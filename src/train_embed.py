"""Contrastive embedding training with Sentence Transformers.

  torchrun --nproc_per_node 8 src/train_embed.py <config.yaml>

`dataset_mixture` is a list of groups, each a `sampling_weight` and `datasets` entries.
An entry is a parquet glob with a `query_prefix`, a `document_prefix` and `in_batch`:
in-batch negatives gathered across ranks, or only the row's own negatives.

Every step draws one dataset for all ranks: a group by its weight, a source of the group
by size^`source_alpha`, then a `<source>@<subset>` file by size.
"""

import argparse
import glob
import json
import os
import re
import shutil
from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence
from functools import partial
from itertools import accumulate, islice

import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from datasets import Dataset, DatasetDict, load_dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.base.evaluation.evaluator import BaseEvaluator
from sentence_transformers.base.losses import gradcache
from sentence_transformers.base.sampler import (
    DefaultBatchSampler,
    NoDuplicatesBatchSampler,
    ProportionalBatchSampler,
)
from sentence_transformers.sentence_transformer.evaluation import (
    InformationRetrievalEvaluator,
)
from sentence_transformers.sentence_transformer.losses import (
    CachedMultipleNegativesRankingLoss,
)
from sentence_transformers.sentence_transformer.modules import Pooling
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import RandomSampler, SubsetRandomSampler
from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

NANO_SPLITS = {
    "hakari-bench/NanoMTEB-v2": {
        "argu_ana": "ArguAna",
        "climate_fever": "ClimateFEVERHardNegatives",
        "cqadupstack_gaming": "CQADupstackGamingRetrieval",
        "cqadupstack_unix": "CQADupstackUnixRetrieval",
        "fever": "FEVERHardNegatives",
        "fi_qa2018": "FiQA2018",
        "hotpot_qa": "HotpotQAHardNegatives",
        "scidocs": "SCIDOCS",
        "touche2020_v3": "Touche2020Retrieval.v3",
        "treccovid": "TRECCOVID",
    },
    "hakari-bench/NanoCoIR": {
        "NanoApps": "AppsRetrieval",
        "NanoCodeFeedbackMT": "CodeFeedbackMT",
        "NanoCodeFeedbackST": "CodeFeedbackST",
        "NanoCodeSearchNet": "COIRCodeSearchNetRetrieval",
        "NanoCodeSearchNetCCR": "CodeSearchNetCCRetrieval",
        "NanoCodeTransOceanContest": "CodeTransOceanContest",
        "NanoCodeTransOceanDL": "CodeTransOceanDL",
        "NanoCosQA": "CosQA",
        "NanoStackOverflowQA": "StackOverflowQA",
        "NanoSyntheticText2SQL": "SyntheticText2SQL",
    },
}


class NanoEvaluator(BaseEvaluator):
    def __init__(self, prompts: dict[str, str], batch_size: int):
        super().__init__()
        self.primary_metric = "nano_mean_cosine_ndcg@10"
        self.evaluators, self.benches = [], defaultdict(list)
        for repo, splits in NANO_SPLITS.items():
            for split, task in splits.items():
                evaluator = InformationRetrievalEvaluator(
                    *self.load(repo, split),
                    name=split,
                    query_prompt=prompts.get(f"{task}-query", prompts.get("query")),
                    batch_size=batch_size,
                    show_progress_bar=False,
                )
                self.evaluators.append(evaluator)
                self.benches[repo.split("/")[1]].append(split)

    @staticmethod
    def load(repo: str, split: str) -> tuple[dict, dict, dict]:
        relevant = defaultdict(set)
        for row in load_dataset(repo, "qrels", split=split):
            ids = row["corpus-id"]
            relevant[row["query-id"]].update(ids if isinstance(ids, list) else [ids])

        corpus = load_dataset(repo, "corpus", split=split)
        corpus = {row["_id"]: row["text"] for row in corpus if row["text"]}
        queries = {
            row["_id"]: row["text"]
            for row in load_dataset(repo, "queries", split=split)
            if row["text"] and row["_id"] in relevant
        }
        return queries, corpus, relevant

    def __call__(
        self,
        model: SentenceTransformer,
        output_path: str | None = None,
        epoch: float = -1,
        steps: int = -1,
    ) -> dict[str, float]:
        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1

        metrics = {}
        for evaluator in self.evaluators[rank::world_size]:
            metrics.update(evaluator(model, output_path, epoch, steps))
        if dist.is_initialized():
            shards = [None] * world_size
            dist.all_gather_object(shards, metrics)
            metrics = {k: v for shard in shards for k, v in shard.items()}

        means = {}
        for bench, splits in self.benches.items():
            scores = [metrics[f"{split}_cosine_ndcg@10"] for split in splits]
            means[f"{bench}_mean_cosine_ndcg@10"] = sum(scores) / len(scores)
        means["nano_mean_cosine_ndcg@10"] = sum(means.values()) / len(means)
        metrics.update(means)

        if output_path is not None and rank == 0:
            path = os.path.join(output_path, "Nano_evaluation_mean_results.jsonl")
            with open(path, "a") as f:
                print(json.dumps({"epoch": epoch, "steps": steps, **means}), file=f)
        return metrics


class KeepBestCheckpoints(TrainerCallback):
    """Keeps the `top_k` checkpoints with the highest `metric`, plus the newest one."""

    def __init__(self, metric: str, top_k: int):
        super().__init__()
        self.metric, self.top_k, self.scores = metric, top_k, {}

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: dict[str, float] | None = None,
        **kwargs,
    ):
        if metrics and self.metric in metrics:
            self.scores[state.global_step] = metrics[self.metric]

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if not state.is_world_process_zero:
            return

        paths = {}
        for path in glob.glob(os.path.join(args.output_dir, "checkpoint-*")):
            if match := re.fullmatch(r"checkpoint-(\d+)", os.path.basename(path)):
                paths[int(match.group(1))] = path
        if not paths:
            return

        scored = [step for step in self.scores if step in paths]
        ranked = sorted(scored, key=self.scores.get, reverse=True)
        keep = {max(paths)} | set(ranked[: self.top_k])
        for step, path in paths.items():
            if step in self.scores and step not in keep:
                shutil.rmtree(path, ignore_errors=True)


class SameSourceBatchSampler(ProportionalBatchSampler):
    """Draws `world_size` consecutive batches from one dataset: a mixture group by its
    weight, then a source of the group by size^alpha (see the module docstring).
    """

    def __init__(
        self,
        *args,
        alpha: float = 1.0,
        groups: dict[str, int] | None = None,
        group_weights: Sequence[float] = (1.0,),
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.alpha, self.groups, self.group_weights = alpha, groups or {}, group_weights
        self.weighted = alpha != 1 or len(group_weights) > 1

    @property
    def world_size(self) -> int:
        return dist.get_world_size() if dist.is_initialized() else 1

    def dataset_order(self) -> Iterable[int]:
        """Dataset index per step: a permutation (not weighted) or weighted draws.

        P(dataset) = P(group) P(source | group) P(`@` subset | source)
                   = weight * total^alpha / mass[group] * size / total,
        where total is the size of the source over its `@` subsets and mass[group]
        the sum of total^alpha over the sources of the group.
        """
        counts = [len(sampler) // self.world_size for sampler in self.batch_samplers]
        if not self.weighted:
            order = [i for i, count in enumerate(counts) for _ in range(count)]
            return SubsetRandomSampler(order, generator=self.generator)

        names = [d[0]["dataset_name"] for d in self.dataset.datasets]
        sizes = [len(d) for d in self.dataset.datasets]
        sources = [name.split("@")[0] for name in names]
        groups = [self.groups.get(name, 0) for name in names]

        total, mass = defaultdict(int), defaultdict(float)
        for source, size in zip(sources, sizes):
            total[source] += size
        for source, group in dict(zip(sources, groups)).items():
            mass[group] += total[source] ** self.alpha

        probs = []
        for g, s, n in zip(groups, sources, sizes):
            p_source = self.group_weights[g] * total[s] ** self.alpha / mass[g]
            probs.append(p_source * n / total[s])

        return torch.multinomial(
            input=torch.tensor(probs),
            num_samples=sum(counts),
            replacement=True,
            generator=self.generator,
        ).tolist()

    def __iter__(self) -> Iterator[list[int]]:
        if self.generator and self.seed is not None:
            self.generator.manual_seed(self.seed + self.epoch)

        offsets = [0] + list(accumulate(len(d) for d in self.dataset.datasets))
        samplers = [iter(sampler) for sampler in self.batch_samplers]

        for i in self.dataset_order():
            group = list(islice(samplers[i], self.world_size))
            if len(group) < self.world_size and self.weighted:
                samplers[i] = iter(self.batch_samplers[i])
                group = list(islice(samplers[i], self.world_size))
            if len(group) == self.world_size:
                for batch in group:
                    yield [idx + offsets[i] for idx in batch]

    def __len__(self) -> int:
        world_size = self.world_size
        return sum(len(s) // world_size * world_size for s in self.batch_samplers)


def batch_sampler_fn(
    dataset: Dataset, *, no_duplicates: bool, in_batch: dict[str, bool], **kwargs
) -> DefaultBatchSampler:
    """NoDuplicates only for in-batch datasets (label rows share their positives)."""
    if no_duplicates and in_batch[dataset[0]["dataset_name"]]:
        return NoDuplicatesBatchSampler(dataset, **kwargs)
    sampler = RandomSampler(dataset, generator=kwargs["generator"])
    return DefaultBatchSampler(sampler, **kwargs)


def backward_hook_sync_once(
    grad_output: Tensor, *, loss_obj: CachedMultipleNegativesRankingLoss, **kwargs
):
    """GradCache chunk backwards without DDP sync, then one all-reduce of the gradients.

    DDP all-reduces on every backward call, and token-budget chunks give each rank a
    different number of chunk backwards, so the ranks deadlock. Assumes
    gradient_accumulation_steps 1 (`.grad` holds only this step).
    """
    model = loss_obj.model
    if not isinstance(model, DistributedDataParallel):
        return gradcache_backward_hook(grad_output, loss_obj=loss_obj, **kwargs)
    with model.no_sync():
        gradcache_backward_hook(grad_output, loss_obj=loss_obj, **kwargs)

    grads = []
    for param in model.module.parameters():
        if param.requires_grad:
            if param.grad is None:
                param.grad = torch.zeros_like(param)
            grads.append(param.grad)

    handles = [dist.all_reduce(grad, async_op=True) for grad in grads]
    for handle in handles:
        handle.wait()
    torch._foreach_div_(grads, dist.get_world_size())


gradcache_backward_hook = gradcache._backward_hook
gradcache._backward_hook = backward_hook_sync_once


def source_name(path: str) -> str:
    """
    .../lightonai/search/msmarco_en.parquet -> lightonai__search__msmarco_en;
    .../classification/neg@2.parquet -> ...__classification__neg@2 (source: up to @).
    """
    stem = re.sub(r"-\d+-of-\d+$", "", os.path.splitext(os.path.basename(path))[0])
    return "__".join(os.path.normpath(path).split(os.sep)[-3:-1] + [stem])


def load_sources(entry: dict) -> dict[str, Dataset]:
    """One dataset per source under the entry's glob (anchor, positive, negative_*)."""
    files = defaultdict(list)
    for path in sorted(glob.glob(os.path.expanduser(entry["path"]))):
        files[source_name(path)].append(path)

    sources = {}
    for name, paths in files.items():
        ds = load_dataset("parquet", data_files=paths, split="train")
        if "query" in ds.column_names:
            ds = ds.rename_columns({"query": "anchor", "document": "positive"})
        negatives = [c for c in ds.column_names if c.startswith("negative")]
        sources[name] = ds.select_columns(["anchor", "positive", *negatives])
    return sources


class HardNegativesOnlyLoss(nn.Module):
    """InfoNCE over (anchor, positive, negatives) of one row; no in-batch negatives.
    With cosent_weight > 0, adds weight * CoSENT over the local batch: every
    (anchor, positive) cosine must exceed every (anchor, negative) cosine of any row.
    """

    def __init__(
        self,
        model: SentenceTransformer,
        infonce_scale: float,
        cosent_scale: float = 20.0,
        cosent_weight: float = 0.0,
    ):
        super().__init__()
        self.model, self.infonce_scale = model, infonce_scale
        self.cosent_scale, self.cosent_weight = cosent_scale, cosent_weight

    def embed(self, features: dict[str, Tensor]) -> Tensor:
        return F.normalize(self.model(features)["sentence_embedding"], dim=-1)

    def forward(
        self, sentence_features: list[dict[str, Tensor]], labels: Tensor | None = None
    ) -> Tensor:
        anchor, *candidates = map(self.embed, sentence_features)
        candidates = torch.stack(candidates, dim=1)

        cos = torch.einsum("bd,bkd->bk", anchor, candidates)
        targets = cos.new_zeros(len(cos), dtype=torch.long)

        loss = F.cross_entropy(self.infonce_scale * cos, targets)
        if self.cosent_weight > 0:
            diff = self.cosent_scale * (cos[:, 1:].reshape(1, -1) - cos[:, :1])
            pairs = torch.cat([diff.new_zeros(1), diff.reshape(-1)])
            loss = loss + self.cosent_weight * torch.logsumexp(pairs, dim=0)
        return loss


def make_loss(model: SentenceTransformer, entry: dict, cfg: dict) -> nn.Module:
    """CachedMNRL with in-batch negatives, or the row-wise hard-negative loss."""
    if entry["in_batch"]:
        return CachedMultipleNegativesRankingLoss(
            model,
            scale=entry["infonce_scale"],
            gather_across_devices=True,
            mini_batch_size=cfg.get("grad_cache_chunk_size", 32),
            mini_batch_num_tokens=cfg.get("grad_cache_chunk_tokens"),
        )
    return HardNegativesOnlyLoss(
        model,
        infonce_scale=entry["infonce_scale"],
        cosent_scale=entry.get("cosent_scale", 0.0),
        cosent_weight=entry.get("cosent_weight", 0.0),
    )


def main(cfg: dict):
    # Create sentence transformer model.
    model = SentenceTransformer(
        cfg["model"],
        trust_remote_code=True,
        prompts=cfg["eval_prompts"],
        model_kwargs={"attn_implementation": "flash_attention_2"},
        processor_kwargs={"chat_template": None},
    )
    model.max_seq_length = cfg["max_seq_length"]
    model[1] = Pooling(model[0].get_word_embedding_dimension(), cfg["pooling_mode"])

    # Prepare the dataset mixture.
    mixture = cfg["dataset_mixture"]
    train, losses, prompts, in_batch, groups = {}, {}, {}, {}, {}
    for group, entries in enumerate(m["datasets"] for m in mixture):
        for entry in entries:
            loss = make_loss(model, entry, cfg)
            query = entry.get("query_prefix", "")
            document = entry.get("document_prefix", "")
            for name, dataset in load_sources(entry).items():
                prefix = {c: document for c in dataset.column_names} | {"anchor": query}
                train[name], losses[name] = dataset, loss
                prompts[name] = {c: p for c, p in prefix.items() if p}
                in_batch[name], groups[name] = entry["in_batch"], group

    # Create training args with batch and dataset sampler.
    nodups = cfg["training_args"].pop("batch_sampler", None) == "no_duplicates"
    batch_sampler = partial(batch_sampler_fn, no_duplicates=nodups, in_batch=in_batch)
    multi_dataset_batch_sampler = partial(
        SameSourceBatchSampler,
        alpha=cfg.get("source_alpha", 1.0),
        groups=groups,
        group_weights=[m.get("sampling_weight", 1.0) for m in mixture],
    )

    args = SentenceTransformerTrainingArguments(
        prompts=prompts,
        batch_sampler=batch_sampler,
        multi_dataset_batch_sampler=multi_dataset_batch_sampler,
        **cfg["training_args"],
    )
    assert args.gradient_accumulation_steps == 1, "backward_hook_sync_once needs 1"

    # Create an evaluator and start training.
    callbacks = []
    if top_k := cfg.get("keep_best_checkpoints"):
        callbacks.append(KeepBestCheckpoints("eval_nano_mean_cosine_ndcg@10", top_k))

    evaluator = NanoEvaluator(model.prompts, batch_size=args.per_device_eval_batch_size)
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=DatasetDict(train),
        loss=losses,
        evaluator=evaluator,
        callbacks=callbacks,
    )
    trainer.train(resume_from_checkpoint=cfg.get("resume_from_checkpoint"))
    if trainer.is_world_process_zero():
        model.save_pretrained(os.path.join(args.output_dir, "final"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    main(yaml.safe_load(open(args.config)))
