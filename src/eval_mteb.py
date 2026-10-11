"""Evaluate a Sentence Transformers model on an MTEB benchmark, one process per task.

  python src/eval_mteb.py --model_name_or_path <model> --task TRECCOVID
  python src/eval_mteb.py --model_name_or_path <model> --queue > queue.sh
  python src/eval_mteb.py --model_name_or_path <model> --summary

With --eval_prompts <yaml> (instruction_template + per-task prompts), the model is
wrapped in mteb's InstructSentenceTransformerModel; otherwise the prompts saved in the
model are used.

Each task result is cached in <output_dir>/results/<model>/<revision>/<task>.json and a
cached task is skipped, so the queue lines can run in parallel (one worker per GPU).
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from collections import defaultdict

import mteb
import yaml
from mteb.benchmarks import _benchmark_metrics as metrics
from mteb.models.instruct_wrapper import InstructSentenceTransformerModel
from mteb.models.model_meta import ModelMeta
from mteb.models.sentence_transformer_wrapper import SentenceTransformerEncoderWrapper


def text_length(task):
    stats = task.metadata.descriptive_stats or {}
    splits = [stats.get(split, {}) for split in task.metadata.eval_splits]
    fields = [v for split in splits for v in split.values() if isinstance(v, dict)]
    return sum(v.get("total_text_length", 0) for v in fields)


def print_summary(cache, tasks, model_name, revision):
    results, by_type = [], defaultdict(list)
    for task in tasks:
        result = cache.load_task_result(task.metadata.name, model_name, revision)
        score = f"{100 * result.get_score():.2f}" if result else "-"
        print(f"{task.metadata.name:40s} {task.metadata.type:20s} {score}")
        if result:
            results.append(result)
            by_type[task.metadata.type].append(result)

    print()
    for task_type, group in sorted(by_type.items()):
        mean = 100 * metrics._compute_mean_task(group)
        print(f"{task_type:20s} {mean:.2f} ({len(group)} tasks)")

    mean_task = 100 * metrics._compute_mean_task(results)
    mean_task_type = 100 * metrics._compute_mean_task_type(results)
    print(f"\nMean(Task)     {mean_task:.2f} ({len(results)}/{len(tasks)} tasks)")
    print(f"Mean(TaskType) {mean_task_type:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--model_kwargs", default="{}")
    parser.add_argument("--eval_prompts")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--output_dir", default="results/mteb")
    parser.add_argument("--benchmark", default="MTEB(eng, v2)")
    parser.add_argument("--task", nargs="*")
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--queue", action="store_true")
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()

    tasks = mteb.get_benchmark(args.benchmark).tasks
    if args.task:
        tasks = [task for task in tasks if task.metadata.name in args.task]

    # Create queue commands to evaluate benchmarks in parallel.
    if args.queue:
        argv = shlex.join(arg for arg in sys.argv if arg != "--queue")
        for task in sorted(tasks, key=text_length, reverse=True):
            print(f"python {argv} --task {task.metadata.name}")
        sys.exit()

    # Summarize the evaluation results.
    cache = mteb.ResultCache(cache_path=args.output_dir)
    if args.summary:
        print_summary(cache, tasks, args.model_name_or_path, args.revision)
        sys.exit()

    kwargs = dict(trust_remote_code=True, model_kwargs=json.loads(args.model_kwargs))
    if args.eval_prompts:
        spec = yaml.safe_load(open(args.eval_prompts))
        model = InstructSentenceTransformerModel(
            args.model_name_or_path,
            revision=args.revision,
            instruction_template=spec["instruction_template"],
            prompts_dict=spec["prompts"],
            apply_instruction_to_passages=False,
            **kwargs,
        )
        model.mteb_model_meta = ModelMeta.create_empty(
            overwrites=dict(name=args.model_name_or_path, revision=args.revision)
        )
    else:
        model = SentenceTransformerEncoderWrapper(
            args.model_name_or_path, revision=args.revision, **kwargs
        )
    model.model.max_seq_length = args.max_seq_length
    mteb.evaluate(
        model,
        tasks,
        cache=cache,
        encode_kwargs={"batch_size": args.batch_size},
        raise_error=False,
        show_progress_bar=False,
    )
