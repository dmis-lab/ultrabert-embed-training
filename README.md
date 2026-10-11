![UltraBERT-Embed](img/ultrabert_embed_banner.jpg)

# UltraBERT-Embed Training

UltraBERT-Embed is a family of English text embedding models built on the UltraBERT encoders. UltraBERT-Embed-150M maps text to a 768-dimensional vector, takes inputs up to 32K tokens, follows task instructions on the query side, and embeds code as well as text. On MTEB(eng, v2) it scores 67.24 Mean(Task), the highest of any dense model under 200M parameters, without distillation from a larger embedding model.

| Model | Parameters | Hidden | Context | Description |
|---|---|---|---|---|
| **[UltraBERT-150M](https://huggingface.co/dmis-lab/UltraBERT-150M)** | 150M | 768 | 32K | Pre-trained encoder |
| **[UltraBERT-380M](https://huggingface.co/dmis-lab/UltraBERT-380M)** | 380M | 1,024 | 32K | Pre-trained encoder |
| **[UltraBERT-Embed-150M](https://huggingface.co/dmis-lab/UltraBERT-Embed-150M)** | 150M | 768 | 32K | Text embedding model built on UltraBERT-150M |

This repository holds the code that trains UltraBERT-Embed from an UltraBERT checkpoint: building the training data, the two contrastive training stages, and MTEB evaluation. To use the model, see the [model card](https://huggingface.co/dmis-lab/UltraBERT-Embed-150M).

## 🚀 Setup

Python 3.12. Install `torch==2.14.0` (CUDA 13.0) first, then

```bash
pip install -r requirements.txt
```

and a `flash-attn` 2.8.3 build that matches your torch and CUDA.

## 🗂️ Data

```bash
python src/download.py --root data
python src/prepare_data.py --root data
```

Every training example comes from a dataset that is already public on the Hugging Face Hub. We did not collect, annotate or generate any data of our own; `prepare_data.py` only reformats the public sources into the layout the trainer reads. We therefore do not redistribute the processed data: it inherits the licences of its sources (MS MARCO, inside several retrieval sets, is for non-commercial research only), and anyone can rebuild it locally with the two commands above.

`download.py` fetches each source at the revision our data was built from into `data/raw/<org>__<repo>/`, limited to the files `prepare_data.py` reads (`--only` selects sources). `prepare_data.py` then writes

```
data/stage1/<source>/<group>/<subset>.parquet   query, document
data/stage2/<source>/<group>/<subset>.parquet   anchor, positive, negative_1 .. negative_k
```

The group names what a pair encodes: `search` (query → document), `similarity` (symmetric pairs), `classification` (text → label phrase), `clustering` (text → another text with the same label). Reformatting means selecting subsets, renaming columns, keeping a fixed number of the mined negatives, splitting files by number of negatives, and removing CornStack rows whose text appears in an MTEB(Code, v1) test set. The raw snapshots take about 1.2 TB, most of it the stage-1 pre-training pairs and CornStack.

| Stage | Sources |
|:--|:--|
| 1 | [lightonai/embeddings-pre-training-curated](https://huggingface.co/datasets/lightonai/embeddings-pre-training-curated), [lightonai/cornstack](https://huggingface.co/datasets/lightonai/cornstack) |
| 2 | [lightonai/embeddings-fine-tuning-filtered-en](https://huggingface.co/datasets/lightonai/embeddings-fine-tuning-filtered-en), [codefuse-ai/F2LLM](https://huggingface.co/datasets/codefuse-ai/F2LLM), [codefuse-ai/F2LLM-v2](https://huggingface.co/datasets/codefuse-ai/F2LLM-v2), [nomic-ai/nomic-embed-supervised-data](https://huggingface.co/datasets/nomic-ai/nomic-embed-supervised-data), [jspringer/open-synthetic-embeddings](https://huggingface.co/datasets/jspringer/open-synthetic-embeddings), [nvidia/embed-nemotron-dataset-v1](https://huggingface.co/datasets/nvidia/embed-nemotron-dataset-v1), [thesofakillers/jigsaw-toxic-comment-classification-challenge](https://huggingface.co/datasets/thesofakillers/jigsaw-toxic-comment-classification-challenge), [ucberkeley-dlab/measuring-hate-speech](https://huggingface.co/datasets/ucberkeley-dlab/measuring-hate-speech) |

## ⚙️ Training

```bash
torchrun --nproc_per_node 8 src/train_embed.py config/stage1.yaml
torchrun --nproc_per_node 8 src/train_embed.py config/stage2.yaml
```

Both stages train with InfoNCE in Sentence Transformers on 8 GPUs, pool the CLS token, and use GradCache so that large batches fit in memory. Each stage writes its final model to `<output_dir>/final`.

**Stage 1: Contrastive Pre-Training.** Starting from UltraBERT-150M, the model learns from weak pairs (question → answer, title → body, docstring → code, paraphrases) with in-batch negatives only, gathered across the 8 GPUs. The batch is 16,384 pairs of up to 2,048 tokens, the temperature 0.02, and every batch comes from a single source, so the negatives are hard to tell apart from the positive. Training runs 33K steps, a little over one epoch: 1K warmup, a constant learning rate of 5e-5 up to step 30K, and a linear decay over the last 3K steps.

**Stage 2: Fine-Tuning.** Starting from the stage-1 model, the model learns from labelled data with mined hard negatives in four groups: English retrieval, code retrieval, similarity, and classification & clustering. Queries carry an E5-mistral style instruction (`Instruct: {task}\nQuery: `), documents are raw, and label phrases take `Label: `. Retrieval data also uses in-batch negatives; similarity, classification and clustering data contrast each row only with its own negatives, since other rows of the batch can be false negatives. The batch is 256, inputs are up to 32,768 tokens so long documents stay whole, and training runs 10K steps at a learning rate of 5e-6 with 5% warmup and a linear decay.

A config lists the training data as `dataset_mixture`, a list of groups:

```yaml
dataset_mixture:
- sampling_weight: 0.5406
  datasets:
  - path: data/stage2/lightonai/search/fever_en.parquet
    query_prefix: "Instruct: Given a claim, retrieve documents that support or refute the claim\nQuery: "
    document_prefix: ""
    in_batch: true
    infonce_scale: 50.0
source_alpha: 0.5
```

Every step draws one dataset for all GPUs: a group by `sampling_weight`, a source of the group by size^`source_alpha`, then a `<source>@<subset>` file by size. `query_prefix` is prepended to the anchor and `document_prefix` to every other column, `infonce_scale` is 1/temperature, and `training_args` are passed to `SentenceTransformerTrainingArguments`.

## 📊 Evaluation

`eval_mteb.py` runs an MTEB benchmark one task per process and caches each result in `results/mteb/results/<model>/<revision>/<task>.json`; a cached task is skipped.

```bash
M=runs/stage2/final
ARGS="--max_seq_length 32768 --model_kwargs {\"attn_implementation\":\"flash_attention_2\",\"dtype\":\"bfloat16\"}"
python src/eval_mteb.py --model_name_or_path $M $ARGS --benchmark "MTEB(eng, v2)" --queue > queue.sh
bash queue.sh
python src/eval_mteb.py --model_name_or_path $M --benchmark "MTEB(eng, v2)" --summary
```

`--queue` prints one command per task, longest first, so the lines can be split across GPUs and run in parallel; `--summary` prints the per-task scores, the task-type means, Mean(Task) and Mean(TaskType). Any mteb benchmark name works, e.g. `"MTEB(Code, v1)"`, `LongEmbed` or `RAR-b`.

Without `--eval_prompts` the model uses the prompts saved with it. With `--eval_prompts <yaml>` it is wrapped in mteb's `InstructSentenceTransformerModel`, which puts a task instruction on the queries (and on both sides of symmetric tasks):

```yaml
instruction_template: "Instruct: {instruction}\nQuery: "
prompts:
  query: "Given a question, retrieve passages that answer the question"
  STS: "Retrieve semantically similar text."
  ArguAna: "Given a claim, find documents that refute the claim"
```

A prompt is looked up by task name, then task type, then `query`.
