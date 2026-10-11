"""Build the stage-1 and stage-2 training data from the raw hub snapshots.

  python src/prepare_data.py [--root DIR] [--only NAME ...]

  raw/<org>__<repo>/                         hub snapshots
  stage1/<source>/<group>/<subset>.parquet   query, document
  stage2/<source>/<group>/<subset>.parquet   anchor, positive, negative_1 .. negative_k

A group names what a pair encodes: `search` (query -> document), `similarity` (symmetric
pairs), `classification` (text -> label phrase). Subsets in EXCLUDED are written under
`stage2-excluded/` instead.
"""

import argparse
import glob
import hashlib
import json
import os
import random
import re
from collections import defaultdict
from collections.abc import Iterator

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Dataset

RAW = {
    "nomic": "nomic-ai__nomic-embed-unsupervised-data",
    "lightonai": "lightonai__embeddings-pre-training-curated",
    "lightonai-ft": "lightonai__embeddings-fine-tuning-filtered-en",
    "lightonai-code": "lightonai__embeddings-fine-tuning-filtered-code",
    "cornstack": "lightonai__cornstack",
    "f2llm": "codefuse-ai__F2LLM",
    "f2llm-v2": "codefuse-ai__F2LLM-v2",
    "nomic-sft": "nomic-ai__nomic-embed-supervised-data",
    "quora": "sentence-transformers__quora-duplicates",
    "open-synthetic": "jspringer__open-synthetic-embeddings",
    "nemotron": "nvidia__embed-nemotron-dataset-v1",
    "jigsaw": "thesofakillers__jigsaw-toxic-comment-classification-challenge",
    "hate-speech": "ucberkeley-dlab__measuring-hate-speech",
}

STAGE1 = {
    "nomic": {
        "search": (
            "paq gooaq yahoo_title_answer yahoo_qa yahoo_title_question amazonqa eli5 "
            "squad"
        ),
        "classification": (
            "wikianswers wikipedia agnews ccnews npr cnn sentence_compression wikihow "
            "altlex quora simplewiki"
        ),
        "clustering": (
            "reddit_title_body s2orc_citation_titles s2orc_title_abstract "
            "s2orc_abstract_citation s2orc_abstract_body codesearch"
        ),
    },
    "lightonai": {
        "search": (
            "paq gooaq_qa amazon_qa yahoo_qa yahoo_answer yahoo_question_body msmarco "
            "beir_dbpedia hlp_wikipedia_cm hlp_wikipedia_dl s2orc_title_abstract "
            "s2orc_citation_titles s2orc_abstract_citation agnews ccnews_en "
            "cnn_dailymail npr reddit reddit_body_comment fw-edu "
            "stackoverflow_title_body"
        ),
        "classification": "altlex quora wikianswers",
    },
}

EXCLUDED = {
    "nomic-sft": "reddit_triples medi_supernli_sampled",
    "quora-duplicates": "neg24",
    "f2llm-v2": (
        "stackexchange stackexchange_part2 healthcaremagic medical_instruction "
        "medical_flashcards medinstruct xsum sentence_compression emotion"
    ),
}

CODE_LANGS = "go java javascript php python ruby".split()

LIGHTONAI_SPLITS = (
    "msmarco_en nq_en hotpotqa_en fever_en squadv2_en trivia_en miracl_en mldr_en"
).split()

S2ORC_NEGATIVES = {"title_abstract": 24, "abstract_citation": 10, "title_citation": 24}

F2LLM_V2 = {
    "search": (
        "stackexchange stackexchange_part2 pubmedqa bioasq healthcaremagic "
        "medical_instruction medquad medical_flashcards medinstruct xsum "
        "sentence_compression dbpedia"
    ),
    "similarity": "qqp pawsx_en anli mscinli",
    "classification": "amazon_polarity",
    "clustering": "emotion",
}
F2LLM_V2_NEGATIVES = {
    "stackexchange": 10,
    "stackexchange_part2": 10,
    "pubmedqa": 10,
    "bioasq": 10,
    "medquad": 10,
    "xsum": 10,
}
INSTRUCTION = re.compile(r"^Instruct:.*?\nQuery:\s*", re.S)

NOMIC_SFT = {
    "reddit_triples": ("search", "reddit_triples", 25),
    "medi_sts_flickr_sampled": ("similarity", "medi_sts_flickr_sampled", 20),
    "medi_supernli_sampled": ("similarity", "medi_supernli_sampled", 20),
    "nli_simcse_50negs_fixed": ("similarity", "nli_simcse_10negs", 10),
}
QUORA_NEGATIVES = 24
OPEN_SYNTHETIC = {
    "LLaMA-3.1-70B-Instruct": "sts@llama70b",
    "LLaMA-3.1-8B-Instruct": "sts@llama8b",
}

NEMOTRON_NEGATIVES = 4
JIGSAW_LABELS = "toxic severe_toxic obscene threat insult identity_hate".split()
HATE_SPEECH_TOXIC, HATE_SPEECH_CLEAN = 0.5, -1.0


def read(path: str, columns: list[str] | None = None) -> dict[str, list]:
    return pq.read_table(path, columns=columns).to_pydict()


def save(table: pa.Table, root: str, folder: str, name: str):
    if name in EXCLUDED.get(folder.split("/")[1], "").split():
        folder = folder.replace("stage2", "stage2-excluded", 1)
    path = os.path.join(root, folder, f"{name}.parquet")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pq.write_table(table, path, compression="zstd")
    print(f"{folder}/{name}: {table.num_rows:,} rows", flush=True)


def pair_table(anchor: list, positive: list, *negatives: list) -> pa.Table:
    names = ["anchor", "positive"]
    names += [f"negative_{i}" for i in range(1, len(negatives) + 1)]
    return pa.table([anchor, positive, *negatives], names=names)


def rows_table(rows: list[list[str]]) -> pa.Table:
    return pair_table(*map(list, zip(*rows)))


def mined_table(base: str, split: str, negatives: int = 10) -> pa.Table:
    """(query, positive, negatives) from the queries / documents / scores layout."""
    file = f"{split}-00000-of-00001.parquet"
    q = read(f"{base}/queries/{file}")
    queries = dict(zip(q["query_id"], q["query"]))
    documents = {}
    for path in sorted(glob.glob(f"{base}/documents/{split}-*.parquet")):
        d = read(path, ["document_id", "document"])
        documents.update(zip(d["document_id"], d["document"]))

    rows = []
    s = read(f"{base}/scores/{file}", ["query_id", "document_ids"])
    for qid, ids in zip(s["query_id"], s["document_ids"]):
        texts = [documents.get(i) for i in ids]
        if qid in queries and len(ids) == negatives + 1 and None not in texts:
            rows.append([queries[qid], *texts])
    return rows_table(rows)


def f2llm_table(path: str, negatives: int, strip: bool = True) -> pa.Table:
    """F2LLM (query, passage, negative_*) with the first `negatives` negatives."""
    file = pq.ParquetFile(path)
    available = sum(name.startswith("negative_") for name in file.schema_arrow.names)
    columns = ["query", "passage"]
    columns += [f"negative_{i}" for i in range(1, min(negatives, available) + 1)]

    tables = []
    for batch in file.iter_batches(20_000, columns=columns):
        texts = [column.to_pylist() for column in batch.columns]
        if strip:
            texts = [[INSTRUCTION.sub("", x or "") for x in column] for column in texts]
        tables.append(pair_table(*texts))
    return pa.concat_tables(tables)


def link_stage1(root: str):
    for source, groups in STAGE1.items():
        base = os.path.join(root, "raw", RAW[source])
        for group, subsets in groups.items():
            folder = os.path.join(root, "stage1", source, group)
            os.makedirs(folder, exist_ok=True)
            for subset in subsets.split():
                files = glob.glob(f"{base}/**/{subset}-*.parquet", recursive=True)
                files += glob.glob(f"{base}/**/{subset}.parquet", recursive=True)
                assert files, f"{source}/{subset}: no parquet under {base}"
                for file in sorted(files):
                    link = os.path.join(folder, os.path.basename(file))
                    if not os.path.islink(link):
                        os.symlink(os.path.relpath(file, folder), link)
                print(f"stage1/{source}/{group}/{subset}: {len(files)} files")


def text_key(text: str) -> bytes:
    return hashlib.md5(" ".join(text.split()).encode()).digest()


def code_test_keys() -> set[bytes]:
    """Keys of every MTEB(Code, v1) test query and corpus text."""
    import mteb

    def texts(split: dict | Dataset) -> list[str]:
        if isinstance(split, dict):
            return [v["text"] if isinstance(v, dict) else v for v in split.values()]
        return list(split["text"])

    def test_splits(task: mteb.AbsTask) -> Iterator[tuple]:
        if getattr(task, "dataset", None):
            for subset in task.dataset.values():
                if subset is not None:
                    split = subset.get("test", next(iter(subset.values())))
                    yield split["queries"], split["corpus"]
        else:
            for lang, splits in task.queries.items():
                for name, queries in splits.items():
                    yield queries, task.corpus[lang][name]

    keys = set()
    for task in mteb.get_benchmark("MTEB(Code, v1)").tasks:
        task.load_data()
        for queries, corpus in test_splits(task):
            for text in texts(queries) + texts(corpus):
                if text and len(text) > 20:
                    keys.add(text_key(text))
    return keys


def cut_code(code: str, rng: random.Random) -> tuple[str, str] | None:
    """Split at a uniform 40-70% of the characters, moved to the next whitespace."""
    k = int(len(code) * rng.uniform(0.4, 0.7))
    while k < len(code) and not code[k].isspace():
        k += 1
    head, tail = code[:k], code[k:]
    return (head, tail) if head.strip() and tail.strip() else None


def prepare_cornstack(root: str):
    """docstring -> code, code -> docstring and code prefix -> rest, without any text
    of an MTEB(Code, v1) test set."""
    base = os.path.join(root, "raw", RAW["cornstack"])
    test = code_test_keys()
    for lang in CODE_LANGS:
        d = read(f"{base}/{lang}", ["query", "document"])
        pairs = [
            (doc, code)
            for doc, code in zip(d["query"], d["document"])
            if text_key(doc) not in test and text_key(code) not in test
        ]
        rng = random.Random(f"cornstack@{lang}")
        cuts = [cut for _, code in pairs if (cut := cut_code(code, rng))]

        subsets = {"dtc": pairs, "ctd": [(c, d) for d, c in pairs], "ctc": cuts}
        for kind, rows in subsets.items():
            query, document = map(list, zip(*rows))
            table = pa.table({"query": query, "document": document})
            save(table, root, "stage1/cornstack/search", f"cornstack@{lang}_{kind}")


def prepare_lightonai(root: str):
    base = os.path.join(root, "raw", RAW["lightonai-ft"])
    for split in LIGHTONAI_SPLITS:
        save(mined_table(base, split), root, "stage2/lightonai/search", split)


def prepare_lightonai_code(root: str):
    base = os.path.join(root, "raw", RAW["lightonai-code"])
    for lang in CODE_LANGS:
        table = mined_table(base, f"CodeSearchNet_ccr_{lang}")
        save(table, root, "stage2/lightonai-code/search", f"csn_ccr@{lang}")


def prepare_s2orc(root: str):
    base = os.path.join(root, "raw", RAW["f2llm"])
    for subset, negatives in S2ORC_NEGATIVES.items():
        table = f2llm_table(f"{base}/s2orc_{subset}.parquet", negatives)
        save(table, root, "stage2/s2orc/search", subset)


def prepare_f2llm_v2(root: str):
    base = os.path.join(root, "raw", RAW["f2llm-v2"])
    for group, subsets in F2LLM_V2.items():
        for subset in subsets.split():
            negatives = F2LLM_V2_NEGATIVES.get(subset, 24)
            table = f2llm_table(f"{base}/{subset}.parquet", negatives)
            save(table, root, f"stage2/f2llm-v2/{group}", subset)


def prepare_f2llm_v2_code(root: str):
    """ocgi (instruction -> code) and xcodeeval_nl2code (problem -> solution), with the
    instructions kept in the queries."""
    base = os.path.join(root, "raw", RAW["f2llm-v2"])
    parts = [f"{base}/ocgi.parquet", f"{base}/ocgi_part2.parquet"]
    ocgi = pa.concat_tables([f2llm_table(p, 10, strip=False) for p in parts])
    save(ocgi, root, "stage2/f2llm-v2/search", "ocgi")

    for path in sorted(glob.glob(f"{base}/xcodeeval_nl2code_*.parquet")):
        lang = re.fullmatch(r"xcodeeval_nl2code_(.+)\.parquet", os.path.basename(path))
        table = f2llm_table(path, 24, strip=False)
        save(table, root, "stage2/f2llm-v2/search", f"xcodeeval_nl2code@{lang[1]}")


def prepare_nomic_sft(root: str):
    base = os.path.join(root, "raw", RAW["nomic-sft"], "data")
    for subset, (group, name, negatives) in NOMIC_SFT.items():
        rows = []
        for path in sorted(glob.glob(f"{base}/{subset}-*.parquet")):
            d = read(path, ["query", "document", "negative"])
            for query, document, negative in zip(*d.values()):
                negative = [x for x in negative if x][:negatives]
                if query and document and len(negative) == negatives:
                    rows.append([query, document, *negative])
        save(rows_table(rows), root, f"stage2/nomic-sft/{group}", name)


def prepare_quora(root: str):
    """triplet-all pairs with at least QUORA_NEGATIVES ranked negatives."""
    base = os.path.join(root, "raw", RAW["quora"], "triplet-all")
    d = read(f"{base}/train-00000-of-00001.parquet")
    pairs = defaultdict(list)
    for anchor, positive, negative in zip(d["anchor"], d["positive"], d["negative"]):
        pairs[anchor, positive].append(negative)

    rows = [
        [*pair, *negatives[:QUORA_NEGATIVES]]
        for pair, negatives in pairs.items()
        if len(negatives) >= QUORA_NEGATIVES
    ]
    save(rows_table(rows), root, "stage2/quora-duplicates/similarity", "neg24")


def prepare_open_synthetic(root: str):
    """LLaMA-generated STS triplets in English, one row per query."""
    import fasttext
    from huggingface_hub import hf_hub_download

    lid = hf_hub_download("facebook/fasttext-language-identification", "model.bin")
    lid = fasttext.load_model(lid)

    def english(text: str) -> bool:
        return lid.predict(text)[0][0] == "__label__eng_Latn"

    base = os.path.join(root, "raw", RAW["open-synthetic"])
    folder = "stage2/open-synthetic-embeddings/similarity"
    keys = ("query", "positive", "negative")
    for generator, name in OPEN_SYNTHETIC.items():
        subset = f"synthetic_{generator}_sts"
        rows, seen = [], set()
        with open(f"{base}/{subset}/{subset}.jsonl") as f:
            for line in f:
                row = json.loads(line)
                texts = [" ".join(row[k].split()) for k in keys]
                query, positive, _ = texts
                if all(texts) and query != positive and query not in seen:
                    if all(map(english, texts)):
                        seen.add(query)
                        rows.append(texts)
        save(rows_table(rows), root, folder, name)


def prepare_nemotron(root: str):
    """English synthetic text -> label phrase, one subset per number of negatives."""
    from fast_langdetect import detect

    def english(text: str) -> bool:
        try:
            return detect(text.replace("\n", " ")[:400])[0]["lang"] == "en"
        except Exception:
            return False

    base = os.path.join(root, "raw", RAW["nemotron"], "SyntheticClassificationData")
    c = read(f"{base}/corpus.parquet")
    labels = dict(zip(c["id"], c["text"]))
    q = read(f"{base}/queries.parquet", ["question", "pos_doc", "neg_doc"])
    pattern = re.compile(r"Instruct: (.*?)\nQuery: (.*)", re.S)

    buckets = defaultdict(list)
    for question, positive, negative in zip(q["question"], q["pos_doc"], q["neg_doc"]):
        match = pattern.match(question)
        text = (match[2] if match else question).strip()
        if not text or len(positive) != 1 or not english(text):
            continue
        if negative := [labels[x["id"]] for x in negative][:NEMOTRON_NEGATIVES]:
            row = [question.strip(), labels[positive[0]["id"]], *negative]
            buckets[len(negative)].append(row)

    folder = "stage2/nemotron-synthetic-classification/classification"
    for k, rows in sorted(buckets.items()):
        save(rows_table(rows), root, folder, f"neg@{k}")


def clean_text(value: object, seen: set[str]) -> str | None:
    text = " ".join(str(value).split())
    if len(text.split()) < 3 or text in seen:
        return None
    seen.add(text)
    return text


def save_toxicity(toxic: list[str], clean: list[str], root: str, source: str):
    """Balanced text -> `toxic` / `not toxic` pairs, the other label as the negative."""
    rng = random.Random(42)
    k = min(len(toxic), len(clean))
    rows = [[text, "toxic", "not toxic"] for text in rng.sample(toxic, k)]
    rows += [[text, "not toxic", "toxic"] for text in rng.sample(clean, k)]
    rng.shuffle(rows)
    save(rows_table(rows), root, f"stage2/{source}/classification", "neg1")


def prepare_jigsaw(root: str):
    import pyarrow.csv as pc

    base = os.path.join(root, "raw", RAW["jigsaw"])
    options = pc.ParseOptions(newlines_in_values=True)
    d = pc.read_csv(f"{base}/train.csv", parse_options=options).to_pydict()

    toxic, clean, seen = [], [], set()
    for i, value in enumerate(d["comment_text"]):
        if text := clean_text(value, seen):
            (toxic if any(d[c][i] for c in JIGSAW_LABELS) else clean).append(text)
    save_toxicity(toxic, clean, root, "jigsaw-toxic-comment")


def prepare_hate_speech(root: str):
    """Comments whose mean annotator score is clearly toxic or clearly not."""
    base = os.path.join(root, "raw", RAW["hate-speech"])
    d = read(f"{base}/data/train-00000-of-00001.parquet")
    scores, texts = defaultdict(list), {}
    for key, text, score in zip(d["comment_id"], d["text"], d["hate_speech_score"]):
        scores[key].append(score)
        texts[key] = text

    toxic, clean, seen = [], [], set()
    for key, values in scores.items():
        text, mean = clean_text(texts[key], seen), sum(values) / len(values)
        if text and mean > HATE_SPEECH_TOXIC:
            toxic.append(text)
        elif text and mean < HATE_SPEECH_CLEAN:
            clean.append(text)
    save_toxicity(toxic, clean, root, "measuring-hate-speech")


PREPARE = {
    "stage1": link_stage1,
    "cornstack": prepare_cornstack,
    "lightonai": prepare_lightonai,
    "lightonai-code": prepare_lightonai_code,
    "s2orc": prepare_s2orc,
    "f2llm-v2": prepare_f2llm_v2,
    "f2llm-v2-code": prepare_f2llm_v2_code,
    "nomic-sft": prepare_nomic_sft,
    "quora": prepare_quora,
    "open-synthetic": prepare_open_synthetic,
    "nemotron": prepare_nemotron,
    "jigsaw": prepare_jigsaw,
    "hate-speech": prepare_hate_speech,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data")
    parser.add_argument("--only", nargs="*", choices=PREPARE, default=list(PREPARE))
    args = parser.parse_args()
    for name in args.only:
        PREPARE[name](args.root)
