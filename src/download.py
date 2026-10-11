"""Download the hub snapshots that prepare_data.py builds from.

  python src/download.py [--root DIR] [--only NAME ...]

Each source is pinned to the revision the data was built from and limited to the files
prepare_data.py reads, under <root>/raw/<org>__<repo>/.
"""

import argparse
import os

from huggingface_hub import snapshot_download

from prepare_data import CODE_LANGS, F2LLM_V2, LIGHTONAI_SPLITS, NOMIC_SFT, RAW, STAGE1

REVISIONS = {
    "nomic": "917bae6ed30ebc80fc8c81ba8e3e34558205d6bb",
    "lightonai": "1bdc6433cfa9de28d3816e0550b1cb3107ffd92d",
    "lightonai-ft": "3464b0de66da5834f165a7eded49f1cc63d0b5eb",
    "lightonai-code": "3e4f7d64fee065a81766a219408284eea44a7d48",
    "cornstack": "44094c61a02b2249f7128a9b86a97dc2e1350a37",
    "f2llm": "c8158be982d16202dda93211c1f7a542159acc3e",
    "f2llm-v2": "d520b8ad02c86d5e5611441c6196ff65d8888927",
    "nomic-sft": "13eef8aabb0ba010967e210c5c89497cae5f71d5",
    "quora": "41f699770310302022a4dd75d4cf903bfef9ea46",
    "open-synthetic": "735d4ca17fbd1d5261359b2b4c03ada06f7a6893",
    "nemotron": "f457c3e2da4af3b9dd2818685d411b26298d7cbb",
    "jigsaw": "50f3c926ec8c07b440cb17b55d8cd80e8484497e",
    "hate-speech": "5468f6e118396646b02a2f691e771f6b6d9502ea",
}

NOMIC_STAGE1 = " ".join(STAGE1["nomic"].values()).split()
LIGHTONAI_STAGE1 = " ".join(STAGE1["lightonai"].values()).split()
LIGHTONAI_FILES = ("queries", "documents", "scores")
F2LLM_V2_FILES = [f"{subset}.parquet" for subset in " ".join(F2LLM_V2.values()).split()]

FILES = {
    "nomic": [f"data/{subset}-*" for subset in NOMIC_STAGE1],
    "lightonai": [f"{subset}[-.]*" for subset in LIGHTONAI_STAGE1],
    "lightonai-ft": [f"{f}/{s}-*" for f in LIGHTONAI_FILES for s in LIGHTONAI_SPLITS],
    "lightonai-code": [f"*/CodeSearchNet_ccr_{lang}-*" for lang in CODE_LANGS],
    "cornstack": [f"{lang}/*" for lang in CODE_LANGS],
    "f2llm": ["s2orc_*.parquet"],
    "f2llm-v2": F2LLM_V2_FILES + ["ocgi*.parquet", "xcodeeval_nl2code_*.parquet"],
    "nomic-sft": [f"data/{subset}-*" for subset in NOMIC_SFT],
    "quora": ["triplet-all/*"],
    "open-synthetic": ["synthetic_*_sts/*"],
    "nemotron": ["SyntheticClassificationData/*"],
    "jigsaw": ["train.csv"],
    "hate-speech": ["data/*"],
}


def download(root: str, name: str):
    snapshot_download(
        repo_id=RAW[name].replace("__", "/", 1),
        repo_type="dataset",
        revision=REVISIONS[name],
        allow_patterns=FILES[name],
        local_dir=os.path.join(root, "raw", RAW[name]),
        max_workers=16,
    )
    print(f"raw/{RAW[name]}: {REVISIONS[name][:8]}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data")
    parser.add_argument("--only", nargs="*", choices=RAW, default=list(RAW))
    args = parser.parse_args()
    for name in args.only:
        download(args.root, name)
