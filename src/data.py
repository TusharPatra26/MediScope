"""Dataset acquisition and loading.

Run ``python -m src.data`` to download every dataset into ``data/raw``.
Raw data is never committed (see .gitignore).
"""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

import pandas as pd

import config


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  [skip] {dest.name} already present ({dest.stat().st_size / 1e6:.1f} MB)")
        return
    print(f"  [get ] {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "mediscope-research/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp, open(dest, "wb") as fh:
        while chunk := resp.read(1 << 20):
            fh.write(chunk)
    print(f"         -> {dest} ({dest.stat().st_size / 1e6:.1f} MB)")


def download_all(include_summarization: bool = True) -> None:
    """Download PubMed 20k RCT, BC5CDR and (optionally) the summarization test split (~58 MB)."""
    print("PubMed 20k RCT (~39 MB)")
    for split, fname in config.PUBMED_RCT_FILES.items():
        _download(f"{config.PUBMED_RCT_BASE}/{fname}", config.DATA_RAW / "pubmed_rct" / fname)
    print("BC5CDR via T-NER (~4 MB)")
    for split, fname in config.BC5CDR_FILES.items():
        _download(f"{config.BC5CDR_BASE}/{fname}", config.DATA_RAW / "bc5cdr" / fname)
    if include_summarization:
        print("PubMed summarization test split (~58 MB)")
        _download(config.SUMM_TEST_URL, config.DATA_RAW / "pubmed_summarization" / "test.parquet")


# ---------------------------------------------------------------------------
# PubMed 20k RCT  (sentence-level rhetorical-role classification)
# ---------------------------------------------------------------------------
def load_pubmed_rct(split: str) -> pd.DataFrame:
    """Parse a PubMed RCT split into one row per sentence.

    Columns: abstract_id, sent_idx, n_sents, label, text.  The official splits are
    made at the abstract level, so no abstract appears in more than one split.
    """
    path = config.DATA_RAW / "pubmed_rct" / config.PUBMED_RCT_FILES[split]
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python -m src.data` first.")
    rows, abs_id, buf = [], None, []

    def flush():
        for i, (lab, txt) in enumerate(buf):
            rows.append((abs_id, i, len(buf), lab, txt))

    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith("###"):
                if buf:
                    flush()
                abs_id, buf = line[3:], []
            elif line.strip():
                lab, _, txt = line.partition("\t")
                buf.append((lab, txt))
        if buf:
            flush()
    return pd.DataFrame(rows, columns=["abstract_id", "sent_idx", "n_sents", "label", "text"])


def stratified_subset(df: pd.DataFrame, n: int, seed: int = config.SEED) -> pd.DataFrame:
    """Class-stratified random subset of ``n`` sentences (all rows if n >= len(df))."""
    if n >= len(df):
        return df.reset_index(drop=True)
    frac = n / len(df)
    parts = [g.sample(frac=frac, random_state=seed) for _, g in df.groupby("label")]
    return pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)


# ---------------------------------------------------------------------------
# BC5CDR  (token classification)
# ---------------------------------------------------------------------------
def load_bc5cdr(split: str) -> list[dict]:
    """Return a list of {"tokens": [...], "tags": [str, ...]} (BIO strings)."""
    path = config.DATA_RAW / "bc5cdr" / config.BC5CDR_FILES[split]
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python -m src.data` first.")
    id2tag = {v: k for k, v in config.BC5CDR_LABELS.items()}
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rec = json.loads(line)
                out.append({"tokens": rec["tokens"], "tags": [id2tag[t] for t in rec["tags"]]})
    return out


def load_summarization_test() -> pd.DataFrame:
    path = config.DATA_RAW / "pubmed_summarization" / "test.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python -m src.data` first.")
    return pd.read_parquet(path)


if __name__ == "__main__":
    download_all(include_summarization="--no-summ" not in sys.argv)


def eval_subset(test_df: pd.DataFrame) -> pd.DataFrame:
    """The fixed stratified test subset every model is compared on (like-for-like benchmarking)."""
    return stratified_subset(test_df, config.TEST_SUBSET_SENTENCES, config.SEED)
