"""Local medical-text summarization: extractive baseline + chunked abstractive T5.

The abstractive model is *pretrained* ``Falconsai/medical_summarization`` (T5-small, Apache-2.0); MediScope
did not fine-tune it. Weights are loaded from ``artifacts/summarization/t5_medical`` (saved by notebook 08);
nothing is downloaded at request time. Summaries are machine generated and **not medically verified**.
"""
from __future__ import annotations

import threading

import numpy as np

import config
from src.preprocessing import split_sentences
from src.utils import ArtifactMissingError

_MODEL = {}
_LOCK = threading.Lock()
PREFIX = "summarize: "


# ---------------------------------------------------------------------------
# Extractive (no model download): TF-IDF sentence graph + PageRank-style centrality
# ---------------------------------------------------------------------------
def extractive_summary(text: str, n_sentences: int = 5, sentences: list[str] | None = None) -> list[dict]:
    """Return the ``n_sentences`` most central sentences in document order, with centrality scores."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    sents = sentences if sentences is not None else split_sentences(text)
    if len(sents) <= n_sentences:
        return [{"index": i, "text": s, "score": 1.0} for i, s in enumerate(sents)]
    X = TfidfVectorizer(stop_words="english", sublinear_tf=True).fit_transform(sents)
    sim = (X @ X.T).toarray()
    np.fill_diagonal(sim, 0)
    row = sim.sum(1, keepdims=True)
    trans = np.divide(sim, row, out=np.zeros_like(sim), where=row > 0)
    r = np.full(len(sents), 1 / len(sents))
    for _ in range(50):
        r_new = 0.15 / len(sents) + 0.85 * trans.T @ r
        if np.abs(r_new - r).sum() < 1e-8:
            break
        r = r_new
    top = sorted(np.argsort(-r)[:n_sentences])
    return [{"index": int(i), "text": sents[i], "score": float(r[i])} for i in top]


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
def chunk_sentences(sentences: list[str], count_tokens, max_tokens: int = config.SUM_MAX_INPUT_TOKENS) -> list[list[str]]:
    """Greedy sentence packing into chunks of at most ``max_tokens`` (sentences are never split)."""
    chunks, cur, cur_n = [], [], 0
    for s in sentences:
        n = count_tokens(s)
        if cur and cur_n + n > max_tokens:
            chunks.append(cur)
            cur, cur_n = [], 0
        cur.append(s)
        cur_n += n
    if cur:
        chunks.append(cur)
    return chunks


class Summarizer:
    def __init__(self, directory):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self._torch = torch
        self.tok = AutoTokenizer.from_pretrained(directory)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(directory).eval()
        self.name = "T5-small (medical_summarization, pretrained, not fine-tuned here)"

    def count_tokens(self, text: str) -> int:
        return len(self.tok(text, add_special_tokens=False)["input_ids"])

    def _generate(self, texts: list[str], min_new: int, max_new: int, num_beams: int = 2) -> list[str]:
        enc = self.tok([PREFIX + t for t in texts], return_tensors="pt", truncation=True,
                       max_length=config.SUM_MAX_INPUT_TOKENS, padding=True)
        with self._torch.no_grad():
            out = self.model.generate(**enc, min_new_tokens=min_new, max_new_tokens=max_new, num_beams=num_beams,
                                      no_repeat_ngram_size=3, early_stopping=True)
        return [s.strip() for s in self.tok.batch_decode(out, skip_special_tokens=True)]

    def summarize(self, text: str, length: str = config.SUM_DEFAULT_LENGTH, max_chunks: int = 6,
                  refine: bool = True) -> dict:
        if length not in config.SUM_LENGTHS:
            raise ValueError(f"length must be one of {list(config.SUM_LENGTHS)}")
        lo, hi = config.SUM_LENGTHS[length]
        sents = split_sentences(text)
        chunks = chunk_sentences(sents, self.count_tokens)
        truncated = len(chunks) > max_chunks
        used = chunks[:max_chunks]
        parts = []
        for ch in used:
            parts.extend(self._generate([" ".join(ch)], lo, hi))
        combined = " ".join(parts)
        final = combined
        refined = False
        if refine and len(parts) > 1:
            final = self._generate([combined], lo, hi)[0]
            refined = True
        return {
            "summary": final, "chunk_summaries": parts, "n_chunks_total": len(chunks), "n_chunks_used": len(used),
            "truncated": truncated, "refined": refined, "input_tokens": sum(self.count_tokens(" ".join(c)) for c in used),
            "summary_tokens": self.count_tokens(final), "length": length, "model": self.name,
        }


def available() -> bool:
    return (config.ART_SUM / "t5_medical" / "config.json").exists()


def get_summarizer() -> Summarizer:
    with _LOCK:
        if "m" not in _MODEL:
            if not available():
                raise ArtifactMissingError(
                    "The summarization model is not available. Run notebooks/08_medical_summarization.ipynb, "
                    "which saves the pretrained T5 weights to artifacts/summarization/t5_medical/."
                )
            _MODEL["m"] = Summarizer(config.ART_SUM / "t5_medical")
        return _MODEL["m"]


def key_findings(sentences: list[str], labels: list[str], k: int = 3) -> list[dict]:
    """Key findings = most central sentences among those a classifier labelled RESULTS/CONCLUSIONS.

    Purely extractive (verbatim sentences), so nothing is invented.
    """
    idx = [i for i, l in enumerate(labels) if l in ("RESULTS", "CONCLUSIONS")]
    if not idx:
        return []
    picks = extractive_summary("", min(k, len(idx)), sentences=[sentences[i] for i in idx])
    return [{"text": p["text"], "label": labels[idx[p["index"]]]} for p in picks]
