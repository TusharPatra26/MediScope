"""Shared helpers: logging, timing, text statistics, key terms and document handling."""
from __future__ import annotations

import io
import json
import logging
import re
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import config
from src.preprocessing import STOPWORDS, split_sentences, tokenize


class ArtifactMissingError(RuntimeError):
    """Raised when a trained model/artifact is not available. Never replaced by a mock."""


class DocumentError(ValueError):
    """Raised for invalid/unsupported/empty uploaded documents or text."""


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    return logging.getLogger("mediscope")


@contextmanager
def timer():
    """``with timer() as t: ...; t.seconds`` -> wall-clock seconds."""

    class _T:
        seconds = 0.0

    t = _T()
    start = time.perf_counter()
    try:
        yield t
    finally:
        t.seconds = time.perf_counter() - start


def set_seed(seed: int = config.SEED) -> None:
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def save_json(obj, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=float), encoding="utf-8")


def load_json(path: Path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def update_json(path: Path, key: str, value) -> None:
    """Merge ``{key: value}`` into a JSON dict file (used to accumulate per-model metrics)."""
    data = load_json(path, {}) or {}
    data[key] = value
    save_json(data, path)


# ---------------------------------------------------------------------------
# Text statistics and key terms
# ---------------------------------------------------------------------------
def text_stats(text: str) -> dict:
    sents = split_sentences(text)
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9'\-]*", text)
    uniq = {w.lower() for w in words}
    n_words = len(words)
    return {
        "characters": len(text),
        "words": n_words,
        "sentences": len(sents),
        "unique_words": len(uniq),
        "lexical_diversity": round(len(uniq) / n_words, 3) if n_words else 0.0,
        "avg_sentence_words": round(n_words / len(sents), 1) if sents else 0.0,
        "avg_word_length": round(sum(map(len, words)) / n_words, 2) if n_words else 0.0,
        "numbers": len(re.findall(r"\b\d+(?:\.\d+)?%?\b", text)),
        "reading_minutes": round(n_words / 230, 1),
    }


def key_terms(text: str, top_k: int = 12) -> list[dict]:
    """Frequency-ranked content unigrams and bigrams (stopwords/negation-aware, no model needed)."""
    toks = [t for t in tokenize(text) if t.isalpha() and len(t) > 2]
    content = [t for t in toks if t not in STOPWORDS]
    uni = Counter(content)
    bi = Counter(
        f"{a} {b}" for a, b in zip(toks, toks[1:]) if a not in STOPWORDS and b not in STOPWORDS
    )
    terms = [{"term": t, "count": c, "kind": "bigram"} for t, c in bi.items() if c >= 2]
    terms += [{"term": t, "count": c, "kind": "unigram"} for t, c in uni.items()]
    terms.sort(key=lambda d: (-d["count"], d["kind"] != "bigram", d["term"]))
    return terms[:top_k]


# ---------------------------------------------------------------------------
# Document handling (PDF / TXT)
# ---------------------------------------------------------------------------
def validate_text(text: str) -> str:
    """Normalise whitespace and enforce min/max length. Returns clean text."""
    text = (text or "").replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) < config.MIN_TEXT_CHARS:
        raise DocumentError(
            f"The text is too short ({len(text)} characters). Provide at least "
            f"{config.MIN_TEXT_CHARS} characters of research text."
        )
    if len(text) > config.MAX_TEXT_CHARS:
        raise DocumentError(
            f"The text is too long ({len(text):,} characters). The limit is "
            f"{config.MAX_TEXT_CHARS:,} characters."
        )
    return text


def extract_text(filename: str, data: bytes) -> str:
    """Extract text from an uploaded PDF/TXT held in memory. Nothing is executed or written."""
    ext = Path(filename or "").suffix.lower()
    if ext not in config.ALLOWED_EXTENSIONS:
        raise DocumentError(
            f"Unsupported file type '{ext or 'unknown'}'. Upload a "
            + " or ".join(sorted(e.lstrip(".").upper() for e in config.ALLOWED_EXTENSIONS))
            + " file."
        )
    if not data:
        raise DocumentError("The uploaded file is empty.")
    if len(data) > config.MAX_UPLOAD_BYTES:
        raise DocumentError(
            f"File is too large ({len(data) / 1e6:.1f} MB). Limit is "
            f"{config.MAX_UPLOAD_BYTES / 1e6:.0f} MB."
        )
    if ext == ".txt":
        for enc in ("utf-8", "utf-16", "latin-1"):
            try:
                text = data.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        else:  # pragma: no cover - latin-1 never fails
            raise DocumentError("Could not decode the text file.")
    else:
        if not data.startswith(b"%PDF"):
            raise DocumentError("The file does not look like a valid PDF.")
        try:
            from pypdf import PdfReader

            reader = PdfReader(io.BytesIO(data))
            if reader.is_encrypted:
                raise DocumentError("Encrypted PDFs are not supported.")
            text = "\n".join((page.extract_text() or "") for page in reader.pages)
        except DocumentError:
            raise
        except Exception as exc:  # corrupt PDF, etc.
            raise DocumentError(f"Could not read the PDF: {exc}") from exc
        if len(text.strip()) < config.MIN_TEXT_CHARS:
            raise DocumentError(
                "No extractable text found in this PDF (it may be a scanned image). "
                "OCR is not supported."
            )
    return validate_text(text)
