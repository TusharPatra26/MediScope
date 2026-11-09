"""Shared text preprocessing.

The *same* functions are used by the training notebooks and by the Flask app, so
training/inference skew cannot creep in. Design goals for biomedical text:

* never drop negations (``no``, ``not``, ``without`` ...) - they flip meaning;
* keep numbers, decimals, percentages and dosage expressions (``5 mg``, ``p<0.05``);
* no dependence on downloaded NLTK corpora for the deployed path.
"""
from __future__ import annotations

import re
import unicodedata

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------
_ABBREV = (
    "et al", "e.g", "i.e", "vs", "fig", "figs", "approx", "ca", "cf", "dr", "no", "nos",
    "mr", "ms", "st", "inc", "ltd", "co", "p.o", "i.v", "i.m", "s.c", "b.i.d", "t.i.d",
    "q.d", "mg", "ml", "kg", "pts", "pt", "resp", "temp", "conc", "ref", "refs",
)
_SENT_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])")


def split_sentences(text: str) -> list[str]:
    """Rule-based sentence splitter that protects common medical abbreviations/decimals."""
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return []
    sentences, start = [], 0
    for m in _SENT_BOUNDARY.finditer(text):
        candidate = text[start : m.start()].strip()
        last_word = re.findall(r"([A-Za-z.]+)\.$", candidate)
        if last_word and last_word[-1].lower().rstrip(".") in _ABBREV:
            continue
        if re.search(r"\b[A-Z]\.$", candidate):  # single initial, e.g. "J. Smith"
            continue
        sentences.append(candidate)
        start = m.end()
    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences


# ---------------------------------------------------------------------------
# Normalisation, tokenisation
# ---------------------------------------------------------------------------
NEGATIONS = {
    "no", "not", "nor", "none", "never", "neither", "cannot", "without", "n't",
    "nobody", "nothing", "nowhere", "against", "free", "absence", "lack", "fail", "failed",
}
# sklearn's list minus negations and a few clinically meaningful words
_KEEP = NEGATIONS | {"more", "less", "most", "least", "before", "after", "between", "per", "over", "under", "above", "below", "high", "low", "full", "front", "empty"}
STOPWORDS = frozenset(w for w in ENGLISH_STOP_WORDS if w not in _KEEP)

_TOKEN_RE = re.compile(
    r"""
    \d+(?:[.,]\d+)*%?           # numbers, decimals, percentages
    | [A-Za-z]+(?:[-'][A-Za-z0-9]+)*   # words incl. hyphenated / contractions
    | [A-Za-z0-9]+              # alnum fall-back (e.g. IL6)
    | [<>=≤≥±]   # comparison operators matter (p < 0.05)
    """,
    re.VERBOSE,
)
_UNIT_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(mg|g|mcg|µg|ug|ml|l|iu|mmhg|kg|cm|mm|mmol|mol|bpm|hz|h|hr|hrs|min|d|wk|mo|yr|yrs)\b",
    re.IGNORECASE,
)
DOSAGE_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s?(?:mg|g|mcg|µg|ug|ml|iu|mmhg|kg|mmol|mol)(?:/(?:kg|day|d|h|hr|ml|l|m2))?\b",
    re.IGNORECASE,
)
_NUM_RE = re.compile(r"\d+(?:[.,]\d+)*")

# Small curated abbreviation map (illustrative, used by the 'expand' strategy)
ABBREVIATIONS = {
    "rct": "randomized controlled trial", "bp": "blood pressure", "hr": "hazard ratio",
    "ci": "confidence interval", "or": "odds ratio", "bmi": "body mass index",
    "hba1c": "glycated hemoglobin", "icu": "intensive care unit", "mi": "myocardial infarction",
    "copd": "chronic obstructive pulmonary disease", "cvd": "cardiovascular disease",
    "sd": "standard deviation", "iqr": "interquartile range", "hiv": "human immunodeficiency virus",
}


def normalize(text: str) -> str:
    """Unicode/whitespace normalisation that does not change content words or numbers."""
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("−", "-").replace("–", "-").replace("—", " - ")
    text = re.sub(r"[‘’]", "'", text)
    text = re.sub(r"[“”]", '"', text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str, lower: bool = True) -> list[str]:
    toks = _TOKEN_RE.findall(normalize(text))
    return [t.lower() for t in toks] if lower else toks


STRATEGIES = ("raw", "clean", "numshape", "nostop", "stem", "expand")
DEFAULT_STRATEGY = "clean"


def _stem(tokens: list[str]) -> list[str]:
    from nltk.stem import PorterStemmer  # algorithmic stemmer, needs no corpus download

    ps = PorterStemmer()
    return [ps.stem(t) if t.isalpha() else t for t in tokens]


def preprocess(text: str, strategy: str = DEFAULT_STRATEGY) -> str:
    """Return a whitespace-joined token string for the chosen strategy.

    raw      lowercase only (no tokenisation changes)
    clean    normalise + regex tokenise, keep numbers/units/negations/operators
    numshape clean + map every number to a shape token (``NUM``/``PCT``)
    nostop   clean + remove stopwords (negations are *kept*)
    stem     clean + Porter stemming
    expand   clean + expand a few medical abbreviations (RCT, BP, CI ...)
    """
    if strategy == "raw":
        return (text or "").lower()
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown preprocessing strategy {strategy!r}; choose from {STRATEGIES}")
    text = normalize(text)
    text = _UNIT_RE.sub(lambda m: f"{m.group(1)} {m.group(2).lower()}", text)
    tokens = tokenize(text)
    if strategy == "numshape":
        tokens = ["pct" if t.endswith("%") else "num" if _NUM_RE.fullmatch(t) else t for t in tokens]
    elif strategy == "nostop":
        tokens = [t for t in tokens if t not in STOPWORDS]
    elif strategy == "stem":
        tokens = _stem(tokens)
    elif strategy == "expand":
        out: list[str] = []
        for t in tokens:
            out.extend(ABBREVIATIONS.get(t, t).split())
        tokens = out
    return " ".join(tokens)


def preprocess_many(texts, strategy: str = DEFAULT_STRATEGY) -> list[str]:
    return [preprocess(t, strategy) for t in texts]


def extract_dosages(text: str) -> list[str]:
    """Dosage-like expressions such as ``5 mg`` or ``10 mg/kg`` (used for UI term highlights)."""
    return [m.group(0) for m in DOSAGE_RE.finditer(normalize(text))]


def word_tokens_with_spans(text: str) -> list[tuple[str, int, int]]:
    """Word-level tokens with character offsets, punctuation split as in BC5CDR.

    Used by NER so predicted BIO spans map back to the original string.
    """
    return [(m.group(0), m.start(), m.end()) for m in re.finditer(r"\w+(?:[-']\w+)*|[^\w\s]", text)]
