"""Task-appropriate evaluation: classification, calibration, entity-level NER, summarization."""
from __future__ import annotations

import time
from collections import defaultdict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
def classification_metrics(y_true, y_pred, labels: list[str]) -> dict:
    """Accuracy, macro/weighted P/R/F1, per-class scores and the confusion matrix."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    out = {"accuracy": float(accuracy_score(y_true, y_pred)), "n": int(len(y_true))}
    for avg in ("macro", "weighted"):
        p, r, f, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, average=avg, zero_division=0
        )
        out.update({f"precision_{avg}": float(p), f"recall_{avg}": float(r), f"f1_{avg}": float(f)})
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    out["per_class"] = {
        lab: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
        for i, lab in enumerate(labels)
    }
    out["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    out["labels"] = list(labels)
    return out


def expected_calibration_error(probs: np.ndarray, y_true_idx: np.ndarray, n_bins: int = 10) -> dict:
    """Top-label ECE with equal-width bins. ``probs`` is (n, k); ``y_true_idx`` are class indices."""
    probs = np.asarray(probs)
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == np.asarray(y_true_idx)).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    ece, bins = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        acc, c = correct[m].mean(), conf[m].mean()
        ece += m.mean() * abs(acc - c)
        bins.append({"lo": float(lo), "hi": float(hi), "n": int(m.sum()), "accuracy": float(acc), "confidence": float(c)})
    return {"ece": float(ece), "bins": bins}


def softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def measure_latency_ms(fn, batch, repeats: int = 5, warmup: int = 1) -> float:
    """Median wall-clock latency (ms) per *item* when calling ``fn(batch)``."""
    for _ in range(warmup):
        fn(batch)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(batch)
        times.append((time.perf_counter() - t0) * 1000 / max(len(batch), 1))
    return float(np.median(times))


# ---------------------------------------------------------------------------
# NER: entity-level (span) evaluation
# ---------------------------------------------------------------------------
def bio_to_spans(tags: list[str]) -> set[tuple[str, int, int]]:
    """BIO tags -> {(type, start, end_exclusive)}. A stray I- starts a new entity (CoNLL convention)."""
    spans, start, etype = set(), None, None
    for i, tag in enumerate(list(tags) + ["O"]):
        if tag.startswith("B-") or tag == "O" or (tag.startswith("I-") and tag[2:] != etype):
            if etype is not None:
                spans.add((etype, start, i))
                etype = None
            if tag != "O":
                etype, start = tag[2:], i
    return spans


def entity_prf(gold: list[list[str]], pred: list[list[str]]) -> dict:
    """Strict entity-level precision/recall/F1: type *and* exact boundaries must match.

    Sequences are evaluated independently (span offsets are per-sentence), counts are summed
    over the corpus, then micro-averaged overall and reported per entity type.
    """
    tp, fp, fn = defaultdict(int), defaultdict(int), defaultdict(int)
    for g, p in zip(gold, pred):
        gs, ps = bio_to_spans(g), bio_to_spans(p)
        for t, *_ in gs & ps:
            tp[t] += 1
        for t, *_ in ps - gs:
            fp[t] += 1
        for t, *_ in gs - ps:
            fn[t] += 1

    def prf(a, b, c):
        p_ = a / (a + b) if a + b else 0.0
        r_ = a / (a + c) if a + c else 0.0
        return p_, r_, (2 * p_ * r_ / (p_ + r_) if p_ + r_ else 0.0)

    types = sorted(set(tp) | set(fp) | set(fn))
    per_type = {}
    for t in types:
        p_, r_, f_ = prf(tp[t], fp[t], fn[t])
        per_type[t] = {"precision": p_, "recall": r_, "f1": f_, "support": tp[t] + fn[t]}
    P, R, F = prf(sum(tp.values()), sum(fp.values()), sum(fn.values()))
    return {
        "precision": P, "recall": R, "f1": F,
        "n_gold": int(sum(tp.values()) + sum(fn.values())),
        "n_pred": int(sum(tp.values()) + sum(fp.values())),
        "per_type": per_type,
    }


# ---------------------------------------------------------------------------
# Summarization
# ---------------------------------------------------------------------------
def rouge_scores(predictions: list[str], references: list[str]) -> dict:
    """Mean ROUGE-1/2/L F-measure (stemmed) over a list of pairs."""
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    agg = defaultdict(list)
    for pred, ref in zip(predictions, references):
        s = scorer.score(ref, pred)
        for k, v in s.items():
            agg[k].append(v.fmeasure)
    return {k: float(np.mean(v)) for k, v in agg.items()}


def numeric_consistency(summary: str, source: str) -> dict:
    """Cheap factual-consistency probe: share of numbers in the summary that occur in the source.

    This is a *proxy* for hallucination of statistics, not a proof of factuality.
    """
    import re

    num = re.compile(r"\d+(?:\.\d+)?")
    s_nums, src_nums = set(num.findall(summary)), set(num.findall(source))
    if not s_nums:
        return {"summary_numbers": 0, "supported": 0, "support_rate": None}
    sup = len(s_nums & src_nums)
    return {"summary_numbers": len(s_nums), "supported": sup, "support_rate": sup / len(s_nums)}
