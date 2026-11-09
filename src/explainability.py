"""Model transparency helpers.

* ``linear_terms``  - exact per-term contributions of a linear TF-IDF model (coef x tf-idf value).
* ``occlusion_terms`` - model-agnostic leave-one-word-out attribution (works for transformers/BiLSTM).

Attributions describe what the *model* relied on, not medical causation.
"""
from __future__ import annotations

import numpy as np


def linear_terms(pipeline, text: str, label_idx_in_classes: int, k: int = 8) -> list[dict]:
    """Top contributing n-grams for one class of a ``Pipeline([tfidf, linear-clf])``."""
    vec, clf = pipeline.steps[0][1], pipeline.steps[-1][1]
    if not hasattr(clf, "coef_"):
        raise ValueError("classifier has no coef_; use occlusion_terms instead")
    X = vec.transform([text])
    names = vec.get_feature_names_out()
    coefs = clf.coef_[label_idx_in_classes]
    idx = X.indices
    contrib = X.data * coefs[idx]
    order = np.argsort(-np.abs(contrib))[:k]
    return [{"term": str(names[idx[i]]), "weight": float(contrib[i])} for i in order]


def occlusion_terms(predict_proba, text: str, target_idx: int, k: int = 8, max_words: int = 80) -> list[dict]:
    """Drop each word in turn; attribution = decrease in the target-class score (positive = supports class)."""
    words = text.split()
    if not words:
        return []
    words = words[:max_words]
    base = predict_proba([" ".join(words)])[0][target_idx]
    variants = [" ".join(words[:i] + words[i + 1:]) for i in range(len(words))]
    probs = predict_proba(variants)[:, target_idx]
    deltas = base - probs
    order = np.argsort(-np.abs(deltas))[:k]
    return [{"term": words[i], "weight": float(deltas[i])} for i in order]


def explain_prediction(clf, text: str, label_idx: int, k: int = 8) -> dict:
    """Pick the best available method for a loaded classifier from ``src.classification``."""
    pipe = getattr(clf, "pipe", None)
    if pipe is not None and hasattr(pipe.steps[-1][1], "coef_"):
        classes = list(pipe.steps[-1][1].classes_)
        from src.classification import LABELS

        pos = classes.index(LABELS[label_idx]) if not isinstance(classes[0], (int, np.integer)) else label_idx
        return {"method": "linear coefficient x TF-IDF value (exact for this model)", "terms": linear_terms(pipe, text, pos, k)}
    return {"method": "occlusion (leave-one-word-out change in class score)",
            "terms": occlusion_terms(clf.predict_proba, text, label_idx, k)}
