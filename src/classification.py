"""Sentence-role classification: model definitions and local inference.

Every model exposes the same interface (:class:`BaseClassifier`) so the Flask app and the
benchmarking notebook treat classical, BiLSTM and transformer models identically.
Models are loaded lazily, once per process, and cached.

IMPORTANT: returned probabilities are **uncalibrated** model scores. They are not a
statement of certainty (see notebook 09 for measured calibration error).
"""
from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

import config
from src.evaluation import softmax
from src.preprocessing import tokenize
from src.utils import ArtifactMissingError, load_json

LABELS = config.CLASS_LABELS
_CACHE: dict[str, "BaseClassifier"] = {}
_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# BiLSTM definition (trained in notebook 05, loaded here for inference)
# ---------------------------------------------------------------------------
def build_vocab(texts, max_size: int = 30_000, min_freq: int = 2) -> dict[str, int]:
    from collections import Counter

    c = Counter(t for s in texts for t in tokenize(s))
    vocab = {"<pad>": 0, "<unk>": 1}
    for tok, n in c.most_common(max_size):
        if n >= min_freq:
            vocab[tok] = len(vocab)
    return vocab


def encode_texts(texts, vocab: dict[str, int], max_len: int = 64) -> np.ndarray:
    out = np.zeros((len(texts), max_len), dtype=np.int64)
    for i, s in enumerate(texts):
        ids = [vocab.get(t, 1) for t in tokenize(s)][:max_len]
        out[i, : len(ids)] = ids
    return out


def make_bilstm(vocab_size: int, n_classes: int = len(LABELS), emb_dim: int = 100,
                hidden: int = 128, dropout: float = 0.3, pretrained=None):
    import torch
    import torch.nn as nn

    class BiLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
            if pretrained is not None:
                self.emb.weight.data.copy_(torch.as_tensor(pretrained))
            self.lstm = nn.LSTM(emb_dim, hidden, batch_first=True, bidirectional=True)
            self.drop = nn.Dropout(dropout)
            self.fc = nn.Linear(hidden * 2, n_classes)

        def forward(self, x):
            mask = (x != 0).unsqueeze(-1)
            h, _ = self.lstm(self.drop(self.emb(x)))
            h = h.masked_fill(~mask, -1e9).max(dim=1).values  # masked max-pool over time
            return self.fc(self.drop(h))

    return BiLSTM()


# ---------------------------------------------------------------------------
# Unified inference wrappers
# ---------------------------------------------------------------------------
class BaseClassifier:
    name: str = ""
    kind: str = ""
    labels = LABELS
    score_type = "softmax probability (uncalibrated)"

    def predict_proba(self, texts: list[str]) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def info(self) -> dict:
        m = (load_json(config.ART_CLS / "metrics.json", {}) or {}).get(self.name, {})
        return {"name": self.name, "kind": self.kind, "score_type": self.score_type,
                "macro_f1": m.get("f1_macro"), "accuracy": m.get("accuracy"),
                "latency_ms": m.get("latency_ms"), "model_mb": m.get("model_mb")}


class SklearnClassifier(BaseClassifier):
    kind = "classical"

    def __init__(self, name: str, pipeline):
        self.name, self.pipe = name, pipeline
        clf = pipeline.steps[-1][1]
        self.has_proba = hasattr(clf, "predict_proba")
        if not self.has_proba:
            self.score_type = "softmax of SVM margins (uncalibrated, not a probability)"
        classes = list(clf.classes_)
        # XGBoost is trained on integer labels in LABELS order; the others on label strings
        self._order = (list(range(len(LABELS))) if isinstance(classes[0], (int, np.integer))
                       else [classes.index(l) for l in LABELS])

    def predict_proba(self, texts):
        if self.has_proba:
            p = self.pipe.predict_proba(texts)
        else:
            p = softmax(self.pipe.decision_function(texts))
        return p[:, self._order]


class BiLSTMClassifier(BaseClassifier):
    kind = "deep"

    def __init__(self, name: str, directory: Path):
        import torch

        meta = load_json(directory / "config.json")
        self.vocab = load_json(directory / "vocab.json")
        self.max_len = meta["max_len"]
        self.model = make_bilstm(len(self.vocab), emb_dim=meta["emb_dim"], hidden=meta["hidden"])
        self.model.load_state_dict(torch.load(directory / "model.pt", map_location="cpu"))
        self.model.eval()
        self.name = name
        self._torch = torch

    def predict_proba(self, texts):
        x = self._torch.as_tensor(encode_texts(texts, self.vocab, self.max_len))
        with self._torch.no_grad():
            return softmax(self.model(x).numpy())


class TransformerClassifier(BaseClassifier):
    kind = "transformer"

    def __init__(self, name: str, directory: Path, max_len: int = config.INFER_MAX_SEQ_LEN):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(directory)
        self.model = AutoModelForSequenceClassification.from_pretrained(directory).eval()
        self.max_len, self.name, self._torch = max_len, name, torch
        id2label = {int(k): v for k, v in self.model.config.id2label.items()}
        self._order = [[id2label[i] for i in range(len(id2label))].index(l) for l in LABELS]

    def predict_proba(self, texts, batch_size: int = 32):
        out = []
        for i in range(0, len(texts), batch_size):
            enc = self.tok(texts[i : i + batch_size], truncation=True, max_length=self.max_len,
                           padding=True, return_tensors="pt")
            with self._torch.no_grad():
                out.append(softmax(self.model(**enc).logits.numpy()))
        return np.vstack(out)[:, self._order]


# ---------------------------------------------------------------------------
# Registry / loading
# ---------------------------------------------------------------------------
def _train_hint(name: str | None) -> str:
    return (f"Model '{name}' is not available." if name else "No classification model is deployed.") + (
        " Train it by running notebooks/04_classical_ml_experiments.ipynb (classical), "
        "05_deep_learning_experiments.ipynb (BiLSTM) or 06_transformer_finetuning.ipynb "
        "(transformers); notebook 10 selects the deployed model."
    )


def available_models() -> dict[str, dict]:
    """Discover trained classifiers on disk: ``{name: {"kind", "path"}}``."""
    found = {}
    for p in sorted((config.ART_CLS / "models").glob("*.joblib")):
        found[p.stem] = {"kind": "classical", "path": p}
    for d in sorted((config.ART_CLS / "transformers").glob("*/")):
        if (d / "config.json").exists():
            found[d.name] = {"kind": "transformer", "path": d}
    d = config.ART_CLS / "bilstm"
    if (d / "model.pt").exists():
        found["bilstm"] = {"kind": "deep", "path": d}
    return found


def deployed_name() -> str | None:
    meta = load_json(config.ART_CLS / "deployed.json")
    return meta.get("name") if meta else None


def get_classifier(name: str | None = None) -> BaseClassifier:
    """Return a cached classifier; raise :class:`ArtifactMissingError` if unavailable."""
    name = name or deployed_name()
    if not name:
        raise ArtifactMissingError(_train_hint(None))
    with _LOCK:
        if name in _CACHE:
            return _CACHE[name]
        spec = available_models().get(name)
        if spec is None:
            raise ArtifactMissingError(_train_hint(name))
        if spec["kind"] == "classical":
            import joblib

            clf: BaseClassifier = SklearnClassifier(name, joblib.load(spec["path"]))
        elif spec["kind"] == "deep":
            clf = BiLSTMClassifier(name, spec["path"])
        else:
            clf = TransformerClassifier(name, spec["path"])
        _CACHE[name] = clf
        return clf


def classify_sentences(sentences: list[str], name: str | None = None) -> list[dict]:
    """Classify each sentence; returns label, score and the full (uncalibrated) distribution."""
    clf = get_classifier(name)
    probs = clf.predict_proba(sentences)
    results = []
    for s, p in zip(sentences, probs):
        order = np.argsort(-p)
        results.append({
            "text": s,
            "label": LABELS[order[0]],
            "score": float(p[order[0]]),
            "alternatives": [{"label": LABELS[i], "score": float(p[i])} for i in order[1:3]],
            "distribution": {LABELS[i]: float(p[i]) for i in range(len(LABELS))},
        })
    return results


def clear_cache() -> None:
    with _LOCK:
        _CACHE.clear()
