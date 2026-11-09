"""Biomedical NER: dictionary baseline, BIO/subword alignment and BioBERT token-classification inference.

Entity types (BC5CDR): Chemical, Disease.  Models load lazily, once per process.
"""
from __future__ import annotations

import re
import threading
from collections import Counter

import numpy as np

import config
from src.evaluation import softmax
from src.preprocessing import split_sentences, word_tokens_with_spans
from src.utils import ArtifactMissingError, load_json, save_json

TAGS = config.NER_TAGS
TAG2ID = {t: i for i, t in enumerate(TAGS)}
_CACHE: dict[str, object] = {}
_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# BIO <-> sub-word alignment (used by notebook 07 for training, here for inference)
# ---------------------------------------------------------------------------
def encode_for_ner(tokenizer, words: list[str], tags: list[str] | None = None, max_len: int = 128):
    """Tokenize pre-split words and align BIO tags to word-pieces.

    Only the **first** word-piece of each word carries the word's label; continuation pieces and
    special tokens get -100 so the loss ignores them. Returns (encoding, aligned_labels, word_ids).
    """
    enc = tokenizer(words, is_split_into_words=True, truncation=True, max_length=max_len)
    word_ids = enc.word_ids()
    labels, prev = [], None
    for wid in word_ids:
        if wid is None or wid == prev:
            labels.append(-100)
        else:
            labels.append(TAG2ID[tags[wid]] if tags is not None else 0)
        prev = wid
    return enc, labels, word_ids


def tags_to_entities(words: list[str], tags: list[str], probs: list[float] | None = None):
    """BIO tags -> [(type, start_word, end_word_exclusive, confidence)]. A stray I- starts an entity."""
    ents, start, etype = [], None, None
    n = len(tags)
    for i in range(n + 1):
        tag = tags[i] if i < n else "O"
        if tag == "O" or tag.startswith("B-") or (tag.startswith("I-") and tag[2:] != etype):
            if etype is not None:
                conf = float(np.mean(probs[start:i])) if probs else None
                ents.append((etype, start, i, conf))
                etype = None
            if tag != "O":
                etype, start = tag[2:], i
    return ents


# ---------------------------------------------------------------------------
# Dictionary baseline
# ---------------------------------------------------------------------------
class DictionaryNER:
    """Greedy longest-match lookup of mentions seen in training data (no context)."""

    def __init__(self, lexicon: dict[tuple[str, ...], str], max_len: int = 6):
        self.lexicon, self.max_len = lexicon, max_len

    @classmethod
    def from_sentences(cls, sentences: list[dict], max_len: int = 6, min_chars: int = 3):
        # a mention string is assigned the type it carries most often in training
        counts: dict[tuple[str, ...], Counter] = {}
        for rec in sentences:
            for etype, a, b, _ in tags_to_entities(rec["tokens"], rec["tags"]):
                key = tuple(w.lower() for w in rec["tokens"][a:b])
                if 0 < len(key) <= max_len and len("".join(key)) >= min_chars:
                    counts.setdefault(key, Counter())[etype] += 1
        return cls({k: c.most_common(1)[0][0] for k, c in counts.items()}, max_len)

    def predict_tags(self, words: list[str]) -> list[str]:
        low = [w.lower() for w in words]
        tags, i = ["O"] * len(words), 0
        while i < len(words):
            for n in range(min(self.max_len, len(words) - i), 0, -1):
                etype = self.lexicon.get(tuple(low[i : i + n]))
                if etype:
                    tags[i] = f"B-{etype}"
                    for j in range(i + 1, i + n):
                        tags[j] = f"I-{etype}"
                    i += n
                    break
            else:
                i += 1
        return tags

    def save(self, path):
        save_json({"max_len": self.max_len, "lexicon": [[list(k), v] for k, v in self.lexicon.items()]}, path)

    @classmethod
    def load(cls, path):
        d = load_json(path)
        return cls({tuple(k): v for k, v in d["lexicon"]}, d["max_len"])


# ---------------------------------------------------------------------------
# Inference wrappers (common interface: tag_words(words_per_sentence) -> tags, probs)
# ---------------------------------------------------------------------------
class DictionaryTagger:
    name, kind, has_confidence = "dictionary", "baseline", False

    def __init__(self, ner: DictionaryNER):
        self.ner = ner

    def tag_words(self, sents):
        return [self.ner.predict_tags(w) for w in sents], [None] * len(sents)


class BioBertTagger:
    name, kind, has_confidence = "biobert", "transformer", True

    def __init__(self, directory):
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        self._torch = torch
        self.tok = AutoTokenizer.from_pretrained(directory)
        self.model = AutoModelForTokenClassification.from_pretrained(directory).eval()
        self.id2tag = {int(k): v for k, v in self.model.config.id2label.items()}

    def tag_words(self, sents, max_len: int = config.NER_MAX_SEQ_LEN, batch_size: int = 16):
        all_tags, all_probs = [], []
        for i in range(0, len(sents), batch_size):
            batch = sents[i : i + batch_size]
            encs = [encode_for_ner(self.tok, w, None, max_len) for w in batch]
            width = max(len(e[0]["input_ids"]) for e in encs)
            ids = torch_pad([e[0]["input_ids"] for e in encs], width, self.tok.pad_token_id)
            mask = torch_pad([[1] * len(e[0]["input_ids"]) for e in encs], width, 0)
            with self._torch.no_grad():
                logits = self.model(input_ids=self._torch.as_tensor(ids),
                                    attention_mask=self._torch.as_tensor(mask)).logits.numpy()
            probs = softmax(logits)
            for b, (enc, _labels, word_ids) in enumerate(encs):
                n_words = len(batch[b])
                tags, conf, seen = ["O"] * n_words, [0.0] * n_words, set()
                for pos, wid in enumerate(word_ids):
                    if wid is None or wid in seen:
                        continue
                    seen.add(wid)
                    k = int(probs[b, pos].argmax())
                    tags[wid], conf[wid] = self.id2tag[k], float(probs[b, pos, k])
                all_tags.append(tags)
                all_probs.append(conf)
        return all_tags, all_probs


def torch_pad(rows, width, value):
    return np.array([r + [value] * (width - len(r)) for r in rows], dtype=np.int64)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def _hint(which: str) -> str:
    return (f"NER model '{which}' is not available. Train it by running "
            "notebooks/07_biomedical_ner.ipynb (it saves artifacts/ner/dictionary.json and artifacts/ner/biobert/).")


def available_taggers() -> list[str]:
    out = []
    if (config.ART_NER / "dictionary.json").exists():
        out.append("dictionary")
    if (config.ART_NER / "biobert" / "config.json").exists():
        out.append("biobert")
    return out


def get_tagger(name: str = "biobert"):
    with _LOCK:
        if name in _CACHE:
            return _CACHE[name]
        if name not in available_taggers():
            raise ArtifactMissingError(_hint(name))
        tagger = (BioBertTagger(config.ART_NER / "biobert") if name == "biobert"
                  else DictionaryTagger(DictionaryNER.load(config.ART_NER / "dictionary.json")))
        _CACHE[name] = tagger
        return tagger


def extract_entities(text: str, model: str = "biobert") -> dict:
    """Run NER on raw text. Offsets refer to the whitespace-normalised text returned as ``text``."""
    tagger = get_tagger(model)
    norm = re.sub(r"\s+", " ", text or "").strip()
    sents, pos, spans = split_sentences(norm), 0, []
    words_per = []
    for s in sents:
        start = norm.find(s, pos)
        start = pos if start < 0 else start
        toks = [(w, a + start, b + start) for w, a, b in word_tokens_with_spans(s)]
        words_per.append(toks)
        pos = start + len(s)
        spans.append(start)
    tags, confs = tagger.tag_words([[w for w, _, _ in t] for t in words_per])
    entities = []
    for toks, tg, cf in zip(words_per, tags, confs):
        for etype, a, b, conf in tags_to_entities([w for w, _, _ in toks], tg, cf if tagger.has_confidence else None):
            s_char, e_char = toks[a][1], toks[b - 1][2]
            entities.append({"text": norm[s_char:e_char], "type": etype, "start": s_char, "end": e_char,
                             "confidence": conf})
    freq = Counter((e["text"].lower(), e["type"]) for e in entities)
    summary = [{"entity": t, "type": ty, "count": c} for (t, ty), c in freq.most_common()]
    return {
        "text": norm,
        "entities": entities,
        "frequency": summary,
        "category_counts": dict(Counter(e["type"] for e in entities)),
        "model": tagger.name,
        "has_confidence": tagger.has_confidence,
    }


def clear_cache() -> None:
    with _LOCK:
        _CACHE.clear()
