"""Feature construction: TF-IDF, handcrafted features, Word2Vec document vectors."""
from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

import config
from src.preprocessing import preprocess, tokenize


class Preprocessor:
    """Picklable callable wrapping ``preprocess`` (lambdas cannot be saved with joblib)."""

    def __init__(self, strategy: str = "clean"):
        self.strategy = strategy

    def __call__(self, text: str) -> str:
        return preprocess(text, self.strategy)


def whitespace_tokenizer(text: str) -> list[str]:
    return text.split()


def build_tfidf(
    strategy: str = "clean",
    ngram_range: tuple[int, int] = (1, 2),
    max_features: int | None = 200_000,
    min_df: int = 2,
    sublinear_tf: bool = True,
) -> TfidfVectorizer:
    """TF-IDF vectoriser whose analyzer applies the shared ``preprocess`` function.

    Because preprocessing happens inside the vectoriser, a fitted pipeline carries its own
    preprocessing and the app cannot apply a different one at inference time.
    """
    return TfidfVectorizer(
        preprocessor=Preprocessor(strategy),
        tokenizer=whitespace_tokenizer,
        token_pattern=None,
        lowercase=False,
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=min_df,
        sublinear_tf=sublinear_tf,
        dtype=np.float32,
    )


def train_word2vec(texts, vector_size: int = 100, window: int = 5, min_count: int = 3,
                   epochs: int = 10, seed: int = config.SEED, workers: int = 1):
    """Train gensim Word2Vec (skip-gram) on tokenised sentences. ``workers=1`` for determinism."""
    from gensim.models import Word2Vec

    sentences = [tokenize(t) for t in texts]
    return Word2Vec(sentences, vector_size=vector_size, window=window, min_count=min_count,
                    sg=1, epochs=epochs, seed=seed, workers=workers)


def mean_word_vector(model, text: str) -> np.ndarray:
    vecs = [model.wv[t] for t in tokenize(text) if t in model.wv]
    return np.mean(vecs, axis=0) if vecs else np.zeros(model.vector_size, dtype=np.float32)


def doc_vectors(model, texts) -> np.ndarray:
    return np.vstack([mean_word_vector(model, t) for t in texts])


def position_features(sent_idx, n_sents) -> np.ndarray:
    """Relative position of a sentence inside its abstract (strongly predictive in PubMed RCT)."""
    sent_idx, n_sents = np.asarray(sent_idx, float), np.asarray(n_sents, float)
    rel = sent_idx / np.maximum(n_sents - 1, 1)
    return np.column_stack([rel, (sent_idx == 0).astype(float), (sent_idx == n_sents - 1).astype(float)])
