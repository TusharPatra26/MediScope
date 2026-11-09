"""Model behaviour: missing artifacts raise (never mock), plus real-model checks when artifacts exist."""
import numpy as np
import pytest

import config
from src import classification as C
from src import ner as N
from src import summarization as S
from src.explainability import occlusion_terms
from src.utils import ArtifactMissingError

HAS_CLS = bool(C.available_models()) and C.deployed_name() is not None
HAS_NER = "biobert" in N.available_taggers()
HAS_SUM = S.available()
ABSTRACT = ("Hypertension is a major risk factor for stroke. We aimed to test whether amlodipine lowers blood pressure. "
            "Patients were randomly assigned to amlodipine or placebo for 8 weeks. Systolic pressure fell by 12 mmHg with amlodipine "
            "versus 3 mmHg with placebo (p < 0.001). Amlodipine effectively lowered blood pressure in this population.")


# ---------------------------------------------------------------- missing artifacts
def test_missing_classifier_raises_with_actionable_hint(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "ART_CLS", tmp_path)
    C.clear_cache()
    with pytest.raises(ArtifactMissingError, match="notebooks/0"):
        C.get_classifier("does_not_exist")
    with pytest.raises(ArtifactMissingError, match="No classification model"):
        C.get_classifier()


def test_missing_ner_and_summarizer_raise(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "ART_NER", tmp_path)
    monkeypatch.setattr(config, "ART_SUM", tmp_path)
    N.clear_cache()
    S._MODEL.clear()
    with pytest.raises(ArtifactMissingError, match="07_biomedical_ner"):
        N.extract_entities("Aspirin causes nausea.", "biobert")
    with pytest.raises(ArtifactMissingError, match="08_medical_summarization"):
        S.get_summarizer()


# ---------------------------------------------------------------- deterministic logic (no trained model needed)
def test_dictionary_ner_longest_match_and_offsets(monkeypatch, tmp_path):
    lex = N.DictionaryNER({("renal", "failure"): "Disease", ("renal",): "Disease", ("cisplatin",): "Chemical"})
    tags = lex.predict_tags(["Cisplatin", "caused", "renal", "failure", "."])
    assert tags == ["B-Chemical", "O", "B-Disease", "I-Disease", "O"]
    path = tmp_path / "d.json"
    lex.save(path)
    assert N.DictionaryNER.load(path).lexicon == lex.lexicon


def test_ner_alignment_marks_only_first_subword():
    from transformers import AutoTokenizer

    try:
        tok = AutoTokenizer.from_pretrained(config.HF_BIOBERT_TOKENIZER, local_files_only=True)
    except Exception:
        pytest.skip("bert-base-cased tokenizer not cached")
    enc, labels, wids = N.encode_for_ner(tok, ["nephrotoxicity", "occurred"], ["B-Disease", "O"], 32)
    assert labels[0] == -100 and labels[-1] == -100               # [CLS]/[SEP]
    first = [i for i, w in enumerate(wids) if w == 0]
    assert labels[first[0]] == N.TAG2ID["B-Disease"] and all(labels[i] == -100 for i in first[1:])
    assert len(first) > 1                                          # word really was split into pieces


def test_tags_to_entities_confidence_mean():
    ents = N.tags_to_entities(["a", "b", "c"], ["B-Disease", "I-Disease", "O"], [0.9, 0.7, 0.99])
    assert ents == [("Disease", 0, 2, pytest.approx(0.8))]


def test_chunking_never_splits_sentences_and_respects_budget():
    sents = [f"Sentence number {i} has several words in it." for i in range(40)]
    chunks = S.chunk_sentences(sents, lambda s: len(s.split()), max_tokens=30)
    assert all(sum(len(s.split()) for s in c) <= 30 for c in chunks)
    assert [s for c in chunks for s in c] == sents


def test_extractive_summary_returns_requested_count_in_order():
    text = " ".join(f"Study {i} measured outcome {i % 3} in patients with condition {i % 2}." for i in range(12))
    out = S.extractive_summary(text, 3)
    assert len(out) == 3 and [o["index"] for o in out] == sorted(o["index"] for o in out)
    assert len(S.extractive_summary("Only one sentence here.", 3)) == 1


def test_key_findings_only_from_results_and_conclusions():
    sents = ["Background text.", "We tested X.", "X improved Y by 10%.", "X is effective."]
    labels = ["BACKGROUND", "OBJECTIVE", "RESULTS", "CONCLUSIONS"]
    kf = S.key_findings(sents, labels, 2)
    assert {k["label"] for k in kf} <= {"RESULTS", "CONCLUSIONS"} and S.key_findings(sents, ["METHODS"] * 4) == []


def test_occlusion_attributes_the_decisive_word():
    def fake(texts):  # a transparent scoring function, not a model substitute: P(class0) rises with 'significant'
        p = np.array([0.9 if "significant" in t.split() else 0.2 for t in texts])
        return np.column_stack([p, 1 - p])

    top = occlusion_terms(fake, "the difference was significant overall", 0, k=1)
    assert top[0]["term"] == "significant" and top[0]["weight"] > 0.5


# ---------------------------------------------------------------- real trained models
@pytest.mark.skipif(not HAS_CLS, reason="no trained classifier in artifacts/ (run notebooks 04-06, 10)")
def test_deployed_classifier_outputs_valid_distribution_and_sensible_roles():
    res = C.classify_sentences([s for s in __import__("src.preprocessing", fromlist=["x"]).split_sentences(ABSTRACT)])
    assert len(res) == 5
    for r in res:
        assert abs(sum(r["distribution"].values()) - 1) < 1e-4 and r["score"] == max(r["distribution"].values())
        assert r["label"] in config.CLASS_LABELS and len(r["alternatives"]) == 2
    labels = [r["label"] for r in res]
    assert labels[0] in ("BACKGROUND", "OBJECTIVE") and labels[-1] in ("CONCLUSIONS", "RESULTS")
    assert "RESULTS" in labels[3]  # the statistics sentence


@pytest.mark.skipif(not HAS_CLS, reason="no trained classifier")
def test_classifier_is_cached_singleton():
    assert C.get_classifier() is C.get_classifier()


@pytest.mark.skipif(not HAS_NER, reason="no BioBERT NER artifact (run notebook 07)")
def test_biobert_ner_output_structure():
    r = N.extract_entities("Cisplatin-induced nephrotoxicity and hypertension were reported after amlodipine.", "biobert")
    assert r["model"] == "biobert" and r["has_confidence"]
    for e in r["entities"]:
        assert r["text"][e["start"]:e["end"]] == e["text"] and e["type"] in config.NER_ENTITY_TYPES and 0 <= e["confidence"] <= 1
    assert sum(r["category_counts"].values()) == len(r["entities"])


@pytest.mark.skipif(not HAS_SUM, reason="no summarizer artifact (run notebook 08)")
def test_summarizer_length_control_and_long_document_chunking():
    s = S.get_summarizer()
    long_doc = " ".join([ABSTRACT] * 12)
    short = s.summarize(ABSTRACT, "short")
    long_ = s.summarize(ABSTRACT, "long")
    assert short["summary"] and short["summary_tokens"] <= long_["summary_tokens"] + 5
    big = s.summarize(long_doc, "short", max_chunks=2)
    assert big["n_chunks_total"] >= 2 and big["n_chunks_used"] == 2 and big["truncated"] is (big["n_chunks_total"] > 2)
