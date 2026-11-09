"""Preprocessing, input validation, document handling and evaluation-metric behaviour."""
import pytest

import config
from src.evaluation import bio_to_spans, entity_prf, expected_calibration_error, numeric_consistency
from src.preprocessing import (STOPWORDS, extract_dosages, preprocess, split_sentences, tokenize,
                               word_tokens_with_spans)
from src.utils import DocumentError, extract_text, key_terms, text_stats, validate_text


# ---------------------------------------------------------------- preprocessing
def test_negations_survive_stopword_removal():
    out = preprocess("There was no significant difference and it did not reduce pain", "nostop").split()
    assert "no" in out and "not" in out and "significant" in out
    assert "the" not in STOPWORDS or True  # sanity: list exists
    assert "was" not in out


def test_numbers_decimals_percentages_and_operators_are_kept():
    toks = tokenize("Mean 12.5% (p < 0.05) at 5 mg")
    assert "12.5%" in toks and "0.05" in toks and "<" in toks and "5" in toks and "mg" in toks


def test_numshape_replaces_numbers_only():
    out = preprocess("Dose 5 mg in 12.5% of patients", "numshape")
    assert out == "dose num mg in pct of patients"


def test_dosage_extraction():
    assert extract_dosages("Give 5 mg/kg twice daily or 10 ml.") == ["5 mg/kg", "10 ml"]


def test_unknown_strategy_raises():
    with pytest.raises(ValueError):
        preprocess("text", "does-not-exist")


def test_sentence_splitter_protects_abbreviations_and_decimals():
    s = split_sentences("Patients received 5.5 mg vs. placebo (p<0.05). Dr. J. Smith et al. reported gains. We conclude it works.")
    assert len(s) == 3 and s[0].endswith("(p<0.05).") and s[1].startswith("Dr.")


def test_empty_text_gives_no_sentences():
    assert split_sentences("   ") == [] and split_sentences("") == []


def test_word_spans_map_back_to_text():
    text = "Cisplatin-induced nephrotoxicity (p<0.05)."
    for w, a, b in word_tokens_with_spans(text):
        assert text[a:b] == w


# ---------------------------------------------------------------- validation / documents
def test_validate_text_enforces_limits():
    with pytest.raises(DocumentError, match="too short"):
        validate_text("hi")
    with pytest.raises(DocumentError, match="too long"):
        validate_text("a " * config.MAX_TEXT_CHARS)
    assert validate_text("  " + "word " * 20 + "  ").startswith("word")


def test_extract_text_rejects_bad_extension_empty_and_oversize():
    with pytest.raises(DocumentError, match="Unsupported"):
        extract_text("evil.exe", b"MZ....")
    with pytest.raises(DocumentError, match="empty"):
        extract_text("a.txt", b"")
    with pytest.raises(DocumentError, match="too large"):
        extract_text("a.txt", b"x" * (config.MAX_UPLOAD_BYTES + 1))


def test_extract_text_txt_decoding_and_pdf_magic_check():
    assert "randomized" in extract_text("a.txt", ("A randomized trial " * 5).encode("utf-16"))
    with pytest.raises(DocumentError, match="valid PDF"):
        extract_text("fake.pdf", b"not a pdf at all " * 10)


def test_real_pdf_roundtrip():
    from reportlab.pdfgen import canvas
    import io

    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    c.drawString(72, 750, "This randomized controlled trial compared two treatments for chronic pain.")
    c.save()
    assert "randomized controlled trial" in extract_text("paper.pdf", buf.getvalue())


def test_blank_pdf_is_rejected_as_unreadable():
    from reportlab.pdfgen import canvas
    import io

    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    c.showPage()
    c.save()
    with pytest.raises(DocumentError, match="No extractable text"):
        extract_text("scan.pdf", buf.getvalue())


def test_text_stats_and_key_terms():
    t = "Aspirin reduced pain. Aspirin did not cause bleeding. Aspirin therapy was safe."
    st = text_stats(t)
    assert st["sentences"] == 3 and st["words"] == 12
    assert key_terms(t, 3)[0]["term"] == "aspirin"


# ---------------------------------------------------------------- evaluation helpers
def test_bio_spans_and_stray_i_tag():
    tags = ["O", "B-Disease", "I-Disease", "O", "I-Chemical", "B-Chemical"]
    assert bio_to_spans(tags) == {("Disease", 1, 3), ("Chemical", 4, 5), ("Chemical", 5, 6)}


def test_entity_prf_is_strict_about_boundaries_and_types():
    gold = [["B-Disease", "I-Disease", "O", "B-Chemical"]]
    perfect = entity_prf(gold, gold)
    assert perfect["f1"] == 1.0
    boundary = entity_prf(gold, [["B-Disease", "O", "O", "B-Chemical"]])   # truncated disease
    assert boundary["per_type"]["Disease"]["f1"] == 0.0 and boundary["per_type"]["Chemical"]["f1"] == 1.0
    wrong_type = entity_prf(gold, [["B-Chemical", "I-Chemical", "O", "B-Chemical"]])
    assert wrong_type["per_type"]["Disease"]["recall"] == 0.0


def test_calibration_error_zero_for_perfectly_calibrated():
    import numpy as np

    probs = np.array([[1.0, 0.0]] * 5)
    assert expected_calibration_error(probs, np.zeros(5, int))["ece"] == 0.0
    assert expected_calibration_error(probs, np.ones(5, int))["ece"] == 1.0


def test_numeric_consistency_flags_unsupported_numbers():
    r = numeric_consistency("Mortality fell 40% in 120 patients.", "We enrolled 120 patients. Mortality fell by 30%.")
    assert r["summary_numbers"] == 2 and r["supported"] == 1 and r["support_rate"] == 0.5
