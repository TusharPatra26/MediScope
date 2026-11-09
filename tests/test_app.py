"""Flask routes, uploads, history, exports and error handling (isolated temp DB)."""
import io
import re

import pytest

import config
from app import create_app
from src import classification as C
from src import ner as N
from src import summarization as S

TEXT = ("Hypertension is a major risk factor for stroke. We aimed to test whether amlodipine lowers blood pressure. "
        "Patients were randomly assigned to amlodipine or placebo for 8 weeks. Systolic pressure fell by 12 mmHg with amlodipine "
        "versus 3 mmHg with placebo (p < 0.001). Amlodipine effectively lowered blood pressure in this population.")


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "INSTANCE_DIR", tmp_path)
    monkeypatch.setattr(config, "HISTORY_DB", tmp_path / "history.db")
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


def token(client, path="/explorer"):
    html = client.get(path).get_data(as_text=True)
    return re.search(r'name="csrf_token" value="([0-9a-f]+)"', html).group(1)


def post(client, path, data, **kw):
    data = {"csrf_token": token(client, path), **data}
    return client.post(path, data=data, content_type="multipart/form-data", **kw)


@pytest.mark.parametrize("path", ["/", "/explorer", "/classifier", "/ner", "/summarizer", "/laboratory", "/history", "/about"])
def test_every_page_renders(client, path):
    r = client.get(path)
    assert r.status_code == 200 and b"MediScope" in r.data and b"Making medical research" in r.data


def test_unknown_page_uses_custom_404(client):
    r = client.get("/nope")
    assert r.status_code == 404 and b"does not exist" in r.data


def test_post_without_csrf_token_is_rejected(client):
    assert client.post("/classifier", data={"text": TEXT}).status_code == 400


def test_empty_and_short_input_are_rejected_with_message(client):
    r = post(client, "/classifier", {"text": "   "})
    assert r.status_code == 200 and b"too short" in r.data
    r = post(client, "/explorer", {"text": TEXT})            # no analysis selected
    assert b"Select at least one analysis" in r.data


def test_too_long_text_rejected(client):
    r = post(client, "/classifier", {"text": "word " * (config.MAX_TEXT_CHARS // 4)})
    assert b"too long" in r.data


def test_invalid_upload_type_and_fake_pdf(client):
    r = post(client, "/classifier", {"file": (io.BytesIO(b"MZ binary"), "malware.exe")})
    assert b"Unsupported file type" in r.data
    r = post(client, "/classifier", {"file": (io.BytesIO(b"definitely not a pdf" * 5), "paper.pdf")})
    assert b"valid PDF" in r.data
    r = post(client, "/classifier", {"file": (io.BytesIO(b""), "empty.txt")})
    assert b"empty" in r.data or b"too short" in r.data


def test_oversized_upload_returns_413(client):
    big = io.BytesIO(b"a" * (config.MAX_UPLOAD_BYTES + 3 * 1024 * 1024))
    r = post(client, "/classifier", {"file": (big, "big.txt")})
    assert r.status_code == 413 and b"too large" in r.data.lower()


def test_missing_models_show_clear_message_not_mock_output(client, monkeypatch, tmp_path):
    monkeypatch.setattr(config, "ART_CLS", tmp_path)
    monkeypatch.setattr(config, "ART_NER", tmp_path)
    monkeypatch.setattr(config, "ART_SUM", tmp_path)
    C.clear_cache(); N.clear_cache(); S._MODEL.clear()
    r = post(client, "/explorer", {"text": TEXT, "steps": ["classify", "ner", "summarize"]})
    html = r.get_data(as_text=True)
    assert r.status_code == 200
    assert "No classification model is deployed" in html and "07_biomedical_ner" in html and "08_medical_summarization" in html
    assert "Dominant role" not in html and "<mark" not in html          # nothing was fabricated


# ---------------------------------------------------------------- real inference end-to-end
needs_all = pytest.mark.skipif(not (C.available_models() and C.deployed_name() and "biobert" in N.available_taggers() and S.available()),
                               reason="trained artifacts missing (run notebooks)")


@needs_all
def test_full_explorer_flow_history_and_exports(client):
    r = post(client, "/explorer", {"text": TEXT, "steps": ["classify", "ner", "summarize"], "length": "short"})
    html = r.get_data(as_text=True)
    assert r.status_code == 200 and "Research structure" in html and "Biomedical entities" in html and "not medically verified" in html
    assert 'class="sent"' in html and "uncalibrated" in html.lower()
    hist = client.get("/history").get_data(as_text=True)
    aid = int(re.search(r"/history/(\d+)\"", hist).group(1))
    assert client.get(f"/history/{aid}").status_code == 200
    txt = client.get(f"/history/{aid}/export.txt")
    assert txt.status_code == 200 and b"MediScope Research Report" in txt.data and b"Disclaimer" in txt.data
    pdf = client.get(f"/history/{aid}/export.pdf")
    assert pdf.status_code == 200 and pdf.data.startswith(b"%PDF")
    assert client.get(f"/history/{aid}/export.docx").status_code == 404
    d = client.post(f"/history/{aid}/delete", data={"csrf_token": token(client, "/history")}, follow_redirects=True)
    assert b"deleted" in d.data and client.get(f"/history/{aid}").status_code == 404


@needs_all
def test_text_upload_is_processed_and_not_stored(client, tmp_path):
    r = post(client, "/classifier", {"file": (io.BytesIO(TEXT.encode()), "abstract.txt")})
    assert r.status_code == 200 and b"Abstract structure" in r.data
    assert not list(tmp_path.glob("*.txt"))                              # nothing written to disk besides the DB/log


@needs_all
def test_ner_page_highlights_entities(client):
    r = post(client, "/ner", {"text": "Cisplatin caused nephrotoxicity and hypertension in treated patients with cancer.", "ner_model": "biobert"})
    assert b'class="ent' in r.data or b"No chemical or disease" in r.data
