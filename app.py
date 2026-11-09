"""MediScope - local healthcare research text analytics (Flask, server-rendered).

Run with ``python app.py``. All inference is local; there is no REST API - pages are rendered
with Jinja2 from HTML form posts.
"""
from __future__ import annotations

import json
import logging
import secrets
import sqlite3
import threading
import time
from collections import Counter
from datetime import date, datetime
from logging.handlers import RotatingFileHandler

from flask import (Flask, Response, abort, flash, g, redirect, render_template, request, session, url_for)

import config
from src import classification as clf_mod
from src import ner as ner_mod
from src import summarization as sum_mod
from src.explainability import explain_prediction
from src.preprocessing import split_sentences
from src.report import to_pdf, to_txt
from src.utils import (ArtifactMissingError, DocumentError, extract_text, key_terms, load_json, text_stats,
                       validate_text)

log = logging.getLogger("mediscope")


# ---------------------------------------------------------------------------
# History (SQLite)
# ---------------------------------------------------------------------------
def db() -> sqlite3.Connection:
    if "db" not in g:
        config.INSTANCE_DIR.mkdir(exist_ok=True)
        g.db = sqlite3.connect(config.HISTORY_DB)
        g.db.row_factory = sqlite3.Row
        g.db.execute(
            "CREATE TABLE IF NOT EXISTS analyses (id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL,"
            " kind TEXT NOT NULL, title TEXT NOT NULL, chars INTEGER NOT NULL, seconds REAL NOT NULL, result TEXT NOT NULL)"
        )
    return g.db


def compact_for_storage(a: dict) -> dict:
    """Keep results, drop the bulk of the document: sentence text is capped and the file itself is never stored."""
    a = json.loads(json.dumps(a))
    if a.get("classification"):
        for s in a["classification"]["sentences"]:
            s["text"] = s["text"][:240]
    if a.get("ner"):
        a["ner"]["segments"] = a["ner"]["segments"][:300]
        a["ner"].pop("entities", None)
    if a.get("summary"):
        a["summary"].pop("chunk_summaries", None)
    return a


def save_analysis(kind: str, analysis: dict) -> int:
    analysis["kind"], analysis["created_at"] = kind, datetime.now().strftime("%Y-%m-%d %H:%M")
    cur = db().execute(
        "INSERT INTO analyses (created_at, kind, title, chars, seconds, result) VALUES (?,?,?,?,?,?)",
        (analysis["created_at"], kind, analysis["doc"]["title"], analysis["doc"]["stats"]["characters"],
         sum(analysis["timings"].values()), json.dumps(compact_for_storage(analysis))),
    )
    db().execute(
        "DELETE FROM analyses WHERE id NOT IN (SELECT id FROM analyses ORDER BY id DESC LIMIT ?)", (config.HISTORY_MAX_ROWS,))
    db().commit()
    analysis["id"] = cur.lastrowid
    return cur.lastrowid


def load_analysis(aid: int) -> dict:
    row = db().execute("SELECT * FROM analyses WHERE id=?", (aid,)).fetchone()
    if row is None:
        abort(404)
    a = json.loads(row["result"])
    a["id"], a["created_at"], a["kind"] = row["id"], row["created_at"], row["kind"]
    return a


# ---------------------------------------------------------------------------
# Analysis services (thin wrappers around src/ modules)
# ---------------------------------------------------------------------------
def classify(text: str, model: str | None) -> dict:
    sents = split_sentences(text)
    truncated = len(sents) > config.MAX_SENTENCES_CLASSIFY
    sents = sents[: config.MAX_SENTENCES_CLASSIFY]
    clf = clf_mod.get_classifier(model)
    results = clf_mod.classify_sentences(sents, model)
    linear = getattr(clf, "pipe", None) is not None and hasattr(clf.pipe.steps[-1][1], "coef_")
    method = ""
    for i, r in enumerate(results):
        if linear or i < config.EXPLAIN_MAX_SENTENCES:
            ex = explain_prediction(clf, r["text"], clf_mod.LABELS.index(r["label"]))
            r["terms"], method = ex["terms"], ex["method"]
    counts = Counter(r["label"] for r in results)
    labels = [r["label"] for r in results]
    return {
        "model": clf.info(), "sentences": results, "truncated": truncated, "explain_method": method,
        "explained_all": linear, "counts": {l: counts.get(l, 0) for l in clf_mod.LABELS},
        "dominant": counts.most_common(1)[0][0] if counts else None,
        "key_findings": sum_mod.key_findings([r["text"] for r in results], labels),
    }


def recognise(text: str, model: str) -> dict:
    cut = len(text) > config.NER_MAX_CHARS
    res = ner_mod.extract_entities(text[: config.NER_MAX_CHARS], model)
    segs, pos = [], 0
    for e in sorted(res["entities"], key=lambda x: x["start"]):
        if e["start"] < pos:
            continue
        if e["start"] > pos:
            segs.append([res["text"][pos:e["start"]], None, None])
        segs.append([res["text"][e["start"]:e["end"]], e["type"], e["confidence"]])
        pos = e["end"]
    segs.append([res["text"][pos:], None, None])
    res["segments"], res["truncated"] = segs, cut
    res["n_entities"] = len(res["entities"])
    return res


def summarise(text: str, length: str) -> dict:
    s = sum_mod.get_summarizer().summarize(text, length=length, max_chunks=config.SUM_MAX_CHUNKS)
    s["extractive"] = sum_mod.extractive_summary(text, 4)
    return s


def analyse(text: str, title: str, steps: list[str], opts: dict) -> dict:
    """Run the selected modules; a missing artifact disables only that module (never a mock)."""
    a = {"doc": {"title": title, "stats": text_stats(text)}, "key_terms": key_terms(text, 14), "timings": {}, "errors": {},
         "classification": None, "ner": None, "summary": None, "text_preview": text[:1200]}
    runners = {
        "classify": ("classification", lambda: classify(text, opts.get("cls_model"))),
        "ner": ("ner", lambda: recognise(text, opts.get("ner_model", "biobert"))),
        "summarize": ("summary", lambda: summarise(text, opts.get("length", config.SUM_DEFAULT_LENGTH))),
    }
    for step in steps:
        key, fn = runners[step]
        t0 = time.perf_counter()
        try:
            a[key] = fn()
        except ArtifactMissingError as exc:
            a["errors"][key] = str(exc)
        except Exception as exc:  # keep the other modules alive
            log.exception("module %s failed", step)
            a["errors"][key] = f"Processing failed: {exc}"
        a["timings"][key] = time.perf_counter() - t0
    return a


def read_input() -> tuple[str, str]:
    """Text from the textarea or an uploaded PDF/TXT (held in memory only; nothing is written to disk)."""
    f = request.files.get("file")
    if f and f.filename:
        data = f.read(config.MAX_UPLOAD_BYTES + 1)
        return extract_text(f.filename, data), f"Uploaded file: {f.filename[:80]}"
    text = validate_text(request.form.get("text", ""))
    first = text.split("\n", 1)[0][:90]
    return text, first or "Pasted text"


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------
def create_app() -> Flask:
    app = Flask(__name__)
    app.config.update(SECRET_KEY=config.SECRET_KEY, MAX_CONTENT_LENGTH=config.MAX_UPLOAD_BYTES + 2 * 1024 * 1024)
    config.INSTANCE_DIR.mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    fh = RotatingFileHandler(config.INSTANCE_DIR / "mediscope.log", maxBytes=500_000, backupCount=2, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(fh)

    @app.teardown_appcontext
    def close_db(_exc):
        d = g.pop("db", None)
        if d is not None:
            d.close()

    # ---- CSRF (session token checked on every POST) ----
    @app.before_request
    def csrf_protect():
        if request.method == "POST":
            expected = session.get("csrf", "")
            if not expected or not secrets.compare_digest(expected, request.form.get("csrf_token", "")):
                abort(400, "Invalid or missing form token. Reload the page and try again.")

    @app.context_processor
    def inject():
        if "csrf" not in session:
            session["csrf"] = secrets.token_hex(16)
        return {
            "csrf_token": session["csrf"], "status": system_status(), "CLASS_LABELS": config.CLASS_LABELS,
            "limits": {"mb": config.MAX_UPLOAD_BYTES // (1024 * 1024), "chars": config.MAX_TEXT_CHARS, "min": config.MIN_TEXT_CHARS},
            "lengths": list(config.SUM_LENGTHS), "default_length": config.SUM_DEFAULT_LENGTH,
        }

    def system_status() -> dict:
        cls = clf_mod.available_models()
        return {"classifiers": list(cls), "taggers": ner_mod.available_taggers(), "summarizer": sum_mod.available(),
                "deployed": clf_mod.deployed_name(), "ready": bool(cls) and bool(ner_mod.available_taggers()) and sum_mod.available()}

    def page(template, crumb, **ctx):
        return render_template(template, crumb=crumb, **ctx)

    def classifier_options():
        info = load_json(config.ART_CLS / "metrics.json", {}) or {}
        return [{"key": k, "label": info.get(k, {}).get("label", k), "default": k == clf_mod.deployed_name()} for k in clf_mod.available_models()]

    # ---- Overview ----
    @app.get("/")
    def index():
        stats = load_json(config.ART_EVAL / "dataset_stats.json", {}) or {}
        facts = stats.get("facts", [])
        fact = facts[date.today().toordinal() % len(facts)] if facts else None
        metrics = load_json(config.ART_CLS / "metrics.json", {}) or {}
        dep = clf_mod.deployed_name()
        ner_m = load_json(config.ART_NER / "metrics.json", {}) or {}
        sum_m = load_json(config.ART_SUM / "metrics.json", {}) or {}
        rows = db().execute("SELECT id, created_at, kind, title, seconds FROM analyses ORDER BY id DESC LIMIT 6").fetchall()
        total = db().execute("SELECT COUNT(*) FROM analyses").fetchone()[0]
        per_day = db().execute("SELECT substr(created_at,1,10) d, COUNT(*) n FROM analyses GROUP BY d ORDER BY d DESC LIMIT 14").fetchall()
        kinds = {r[0] for r in db().execute("SELECT DISTINCT kind FROM analyses")}
        milestones = [
            {"label": "First analysis", "done": total >= 1}, {"label": "Ten analyses", "done": total >= 10},
            {"label": "All four modules used", "done": {"explorer", "classifier", "ner", "summarizer"} <= kinds},
        ]
        return page("index.html", ["Overview"], fact=fact, deployed=metrics.get(dep) if dep else None, ner=(ner_m.get("biobert") if ner_m else None),
                    summ=(sum_m.get("systems", {}).get("T5-small medical") if sum_m else None), dataset=stats, recent=rows,
                    timeline=list(reversed([dict(r) for r in per_day])), milestones=milestones, total=total)

    # ---- Research Explorer (integrated) ----
    @app.route("/explorer", methods=["GET", "POST"])
    def explorer():
        ctx = {"cls_options": classifier_options(), "analysis": None}
        if request.method == "POST":
            steps = [s for s in request.form.getlist("steps") if s in ("classify", "ner", "summarize")]
            try:
                if not steps:
                    raise DocumentError("Select at least one analysis to run.")
                text, title = read_input()
                a = analyse(text, title, steps, {"cls_model": request.form.get("cls_model") or None, "ner_model": request.form.get("ner_model", "biobert"),
                                                 "length": request.form.get("length", config.SUM_DEFAULT_LENGTH)})
                save_analysis("explorer", a)
                ctx["analysis"], ctx["total_seconds"] = a, sum(a["timings"].values())
            except DocumentError as exc:
                flash(str(exc), "error")
        return page("explorer.html", ["Research Explorer"], **ctx)

    # ---- Abstract Classifier ----
    @app.route("/classifier", methods=["GET", "POST"])
    def classifier():
        ctx = {"cls_options": classifier_options(), "analysis": None}
        if request.method == "POST":
            try:
                text, title = read_input()
                a = analyse(text, title, ["classify"], {"cls_model": request.form.get("cls_model") or None})
                if a["errors"]:
                    flash(a["errors"]["classification"], "error")
                else:
                    save_analysis("classifier", a)
                ctx["analysis"] = a if not a["errors"] else None
            except DocumentError as exc:
                flash(str(exc), "error")
        return page("classifier.html", ["Abstract Classifier"], **ctx)

    # ---- Entity Extraction ----
    @app.route("/ner", methods=["GET", "POST"])
    def ner():
        ctx = {"analysis": None, "taggers": ner_mod.available_taggers()}
        if request.method == "POST":
            try:
                text, title = read_input()
                a = analyse(text, title, ["ner"], {"ner_model": request.form.get("ner_model", "biobert")})
                if a["errors"]:
                    flash(a["errors"]["ner"], "error")
                else:
                    save_analysis("ner", a)
                ctx["analysis"] = a if not a["errors"] else None
            except DocumentError as exc:
                flash(str(exc), "error")
        return page("ner.html", ["Entity Extraction"], **ctx)

    # ---- Summarization Studio ----
    @app.route("/summarizer", methods=["GET", "POST"])
    def summarizer():
        ctx = {"analysis": None, "source": None}
        if request.method == "POST":
            try:
                text, title = read_input()
                a = analyse(text, title, ["summarize"], {"length": request.form.get("length", config.SUM_DEFAULT_LENGTH)})
                if a["errors"]:
                    flash(a["errors"]["summary"], "error")
                else:
                    save_analysis("summarizer", a)
                    ctx["source"] = split_sentences(text)[:120]
                ctx["analysis"] = a if not a["errors"] else None
            except DocumentError as exc:
                flash(str(exc), "error")
        return page("summarizer.html", ["Summarization Studio"], **ctx)

    # ---- Model Laboratory ----
    @app.get("/laboratory")
    def laboratory():
        bench = load_json(config.ART_EVAL / "benchmark.json")
        return page("laboratory.html", ["Model Laboratory"], bench=bench, deployed=load_json(config.ART_CLS / "deployed.json"),
                    ner=load_json(config.ART_NER / "metrics.json"), summ=load_json(config.ART_SUM / "metrics.json"),
                    dataset=load_json(config.ART_EVAL / "dataset_stats.json"), calibration=load_json(config.ART_EVAL / "calibration.json"),
                    ablations=load_json(config.ART_CLS / "ablations.json"), prep=load_json(config.ART_CLS / "preprocessing_choice.json"))

    # ---- Research History ----
    @app.get("/history")
    def history():
        rows = db().execute("SELECT id, created_at, kind, title, chars, seconds FROM analyses ORDER BY id DESC").fetchall()
        return page("history.html", ["Research History"], rows=rows)

    @app.get("/history/<int:aid>")
    def history_view(aid):
        return page("report.html", ["Research History", f"Report #{aid}"], analysis=load_analysis(aid))

    @app.post("/history/<int:aid>/delete")
    def history_delete(aid):
        db().execute("DELETE FROM analyses WHERE id=?", (aid,))
        db().commit()
        flash(f"Report #{aid} deleted.", "ok")
        return redirect(url_for("history"))

    @app.post("/history/clear")
    def history_clear():
        db().execute("DELETE FROM analyses")
        db().commit()
        flash("All history deleted.", "ok")
        return redirect(url_for("history"))

    @app.get("/history/<int:aid>/export.<fmt>")
    def history_export(aid, fmt):
        a = load_analysis(aid)
        if fmt == "txt":
            return Response(to_txt(a), mimetype="text/plain; charset=utf-8",
                            headers={"Content-Disposition": f"attachment; filename=mediscope_report_{aid}.txt"})
        if fmt == "pdf":
            return Response(to_pdf(a), mimetype="application/pdf",
                            headers={"Content-Disposition": f"attachment; filename=mediscope_report_{aid}.pdf"})
        abort(404)

    @app.get("/about")
    def about():
        return page("about.html", ["About"], dataset=load_json(config.ART_EVAL / "dataset_stats.json"))

    # ---- Centralised error handling ----
    @app.errorhandler(ArtifactMissingError)
    def artifact_missing(exc):
        return render_template("error.html", crumb=["Error"], code=503, title="Model not available", message=str(exc)), 503

    @app.errorhandler(413)
    def too_large(_exc):
        return render_template("error.html", crumb=["Error"], code=413, title="File too large",
                               message=f"Uploads are limited to {config.MAX_UPLOAD_BYTES // (1024 * 1024)} MB."), 413

    @app.errorhandler(400)
    @app.errorhandler(404)
    def client_error(exc):
        code = getattr(exc, "code", 400)
        msg = "That page does not exist." if code == 404 else (getattr(exc, "description", None) or "Bad request.")
        return render_template("error.html", crumb=["Error"], code=code, title="Not found" if code == 404 else "Bad request", message=msg), code

    @app.errorhandler(Exception)
    def server_error(exc):
        if hasattr(exc, "code") and isinstance(exc.code, int) and exc.code < 500:
            return exc
        log.exception("Unhandled error")
        return render_template("error.html", crumb=["Error"], code=500, title="Something went wrong",
                               message="The error has been logged locally. Your input was not stored."), 500

    @app.template_filter("pct")
    def pct(v, digits=1):
        return "n/a" if v is None else f"{v * 100:.{digits}f}%"

    @app.template_filter("num")
    def num(v, digits=3):
        return "n/a" if v is None else f"{v:.{digits}f}"

    return app


def preload(app: Flask) -> None:
    """Warm the models once in the background so the first request is not slow."""
    def _go():
        for name, fn in (("classifier", lambda: clf_mod.get_classifier()), ("ner", lambda: ner_mod.get_tagger("biobert")),
                         ("summarizer", sum_mod.get_summarizer)):
            try:
                t0 = time.perf_counter()
                fn()
                log.info("preloaded %s in %.1fs", name, time.perf_counter() - t0)
            except ArtifactMissingError as exc:
                log.warning("%s not available: %s", name, exc)
            except Exception:
                log.exception("failed to preload %s", name)
    threading.Thread(target=_go, daemon=True).start()


app = create_app()

if __name__ == "__main__":
    preload(app)
    print("MediScope running locally at http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False, threaded=True)
