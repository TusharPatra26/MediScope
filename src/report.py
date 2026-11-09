"""Export of an analysis (the dict stored in history) to plain text or PDF."""
from __future__ import annotations

import io
import textwrap

DISCLAIMER = (
    "MediScope is an educational research tool. Outputs are automated, uncalibrated model predictions and "
    "machine-generated text; they are not medical advice, diagnoses or verified facts."
)


def _lines(a: dict) -> list[tuple[str, str]]:
    """Flatten an analysis into (style, text) lines shared by both exporters."""
    out: list[tuple[str, str]] = [("title", "MediScope Research Report"), ("meta", f"Generated: {a.get('created_at', '')}  |  Analysis #{a.get('id', '-')}")]
    d = a.get("doc", {})
    out += [("h", "Document overview"), ("p", d.get("title", ""))]
    st = d.get("stats", {})
    if st:
        out.append(("p", ", ".join(f"{k.replace('_', ' ')}: {v}" for k, v in st.items())))
    c = a.get("classification")
    if c:
        out += [("h", "Research structure (sentence roles)"),
                ("p", f"Model: {c['model']['name']} ({c['model']['kind']}). Scores are {c['model']['score_type']}.")]
        out.append(("p", "Role counts: " + ", ".join(f"{k} {v}" for k, v in c["counts"].items())))
        for i, s in enumerate(c["sentences"], 1):
            out.append(("li", f"[{s['label']} {s['score']:.2f}] {s['text']}"))
        if c.get("key_findings"):
            out.append(("h2", "Key findings (verbatim sentences classified RESULTS/CONCLUSIONS)"))
            out += [("li", k["text"]) for k in c["key_findings"]]
    n = a.get("ner")
    if n:
        out += [("h", "Biomedical entities"), ("p", f"Model: {n['model']}. Counts: " + (", ".join(f"{k} {v}" for k, v in n["category_counts"].items()) or "none"))]
        out += [("li", f"{e['entity']} ({e['type']}) x{e['count']}") for e in n["frequency"][:40]]
    s = a.get("summary")
    if s:
        out += [("h", "Generated summary"), ("p", s["summary"]),
                ("meta", f"Model: {s['model']}. Length: {s['length']}. Chunks used: {s['n_chunks_used']}/{s['n_chunks_total']}"
                         + (" (document truncated)" if s.get("truncated") else "") + ". Not medically verified.")]
    if a.get("key_terms"):
        out += [("h", "Key terminology"), ("p", ", ".join(f"{t['term']} ({t['count']})" for t in a["key_terms"]))]
    out += [("h", "Processing"), ("p", ", ".join(f"{k}: {v:.2f}s" for k, v in a.get("timings", {}).items()))]
    for k, msg in a.get("errors", {}).items():
        out.append(("p", f"[{k} unavailable] {msg}"))
    out += [("h", "Disclaimer"), ("p", DISCLAIMER)]
    return out


def to_txt(a: dict) -> str:
    buf = []
    for style, text in _lines(a):
        if style == "title":
            buf += [text, "=" * len(text)]
        elif style in ("h", "h2"):
            buf += ["", text, "-" * len(text)]
        elif style == "li":
            buf += textwrap.wrap(text, 96, initial_indent="  - ", subsequent_indent="    ")
        else:
            buf += textwrap.wrap(text, 100) or [""]
    return "\n".join(buf) + "\n"


def to_pdf(a: dict) -> bytes:
    from xml.sax.saxutils import escape

    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

    ink, forest = colors.HexColor("#14282A"), colors.HexColor("#0E6B63")
    st = {
        "title": ParagraphStyle("t", fontName="Times-Bold", fontSize=22, leading=26, textColor=forest, spaceAfter=4),
        "meta": ParagraphStyle("m", fontName="Helvetica", fontSize=8, leading=11, textColor=colors.HexColor("#5A6F6D"), spaceAfter=6),
        "h": ParagraphStyle("h", fontName="Times-Bold", fontSize=14, leading=18, textColor=forest, spaceBefore=12, spaceAfter=4),
        "h2": ParagraphStyle("h2", fontName="Times-Italic", fontSize=11, leading=14, textColor=ink, spaceBefore=6, spaceAfter=2),
        "p": ParagraphStyle("p", fontName="Helvetica", fontSize=9.5, leading=13, textColor=ink, spaceAfter=3),
        "li": ParagraphStyle("li", fontName="Helvetica", fontSize=9, leading=12, leftIndent=10, textColor=ink),
    }
    story = []
    for style, text in _lines(a):
        story.append(Paragraph(escape(text), st[style]))
        if style == "title":
            story.append(Spacer(1, 2 * mm))
    buf = io.BytesIO()
    SimpleDocTemplate(buf, pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm, topMargin=18 * mm, bottomMargin=18 * mm,
                      title="MediScope Research Report", author="MediScope").build(story)
    return buf.getvalue()
