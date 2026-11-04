"""Create portable reports from the exact completed run selected in the studio."""
import csv
import io
import json
from pathlib import Path
from threading import Lock
from xml.sax.saxutils import escape

import reportlab
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Table, TableStyle

TITLE = "Macroeconomic Analysis Studio with LLM prompt writing"
_font_lock = Lock()


def _decisions(audit):
    return [e for e in audit.get("events", []) if e.get("phase") != "working"]


def _metadata(r):
    a = r["audit"]
    yield "Summary", r["summary"]
    yield "Interpretation", r["interpretation"]
    yield "Data", r["source"]
    yield "Mode", a["agent_mode"]
    yield "Prompt", a["original_prompt"]
    if a.get("analysis_prompt", a["original_prompt"]) != a["original_prompt"]:
        yield "Prompt used for analysis", a["analysis_prompt"]
    if a.get("writing_choice"):
        c = a["writing_choice"]
        yield "Suggested wording", c["suggestion"]
        yield "Wording choice", c["action"]
        yield "Selected wording", c["used"]
    for index, exchange in enumerate(a.get("prompt_clarifications", []), 1):
        yield f"Clarification {index} question", exchange["question"]
        yield f"Clarification {index} answer", exchange["answer"]
    for change in r["feedback"]:
        yield f"Feedback for version {change['version']}", change["text"]
        if change.get("used_text", change["text"]) != change["text"]:
            yield f"Feedback used for version {change['version']}", change["used_text"]
    if r.get("previous_summary"):
        yield "Previous result", r["previous_summary"]
        for change in r["changes"]:
            yield "Change in this version", change
    if r["task"] == "forecast":
        yield "Holdout start", a["holdout_start"]
        yield "Holdout end", a["holdout_end"]
        yield "Months scored", a["holdout_months_scored"]
        yield "Forecast month", a["next_month"]["month"]
        yield "Point forecast", a["next_month"]["forecast"]
        for name, value in a["weights"].items():
            yield f"{name} weight", value
    elif r["task"] in ("cross_dml","ts_dml"):
        for key in ("target","term","controls","n","folds","uncertainty","verdict","limits"):
            yield key, a[key]
        if a["estimate"] is not None:
            for key,value in a["estimate"].items():
                yield key,value
    elif r["task"]=="family_forecast":
        for key in ("target","mode","base","learner","choice","validation_span","forecast_month","next_components"):
            yield key,a[key]
    elif r["task"] == "exploration":
        for key in ("target","trained","validation","reserved","common_months","recommended_model"):
            yield key,a[key]
        for key in ("holdout_rmse","holdout_scored_months","forecast_month","forecast_value"):
            if key in a:
                yield key,a[key]
        for name, score in a["models"].items():
            yield name + " validation RMSE", score
    elif r["task"] == "rcf_dml_screen":
        for key in ("target", "term", "controls", "n", "calendar_months", "first_month", "last_month", "limits"):
            yield key, a[key]
    else:
        for key in ("target", "term", "n", "first_month", "last_month", "hac_lags"):
            yield key, a[key]


def csv_report(r):
    audit = r["audit"]
    rows = [dict(section="run", label=k, value=v) for k, v in _metadata(r)]
    rows.extend({"section": "results", **dict(zip(r["columns"], row))} for row in r["rows"])
    for row in r.get("holdout_forecasts", []):
        rows.append({"section": "holdout", **row})
    for e in _decisions(audit):
        rows.append(dict(section="review", round=e["round"], role=e["role"],
                         decision=e.get("decision", ""), value=e["message"],
                         code=e.get("code", ""), suggested_prompt=e.get("suggested_prompt", ""), criteria=json.dumps(e["criteria"], ensure_ascii=False) if "criteria" in e else "", evidence=json.dumps(e["evidence"]) if "evidence" in e else ""))
    if not _decisions(audit):
        rows.append(dict(section="review", value="Python reference run. No agent review."))
    rows.append(dict(section="program", code=audit["python_script"]))
    columns = ["run_id", "version", *dict.fromkeys(k for row in rows for k in row)]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns)
    writer.writeheader()
    for row in rows:
        row = {"run_id": r["id"], "version": r["version"], **row}
        # Keep numerical values numeric while treating user-supplied text as spreadsheet text.
        writer.writerow({k: "'" + v if isinstance(v, str) and v.lstrip().startswith(("=", "+", "-", "@"))
                         else v for k, v in row.items()})
    return buffer.getvalue().encode("utf-8-sig")


def _fonts():
    with _font_lock:
        if "StudioSans" not in pdfmetrics.getRegisteredFontNames():
            fonts = Path(reportlab.__file__).parent / "fonts"
            for name, file in [("StudioSans", "Vera.ttf"), ("StudioBold", "VeraBd.ttf")]:
                pdfmetrics.registerFont(TTFont(name, str(fonts/file)))


def pdf_report(r):
    _fonts()
    ink, teal, muted = "#193038", "#176552", "#48615d"
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter, leftMargin=43, rightMargin=43,
                            topMargin=54, bottomMargin=48, title=TITLE)
    widths = letter[0] - doc.leftMargin - doc.rightMargin - 12
    styles = {
        "title": ("StudioBold", 24, 30, ink), "head": ("StudioBold", 13, 18, teal),
        "body": ("StudioSans", 9, 13, ink), "label": ("StudioBold", 8, 12, muted),
        "cell": ("StudioSans", 8, 11, ink), "col": ("StudioBold", 7.5, 10, "#ffffff"),
        "code": ("Courier", 7, 10, ink)}
    styles = {k: ParagraphStyle(k, fontName=f, fontSize=s, leading=l, textColor=colors.HexColor(c),
                                spaceAfter=6 if k in ("title", "head", "body") else 1,
                                spaceBefore=9 if k=="head" else 0, keepWithNext=k in ("head","label"),
                                splitLongWords=True) for k,(f,s,l,c) in styles.items()}

    def para(value, kind="body"):
        safe = escape("".join(c for c in str(value) if ord(c) >= 32 or c in "\t\n"))
        return Paragraph(safe.replace("\n", "<br/>"), styles[kind])

    def table(headers, body, fractions):
        formatted = [[para(str(h).replace("_", " "), "col") for h in headers]]
        for row in body:
            formatted.append([para(f"{v:.4f}" if isinstance(v, float) else v, "cell") for v in row])
        t = Table(formatted, colWidths=[widths*f for f in fractions], repeatRows=1, hAlign="LEFT")
        t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor(teal)),
                               ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#eff5f2")]),
                               ("VALIGN",(0,0),(-1,-1),"TOP"),("TOPPADDING",(0,0),(-1,-1),4),
                               ("BOTTOMPADDING",(0,0),(-1,-1),4)]))
        return t

    a = r["audit"]
    story = [para(f"VERSION {r['version']}  /  {a['agent_mode']}", "label"),
             para({"family_forecast":"Forecast method comparison","exploration":"Time series exploration and model choice","forecast":"Forecast results","rcf_dml_screen":"Time-series DML suitability","cross_dml":"Cross-sectional DML","ts_dml":"Monthly time-series DML"}.get(r["task"], "Estimation results"), "title"),
             para(r["summary"]), para(r["interpretation"]), para(r["source"], "label")]
    if r["feedback"]:
        story.append(para("User feedback and revision", "head"))
        for f in r["feedback"]:
            story.append(para(f"Version {f['version']}  /  {f['text']}"))
            if f.get("used_text", f["text"]) != f["text"]:
                story.append(para("Feedback used: " + f["used_text"]))
        story += [para("Previous result", "label"), para(r["previous_summary"])]
        story += [para(item) for item in r["changes"]]
    if r["task"]=="forecast":
        story.append(para("Holdout performance", "head"))
        story.append(para(f"{a['holdout_start']} to {a['holdout_end']}  /  {a['holdout_months_scored']} scored months"))
        story.append(para("Ensemble weights: " + ", ".join(f"{k.replace('_',' ')} {v:.1%}" for k,v in a["weights"].items())))
    elif r["task"] in ("cross_dml","ts_dml"):
        if a["estimate"] is not None:
            story.append(table(list(a["estimate"]),[list(a["estimate"].values())],[.25]*4))
        story.append(para("Design assessment", "head"))
    elif r["task"]=="family_forecast":
        story.append(para("Validation, holdout and next forecast","head"))
        if a["next_components"]:
            parts=a["next_components"]
            story.append(para(f"Next classical forecast {parts['Baseline']:,.3f} plus ML correction {parts['ML correction']:+,.3f}."))
    elif r["task"] == "exploration":
        story += [para("Features and rolling validation", "head"), para("Initial training: " + a["trained"] +
                 ". Validation: " + a["validation"] + ". Common months scored: " + str(a["common_months"]) + ".")]
    elif r["task"] == "rcf_dml_screen":
        story.append(para("Evidence for the three conditions", "head"))
    else:
        story.append(para("Coefficient estimates and 95% intervals", "head"))
        story.append(para(f"Sample: {a['first_month']} to {a['last_month']}, {a['n']} months. HAC bandwidth: {a['hac_lags']}."))
    ncols = len(r["columns"])
    widths_table = [.28,.26,.46] if r["task"]=="exploration" else [.26,.20,.54] if r["task"] in ("rcf_dml_screen","cross_dml","ts_dml") else [.32]+[.68/(ncols-1)]*(ncols-1)
    story.append(table(r["columns"], r["rows"], widths_table))
    story += [para("Research prompt", "head"), para(a["original_prompt"])]
    if a.get("analysis_prompt", a["original_prompt"]) != a["original_prompt"]:
        story += [para("Prompt used for analysis", "label"), para(a["analysis_prompt"])]
    if a.get("prompt_clarifications"):
        story.append(para("Clarifications used for this analysis", "head"))
        for item in a["prompt_clarifications"]:
            story += [para(item["question"], "label"), para(item["answer"])]
    story.append(para("Agent review", "head"))
    for e in _decisions(a):
        decision = e.get("decision") or ("Code" if e.get("code") else "Response")
        stage = f"ROUND {e['round']}" if e["round"] else "PROMPT REVIEW"
        story += [para(f"{stage}   /   {e['role']}   /   {decision.replace('_',' ').upper()}", "label"),
                  para(e["message"])]
        if e.get("suggested_prompt"):
            story += [para("Suggested wording", "label"), para(e["suggested_prompt"])]
        if e.get("criteria"):
            story.append(para("\n".join(f"{key.title()}: {value}" for key,value in e["criteria"].items())))
    if not _decisions(a):
        story.append(para("Python reference run. No agent review."))
    for round_info in a.get("rounds", []):
        ev = round_info.get("evidence", {})
        if ev.get("features"):
            story += [para(f"Round {round_info['round']} features", "label"), para(", ".join(ev["features"]))]
    story += [para("Executed Python", "head"), para(a["python_script"], "code")]
    months = r.get("holdout_forecasts", [])
    if months:
        story.append(para("Forecasts by target month", "head"))
        headers = list(months[0])
        story.append(table(headers, [[row[k] for k in headers] for row in months], [1/len(headers)]*len(headers)))

    def page(c, document):
        c.setFillColor(colors.HexColor(teal))
        c.setFont("StudioBold",8); c.drawString(43,letter[1]-31,TITLE)
        c.setStrokeColor(colors.HexColor("#cddbd5")); c.line(43,letter[1]-38,letter[0]-43,letter[1]-38)
        c.setFillColor(colors.HexColor(muted)); c.setFont("StudioSans",7)
        c.drawString(43,27, f"Analysis {r['id'][:8]}  /  Version {r['version']}")
        c.drawRightString(letter[0]-43,27,f"Page {document.page}")
    doc.build(story,onFirstPage=page,onLaterPages=page)
    return buf.getvalue()
