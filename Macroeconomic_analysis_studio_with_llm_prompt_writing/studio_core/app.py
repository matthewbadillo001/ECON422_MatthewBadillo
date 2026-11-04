"""A local-only web form. One streaming request owns one analysis and its trace."""
import io
from html import escape
import json
import re
import threading
import webbrowser
from collections import OrderedDict
from uuid import uuid4
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files

import pandas as pd

from studio_core.example_prompt import RETAIL_PROMPTS, RETAIL_CODE, RETAIL_SOURCE, RETAIL_ID, EXPLORE_PROMPTS, exploration_code
from hybrid_forecasting import BASES, LEARNERS, family_development, family_forecast
from studio_core.family_prompts import PRESETS as FAMILY_PRESETS, family_intent, template as family_template, default_config
from studio_core.family_prompts import FIELDS as FAMILY_FIELDS, CONFIGS as FAMILY_CONFIGS, fred_task, fred_family_config
from exploration import monthly_series, explore_series
from studio_core.data_io import load_retail_demo, retail_sample, data_structure
from studio_core.example_prompt import RETAIL_FIELDS, RETAIL_FRED_FIELDS, EXPLORE_FIELDS, EXPLORE_FRED_FIELDS, csv_explore_fields
from studio_core.example_prompt import REGRESSION_FIELDS, REGRESSION_PROMPT, REGRESSION_CODE
from studio_core.analysis_tools import monthly_features, ols_hac, rcf_dml_screen, super_learner
from studio_core.data_io import load_demo, prepare_airfare_sample
from dml import cross_dml, cross_section, ts_dml, load_cross_demo, load_activity_demo
from studio_core.example_prompt import CROSS_FIELDS, CROSS_PROMPT, CROSS_SCREEN_FIELDS, CROSS_SCREEN_PROMPT, TS_EST_FIELDS, TS_EST_FRED_FIELDS, TS_EST_PROMPT, TS_EST_FRED_PROMPT, ESTIMATION_EXAMPLES
from studio_core.example_prompt import (AIRFARE_FRED_COLUMNS, AIRFARE_FRED_PROMPT, DML_CODE, DML_FIELDS, DML_FRED_FIELDS, DML_FRED_PROMPT, DML_PROMPT, FORECAST_CODE, FORECAST_FIELDS,
                             FORECAST_PROMPT, FIELD_HELP, INCOMPLETE_CLARITY, ROUGH_FIELDS,
                             airfare_january_example, assembled_fields, fred_starter)
from studio_core.sources import import_fred, requested_source, resembles_fred_key
from studio_core.web_sources import PROVIDERS, import_web_source
from studio_core.reporting import csv_report, pdf_report
from studio_core.studio import BUNDLED_EXAMPLES, MONTHLY_DEMO, PromptNeedsAnswer, PromptRewriteOffered, StudioAgents, analyze, parse_program, replay_script


def present(result):
    audit = result.audit
    report = {"task": audit["task"], "columns": [result.metrics.index.name, *result.metrics.columns],
              "interpretation": audit["interpretation"], "rows": result.metrics.reset_index().values.tolist(), "audit": audit}
    if audit["task"] == "forecast":
        future = audit["next_month"]
        forecast = future["forecast"]
        detail = (f'{forecast:.3f} in target units' if forecast is not None else future["reason"])
        rmse = result.metrics.loc["super_learner", "RMSE"]
        report["summary"] = (f'{audit["accepted_plan"]["target"]} forecast for {future["month"]}: '
                             f'{detail}. Holdout RMSE: {rmse:.3f} over {audit["holdout_months_scored"]} months.')
        report["holdout_forecasts"] = result.predictions.rename_axis("month").reset_index().astype(
            {"month": str}).to_dict("records")
    elif audit["task"]=="family_forecast":
        report["summary"]=audit["summary"]
        report["holdout_forecasts"]=audit["holdout_predictions"]
        if audit.get("rounds") and audit["next_components"]:
            parts=audit["next_components"]
            total=parts["Baseline"]+parts["ML correction"]
            report["summary"]+=(f" For {audit['forecast_month']}, the classical forecast is {parts['Baseline']:,.3f} "
                                f"and the ML correction is {parts['ML correction']:+,.3f}, giving {total:,.3f}.")
    elif audit["task"] == "exploration":
        report["summary"] = audit["verdict"]
        report["interpretation"] = audit["interpretation"]
    elif audit["task"] == "rcf_dml_screen":
        report["summary"] = audit["verdict"]
        sample = (f'{audit["first_month"]} to {audit["last_month"]}' if audit["n"] else "no usable span")
        report["interpretation"] = (f'Y = {audit["target"]}; D = {audit["term"]}. Controls: '
            f'{", ".join(audit["controls"]) or "intercept only"}. Longest contiguous complete span: '
            f'{sample} ({audit["n"]} of {audit["calendar_months"]} calendar months). {audit["limits"]}')
    elif audit["task"] in ("cross_dml","ts_dml"):
        estimate = audit["estimate"]
        report["summary"] = audit["verdict"]
        if estimate is not None:
            report["summary"] += (f' Coefficient on {audit["term"]}: {estimate["coefficient"]:.4f} '
                                  f'(nominal 95% interval {estimate["lower_95"]:.4f} to {estimate["upper_95"]:.4f}).')
        if audit["task"] == "ts_dml":
            sample = f'{audit["n"]} contiguous months ({audit["first_month"]} through {audit["last_month"]}).'
            if estimate is None:
                report["summary"] += " Coefficient and interval withheld."
        else:
            if estimate is None and audit["requested_estimate"]:
                report["summary"] += " Coefficient and interval withheld."
            sample = f'{audit["n"]} complete independent units from {audit["input_rows"]} input rows.'
        report["interpretation"] = (f'Y = {audit["target"]}, D = {audit["term"]}. Controls: '
            f'{", ".join(audit["controls"]) or "intercept only"}. {sample} {audit["uncertainty"]}. {audit["limits"]}')
    else:
        term = audit["term"]
        row = result.metrics.loc[term]
        report["summary"] = (f'For {audit["target"]}, the adjusted coefficient on {term} is '
                             f'{row.estimate:.3f} (95% HAC interval {row.lower_95:.3f} to '
                             f'{row.upper_95:.3f}, n = {audit["n"]}).')
    return report


def reference_forecast(data=None, *, prompt=None, code=FORECAST_CODE, source=None, external=None):
    (_, features), (_, options) = parse_program(code)
    result = super_learner(monthly_features(load_demo() if data is None else data, **features), **options)
    if external is None:
        external = data is not None and source not in (RETAIL_SOURCE, MONTHLY_DEMO)
    weights = result.audit["weights"]
    explanation = (f"The reference forecast uses {weights['lasso']:.1%} LASSO and "
                   f"{weights['random_forest']:.1%} random forest, learned from earlier months. "
                   "The table compares one-month-ahead errors over the final 24 months.")
    result.audit.update(agent_mode="Python reference only", original_prompt=prompt or (FORECAST_PROMPT if data is None else AIRFARE_FRED_PROMPT),
                        reference_code=code,
                        python_script=replay_script(code, require_csv=external, retail_demo=source==RETAIL_SOURCE, retail_window=prompt in RETAIL_PROMPTS,
                                                    prepare_airfare=external and code == FORECAST_CODE), interpretation=explanation)
    return result


def reference_family(data,prompt,source,code):
    (_,features),(_,opts)=parse_program(code)
    result=family_forecast(monthly_series(data,**features),**opts)
    result.audit.update(agent_mode="Python reference only",original_prompt=prompt,reference_code=code,
                        python_script=replay_script(code,require_csv=source not in (RETAIL_SOURCE, "Bundled simulated AR activity"),
                                                    retail_demo=source==RETAIL_SOURCE, activity_demo=source=="Bundled simulated AR activity"))
    return result


def reference_exploration(data, prompt, source, code):
    (_, features), (_, options) = parse_program(code)
    result = explore_series(monthly_series(data, **features), **options)
    result.audit.update(agent_mode="Python reference only", original_prompt=prompt, reference_code=code,
                        python_script=replay_script(code, require_csv=source!=RETAIL_SOURCE, retail_demo=source==RETAIL_SOURCE, retail_window=prompt in EXPLORE_PROMPTS))
    return result


def approved_code(report):
    rounds = report["audit"].get("rounds", [])
    if rounds:
        return rounds[-1]["validated_code"]
    if "reference_code" in report["audit"]:
        return report["audit"]["reference_code"]
    if report["task"] in ("cross_dml","ts_dml"):
        return ESTIMATION_EXAMPLES[report["audit"]["original_prompt"]]
    return DML_CODE if report["task"] == "rcf_dml_screen" else FORECAST_CODE


def reference_screen(data, prompt, code=DML_CODE, *, external=None):
    if external is None:
        external = data is not None
    (_, features), (_, options) = parse_program(code)
    result = rcf_dml_screen(monthly_features(load_demo() if data is None else data, **features), **options)
    result.audit.update(agent_mode="Python reference only", original_prompt=prompt,
                        reference_code=code, python_script=replay_script(code, require_csv=external, prepare_airfare=prompt==DML_FRED_PROMPT))
    return result


def reference_dml(data, prompt, external, activity_demo=False, *, code=None):
    code = code or ESTIMATION_EXAMPLES[prompt]
    (constructor, features), (method, options) = parse_program(code)
    if constructor == "cross_section":
        result = cross_dml(cross_section(data, **features), **options)
    else:
        result = ts_dml(monthly_features(data, **features), **options)
    result.audit.update(agent_mode="Python reference only", original_prompt=prompt,
                        reference_code=code, python_script=replay_script(code, require_csv=external, prepare_airfare=prompt==TS_EST_FRED_PROMPT, activity_demo=activity_demo))
    return result


def changes_since(parent, current):
    def settings(report):
        (_, features), (method, options) = parse_program(approved_code(report))
        return {**features, "method": method, **options}
    before, after = settings(parent), settings(current)
    changes = [f"{key}: {before.get(key, '—')} → {after.get(key, '—')}"
               for key in dict.fromkeys([*before, *after]) if before.get(key) != after.get(key)]
    if not changes:
        return ["The executed model settings are the same as in the previous version."]
    if (parent["task"] == current["task"] == "forecast" and
            parent["audit"]["holdout_end"] == current["audit"]["holdout_end"]):
        changes.append("The evaluations share holdout months. Comparisons informed by earlier results are exploratory.")
    return changes


def prompt_fields(value):
    if not isinstance(value, dict) or set(value) != set(FIELD_HELP):
        raise ValueError("Supply all five field names.")
    fields = {key: value[key].strip() for key in FIELD_HELP}
    if len(assembled_fields(fields)) > 8000:
        raise ValueError("The combined prompt exceeds 8,000 characters.")
    return fields


def field_wording(fields, columns, targets, kind="monthly"):
    try:
        choices = StudioAgents().suggest_field_wording(fields, columns, targets)
        return {"source": "Reviewer suggestions", "candidates": choices}
    except (RuntimeError, ValueError):
        choices = {key: FIELD_HELP[key]["continuations" if fields[key] else "options"] for key in targets}
        if "structure" in targets and not fields["structure"]:
            question = fields["clarity"].casefold()
            if any(word in question for word in ("forecast", "predict")):
                choices["structure"] = [
                    "Fit the model using earlier months, evaluate one-month-ahead predictions, then report the next forecast.",
                    "Compare forecast errors with a last-month benchmark and explain the difference to the reader.",
                ]
            elif any(word in question for word in ("estimat", "association", "coefficient")):
                choices["structure"] = [
                    "Define the outcome and controls, fit the regression, then interpret the coefficient and its confidence interval.",
                    "Describe the comparison, explain the adjustment variables, and end with what the estimate means in the original units.",
                ]
        return {"source": "Built-in starters (local reviewer suggestions unavailable)", "candidates": choices}


class StudioServer(ThreadingHTTPServer):
    """Keep the latest 40 completed versions local to this server session."""
    def __init__(self, address, handler):
        super().__init__(address, handler)
        self.runs = OrderedDict()
        self.run_lock = threading.Lock()

    def remember(self, report, data):
        with self.run_lock:
            self.runs[report["id"]] = (report, data)
            while len(self.runs) > 40:
                self.runs.popitem(last=False)

    def lookup(self, run_id):
        with self.run_lock:
            return self.runs.get(run_id)


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/reports/"):
            parts = self.path.removeprefix("/reports/").split(".")
            if len(parts) != 2 or parts[1] not in ("pdf", "csv"):
                self.send_error(404)
                return
            saved = self.server.lookup(parts[0])
            if saved is None:
                self.send_error(410, "This run is no longer in this studio session.")
                return
            report, _ = saved
            fmt = parts[1]
            body = pdf_report(report) if fmt == "pdf" else csv_report(report)
            self.send_response(200)
            self.send_header("Content-Type", "application/pdf" if fmt == "pdf" else "text/csv; charset=utf-8")
            name = f"{report['task']}_v{report['version']}_{parts[0][:8]}_results_and_review.{fmt}"
            self.send_header("Content-Disposition", f'attachment; filename="{name}"')
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path != "/":
            self.send_error(404)
            return
        page = files("studio_core").joinpath("studio.html").read_text(encoding="utf-8")
        def options(catalog):
            return "".join(f'<option value="{escape(key)}">{escape(label)}</option>'
                           for key, label in catalog.items())

        body = (page.replace("__FORECAST_FIELDS__", json.dumps(RETAIL_FIELDS))
                .replace("__RETAIL_FRED_FIELDS__", json.dumps(RETAIL_FRED_FIELDS))
                .replace("__EXPLORE_FIELDS__", json.dumps(EXPLORE_FIELDS))
                .replace("__EXPLORE_FRED_FIELDS__", json.dumps(EXPLORE_FRED_FIELDS))
                .replace("__FAMILY_BASES__", options({k:v for k,v in BASES.items() if not k.startswith("ar") and k != "ma1"}))
                .replace("__FAMILY_MA__", options({"ma1": BASES["ma1"]}))
                .replace("__FAMILY_AR__", options({k:v for k,v in BASES.items() if k.startswith("ar")}))
                .replace("__FAMILY_LEARNERS__", options(LEARNERS))
                .replace("__FAMILY_FIELDS__",json.dumps(FAMILY_FIELDS))
                .replace("__FAMILY_CONFIGS__",json.dumps(FAMILY_CONFIGS))
                .replace("__REGRESSION_FIELDS__",json.dumps(REGRESSION_FIELDS))
                .replace("__CROSS_FIELDS__", json.dumps(CROSS_FIELDS))
                .replace("__CROSS_SCREEN_FIELDS__", json.dumps(CROSS_SCREEN_FIELDS))
                .replace("__TS_EST_FIELDS__", json.dumps(TS_EST_FIELDS))
                .replace("__TS_EST_FRED_FIELDS__", json.dumps(TS_EST_FRED_FIELDS))
                .replace("__DML_FIELDS__", json.dumps(DML_FIELDS))
                .replace("__DML_FRED_FIELDS__", json.dumps(DML_FRED_FIELDS))
                .replace("__ROUGH_FIELDS__", json.dumps(ROUGH_FIELDS)).replace("__FIELD_HELP__", json.dumps(FIELD_HELP))
                .replace("__INCOMPLETE_CLARITY__", json.dumps(INCOMPLETE_CLARITY)).encode())
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, value, status=200):
        body = json.dumps(value).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        host = f"127.0.0.1:{self.server.server_port}"
        if (self.path not in ("/analyze", "/practice-review", "/family-template", "/writing-hints", "/sources/fred", "/sources/web") or self.headers.get("Host") != host or
                self.headers.get("Origin", f"http://{host}") != f"http://{host}" or
                self.headers.get_content_type() != "application/json"):
            self.send_error(403)
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 4000000:
                raise ValueError("The request must be under 4 MB.")
            args = json.loads(self.rfile.read(size))
            if self.path in ("/sources/fred", "/sources/web"):
                try:
                    imported = (import_fred(args.get("series_ids"), args.get("api_key"),
                                            airfare_example=args.get("recipe") == "airfare_january_2025",
                                            retail_example=args.get("recipe") == "retail_2005_2024")
                                if self.path == "/sources/fred" else import_web_source(args))
                    if self.path == "/sources/fred" and args.get("starter_task") and not args.get("recipe"):
                        imported["starter"], _ = fred_starter(imported["series"], args["starter_task"])
                        if args["starter_task"].startswith("family:"):
                            imported["family_config"]=fred_family_config(args["starter_task"],imported["series"])
                except ValueError as exc:
                    self.send_json({"error": str(exc)}, 400)
                else:
                    self.send_json(imported)
                return
            user_text = [args.get(k) for k in ("prompt", "prompt_fields", "feedback", "clarifications", "wording")]
            written = json.dumps(user_text)
            if resembles_fred_key(written) or re.search(
                    r"(?i)\b(?:api[_ -]?key|access[_ -]?token|x-api-key)\s*[:=]\s*[^\s\"&]{6,}|"
                    r"\bBearer\s+[A-Za-z0-9._~+/=-]{8,}", written):
                self.send_json({"error": "Remove the possible API key from the prompt or reply. Enter it only in the selected provider’s masked key field."}, 422)
                return
            if self.path=="/practice-review":
                fields=prompt_fields(args.get("prompt_fields"))
                if not any(fields.values()):
                    self.send_json({"error":"Write at least one field to review."},400)
                    return
                try:
                    notes=StudioAgents().review_writing(fields)
                    source="Local Llama writing feedback"
                except (RuntimeError,ValueError):
                    notes={k:FIELD_HELP[k]["question"] for k in fields}
                    source="Self-review questions (local Llama unavailable)"
                self.send_json({"source":source,"notes":notes})
                return
            if self.path=="/family-template":
                try:
                    fields,_=family_template(args)
                    self.send_json({"fields":fields})
                except (ValueError,KeyError,TypeError) as exc:
                    self.send_json({"error":str(exc)},400)
                return
            if self.path == "/writing-hints":
                fields = prompt_fields(args["prompt_fields"])
                targets = args["targets"]
                if (not isinstance(targets, list) or not 1 <= len(targets) <= len(FIELD_HELP)
                        or any(key not in FIELD_HELP for key in targets) or len(set(targets)) != len(targets)):
                    raise ValueError("Choose one or more prompt criteria.")
                if args.get("practice_only"):
                    self.send_json(field_wording(fields, [], targets))
                    return
                # Only draft text goes to the reviewer while typing. Never send CSV rows.
                provider = requested_source(assembled_fields(fields), args.get("source_mode", "auto"))
                columns = []
                hint_kind = args.get("data_kind","auto")
                if not args.get("custom_csv") and provider == "local":
                    try:
                        hint_kind = data_structure(assembled_fields(fields), args.get("data_kind","auto"))
                    except ValueError:
                        hint_kind = None
                    if hint_kind:
                        columns = (load_cross_demo() if hint_kind=="cross_section" else load_activity_demo() if args.get("source_mode")=="ar_demo" else load_retail_demo() if args.get("source_mode")=="retail_demo" else load_demo()).columns
                self.send_json(field_wording(fields, columns, targets, hint_kind))
                return
            parent = None
            revision = None
            feedback = []
            mode = args.get("mode", "live")
            if "parent_id" in args:
                saved = self.server.lookup(args["parent_id"])
                if saved is None:
                    self.send_error(410, "Previous run expired. Start a new analysis.")
                    return
                parent, data = saved
                kind = parent.get("data_kind", "monthly")
                note = args["feedback"].strip()
                if not 1 <= len(note) <= 2000:
                    raise ValueError("Feedback must contain 1 to 2,000 characters.")
                version = parent["version"] + 1
                prompt, source = parent["audit"]["original_prompt"], parent["source"]
                feedback = [*parent["feedback"], {"version": version, "text": note}]
                revision = {"previous_approved_code": approved_code(parent),
                            "prior_analysis_prompt": parent["audit"].get("analysis_prompt", prompt),
                            "prior_prompt_clarifications": parent["audit"].get("prompt_clarifications", []),
                            "previous_interpretation": parent["interpretation"],
                            "previous_user_feedback": [c.get("used_text", c["text"]) for c in feedback[:-1]],
                            "latest_user_feedback": note}
                mode = "live"
            else:
                version = 1
                fields = args.get("prompt_fields")
                if fields is None:
                    prompt = args["prompt"]  # Public free-text API.
                else:
                    fields = prompt_fields(fields)
                    prompt = assembled_fields(fields)
                csv = args.get("csv")
                data = pd.read_csv(io.StringIO(csv)) if csv is not None else None
                source_text = [prompt, *(turn["answer"] for turn in (args.get("clarifications") or []))]
                wording = args.get("wording") or {}
                if wording.get("action") in ("edit", "use"):
                    source_text.append(wording.get("edited") or wording["suggestion"])
                provider = requested_source("\n".join(source_text), args.get("source_mode", "auto")) if data is None else "local"
                source = str(args.get("filename") or "Uploaded monthly CSV")[:200] if data is not None else "Bundled simulated monthly data"
        except (AttributeError, KeyError, ValueError, TypeError):
            self.send_error(400, "Invalid request, feedback, or CSV")
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

        def emit(event):
            self.wfile.write((json.dumps(event, allow_nan=False) + "\n").encode())
            self.wfile.flush()

        try:
            if parent is None:
                requested_kind=args.get("data_kind","auto")
                kind = data_structure(prompt,requested_kind,data)
                if kind == "cross_section" and provider != "local":
                    raise ValueError("These API importers supply monthly series. Choose the independent-household example or upload a cross-sectional CSV.")
                if args.get("source_mode") == "ar_demo" and kind == "cross_section":
                    raise ValueError("The activity simulation is a monthly time series. Select independent units and the household data or CSV.")
                cfg=default_config(prompt,args.get("family_config"))
                if (csv is not None and cfg and cfg.get("source") in ("bundled","activity")
                        and assembled_fields(family_template(cfg)[0])==prompt):
                    raise ValueError("Choose My monthly CSV under forecast settings, enter its target and units, and apply the settings.")
                if cfg and cfg.get("source") in ("csv","fred") and data is None and provider=="local":
                    raise ValueError("Import or upload the monthly data named in this template.")
                if data is None and provider == "local" and prompt in BUNDLED_EXAMPLES:
                    _, load, source = BUNDLED_EXAMPLES[prompt]
                    data = load()
                if data is None and provider=="local" and cfg and cfg.get("source")=="activity":
                    data, source = load_activity_demo(), "Bundled simulated AR activity"
                if data is None and provider=="local" and (cfg or family_intent(prompt)) and args.get("source_mode") in (None,"local","auto","retail_demo"):
                    data,source=load_retail_demo(),RETAIL_SOURCE
                if data is None and provider=="local" and (args.get("source_mode")=="ar_demo" or prompt==TS_EST_PROMPT):
                    data,source=load_activity_demo(),"Bundled simulated AR activity"
                if data is None and provider == "local" and (args.get("source_mode")=="retail_demo" or prompt in (*RETAIL_PROMPTS, *EXPLORE_PROMPTS)):
                    if kind != "monthly":
                        raise ValueError("Retail sales require monthly observations.")
                    data, source = load_retail_demo(), RETAIL_SOURCE
                if kind == "cross_section" and data is None:
                    data = load_cross_demo()
                    source = "Bundled simulated independent households"
                elif csv is not None and not args.get("filename"):
                    source = "Uploaded CSV"
            if kind == "cross_section" and (prompt in (DML_PROMPT, DML_FRED_PROMPT, FORECAST_PROMPT, AIRFARE_FRED_PROMPT, TS_EST_FRED_PROMPT) or prompt in ESTIMATION_EXAMPLES and prompt not in (CROSS_PROMPT,CROSS_SCREEN_PROMPT)):
                raise ValueError("This example requires monthly observations. Choose a cross-sectional DML example for independent units.")
            if parent is None and mode != "reference" and fields is not None and not all(fields.values()):
                missing = [k for k in FIELD_HELP if not fields[k]]
                message = "Please complete " + ", ".join(k.title() for k in missing) + "."
                emit({"type":"event", "phase":"working", "round":0, "role":"Prompt reviewer",
                      "message":"Preparing candidate answers for the blank fields…"})
                columns = [] if provider != "local" else (load_demo() if data is None else data).columns
                wording = field_wording(fields, columns, missing, kind)
                emit({"type":"event", "round":0, "role":"Prompt reviewer",
                      "message":message, "decision":"ask_user"})
                emit({"type":"needs_fields", "message":message, **wording})
                return
            family_ref = parent is None and provider=="fred" and bool(default_config(prompt,args.get("family_config")))
            fred_reference = mode == "reference" and prompt in (AIRFARE_FRED_PROMPT,DML_FRED_PROMPT,TS_EST_FRED_PROMPT, RETAIL_PROMPTS[1], EXPLORE_PROMPTS[1])
            if fred_reference and data is None and provider != "fred":
                raise ValueError("Select FRED API for this FRED reference example.")
            if parent is None and (mode != "reference" or fred_reference or family_ref) and provider != "local":
                settings = PROVIDERS.get(provider, {"name":"FRED"})
                if family_ref:
                    cfg=default_config(prompt,args.get("family_config"))
                    settings={**settings,"series_ids":[cfg["target"]],"starterTask":fred_task(cfg),"family_fields":family_template(cfg)[0]}
                elif provider == "fred" and airfare_january_example(prompt, fields):
                    settings = {**settings, "series_ids":list(AIRFARE_FRED_COLUMNS), "recipe":"airfare_january_2025"}
                elif provider=="fred" and (prompt in (*RETAIL_PROMPTS,*EXPLORE_PROMPTS) or
                        fields and fields["context"] in (RETAIL_FRED_FIELDS["context"],EXPLORE_FRED_FIELDS["context"])):
                    settings = {**settings,"series_ids":[RETAIL_ID],"recipe":"retail_2005_2024",
                                "example_task":"exploration" if prompt in EXPLORE_PROMPTS or (fields and fields["context"]==EXPLORE_FRED_FIELDS["context"]) else "forecast"}
                emit({"type":"needs_source", "provider":provider, "settings":settings,
                      "message":"Connect " + settings["name"] + " to import the observations."})
                return
            if parent is None and data is not None and airfare_january_example(prompt, fields):
                # Keep the same prepared frame for feedback and the exported analysis.
                data = prepare_airfare_sample(data)
            if parent is None and data is not None and prompt in (*RETAIL_PROMPTS,*EXPLORE_PROMPTS):
                data = retail_sample(data)
            if mode == "reference" and args.get("fred_task"):
                if data is None or "date" not in data or kind != "monthly":
                    raise ValueError("Import the monthly FRED series before running its example.")
                task = args["fred_task"]
                starter, code = fred_starter([c for c in data if c != "date"], task)
                if assembled_fields(starter) != prompt:
                    raise ValueError("Use Run with local Llama for an edited question.")
                if task.startswith("family:"):
                    result=reference_family(data,prompt,source,code)
                elif task == "exploration":
                    result = reference_exploration(data, prompt, source, code)
                elif task == "forecast":
                    result = reference_forecast(data, prompt=prompt, code=code)
                elif task == "rcf_dml_screen":
                    result = reference_screen(data, prompt, code=code)
                else:
                    result = reference_dml(data, prompt, external=True, code=code)
            elif mode=="reference" and (prompt in FAMILY_PRESETS or args.get("family_config")):
                cfg=default_config(prompt,args.get("family_config"))
                fields,code=family_template(cfg)
                if assembled_fields(fields)!=prompt or kind!="monthly" or data is None or provider!="local":
                    raise ValueError("Select the matching monthly data. Use local Llama for edited prompt wording.")
                result=reference_family(data,prompt,source,code)
            elif mode=="reference" and args.get("exploration_target"):
                target = args["exploration_target"]
                if csv is None or kind!="monthly" or prompt!=assembled_fields(csv_explore_fields(target,args.get("exploration_units"))):
                    raise ValueError("Upload the matching monthly CSV. Use local Llama for free-form edits.")
                result = reference_exploration(data, prompt, source, exploration_code(target))
            elif mode=="reference" and prompt in (*RETAIL_PROMPTS,*EXPLORE_PROMPTS):
                if kind != "monthly" or provider!="local" or data is None:
                    raise ValueError("Use the included retail sales sample, matching uploaded data, or the FRED preset.")
                if prompt in RETAIL_PROMPTS:
                    result = reference_forecast(data, prompt=prompt, code=RETAIL_CODE, source=source, external=csv is not None)
                else:
                    result = reference_exploration(data, prompt, source, exploration_code(RETAIL_ID))
            elif mode=="reference" and prompt==REGRESSION_PROMPT:
                if kind!="monthly" or provider!="local" or csv is not None:
                    raise ValueError("Choose the included monthly simulation for this regression example.")
                (_,features),(_,options)=parse_program(REGRESSION_CODE)
                result=ols_hac(monthly_features(data,**features),**options)
                result.audit.update(agent_mode="Python reference only",original_prompt=prompt,reference_code=REGRESSION_CODE,
                                    python_script=replay_script(REGRESSION_CODE))
            elif mode=="reference" and prompt in ESTIMATION_EXAMPLES:
                if (prompt in (CROSS_PROMPT,CROSS_SCREEN_PROMPT)) != (kind=="cross_section") or provider!="local":
                    raise ValueError("Choose the matching data structure and example.")
                emit({"type":"status","message":"Cross-fitting the nuisance models and checking the DML specification…"})
                result = reference_dml(data if data is not None else load_demo(), prompt, external=csv is not None, activity_demo=source=="Bundled simulated AR activity")
            elif mode == "reference" and prompt in (DML_PROMPT,DML_FRED_PROMPT):
                if kind != "monthly":
                    raise ValueError("The RCF screen applies only to monthly time series.")
                if provider != "local":
                    raise ValueError("Use the included data or a compatible monthly CSV for the Python-only DML example.")
                emit({"type":"status","message":"Assessing the three RCF-DML conditions in Python…"})
                result = reference_screen(data, prompt, external=csv is not None)
            elif mode == "reference":
                if not ((prompt == FORECAST_PROMPT and csv is None) or (fred_reference and csv is not None)):
                    raise ValueError("Use the unchanged Forecast example with included data or its prepared FRED sample.")
                emit({"type": "status", "message": "Fitting the reference models in Python…"})
                result = reference_forecast(data, prompt=prompt, source=source, external=csv is not None)
            else:
                emit({"type": "status", "message": "Connecting to local Llama…"})
                result = analyze(prompt, data, revision=revision, clarifications=args.get("clarifications"),
                                 pause_for_rewrite=True, wording=args.get("wording"), data_source=source, data_kind=kind, family_config=args.get("family_config"), on_event=lambda e: emit({"type": "event", **e}))
            if parent and "writing_choice" in result.audit:
                feedback[-1]["used_text"] = result.audit["writing_choice"]["used"]
            report = present(result)
            report.update(id=uuid4().hex, version=version, source=source, data_kind=kind, feedback=feedback,
                          parent_id=parent["id"] if parent else None,
                          previous_summary=parent["summary"] if parent else None)
            report["changes"] = changes_since(parent, report) if parent else []
            self.server.remember(report, data)
            emit({"type": "result", "report": report})
        except PromptRewriteOffered as exc:
            emit({"type": "rewrite_offered", "suggestion": str(exc)})
        except PromptNeedsAnswer as exc:
            if len(args.get("clarifications") or []) >= 6:
                emit({"type": "stopped", "message": "Six replies have been reviewed. Edit the original prompt "
                      "and submit again. Last question: " + str(exc)})
            else:
                emit({"type": "needs_input", "question": str(exc)})
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception as exc:
            # Preserve streamed review history; no successful final result on failure.
            try:
                emit({"type": "stopped", "message": str(exc)})
            except (BrokenPipeError, ConnectionResetError):
                pass


def launch(port=0, open_browser=True):
    with StudioServer(("127.0.0.1", port), Handler) as server:
        url = f"http://127.0.0.1:{server.server_port}/"
        print(f"Open {url} in your browser. Press Ctrl+C here to stop.", flush=True)
        if open_browser:
            threading.Timer(0.25, lambda: webbrowser.open(url)).start()
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
