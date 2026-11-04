"""Write, run, critique, revise, and explicitly agree on the same Python program."""
import ast
import hashlib
import re
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import Field, create_model

from studio_core.example_prompt import RETAIL_PROMPTS, RETAIL_CODE, RETAIL_SOURCE, RETAIL_ID, EXPLORE_PROMPTS, exploration_code, csv_exploration_target
from hybrid_forecasting import BASES, LEARNERS, family_development, family_forecast
from studio_core.family_prompts import PRESETS as FAMILY_PRESETS, family_intent, template as family_template, default_config
from studio_core.family_prompts import general_joint_request
from exploration import monthly_series, exploration_development, evaluate_recommendation
from studio_core.data_io import load_retail_demo, retail_sample, data_structure
from studio_core.agents import OllamaAgents
from dml import cross_section, cross_dml, ts_dml, load_cross_demo, load_activity_demo
from studio_core.example_prompt import (ESTIMATION_EXAMPLES, CROSS_PROMPT, CROSS_SCREEN_PROMPT, REGRESSION_PROMPT,
    REGRESSION_CODE, TS_EST_PROMPT, TS_EST_FRED_PROMPT, fred_starter)
from studio_core.analysis_tools import forecast_development, monthly_features, ols_hac, rcf_dml_screen, super_learner
from studio_core.data_io import load_demo, prepare_airfare_sample
from studio_core.example_prompt import AIRFARE_FRED_PROMPT, DML_CODE, DML_FRED_PROMPT, DML_PROMPT, FIELD_HELP, FORECAST_CODE, FORECAST_PROMPT, INCOMPLETE_CLARITY, ROUGH_EXAMPLE, ROUGH_FIELDS, assembled_fields
from studio_core.spec import Lag, Record, Review
from studio_core.workflow import ReviewStopped

ESTIMATION_PROMPT = (
    "Using the simulated monthly data, estimate the association between last month's "
    "fuel_inflation and this month's fare_inflation. Control for fare_inflation lags 1, "
    "2 and 3 and annual seasonality. Report the fuel coefficient and a 95% confidence "
    "interval using HAC standard errors with 3 monthly lags."
)
# The complete prompt identifies the program and the dataset for a worked example.
# Free-form drafts, external imports and uploaded tables take their own paths.
MONTHLY_DEMO = "Bundled simulated monthly data"
BUNDLED_EXAMPLES = {
    FORECAST_PROMPT: (FORECAST_CODE, load_demo, MONTHLY_DEMO),
    REGRESSION_PROMPT: (REGRESSION_CODE, load_demo, MONTHLY_DEMO),
    DML_PROMPT: (DML_CODE, load_demo, MONTHLY_DEMO),
    TS_EST_PROMPT: (ESTIMATION_EXAMPLES[TS_EST_PROMPT], load_activity_demo, "Bundled simulated AR activity"),
    **{p: (ESTIMATION_EXAMPLES[p], load_cross_demo, "Bundled simulated independent households")
       for p in (CROSS_PROMPT, CROSS_SCREEN_PROMPT)},
    RETAIL_PROMPTS[0]: (RETAIL_CODE, load_retail_demo, RETAIL_SOURCE),
    EXPLORE_PROMPTS[0]: (exploration_code(RETAIL_ID), load_retail_demo, RETAIL_SOURCE),
    **{p: (code, load_retail_demo, RETAIL_SOURCE) for p, code in FAMILY_PRESETS.items()},
}


def check_example_columns(code, data):
    (constructor, design), _ = parse_program(code)
    if constructor == "cross_section":
        required = [design["target"], design["exposure"], *design["controls"]]
    else:
        required = ["date", design["target"], *design.get("predictors", [])]
    missing = [name for name in required if name not in data.columns]
    if missing:
        raise ValueError("The selected data is missing columns required by this example: "
                         + ", ".join(missing) + ". Select the worked example to use its included data, "
                         "or provide a CSV with those columns.")


FEATURE_KEYS = {"target", "target_lags", "predictors", "predictor_lags", "seasonality"}
ANALYSES = {"family_forecast":{"mode","base","learner","holdout_months"}, "explore_series":{"validation_months"}, "super_learner": {"holdout_months"}, "ols_hac": {"term", "hac_lags"}, "rcf_dml_screen": {"term"}, "ts_dml": {"term","hac_lags"},
            "cross_dml": {"estimate"}}
DESIGNS = {"monthly_series":{"target"}, "monthly_features":FEATURE_KEYS, "cross_section":{"target","exposure","controls"}}
GUIDE = f"""For hybrid (classical + ML), ML only, classical only, or a comparison of forecasting families, use
design = monthly_series(data, target="actual column")
result = family_forecast(design, mode="hybrid", base="seasonal_trend", learner="forest", holdout_months=24)
Modes: hybrid, classical, ml, compare. Classical component keys: {', '.join(BASES)}. ML component keys: {', '.join(LEARNERS)}.
The ma1 baseline fits MA(1) to a mean and last month’s forecast error. The moving_average3 benchmark averages the last three observed levels.
Use base="" for ml and learner="" for classical. The hybrid adds an ML prediction of the classical model's
earlier one-step errors. Ask for an unresolved target or combination. A user's phrase joint modeling means
a hybrid classical plus ML forecast.
For exploratory time-series features and model recommendations, use design = monthly_series(data, target="actual_column") followed by result = explore_series(design, validation_months=24). It measures coverage, trend, seasonality, persistence and volatility and compares five models on earlier rolling dates while reserving the final 24 months. It takes one target and no external predictors.
Available workflows for monthly data: (1) one-month-ahead Super Learner of
LASSO and random forest with chronological validation; (2) OLS adjusted linear
association and a pointwise 95% calendar-time Newey-West interval; (3) a three-condition
suitability assessment for reverse-cross-fitted time-series DML. For (3) choose Y as
target, D as one observed lag term from the design, and the remaining lags/seasonality
as X. Use result = rcf_dml_screen(design, term="fuel_inflation__lag1") with the
actual requested exposure term. Clarify if Y, D, its timing or important controls
are unresolved. For a requested monthly DML slope use ts_dml(design, term="fuel_inflation__lag1", hac_lags=3).
It runs the screen and withholds estimation when flagged or inconclusive. For independent units use
design = cross_section(data, target="earnings", exposure="training_hours", controls=["education","experience"])
result = cross_dml(design, estimate=True)
Set estimate=False for an assessment-only request. Use the user's actual Y, D, X. Independent rows
require no date, time-reversibility test, or lags. This PLR coefficient needs assumptions for causal
interpretation; no estimator establishes them. Never substitute OLS/forecasting for a DML request.
The single super_learner call handles every stage of a forecast: it fits both base
models, estimates nonnegative weights summing to one by minimizing squared error on
earlier chronological out-of-fold predictions, evaluates the base models and ensemble
by RMSE and MAE on the requested holdout, and produces the next-month forecast.
Asking for the model comparison, weights and final prediction is ONE supported task.
The package supplies the combination rule. The user need not specify an optimizer or
provide weights. Reviewers get development evidence first; holdout metrics and the
next-month forecast are deliberately computed only after code approval.
Example forecasting program:
design = monthly_features(data, target="MRTSSM44000USN", target_lags=[1,2,3,12], predictors=[], predictor_lags=[], seasonality=True)
result = super_learner(design, holdout_months=24)
For estimation replace the second line with, for example:
result = ols_hac(design, term="fuel_inflation__lag1", hac_lags=3)
Use the actual column names and requested settings. All inputs are calendar lags
of 1 to 12 months relative to the target month. Need one or more target lags;
external predictors are optional. Use [] for both external lists when absent.
Annual seasonality adds sine/cosine. Forecast holdout is 3 to 60 months.
Inference reports a lagged coefficient in the design, with HAC bandwidth 0 to 24.
If unspecified use target lags [1,2,3], lag [1] for requested external predictors,
no seasonality unless requested, 24 holdout months or HAC bandwidth 3. Clarify
when the forecast target (or exposure for estimation) or intended analysis is unclear, or required methods are
unsupported. Return Python True/False, no imports or markdown fences in code.
Only the fixed API above is available. Never select specifications by a desired
sign, significance, or favorable holdout error."""


class ScreenExplanation(Record):
    reason: str = Field(min_length=1, max_length=800)


class OwnLagDesign(Record):
    exposure_lag: Lag
    control_lags: list[Lag] = Field(max_length=11)
    seasonality: bool


class OwnLagProposal(Record):
    design: OwnLagDesign | None
    reason: str = Field(min_length=1, max_length=800)


class WritingCriteria(Record):
    clarity: str
    structure: str
    context: str
    role: str
    examples: str


class FieldOptions(Record):
    first: str = Field(min_length=1, max_length=450)
    second: str = Field(min_length=1, max_length=450)


class PromptAssessment(Record):
    decision: Literal["accept", "ask_user"]
    message: str = Field(min_length=1, max_length=800)
    criteria: WritingCriteria
    suggested_prompt: str = Field(max_length=2400, description="A copy-ready rewrite for rough writing; empty if already clear.")


class WordingChoice(Record):
    action: Literal["keep", "use", "edit"]
    suggestion: str = Field(min_length=1, max_length=2400)
    edited: str = Field(default="", max_length=2400)

    def selected_text(self, original):
        if self.action == "keep":
            return original
        selected = (self.edited if self.action == "edit" else self.suggestion).strip()
        if not selected:
            raise ValueError("Enter your wording or keep the original.")
        return selected


class PromptRewriteOffered(ReviewStopped):
    """A runnable draft has an optional replacement for the author to consider."""


class CompleteExampleAssessment(PromptAssessment):
    decision: Literal["accept"]
    suggested_prompt: Literal[""] = ""


class RoughExampleAssessment(PromptAssessment):
    decision: Literal["accept"]
    suggested_prompt: str = Field(min_length=10, max_length=2400)


class PromptExchange(Record):
    question: str = Field(min_length=1, max_length=1200)
    answer: str = Field(min_length=1, max_length=2000)


class PromptNeedsAnswer(ReviewStopped):
    """The current input needs a user answer before analysis can proceed."""


def target_question(text, *, forecast_known=False):
    """Distinguish a forecast-target question from a request about other model inputs."""
    text = " ".join(text.casefold().split())
    if not forecast_known and not re.search(r"\b(?:forecast|predict)", text):
        return False
    if re.search(r"\b(predictors?|lags?|horizons?|controls?|holdout|frequency|methods?|confounders?|units|covariates?|bandwidth|sample period|dates?|data source|seasonality)\b", text):
        return False
    return bool(re.search(
        r"\b(?:which|what|choose|select|specify|identify|undecided)\b.{0,80}?\b(?:series|column|variable|outcome|target)\b.{0,90}?\b(?:forecast|predict)|"
        r"\b(?:which|what|choose|select|specify|identify|clarify)\b.{0,55}?\b(?:forecast(?:ing)? target|target (?:series|variable|column)|outcome to predict)\b|"
        r"\b(?:forecast(?:ing)? target|target (?:series|variable|column))\b.{0,35}?\b(?:missing|unspecified|unresolved|unclear|undecided)\b", text))


def chosen_forecast_target(exchanges, data, *, forecast_known=False):
    for item in reversed(exchanges):
        if target_question(item["question"], forecast_known=forecast_known):
            answer = item["answer"].strip().rstrip(".").strip(" '\"").strip(chr(96)).casefold()
            for name in data:
                key = str(name).casefold()
                named = answer == key or re.match(r"(?:forecast|predict|(?:the )?target is)\s+" + re.escape(key)
                                                  + r"(?:$|[, ]+and (?:include|use|report)\b)", answer)
                if name != "date" and named and pd.to_numeric(data[name], errors="coerce").notna().any():
                    return name
            break
    return None


class Interpretation(Record):
    text: str = Field(min_length=1, max_length=2000)


class Draft(Record):
    decision: Literal["run", "ask_user"]
    code: str = Field(max_length=4000)
    reason: str = Field(min_length=1)


class WorkedForecastDraft(Draft):
    """The bundled data and unedited worked case already specify all required inputs."""
    decision: Literal["run"]


class ClarificationCheck(Record):
    decision: Literal["revise", "ask_user"]
    reason: str = Field(min_length=1)


def _script_header(stmt):
    # Only recognize familiar scaffolding. The studio owns its imports and data source.
    if isinstance(stmt, ast.ImportFrom):
        return (stmt.level == 0 and stmt.module in {"studio_core.analysis_tools", "monthly_forecast", "toy_superlearner.analysis_tools", "superlearner.ensemble", "dml", "rcf_screen", "exploration", "hybrid_forecasting"}
                and all(a.name in {*DESIGNS, *ANALYSES} and a.asname is None for a in stmt.names))
    return ast.unparse(stmt) in {
        "import pandas as pd",
        "data = pd.read_csv('data.csv')",
        "data = pd.read_csv('monthly_airfares.csv')",
    }


def _report_print(stmt):
    # Recognize simple result displays only. Never execute the display expression.
    if not isinstance(stmt, ast.Expr) or not isinstance(stmt.value, ast.Call):
        return False
    call = stmt.value
    if not isinstance(call.func, ast.Name) or call.func.id != "print" or call.keywords:
        return False
    for arg in call.args:
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            continue
        while isinstance(arg, ast.Subscript) and isinstance(arg.slice, ast.Constant):
            arg = arg.value
        if isinstance(arg, ast.Attribute) and arg.attr in {"audit", "metrics", "predictions"}:
            arg = arg.value
        if not isinstance(arg, ast.Name) or arg.id != "result":
            return False
    return True


def _validated_program(source):
    """Return the effective source and its tool calls. All model source is treated as data."""
    if len(source) > 4000:
        raise ValueError("Program too long.")
    source = source.strip()
    lines = source.splitlines()
    if len(lines) > 1 and lines[0].lower() in {"```", "```python", "```py"} and lines[-1] == "```":
        source = "\n".join(lines[1:-1]).strip()
    try:
        statements = ast.parse(source).body
    except SyntaxError as exc:
        # Some analyst replies collapse the newline between the two allowed assignments.
        source, count = re.subn(r"(?<=\))[ \t]+(?=result\s*=\s*(?:super_learner|ols_hac|rcf_dml_screen|ts_dml|cross_dml|explore_series|family_forecast)\()", "\n", source, count=1)
        if not count:
            raise ValueError(f"Invalid Python: {exc.msg}") from exc
        try:
            statements = ast.parse(source).body
        except SyntaxError as error:
            raise ValueError(f"Invalid Python: {error.msg}") from error
    start, end = 0, len(statements)
    while start < end and _script_header(statements[start]):
        start += 1
    while end > start and _report_print(statements[end-1]):
        end -= 1
    if start or end != len(statements):
        statements = statements[start:end]
        source = "\n".join(ast.unparse(stmt) for stmt in statements)
    if len(statements) != 2:
        raise ValueError("Expected two assignments, design = ... and result = ...")
    steps = []
    for stmt, variable, input_name, allowed in zip(
            statements, ("design", "result"), ("data", "design"),
            (DESIGNS, ANALYSES)):
        if not (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and
                isinstance(stmt.targets[0], ast.Name) and stmt.targets[0].id == variable and
                isinstance(stmt.value, ast.Call) and isinstance(stmt.value.func, ast.Name)):
            raise ValueError(f"Expected {variable} = approved_function({input_name}, ...).")
        call = stmt.value
        name = call.func.id
        if (name not in allowed or len(call.args) != 1 or not isinstance(call.args[0], ast.Name)
                or call.args[0].id != input_name):
            raise ValueError(f"Allowed calls here: {list(allowed)} with first argument {input_name}.")
        keys = [kw.arg for kw in call.keywords]
        if set(keys) != allowed[name] or len(keys) != len(set(keys)):
            raise ValueError(f"{name} requires keyword arguments {sorted(allowed[name])}.")
        kwargs = {}
        for kw in call.keywords:
            nodes = kw.value.elts if isinstance(kw.value, ast.List) else [kw.value]
            if not all(isinstance(n, ast.Constant) and type(n.value) in (bool, str, int)
                       for n in nodes):
                raise ValueError("Arguments must be strings, integers, booleans, or lists of them.")
            kwargs[kw.arg] = ast.literal_eval(kw.value)
        steps.append((name, kwargs))
    if (steps[0][0] == "cross_section") != (steps[1][0] == "cross_dml"):
        raise ValueError("Pair cross_section with cross_dml. Monthly methods need monthly_features.")
    if (steps[0][0] == "monthly_series") != (steps[1][0] in ("explore_series","family_forecast")):
        raise ValueError("Use monthly_series for exploration and forecast-family programs.")
    return source, steps


def parse_program(source):
    return _validated_program(source)[1]


def replay_script(code, *, require_csv=False, prepare_airfare=False, activity_demo=False, retail_demo=False, retail_window=False):
    (constructor,_), (method,_) = parse_program(code)
    imports = {
        "family_forecast":"hybrid_forecasting", "cross_section":"dml", "cross_dml":"dml", "ts_dml":"dml",
        "monthly_series":"exploration", "explore_series":"exploration",
        "rcf_dml_screen":"rcf_screen", "super_learner":"superlearner.ensemble",
        "ols_hac":"studio_core.analysis_tools", "monthly_features":"studio_core.analysis_tools",
    }
    lines=['"""Run with python analysis.py [same_input.csv]."""',"import sys","import pandas as pd"]
    for name in (constructor,method):
        lines.append(f"from {imports[name]} import {name}")
    if require_csv:
        independent = constructor=="cross_section"
        label="or original independent-unit input CSV" if independent else "monthly input CSV"
        lines += ["","if len(sys.argv) != 2:", f"    raise SystemExit('Pass the downloaded {label} as the argument.')",
                  "data = pd.read_csv(sys.argv[1])"]
    else:
        if constructor=="cross_section" or activity_demo:
            module="dml"
            loader="load_activity_demo" if activity_demo else "load_cross_demo"
            call=loader+"()"
        else:
            module="studio_core.data_io"
            loader="load_retail_demo" if retail_demo else "load_demo"
            call=loader+"()"
        lines += [f"from {module} import {loader}","",f"data = pd.read_csv(sys.argv[1]) if len(sys.argv) > 1 else {call}"]
    for active,name in ((retail_window,"retail_sample"),(prepare_airfare,"prepare_airfare_sample")):
        if active:
            lines += [f"from studio_core.data_io import {name}",f"data = {name}(data)"]
    lines += ["",code.strip(),"","print(result.metrics.to_string())","print(result.audit)",""]
    return "\n".join(lines)


class StudioAgents(OllamaAgents):
    def review_writing(self, fields):
        return self._call(
            "You are an ECON 422 writing tutor. Give one short, specific revision note for each of "
            "Clarity, Structure, Context, Role, and Examples, using the student's words. For a blank "
            "field, ask a focused question. When an entry already works, say what it specifies. "
            "Review the writing only. The student has supplied no data for analysis. Do not invent "
            "data, report empirical findings, generate code, request credentials, or rewrite the draft.",
            WritingCriteria, {"draft":fields}).model_dump()


    def suggest_field_wording(self, fields, columns, targets):
        schema = create_model("CandidateAnswers", __base__=Record,
                              **{key: (FieldOptions, ...) for key in targets})
        reply = self._call(
            "You are the Prompt Reviewer helping an economics student complete a five-field form. "
            "Give TWO short options for each requested field. For a blank field, write a complete answer. "
            "For a field with text, write a single NEXT sentence that can be appended to the draft. "
            "Never repeat or rewrite the existing text. If column names are unavailable, use only "
            "variables named by the student. Each "
            "candidate must be ready to paste into that field, in the student's voice. Use the "
            "submitted fields and actual column names. Structure gives the order of the work; "
            "role names the perspective and audience; examples give a concrete timing or interpretation "
            "example. Do not silently decide an unresolved economic target, method or causal "
            "assumption. If the target is explicitly undecided, use target-neutral wording in other "
            "fields. The student will choose or write their own answers. Do not write analysis code.",
            schema, {"written_fields": {k:v for k,v in fields.items() if v},
                     "requested_fields": {k:FIELD_HELP[k]["question"] for k in targets},
                     "available_columns": list(columns)}).model_dump()
        result = {}
        for key in targets:
            options = list(dict.fromkeys(text.strip() for text in reply[key].values() if text.strip()))
            if not options:
                raise ValueError("No candidate text was produced.")
            result[key] = options
        return result

    def review_prompt(self, context):
        instructions = (
            "You are the Prompt Reviewer, an economics writing tutor before the analyst. "
            "Assess the request against the available data columns, task, user revisions and "
            "clarification answers. Give one concise, actionable note for EACH of clarity, "
            "structure, context, role and examples. Accept a runnable question even if its style "
            "could improve. Role and examples are optional. Never invent an outcome or material "
            "identification assumption. When a missing or conflicting requirement would change "
            "the analysis, return ask_user and put ONE specific question in message. Offer "
            "actual column names or supported choices if helpful. If the requested method is "
            "unsupported, explain the limitation and ask whether a supported task would work. "
            "When ready, return accept with a short reason. Do not write code. "
            "Coach by example. If the wording is rough, choppy, vague, or hard to follow, "
            "put ONE complete copy-ready rewrite in suggested_prompt. Use the student's "
            "perspective and explain one specific improvement in message. For a revision "
            "rewrite only the latest feedback, in the context of the prior approved analysis. "
            "Preserve the intended task, variables, timing, requested design, constraints "
            "and audience. Data displayed in this context are already loaded. Never request credentials in a clarification. Do not convert association into causation, invent methods or "
            "results, or turn package defaults into supposed user choices. Use concise "
            "natural language. A short but clear prompt needs no rewrite: use an empty "
            "string. When an essential detail is missing, still ask the focused question "
            "in message; any illustrative wording must mark the missing detail with a "
            "placeholder like [outcome]. A writing suggestion never resolves a missing fact. "

            "Hybrid classical + ML forecasting, ML only, classical only and matched model comparisons are supported. Treat joint modeling as hybrid forecasting. "
            "An exploratory time-series report with a target, measured features and a recommendation from rolling validation is supported. "
            "Supported: one-month-ahead monthly Super Learner forecasting with LASSO and random "
            "forest, lagged monthly OLS association with a 95% HAC interval, or an empirical "
            "three-condition RCF-DML suitability screen for an observed lagged D and outcome Y. "
            "For an estimate, the lab supports monthly reverse-cross-fitted DML conditional on the three checks, "
            "and partially linear DML for independent cross-sectional units with shuffled folds and robust SEs. "
            "Ask for Y, D, and the intended controls; ask for date/lag information only for monthly designs. "
            "It cannot certify conditions or supply a DML causal coefficient. Forecast error "
            "metrics and learning convex ensemble weights are built in. Defaults are target lags "
            "1–3, external lag 1 if requested, no seasonal terms unless requested, a 24-month "
            "holdout or HAC bandwidth 3. Do not ask users to specify these defaults. Use previous "
            "answers and the prior approved code for revisions. confirmed_forecast_target, when supplied, "
            "is the student's validated choice. Do not ask for the outcome again. The latest answer resolves "
            "earlier ambiguities. An expected economic implication is only motivation."
        )
        schema = PromptAssessment
        if context.get("required_task") == "rcf_dml_screen":
            instructions += (" This is an assessment of a monthly series. Its own lag may be D, and X may be empty. "
                             "Use the loaded series IDs and respect a request to evaluate levels. A possible failure "
                             "of a condition is a reason to run the assessment.")
        if context.get("confirmed_fred_screen") or context.get("worked_dml_screen") or context.get("worked_dml_code"):
            instructions += " This complete DML example is supported. Approve its specification."
            schema = CompleteExampleAssessment
        if context.get("family_code") or context.get("feature_code") or context.get("retail_code"):
            instructions += " The data and analysis are fully specified. Assess the five fields and approve."
            schema = CompleteExampleAssessment
        if context.get("built_in_forecast"):
            instructions += " This is the complete, prevalidated worked forecast. Give the five notes and approve it."
            schema = CompleteExampleAssessment
        elif (not context.get("user_revision") and not context.get("clarification_answers")
              and not context.get("wording_selected") and context["original_prompt"] in {"1. Clarity\n"+ROUGH_EXAMPLE, assembled_fields(ROUGH_FIELDS)}
              and RETAIL_ID in context["columns"]):
            instructions += " This deliberately rough teaching example needs a copy-ready rewrite."
            schema = RoughExampleAssessment
        if context.get("wording_selected"):
            instructions += " The author selected the wording. Check its meaning, ask about any essential gap, and give no further style rewrite."
        return self._call(instructions, schema, context)

    def draft(self, context):
        role = ("You are the analyst. data is already provided and the app displays the results. "
                "The code field starts with design = and ends with result = ... . "
                "Return only those two assignments, without a script header or output statements. "
                "Use the clarification answers to resolve the original prompt. Apply user revision instructions when present. Preserve other settings and valid reviewer feedback. ")
        if context.get("family_code") or context.get("feature_code") or context.get("retail_code"):
            code = context.get("family_code") or context.get("feature_code") or context["retail_code"]
            explanation = self._call("Explain the validated Python specification for the selected target. "
                                     "Describe what it will measure without predicting a winner before running.",
                                     ScreenExplanation, {**context, "validated_code":code})
            return Draft(decision="run", code=code, reason=explanation.reason)
        if context.get("confirmed_fred_screen"):
            explanation = self._call(
                "You are the analyst. The author selected a complete FRED assessment. The studio binds its "
                "validated Python below. Explain the chosen Y, lagged D and controls in one sentence. "
                "The diagnostics have yet to run.", ScreenExplanation, context)
            return Draft(decision="run", code=context["confirmed_fred_screen"], reason=explanation.reason)
        if context.get("worked_dml_code"):
            return self._call(role+"Produce this worked specification:\n"+context["worked_dml_code"],WorkedForecastDraft,context)
        if context.get("worked_dml_screen"):
            return self._call(role + "Generate the complete worked DML screen with this specification:\n"
                              + DML_CODE, WorkedForecastDraft, context)
        if context.get("single_screen_series"):
            series = context["single_screen_series"]
            proposed = self._call(
                "You are the assessment analyst. Select the requested exposure lag and distinct additional "
                "control lags of the ONE loaded monthly series, in months (1 to 12). Last month means 1. "
                "Use an empty control_lags list for no additional controls. Omit seasonal terms unless requested. "
                "The series remains in its imported units. Read the user's clarification answers and revisions. "
                "If the exposure timing is unresolved, other variables are required, or the question needs "
                "transformations or a sample restriction this design cannot express, return design=null and one "
                "specific question. Otherwise return the lag design with a short reason.",
                OwnLagProposal, context)
            if proposed.design is None:
                return Draft(decision="ask_user", code="", reason=proposed.reason)
            plan = proposed.design
            lags = sorted({plan.exposure_lag, *plan.control_lags})
            code = (f"design = monthly_features(data, target={series!r}, target_lags={lags!r}, "
                    f"predictors=[], predictor_lags=[], seasonality={plan.seasonality!r})\n"
                    f"result = rcf_dml_screen(design, term={f'{series}__lag{plan.exposure_lag}'!r})")
            return Draft(decision="run", code=code, reason=proposed.reason)
        if context.get("built_in_forecast"):
            return self._call(role + "This is the complete January 2025 worked forecast. The loaded data already "
                              "contain fare_inflation and fuel_inflation as monthly percentage changes. "
                              "Use the provided data directly. Generate the two calls for this task.\n" + GUIDE,
                              WorkedForecastDraft, context)
        if context.get("confirmed_forecast_target") and any(
                step.get("target_already_answered") for step in context.get("previous_rounds", [])):
            return self._call(role + "The target question is settled. Generate code using confirmed_forecast_target.\n"
                              + GUIDE, WorkedForecastDraft, context)
        return self._call(role + "Ask a question only for missing information that prevents a "
                          "supported analysis. Use an empty code string for clarification.\n"
                          + GUIDE, Draft, context)

    def check_clarification(self, context):
        if context.get("required_task") == "rcf_dml_screen":
            return self._call(
                "Check whether this assessment actually needs clarification. A series can be both Y and "
                "the source of the lagged exposure, and the user can request no additional controls. "
                "Use any previous answers. Ask one question only for an unresolved specification. If the "
                "answer is already explicit, return revise and identify the requested series and lag.",
                ClarificationCheck, context)
        return self._call("Check the analyst's request for clarification against the user's "
                          "question and package capabilities. If the package or prompt already "
                          "supplies the information, return revise and tell the analyst exactly how "
                          "to proceed. If input really is missing or the requested method is "
                          "unsupported, return ask_user with one specific user-facing question.\n"
                          + GUIDE, ClarificationCheck, context)

    def critic(self, context):
        if context.get("review_stage") == "findings":
            return self._call("You are the methods critic commenting on the completed time-series DML screen. "
                              "Python verified the executed variable roles. Y is the outcome, D is the exposure, "
                              "and X lists the controls. Explain the diagnostic evidence for these roles. "
                              "Any recommendation to change the model concerns a future analysis.", Review, context)
        if context.get("review_stage") == "specification":
            return self._call("You are the methods critic. Check only whether the outcome, exposure, controls, "
                              "lags, transformations and sample match all five prompt fields and any user changes. "
                              "Compare the executed Y, D and X separately. A lag of Y can be a requested control "
                              "alongside a distinct exposure D. Accept a matching "
                              "design even when DML might be unsuitable. Revise for a specific mismatch or "
                              "ask for a genuinely missing design choice.", Review, context)
        return self._call("You are the independent methods critic. Review the SAME Python and "
                          "the observed execution evidence against the original prompt as amended by the user feedback. Accept "
                          "a correctly implemented and appropriate workflow, revise a concrete "
                          "mistake, or ask_user if the question is unsupported or ambiguous. "
                          "An expected implication is motivation only. Do not ask for final "
                          "holdout performance, coefficient signs or p-values during review.\n"
                          + GUIDE, Review, context)

    def confirm(self, context):
        if context.get("review_stage") == "specification":
            return self._call("You are the analyst confirming the design review. Check that the critic "
                              "and the executed specification agree with the user's requested variables, "
                              "timing and sample. The suitability verdict is calculated separately.", Review, context)
        return self._call("You are the original analyst after the critic's review. Check your "
                          "exact current code and execution evidence one last time. Return accept "
                          "only if you agree it answers the prompt including later user changes, revise for a concrete "
                          "remaining issue, or ask_user for missing information. Do not require a "
                          "desired finding before acceptance.\n" + GUIDE, Review, context)

    def interpret(self, context):
        return self._call("You are the analyst. Explain these fixed, computed results for the "
                          "user in 1 to 3 plain English sentences. Address the original question as revised by user feedback. "
                          "Use only values provided; round only when useful. Never suggest a causal "
                          "effect from an association or a general guarantee from one holdout. "
                          "For a family forecast, distinguish validation selection and holdout scoring, and explain the baseline plus correction if present. "
                          "For exploration, name the winning model, cite a measured feature and validation RMSE, "
                          "and distinguish the later holdout and forecast from the model selection. "
                          "For an RCF-DML screen retain the reported verdict and distinguish concerns, "
                          "inconclusive checks and no flag detected. No causal estimate or pass certificate. "
                          "On revision, address the critic without changing models or numbers.",
                          Interpretation, context)

    def audit_interpretation(self, context):
        return self._call("You are the critic. Verify the analyst's interpretation against the "
                          "fixed Python results and the prompt as revised by user feedback. Check direction, numbers, units, "
                          "dates, uncertainty and economic meaning. Accept a supported concise answer. "
                          "Use revise only for a concrete mistake in the interpretation. The numerical "
                          "analysis is frozen. Do not request model tuning or external evidence.",
                          Review, context)


def analyze(prompt, data=None, model="llama3.1:8b", *, on_event=None, required_task=None, revision=None, clarifications=None, pause_for_rewrite=False, wording=None, data_source=None, data_kind="auto", family_config=None):
    """Run the real local model. A review limit or disagreement returns no final analysis."""
    worked = prompt in (FORECAST_PROMPT, AIRFARE_FRED_PROMPT)
    built_in_forecast = worked and not wording and not clarifications and revision is None and required_task in (None, "forecast")
    if isinstance(data, (str, Path)):
        data = pd.read_csv(data)
    kind = data_structure(prompt, data_kind, data)
    if data is not None and data_source is None:
        data_source = "Supplied data"
    if data is None:
        if prompt in (AIRFARE_FRED_PROMPT, DML_FRED_PROMPT, TS_EST_FRED_PROMPT):
            raise ValueError("Import the FRED airfare and jet fuel series before running this example.")
        if prompt in (RETAIL_PROMPTS[1], EXPLORE_PROMPTS[1]):
            raise ValueError("Import the monthly retail sales series for the FRED example.")
        if family_config and family_config.get("source") in ("csv","fred"):
            raise ValueError("Supply the selected monthly data for this forecast.")
        if prompt in BUNDLED_EXAMPLES:
            _, load, data_source = BUNDLED_EXAMPLES[prompt]
            data = load()
        elif family_config and family_config.get("source") == "activity":
            data, data_source = load_activity_demo(), "Bundled simulated AR activity"
        elif family_intent(prompt) or general_joint_request(prompt):
            data,data_source=load_retail_demo(),RETAIL_SOURCE
        else:
            data = load_cross_demo() if kind == "cross_section" else load_demo()
    if prompt in (*RETAIL_PROMPTS, *EXPLORE_PROMPTS) and revision is None:
        data = retail_sample(data)
    if (worked or prompt in (DML_FRED_PROMPT,TS_EST_FRED_PROMPT)) and revision is None:
        data = prepare_airfare_sample(data)
    if revision is None and prompt in BUNDLED_EXAMPLES and not (wording or clarifications):
        check_example_columns(BUNDLED_EXAMPLES[prompt][0], data)
    return _analyze(prompt, data, StudioAgents(model), on_event, required_task, built_in_forecast=built_in_forecast, revision=revision, clarifications=clarifications,
                    pause_for_rewrite=pause_for_rewrite, wording=wording, data_source=data_source, data_kind=kind, family_config=family_config)


def _analyze(prompt, data, agents, on_event=None, required_task=None, *, built_in_forecast=False, revision=None, clarifications=None, pause_for_rewrite=False, wording=None, data_source=None, data_kind="auto", family_config=None):
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 8000:
        raise ValueError("Write a question of 1 to 8,000 characters.")
    if not isinstance(data, pd.DataFrame) or data.empty:
        raise ValueError("Supply a nonempty table.")
    family_mode=family_intent(prompt)
    if not family_mode and general_joint_request(prompt):
        for turn in reversed(clarifications or []):
            if isinstance(turn,dict):
                family_mode=family_intent(str(turn.get("answer","")))
                if family_mode: break
    if general_joint_request(prompt) and not family_mode and not revision and not wording:
        raise PromptNeedsAnswer("For a hybrid forecast, which monthly target, classical baseline, and ML correction should I use? For example, specify AR(3) and gradient boosting.",[])
    kind=data_structure(prompt,data_kind,data)
    if kind == "monthly" and "date" not in data:
        raise ValueError("Monthly analysis requires a date column. For independent units choose Cross-sectional.")

    original_draft = revision["latest_user_feedback"] if revision else prompt
    choice = WordingChoice.model_validate(wording) if wording is not None else None
    selected = choice.selected_text(original_draft) if choice else original_draft
    effective_prompt = revision.get("prior_analysis_prompt", prompt) if revision else selected
    original_screen_request = not revision and not clarifications and (choice is None or choice.action == "keep")
    worked_dml = original_screen_request and effective_prompt in (DML_PROMPT, DML_FRED_PROMPT)
    request = effective_prompt.split("2. Structure")[0]
    mentions_dml = re.search(r"\b(?:DML|double machine learning|reverse.cross.fitt\w*)\b", request, re.I)
    asks_if_suitable = re.search(r"can (?:I|we) (?:use|apply)|can .{0,40} be (?:used|applied)|(?:is|whether|if) .{0,45}(?:suitable|appropriate|applicable|eligible)", request, re.I)
    wants_screen = asks_if_suitable or re.search(r"assess|screen|suitab|condition|applicab|assumption|appropriate|eligible", request, re.I)
    excludes_dml = re.search(r"(?:do not|don't|without|no)\s+(?:use\s+|assess\s+)?(?:RCF-)?DML\b", request, re.I)
    screen_only = re.compile(r"(?:screen|assessment|diagnostics|conditions) only|(?:do not|don't) estimate|without estimating|"
                             r"(?:no|without) (?:an? )?(?:coefficient|point estimate)|(?:do not|don't) (?:report|fit) (?:an? |the )?coefficient", re.I)
    assessment_only = screen_only.search(request)
    asks_both = re.search(r"(?:then|and) (?:estimate|report|compute)",request,re.I)
    wants_estimate = (bool(re.search(r"estimat(?:e|ion)|coefficient|effect|interval",request,re.I))
                      and not assessment_only and (not asks_if_suitable or asks_both))
    family_mode=family_intent(request) or family_mode
    if family_mode:
        if kind!="monthly": raise ValueError("Forecasting families require monthly observations.")
        required_task="family_forecast"
    elif kind == "cross_section":
        if not revision and (excludes_dml or (not mentions_dml and re.search(r"forecast|predict(?:ion)?", request, re.I))):
            raise ValueError("The independent-unit workflow supports DML estimation and assessment. For a monthly forecast, select monthly time series and provide a date column.")
        required_task = "cross_dml"
    elif required_task is None and mentions_dml and not excludes_dml:
        if not wants_estimate and not wants_screen and "\n\n2. Structure\n" in effective_prompt:
            structure = effective_prompt.split("\n\n2. Structure\n", 1)[1].split("\n\n3. Context", 1)[0]
            wants_screen = re.search(r"assess|screen|diagnostic|three conditions", structure, re.I)
            wants_estimate = (re.search(r"estimate|coefficient|confidence interval|HAC uncertainty", structure, re.I)
                              and not screen_only.search(structure))
        required_task = "ts_dml" if wants_estimate else "rcf_dml_screen" if wants_screen else "ts_dml"
    if kind == "monthly" and required_task is None and (effective_prompt in EXPLORE_PROMPTS or
            re.search(r"explor(?:e|atory|ation).{0,60}(?:series|monthly|seasonal|features)|recommend (?:a |the |which )?(?:forecasting )?model", request, re.I)):
        required_task = "exploration"
    if revision:
        match = re.search(r"result\s*=\s*(rcf_dml_screen|ts_dml|cross_dml|family_forecast)\(",revision.get("previous_approved_code",""))
        if match:
            previous = match[1]
            required_task = previous
            if previous == "rcf_dml_screen" and re.search(r"estimate|coefficient", selected, re.I):
                required_task = "ts_dml"
    worked_code = ESTIMATION_EXAMPLES.get(effective_prompt) if not (revision or clarifications or choice) else None
    history, events = [], []
    locked = None
    context = {"original_prompt": effective_prompt, "columns": list(data.columns), "rows": len(data),
               "dataset": ("Independent cross-sectional units; rows are not months." if kind=="cross_section"
                           else "Monthly observations. Forecasts use preceding calendar months."),
               "required_task": "forecast" if built_in_forecast else (required_task or "Infer from original prompt."),
               "built_in_forecast": built_in_forecast, "worked_dml_screen": worked_dml, "worked_dml_code":worked_code}

    if not (revision or clarifications or choice):
        cfg=default_config(effective_prompt,family_config)
        if cfg:
            field_values,code=family_template(cfg)
            if assembled_fields(field_values)==effective_prompt:
                context["family_code"]=code
        if effective_prompt in RETAIL_PROMPTS:
            context["retail_code"] = RETAIL_CODE
            context["required_task"] = required_task = "forecast"
        if effective_prompt in EXPLORE_PROMPTS:
            context["feature_code"] = exploration_code(RETAIL_ID)
        elif required_task == "exploration" and data_source and data_source.startswith("FRED monthly data ("):
            ids = [col for col in data if col != "date"]
            try:
                starter, code = fred_starter(ids, "exploration")
            except (TypeError, ValueError):
                starter = None
            if starter and assembled_fields(starter) == effective_prompt:
                context["feature_code"] = code
        if required_task == "exploration" and not context.get("feature_code"):
            target_column = csv_exploration_target(effective_prompt, data.columns)
            if target_column:
                context["feature_code"] = exploration_code(target_column)
    if kind == "monthly" and required_task == "rcf_dml_screen":
        series = [col for col in data if col != "date"]
        if len(series) == 1 and isinstance(series[0], str):
            context["single_screen_series"] = series[0]
        if (original_screen_request and data_source
                and data_source.startswith("FRED monthly data (")
                and all(isinstance(col, str) for col in series)
                and data_source == f"FRED monthly data ({', '.join(series)})"):
            try:
                fields, code = fred_starter(series, "rcf_dml_screen")
            except ValueError:
                fields = None  # A user-supplied CSV filename can resemble an import label.
            if fields and assembled_fields(fields) == effective_prompt:
                context["confirmed_fred_screen"] = code

    if data_source:
        context["loaded_data_source"] = data_source
    if revision:
        context["user_revision"] = {k: v for k, v in revision.items() if k != "previous_interpretation"}
        context["user_revision"]["latest_user_feedback"] = selected
        context["revision_instruction"] = ("Use the previous accepted code as the starting point. "
            "Apply the latest user feedback; it supersedes conflicting earlier instructions. "
            "Preserve the data and every setting the user did not ask to change. If the user "
            "only requests new wording, retain the model. Ask for clarification if unsupported.")

    def announce(round_no, role, message, **extra):
        event = {"round": round_no, "role": role, "message": message, **extra}
        events.append(event)
        if on_event:
            on_event(event)

    def working(round_no, role, message):
        if on_event:
            on_event({"round": round_no, "role": role, "message": message, "phase": "working"})

    exchanges = [] if clarifications is None else clarifications
    if not isinstance(exchanges, list) or len(exchanges) > 6:
        raise ValueError("Use at most six clarification replies for one prompt.")
    exchanges = [PromptExchange.model_validate(item).model_dump() for item in exchanges]
    target = (chosen_forecast_target(exchanges, data, forecast_known=INCOMPLETE_CLARITY in effective_prompt or required_task == "forecast")
              if kind=="monthly" and required_task in (None,"forecast") else None)
    if exchanges:
        context["clarification_answers"] = exchanges
        if target:
            effective_prompt = effective_prompt.replace(INCOMPLETE_CLARITY, f"Forecast {target} one month ahead.")
            effective_prompt += (f"\n\nThe user confirmed {target} as the forecast target. "
                                 "This supersedes earlier instructions to ask for a series.")
            context.update(confirmed_forecast_target=target, required_task="forecast")
        effective_prompt += "\n\nUser clarifications (latest decisions take precedence):\n" + "\n".join(
            ("Forecast target" if target and target_question(turn["question"], forecast_known=True) else turn["question"])
            + " → " + turn["answer"] for turn in exchanges)
        context["original_prompt"] = effective_prompt
    for exchange in exchanges:
        announce(0, "Prompt reviewer", exchange["question"], decision="ask_user")
        announce(0, "User", exchange["answer"], decision="answered")
    if choice:
        context["wording_selected"] = True
        announce(0, "Prompt reviewer", choice.suggestion, decision="suggestion")
        labels = {"keep": "Kept the original wording.", "use": "Used the suggested wording.",
                  "edit": "Edited the suggested wording."}
        announce(0, "User", labels[choice.action] + "\n" + selected, decision="answered")
    working(0, "Prompt reviewer", "Checking the research question against the data and five writing criteria…")
    short_unmatched = bool(exchanges and not target and target_question(exchanges[-1]["question"])
                           and re.fullmatch(r"[\w.-]+", exchanges[-1]["answer"].strip()))
    if not revision and INCOMPLETE_CLARITY in effective_prompt and (not exchanges or short_unmatched):
        names = [str(k) for k in data if k != "date" and pd.to_numeric(data[k], errors="coerce").notna().any()]
        question = "Which series should I forecast? Choose one of these columns: " + ", ".join(names) + "."
        if short_unmatched:
            question = "That answer does not match a numeric column. " + question
        announce(0, "Prompt reviewer", question, decision="ask_user")
        raise PromptNeedsAnswer(question, history)
    assessment = agents.review_prompt(context)
    suggestion = assessment.suggested_prompt.strip() if choice is None else ""
    if suggestion == original_draft.strip():
        suggestion = ""
    if assessment.decision == "ask_user" and target and target_question(assessment.message, forecast_known=True):
        announce(0, "Studio", f"The forecast target is confirmed as {target}. Continuing to the analyst.", decision="answered")
        suggestion = ""
    else:
        announce(0, "Prompt reviewer", assessment.message, decision=assessment.decision,
                 criteria=assessment.criteria.model_dump(), **({"suggested_prompt": suggestion} if suggestion else {}))
        if assessment.decision == "ask_user":
            raise PromptNeedsAnswer(assessment.message, history)
    if suggestion and pause_for_rewrite:
        raise PromptRewriteOffered(suggestion, history)

    for attempt in range(1, 4):
        entry = {"round": attempt}
        history.append(entry)
        try:
            working(attempt, "Analyst", "Preparing the confirmed FRED assessment…" if context.get("confirmed_fred_screen") else "Writing the analysis code…")
            draft = agents.draft({**context, "previous_rounds": history[:-1]})
            entry["analyst"] = draft.model_dump()
            if draft.decision == "ask_user":
                announce(attempt, "Analyst", draft.reason, decision="clarification proposed")
                if target and target_question(draft.reason, forecast_known=True):
                    entry["target_already_answered"] = True
                    raise ValueError(f"The user already chose {target}. Generate the forecast code with that target.")
                if required_task == "rcf_dml_screen" and any(
                        step.get("clarification_check", {}).get("decision") == "revise"
                        and step["analyst"]["reason"].strip() == draft.reason.strip() for step in history[:-1]):
                    raise ReviewStopped("The analyst repeated a clarification the critic already addressed: "
                                        + draft.reason + " Review the outcome and exposure lag in Clarity.", history)
                working(attempt, "Critic", "Checking the clarification request…")
                check = agents.check_clarification({**context, "analyst_question": draft.reason})
                entry["clarification_check"] = check.model_dump()
                announce(attempt, "Critic", check.reason, decision=check.decision)
                if check.decision == "ask_user":
                    if target and target_question(check.reason, forecast_known=True):
                        entry["target_already_answered"] = True
                        raise ValueError(f"Use the confirmed forecast target {target} in the next draft.")
                    raise PromptNeedsAnswer(check.reason, history)
                continue
            proposed = draft.code.strip()
            announce(attempt, "Analyst", draft.reason, code=proposed)
            code, steps = _validated_program(proposed)
            entry["validated_code"] = code
            if code != proposed:
                announce(attempt, "Python", "Using these two validated calls with the selected data. "
                         "The studio handles imports, data loading, and result display.", code=code)
            if built_in_forecast and steps != parse_program(FORECAST_CODE):
                raise ValueError("Match the complete built-in forecast specification: " + FORECAST_CODE)
            if worked_dml and steps != parse_program(DML_CODE):
                raise ValueError("Use the worked RCF-DML screen specification: " + DML_CODE)
            if context.get("confirmed_fred_screen") and steps != parse_program(context["confirmed_fred_screen"]):
                raise ValueError("Match the confirmed FRED assessment: " + context["confirmed_fred_screen"])
            if worked_code and steps != parse_program(worked_code):
                raise ValueError("Match the worked DML specification: " + worked_code)
            preset = context.get("family_code") or context.get("feature_code") or context.get("retail_code")
            if preset and steps != parse_program(preset):
                raise ValueError("Match the confirmed example: " + preset)
            (design_method, features), (method, kwargs) = steps
            task = method if method=="family_forecast" else {"explore_series":"exploration", "super_learner":"forecast","ols_hac":"estimation","rcf_dml_screen":"rcf_dml_screen","ts_dml":"ts_dml","cross_dml":"cross_dml"}[method]
            if target and (task != "forecast" or features["target"] != target):
                raise ValueError(f"The user selected a forecast of {target}. Use that target and super_learner.")
            if task=="family_forecast" and not revision and family_mode and kwargs["mode"]!=family_mode:
                raise ValueError("Use the requested forecast family: "+family_mode)
            if required_task and task != required_task:
                raise ValueError(f"This entry point requires {required_task}.")
            if (kind == "cross_section") != (design_method == "cross_section"):
                raise ValueError("Match the design to the selected data structure.")
            if not revision and method == "cross_dml" and wants_screen and not wants_estimate and kwargs["estimate"]:
                raise ValueError("Use estimate=False for an assessment-only question.")
            if not revision and method == "cross_dml" and wants_estimate and not kwargs["estimate"]:
                raise ValueError("Use estimate=True for the requested coefficient; the support diagnostic may withhold it.")
            key = (task, features.get("target",tuple(features.get("outcomes",[]))), kwargs.get("holdout_months", kwargs.get("validation_months")), features.get("exposure") or kwargs.get("term"))
            if task == "rcf_dml_screen":
                for prior in history[:-1]:
                    review = prior.get("analyst_confirmation") or prior.get("critic", {})
                    if review.get("decision") == "revise" and parse_program(prior["validated_code"]) == steps:
                        raise ReviewStopped("The analyst proposed the same assessment again after this feedback: "
                                            + review["reason"] + " Review the requested specification in Clarity.", history)
            if locked and key != locked:
                raise ValueError("The task, target, forecast holdout and screen exposure are locked once review evidence is seen.")
            working(attempt, "Python", "Fitting the proposed model…")
            design = {"cross_section":cross_section, "monthly_features":monthly_features, "monthly_series":monthly_series}[design_method](data, **features)
            if task == "forecast":
                _, _, _, evidence = forecast_development(design, **kwargs)
            elif task=="family_forecast":
                evidence=family_development(design,**kwargs)
            elif task == "exploration":
                fitted = exploration_development(design, **kwargs)
                evidence = {**fitted.audit, "features":fitted.metrics.to_dict("index")}
            elif task == "rcf_dml_screen":
                fitted = rcf_dml_screen(design, **kwargs)
                evidence = {k:fitted.audit[k] for k in ("task","target","term","controls","n","first_month","last_month","verdict","seasonality")}
                evidence["diagnostics"] = fitted.metrics.to_dict("index")
            elif task in ("ts_dml","cross_dml"):
                fitted = (ts_dml if task=="ts_dml" else cross_dml)(design, **kwargs)
                evidence = {k:fitted.audit[k] for k in ("task","target","term","controls","n","uncertainty")}
            else:
                fitted = ols_hac(design, **kwargs)
                evidence = fitted.audit.copy()  # No point estimates or significance during review.
            locked = key
            entry["evidence"] = evidence
            code_id = hashlib.sha256(code.encode()).hexdigest()[:12]
            entry["revision_id"] = code_id
            evidence_context = {**context, "code": code, "revision_id": code_id, "evidence": evidence}
            # An assessment ends with its three diagnostics, for every possible verdict.
            # An estimate continues to coefficient review if its screen clears.
            completed_screen = (task == "rcf_dml_screen" or
                                (task == "ts_dml" and fitted.audit["screen_status"] in ("does_not_pass", "inconclusive")))
            if completed_screen:
                variable_roles = {"Y": fitted.audit["target"], "D": fitted.audit["term"], "X": fitted.audit["controls"]}
                evidence_context["variable_roles"] = variable_roles
                if context.get("confirmed_fred_screen") or worked_dml or worked_code:
                    # These exact calls already matched the user's complete example above.
                    controls = ", ".join(variable_roles["X"]) or "none"
                    entry["specification_check"] = (f"Matched the requested design. Outcome Y = {variable_roles['Y']}. "
                                                    f"Exposure D = {variable_roles['D']}. Controls X = {controls}.")
                    announce(attempt, "Python", entry["specification_check"], evidence=evidence)
                    break
                # Review a custom design on its merits before asking for comments on its findings.
                evidence_context.update(review_stage="specification", evidence={k:fitted.audit[k] for k in
                    ("task","target","term","controls","seasonality","n","first_month","last_month")})
            announce(attempt, "Python", "Executed the proposed analysis for review.", evidence=evidence_context["evidence"])
            working(attempt, "Critic", "Checking the requested specification…" if completed_screen else "Checking the executed code and evidence…")
            critic = agents.critic(evidence_context)
            entry["critic"] = critic.model_dump()
            announce(attempt, "Critic", critic.reason, decision=critic.decision)
            if critic.decision == "ask_user":
                if target and target_question(critic.reason, forecast_known=True):
                    raise ValueError(f"The executed forecast uses the confirmed target {target}. Review that code.")
                raise PromptNeedsAnswer(critic.reason, history)
            if critic.decision == "revise":
                if context.get("confirmed_fred_screen"):
                    raise ReviewStopped("Review of the confirmed FRED assessment stopped: " + critic.reason
                                        + " Edit the question for a different specification or run the Python-only assessment.", history)
                continue
            working(attempt, "Analyst", "Checking the critic’s decision…")
            analyst = agents.confirm({**evidence_context, "critic": critic.model_dump()})
            entry["analyst_confirmation"] = analyst.model_dump()
            announce(attempt, "Analyst", analyst.reason, decision=analyst.decision)
            if analyst.decision == "ask_user":
                if target and target_question(analyst.reason, forecast_known=True):
                    raise ValueError(f"The executed forecast uses the confirmed target {target}. Review that code.")
                raise PromptNeedsAnswer(analyst.reason, history)
            if analyst.decision == "revise":
                if context.get("confirmed_fred_screen"):
                    raise ReviewStopped("Review of the confirmed FRED assessment stopped: " + analyst.reason
                                        + " Edit the question for a different specification or run the Python-only assessment.", history)
                continue
        except ReviewStopped:
            raise
        except (ValueError, TypeError) as exc:
            entry["execution_feedback"] = str(exc)
            announce(attempt, "Python", str(exc), decision="revise")
            continue
        break
    else:
        if "execution_feedback" in entry and not any("evidence" in r for r in history):
            raise ReviewStopped("The analyst could not produce valid analysis code. "
                                f"Last validation message: {entry['execution_feedback']}", history)
        raise ReviewStopped("No agreement after three rounds. Review the exchange and refine the prompt.", history)

    # Final forecasts are evaluated only after both roles accept this exact program.
    working(attempt, "Python", "Preparing the numerical results…")
    if task=="forecast":
        result=super_learner(design,**kwargs)
    elif task=="family_forecast":
        result=family_forecast(design,**kwargs)
    else:
        result=fitted
    if task == "exploration":
        result = evaluate_recommendation(design, fitted)
    readbacks = []
    if completed_screen:
        # Publish the computed verdict even if a reviewer wants changes to future estimation.
        interpretation = Interpretation(text=result.audit["verdict"])
        announce(attempt, "Python", result.audit["verdict"], decision="screen_complete",
                 evidence=result.metrics.to_dict("index"))
        working(attempt, "Critic", "Commenting on the completed screen…")
        try:
            note = agents.critic({**context, "review_stage":"findings", "code":code,
                                  "variable_roles":variable_roles, "verdict":result.audit["verdict"],
                                  "diagnostics":result.metrics.to_dict("index")})
            entry["findings_comment"] = note.model_dump()
            if worked_dml and note.decision != "accept":
                announce(attempt, "Studio", "The critic suggested another review. The executed design matches "
                         "the template. Use the three diagnostic findings to decide whether to pursue estimation.",
                         decision="review_note")
            else:
                announce(attempt, "Critic", note.reason, decision="review_note")
        except (ValueError, RuntimeError, OSError) as exc:
            entry["comment_error"] = str(exc)
            announce(attempt, "Studio", "The critic's comment was unavailable. The diagnostic results are complete.",
                     decision="review_note")
    else:
        final_inputs = {**context, "code": code, "numerical_result": result.audit.copy(),
                        "table": result.metrics.to_dict("index")}
        if revision:
            final_inputs["previous_interpretation"] = revision.get("previous_interpretation", "")
        for pass_no in (1, 2):
            try:
                working(attempt, "Analyst", "Interpreting the computed results…")
                interpretation = agents.interpret({**final_inputs, "prior_feedback": readbacks})
                announce(attempt, "Analyst · result review", interpretation.text)
                working(attempt, "Critic", "Checking the interpretation against the numbers…")
                assessment = agents.audit_interpretation({**final_inputs, "interpretation": interpretation.text})
                readbacks.append({"interpretation": interpretation.text, "critic": assessment.model_dump()})
                announce(attempt, "Critic · result review", assessment.reason, decision=assessment.decision)
                if assessment.decision == "accept":
                    break
                if assessment.decision == "ask_user":
                    raise ReviewStopped(assessment.reason, history + [{"result_review": readbacks}])
            except ReviewStopped:
                raise
            except ValueError as exc:
                readbacks.append({"format_feedback": str(exc)})
                announce(attempt, "Python", f"Result review pass {pass_no}: {exc}", decision="revise")
        else:
            raise ReviewStopped("No agreement on the interpretation after two passes. "
                                "The model and its numerical results stayed fixed.",
                                history + [{"result_review": readbacks}])

    inherited = revision.get("prior_prompt_clarifications", []) if revision else []
    result.audit["prompt_clarifications"] = [*inherited, *exchanges]
    result.audit.update({"interpretation": interpretation.text, "result_review": readbacks})
    result.audit["analysis_prompt"] = effective_prompt
    if choice:
        result.audit["writing_choice"] = {**choice.model_dump(), "original": original_draft, "used": selected}
    external_airfare = (prompt in (AIRFARE_FRED_PROMPT, DML_FRED_PROMPT, TS_EST_FRED_PROMPT) or
                        (prompt == FORECAST_PROMPT and bool(data_source and data_source != "Bundled simulated monthly data")))
    external_screen = task in ("rcf_dml_screen","ts_dml","cross_dml") and data_source not in (None, "Bundled simulated monthly data","Bundled simulated independent households","Bundled simulated AR activity")
    require_csv = external_airfare or external_screen or data_source not in (None, "Bundled simulated monthly data", "Bundled simulated independent households", "Bundled simulated AR activity", RETAIL_SOURCE)
    result.audit.update({"agent_mode": agents.name, "original_prompt": prompt, "rounds": history,
                         "events": events, "approved_revision": code_id,
                         "python_script": replay_script(code, require_csv=require_csv,
                                                        prepare_airfare=external_airfare, retail_demo=data_source==RETAIL_SOURCE, retail_window=prompt in (*RETAIL_PROMPTS,*EXPLORE_PROMPTS), activity_demo=data_source=="Bundled simulated AR activity")})
    return result
