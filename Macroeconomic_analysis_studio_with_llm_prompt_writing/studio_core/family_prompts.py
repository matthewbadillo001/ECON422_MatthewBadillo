"""Forecast-family examples use one reusable, auditable two-call program."""
import re
from hybrid_forecasting import BASES, LEARNERS, FAMILIES
from studio_core.example_prompt import RETAIL_ID, RETAIL_CONTEXT, assembled_fields

EXAMPLES={
    "seasonForest":("hybrid","seasonal_trend","forest"),
    "trendLasso":("hybrid","trend","lasso"),
    "arForest":("hybrid","ar1","forest"),
    "mlForest":("ml","","forest"),"mlLasso":("ml","","lasso"),
    "classicAR":("classical","ar1",""),"classicTrend":("classical","seasonal_trend",""),
    "classicMA":("classical","ma1",""),
    "familyCompare":("compare","seasonal_trend","forest"),
}


def template(config):
    if not isinstance(config,dict):
        raise ValueError("Supply forecast settings as an object.")
    mode,base,learner=config["mode"],config.get("base",""),config.get("learner","")
    if mode=="classical":learner=""
    if mode=="ml":base=""
    if mode not in (*FAMILIES,"compare") or (mode!="ml" and base not in BASES) or (mode!="classical" and learner not in LEARNERS):
        raise ValueError("Select a forecast family and the components it uses.")
    target=config["target"]
    if not isinstance(target,str) or not 1<=len(target)<=100 or not target.isprintable() or target.strip()!=target or target=="date":
        raise ValueError("Enter a numeric target column, distinct from date.")
    source=config.get("source","bundled")
    if source not in ("bundled","activity","csv","fred"):
        raise ValueError("Choose a series source from the forecast settings.")
    bundled_targets = {"bundled": RETAIL_ID, "activity": "activity"}
    if source in bundled_targets and target != bundled_targets[source]:
        raise ValueError(f"The selected dataset uses the target {bundled_targets[source]}.")
    units=config.get("units","target units")
    if not isinstance(units,str) or not units.strip() or len(units)>100 or not units.isprintable():
        raise ValueError("Enter units such as percent or millions of dollars.")
    if mode=="hybrid":
        article = "an" if learner == "elastic_net" else "a"
        clarity=f"Use hybrid forecasting for {target}: {BASES[base]} plus {article} {LEARNERS[learner]} forecast of its errors."
        structure="Build earlier one-step baseline errors. Train ML on those errors and add its correction to the classical forecast. Report both contributions for the next month."
    elif mode=="classical":
        clarity=f"Use classical only forecasting for {target} with {BASES[base]}."
        structure="Fit the classical specification and report its one-month-ahead forecast."
    elif mode=="ml":
        clarity=f"Use ML only forecasting for {target} with {LEARNERS[learner]}."
        structure="Predict the series directly from lags 1, 2, 3 and 12 and annual sine and cosine features."
    else:
        clarity=f"Compare forecast families for {target}: {BASES[base]}, {LEARNERS[learner]}, and their additive hybrid."
        structure="Compare classical only, ML only, and hybrid on the same validation months. Select the lowest validation RMSE before scoring the final holdout. Report all next-month forecasts."
    if base == "ma1":
        structure += " Fit a constant mean and one lagged shock. Show how the previous month's estimated forecast error enters the next baseline forecast."
    context=RETAIL_CONTEXT if source=="bundled" else f"Use my monthly CSV with a date column and target {target} in {units}."
    if source=="activity":
        context="Use the included monthly simulated activity series from January 1980 through December 2019. The target is activity in simulated index units."
    if source=="fred":
        context=f"Use monthly FRED observations for {target} in their original units over the imported sample."
    fields=dict(clarity=clarity,structure=structure+" Reserve 24 validation months before the last 24 holdout months.",
        context=context,role="Explain the forecast to an ECON 422 student in the target’s original units.",
        examples="Show the next forecast and the size of a typical error. Only use lags known at each forecast origin.")
    code=(f"design = monthly_series(data, target={target!r})\nresult = family_forecast(design, mode={mode!r}, "
          f"base={base!r}, learner={learner!r}, holdout_months=24)")
    return fields,code


CONFIGS={key:dict(mode=mode,base=base,learner=learner,source="bundled",target=RETAIL_ID,units="millions of dollars")
         for key,(mode,base,learner) in EXAMPLES.items()}
FIELDS={key:template(config)[0] for key,config in CONFIGS.items()}
PRESETS={assembled_fields(FIELDS[key]):template(cfg)[1] for key,cfg in CONFIGS.items()}

def family_intent(prompt):
    clarity=prompt.split("2. Structure",1)[0]
    patterns=[("compare",r"compare (?:forecast |forecasting |model )?families|compare.{0,80}classical.{0,80}(?:ML|learning)"),
              ("hybrid",r"\bhybrid\b|classical.{0,35}(?:\+|plus|combined with|and)\s*(?:ML\b|machine learning)"),
              ("ml",r"\b(?:ML|machine learning)[ -]only\b"),("classical",r"\bclassical[ -]only\b")]
    matched=next((mode for mode,pattern in patterns if re.search(pattern,clarity,re.I)),None)
    if matched:
        return matched
    if (re.search(r"trend|seasonal|(?:AR|MA)\s*\(\s*\d+\s*\)|autoregress|mean|exponential smoothing|moving average|last month|same month last year|na[iï]ve",clarity,re.I)
            and re.search(r"forest|\bRF\b|LASSO|ridge|elastic[ -]net|boosting",clarity,re.I)
            and re.search(r"residual|correction|remaining|baseline|\bplus\b|\+",clarity,re.I)):
        return "hybrid"
    if re.search(r"\bMA\s*\(\s*1\s*\)", clarity, re.I) and re.search(r"forecast|predict", clarity, re.I):
        return "classical"
    return None


def fred_task(config):
    return ":".join(["family",config["mode"],config.get("base",""),config.get("learner","")])


def fred_family_config(task,series):
    parts=task.split(":")
    if len(parts)!=4 or len(series)!=1:
        raise ValueError("Select exactly one FRED target for this forecast.")
    return dict(mode=parts[1],base=parts[2],learner=parts[3],target=series[0],source="fred",units="original series units")


def default_config(prompt,provided=None):
    if provided:
        return provided
    return next((CONFIGS[key] for key,fields in FIELDS.items() if assembled_fields(fields)==prompt),None)


def general_joint_request(prompt):
    return re.search(r"\bjoint(?:ly)?\s+(?:model(?:ing|ling)?|fit|estimate|analy[sz]e)",prompt.split("2. Structure",1)[0],re.I)
