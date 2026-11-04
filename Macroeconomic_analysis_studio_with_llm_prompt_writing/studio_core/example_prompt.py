"""The five fields shown in the built-in forecast example, in display order."""
import re


INCOMPLETE_CLARITY = ("I want a one-month-ahead forecast, but I have not decided which series to forecast. "
                      "Please ask me to choose.")

FORECAST_FIELDS = {
    "clarity": "Produce one forecast: predict January 2025 fare_inflation, one month after the sample ends.",
    "structure": ("Run the package's Super Learner once. Choose nonnegative LASSO and random forest "
                  "weights that sum to one by minimizing squared error on earlier chronological "
                  "out-of-fold predictions. Score one-month-ahead forecasts over the final 24 months "
                  "using RMSE and MAE. Report the ensemble weights and January forecast."),
    "context": ("Use the included simulated monthly CSV. Both series are monthly percentage changes. "
                "Use fare_inflation lags 1, 2 and 3, fuel_inflation lag 1, and annual seasonality. "
                "Assume the rates are available at month-end."),
    "role": "Act as a transportation economist explaining the forecast to an airline planning team.",
    "examples": ("For January 2025, use October, November and December 2024 fare inflation and "
                 "December 2024 fuel inflation. State the forecast as a monthly percentage change."),
}
FORECAST_PROMPT = "\n\n".join(
    f"{index}. {field.title()}\n{text}" for index, (field, text) in enumerate(FORECAST_FIELDS.items(), 1)
)

FORECAST_CODE = (
    'design = monthly_features(data, target="fare_inflation", target_lags=[1, 2, 3], '
    'predictors=["fuel_inflation"], predictor_lags=[1], seasonality=True)\n'
    'result = super_learner(design, holdout_months=24)'
)

AIRFARE_FRED_COLUMNS = {"CUSR0000SETG01": "fare_inflation", "MJFUELUSGULF": "fuel_inflation"}
AIRFARE_FRED_FIELDS = {**FORECAST_FIELDS, "context": (
    "Use FRED's CUSR0000SETG01 (airline fare CPI) and MJFUELUSGULF (Gulf Coast jet fuel price). "
    "Use fare_inflation and fuel_inflation, calculated as one-calendar-month percentage changes "
    "in those levels. Use January 2005 through December 2024, fare_inflation lags 1, 2 and 3, "
    "fuel_inflation lag 1 and annual seasonality. This retrospective exercise uses today's vintage "
    "and assumes the preceding month's observations are available when predicting.")}

# Deliberately informal, but the economic target and requested comparison are clear.
ROUGH_EXAMPLE = "retail sales next month pls. past sales incl last year. lasso and forest together. how good last 24 months?"

ROUGH_FIELDS = {
    "clarity": ROUGH_EXAMPLE,
    "structure": "fit lasso + forest, mix them, report error",
    "context": "included monthly retail sales MRTSSM44000USN, millions of dollars",
    "role": "economist explaining to students",
    "examples": "use december sales and last january for this january",
}


def assembled_fields(fields):
    return "\n\n".join(f"{i}. {key.title()}\n{fields[key].strip()}"
                       for i, key in enumerate(FORECAST_FIELDS, 1) if fields[key].strip())

REGRESSION_FIELDS = {
    'clarity': 'Estimate the association between last month’s fuel_inflation and this month’s fare_inflation, after adjusting for recent fare changes.',
    'structure': 'First fit the monthly OLS regression. Then report the fuel coefficient and its 95% confidence interval using HAC standard errors with a three-month bandwidth. Conclude by explaining the coefficient in percentage points.',
    'context': 'Use the simulated monthly CSV. Both rates are monthly percentage changes. Include fuel_inflation lag 1, fare_inflation lags 1, 2 and 3, and annual seasonality. Use all months with complete inputs.',
    'role': 'Act as an applied economist teaching ECON 422. Explain what the adjusted association tells a student about these data.',
    'examples': 'Compare two months with the same recent airfare history and seasonal terms. If the preceding month’s fuel inflation is one percentage point higher in one case, how much does fitted airfare inflation differ?',
}
REGRESSION_PROMPT = assembled_fields(REGRESSION_FIELDS)
REGRESSION_CODE = (FORECAST_CODE.split("\n")[0] + '\nresult = ols_hac(design, term="fuel_inflation__lag1", hac_lags=3)')

AIRFARE_FRED_PROMPT = assembled_fields(AIRFARE_FRED_FIELDS)

DML_FIELDS = {
    "clarity": "Can I use reverse-cross-fitted DML to study fare_inflation as Y and last month's fuel_inflation as D?",
    "structure": "Assess stationarity and weak dependence, time reversibility, and conditional stability. "
                 "Report the evidence for each and recommend whether to pursue or revise this specification.",
    "context": "Use the included monthly simulated rates. Use fare_inflation lags 1, 2 and 3 and annual "
               "seasonality as X. The exposure D is fuel_inflation lag 1.",
    "role": "Explain the diagnostics to an ECON 422 student deciding whether reverse cross-fitting is defensible.",
    "examples": "Check whether neighboring training-block information improves held-out prediction beyond X. "
                "Keep the conclusion conditional on the limits of these checks.",
}
DML_PROMPT = assembled_fields(DML_FIELDS)
DML_FRED_FIELDS = {**DML_FIELDS, "context": (
    "Use FRED CUSR0000SETG01 and MJFUELUSGULF as one-calendar-month percentage changes named "
    "fare_inflation and fuel_inflation. Screen January 2005 through December 2024, using current-vintage data. "
    "Use fare_inflation lags 1, 2 and 3 and annual seasonality as X. D is fuel_inflation lag 1.")}
DML_FRED_PROMPT = assembled_fields(DML_FRED_FIELDS)
DML_CODE = (FORECAST_CODE.split("\n")[0] + '\nresult = rcf_dml_screen(design, term="fuel_inflation__lag1")')


CROSS_FIELDS = {
    "clarity":"Estimate a cross-sectional DML coefficient for earnings (Y) on training_hours (D).",
    "structure":"Use partially linear DML with five shuffled folds. Report the coefficient, robust standard error, 95% interval and exposure-support diagnostic.",
    "context":"Each simulated household contributes one independent row. Adjust for education and experience, measured before training. No dates or lags are needed.",
    "role":"Explain to ECON 422 students the difference between an adjusted slope and a causal effect.",
    "examples":"Interpret one additional training hour in earnings units. The known slope in the simulation is 1.5.",
}
CROSS_CODE = ('design = cross_section(data, target="earnings", exposure="training_hours", controls=["education", "experience"])'
              '\nresult = cross_dml(design, estimate=True)')
TS_EST_FARE = {**DML_FIELDS, "clarity":"Estimate the time-series DML coefficient of last month's fuel_inflation on current fare_inflation if the RCF screen allows.",
                 "structure":"Assess all three RCF conditions first. If no warning appears, report a reverse-cross-fitted PLR slope with three-lag HAC uncertainty. Otherwise withhold the estimate."}
TS_EST_FRED_FIELDS = {**TS_EST_FARE, "context":DML_FRED_FIELDS["context"]}
TS_EST_FRED_CODE = DML_CODE.split("\n")[0] + '\nresult = ts_dml(design, term="fuel_inflation__lag1", hac_lags=3)'
TS_EST_FIELDS = {"clarity":"Estimate monthly time-series DML persistence in simulated activity.",
                "structure":TS_EST_FARE["structure"], "context":"Use the simulated Gaussian monthly AR(1) activity series. Y is current activity, D is activity lag 1, and X is activity lag 2. Use no seasonality.",
                "role":"Explain the adjusted persistence estimate to an ECON 422 student.",
                "examples":"Compare the coefficient on last month’s activity with the known simulation value of 0.4."}
TS_EST_CODE = ('design = monthly_features(data, target="activity", target_lags=[1, 2], predictors=[], predictor_lags=[], seasonality=False)'
               '\nresult = ts_dml(design, term="activity__lag1", hac_lags=3)')
CROSS_SCREEN_FIELDS = {**CROSS_FIELDS,
    "clarity":"Assess cross-sectional DML suitability for earnings (Y), training_hours (D), and the specified controls.",
    "structure":"Report sampling and identification questions, usable units, exposure support and held-out nuisance fit. Give a design assessment only.",
    "examples":"Check whether training hours vary after adjustment for education and experience."}
CROSS_SCREEN_PROMPT = assembled_fields(CROSS_SCREEN_FIELDS)
CROSS_PROMPT = assembled_fields(CROSS_FIELDS)
TS_EST_PROMPT = assembled_fields(TS_EST_FIELDS)
TS_EST_FRED_PROMPT = assembled_fields(TS_EST_FRED_FIELDS)
ESTIMATION_EXAMPLES = {CROSS_PROMPT:CROSS_CODE, CROSS_SCREEN_PROMPT:CROSS_CODE.replace("estimate=True","estimate=False"), TS_EST_PROMPT:TS_EST_CODE, TS_EST_FRED_PROMPT:TS_EST_FRED_CODE}


def fred_starter(series, task):
    """A reviewable prompt and matching Python program for custom FRED levels."""
    if not (1 <= len(series) <= 6 and len(set(series)) == len(series) and all(
            re.fullmatch(r"[A-Z0-9][A-Z0-9_.-]{0,39}", name) and name != "DATE" for name in series)):
        raise ValueError("Supply one to six distinct FRED series IDs.")
    y, others = series[0], series[1:]
    if isinstance(task,str) and task.startswith("family:"):
        from studio_core.family_prompts import fred_family_config,template
        return template(fred_family_config(task,series))
    if task == "exploration":
        context = f"Use all available monthly FRED observations for {y} in its original units. Explore this target on its own."
        return exploration_fields(y, context), exploration_code(y)
    d = others[0] if others else y
    reference = {
        "forecast": ("Forecast next month's " + y + " with a Super Learner.",
                     "Fit LASSO and random forest using lag 1 of each selected series. Evaluate the final 24 months using RMSE and MAE. Report the weights and next forecast.",
                     "super_learner(design, holdout_months=24)"),
        "rcf_dml_screen": (f"Assess whether reverse-cross-fitted DML is suitable for monthly {y} as Y and last month's {d} as D.",
                           DML_FIELDS["structure"], f"rcf_dml_screen(design, term={d+'__lag1'!r})"),
        "ts_dml": (f"Estimate the monthly time-series DML coefficient of last month's {d} on current {y} if the RCF screen allows.",
                   TS_EST_FIELDS["structure"], f"ts_dml(design, term={d+'__lag1'!r}, hac_lags=3)")}
    if task not in reference:
        raise ValueError("Choose forecast, time-series DML estimate, or time-series DML assessment.")
    clarity, structure, result = reference[task]
    controls = [y, *others[1:]] if others else []
    timing = ("Use one monthly lag of each series." if task == "forecast" else
              ("Adjust for lag 1 of " + ", ".join(controls) + "." if controls else "Use no additional controls."))
    fields = {"clarity":clarity, "structure":structure,
              "context":f"Use all available monthly FRED observations for {', '.join(series)} in their original units. {timing} Use no seasonal terms.",
              "role":"Explain the result and its limitations to an ECON 422 student.",
              "examples":f"For a June observation, use May's {d} as a lagged input."}
    design = (f"design = monthly_features(data, target={y!r}, target_lags=[1], "
              f"predictors={others!r}, predictor_lags={[1] if others else []}, seasonality=False)")
    return fields, design + "\nresult = " + result


def airfare_january_example(prompt, fields):
    return (prompt in {FORECAST_PROMPT, AIRFARE_FRED_PROMPT, DML_FRED_PROMPT, TS_EST_FRED_PROMPT}
            or (fields is not None and fields["clarity"] in (DML_FIELDS["clarity"], TS_EST_FRED_FIELDS["clarity"]) and fields["context"] == DML_FRED_FIELDS["context"])
            or (fields is not None and fields["clarity"] == FORECAST_FIELDS["clarity"]
                and fields["context"] in (FORECAST_FIELDS["context"], AIRFARE_FRED_FIELDS["context"])))


FIELD_HELP = {
    "clarity": {
        "question": "What outcome and question should the analyst focus on?",
        "continuations": ["Report the answer in the units of the outcome.",
                          "Explain what the result tells us about my economic question."],
        "options": ["Help me choose which available series to forecast one month ahead.",
                    "I want to estimate a monthly association. Ask me which outcome and predictor to use."]},
    "structure": {
        "question": "What should the analyst do first, then report?",
        "continuations": ["Then explain how the model will be evaluated.",
                          "Finish by interpreting the main result in the original units."],
        "options": ["Describe the inputs, fit the model, and end with the main result in plain language.",
                    "Start with the main finding, then show the method and the evidence behind it."]},
    "context": {
        "question": "Which data, timing, or assumptions matter for your question?",
        "continuations": ["Use only information available at the time of the analysis.",
                          "Identify missing months before constructing lags."],
        "options": ["Use the monthly dataset I selected. Ask if you need the units of a series.",
                    "Use calendar-month lags and account for any missing months."]},
    "role": {
        "question": "Who should the analyst speak as, and who is the audience?",
        "continuations": ["Define technical terms when explaining the result to the audience.",
                          "Explain which parts of the conclusion depend on the assumptions."],
        "options": ["Act as an economist explaining the results to ECON 422 students.",
                    "Act as an economic analyst briefing a policy audience."]},
    "examples": {
        "question": "What concrete case would help illustrate your instructions?",
        "continuations": ["Show which observations are available at that point.",
                          "Express the result for this example in the units of the outcome."],
        "options": ["Walk through one month and name the observations that can be used at that point.",
                    "Use one row from the selected monthly data to illustrate the timing of the variables."]},
}


RETAIL_ID = "MRTSSM44000USN"
RETAIL_SOURCE = "Included U.S. retail sales, Jan 2005–Dec 2024"
RETAIL_CONTEXT = ("Use the included U.S. monthly retail sales snapshot, MRTSSM44000USN, January 2005 through December 2024. "
                  "Sales are nominal, in millions of dollars, and not seasonally adjusted. "
                  "This retrospective exercise assumes each preceding observation has been released.")
RETAIL_FRED_CONTEXT = RETAIL_CONTEXT.replace("the included U.S. monthly retail sales snapshot", "FRED monthly U.S. retail sales")
RETAIL_FIELDS = {
    "clarity":"Forecast U.S. retail sales for January 2025, the month after the sample ends.",
    "structure":"Combine LASSO and random forest in a Super Learner. Learn weights from chronological out-of-fold predictions. Report weights, RMSE and MAE on the final 24 months, and the January forecast.",
    "context":RETAIL_CONTEXT + " Use own lags 1, 2, 3 and 12 plus annual seasonality. Use no external predictors.",
    "role":"Brief a retail planning team deciding how much sales revenue to expect after the holidays.",
    "examples":"For January 2025, use October through December 2024 and January 2024. Report sales in millions of dollars.",
}
RETAIL_FRED_FIELDS = {**RETAIL_FIELDS, "context":RETAIL_FIELDS["context"].replace(RETAIL_CONTEXT, RETAIL_FRED_CONTEXT)}
RETAIL_CODE = (f'design = monthly_features(data, target="{RETAIL_ID}", target_lags=[1, 2, 3, 12], predictors=[], predictor_lags=[], seasonality=True)\n'
               'result = super_learner(design, holdout_months=24)')


def exploration_fields(target, context):
    return {
        "clarity":f"Explore the monthly time series {target} and recommend a model for one-month-ahead forecasting.",
        "structure":"Report coverage, trend, seasonality, lag 1 and lag 12 correlations, and volatility. Compare five models by rolling validation RMSE over 24 months before the final 24-month holdout. Recommend the lowest-RMSE model and explain why. Then evaluate that fixed choice on the holdout and forecast the next month.",
        "context":context,
        "role":"Brief an ECON 422 student. Connect the measured features to a defensible forecasting choice.",
        "examples":"Compare using last month with using the same month last year. Name the recommended model, its validation RMSE and the months reserved for a later evaluation.",
    }


def exploration_code(target):
    return f'design = monthly_series(data, target={target!r})\nresult = explore_series(design, validation_months=24)'


EXPLORE_FIELDS = exploration_fields(RETAIL_ID, RETAIL_CONTEXT)
EXPLORE_FRED_FIELDS = exploration_fields(RETAIL_ID, RETAIL_FRED_CONTEXT)
RETAIL_PROMPTS = tuple(assembled_fields(f) for f in (RETAIL_FIELDS, RETAIL_FRED_FIELDS))
EXPLORE_PROMPTS = tuple(assembled_fields(f) for f in (EXPLORE_FIELDS, EXPLORE_FRED_FIELDS))

def csv_explore_fields(target, units):
    if not all(isinstance(value, str) and 0 < len(value) <= 100 for value in (target, units)) or target == "date":
        raise ValueError("Supply the name of a numeric target column and its units.")
    return exploration_fields(target, f"Use the uploaded monthly CSV. The target column is {target}, measured in {units}. Keep the data in the supplied units.")

def csv_exploration_target(prompt, columns):
    # Recognize the complete generated five-field prompt. Edits go to the general analyst.
    match = re.search(r"^3\. Context\nUse the uploaded monthly CSV\. The target column is ([^\n]+?), measured in ([^\n]+?)\. Keep the data in the supplied units\.$", prompt, re.M)
    if match and match[1] in columns and max(map(len, match.groups())) <= 100:
        if prompt == assembled_fields(csv_explore_fields(*match.groups())):
            return match[1]
    return None
