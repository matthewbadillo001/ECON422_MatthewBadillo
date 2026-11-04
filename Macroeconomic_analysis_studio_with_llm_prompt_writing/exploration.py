"""Exploratory features and rolling, one-month-ahead model comparisons."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from studio_core.data_io import monthly_frame
from studio_core.workflow import save_bundle


def monthly_series(data, *, target):
    frame = monthly_frame(data)
    if target not in frame:
        raise ValueError(f"Choose a target from the uploaded columns: {', '.join(frame.columns)}.")
    series = pd.to_numeric(frame[target], errors="raise").astype(float)
    if np.isinf(series).any():
        raise ValueError(f"{target} contains infinite observations.")
    return series


@dataclass
class Exploration:
    metrics: pd.DataFrame
    audit: dict

    def save(self, destination):
        save_bundle(destination, {"series_report.csv": self.metrics}, self.audit)


def _describe(series):
    observed = series.dropna()
    rows = [("Coverage", f"{len(observed)} of {len(series)} calendar months observed",
             f"{series.index[0]} through {series.index[-1]}. Missing months keep their place in the calendar."
             if len(series) else "No months before validation.")]
    if len(observed) < 2:
        return rows
    t = np.arange(len(series))
    slope, intercept = np.polyfit(t[series.notna()], observed, 1)
    residual = series - (intercept + slope * t)
    rows.append(("Linear trend", f"{slope:+,.2f} units per month", "Slope of the fitted level against elapsed months."))
    month = residual.groupby(residual.index.month)
    if len(month.count()) == 12 and month.count().min() >= 2:
        averages = month.mean()
        peak, trough = (pd.Timestamp(2000, int(idx), 1).strftime("%B") for idx in (averages.idxmax(), averages.idxmin()))
        rows.append(("Annual seasonality", f"Peak {peak}, trough {trough}",
                     f"Detrended month averages differ by {averages.max()-averages.min():,.2f} units from peak to trough."))
    else:
        rows.append(("Annual seasonality", "Insufficient coverage", "Need at least two observations in each calendar month."))
    for lag in (1, 12):
        pairs = pd.concat([series, series.shift(lag)], axis=1).dropna()
        acf = pairs.iloc[:,0].corr(pairs.iloc[:,1]) if len(pairs) >= 3 and pairs.nunique().min() > 1 else None
        rows.append((f"Lag {lag} correlation", f"{acf:.3f}" if acf is not None else "Unavailable",
                     f"Level correlation across {len(pairs)} pairs exactly {lag} month(s) apart. Trend and seasonality can contribute."))
    changes = series.diff().dropna()
    rows.append(("Volatility", f"{changes.std():,.2f} units" if len(changes) > 1 else "Unavailable",
                 "Standard deviation of one-calendar-month changes."))
    return rows


def _candidates(series):
    # Each row describes its target month. Only lagged observations enter a forecast.
    t = np.arange(len(series), dtype=float)
    calendar = np.column_stack((np.ones(len(t)), t, np.eye(12)[series.index.month-1, 1:]))
    lag = series.shift().to_numpy()
    return {"Last month": lag, "Same month last year": series.shift(12).to_numpy(),
            "AR(1)": np.column_stack((np.ones(len(t)), lag)),
            "Trend + month effects": calendar,
            "Trend + month effects + AR(1)": np.column_stack((calendar, lag))}


def _prediction(x, y, i):
    if x.ndim == 1:
        return x[i]
    usable = np.isfinite(x[:i]).all(axis=1) & np.isfinite(y[:i])
    if usable.sum() < max(24, x.shape[1]+1) or not np.isfinite(x[i]).all():
        return np.nan
    coef, _, rank, _ = np.linalg.lstsq(x[:i][usable], y[:i][usable], rcond=None)
    if rank < x.shape[1]:
        # Redundant lags can still identify a prediction in the training row space.
        training = x[:i][usable]
        weights = np.linalg.lstsq(training.T, x[i], rcond=None)[0]
        if not np.allclose(training.T @ weights, x[i], atol=1e-9, rtol=1e-9):
            return np.nan
    return x[i] @ coef


def exploration_development(design, *, validation_months=24):
    """Reserve the last 24 calendar months. Rank models only on preceding validation dates."""
    if type(validation_months) is not int or not 6 <= validation_months <= 60:
        raise ValueError("Choose a validation window between 6 and 60 months.")
    holdout = 24
    series = design
    dev_end = max(0, len(series) - holdout)
    val_start = max(0, dev_end - validation_months)
    initial = series.iloc[:val_start]
    rows = _describe(initial)
    feature_count = len(rows)
    evidence = "Feature estimates describe the initial training period. Validation and holdout observations are excluded."
    scores = {}
    winner = None
    common = 0
    if initial.count() >= 36:
        development = series.iloc[:dev_end]
        actual = development.to_numpy()
        inputs = _candidates(development)
        errors = []
        for i in range(val_start, dev_end):
            forecasts = [_prediction(x, actual, i) for x in inputs.values()]
            if np.isfinite([actual[i], *forecasts]).all():
                errors.append(np.asarray(forecasts) - actual[i])
        common = len(errors)
        if common >= max(6, (validation_months+1)//2):
            rmse = np.sqrt(np.mean(np.square(errors), axis=0))
            scores = {model:float(value) for model,value in zip(inputs, rmse)}
            winner = min(scores, key=scores.get)
            for model, value in sorted(scores.items(), key=lambda item:item[1]):
                rows.append((model, f"RMSE {value:,.2f}", f"Rolling one-step errors on the same {common} validation months."))
    if winner:
        explanation = {"Last month":"Recent levels alone supplied the most accurate forecast in this comparison.",
                       "Same month last year":"Repeating the prior year's value captured enough of the calendar pattern to win.",
                       "AR(1)":"Estimating persistence from the previous month improved the validation forecasts.",
                       "Trend + month effects":"A changing level and a separate effect for each calendar month fit the validation pattern.",
                       "Trend + month effects + AR(1)":"Allowing for trend, calendar month and the preceding observation fit the validation pattern."}
        verdict = f"Recommend {winner} for one-month-ahead forecasts of {series.name}."
        evidence += f" {explanation[winner]} Its validation RMSE was {scores[winner]:,.2f} target units, the lowest among the five candidates."
        features = {name:finding for name,finding,_ in rows}
        evidence += (f" The training sample had a trend of {features['Linear trend']}, "
                     f"a lag-1 correlation of {features['Lag 1 correlation']} and a lag-12 correlation of "
                     f"{features['Lag 12 correlation']}.")
    else:
        verdict = "Insufficient evidence to recommend a forecasting model."
        evidence += (f" The comparison needs 36 observed initial months and a common score for all five models in "
                     f"at least {max(6,(validation_months+1)//2)} validation months. It had {initial.count()} initial observations "
                     f"and {common} commonly scored months.")
    def span(idx):
        return f"{idx[0]} through {idx[-1]}" if len(idx) else "No months"
    audit = dict(task="exploration", target=str(series.name), trained=span(initial.index),
                 validation=span(series.index[val_start:dev_end]), reserved=span(series.index[dev_end:]),
                 models=scores, recommended_model=winner, common_months=common, verdict=verdict,
                 interpretation=evidence + " The final 24 months are reserved for a separate forecast evaluation.")
    rows.insert(feature_count, ("Rolling validation", audit["validation"],
                 f"{common} calendar months scored across all candidates. Earlier observations are available at each forecast origin."))
    rows.append(("Reserved evaluation", audit["reserved"], "Excluded from feature measurement and model selection."))
    return Exploration(pd.DataFrame(rows, columns=["Feature or model","Finding","Meaning"]).set_index("Feature or model"), audit)

def evaluate_recommendation(series, result):
    """Evaluate only the preselected specification, then issue its next forecast."""
    model = result.audit["recommended_model"]
    if not model:
        return result
    # Append an unlabeled target month. At origin i even the regression fit stops at i-1.
    extended = series.reindex(pd.period_range(series.index[0], series.index[-1]+1, freq="M"))
    x = _candidates(extended)[model]
    y = extended.to_numpy()
    holdout = range(len(series)-24, len(series))
    observations = [(series.index[i], y[i], _prediction(x,y,i)) for i in holdout]
    valid = [(str(date), actual, predicted) for date,actual,predicted in observations if np.isfinite([actual,predicted]).all()]
    rmse = float(np.sqrt(np.mean([(actual-predicted)**2 for _,actual,predicted in valid]))) if valid else None
    next_value = _prediction(x,y,len(series))
    forecast = float(next_value) if np.isfinite(next_value) else None
    next_date = str(extended.index[-1])
    result.audit.update(holdout_rmse=rmse, holdout_scored_months=[date for date,_,_ in valid],
                        forecast_month=next_date, forecast_value=forecast)
    result.metrics.loc["Selected model on holdout"] = [
        f"RMSE {rmse:,.2f}" if rmse is not None else "Unavailable",
        f"{len(valid)} of 24 months scored after model selection. The chosen specification refits at each origin."]
    result.metrics.loc["Selected model forecast"] = [
        f"{forecast:,.2f} units" if forecast is not None else "Unavailable",
        f"One-month-ahead prediction for {next_date}, using the chosen specification and available earlier observations."]
    result.audit["interpretation"] = result.audit["interpretation"].replace(
        "The final 24 months are reserved for a separate forecast evaluation.",
        "Only the selected model is then scored on the final 24 months and used to forecast the next month.")
    return result


def explore_series(design, *, validation_months=24):
    developed = exploration_development(design, validation_months=validation_months)
    return evaluate_recommendation(design, developed)
