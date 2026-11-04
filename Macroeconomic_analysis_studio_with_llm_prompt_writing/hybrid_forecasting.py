"""One-step classical, machine-learning, and additive hybrid forecasts."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from studio_core.workflow import save_bundle

BASES = {
    "naive": "Last month",
    "seasonal_naive": "Same month last year",
    "moving_average3": "3-month moving average",
    "mean": "Historical mean",
    "ses": "Exponential smoothing (α = 0.3)",
    "trend": "Linear trend",
    "seasonal_trend": "Trend + calendar months",
    "ma1": "MA(1)",
    **{f"ar{order}": f"AR({order})" for order in range(1, 13)},
}
LEARNERS = {
    "forest": "Random forest",
    "lasso": "LASSO",
    "ridge": "Ridge regression",
    "elastic_net": "Elastic net",
    "boosting": "Gradient boosting",
}
# These baselines forecast directly from observed calendar lags, with no fitted weights.
FIXED_LAGS = {"naive": (1,), "seasonal_naive": (12,), "moving_average3": (1, 2, 3)}
# SES uses a fixed update rate; like the lag benchmarks, it has no coefficients to estimate.
FIXED_BASES = {*FIXED_LAGS, "ses"}
FAMILIES={"classical":"Classical only","ml":"ML only","hybrid":"Hybrid"}
VALIDATION_MONTHS=24


@dataclass
class FamilyForecast:
    metrics: pd.DataFrame
    audit: dict

    def save(self,destination):
        save_bundle(destination,{"forecast_comparison.csv":self.metrics,
                    "holdout_forecasts.csv":pd.DataFrame(self.audit["holdout_predictions"])},self.audit)


def _classical_features(series, base):
    if base == "ses":
        level = series.ewm(alpha=0.3, adjust=False, ignore_na=True).mean()
        return level.shift().to_numpy()[:, None]
    if base in FIXED_LAGS:
        return np.column_stack([series.shift(lag) for lag in FIXED_LAGS[base]])
    cols = [np.ones(len(series))]
    if base.startswith("ar"):
        order = int(base[2:])
        cols.extend(series.shift(lag) for lag in range(1, order + 1))
    elif base in ("trend", "seasonal_trend"):
        cols.append(np.arange(len(series), dtype=float))
        if base == "seasonal_trend":
            cols.extend(np.eye(12)[series.index.month - 1, 1:].T)
    return np.column_stack(cols)


def _fit_base(x, y, end, base):
    if base in FIXED_BASES:
        return np.full(x.shape[1], 1 / x.shape[1])
    complete = np.isfinite(y[:end]) & np.isfinite(x[:end]).all(axis=1)
    if complete.sum() < 24:
        return None
    coef, _, rank, _ = np.linalg.lstsq(x[:end][complete], y[:end][complete], rcond=None)
    # Calendar effects need coverage of each month. Collinear AR lags can
    # still forecast a constant series using the minimum-norm solution.
    return coef if base.startswith("ar") or rank == x.shape[1] else None


def _ma1_forecasts(y, start, end, include_history):
    """Conditional least squares for y[t] = mean + e[t] + theta * e[t-1].

    For each invertible theta, filter the centered data and the intercept in
    parallel. Their difference gives the innovation at any candidate mean.
    The three running sums profile out the mean at each training cutoff.
    """
    observed = y[:start][np.isfinite(y[:start])]
    if len(observed) < 24:
        raise ValueError("MA(1) needs at least 24 observed training months.")
    center = observed[0]
    # Start with zero so a constant series selects the no-shock model on a tie.
    positive = np.arange(1, 100) / 100
    grid = np.r_[0., positive, -positive]
    raw_error = np.zeros_like(grid)
    loading = np.zeros_like(grid)
    sum_squared = np.zeros_like(grid)
    sum_cross = np.zeros_like(grid)
    sum_loading = np.zeros_like(grid)
    forecasts = np.full(end, np.nan)
    count = 0

    def fit():
        means = sum_cross / sum_loading
        best = np.argmin(sum_squared - means * sum_cross)
        return best, means[best]

    for t in range(start):
        if include_history and count >= 24:
            k, mean = fit()
            forecasts[t] = center + mean + grid[k] * (raw_error[k] - mean * loading[k])
        if not np.isfinite(y[t]):
            # Each contiguous run starts with zero expected unobserved shock.
            raw_error.fill(0)
            loading.fill(0)
            continue
        raw_error = (y[t] - center) - grid * raw_error
        loading = 1 - grid * loading
        sum_squared += raw_error ** 2
        sum_cross += raw_error * loading
        sum_loading += loading ** 2
        count += 1

    k, mean = fit()
    theta = grid[k]
    last_error = raw_error[k] - mean * loading[k]
    level = center + mean
    fit_info = {"mean": float(level), "theta": float(theta), "last_innovation": float(last_error)}
    for t in range(start, end):
        forecasts[t] = level + theta * last_error
        last_error = y[t] - forecasts[t] if np.isfinite(y[t]) else 0.
    return forecasts, fit_info


def _baseline_forecasts(series, base, start, end, include_history):
    y = series.to_numpy()
    if base == "ma1":
        return _ma1_forecasts(y, start, end, include_history)
    x = _classical_features(series, base)
    coef = _fit_base(x, y, start, base)
    if coef is None:
        raise ValueError(f"{BASES[base]} needs at least 24 complete training months after lagging "
                         "and enough data to fit its terms.")
    forecasts = np.full(end, np.nan)
    forecasts[start:end] = x[start:end] @ coef
    if include_history:
        origins = np.flatnonzero(np.isfinite(y[:start]) & np.isfinite(x[:start]).all(axis=1))
        for i in origins:
            earlier = coef if base in FIXED_BASES else _fit_base(x, y, i, base)
            if earlier is not None:
                forecasts[i] = x[i] @ earlier
    return forecasts, {}


def _learner(name):
    if name == "forest":
        return RandomForestRegressor(n_estimators=80, min_samples_leaf=4, random_state=422, n_jobs=1)
    if name == "boosting":
        return GradientBoostingRegressor(n_estimators=60, max_depth=2, min_samples_leaf=4, random_state=422)
    linear = {
        "lasso": Lasso(alpha=.025, max_iter=10000),
        "ridge": Ridge(alpha=1.0),
        "elastic_net": ElasticNet(alpha=.025, l1_ratio=.5, max_iter=10000),
    }
    return TransformedTargetRegressor(
        regressor=make_pipeline(StandardScaler(), linear[name]),
        transformer=StandardScaler())


def _block(s, families, base, learner, start, end):
    """Fit at the block origin. Update observed lags for each later one-step forecast."""
    y = s.to_numpy()
    test = np.arange(start, end)
    if "ml" in families or "hybrid" in families:
        features = pd.concat([s.shift(lag) for lag in (1, 2, 3, 12)], axis=1)
        angle = 2 * np.pi * (s.index.month - 1) / 12
        features["sin"], features["cos"] = np.sin(angle), np.cos(angle)
        x = features.to_numpy()
        test = test[np.isfinite(x[test]).all(axis=1)]
    if "classical" in families or "hybrid" in families:
        baseline, fit_info = _baseline_forecasts(s, base, start, end, "hybrid" in families)
        test = test[np.isfinite(baseline[test])]
    pred = pd.DataFrame(index=s.index[test])
    components = pd.DataFrame(index=pred.index)
    if not len(test):
        return pred, components
    if "classical" in families or "hybrid" in families:
        components["Baseline"] = baseline[test]
        components.attrs.update(fit_info)
        if "classical" in families:
            pred["classical"] = components["Baseline"]
    if "ml" in families or "hybrid" in families:
        train = np.isfinite(y[:start]) & np.isfinite(x[:start]).all(axis=1)
        if train.sum() < 24:
            raise ValueError("The ML fit needs at least 24 earlier complete months after lagging.")
        if "ml" in families:
            pred["ml"] = _learner(learner).fit(x[:start][train], y[:start][train]).predict(x[test])
        if "hybrid" in families:
            # Each training residual uses only information prior to its target month.
            errors = y[:start] - baseline[:start]
            usable = train & np.isfinite(errors)
            if usable.sum() < 24:
                raise ValueError("The correction needs 24 historical one-step errors after the baseline warm-up.")
            correction = _learner(learner).fit(x[:start][usable], errors[usable])
            components["ML correction"] = correction.predict(x[test])
            pred["hybrid"] = components.sum(axis=1, min_count=2)
    return pred, components


def _validate(mode,base,learner,holdout_months):
    if mode not in (*FAMILIES,"compare"):
        raise ValueError("Choose classical, ml, hybrid, or compare.")
    if mode!="ml" and base not in BASES:
        raise ValueError("Choose a classical component from " + ", ".join(BASES) + ".")
    if mode!="classical" and learner not in LEARNERS:
        raise ValueError("Choose an ML component from " + ", ".join(LEARNERS) + ".")
    if type(holdout_months)!=int or not 3<=holdout_months<=60:
        raise ValueError("Choose 3 to 60 holdout months.")


def _evaluate(s,families,base,learner,start,end):
    preds,_=_block(s,families,base,learner,start,end)
    if not len(preds):
        raise ValueError("No complete forecast inputs in the evaluation window.")
    preds["Actual"]=s.reindex(preds.index)
    scored=preds.dropna()
    if len(scored)<max(3,(end-start+1)//2):
        raise ValueError("Too few months with actual outcomes and complete forecast inputs for comparison.")
    errors=scored[list(families)].subtract(scored.Actual,axis=0)
    return scored, np.sqrt(errors.pow(2).mean()), errors.abs().mean()


def _development(series,mode,base,learner,holdout_months):
    _validate(mode,base,learner,holdout_months)
    families=list(FAMILIES) if mode=="compare" else [mode]
    split=len(series)-holdout_months
    first=split-VALIDATION_MONTHS
    if first<36:
        raise ValueError("Supply a longer monthly series. The analysis reserves separate validation and holdout windows.")
    valid,rmse,_=_evaluate(series.iloc[:split],families,base,learner,first,split)
    return {"validation_rmse":rmse.to_dict(),"choice":rmse.idxmin(),"validation_months":list(map(str,valid.index)),
            "validation_span":f"{series.index[first]} through {series.index[split-1]}",
            "target":series.name, "mode":mode,"base":base,"learner":learner}


def family_development(design,*,mode,base,learner,holdout_months):
    return _development(design,mode,base,learner,holdout_months)


def family_forecast(design,*,mode,base,learner,holdout_months=24):
    """Choose by validation if comparing, then score all declared models on untouched months."""
    a=_development(design,mode,base,learner,holdout_months)
    families=list(a["validation_rmse"])
    cutoff=len(design)-holdout_months
    hold,rmse,mae=_evaluate(design,families,base,learner,cutoff,len(design))
    future=design.reindex(pd.period_range(design.index[0],design.index[-1]+1,freq="M"))
    nxt,components=_block(future,families,base,learner,len(design),len(future))
    if nxt.empty:
        raise ValueError("The last observations are missing a calendar lag required for the selected forecast.")
    scores=pd.DataFrame({"Validation RMSE":a["validation_rmse"],"Holdout RMSE":rmse,
        "Holdout MAE":mae,"Next forecast":nxt.iloc[0]})
    scores=scores.loc[families].rename(index=FAMILIES).rename_axis("Forecaster")
    picked=FAMILIES[a["choice"]]
    month=str(future.index[-1])
    heading=(f"Validation recommends {picked} for {design.name}." if mode=="compare" else
             f"{picked} forecast of {design.name} for {month}.")
    detail=f"Validation covers {a['validation_span']}. Holdout errors use {len(hold)} common months from {design.index[cutoff]} through {design.index[-1]}."
    if base == "ma1" and components.attrs:
        ma = dict(components.attrs)
        a["ma1"] = ma
        detail += (f" The MA(1) forecast starts at the fitted mean ({ma['mean']:,.3f}) and adds "
                   f"{ma['theta']:.2f} times the last estimated shock ({ma['last_innovation']:,.3f}).")
    parts={}
    if "hybrid" in families:
        parts={k:float(v) for k,v in components.iloc[0].items()}
        detail+=(f" For {month}, {BASES[base]} predicts {parts['Baseline']:,.3f} and {LEARNERS[learner]} adds "
                 f"{parts['ML correction']:+,.3f}, giving {nxt['hybrid'].iloc[0]:,.3f} in target units.")
    a.update(task="family_forecast",summary=heading,interpretation=detail,forecast_month=month,
        next_components=parts,holdout_predictions=hold.rename(columns=FAMILIES).rename_axis("month").reset_index().astype({"month":str}).to_dict("records"))
    return FamilyForecast(scores,a)
