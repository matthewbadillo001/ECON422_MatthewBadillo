"""The index denotes the target month. All observed predictors are lagged."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

NAMES = ["lasso", "random_forest"]


def make_design(frame, plan):
    required = [plan.target, *plan.predictors]
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"Unknown columns: {sorted(missing)}")
    series = frame[required].apply(pd.to_numeric, errors="raise").astype(float)
    if np.isinf(series.to_numpy()).any():
        raise ValueError("The selected series contain infinite values.")
    index = pd.period_range(frame.index[0], frame.index[-1] + 1, freq="M", name="month")
    series = series.reindex(index)
    features = {}
    for column in required:
        lags = plan.target_lags if column == plan.target else plan.predictor_lags
        for lag in lags:
            features[f"{column}__lag{lag}"] = series[column].shift(lag)
    if plan.seasonality:
        angle = 2 * np.pi * (index.month - 1) / 12
        features["calendar__sin"] = np.sin(angle)
        features["calendar__cos"] = np.cos(angle)
    return pd.DataFrame(features, index=index), series[plan.target]


def base_predictions(X, y, train, test):
    models = [
        make_pipeline(StandardScaler(), Lasso(alpha=0.025, max_iter=10000)),
        RandomForestRegressor(n_estimators=100, min_samples_leaf=4,
                              max_features=1.0, random_state=422, n_jobs=1),
    ]
    return np.column_stack([m.fit(X.loc[train], y.loc[train]).predict(X.loc[test])
                            for m in models])


def convex_weights(actual, predictions):
    """Exact squared-error solution for two nonnegative weights summing to one."""
    lasso, forest = predictions.T
    delta = lasso - forest
    denom = delta @ delta
    w = 0.5 if denom == 0 else float(np.clip((delta @ (actual - forest)) / denom, 0, 1))
    return np.array([w, 1 - w])


def score(predictions):
    records = {}
    for name in predictions.columns.drop("actual"):
        e = predictions[name] - predictions.actual
        records[name] = {"RMSE": float(np.sqrt(np.mean(e**2))), "MAE": float(np.mean(abs(e)))}
    return pd.DataFrame.from_dict(records, orient="index").rename_axis("model")


def development(frame, plan):
    # The controller passes only the development prefix.
    X, y = make_design(frame, plan)
    usable = X.index[X.notna().all(axis=1) & y.notna()]
    if len(usable) < 60:
        raise ValueError("Need at least 60 complete pre-holdout target months after lagging.")
    initial = max(36, len(usable) // 2)
    parts, folds = [], []
    for number, indexes in enumerate(np.array_split(np.arange(initial, len(usable)), 4), 1):
        test = usable[indexes]
        train = usable[:indexes[0]]
        pred = base_predictions(X, y, train, test)
        block = pd.DataFrame(pred, index=test, columns=NAMES)
        block.insert(0, "actual", y.loc[test])
        block["fold"] = number
        parts.append(block)
        folds.append({"fold": number, "train_first": str(train[0]), "train_last": str(train[-1]),
                      "train_n": len(train), "validate_first": str(test[0]),
                      "validate_last": str(test[-1]), "validate_n": len(test)})
    oof = pd.concat(parts)
    weights = convex_weights(oof.actual.to_numpy(), oof[NAMES].to_numpy())
    oof["super_learner"] = oof[NAMES].to_numpy() @ weights
    evidence = {"features": list(X), "complete_development_months": len(usable),
                "calendar_months_in_development": len(frame),
                "oof_months": len(oof), "folds": folds, "weights": dict(zip(NAMES, weights.tolist())),
                "base_oof_scores": score(oof[["actual", *NAMES]]).to_dict("index")}
    return weights, oof, evidence


def evaluate(frame, plan, weights):
    X, y = make_design(frame, plan)
    complete = X.notna().all(axis=1) & y.notna()
    cutoff = frame.index[-1] - plan.holdout_months + 1
    train = X.index[complete & (X.index < cutoff)]
    eligible = complete & (X.index >= cutoff) & y.shift(1).notna()
    test = X.index[eligible]
    if not len(test):
        raise ValueError("No complete holdout months for a common model comparison.")
    pred = base_predictions(X, y, train, test)
    out = pd.DataFrame(pred, index=test, columns=NAMES)
    out.insert(0, "actual", y.loc[test])
    out["super_learner"] = pred @ weights
    out["last_observation"] = y.shift(1).loc[test]
    next_month = X.index[-1]
    future = {"month": str(next_month), "forecast": None}
    if X.loc[next_month].notna().all():
        final = base_predictions(X, y, X.index[complete], [next_month])[0]
        future["forecast"] = float(final @ weights)
    else:
        future["reason"] = "Missing latest lagged values required by the accepted plan."
    return out, future
