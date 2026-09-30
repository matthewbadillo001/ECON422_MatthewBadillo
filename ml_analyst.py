"""ML Analyst: validate data, tune six regression methods, hand off evidence.

This is a bounded, rule-based analysis agent. All actions run locally in Python.
All learned preprocessing stays inside cross-validation pipelines.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import importlib.metadata
import json
import platform
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.model_selection import GridSearchCV, KFold, TimeSeriesSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor


@dataclass(frozen=True)
class AnalysisConfig:
    methods: tuple[str, ...] | None = None  # None runs all six candidates
    target: str = "growth_next_year"
    test_size: float = 0.20
    folds: int = 5
    seed: int = 422
    split: str = "random"  # independent rows only; use 'time' for a single series
    time_column: str | None = None
    gap: int = 0  # purge rows between estimation and evaluation in time mode
    n_jobs: int = 1
    dataset_label: str = "Simulated regional economic observations"
    target_units: str = "percentage points"


def make_demo_data(n=800, seed=422):
    """Independent synthetic regions; no real economic estimates or forecasts."""
    rng = np.random.default_rng(seed)
    unemployment = rng.uniform(3, 11, n)
    spread = rng.uniform(0.3, 4.5, n)
    inflation = rng.normal(2.5, 1.2, n)
    investment = rng.normal(3, 2, n)
    confidence = 100 - 2 * unemployment - 3 * spread + rng.normal(0, 5, n)
    policy_rate = 1 + 0.65 * inflation + rng.normal(0, 0.7, n)
    growth = (3.5 - 0.25 * unemployment + 0.4 * investment
              - 0.18 * (inflation - 2)**2
              - 2.5 * ((unemployment > 6.5) & (spread > 2.3))
              + 0.015 * (confidence - 80) + rng.normal(0, 0.65, n))
    df = pd.DataFrame(dict(unemployment=unemployment, credit_spread=spread,
                           inflation=inflation, investment_growth=investment,
                           confidence=confidence, policy_rate=policy_rate,
                           growth_next_year=growth))
    # Missing predictors are imputed inside each estimation fold.
    for c in ["confidence", "investment_growth"]:
        df.loc[rng.choice(n, size=max(1, n // 40), replace=False), c] = np.nan
    return df


def prepare_data(df, features, config):
    """Split once. Return the sealed testing data to the notebook, not ML Analyst."""
    if config.split not in {"random", "time"}:
        raise ValueError("split must be random or time")
    if not 0 < config.test_size < 0.5 or config.folds < 2 or config.gap < 0:
        raise ValueError("Invalid test_size, folds, or gap")
    if not features or len(features) != len(set(features)):
        raise ValueError("Provide a nonempty list of distinct predictor names")
    if config.target in features or config.time_column in features:
        raise ValueError("Exclude the target and raw time column from predictors")
    if len(df) < 80:
        raise ValueError("This teaching workflow requires at least 80 rows")
    frame = df.copy().reset_index(drop=True)
    frame.index.name = "row_id"
    if config.split == "time":
        if not config.time_column:
            raise ValueError("Time mode requires time_column")
        dates = pd.to_datetime(frame[config.time_column], errors="raise")
        if dates.isna().any() or dates.duplicated().any():
            raise ValueError("Time mode requires one unique, nonmissing time per row; panel data need a custom splitter")
        frame = frame.loc[dates.sort_values(kind="stable").index]
    elif config.gap:
        raise ValueError("gap is only used in time mode")
    X = frame.loc[:, features].apply(pd.to_numeric, errors="raise").astype(float)
    y = pd.to_numeric(frame[config.target], errors="raise").astype(float)
    if not np.isfinite(y).all() or y.nunique() < 2:
        raise ValueError("Target must be finite, nonmissing, and nonconstant")
    if np.isinf(X.to_numpy()).any() or X.isna().all().any():
        raise ValueError("Predictors cannot contain infinity or be entirely missing")
    if config.split == "time":
        cut = len(frame) - int(np.ceil(len(frame) * config.test_size))
        end = cut - config.gap
        if end < config.folds * 5:
            raise ValueError("Too few estimation rows after the time gap")
        Xe, Xt, ye, yt = X.iloc[:end], X.iloc[cut:], y.iloc[:end], y.iloc[cut:]
    else:
        Xe, Xt, ye, yt = train_test_split(X, y, test_size=config.test_size,
                                        random_state=config.seed)
    if Xe.isna().all().any():
        raise ValueError("A predictor is entirely missing in the estimation sample")
    return Xe, Xt, ye, yt


def candidate_specs(seed):
    return {
        "Ridge": (Ridge(), {"model__alpha": [0.01, 0.1, 1, 10, 100]}, True),
        "Lasso": (Lasso(max_iter=20000),
                  {"model__alpha": [0.001, 0.01, 0.1, 0.5, 1]}, True),
        "Elastic Net": (ElasticNet(max_iter=20000),
                        {"model__alpha": [0.001, 0.01, 0.1, 0.5],
                         "model__l1_ratio": [0.2, 0.5, 0.8]}, True),
        "CART": (DecisionTreeRegressor(random_state=seed),
                 {"model__max_depth": [2, 4, 6],
                  "model__min_samples_leaf": [5, 15]}, False),
        "Random Forest": (RandomForestRegressor(n_estimators=160, random_state=seed, n_jobs=1),
                          {"model__max_features": [0.7, 1.0],
                           "model__min_samples_leaf": [2, 8],
                           "model__max_depth": [None, 6]}, False),
        "Gradient Boosting": (GradientBoostingRegressor(random_state=seed),
                              {"model__n_estimators": [100, 200],
                               "model__learning_rate": [0.03, 0.1],
                               "model__max_depth": [1, 2, 3]}, False),
    }


class MLAnalyst:
    """Owns estimation data and a fixed menu of model-fitting tools."""
    def __init__(self, config=AnalysisConfig(), progress=None):
        self.config = config
        self.progress = progress

    def _notify(self, stage, message):
        print(f"ML Analyst: {message}", flush=True)
        if self.progress:
            self.progress("ML Analyst", stage, message)

    def run(self, X_est, y_est, output_dir):
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=False)  # do not mix old and new runs
        cfg = self.config
        splitter = (TimeSeriesSplit(n_splits=cfg.folds, gap=cfg.gap)
                    if cfg.split == "time" else
                    KFold(n_splits=cfg.folds, shuffle=True, random_state=cfg.seed))
        splits = list(splitter.split(X_est))
        for a, _ in splits:
            if X_est.iloc[a].isna().all().any():
                raise ValueError("An estimation fold has an entirely missing feature")
        specs = candidate_specs(cfg.seed)
        names = tuple(specs) if cfg.methods is None else tuple(cfg.methods)
        if not names or len(set(names)) != len(names) or any(n not in specs for n in names):
            raise ValueError("Choose one or more distinct supported methods")
        records, models, log = [], {}, []
        for name in names:
            model, grid, scale = specs[name]
            steps = [("imputer", SimpleImputer(strategy="median", keep_empty_features=True))]
            if scale:
                steps.append(("scaler", StandardScaler()))
            steps.append(("model", model))
            search = GridSearchCV(Pipeline(steps), grid, cv=splits,
                                  scoring="neg_mean_squared_error", refit=True,
                                  n_jobs=cfg.n_jobs, error_score="raise")
            self._notify("Running", f"Tuning {name}")
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                search.fit(X_est, y_est)
            fold_mse = [-float(search.cv_results_[f"split{k}_test_score"][search.best_index_])
                        for k in range(cfg.folds)]
            records.append(dict(model=name, cv_mse=float(np.mean(fold_mse)),
                                cv_rmse=float(np.sqrt(np.mean(fold_mse))),
                                fold_mse_sd=float(np.std(fold_mse, ddof=1)),
                                best_params=json.dumps(search.best_params_)))
            self._notify("Running", f"{name} completed; CV MSE = {np.mean(fold_mse):.4f}")
            models[name] = search.best_estimator_
            search_table = pd.DataFrame(search.cv_results_)
            search_table.to_csv(out / (name.lower().replace(" ", "_") + "_search.csv"), index=False)
            log.append(dict(agent="ML Analyst", model=name, fold_mse=fold_mse,
                            warnings=sorted(set(str(w.message) for w in caught))))
        leaderboard = pd.DataFrame(records).sort_values(["cv_mse", "model"]).reset_index(drop=True)
        baseline = DummyRegressor(strategy="mean").fit(X_est, y_est)
        data_hash = hashlib.sha256(pd.util.hash_pandas_object(
            X_est.assign(__target__=np.asarray(y_est)), index=True).values.tobytes()).hexdigest()
        manifest = dict(config=asdict(cfg), n_estimation=len(X_est),
                        features=list(X_est.columns), estimation_row_ids=X_est.index.tolist(),
                        estimation_hash=data_hash, selection_rule="Minimum mean CV MSE; exact ties by model name",
                        cv_is_selection_score=True, python=platform.python_version(),
                        versions={p: importlib.metadata.version(p)
                                  for p in ["numpy", "pandas", "scikit-learn", "joblib"]})
        leaderboard.to_csv(out / "leaderboard.csv", index=False)
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        (out / "ml_analyst_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
        # Only load joblib files produced by this trusted workflow.
        joblib.dump(dict(models=models, baseline=baseline, X_est=X_est,
                         leaderboard=leaderboard, manifest=manifest), out / "analysis_results.joblib")
        self._notify("Completed", "Selected models and CV results saved; testing outcomes remain unused.")
        return out / "analysis_results.joblib"
