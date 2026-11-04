"""Public numerical functions available to Llama-generated Python programs."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from pydantic import Field

from studio_core.data_io import monthly_frame
from superlearner.ensemble import forecast_development, super_learner
from superlearner.forecasting import make_design
from studio_core.spec import Plan, Record
from rcf_screen import rcf_dml_screen
from dml import cross_section, cross_dml, ts_dml
from studio_core.workflow import save_bundle


@dataclass
class Design:
    frame: pd.DataFrame
    options: dict


def monthly_features(data, *, target, target_lags, predictors, predictor_lags, seasonality):
    options = dict(target=target, target_lags=target_lags, predictors=predictors,
                   predictor_lags=predictor_lags, seasonality=seasonality)
    Plan(**options, holdout_months=24)  # Validate the shared calendar-feature specification.
    return Design(monthly_frame(data), options)


class InferenceOptions(Record):
    term: str
    hac_lags: int = Field(ge=0, le=24)


@dataclass
class Estimate:
    metrics: pd.DataFrame
    audit: dict

    def save(self, destination):
        save_bundle(destination, {"coefficients.csv": self.metrics}, self.audit)


def ols_hac(design, *, term, hac_lags):
    """Conditional monthly association with a two-sided asymptotic 95% HAC interval."""
    options = InferenceOptions(term=term, hac_lags=hac_lags)
    X, y = make_design(design.frame, Plan(**design.options, holdout_months=24))
    if options.term not in X.columns:
        raise ValueError(f"Select a coefficient from {list(X)}.")
    complete = X.notna().all(axis=1) & y.notna()
    dates = X.index[complete]
    X = X.loc[complete].copy()
    X.insert(0, "intercept", 1.0)
    A = X.to_numpy()
    n, k = A.shape
    if n < max(36, 3*k, 2*hac_lags + 1):
        raise ValueError("Insufficient complete months for this model and HAC bandwidth.")
    b, _, rank, _ = np.linalg.lstsq(A, y.loc[dates].to_numpy(), rcond=None)
    if rank != k:
        raise ValueError("The regression has collinear terms. Revise the controls.")
    # Zero scores at excluded calendar months preserve actual lag distances in the HAC sum.
    scores = np.zeros((len(design.frame), k))
    positions = design.frame.index.get_indexer(dates)
    scores[positions] = A * (y.loc[dates].to_numpy() - A @ b)[:, None]
    meat = scores.T @ scores
    for lag in range(1, hac_lags + 1):
        cross = scores[lag:].T @ scores[:-lag]
        meat += (1 - lag/(hac_lags+1)) * (cross + cross.T)
    bread = np.linalg.inv(A.T @ A)
    se = np.sqrt(np.maximum(np.diag(bread @ meat @ bread.T) * n / (n-k), 0))
    z = 1.959963984540054
    table = pd.DataFrame({"estimate": b, "HAC_SE": se, "lower_95": b-z*se, "upper_95": b+z*se},
                         index=X.columns).rename_axis("term")
    audit = {"task": "estimation", "target": design.options["target"], "term": term,
             "n": n, "first_month": str(dates[0]), "last_month": str(dates[-1]),
             "hac_lags": hac_lags, "features": list(X.columns[1:]),
             "interpretation": "Adjusted linear association with a pointwise 95% HAC interval."}
    return Estimate(table, audit)
