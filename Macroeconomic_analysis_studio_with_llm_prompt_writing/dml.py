"""Partially linear DML for independent units or a monthly lag specification."""
from dataclasses import dataclass
import re

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, StratifiedKFold

from rcf_screen import Assessment, FLAG, QUIET, UNKNOWN, oriented_block, rcf_dml_screen


from studio_core.data_io import data_structure


def load_cross_demo(seed=422, n=800):
    """Independent simulated households; one more training hour raises earnings by 1.5 units."""
    if type(n) is not int or n < 1:
        raise ValueError("Choose a positive integer sample size.")
    rng = np.random.default_rng(seed)
    education, experience = rng.normal(size=(2,n))
    hours = 8 + education + .8*np.sin(experience) + rng.normal(size=n)
    earnings = 20 + 1.5*hours + 2*education + np.sin(experience) + rng.normal(size=n)
    return pd.DataFrame(dict(earnings=earnings, training_hours=hours, education=education, experience=experience))


def load_activity_demo(seed=422, n=480):
    """Stationary Gaussian AR(1) activity, with slope 0.4 and unit innovation variance."""
    if type(n) is not int or n < 1:
        raise ValueError("Choose a positive integer sample size.")
    rng = np.random.default_rng(seed)
    activity = rng.normal(size=n)
    activity[0] /= np.sqrt(1-.4**2)
    for t in range(1,n):
        activity[t] += .4*activity[t-1]
    return pd.DataFrame({"date":pd.date_range("1980-01-01",periods=n,freq="MS"), "activity":activity})


@dataclass
class CrossSection:
    data: pd.DataFrame
    target: str
    exposure: str
    controls: list


def cross_section(data, *, target, exposure, controls):
    if not isinstance(data, pd.DataFrame) or not isinstance(controls,list) or not all(
            isinstance(c,str) and c.strip() for c in [target,exposure,*controls]):
        raise ValueError("Supply a table, named outcome and exposure, and a list of numeric controls.")
    columns = [target,exposure,*controls]
    if len(set(columns)) != len(columns) or data.columns.has_duplicates:
        raise ValueError("Use distinct outcome, exposure and control columns.")
    if not set(columns) <= set(data.columns):
        raise ValueError(f"Unknown cross-sectional columns: {sorted(set(columns)-set(data.columns))}.")
    frame=data[columns].apply(pd.to_numeric,errors="raise")
    if np.isinf(frame.to_numpy(dtype=float)).any():
        raise ValueError("Replace infinite values with missing values.")
    return CrossSection(frame, target, exposure, controls)


def _residuals(train, test, controls, y, d):
    """Both nuisance learners see only this fold's training rows."""
    residuals = []
    for name in (y,d):
        if controls:
            model = RandomForestRegressor(n_estimators=80,min_samples_leaf=6,random_state=422,n_jobs=1)
            model.fit(train[controls],train[name])
            fitted = model.predict(test[controls])
        else:
            fitted = train[name].mean()
        residuals.append(test[name].to_numpy()-fitted)
    return np.column_stack(residuals)


def _interval(theta, se):
    return {"coefficient":float(theta), "SE":float(se), "lower_95":float(theta-1.96*se),
            "upper_95":float(theta+1.96*se)}


def cross_dml(design, *, estimate=True):
    """Pooled partialling-out score with shuffled independent-unit folds and robust inference."""
    if type(estimate) is not bool or not isinstance(design,CrossSection):
        raise ValueError("Use cross_section with cross_dml and specify estimate=True or False.")
    y,d,x = design.target,design.exposure,design.controls
    data = design.data.dropna().reset_index(drop=True)
    n=len(data)
    binary = bool(n and set(data[d].unique()) <= {0,1})
    limits = ("Independence of units, adequate nuisance learning and control of confounding need substantive justification. "
              "A causal interpretation also requires appropriate pre-exposure controls and exogeneity.")
    rows = [("Independent units","Requires justification",
             "Use one independent unit per row. Repeated, clustered or panel observations require a different split and uncertainty calculation."),
            ("Timing and confounding","Requires justification",
             "Explain why the chosen X precedes the exposure and captures common causes of Y and D. These assumptions cannot be verified from fit.")]
    details = {"task":"cross_dml","target":y,"term":d,"controls":x,"n":n,"input_rows":len(design.data),
               "folds":5,"binary":binary,"limits":limits,"interpretation":limits,
               "uncertainty":"independent-unit heteroskedasticity-robust","requested_estimate":estimate,"estimate":None}
    issue = None
    if n < max(60, 3*(len(x)+2)):
        issue = "Too few complete independent units for five-fold nuisance fitting."
    elif data[d].nunique() < 2 or data[y].nunique() < 2:
        issue = "The outcome or exposure has no usable variation."
    elif binary and data[d].value_counts().min() < 10:
        issue = "Fewer than ten complete units in one exposure group."
    else:
        # OOF model error alone can create spurious residual variation when D is exactly spanned by X.
        a = np.column_stack([np.ones(n),data[x].to_numpy()])
        linear_resid = data[d].to_numpy() - a @ np.linalg.lstsq(a,data[d],rcond=None)[0]
        if np.var(linear_resid) <= 1e-8*np.var(data[d]):
            issue = "D is constant or nearly a linear combination of the supplied controls."
    if issue:
        rows.append(("Exposure support and sample size",UNKNOWN,issue))
    else:
        folds = (StratifiedKFold(5,shuffle=True,random_state=422).split(data,data[d]) if binary else
                 KFold(5,shuffle=True,random_state=422).split(data))
        residual = np.empty((n,2))
        for train,test in folds:
            residual[test]=_residuals(data.iloc[train],data.iloc[test],x,y,d)
        u,v = residual.T
        ratio = float(np.mean(v*v)/np.var(data[d]))
        weak = ratio < .01
        rows.append(("Residual exposure variation",FLAG if weak else QUIET,
                     f"Out-of-fold exposure MSE / exposure variance = {ratio:.3g}. The teaching warning threshold is 0.01."))
        details.update(exposure_mse_ratio=ratio,nuisance_rmse_Y=float(np.sqrt(np.mean(u*u))),
                       nuisance_rmse_D=float(np.sqrt(np.mean(v*v))))
        rows.append(("Held-out nuisance prediction","Descriptive",
                     f"Outcome RMSE {details['nuisance_rmse_Y']:.3g}; exposure RMSE {details['nuisance_rmse_D']:.3g}. These measure predictive fit, not causal validity."))
        if binary:
            propensity = data[d].to_numpy()-v
            extreme=float(np.mean((propensity<.05)|(propensity>.95)))
            poor=extreme > .10
            rows.append(("Binary exposure overlap",FLAG if poor else QUIET,
                         f"{extreme:.1%} of out-of-fold propensities are outside [0.05, 0.95]; flag if over 10%."))
            weak = weak or poor
        if weak:
            issue = "The exposure-support diagnostic raised a concern."
        elif estimate:
            theta = (v @ u)/(v @ v)
            influence = v*(u-theta*v)/(v @ v)
            details["estimate"] = _interval(theta,np.sqrt(n/(n-1)*(influence@influence)))
    verdict = ("Review the cross-sectional design. " + issue if issue else
               "Cross-sectional PLR is a candidate, conditional on the sampling and identification assumptions.")
    details["verdict"]=verdict
    table=pd.DataFrame(rows,columns=["Condition","Assessment","Evidence"]).set_index("Condition")
    return Assessment(table,details)


def ts_dml(design, *, term, hac_lags=3):
    """Mean of fold PLR slopes after the RCF screen, with calendar-time HAC influence contributions."""
    if type(hac_lags) is not int or not 0 <= hac_lags <= 24:
        raise ValueError("Use an integer HAC bandwidth of 0 to 24 months.")
    report = rcf_dml_screen(design,term=term)
    audit=report.audit
    audit.update(task="ts_dml",estimate=None,hac_lags=hac_lags,folds=8,
                 uncertainty=f"calendar-time Bartlett HAC ({hac_lags} lags)")
    audit["limits"] += " The nominal interval assumes the model, score and nuisance-rate conditions and does not adjust for screening."
    if audit["screen_status"] != "no_flags":
        return report
    spec=design.options
    frame=design.frame.loc[audit["first_month"]:audit["last_month"],[spec["target"],*spec["predictors"]]].astype(float)
    blocks=[frame.iloc[idx] for idx in np.array_split(np.arange(len(frame)),8)]
    forward=[oriented_block(b,1,spec,term) for b in blocks]
    reverse=[oriented_block(b,-1,spec,term) for b in blocks]
    controls=[c for c in forward[0] if c not in {"__Y","__D"}]
    contributions=pd.Series(0.,index=frame.index)
    fits=[]
    for k,test in enumerate(forward):
        train=pd.concat(forward[:k]+reverse[k+1:])
        u,v=_residuals(train,test,controls,"__Y","__D").T
        denom=v@v
        if denom<=1e-10*len(v)*np.var(frame[term.rsplit("__lag",1)[0]]) or denom==0:
            audit["verdict"]="Withhold the estimate. Too little residual exposure variation in a held-out block."
            return report
        slope=(v@u)/denom
        contributions.loc[test.index]=v*(u-slope*v)/(8*denom)
        fits.append({"fold":k+1,"train_n":len(train),"test_n":len(test),"first":str(test.index[0]),
                     "last":str(test.index[-1]),"slope":float(slope)})
    g=contributions.to_numpy()
    variance=g@g
    for lag in range(1,hac_lags+1):
        variance += 2*(1-lag/(hac_lags+1))*(g[lag:]@g[:-lag])
    audit.update(estimate=_interval(np.mean([f["slope"] for f in fits]),np.sqrt(max(0.,variance))),
                 fold_details=fits,scored_n=sum(f["test_n"] for f in fits))
    return report
