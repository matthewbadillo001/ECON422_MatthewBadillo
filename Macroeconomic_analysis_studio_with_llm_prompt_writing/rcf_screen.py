"""Teaching diagnostics for reverse-cross-fitted DML on a monthly partially linear design."""
from dataclasses import dataclass
from itertools import combinations
from math import erfc, sqrt

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from studio_core.workflow import save_bundle


NAMES = ("Stationarity and weak dependence", "Time reversibility", "Conditional stability")
FLAG, QUIET, UNKNOWN = "Concern", "No flag detected", "Inconclusive"
LIMIT = ("These are limited empirical checks of the specified design. A quiet screen does not establish "
         "the population assumptions, causal identification, or the accuracy of nuisance estimation.")


@dataclass
class Assessment:
    metrics: pd.DataFrame
    audit: dict

    def save(self, destination):
        files = {"rcf_diagnostics.csv" if self.audit["task"]=="rcf_dml_screen" else "dml_diagnostics.csv": self.metrics}
        if self.audit.get("estimate") is not None:
            files["coefficient.csv"] = pd.DataFrame([self.audit["estimate"]], index=[self.audit["term"]])
        save_bundle(destination, files, self.audit)


def _long_run_variance(x):
    centered = x - x.mean()
    n = len(x)
    bandwidth = max(1, int(4 * (n / 100)**(2/9)))
    total = centered @ centered
    for h in range(1, bandwidth + 1):
        total += 2 * (1 - h / (bandwidth + 1)) * (centered[h:] @ centered[:-h])
    return max(0., total / n)


def _stationarity(z):
    stats, persistent, incomplete = {}, [], []
    for name, col in z.items():
        x = col.to_numpy()
        variance = _long_run_variance(x)
        if variance <= 1e-12:
            incomplete.append(name)
            continue
        kpss = float(np.sum(np.cumsum(x - x.mean())**2) / (len(x)**2 * variance))
        acf = [float(x[h:] @ x[:-h] / (x @ x)) for h in (6, 12)]
        stats[name] = {"level_KPSS": kpss, "ACF6": acf[0], "ACF12": acf[1]}
        if min(map(abs, acf)) >= .40:
            persistent.append(name)
    if not stats:
        return UNKNOWN, "No series with estimable long-run variance.", stats
    largest = max(stats, key=lambda name: stats[name]["level_KPSS"])
    flagged = stats[largest]["level_KPSS"] > .739 or bool(persistent)
    text = (f"Maximum level-KPSS {stats[largest]['level_KPSS']:.3f} ({largest}), per-series 1% reference 0.739. "
            f"|ACF| ≥ 0.40 at both 6 and 12 months: {', '.join(persistent) or 'none'}.")
    if incomplete:
        text += " Unassessable: " + ", ".join(incomplete) + "."
    return FLAG if flagged else UNKNOWN if incomplete else QUIET, text, stats


def _reversibility(z):
    # For a reversible joint process each antisymmetric moment has expectation zero.
    bounded = np.tanh(z.to_numpy())
    moments = []
    for h in (1, 3):
        a, b = bounded[:-h], bounded[h:]
        for i, name in enumerate(z):
            moments.append((f"{name}, nonlinear marginal, lag {h}", a[:,i]**2*b[:,i] - b[:,i]**2*a[:,i]))
        for i, j in combinations(range(z.shape[1]), 2):
            moments.append((f"{z.columns[i]} / {z.columns[j]}, joint lead–lag {h}",
                            a[:,i]*b[:,j] - b[:,i]*a[:,j]))
    tests = {}
    for label, sample in moments:
        variance = _long_run_variance(sample)
        if variance > 1e-14:
            statistic = sample.mean() * sqrt(len(sample) / variance)
            p = erfc(abs(statistic) / sqrt(2))
            tests[label] = min(1., len(moments) * p)
    if not tests:
        return UNKNOWN, "No informative direction-sensitive moment could be computed.", tests
    name = min(tests, key=tests.get)
    status = FLAG if tests[name] < .05 else QUIET if len(tests) == len(moments) else UNKNOWN
    return status, (f"Smallest Bonferroni-adjusted HAC p = {tests[name]:.3g} ({name}); "
                    f"{len(tests)}/{len(moments)} usable forward/reverse contrasts."), tests


def oriented_block(block, direction, spec, term):
    """Build lagged variables locally, from raw observations in one oriented block."""
    block = block.iloc[::direction]
    x = {}
    for name in (spec["target"], *spec["predictors"]):
        for lag in (spec["target_lags"] if name == spec["target"] else spec["predictor_lags"]):
            x[f"{name}__lag{lag}"] = block[name].shift(lag)
    if spec["seasonality"]:
        angle = 2*np.pi*(block.index.month-1)/12
        x["calendar__sin"], x["calendar__cos"] = np.sin(angle), np.cos(angle)
    design = pd.DataFrame(x, index=block.index)
    design["__Y"] = block[spec["target"]]
    design["__D"] = design.pop(term)
    return design.dropna()


def _edge_records(blocks, j, direction, spec, term, exposure):
    design = oriented_block(blocks[j], direction, spec, term)
    records = []
    for side in (-1, 1):
        neighbor = j + direction * side
        if not 0 <= neighbor < len(blocks):
            continue
        border = blocks[neighbor].iloc[-1 if neighbor < j else 0]
        edge = (design.iloc[:3] if side < 0 else design.iloc[-3:][::-1]).copy()
        edge["__side"] = side
        edge["__distance"] = abs(edge.index.asi8 - border.name.ordinal)
        edge["__boundary_y"], edge["__boundary_d"] = border[spec["target"]], border[exposure]
        edge["__donor"] = neighbor
        records.append(edge)
    return pd.concat(records)


def _stability(frame, spec, term, exposure):
    k = 8
    maxlag = max([*spec["target_lags"], *spec["predictor_lags"]])
    minimum = max(96, k*(maxlag+6))
    if len(frame) < minimum:
        return UNKNOWN, f"Need {minimum} contiguous months for eight blocks with these lags; have {len(frame)}.", {}
    blocks = [frame.iloc[idx] for idx in np.array_split(np.arange(len(frame)), k)]
    forward = [_edge_records(blocks, j, 1, spec, term, exposure) for j in range(k)]
    backward = [_edge_records(blocks, j, -1, spec, term, exposure) for j in range(k)]
    responses = ["__Y", "__D"]
    boundary = ["__boundary_y"] if spec["target"] == exposure else ["__boundary_y", "__boundary_d"]
    base_cols = [c for c in forward[0] if c not in {*responses, "__boundary_y", "__boundary_d", "__donor"}]
    edges = pd.concat(forward)
    controls = edges[[c for c in base_cols if c not in {"__side", "__distance"}]].to_numpy()
    controls = np.column_stack((np.ones(len(edges)), controls))
    d = edges["__D"].to_numpy()
    residual = d - controls @ np.linalg.lstsq(controls, d, rcond=None)[0]
    if np.var(d) == 0 or np.var(residual) <= 1e-8*np.var(d):
        return UNKNOWN, "The exposure has negligible variation after the specified controls on the boundary rows.", {}
    losses, fold_wins, heldout = np.zeros((2, 2)), np.zeros(2, dtype=int), []
    for j, test in enumerate(forward):
        train = pd.concat(forward[:j] + backward[j+1:])
        train = train.loc[train["__donor"] != j]  # No training feature can use the evaluation block.
        if len(train) < 2*(len(base_cols)+len(boundary)+1):
            return UNKNOWN, "Too few training boundary records for the requested number of controls.", {}
        fold = []
        for cols in (base_cols, base_cols + boundary):
            model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
            model.fit(train[cols], train[responses])
            err = test[responses].to_numpy() - model.predict(test[cols])
            fold.append(np.sum(err**2, axis=0))
        losses += fold
        fold_wins += fold[1] < fold[0]
        heldout.append(test[responses].to_numpy())
    values = np.vstack(heldout)
    total = np.sum((values-values.mean(axis=0))**2, axis=0)
    if (total <= 0).any() or (losses[0] <= 1e-8*total).any():
        return UNKNOWN, "The outcome or exposure has too little residual variation for this comparison.", {}
    gains = 1 - losses[1] / losses[0]
    status = FLAG if ((gains >= .10) & (fold_wins > k/2)).any() else QUIET
    details = {"gain_Y":float(gains[0]), "gain_D":float(gains[1]), "folds_improved":fold_wins.tolist(),
               "folds":k, "heldout_edge_rows":len(values)}
    return status, (f"Relative held-out MSE gains from neighboring block values: {gains[0]:+.1%} for Y "
                    f"and {gains[1]:+.1%} for D across {len(values)} edge months ({fold_wins[0]}/8 and {fold_wins[1]}/8 folds improved). "
                    "Flag rule: ≥10% gain and improvement in a majority of folds for either variable."), details


def rcf_dml_screen(design, *, term):
    """Screen the three RCF-DML requirements for Y, a named lagged exposure D, and the remaining controls X."""
    spec = design.options
    names = [spec["target"], *spec["predictors"]]
    missing = set(names) - set(design.frame.columns)
    if missing:
        raise ValueError(f"Unknown columns: {sorted(missing)}")
    terms = {f"{name}__lag{lag}":name for name in names
             for lag in (spec["target_lags"] if name == spec["target"] else spec["predictor_lags"])}
    if term not in terms:
        raise ValueError(f"Choose the observed exposure term D from {list(terms)}.")
    if len(names) > 12:
        raise ValueError("Use at most 12 distinct series in this teaching screen.")
    raw = design.frame[names].apply(pd.to_numeric, errors="raise")
    if np.isinf(raw.to_numpy(dtype=float)).any():
        raise ValueError("Replace infinite observations with missing values before screening.")
    complete = raw.notna().all(axis=1)
    spans = [part for _, part in raw[complete].groupby((~complete).cumsum()[complete])]
    frame = max(spans, key=lambda part: (len(part), part.index[-1])) if spans else raw.iloc[:0]
    n = len(frame)
    controls = [k for k in terms if k != term] + (["calendar sine/cosine"] if spec["seasonality"] else [])
    audit = {"task":"rcf_dml_screen", "target":spec["target"], "term":term, "controls":controls,
             "n":n, "calendar_months":len(raw), "complete_months":int(complete.sum()),
             "first_month":str(frame.index[0]) if n else None, "last_month":str(frame.index[-1]) if n else None,
             "seasonality":spec["seasonality"], "limits":LIMIT}
    rows = [(UNKNOWN,f"Need 60 contiguous complete months; have {n}.",{})] * 2
    if n >= 60:
        observed = frame.to_numpy(dtype=float)
        if spec["seasonality"]:
            a = 2*np.pi*(frame.index.month-1)/12
            calendar = np.column_stack((np.ones(n), np.sin(a), np.cos(a)))
            observed = observed - calendar @ np.linalg.lstsq(calendar, observed, rcond=None)[0]
        centered = observed - observed.mean(axis=0)
        scales = centered.std(axis=0)
        stable_scale = scales > 1e-12*np.maximum(1., np.max(np.abs(frame.to_numpy()), axis=0))
        z = pd.DataFrame(centered / np.where(stable_scale, scales, 1.), columns=names)
        z.loc[:,~stable_scale] = 0.
        rows = [_stationarity(z), _reversibility(z)]
    rows.append(_stability(frame, spec, term, terms[term]))
    actions = (
        "Inspect shifts or persistent dynamics before choosing transformations.",
        "Investigate the forward/reverse difference before using reversed blocks.",
        "Revisit the controls and lag history, then repeat the boundary comparison.",
    )
    rows = [(s, text + (" " + action if s == FLAG else ""), details)
            for (s, text, details), action in zip(rows, actions)]
    flagged = [name.lower() for name,(status,_,_) in zip(NAMES, rows) if status == FLAG]
    if flagged:
        status = "does_not_pass"
        flagged_names = ", ".join(flagged[:-1]) + " and " + flagged[-1] if len(flagged)>1 else flagged[0]
        verdict = f"This specification does not pass the time-series DML screen. The diagnostics flag {flagged_names}."
    elif any(s == UNKNOWN for s, _, _ in rows):
        status = "inconclusive"
        verdict = "The time-series DML screen is inconclusive. The data provide insufficient evidence to recommend reverse-cross-fitted DML."
    else:
        status = "no_flags"
        verdict = "No warning detected. RCF-DML is a candidate, subject to substantive justification."
    audit.update(screen_status=status, verdict=verdict, evidence={name:d for name,(_,_,d) in zip(NAMES,rows)},
                 interpretation=LIMIT)
    table = pd.DataFrame([(s,txt) for s,txt,_ in rows],columns=["Assessment","Evidence"],index=pd.Index(NAMES,name="Condition"))
    return Assessment(table,audit)
