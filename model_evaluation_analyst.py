"""Model Evaluation Analyst: select, evaluate, explain, and report locally.

Python selects by CV MSE and assembles the report from computed evidence.
"""
from pathlib import Path
import base64
import html
import importlib.metadata
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def explain_pipeline(pipeline, background, X):
    """Explain the fitted model on identically preprocessed, one-to-one features.

Interventional SHAP uses an empirical estimation-sample background. It can
combine correlated features unrealistically and does not identify causal effects.
"""
    B = pipeline[:-1].transform(background)
    Z = pipeline[:-1].transform(X)
    model = pipeline.named_steps["model"]
    if hasattr(model, "coef_"):
        explainer = shap.LinearExplainer(model, shap.maskers.Independent(B, max_samples=len(B)))
        raw = explainer(Z)
        kind = "LinearExplainer with independent background masker"
    else:
        explainer = shap.TreeExplainer(model, data=B,
                                      feature_perturbation="interventional",
                                      model_output="raw")
        raw = explainer(Z, check_additivity=True)
        kind = "TreeExplainer with interventional background"
    values = np.asarray(raw.values)
    bases = np.broadcast_to(np.asarray(raw.base_values).reshape(-1), (len(X),))
    pred = pipeline.predict(X)
    error = float(np.max(np.abs(bases + values.sum(axis=1) - pred)))
    if not np.allclose(bases + values.sum(axis=1), pred, atol=1e-5, rtol=1e-5):
        raise RuntimeError(f"SHAP additivity failed: max error={error}")
    # Plot values in original feature units, with fitted median imputation.
    display_data = pipeline.named_steps["imputer"].transform(X)
    explanation = shap.Explanation(values=values, base_values=bases,
                                   data=display_data, feature_names=list(X.columns))
    return explanation, kind, error


def _table_md(df):
    def cell(v):
        return str(v).replace("|", "\\|").replace("\n", " ")
    return ("| " + " | ".join(map(cell, df.columns)) + " |\n| "
            + " | ".join(["---"] * len(df.columns)) + " |\n"
            + "\n".join("| " + " | ".join(cell(v) for v in row) + " |"
                         for row in df.itertuples(index=False, name=None)))


def selection_summary(board):
    """Explain the existing CV-based choice; board is sorted by CV MSE/name."""
    chosen = str(board.iloc[0]["model"])
    score = float(board.iloc[0]["cv_mse"])
    if len(board) == 1:
        return (f"The Model Evaluation Analyst will use {chosen}, the only method "
                f"specified by the user (CV MSE = {score:.4f}). Cross-validation "
                "tuned its hyperparameters; no comparison across methods was performed. "
                f"Testing evaluation and SHAP analysis will use {chosen} only.")
    comparison = "; ".join(
        f"{row['model']}: CV MSE = {float(row['cv_mse']):.4f}"
        for _, row in board.iterrows()
    )
    tied = board[board["cv_mse"] == score]
    tie_note = (
        " Several candidates have exactly the same minimum CV MSE; "
        "the model-name alphabetical tie-break rule selects this model."
        if len(tied) > 1 else ""
    )
    return (f"The Model Evaluation Analyst selected {chosen} for testing evaluation "
            f"and SHAP analysis because it has the lowest mean CV MSE among the "
            f"{len(board)} user-selected methods. {comparison}.{tie_note} "
            f"SHAP analysis will explain {chosen} only. Testing outcomes and SHAP "
            "values were not used to select the model.")


class ModelEvaluationAnalyst:
    def __init__(self, seed=422, background_size=100, max_explain=120,
                 progress=None):
        if background_size < 2 or max_explain < 2:
            raise ValueError("background_size and max_explain must be at least two")
        self.seed = seed
        self.background_size = background_size
        self.max_explain = max_explain
        self.progress = progress

    def _notify(self, stage, message):
        print(f"Model Evaluation Analyst: {message}", flush=True)
        if self.progress:
            self.progress("Model Evaluation Analyst", stage, message)

    def run(self, analysis_results_path, X_test, y_test, output_dir):
        # Trust boundary: joblib can execute code. Use your own ML Analyst output only.
        analysis_results = joblib.load(analysis_results_path)
        board = analysis_results["leaderboard"].sort_values(["cv_mse", "model"]).reset_index(drop=True)
        if len(board) < 1 or not np.isfinite(board.cv_mse).all():
            raise ValueError("Expected at least one valid CV result")
        if list(X_test.columns) != analysis_results["manifest"]["features"]:
            raise ValueError("Testing features/order differ from the analysis_results")
        if set(X_test.index) & set(analysis_results["X_est"].index):
            raise ValueError("Estimation/testing row IDs overlap")
        if len(X_test) != len(y_test) or not X_test.index.equals(y_test.index):
            raise ValueError("Testing X and y must have matching row IDs and order")
        if len(X_test) < 2 or not np.isfinite(y_test).all():
            raise ValueError("Testing outcomes must be finite, with at least two rows")
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=False)
        cfg = analysis_results["manifest"]["config"]
        # The choice is frozen and written BEFORE test predictions or SHAP.
        chosen = str(board.iloc[0]["model"])
        runner = str(board.iloc[1]["model"]) if len(board) > 1 else None
        gap = float((board.iloc[1].cv_mse - board.iloc[0].cv_mse)
                    / max(board.iloc[0].cv_mse, 1e-12)) if len(board) > 1 else None
        decision = dict(selected_model=chosen, runner_up=runner,
                        selection_rule=analysis_results["manifest"]["selection_rule"],
                        runner_up_relative_mse_gap=gap, near_tie=gap is not None and gap <= .02,
                        test_used_for_selection=False, shap_used_for_selection=False)
        decision["selection_explanation"] = selection_summary(board)
        (out / "selection.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
        self._notify("Running", f"MODEL CHOICE: {chosen}. " + decision["selection_explanation"])
        pipe = analysis_results["models"][chosen]
        pred = pipe.predict(X_test)
        baseline_pred = analysis_results["baseline"].predict(X_test)
        metrics = dict(test_mse=float(mean_squared_error(y_test, pred)),
                       test_rmse=float(np.sqrt(mean_squared_error(y_test, pred))),
                       test_mae=float(mean_absolute_error(y_test, pred)),
                       test_r2=float(r2_score(y_test, pred)),
                       baseline_rmse=float(np.sqrt(mean_squared_error(y_test, baseline_pred))))
        metrics["beats_baseline"] = metrics["test_rmse"] < metrics["baseline_rmse"]
        predictions = pd.DataFrame({"observed": y_test, "predicted": pred,
                                    "baseline_prediction": baseline_pred}, index=X_test.index)
        predictions.to_csv(out / "test_predictions.csv")
        background = analysis_results["X_est"].sample(min(self.background_size, len(analysis_results["X_est"])),
                                            random_state=self.seed)
        X_explain = (X_test if len(X_test) <= self.max_explain else
                     X_test.sample(self.max_explain, random_state=self.seed)).sort_index()
        self._notify("Running", f"Computing SHAP for {len(X_explain)} testing observations")
        explanation, kind, error = explain_pipeline(pipe, background, X_explain)
        importance = pd.DataFrame({"feature": X_test.columns,
                                   "mean_abs_shap": np.abs(explanation.values).mean(axis=0)})
        importance = importance.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
        importance.to_csv(out / "shap_importance.csv", index=False)
        pd.DataFrame(explanation.values, index=X_explain.index,
                     columns=X_explain.columns).to_csv(out / "shap_values.csv")
        local = pd.DataFrame({"feature": X_explain.columns,
                              "value_after_imputation": explanation.data[0],
                              "shap": explanation.values[0]})
        local.to_csv(out / "local_explanation.csv", index=False)
        shap_meta = dict(explainer=kind, definition="interventional empirical-background SHAP",
                         background_row_ids=background.index.tolist(),
                         explained_row_ids=X_explain.index.tolist(),
                         background_size=len(background), n_explained=len(X_explain),
                         max_additivity_error=error,
                         baseline_prediction=float(explanation.base_values[0]),
                         local_row_id=int(X_explain.index[0]),
                         local_prediction=float(pipe.predict(X_explain.iloc[[0]])[0]),
                         shap_version=importlib.metadata.version("shap"))
        (out / "shap_metadata.json").write_text(json.dumps(shap_meta, indent=2), encoding="utf-8")
        self._notify("Running", f"SHAP reconstruction passed (max error {error:.2e}); drawing figures")
        self._plots(board, y_test, pred, explanation, out, cfg["target_units"])
        notes = [
            "CV scores tune hyperparameters and select the model. They can be optimistic after selection; the untouched testing sample provides the final predictive assessment.",
            "A lower observed error does not establish statistically significant superiority. Fold standard deviations describe variation, not confidence intervals.",
            "The baseline predicts the estimation-sample mean. A testing result that fails to improve on it is flagged for further review, without switching models on the same testing sample.",
            "Mean absolute SHAP summarizes the magnitude of prediction contributions on the explained subset. It is not a decomposition of R-squared or a measure of predictive accuracy.",
            "A positive SHAP value raises this observation's prediction relative to the background baseline under the chosen SHAP definition. It does not imply better fit or a causal effect.",
            "Interventional masking can create unlikely feature combinations when predictors are correlated. Attributions depend on the background and the fitted model.",
            "This workflow predicts a continuous outcome. Causal effects, treatment-effect heterogeneity, clustered/panel inference, and uncertainty intervals require additional design."]
        if cfg["split"] == "random":
            notes.append("Random splitting assumes independent rows. For real time-series or panel data, redesign the split to match the intended forecasting task and information availability.")
        else:
            notes.append("Time mode uses forward folds and a final contiguous testing block. Set the gap for label overlap and use predictors available at each forecast origin. Release lags and revised data are the user's responsibility.")
        if decision["near_tie"]:
            notes.append("The runner-up is within 2% of the best CV MSE. This is a descriptive near-tie flag, not a significance test.")
        if not metrics["beats_baseline"]:
            notes.append("REVIEW REQUIRED: the selected model did not beat the mean-prediction baseline on testing RMSE.")
        evidence = dict(dataset=cfg["dataset_label"], target=cfg["target"],
                        units=cfg["target_units"], decision=decision, metrics=metrics,
                        n_estimation=len(analysis_results["X_est"]), n_testing=len(X_test),
                        config=cfg, selected_parameters=json.loads(board.iloc[0].best_params),
                        leaderboard=board.to_dict("records"), shap=shap_meta,
                        importance=importance.to_dict("records"), limitations=notes)
        (out / "evidence.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
        self._notify("Running", "Writing the report from computed evidence")
        self._report(evidence, board, importance, local, notes, out)
        (out / "evaluation_log.json").write_text(json.dumps({"decision": decision,
            "additivity_passed": True, "execution": "local Python"}, indent=2), encoding="utf-8")
        self._notify("Completed", "SHAP checks passed; final_report.html is ready")
        return dict(report=out / "final_report.html", evidence=evidence, importance=importance,
                    explanation=explanation, selection=decision)

    @staticmethod
    def _plots(board, y, pred, explanation, out, units):
        plt.rcParams.update({"font.size": 11, "figure.dpi": 140})
        fig, ax = plt.subplots(figsize=(8, 4.4))
        ax.barh(board.model[::-1], board.cv_rmse[::-1], color="#36556e")
        ax.set_xlabel("Square root of mean CV MSE (" + units + ")")
        ax.set_title("Model comparison within the estimation sample")
        fig.tight_layout(); fig.savefig(out / "model_comparison.png"); plt.close(fig)
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.scatter(y, pred, alpha=.6, color="#36556e", s=18)
        low, high = min(np.min(y), np.min(pred)), max(np.max(y), np.max(pred))
        ax.plot([low, high], [low, high], color="black", linestyle="--")
        ax.set(xlabel="Observed outcome (" + units + ")", ylabel="Predicted outcome (" + units + ")",
               title="Untouched testing sample")
        fig.tight_layout(); fig.savefig(out / "test_predictions.png"); plt.close(fig)
        shap.plots.bar(explanation, max_display=10, show=False)
        plt.gcf().set_size_inches(8, 4.5); plt.tight_layout()
        plt.savefig(out / "shap_global.png", bbox_inches="tight"); plt.close("all")
        shap.plots.beeswarm(explanation, max_display=10, show=False)
        plt.savefig(out / "shap_beeswarm.png", bbox_inches="tight"); plt.close("all")
        shap.plots.waterfall(explanation[0], max_display=10, show=False)
        plt.savefig(out / "shap_local.png", bbox_inches="tight"); plt.close("all")

    @staticmethod
    def _report(e, board, importance, local, notes, out):
        d, m, s = e["decision"], e["metrics"], e["shap"]
        paragraphs = [
            f"The ML Analyst fitted {len(board)} user-selected method(s). The Model Evaluation Analyst generated this report locally from computed results.",
            f"Dataset: {e['dataset']}. Outcome: {e['target']} ({e['units']}).",
            d.get("selection_explanation") or selection_summary(board),
            f"Testing RMSE is {m['test_rmse']:.3f} {e['units']}, MAE is {m['test_mae']:.3f}, and R-squared is {m['test_r2']:.3f}. The mean-prediction baseline has RMSE {m['baseline_rmse']:.3f}. Status: " + ("better than the baseline on this testing sample." if m['beats_baseline'] else "further review required because the baseline performs better or equally well."),
            f"The estimation sample contains {e['n_estimation']} rows and the testing sample contains {e['n_testing']} rows. Split: {e['config']['split']}. CV folds: {e['config']['folds']}. Seed: {e['config']['seed']}. Imputation and scaling were learned within each estimation fold.",
            f"The largest mean absolute SHAP contribution is for {importance.iloc[0]['feature']} ({importance.iloc[0]['mean_abs_shap']:.3f} {e['units']}). This ranks prediction contributions, without assigning a universal positive or negative direction.",
            f"SHAP explains {s['n_explained']} testing observations using {s['background_size']} estimation observations as background. For example row {s['local_row_id']}, baseline {s['baseline_prediction']:.3f} plus all feature contributions equals prediction {s['local_prediction']:.3f}. Maximum numerical reconstruction error: {s['max_additivity_error']:.2e}."]
        b = board[["model", "cv_mse", "cv_rmse", "fold_mse_sd"]].round(4)
        imp = importance.round(4)
        loc = local.round(4)
        md = ["# Two-agent economic prediction report",
              "## Model Evaluation Analyst decision",
              "**Model to use for testing evaluation and SHAP: " + str(d["selected_model"]) + "**",
              *paragraphs,
              "## Candidate-method comparison", _table_md(b),
              "CV RMSE here is sqrt(mean fold MSE). Fold MSE SD is descriptive.",
              "Selected hyperparameters: " + json.dumps(e["selected_parameters"]),
              "## Global SHAP contributions", _table_md(imp),
              "## One observation", _table_md(loc),
              "## Interpretation and review", *["- " + n for n in notes]]
        figures = ["model_comparison", "test_predictions", "shap_global", "shap_beeswarm", "shap_local"]
        md.extend("![" + name.replace("_", " ") + "](" + name + ".png)" for name in figures)
        (out / "final_report.md").write_text("\n\n".join(md), encoding="utf-8")
        esc = lambda value: html.escape(str(value))
        p = lambda text: "<p>" + esc(text) + "</p>"
        def figure(name, caption):
            encoded = base64.b64encode((out / (name + ".png")).read_bytes()).decode()
            return ('<figure><img loading="lazy" alt="' + esc(caption)
                    + '" src="data:image/png;base64,' + encoded + '">'
                    + '<figcaption>' + esc(caption) + '</figcaption></figure>')
        def table(frame):
            return '<div class="table-scroll">' + frame.to_html(index=False, escape=True, border=0) + '</div>'
        def section(title, hint, content):
            return ('<details><summary><strong>' + esc(title) + '</strong><span>'
                    + esc(hint) + '</span></summary><div class="detail-body">'
                    + content + '</div></details>')

        chosen = str(d['selected_model'])
        if len(board) == 1:
            rationale = (f"The Model Evaluation Analyst used {chosen}, the only method you selected. "
                         "Cross-validation tuned its settings; no comparison across methods was performed.")
        else:
            rationale = (f"The Model Evaluation Analyst selected {chosen} from your {len(board)} methods "
                         f"because it achieved the lowest cross-validation MSE ({float(board.iloc[0].cv_mse):.4f}).")
            if int((board.cv_mse == board.iloc[0].cv_mse).sum()) > 1:
                rationale += " Several methods tied exactly; the alphabetical model-name rule determined the choice."
        baseline = float(m['baseline_rmse'])
        if m['beats_baseline'] and baseline > 0:
            reduction = 100 * (1 - float(m['test_rmse']) / baseline)
            performance = (f"Testing RMSE was {m['test_rmse']:.3f} {e['units']}, "
                           f"{reduction:.1f}% lower than the simple mean-prediction benchmark ({baseline:.3f}).")
            status = "Lower testing error than the benchmark"
        else:
            performance = (f"Testing RMSE was {m['test_rmse']:.3f} {e['units']}, "
                           f"compared with {baseline:.3f} for the simple mean-prediction benchmark. "
                           "The selected model did not improve on this benchmark.")
            status = "Review needed: benchmark not improved"
        top_feature = str(importance.iloc[0]['feature'])
        finding = (f"{top_feature} had the largest average absolute SHAP contribution to {chosen}'s "
                   "predictions on the explained observations. This describes prediction contributions, "
                   "not a causal effect or improved accuracy.")
        alerts = []
        if not m['beats_baseline']:
            alerts.append("Review the predictors and study design before relying on this model; it did not beat the benchmark.")
        if d.get('near_tie'):
            alerts.append("The next candidate is within 2% of the lowest CV MSE. Treat the ranking as a close result.")
        alerts.append("The model choice used estimation-sample CV scores. Testing results and SHAP did not determine the winner.")
        if 'simulat' in str(e['dataset']).lower():
            alerts.append("These are simulated data for demonstration, not empirical findings about actual economies.")

        body = ('<header><p class="eyebrow">YOUR ANALYSIS REPORT</p><h1>Results at a glance</h1>'
                + '<p class="dataset">' + esc(e['dataset']) + ' · Outcome: ' + esc(e['target']) + '</p></header>'
                + '<main><section class="overview" aria-labelledby="decision-heading">'
                + '<p class="eyebrow">MODEL EVALUATION ANALYST DECISION</p>'
                + '<h2 id="decision-heading">Selected model: ' + esc(chosen) + '</h2>' + p(rationale)
                + '<div class="takeaway"><h3>How well did it predict?</h3>' + p(performance)
                + '<p class="status">' + esc(status) + '</p></div>'
                + '<div class="takeaway"><h3>What explains its predictions?</h3>' + p(finding) + '</div>'
                + '<div class="review"><h3>What to keep in mind</h3><ul>'
                + ''.join('<li>' + esc(a) + '</li>' for a in alerts) + '</ul></div></section>'
                + '<div class="details-heading"><h2>Explore the details</h2>'
                + '<p>Open only the sections you need. All charts and tables are included in this file.</p></div>'
                + '<nav class="controls" aria-label="Report controls" hidden>'
                + '<button type="button" id="expand">Open all details</button>'
                + '<button type="button" id="collapse">Close all details</button>'
                + '<button type="button" id="print">Print / Save PDF</button></nav>'
                + '<p class="print-note">Printing includes the summary and the sections you have opened.</p>')
        body += section('Why this model?', 'Selection rationale and comparison of your chosen methods',
                        p(d.get('selection_explanation') or selection_summary(board))
                        + table(b) + p('Lower CV MSE is better. CV scores are used for selection and can be optimistic after tuning. Fold MSE SD describes variation, not a confidence interval.')
                        + figure('model_comparison', 'Candidate comparison on the estimation sample: lower CV RMSE is better.'))
        metrics_table = pd.DataFrame({
            'Measure': ['Testing RMSE', 'Testing MAE', 'Testing R-squared', 'Benchmark RMSE'],
            'Value': [f"{m['test_rmse']:.4f}", f"{m['test_mae']:.4f}", f"{m['test_r2']:.4f}", f"{baseline:.4f}"]})
        body += section('How accurate are the predictions?', 'Results on observations held aside for testing',
                        p(performance) + table(metrics_table)
                        + p(f"RMSE and MAE are measured in {e['units']}; smaller values indicate lower prediction error. R-squared is unitless. The benchmark predicts the estimation-sample mean for everyone.")
                        + figure('test_predictions', 'Observed and predicted outcomes in the untouched testing sample.'))
        body += section('Which features influence predictions?', 'Global SHAP importance and the distribution of contributions',
                        p(finding) + figure('shap_global', 'Average absolute SHAP contribution: larger values mean larger prediction contributions.')
                        + figure('shap_beeswarm', 'SHAP contributions across explained observations; contributions can raise or lower a prediction.')
                        + table(imp) + p(f"SHAP explains {s['n_explained']} testing observations using {s['background_size']} estimation observations as background."))
        body += section('Explain one prediction', 'An individual result, broken down by feature',
                        p(f"For observation {s['local_row_id']}, the SHAP baseline is {s['baseline_prediction']:.3f} and the model prediction is {s['local_prediction']:.3f} {e['units']}. Feature contributions bridge that difference.")
                        + figure('shap_local', 'One observation: how feature contributions move the prediction from the SHAP baseline.')
                        + table(loc) + p('Positive contributions raise this prediction relative to the SHAP baseline; negative contributions lower it. Neither sign establishes better fit or causality.'))
        body += section('Data and analysis settings', 'Sample sizes, model settings, and numerical checks',
                        p(paragraphs[4]) + '<h3>Selected model settings</h3><pre>'
                        + esc(json.dumps(e['selected_parameters'], indent=2)) + '</pre>'
                        + '<h3>Run configuration</h3><pre>' + esc(json.dumps(e['config'], indent=2)) + '</pre>'
                        + p(f"Maximum SHAP reconstruction error: {s['max_additivity_error']:.2e}. This checks that the baseline plus contributions reproduces the model prediction."))
        body += section('Limitations and interpretation', 'Assumptions and cautions for using the findings',
                        '<ul>' + ''.join('<li>' + esc(n) + '</li>' for n in notes) + '</ul>')
        body += '<footer>Generated locally by the ML Analyst and Model Evaluation Analyst.</footer></main>'
        style = '''
        :root{color-scheme:light;font:16px/1.65 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:#243340;background:#f5f7fa}
        *{box-sizing:border-box}body{max-width:1040px;margin:0 auto;padding:36px 24px 55px}
        h1,h2,h3{line-height:1.25;color:#152a3b}h1{font-size:clamp(28px,4vw,42px);margin:8px 0 12px}
        h2{font-size:24px;margin:10px 0 16px}h3{font-size:18px;margin:12px 0 6px}p{margin:8px 0 16px}
        .eyebrow{font-size:12px;font-weight:750;letter-spacing:1.4px;color:#486779;margin:0}.dataset{color:#586776}
        .overview{background:white;border:1px solid #d7e1e9;border-top:5px solid #315f79;border-radius:12px;padding:26px;margin:26px 0}
        .takeaway{padding:10px 0;border-top:1px solid #e2e8ef}.status{font-size:14px;font-weight:650}.review{background:#f4f6f9;padding:14px 18px;border-radius:8px}
        li{margin:7px 0}.details-heading{margin-top:32px}.controls{display:flex;flex-wrap:wrap;gap:10px;margin:16px 0}.controls[hidden]{display:none}
        button{font:inherit;background:white;color:#284e66;border:1px solid #aabbc9;border-radius:7px;padding:8px 14px;cursor:pointer}
        button:hover{background:#edf3f7}button:focus-visible,summary:focus-visible{outline:3px solid #4d85ba;outline-offset:3px}
        .print-note,figcaption,footer{font-size:13px;color:#596978}details{margin:14px 0;border:1px solid #d5dfe7;border-radius:10px;background:white}
        summary{cursor:pointer;padding:18px 20px}summary strong{font-size:18px}summary span{display:block;font-size:14px;color:#5c6c79;margin:4px 0 0 20px}
        details[open] summary{border-bottom:1px solid #e0e7ee}.detail-body{padding:20px 24px}.table-scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}
        th,td{text-align:left!important;border-bottom:1px solid #dde5ec;padding:10px}th{background:#f2f5f8}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px;background:#f5f7fa;padding:15px}
        figure{margin:24px 0}img{max-width:100%;height:auto}footer{margin-top:30px} @media(max-width:600px){body{padding:20px 14px}.overview,.detail-body{padding:18px}}
        @media print{body{max-width:none;padding:0;background:white;font-size:11pt}.controls,.print-note{display:none!important}details:not([open]){display:none}details,.overview{border:0}summary{list-style:none;padding:12px 0}.detail-body{padding:0}img{max-height:650px;object-fit:contain}figure,table,.review{break-inside:avoid}h2,h3,summary{break-after:avoid}}
        '''
        script = '''<script>
        document.querySelector('.controls').hidden=false;
        document.getElementById('expand').onclick=()=>document.querySelectorAll('details').forEach(d=>d.open=true);
        document.getElementById('collapse').onclick=()=>document.querySelectorAll('details').forEach(d=>d.open=false);
        document.getElementById('print').onclick=()=>window.print();
        </script>'''
        page = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
                '<meta name="viewport" content="width=device-width,initial-scale=1">'
                '<title>Your analysis report</title><style>' + style + '</style></head><body>'
                + body + script + '</body></html>')
        (out / "final_report.html").write_text(page, encoding="utf-8")
        # Markdown is a full, portable text companion to the interactive HTML.
        summary_md = ['# Analysis summary', '**Selected model: ' + chosen + '**', rationale,
                      performance, finding, *['- ' + a for a in alerts], '---']
        (out / "final_report.md").write_text("\n\n".join(summary_md + md), encoding="utf-8")
