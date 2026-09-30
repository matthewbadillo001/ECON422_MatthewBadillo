import streamlit as st
import pandas as pd
import numpy as np

# ============================================================
# ECON 422: Cross-Validation Click-Through Exercise
# Run from Terminal:
#   python3 -m pip install streamlit pandas numpy
#   python3 -m streamlit run \(Sep2\)\ CrossValidation_illust.py
# ============================================================

st.set_page_config(
    page_title="ECON 422: Cross-Validation Exercise",
    layout="wide"
)

# ------------------------------------------------------------
# Toy data from the classroom illustration
# ------------------------------------------------------------
y = np.array([1, 2, 10, 1, 2, 3], dtype=float)
test_y = np.array([1.5, 2.5, 3.5], dtype=float)
fold = np.array([1, 1, 1, 2, 2, 2])

data = pd.DataFrame({
    "i": np.arange(1, 7),
    "y": y,
    "Fold": fold
})

# Candidate hyperparameters:
# a = 1 -> LAD / median
# a = 2 -> OLS / mean
candidates = {
    1: "LAD / median",
    2: "OLS / mean"
}

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------
def fit_theta(a, values):
    if a == 1:
        return float(np.median(values))
    if a == 2:
        return float(np.mean(values))


def mse(values, theta):
    return float(np.mean((values - theta) ** 2))


# Fold 1 is validation, Fold 2 is training
train_fold2 = y[fold == 2]
valid_fold1 = y[fold == 1]

theta_f1 = {a: fit_theta(a, train_fold2) for a in candidates}
loss_f1 = {a: mse(valid_fold1, theta_f1[a]) for a in candidates}

# Fold 2 is validation, Fold 1 is training
train_fold1 = y[fold == 1]
valid_fold2 = y[fold == 2]

theta_f2 = {a: fit_theta(a, train_fold1) for a in candidates}
loss_f2 = {a: mse(valid_fold2, theta_f2[a]) for a in candidates}

cv_loss = {
    a: (loss_f1[a] + loss_f2[a]) / 2
    for a in candidates
}

selected_a = min(cv_loss, key=cv_loss.get)
final_theta = fit_theta(selected_a, y)
test_mse = mse(test_y, final_theta)

# ------------------------------------------------------------
# Step descriptions
# ------------------------------------------------------------
actions = {
    1: "Split the full data into an estimation sample and a testing sample.",
    2: "Split the estimation sample into two folds.",
    3: "Fold 1 becomes validation; Fold 2 becomes training.",
    4: "Fit each candidate hyperparameter using Fold 2.",
    5: "Evaluate both fitted models on Fold 1.",
    6: "Switch roles: Fold 2 becomes validation; Fold 1 becomes training.",
    7: "Fit on Fold 1 and evaluate both candidates on Fold 2.",
    8: "Average validation losses and select the best hyperparameter.",
    9: "Refit the selected model on the full estimation sample.",
    10: "Evaluate the final model on the testing sample."
}


def role_for_observation(current_step, obs_fold):
    if current_step == 1:
        return "Estimation sample"

    if current_step == 2:
        return f"Fold {obs_fold}"

    if 3 <= current_step <= 5:
        return "Validation" if obs_fold == 1 else "Training"

    if 6 <= current_step <= 7:
        return "Validation" if obs_fold == 2 else "Training"

    if current_step == 8:
        return "Used in CV summary"

    if current_step == 9:
        return "Final refit"

    return "Final refit"


def card(label, value, role):
    styles = {
        "Training": ("#e8f4ea", "#1d5e2a"),
        "Validation": ("#fff1d6", "#7a4d00"),
        "Estimation sample": ("#eef2f7", "#263849"),
        "Fold 1": ("#eef2f7", "#263849"),
        "Fold 2": ("#eef2f7", "#263849"),
        "Used in CV summary": ("#e9ecff", "#303b83"),
        "Final refit": ("#e9ecff", "#303b83"),
        "Held-out test": ("#f4f4f4", "#555555"),
        "Testing sample": ("#eef2f7", "#263849"),
    }

    bg, fg = styles.get(role, ("#f4f4f4", "#333333"))

    return f"""
    <div style="
        border:1px solid #d9d9d9;
        border-radius:10px;
        padding:12px 8px;
        text-align:center;
        background:{bg};
        color:{fg};
        min-height:120px;">
        <div style="font-size:14px;"><b>{label}</b></div>
        <div style="font-size:28px; margin:7px 0;">{value:g}</div>
        <div style="font-size:14px;"><b>{role}</b></div>
    </div>
    """


# ------------------------------------------------------------
# Session state
# ------------------------------------------------------------
if "step" not in st.session_state:
    st.session_state.step = 1

# ------------------------------------------------------------
# Title
# ------------------------------------------------------------
st.title("ECON 422: Macroeconomics & Machine Learning")
st.header("Cross-Validation: Click-Through Exercise")

with st.expander("Setup", expanded=False):

    st.markdown(
        r"""
Suppose we want to predict $y$ using a **constant prediction**
$\widehat y=\theta$. Before estimating the final $\theta$, we must choose
the hyperparameter $a$, which determines the loss function used to fit
$\theta$.
"""
    )

    st.markdown("**Training loss:**")

    st.latex(
        r"L_a(\theta)=\frac{1}{n}\sum_{i=1}^{n}|y_i-\theta|^a,\qquad a\in\{1,2\}"
    )

    st.markdown(
        r"""
- **$a=1$: absolute-error loss (LAD)** $\Rightarrow$ the fitted
  $\widehat\theta$ is the sample **median**.
- **$a=2$: squared-error loss (OLS)** $\Rightarrow$ the fitted
  $\widehat\theta$ is the sample **mean**.
"""
    )

    st.markdown("**Cross-validation loss:**")

    st.latex(
        r"\operatorname{MSE}_{\mathrm{val}}=\frac{1}{n_{\mathrm{val}}}"
        r"\sum_{i\in\mathcal I_{\mathrm{val}}}"
        r"\left(y_i-\widehat{\theta}_{a,-k}\right)^2"
    )

    st.markdown(
        r"""
We begin by splitting the available observations into two separate parts:

- an **estimation sample**, used for model fitting and cross-validation;
- a **testing sample**, set aside and not used during model selection.

Within the estimation sample, we use **2-fold cross-validation** to choose
between $a=1$ and $a=2$. After choosing $a$, we refit the selected model on
the full estimation sample. Only then do we evaluate the final fitted model
on the testing sample.
"""
    )

    st.markdown("**Toy samples used in this exercise:**")
    st.latex(r"y_{\mathrm{est}}=(1,2,10,1,2,3)")
    st.latex(r"y_{\mathrm{test}}=(1.5,2.5,3.5)")

    st.info(
        "Goal: use only the estimation sample for model selection and refitting. "
        "Keep the testing sample separate until the final evaluation."
    )

st.write(
    "Use the step navigator on the right. The training/validation roles, fitted "
    "parameter, validation losses, selected hyperparameter, final refit, and testing evaluation are "
    "revealed progressively."
)

step = st.session_state.step

# ------------------------------------------------------------
# Two-column classroom layout
# ------------------------------------------------------------
main_col, nav_col = st.columns([4.8, 1.35], gap="large")

with nav_col:
    st.markdown("### Steps")

    for s in range(1, 11):
        if st.button(
            f"Step {s}",
            type="primary" if step == s else "secondary",
            use_container_width=True,
            key=f"step_{s}"
        ):
            st.session_state.step = s
            st.rerun()

    st.markdown("---")

    if st.button(
        "← Previous",
        use_container_width=True,
        disabled=(step == 1),
        key="previous_step"
    ):
        st.session_state.step = max(1, step - 1)
        st.rerun()

    if st.button(
        "Next →",
        use_container_width=True,
        disabled=(step == 10),
        key="next_step"
    ):
        st.session_state.step = min(10, step + 1)
        st.rerun()

    st.caption(f"Current: Step {step}")

with main_col:
    st.subheader(f"Step {step}: {actions[step]}")

    # ------------------------------------------------------------
    # Observation cards
    # ------------------------------------------------------------
    if step <= 9:
        st.markdown("### Estimation Sample")

        cols = st.columns(6)

        for j, row in data.iterrows():
            role = role_for_observation(step, int(row["Fold"]))

            if step == 1:
                obs_label = f"Observation {int(row['i'])}"
                role = "Estimation sample"
            else:
                obs_label = f"Observation {int(row['i'])} · Fold {int(row['Fold'])}"

            with cols[j]:
                st.markdown(
                    card(
                        label=obs_label,
                        value=row["y"],
                        role=role
                    ),
                    unsafe_allow_html=True
                )

        if step == 1:
            st.markdown("### Testing Sample")

            test_cols = st.columns(3)
            for j, value in enumerate(test_y):
                with test_cols[j]:
                    st.markdown(
                        card(
                            label=f"Test Observation {j + 1}",
                            value=value,
                            role="Held-out test"
                        ),
                        unsafe_allow_html=True
                    )

        elif 2 <= step <= 9:
            st.caption(
                "Testing sample: held out from Step 1 and untouched during Steps 2–9."
            )

    else:
        st.markdown("### Testing Sample")

        test_cols = st.columns(3)
        for j, value in enumerate(test_y):
            with test_cols[j]:
                st.markdown(
                    card(
                        label=f"Test Observation {j + 1}",
                        value=value,
                        role="Testing sample"
                    ),
                    unsafe_allow_html=True
                )

    # ------------------------------------------------------------
    # Progressive explanation
    # ------------------------------------------------------------
    st.markdown("---")

    if step == 1:
        st.markdown("### Split into estimation and testing samples")

        st.markdown(
            "Before cross-validation begins, separate the available observations "
            "into an **estimation sample** and a **testing sample**."
        )

        sample_left, sample_right = st.columns(2)

        with sample_left:
            st.markdown("#### Estimation sample")
            st.latex(r"y_{\mathrm{est}}=(1,2,10,1,2,3)")
            st.success("Used in Steps 2–9")

        with sample_right:
            st.markdown("#### Testing sample")
            st.latex(r"y_{\mathrm{test}}=(1.5,2.5,3.5)")
            st.warning("Set aside until Step 10")

        st.markdown(
            "The testing sample does **not** participate in fold construction, "
            "hyperparameter selection, or the final refit."
        )

    elif step == 2:
        st.markdown("### Split the estimation sample into two folds")

        st.latex(
            r"\text{Fold 1}=(1,2,10)\qquad \text{Fold 2}=(1,2,3)"
        )

        st.markdown(
            """
    The folds will alternate between training and validation.
    """
        )

    elif step == 3:
        st.markdown("### First cross-validation split")

        st.latex(
            r"\underbrace{\text{Fold 2}}_{\text{training}}"
            r"\;\longrightarrow\;\text{fit model}\;\longrightarrow\;"
            r"\underbrace{\text{Fold 1}}_{\text{validation}}"
        )

    elif step == 4:
        st.markdown("### Fit both candidate hyperparameters on Fold 2")

        st.markdown("**Training data:**")
        st.latex(r"\mathcal I_{\mathrm{train}}=\{1,2,3\}")

        with st.expander("Candidate 1: a = 1 — LAD / median", expanded=False):

            st.latex(
                r"\widehat{\theta}_{1,-1}"
                r"=\arg\min_{\theta}\frac{1}{3}"
                r"\sum_{y_i\in\{1,2,3\}}|y_i-\theta|"
            )

            st.markdown("**Why does minimizing absolute deviations give the median?**")

            st.markdown(
                r"""
    Consider the general absolute-deviation objective

    $$
    Q(\theta)=\sum_{i=1}^{n}|y_i-\theta|.
    $$

    If we move $\theta$ slightly to the right:

    - the distance to every observation **below** $\theta$ increases;
    - the distance to every observation **above** $\theta$ decreases.

    So, if more than half of the observations are above $\theta$, moving right
    reduces the total absolute distance. If more than half are below $\theta$,
    moving left reduces it.

    The total distance stops decreasing when at least half of the observations are
    on each side. That point is a **median**.
    """
            )

            st.latex(
                r"\theta<\operatorname{median}(y)"
                r"\quad\Rightarrow\quad"
                r"\text{moving right lowers }Q(\theta)"
            )

            st.latex(
                r"\theta>\operatorname{median}(y)"
                r"\quad\Rightarrow\quad"
                r"\text{moving left lowers }Q(\theta)"
            )

            st.markdown(
                r"""
    For an odd number of observations, the minimizer is the middle observation.
    For an even number, every value between the two middle observations minimizes
    the sum of absolute deviations; the usual sample median is one such minimizer.
    """
            )

            st.markdown("For the training sample $(1,2,3)$:")

            st.latex(r"Q(1)=|1-1|+|2-1|+|3-1|=3")
            st.latex(r"Q(2)=|1-2|+|2-2|+|3-2|=2")
            st.latex(r"Q(3)=|1-3|+|2-3|+|3-3|=3")

            st.latex(
                r"\widehat{\theta}_{1,-1}"
                r"=\operatorname{median}(1,2,3)=2"
            )

        with st.expander("Candidate 2: a = 2 — OLS / mean", expanded=False):

            st.latex(
                r"\widehat{\theta}_{2,-1}"
                r"=\arg\min_{\theta}\frac{1}{3}"
                r"\sum_{y_i\in\{1,2,3\}}(y_i-\theta)^2"
            )

            st.markdown("**Why does minimizing squared deviations give the mean?**")

            st.markdown(
                r"""
    Consider the squared-error objective

    $$
    Q(\theta)=\sum_{i=1}^{n}(y_i-\theta)^2.
    $$

    Differentiate with respect to $\theta$:
    """
            )

            st.latex(
                r"\frac{dQ(\theta)}{d\theta}"
                r"=-2\sum_{i=1}^{n}(y_i-\theta)"
            )

            st.markdown("At the minimum, the first-order condition is")

            st.latex(
                r"-2\sum_{i=1}^{n}(y_i-\theta)=0"
            )

            st.latex(
                r"\sum_{i=1}^{n}y_i-n\theta=0"
            )

            st.latex(
                r"\theta=\frac{1}{n}\sum_{i=1}^{n}y_i=\bar y"
            )

            st.markdown(
                r"""
    The second derivative is positive,
    """
            )

            st.latex(
                r"\frac{d^2Q(\theta)}{d\theta^2}=2n>0"
            )

            st.markdown(
                r"""
    so this stationary point is the minimum. Therefore, the constant that minimizes
    the sum of squared deviations is the **sample mean**.
    """
            )

            st.markdown("For the training sample $(1,2,3)$:")

            st.latex(
                r"\widehat{\theta}_{2,-1}"
                r"=\frac{1+2+3}{3}=2"
            )

        st.info(
            "Both candidate loss functions happen to produce the same fitted "
            "value in this fold: theta-hat = 2."
        )

        fit_table = pd.DataFrame({
            "a": [1, 2],
            "Estimator": [candidates[1], candidates[2]],
            "Training data": ["(1, 2, 3)", "(1, 2, 3)"],
            "Fitted theta": [theta_f1[1], theta_f1[2]]
        })

        st.dataframe(fit_table, hide_index=True, use_container_width=True)

    elif step == 5:
        st.markdown("### Validate the fitted models on Fold 1")

        with st.expander("Show procedure", expanded=False):

            st.markdown("**Validation data:**")
            st.latex(r"\mathcal I_{\mathrm{val}}=\{1,2,10\}")

            st.markdown("**Recall from Step 4:** both candidates were fitted on Fold 2 and gave")

            st.latex(
                r"\widehat{\theta}_{1,-1}=2"
                r"\qquad"
                r"\widehat{\theta}_{2,-1}=2"
            )

            st.markdown(
                "Step 5 does **not** fit the models again. It takes the fitted "
                "values from Step 4 and evaluates them on the held-out Fold 1."
            )

            st.markdown("**Compute the held-out MSE:**")

            st.latex(
                r"\operatorname{MSE}_{\mathrm{Fold\,1}}"
                r"=\frac{(1-2)^2+(2-2)^2+(10-2)^2}{3}"
            )

            st.latex(
                r"\operatorname{MSE}_{\mathrm{Fold\,1}}"
                r"=\frac{1+0+64}{3}"
                r"=\frac{65}{3}"
                r"\approx 21.667"
            )

            st.info(
                "Because both candidates produced the same fitted theta in Step 4, "
                "they also receive the same validation loss on Fold 1."
            )

        loss_table = pd.DataFrame({
            "a": [1, 2],
            "Estimator": [candidates[1], candidates[2]],
            "Fitted theta": [theta_f1[1], theta_f1[2]],
            "Validation data": ["(1, 2, 10)", "(1, 2, 10)"],
            "Fold 1 validation MSE": [loss_f1[1], loss_f1[2]]
        })

        st.dataframe(
            loss_table.style.format({
                "Fitted theta": "{:.3f}",
                "Fold 1 validation MSE": "{:.3f}"
            }),
            hide_index=True,
            use_container_width=True
        )

    elif step == 6:
        st.markdown("### Switch the roles of the folds")

        st.latex(
            r"\underbrace{\text{Fold 1}}_{\text{training}}"
            r"\;\longrightarrow\;\text{fit model}\;\longrightarrow\;"
            r"\underbrace{\text{Fold 2}}_{\text{validation}}"
        )

        st.markdown(
            """
    Every fold should serve as validation.
    """
        )

    elif step == 7:
        st.markdown("### Fit on Fold 1 and validate on Fold 2")

        st.markdown("**Recall from Step 6:** the fold roles were switched.")

        st.latex(
            r"\underbrace{\text{Fold 1}}_{\text{training}:\,(1,2,10)}"
            r"\qquad"
            r"\underbrace{\text{Fold 2}}_{\text{validation}:\,(1,2,3)}"
        )

        st.markdown("**Training data:**")
        st.latex(r"\mathcal I_{\mathrm{train}}=\{1,2,10\}")

        with st.expander("Candidate 1: a = 1 — LAD / median", expanded=False):

            st.markdown("Because $a=1$, fit the constant by minimizing absolute deviations.")

            st.latex(
                r"\widehat{\theta}_{1,-2}"
                r"=\operatorname{median}(1,2,10)=2"
            )

            st.markdown("Now validate on Fold 2:")
            st.latex(r"\mathcal I_{\mathrm{val}}=\{1,2,3\}")

            st.latex(
                r"\operatorname{MSE}_{2}(a=1)"
                r"=\frac{(1-2)^2+(2-2)^2+(3-2)^2}{3}"
                r"=\frac{2}{3}"
                r"\approx 0.667"
            )

        with st.expander("Candidate 2: a = 2 — OLS / mean", expanded=False):

            st.markdown("Because $a=2$, fit the constant by minimizing squared deviations.")

            st.latex(
                r"\widehat{\theta}_{2,-2}"
                r"=\operatorname{mean}(1,2,10)"
                r"=\frac{1+2+10}{3}"
                r"=\frac{13}{3}"
                r"\approx 4.333"
            )

            st.markdown("Now validate on Fold 2:")
            st.latex(r"\mathcal I_{\mathrm{val}}=\{1,2,3\}")

            st.latex(
                r"\operatorname{MSE}_{2}(a=2)"
                r"=\frac{"
                r"(1-\frac{13}{3})^2"
                r"+(2-\frac{13}{3})^2"
                r"+(3-\frac{13}{3})^2"
                r"}{3}"
            )

            st.latex(
                r"\operatorname{MSE}_{2}(a=2)"
                r"=\frac{55}{9}"
                r"\approx 6.111"
            )

        split2_table = pd.DataFrame({
            "a": [1, 2],
            "Estimator": [candidates[1], candidates[2]],
            "Fitted theta": [theta_f2[1], theta_f2[2]],
            "Fold 2 validation MSE": [loss_f2[1], loss_f2[2]]
        })

        st.dataframe(
            split2_table.style.format({
                "Fitted theta": "{:.3f}",
                "Fold 2 validation MSE": "{:.3f}"
            }),
            hide_index=True,
            use_container_width=True
        )

    elif step == 8:
        st.markdown("### Average the validation losses")

        with st.expander("Show procedure", expanded=False):

            st.markdown(
                "For each candidate hyperparameter, average its held-out losses "
                "across the two folds."
            )

            st.markdown("#### Where did the losses come from?")

            st.markdown(
                r"**Step 5:** Validate on Fold 1. Both candidates used "
                r"$\widehat{\theta}=2$, giving"
            )

            st.latex(
                r"\operatorname{MSE}_{\mathrm{Fold\,1}}"
                r"=\frac{(1-2)^2+(2-2)^2+(10-2)^2}{3}"
                r"=\frac{65}{3}"
            )

            st.markdown(
                r"**Step 7, Candidate 1:** fit $a=1$ on Fold 1, obtaining "
                r"$\widehat{\theta}_{1,-2}=2$, and validate on Fold 2:"
            )

            st.latex(
                r"\operatorname{MSE}_{\mathrm{Fold\,2}}(a=1)"
                r"=\frac{(1-2)^2+(2-2)^2+(3-2)^2}{3}"
                r"=\frac{2}{3}"
            )

            st.markdown(
                r"**Step 7, Candidate 2:** fit $a=2$ on Fold 1, obtaining "
                r"$\widehat{\theta}_{2,-2}=13/3$, and validate on Fold 2:"
            )

            st.latex(
                r"\operatorname{MSE}_{\mathrm{Fold\,2}}(a=2)"
                r"=\frac{55}{9}"
            )

            st.markdown("---")
            st.markdown("#### Now average the two fold losses")

            st.markdown("#### Candidate $a=1$")

            st.markdown(
                r"$65/3$ is from **Step 5** and $2/3$ is from **Step 7**."
            )

            st.latex(
                r"\operatorname{CV}(1)"
                r"=\frac{1}{2}"
                r"\left("
                r"\underbrace{\frac{65}{3}}_{\text{Step 5}}"
                r"+"
                r"\underbrace{\frac{2}{3}}_{\text{Step 7}}"
                r"\right)"
            )

            st.latex(
                r"\operatorname{CV}(1)"
                r"=\frac{67}{6}"
                r"\approx 11.167"
            )

            st.markdown("#### Candidate $a=2$")

            st.markdown(
                r"$65/3$ is from **Step 5** and $55/9$ is from **Step 7**."
            )

            st.latex(
                r"\operatorname{CV}(2)"
                r"=\frac{1}{2}"
                r"\left("
                r"\underbrace{\frac{65}{3}}_{\text{Step 5}}"
                r"+"
                r"\underbrace{\frac{55}{9}}_{\text{Step 7}}"
                r"\right)"
            )

            st.latex(
                r"\operatorname{CV}(2)"
                r"=\frac{125}{9}"
                r"\approx 13.889"
            )

            st.markdown("**Choose the candidate with the smaller CV loss:**")

            st.latex(
                r"\widehat a"
                r"=\arg\min_{a\in\{1,2\}}\operatorname{CV}(a)"
                r"=1"
            )

        summary = pd.DataFrame({
            "a": [1, 2],
            "Estimator": [candidates[1], candidates[2]],
            "Loss: Fold 1": [loss_f1[1], loss_f1[2]],
            "Loss: Fold 2": [loss_f2[1], loss_f2[2]],
            "CV(a)": [cv_loss[1], cv_loss[2]]
        })

        st.dataframe(
            summary.style.format({
                "Loss: Fold 1": "{:.3f}",
                "Loss: Fold 2": "{:.3f}",
                "CV(a)": "{:.3f}"
            }),
            hide_index=True,
            use_container_width=True
        )

        st.success(
            f"Selected hyperparameter: a = {selected_a} "
            f"({candidates[selected_a]})"
        )

    elif step == 9:
        st.markdown("### Refit on the full estimation sample")

        with st.expander("Show procedure", expanded=False):

            st.markdown("**Recall from Step 8:**")

            st.latex(
                r"\operatorname{CV}(1)=\frac{67}{6}"
                r"\;<\;"
                r"\operatorname{CV}(2)=\frac{125}{9}"
            )

            st.markdown("Therefore Step 8 selected")
            st.latex(r"\widehat a=1")

            st.markdown("So we return to the **full estimation sample**:")
            st.latex(r"y=(1,2,10,1,2,3)")

            st.markdown("Because $a=1$, the final estimator minimizes absolute loss.")

            st.latex(
                r"\widehat{\theta}_{\mathrm{final}}"
                r"=\arg\min_{\theta}"
                r"\sum_{i=1}^{6}|y_i-\theta|"
            )

            st.markdown(
                "From Step 4, minimizing the sum of absolute deviations gives a median."
            )

            st.markdown("Sort the observations:")
            st.latex(r"(1,1,2,2,3,10)")

            st.latex(
                r"\widehat{\theta}_{\mathrm{final}}"
                r"=\operatorname{median}(1,1,2,2,3,10)"
                r"=\frac{2+2}{2}"
                r"=2"
            )

        col1, col2 = st.columns(2)

        col1.metric("Selected a", selected_a)
        col2.metric("Final fitted θ", f"{final_theta:.3f}")

    elif step == 10:
        st.markdown("### Evaluate on the testing sample")

        st.markdown(
            "Now bring back the testing observations that were separated in Step 1."
        )

        st.markdown("**Recall from Step 9:**")
        st.latex(r"\widehat{\theta}_{\mathrm{final}}=2")

        st.markdown("**Testing sample from Step 1:**")
        st.latex(r"y_{\mathrm{test}}=(1.5,2.5,3.5)")

        with st.expander("Show procedure", expanded=False):

            st.markdown(
                "The final model is a constant-prediction model, so every "
                "testing observation receives the prediction"
            )

            st.latex(
                r"\widehat y"
                r"=\widehat{\theta}_{\mathrm{final}}"
                r"=2"
            )

            st.markdown("Compute the testing-sample MSE:")

            st.latex(
                r"\operatorname{MSE}_{\mathrm{test}}"
                r"=\frac{(1.5-2)^2+(2.5-2)^2+(3.5-2)^2}{3}"
            )

            st.latex(
                r"\operatorname{MSE}_{\mathrm{test}}"
                r"=\frac{0.25+0.25+2.25}{3}"
                r"=\frac{2.75}{3}"
                r"\approx 0.917"
            )

        test_df = pd.DataFrame({
            "Test y": test_y,
            "Prediction": np.repeat(final_theta, len(test_y))
        })

        st.dataframe(test_df, hide_index=True, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Selected a", selected_a)
        c2.metric("Final θ", f"{final_theta:.3f}")
        c3.metric("Test MSE", f"{test_mse:.3f}")

        st.success(
            "The testing sample was not used anywhere in Steps 2–9. "
            "It is used only here to evaluate the final fitted model."
        )

