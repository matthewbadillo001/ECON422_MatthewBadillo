# Macroeconomic Analysis Studio with LLM prompt writing

**Open a local page, write your question, and watch three Llama agents work.** Version 0.15.0 takes a contextual research prompt, produces a small executable Python program, runs it, and shows the review exchange. The analyst and critic review the program before estimation. When a validated time-series DML screen finds a concern or is inconclusive, Python publishes the diagnostic report and the critic can comment on it.

Hover over **LOCAL ECONOMETRICS LAB** at the top of the studio to reveal the optional name box. Click in the box and type, or tap the heading to start editing. Keyboard users can Tab to the heading and then to the box. Enter or Escape closes the editor. It updates as you type and survives a refresh in the same tab. Clear the field to restore **LOCAL ECONOMETRICS LAB**.

## Start on a Mac

Install Python 3.10 or newer. For optional local Llama review, install [Ollama for macOS](https://ollama.com/download/mac). Save **Macroeconomic analysis studio with llm prompt writing.zip** in Downloads. If an earlier studio is running, press Control+C in its terminal. Paste this block into Terminal:

```bash
cd "$HOME/Downloads" && \
unzip -oq "Macroeconomic analysis studio with llm prompt writing.zip" -d studio-workspaces && \
bash "studio-workspaces/Macroeconomic analysis studio with llm prompt writing/run_studio.sh"
```

The launcher installs the Python dependencies on first use and opens the studio. Selecting a worked example fills the five fields and selects its included dataset. Click the highlighted Python Run button to calculate the results. To use the local analyst and critic on your own prompts, open Ollama and run `ollama pull llama3.1:8b` once. Use the new browser tab opened by the launcher. The **Question and context** area has five editable boxes. Keep Terminal open while using the app and press Control+C to stop.

For subsequent sessions, run this single command:

```bash
bash "$HOME/Downloads/studio-workspaces/Macroeconomic analysis studio with llm prompt writing/run_studio.sh"
```

## Find the analysis code

The studio opens in **Forecasting → Prompt writing**. Opening **Estimation** also starts with five empty fields.

| Workspace | Start here |
|---|---|
| Forecasting | Choose **Prompt writing** to write a forecast or exploration. Browse Hybrid: classical + ML, ML only, Classical only, or Explore & compare for worked examples. |
| Estimation | Choose **Prompt writing** to write an estimation question. Expand **Try a writing exercise** for an incomplete or rough draft about training hours and earnings. Regression, Cross-sectional DML, and Time-series DML have worked examples. |

Write in Clarity, Structure, Context, Role and Examples. **Review my prompt** gives writing feedback. Choose the data and click **Run with local Llama** to analyze the question. Choosing **Prompt writing** or switching workspaces clears the form and its data selection. Browsing methods keeps your own wording. Selecting a worked card prepares the prompt and data. Click its Python Run button for numerical results, or choose **Run with local Llama** for agent review. Selecting another example clears the previous results and review. Writing exercises load drafts for practice. If you explicitly select FRED, import your data before running the external example.

| Method or component | File |
|---|---|
| Cross-sectional and monthly time-series DML | `dml.py` |
| Three-condition reverse-cross-fitting screen | `rcf_screen.py` |
| Series exploration and model choice | `exploration.py` |
| Classical forecasting, direct ML, and additive classical + ML hybrids | `hybrid_forecasting.py` |
| Super Learner weights, base fits and evaluation | `superlearner/ensemble.py`, `superlearner/forecasting.py` |
| Hybrid, ML-only, classical-only, and comparison prompts | `studio_core/family_prompts.py` |
| Other prompts, local interface and analysis routing | `studio_core/example_prompt.py`, `studio_core/studio.html`, `studio_core/studio.py` |

## Hybrid forecasting for ECON 422

Under **Forecasting → Hybrid: classical + ML**, select **Build a hybrid** to choose both components. The three worked cards provide starting points. Select **Use these forecast settings** to fill the five prompt fields. Click **Run hybrid forecast (Python only)** to calculate the results. To compare the same two components against each other and their hybrid, use **Explore & compare → Compare forecast families** and change its forecast settings.

| Classical baseline | ML learner |
|---|---|
| Last month, same month last year, 3-month moving average, historical mean, exponential smoothing (α = 0.3), linear trend, trend with calendar months, MA(1), or AR(1) through AR(12) | Random forest, LASSO, ridge regression, elastic net, or gradient boosting |

**Classical only → MA(1)** forecasts next month as the fitted mean plus a coefficient times this month’s estimated shock (actual minus its one-step forecast). The separate **3-month moving average** averages three observed values. The MA(1) fit minimizes squared one-step errors over coefficients from −0.99 through 0.99 in steps of 0.01, fitting the mean for each coefficient. The initial shock and unobserved shocks after a calendar gap are set to zero. Validation and holdout use coefficients fitted before each block and update the shock as each new observation arrives. The hybrid learns from historical errors calculated with the data available at each forecast origin.

Any baseline can be paired with any learner. For example, **AR(3) + gradient boosting** fits the baseline using the previous three months and trains the learner on earlier one-step errors. The correction uses observed target lags 1, 2, 3, and 12 plus sine and cosine of the calendar month. The report shows the next baseline forecast, the signed correction, and their sum.

You can run offline with the included U.S. retail sales or simulated activity series, upload a monthly CSV with a date and numeric target, or select a monthly FRED series. Edited prompt wording can be run with local Llama.

**ML only → Choose an ML model** uses the selected learner directly on the series. **Classical only → Choose a classical model** uses the selected baseline. Under **Explore & compare**, **Compare forecast families** evaluates a matched classical baseline, a direct ML learner, and their additive hybrid. It selects among those three using the 24 validation months before the holdout. Each procedure is fitted before each evaluation block and uses updated lag information for one-step predictions within it. The coefficients and learners stay fixed within that block. The selected family is fixed before holdout scoring. All forecast families use the same scoring dates, units, and next-month target, and refit at the end of the sample. The older **Explore a time series** exercise still compares its five stated classical models.

## Try it

Choose **Forecasting → Prompt writing** or **Estimation → Prompt writing**, write in the five fields, select the data, and click **Run with local Llama**. You can use **Review my prompt** first for writing feedback. For a worked example, select **Forecasting → ML only → Forecast example** or **Estimation → Regression → Estimation example**. Click **Run with local Llama** for edited prompts. A monthly CSV needs a date column and numeric analysis variables.

| Write a prompt like this | Final result |
| --- | --- |
| Forecast January 2025 U.S. retail sales using monthly lags 1, 2, 3 and 12 and annual seasonality. Combine LASSO and random forest and evaluate the last 24 months. | Next-month prediction, development weights, holdout errors and actual-versus-predicted values in the downloadable results. |
| Estimate the association between last month's fuel_inflation and this month's fare_inflation. Control for fare_inflation lags 1–3 and annual seasonality. Give the fuel coefficient and a 95% interval with a HAC bandwidth of three months. | Adjusted OLS coefficient, HAC standard error and two-sided 95% interval, plus a coefficient table. |

Tell the agents what question you want answered, what each series measures, which lags would have been observed and how you intend to use the result. This lab supports monthly forecasts, adjusted OLS associations, and partially linear DML assessment and estimation for both independent cross-sectional units and monthly time series.

For a first live run, select **ML only → Forecast example**, leave the CSV field blank, and click **Run with local Llama**. For a comparison, **Run reference forecast (Python only)** fits the model immediately on the same included data without calling the agents. The five fields fully specify the January 2025 forecast, so the analyst proceeds to code generation. For the unchanged example the analyst explains the validated code and both roles review the results. Edited prompts follow the general analysis path. If the analyst requests clarification, the critic first checks whether the prompt and package defaults already resolve the issue.

To reproduce the numerical example independently of the local model, run `.venv/bin/python -m studio_core demo` from the extracted folder. This runs the retail sales reference forecast on the included snapshot.

## Explore a monthly time series

Click **Explore & compare → Explore a time series → Run exploration (Python only)**. The five prompt boxes also work with **Run with local Llama**. The report measures coverage, a linear trend in target units per month, the detrended month-of-year pattern, lag-1 and lag-12 level correlations, and volatility of monthly changes. It estimates these features on the initial training period.

Five candidates forecast the same subsequent validation months: last observation, same month last year, AR(1), linear trend with calendar-month effects, and the same model plus lag 1. Regressions refit using only past observations at each origin. The model with the lowest rolling RMSE is recommended. With too little common evidence the report gives no ranking. The final 24 months are excluded from feature measurement and selection. After the choice is locked, that model is scored on the holdout using expanding-window refits, and forecasts the next month. The separate **Forecast example** evaluates its stated Super Learner on the same holdout, with its base fits frozen during holdout scoring. Compare those two forecasting procedures on common dates.

To reuse the exploration prompt, expand **Adapt the exploration template to a monthly CSV**, enter the target column and units, apply, and upload a CSV with a date column. Use the Python-only button for those generated fields, or edit the five boxes and run with Llama. The guide includes a fill-in version. The example snapshot is in `studio_core/data/monthly_retail_sales.csv`. Its 240 observations are from the Census Bureau's unadjusted nominal retail trade series via FRED, updated October 15, 2026. This is a retrospective exercise with revised historical figures.

## DML for independent units and monthly time series

Open **Estimation** and choose **Cross-sectional DML estimate** or **Time-series DML estimate**, then click **Run DML estimate (Python only)** to see the results. Both bundled examples run locally without an API key. Each also works with **Run with local Llama**. For assessment alone, choose the corresponding **DML assessment** example. A coefficient is released only when the applicable numerical checks allow estimation. The assumptions behind causal interpretation need separate substantive justification.

| | Cross-sectional | Monthly time series |
| --- | --- | --- |
| Data | One row per independent unit. Supply numeric Y, D and X; a date column is optional and ignored unless explicitly selected. | One observation per calendar month with a date. Select a lagged D, the outcome Y, and other observed lags as X. |
| Worked example | 800 simulated households. Adjust earnings for standardized education and experience to study one additional training hour. The known slope is 1.5. | 480 simulated months of stationary Gaussian AR(1) activity, with true persistence 0.4. Estimate the coefficient on lag 1, controlling for lag 2. Choose FRED to switch to the airfare and lagged fuel example. |
| Assessment | Independent sampling and pre-exposure controls require research-design justification. Inspect residual exposure variation, held-out nuisance errors and binary exposure overlap when relevant. | Assess stationarity/weak dependence, reversibility and conditional stability. A concern or inconclusive result on any row withholds the RCF estimate and interval. |
| Estimation | Estimate E[Y given X] and E[D given X] out of fold, then regress residual Y on residual D without an intercept. Pool the score across five shuffled folds (stratified for binary D). Report an independent-unit heteroskedasticity-robust SE. | Use eight contiguous folds in the longest complete span. Past training blocks run forward. Future training blocks run backward. Rebuild all lags inside each block. Average the eight fold-specific residual slopes. Use Bartlett HAC on the calendar-time influence contributions with the requested bandwidth. |

Both estimators fit separate, fixed random forests to Y and D on each training split (80 trees, minimum leaf size 6, seed 422). With no controls they use training means. The cross-sectional path drops rows missing selected analysis variables and reports the resulting count. Its teaching screens withhold estimation for fewer than max(60, 3 × (number of controls + 2)) complete units, insufficient residual variation, fewer than ten units in either binary exposure group, or more than 10% of out-of-fold propensities outside [0.05, 0.95]. Held-out RMSE measures predictive error and cannot establish nuisance convergence rates or rule out confounding. The interval is a nominal pointwise large-sample interval. For heterogeneous binary treatment effects the partially linear slope need not equal the ATE or ATT.

The time-series estimator uses the same sample as the screen and discards the initial lag window separately within each oriented block. Its fold slopes receive equal weights. The HAC calculation incorporates cross-fold covariances at actual calendar distances by placing zero influence contributions at discarded months. Reported intervals assume the DML conditions and do not adjust for the preceding screen. This teaching implementation uses fixed forests and does not implement fold-specific Goldilocks tuning. The independent-unit path has no time-reversibility requirement. Panel, repeated-measures, spatially dependent and cluster-robust designs require different splitting and uncertainty calculations. [DoubleML PLR model](https://docs.doubleml.org/stable/guide/models.html) · [Ciganovic, D’Amario and Tancioni](https://doi.org/10.1093/ectj/utag019).

For a custom CSV, set **Data structure** to **Cross-sectional (independent units)** or **Monthly time series** and name Y, D and X in the prompt. Use **Run with local Llama** for edited specifications. The API connectors in this lab supply monthly series. A cross-sectional sample can be supplied through CSV or `load_cross_demo()`. The simulated monthly persistence example can be regenerated with `load_activity_demo()`. CSV/PDF reports and replay scripts use the selected design and dataset.

## Assess whether reverse-cross-fitted DML is suitable

Choose **Time-series DML assessment**, then **Run DML assessment (Python only)** for the reproducible worked example or **Run with local Llama** for the agent-reviewed path. The outcome Y is current fare inflation, the exposure D is last month's fuel inflation, and the controls X are three fare lags and annual sine/cosine terms. For another design, give the reviewer the outcome, the observed lagged exposure and its timing, and the controls. The unchanged example can also use a compatible CSV. Choose **FRED API** with this example to import the two airfare and fuel series for that DML specification. The lab prefills the IDs and computes monthly percentage changes through December 2024. Other custom specifications and imported columns can be assessed through the live prompt path. Python supplies the verdict and a row of evidence for each requirement. Completed screens support feedback, CSV and PDF reports, and downloadable Python.

The screen uses the longest contiguous run of months observed for all underlying series (ties go to the most recent run). It reports the dates and sample size and leaves gaps intact. The first two checks need at least 60 months. When seasonality is requested, those two checks use series residualized on a constant and the annual sine/cosine terms. The stability check includes the calendar controls directly.

| Condition | Teaching diagnostic | How to read a concern |
| --- | --- | --- |
| Stationarity and weak dependence | Level KPSS for each underlying series with a Bartlett long-run variance and bandwidth floor of 4 × (n/100)^(2/9). Flag any statistic above the per-series 1% reference value of 0.739, or absolute sample autocorrelations of at least 0.40 at both 6 and 12 months. | Inspect trends, breaks, and persistence. Several KPSS comparisons increase the chance of a false alert. The ACF threshold is descriptive and cannot test mixing. |
| Time reversibility | Test direction-sensitive moments of the joint stochastic series at lags 1 and 3. After standardizing and applying tanh, compare forward and reversed nonlinear within-series moments and cross-series lag products. Use approximate two-sided HAC p-values and a Bonferroni adjustment across the prespecified contrasts. Flag below 0.05. | Forward and reversed blocks differ on at least one measured feature. Symmetry of these few moments is only a necessary implication. Interpret p-values conditionally on stationarity, weak dependence and adequate sample size. |
| Conditional stability | Divide the uninterrupted span into eight adjacent blocks. Train fixed ridge models on the edges of the other blocks, reversing future training blocks and rebuilding lags inside each block. Compare held-out MSE for Y and D given X, side and distance to the boundary, with and without neighboring-block values of the outcome and exposure series. Training features never use the evaluation block. | Flag when either MSE falls by at least 10% and improves in at least five folds. Check missing dynamics or controls. This predictive proxy can miss nonlinear dependence or flag nuisance-model misspecification; it does not directly test the conditional score-bias assumption. |

The boundary probe needs at least 96 months and enough observations per block for the lags and controls. A concern on any row recommends revising the design. An unassessable row without any concern gives an inconclusive verdict. Three quiet rows make RCF-DML a candidate for further justification. None of these results verifies identification, nuisance rates or causal interpretation. The thresholds and probes are teaching choices; they are not a decision rule from the paper. [Ciganovic, D’Amario and Tancioni, *Double machine learning for time series*](https://doi.org/10.1093/ectj/utag019).

## Import monthly data from FRED

For the retail **Forecast example** or **Explore a time series** select **FRED API** to fetch the retail trade series with your own masked key. The editable default is `MRTSSM44000USN`. This preset keeps the January 2005 to December 2024 period and the nominal sales level in millions of dollars. Replacing the ID offers a new question using the first chosen series. The included snapshot requires no API key. [U.S. Census Bureau via FRED, Retail Sales: Retail Trade](https://fred.stlouisfed.org/series/MRTSSM44000USN)

In the separate time-series DML assessment the airfare and fuel IDs are editable defaults. To try the time-series DML screen on CPI levels, select **Time-series DML assessment**, choose **FRED API**, and click **Run DML assessment (Python only)** to open the connection form. Replace both IDs with `CPIAUCSL`, enter your FRED key, and click **Import and review prompt**. The five boxes now describe the CPI in its original units. Review them and click **Run DML assessment (Python only)** again. For an edited specification, choose **Run with local Llama**. **Change FRED series IDs** reopens the connection form. Plain IDs and individually backtick-quoted IDs are accepted. **Run with local Llama** uses the confirmed series and lag design for the unedited FRED assessment. For a custom single-series assessment, the analyst specifies the exposure lag and any control lags, and the studio builds the corresponding Python calls. The critic sees the three diagnostic results. A stationarity concern is a finding to report in levels. If any diagnostic flags a concern, the completed report says that the specification does not pass the time-series DML screen and names the flagged conditions. If no diagnostic flags a concern but a check lacks evidence, the verdict is inconclusive. The critic can recommend changes for a future run. Both outcomes support CSV and PDF downloads and withhold the coefficient in estimation mode. To study inflation instead, change the question and supply the transformed series.

For custom IDs, the studio uses all available monthly observations in original units. In the suggested DML specification the first ID is the outcome. For one ID its own first lag is the exposure. For multiple IDs the second ID's first lag is the exposure, and first lags of the outcome and any additional series are controls. Check the proposed roles before running. Keeping the original airfare and fuel pair in the connection form uses the historical percentage-change example.

To forecast unemployment, open **Forecasting**, enter a question about monthly unemployment in percent, and select **FRED API** as the input. Enter `UNRATE` and your key in the import form. Keys belong in the masked field. A custom question can use up to six FRED IDs; specify their roles in the prompt.

The studio makes a separate HTTPS request to FRED for each selected series. The key is used only for that import and cleared from the entry box. Agents receive the names of the loaded columns and the source. Clarification replies, writing revisions, and feedback on a completed result reuse the imported observations. A new independent import requires the key again. Get a key through [FRED](https://fred.stlouisfed.org/docs/api/api_key.html).

On macOS, if Python cannot establish or finish the HTTPS request, the studio retries that series with the built-in, certificate-verifying HTTPS client. A reply from FRED such as an invalid-key error or rate limit is shown directly. If both connection attempts fail, the panel reports the reason from each client. The key stays out of error messages and Terminal command arguments.

To check the API connection from your Mac without entering a key, run the command below. An HTTP 400 for the missing key means an HTTP response came back. A certificate, DNS, or timeout error gives you a specific connection issue to investigate.

```bash
/usr/bin/curl --disable --silent --show-error --max-time 20 -o /dev/null -w 'HTTP %{http_code}\n' \
  'https://api.stlouisfed.org/fred/series/observations?series_id=UNRATE&file_type=json'
```

The importer retains reported units, requests monthly frequency and averages higher-frequency observations. Annual and quarterly inputs cannot be converted to monthly observations here. The data are the vintage available when you import, so historical holdout scores use those revised values. For replay, click **Download imported FRED data (CSV)** and run the exported script with that file, for example `.venv/bin/python analysis_v1.py fred_monthly_input.csv`. EIA, Alpha Vantage and Custom website API are also available in the Data source menu. The included simulated data and a CSV upload require no key.

## Import from other websites

In Forecasting, write a monthly electricity price forecasting question, select **EIA API** as its data input and click **Run with local Llama**. The form opens with the official EIA endpoint for Colorado's monthly residential electricity price, the date field `period`, and the analysis column `electricity_price` mapped from EIA's `price`. Enter an EIA key and click **Import data and continue**. The forecast uses cents per kilowatthour. For Alpha Vantage, select **Alpha Vantage API** or name Alpha Vantage in your prompt. In its connection form, **Use example connection settings** loads its monthly U.S. unemployment series. You can edit the query and fields for another monthly series on that provider. [EIA API documentation](https://www.eia.gov/opendata/documentation.php) · [Alpha Vantage documentation](https://www.alphavantage.co/documentation/).

For another site, select **Custom website API**. Paste the official HTTPS GET endpoint with its series filters in the connection form, leaving the credential out of the URL. The form displays the destination hostname. Choose the authentication method from the provider's documentation. Choose a query parameter (for example `api_key`), named header (for example `X-API-Key`), bearer token, or no key. Enter a credential only in the masked box. The import sends only the date and mapped numeric columns, plus a source label, into the existing analysis workflow. A follow-up or feedback revision reuses the imported data.

For example, if a JSON response looks like the one below, enter `observations` as the records path, `month` as the date field, and `price=mean_price` as the numeric field. The analyst will see columns named `date` and `price`.

```json
{"observations":[{"month":"2025-01","mean_price":18.4},{"month":"2025-02","mean_price":18.7}]}
```

A JSON list at the root needs no records path. For CSV, enter its date and numeric column headers. Use an endpoint returning one page with at most one record per month. Date fields accept YYYY-MM and YYYY-MM-DD. The custom importer supports public HTTPS GET requests on port 443 with direct network access and a response under 2 MB. Pagination, interactive OAuth login, POST-only APIs and merging different providers in one run require preprocessing or a monthly CSV upload. An existing bearer access token can be used directly.

## Complete an incomplete prompt

Select **Forecast example** and delete the text in **Structure**. An amber outline and a **Needs detail** label appear immediately. After a short typing pause, the box offers clickable starter wording. The local Llama reviewer uses the current draft to tailor the suggestions. For a forecast, one possible Structure answer is “Fit on earlier months, score one-month-ahead forecasts on the final 24 months, then report the next forecast and its units.”

Click a suggestion to fill an empty box. Try typing your own first step instead. The highlight clears, and the reviewer can offer a next sentence. Clicking that suggestion appends it on a new line and preserves your first step. Keep typing to ignore a suggestion, choose **Write my own** to hide help for that box, or turn off **Show writing suggestions while I type** for the whole form. Toggle it back on to restore help. Built-in starters are available while Llama responds. Writing suggestions do not start an analysis or send CSV rows to the reviewer.

**Try an incomplete prompt** leaves Structure, Role and Examples blank so you can practice all three. Click **Review my prompt** to review the wording. When ready, click **Run with local Llama** in the same workspace. When **One more detail** asks which series to forecast, enter `MRTSSM44000USN` and click **Send answer and continue** once. The studio matches the reply against the data and carries the confirmed outcome into code generation and review. The original draft and clarification remain in the report.

This check covers all five boxes in the browser form. Free-text notebook and CLI prompts continue to use the existing writing reviewer.

## Try the writing coach

In **Forecasting → Prompt writing**, expand **Try a writing exercise** and select **Try a rough prompt**. Click **Review my prompt**, revise the five fields, then click **Run with local Llama**. The example has a clear forecasting goal but reads like shorthand. The Prompt Reviewer gives a copy-ready rewrite and identifies an improvement. Edit the suggested wording if desired, then choose **Use this wording and continue** or **Keep my wording and continue**. The analyst waits for this choice. If your writing is already clear, the reviewer can send it straight to the analyst.

If an essential detail is missing, answer the reviewer’s question first. A sample sentence may contain a placeholder, such as [outcome], to show what information belongs there. The original wording, selected wording, and review trail are available in the downloads. The same choices apply to feedback on a completed analysis. Python-only reference runs do not call the writing coach.

## What happens after submission

1. **Prompt reviewer checks the request.** A separate local Llama role gives feedback on Clarity, Structure, Context, Role, and Examples. If essential information is missing it asks one focused question and pauses before model fitting. Answer in the **One more detail** panel and click **Send answer and continue**. The reviewer checks the original prompt, its data, and your replies before handing the task to the analyst. The browser first checks for a response in every box; the reviewer then considers whether the completed request is actionable. Use **Try an incomplete prompt** to practice choosing a forecast target.
2. **Analyst writes Python.** The local model generates two executable calls to the package's documented API. The first chooses the outcome, lagged inputs and seasonality. The second selects the forecasting or inference function. Python validates syntax, method, column names and arguments. If Llama wraps the two calls in a familiar script header or simple result prints, the studio displays the two effective calls and uses the selected data. The original draft stays in the transcript. Review, execution and replay use the validated calls. The permitted Python subset excludes imports, filesystem access and shell commands.
3. **Python executes the proposed analysis for review.** For forecasting, the agents see development-fold evidence, features and base-model diagnostics. For estimation, Python fits the regression and shows the specification and sample used. Coefficient magnitudes and interval endpoints are released after acceptance. A time-series DML assessment reports three diagnostic findings and a suitability verdict. In the worked assessment, Python checks that the code matches the requested outcome, exposure lag, and controls before publishing the verdict.
4. **Critic reviews and analyst responds.** Each revision is rerun. After the critic accepts, the analyst independently confirms the same code and evidence. The interface shows their messages, the code, and the execution evidence. A missing design choice triggers a question. A disagreement about the program can require a revision. For a time-series DML assessment, the outcome appears under **Assessment complete**. The critic can comment on the findings under **Review note**.
5. **The agents review the numerical result.** After the code is accepted, Python computes the final table. The analyst interprets it and the critic checks the wording against the actual numbers. A second pass can correct the interpretation with the model and numbers frozen. The user receives the table and accepted explanation together. Download **CSV** for spreadsheet-ready results and review records, **PDF** for a paginated report, or **Python** for the runnable program. The round display shows which worker is active and each response carries a labeled decision badge. Reduced-motion settings are respected. Agreement records both agents' methodological judgment. Statistical validity still depends on the data and assumptions.

The generated script runs as `python analysis.py` with the simulated data, or `python analysis.py my_monthly.csv` to replay on the same custom CSV used in the page. A new Llama call is unnecessary for replay.

## Give feedback on a completed result

On the results panel, enter a change such as “Add a six-month sales lag and keep the forecast holdout fixed.” Click **Apply feedback with local Llama**. Both roles receive the previous approved code and the requested change. The revision uses the same input data and repeats numerical execution and review. Feedback on a Python-only reference starts a live Llama revision. Use the version selector to revisit and export either result. Each revised report lists the feedback, previous headline result and actual changes in model settings. Failed revisions leave the previous result available.

Reports and input data are retained in memory for the latest 40 completed versions while the local server runs. Download any reports you want to keep before closing the app. Starting a new question starts a new version list on the page.

## Python, notebook and terminal

```python
from studio_core.api import analyze
from studio_core.example_prompt import RETAIL_PROMPTS

result = analyze(RETAIL_PROMPTS[0])
print(result.metrics.round(3))
result.save("my_results")  # choose a new folder
```

Use `examples/retail_sales_walkthrough.ipynb` for the live prompt-led notebook, or use `econ422-studio ask "your question" --out my_results` in Terminal. Both use the same workflow as the page. The optional `econ422-studio demo` command runs the retail sales reference forecast offline and requires no Llama model. The live page never substitutes the scripted demo if Ollama is unavailable.

## The simulated economy

`studio_core/data/monthly_airfares.csv` contains 239 observed months between January 2005 and December 2024. The rates are simulated monthly percentage changes. The fuel series is persistent. Airfare inflation depends on last month's airfare inflation, last month's fuel inflation, a threshold for large fuel increases, an annual seasonal cycle, and a random shock. This gives the linear and tree learners different patterns to learn.

The simulation runs continuously for 240 months, then removes April 2012. Before lagging, the package restores the full monthly calendar. In May 2012 the one-month fuel lag is therefore missing. Training and scoring use complete rows. Use `simulate_data(seed=422)` to reproduce the CSV or supply a different seed to try another simulated economy.

## What the Super Learner learns

1. Reserve the last 24 calendar months for the demo evaluation. Only the earlier development period goes to the analyst–critic loop.
2. Use the first half of complete development rows (at least 36) for initial training. Split the remaining dates into four successive validation blocks. Within each fold, fit LASSO (including its scaler) and random forest using only earlier rows. Predict each validation month using observations available through the prior month.
3. Pool the two columns of out-of-fold predictions. Choose the weight `w` between 0 and 1 that minimizes the sum of squared validation errors for `w × LASSO + (1 − w) × forest`. Each component forecast was made before its target was used for training. The weight-fitting error is an in-sample criterion for the combination.
4. After critic acceptance, fit both base learners on all complete development rows. Freeze these fitted models and the accepted weights for holdout scoring. Update lagged inputs as each new month is observed. This yields a sequence of one-month-ahead forecasts; it does not assume all 24 months can be forecast from the first origin.
5. Compare both learners, the Super Learner and a last-observation benchmark on identical holdout dates. Separately refit the base learners using all available labeled rows and retain the weights to produce one forecast beyond the CSV. If required latest inputs are missing, the audit gives the reason.

## Estimation details

The inference tool estimates a linear projection of the current outcome on the specified lagged regressors, an intercept and optional annual sine/cosine. It uses a Bartlett HAC sandwich with calendar-month lag distances, a finite-sample factor n/(n−k), and a normal critical value for its 95% interval. Missing calendar months contribute zero scores. Interpretation assumes a suitable, stable monthly relationship and weak dependence for large-sample inference. The interval is pointwise for the specified coefficient. The tool supports associational interpretation. A causal effect would require an identification design.

## Reproducibility

The retail Super Learner forecasts January 2025 sales of about 587,399 million dollars, with holdout RMSE around 40,115 million dollars. Both numbers use the bundled historical snapshot. Llama can propose a different program for a different prompt. Run `python -m unittest discover -s tests -v` after installation. Automated tests exercise the prompt and review protocol with controlled model replies.

Forecast method: [Super Learner](https://pubmed.ncbi.nlm.nih.gov/17910531/). Local model interface: [Ollama structured outputs](https://docs.ollama.com/capabilities/structured-outputs).

## ECON 422 practice

Open `ECON422_Studio_Guide_and_Exercises.docx` for the Mac launch instructions and two five-field prompt exercises. Problem 1 uses the included historical retail sales snapshot and the five-field exploration and forecast templates. The CPI assessment uses `examples/data/CPIAUCSL_2015_2024.csv` and needs no API key. The CSV contains monthly CPI index levels from January 2015 through December 2024, from the U.S. Bureau of Labor Statistics via FRED, series [CPIAUCSL](https://fred.stlouisfed.org/series/CPIAUCSL).
