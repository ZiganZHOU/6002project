# Bayesian Machine Learning for Cardiovascular Disease Diagnosis

Course project for **Bayesian Machine Learning**. The project uses Bayesian
probit and logistic regression to perform an automated medical diagnosis task:
predicting whether a patient will develop coronary heart disease (CHD) within
10 years.

The implementation is intentionally probability-focused. Instead of reporting
only a hard 0/1 label, the Bayesian models produce posterior distributions over
patient risk and coefficient effects.

## Project Fit

This repository directly addresses the project prompt:

- **Bayesian model**: Bayesian logistic and Bayesian probit regression in PyMC.
- **Machine learning task**: binary supervised classification.
- **Automated disease diagnosis**: 10-year CHD risk prediction from clinical
  and demographic covariates.
- **Critical Bayesian output**: posterior predictive risk intervals, posterior
  coefficient uncertainty, WAIC/PSIS-LOO comparison, and cost-sensitive
  decision thresholds.

## Methodological Contributions

The project now includes three method-level extensions beyond a direct
application of Bayesian logistic/probit regression:

1. **Grouped hierarchical shrinkage prior**: predictors are grouped by clinical
   meaning (demographics, behavior, medical history, vitals, labs), and each
   group receives a learned shrinkage scale. This lets the model infer which
   clinical feature families need stronger or weaker regularization.
2. **Posterior expected-loss triage with reject option**: the final decision is
   not forced into low/high risk. The model can output `refer` when posterior
   risk is too uncertain or expected loss favors additional testing.
3. **Bayesian optimization for triage policy tuning**: a Gaussian-process
   surrogate with a lower-confidence-bound acquisition function tunes the
   triage parameters `q_low`, `q_high`, and internal referral cost using saved
   posterior risk samples, without rerunning MCMC.

## Dataset

**Framingham Heart Study Dataset**  
Source: [Kaggle mirror](https://www.kaggle.com/datasets/aasheesh200/framingham-heart-study-dataset)

Download `framingham.csv` and place it in `data/`. Raw data are intentionally
not tracked by git.

Local dataset snapshot used during development:

- Rows: 4,240
- Predictors: 15
- Target: `TenYearCHD`
- Positive class: 644 patients, or 15.2%
- Largest missingness source: `glucose` with 388 missing values

## Setup

```powershell
python -m venv venv
venv\Scripts\activate
python -m pip install -r requirements.txt
```

If you already have the virtual environment and only need to install packages:

```powershell
venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run Workflow

The project no longer depends on notebooks. Use the root runner for both full
and stepwise execution.

```powershell
# Full project: EDA, preprocessing, baseline, sampling, evaluation, optimization
python run_project.py all

# Quick non-Bayesian check when PyMC is not installed yet
python run_project.py all --skip-bayes

# Stepwise execution
python run_project.py eda
python run_project.py preprocess
python run_project.py baseline
python run_project.py bayes
python run_project.py evaluate
python run_project.py optimize
```

For a short smoke test of the Bayesian code, reduce the sampler settings:

```powershell
python run_project.py bayes --draws 500 --tune 500 --chains 2 --no-shrinkage
python run_project.py evaluate --prediction-draws 500
```

The `bayes` step is the long-running MCMC step. It writes
`data/idata_logistic.nc`, `data/idata_logistic_shrink.nc`, and
`data/idata_probit.nc`, and `data/idata_hierarchical_logistic.nc`; those files
are ignored because they can be large. Plots, metrics, posterior risk samples,
Bayesian-optimization history, and triage decisions are written to `outputs/`.

## Methodology

1. **Exploratory analysis**
   - Check class imbalance, missingness, feature distributions, and
     correlations.
   - Compute VIF on train-only imputed and standardized data with an intercept,
     avoiding inflated VIF from raw uncentered measurements.

2. **Leakage-safe preprocessing**
   - Split raw data first with stratification.
   - Fit `IterativeImputer` on the training set only.
   - Restore simple domain constraints after imputation: binary variables are
     rounded/clipped to 0/1, education is clipped to its ordinal range, and
     physiologic measurements are clipped to nonnegative values.
   - Fit `StandardScaler` on training data only and transform train/test.

3. **Models**
   - Frequentist logistic regression as a sanity-check baseline.
   - Bayesian logistic regression with weakly informative Normal priors.
   - Bayesian probit regression with the same design matrix and comparable
     priors.
   - Shrinkage prior sensitivity for the logistic model.
   - Grouped hierarchical Bayesian logistic regression with learned
     group-specific shrinkage scales.

4. **Evaluation**
   - ROC-AUC, PR-AUC, Brier score, calibration curves.
   - MCMC diagnostics: divergences, R-hat, effective sample size.
   - WAIC and PSIS-LOO for Bayesian model comparison.
   - Bayesian odds-ratio summaries for logistic regression only.
   - Posterior predictive risk intervals for individual patients.
   - Asymmetric clinical loss with threshold
     `C_FP / (C_FP + C_FN)`.
   - Three-action posterior triage: `low`, `high`, and `refer`.
   - Bayesian optimization of triage-policy parameters on held-out posterior
     risk samples.

## Baseline Sanity Check

The baseline results are intentionally not hard-coded in the README. Run:

```powershell
python run_project.py baseline
```

This writes the reproducible baseline metrics and decision-cost outputs to:

```text
outputs/baseline_metrics.json
outputs/baseline_predictions.csv
outputs/baseline_cost_curve.csv
outputs/baseline_cost_curve.png
```

Because the target is imbalanced, the project treats accuracy as secondary.
The main workflow emphasizes PR-AUC, calibration, posterior uncertainty, and
cost-sensitive triage rather than a default 0.5 classification threshold.

## Project Structure

```text
.
|-- data/                      # Raw data and generated traces; raw CSV not tracked
|-- outputs/                   # Generated metrics/plots; not tracked
|-- report/
|   `-- report.md
|-- src/
|   |-- preprocess.py
|   |-- model.py
|   `-- evaluate.py
|-- run_project.py             # Full and stepwise project runner
|-- requirements.txt
`-- README.md
```

## Notes

- Odds ratios are valid for the logistic link only: `OR = exp(beta)`.
- Probit coefficients live on the latent-normal scale and should not be
  exponentiated as odds ratios.
- This is an educational risk-modeling project, not a clinical deployment
  system.
