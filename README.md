# Generalizable Bayesian Optimization for Medical Risk Triage

Course project for **Bayesian Machine Learning**. The project started as a
Bayesian cardiovascular disease diagnosis workflow and has been extended into a
generalizable decision-calibration framework:

1. A probabilistic risk model estimates patient-level disease risk.
2. The model exposes uncertainty through posterior or bootstrap risk samples.
3. A three-action triage policy converts risk into `low`, `high`, or `refer`.
4. Bayesian optimization calibrates the triage policy under asymmetric clinical
   costs.
5. Additional datasets test whether the decision-calibration layer transfers
   beyond the original cardiovascular task.

The main Bayesian case study remains the Framingham 10-year CHD prediction
task. Cross-dataset experiments use lightweight bootstrap logistic risk samples
so that generalization and ablation studies can be run quickly without repeated
MCMC.

## Project Fit

- **Bayesian model**: PyMC Bayesian logistic/probit regression for the main
  Framingham case study.
- **Bayesian output**: posterior predictive risk samples, risk intervals,
  coefficient uncertainty, WAIC/PSIS-LOO, and cost-sensitive decisions.
- **Optimization contribution**: Gaussian-process Bayesian optimization tunes
  the downstream triage-policy parameters using saved risk samples.
- **Generalization claim**: the optimized decision layer is evaluated on
  multiple binary medical datasets, not only on the original CHD dataset.
- **Ablation claim**: BO triage is compared against fixed policy, BO without
  transfer initialization, random search, forced binary decisions, and
  mean-risk-only decision making.

## Datasets

Raw external datasets are stored in `data/raw_external/`. Modeling-ready CSVs
are stored in `data/` with a unified binary target column named `target`, except
for the original Framingham file where the target remains `TenYearCHD`.

| Dataset | Local File | Role | Notes |
|---|---|---|---|
| Framingham Heart Study | `data/framingham.csv` | Main Bayesian case study | 10-year CHD risk, 4,240 rows, 15 predictors |
| Breast Cancer Wisconsin Diagnostic | `data/breast_cancer_wisconsin_diagnostic.csv` | Cross-disease generalization | Malignant vs benign, clean numeric data |
| Mammographic Mass | `data/mammographic_mass.csv` | Triage-style generalization | Malignant vs benign, has missing values |
| UCI Cleveland Heart Disease | `data/heart_disease_cleveland.csv` | Same-domain generalization | Heart disease present vs absent |
| CDC Diabetes Health Indicators | `data/cdc_diabetes_health_indicators.csv` | Large-data robustness | Diabetes/prediabetes vs healthy; sampled by default |

The UCI `.data` files are preserved as raw files. CSV versions add explicit
column names and a unified binary `target` column for reproducible modeling.

## Setup

```powershell
python -m venv venv
venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Run Workflow

```powershell
# Full project: Framingham analysis plus cross-dataset generalization
python run_project.py all

# Skip expensive PyMC sampling when only checking the non-Bayesian parts
python run_project.py all --skip-bayes

# Skip cross-dataset generalization if you only need the main case study
python run_project.py all --skip-generalization
```

Stepwise execution:

```powershell
python run_project.py eda
python run_project.py preprocess
python run_project.py baseline
python run_project.py bayes
python run_project.py evaluate
python run_project.py optimize
python run_project.py generalize
python run_project.py ablate
```

For a short smoke test of the Bayesian code:

```powershell
python run_project.py bayes --draws 500 --tune 500 --chains 2 --no-shrinkage
python run_project.py evaluate --prediction-draws 500
```

For a short smoke test of the new generalization layer:

```powershell
python run_project.py generalize `
  --generalization-datasets breast_cancer `
  --generalization-bootstrap-samples 5 `
  --generalization-bo-initial 3 `
  --generalization-bo-iter 2 `
  --generalization-bo-candidates 50
```

Transfer initialization is enabled by default. To disable it for sensitivity
checks:

```powershell
python run_project.py generalize --no-transfer-initialization
```

The `bayes` step is the long-running MCMC step. It writes trace files to
`data/*.nc`. The `generalize`/`ablate` steps write summaries, BO histories, and
ablation outputs to `outputs/generalization/`, including per-dataset
convergence plots and a cost-referral trade-off plot.

## Methodology

1. **Leakage-safe preprocessing**
   - Split raw data before imputation and scaling.
   - Fit imputers/scalers on training data only.
   - Restore simple clinical constraints where relevant.
   - Standardize numeric predictors before logistic/Bayesian models.

2. **Main Bayesian risk modeling**
   - Frequentist logistic regression is used as a sanity-check baseline.
   - Bayesian logistic regression and Bayesian probit regression model CHD risk.
   - A grouped hierarchical Bayesian logistic model learns group-specific
     shrinkage across clinical feature families.

3. **Uncertainty-aware decision layer**
   - Risk samples are used to estimate mean risk and uncertainty around the
     cost-based decision threshold.
   - The triage policy can output `refer` instead of forcing every uncertain
     patient into low/high risk.

4. **Bayesian optimization**
   - Tuned parameters are `q_low`, `q_high`, and internal referral cost.
   - The objective is average clinical cost plus a penalty for referral rate.
   - The acquisition function uses a dynamic lower-confidence-bound schedule:
     more exploration early, more exploitation later.
   - Cross-dataset transfer initialization can seed a new task with previously
     optimized triage-policy parameters.

5. **Generalization and ablation**
   - External datasets reuse the same policy optimizer.
   - Bootstrap logistic ensembles provide fast risk samples for auxiliary
     experiments.
   - BO triage is compared with fixed triage, BO without transfer initialization,
     random search, no-refer forced binary decisions, and mean-risk-only triage.

## Evaluation

Predictive accuracy is evaluated with:

- ROC-AUC
- PR-AUC
- Brier score
- Calibration curves
- Accuracy, sensitivity, specificity, and precision

Decision quality is evaluated with:

- Cost-sensitive threshold performance
- Average clinical cost
- Referral rate
- Coverage
- Decided-case accuracy
- False negatives and false positives
- BO best-so-far convergence curves
- Cost-referral trade-off plots

Generalization is evaluated by asking whether the same decision-calibration
framework improves or remains competitive across breast cancer, mammographic
mass, Cleveland heart disease, and CDC diabetes datasets.

Robustness can be checked by changing:

- random seed
- train/test split
- false-negative cost
- referral cost
- BO budget
- bootstrap sample count
- CDC row cap

## Project Structure

```text
.
|-- data/
|   |-- raw_external/          # Original downloaded UCI/Kaggle files
|   |-- framingham.csv
|   |-- breast_cancer_wisconsin_diagnostic.csv
|   |-- mammographic_mass.csv
|   |-- heart_disease_cleveland.csv
|   `-- cdc_diabetes_health_indicators.csv
|-- outputs/                   # Generated metrics/plots; not tracked
|-- report/
|   `-- report.md
|-- src/
|   |-- datasets.py
|   |-- preprocess.py
|   |-- model.py
|   |-- evaluate.py
|   `-- generalization.py
|-- run_project.py
|-- requirements.txt
`-- README.md
```

## Notes

- Odds ratios are valid for logistic models only: `OR = exp(beta)`.
- Probit coefficients live on the latent-normal scale and should not be
  exponentiated as odds ratios.
- Auxiliary bootstrap experiments are used for fast generalization checks; the
  main Bayesian posterior analysis remains the Framingham PyMC workflow.
- This is an educational risk-modeling project, not a clinical deployment
  system.
