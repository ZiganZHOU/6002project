# Bayesian Probit/Logistic Regression for Cardiovascular Disease Diagnosis

A graduate course project applying Bayesian machine learning to automated disease diagnosis, using the **Framingham Heart Study** dataset from Kaggle.

## Project Overview

This project constructs Bayesian probit/logistic regression models to predict the **10-year risk of cardiovascular disease (CVD)**. We compare:

- Classical (frequentist) logistic regression
- Bayesian logistic regression
- Bayesian probit regression

Key strengths of this workflow:
- Leakage-safe preprocessing (split first, fit transforms on train only)
- Full posterior predictive risk distributions (not point-estimate-only prediction)
- Clinical interpretability via Bayesian Odds Ratios (logistic only)
- Cost-sensitive Bayesian decision thresholding for imbalanced diagnosis

## Dataset

**Framingham Heart Study Dataset**  
Source: [Kaggle](https://www.kaggle.com/datasets/aasheesh200/framingham-heart-study-dataset)

Download `framingham.csv` and place it in the `data/` directory. The dataset contains ~4,000 patients and 15 features including:

| Feature | Description |
|---|---|
| `age` | Age of the patient |
| `male` | Sex (1 = male) |
| `currentSmoker` | Current smoker status |
| `cigsPerDay` | Cigarettes per day |
| `BPMeds` | Blood pressure medication |
| `prevalentStroke` | Prior stroke history |
| `prevalentHyp` | Hypertension |
| `diabetes` | Diabetes status |
| `totChol` | Total cholesterol |
| `sysBP` | Systolic blood pressure |
| `diaBP` | Diastolic blood pressure |
| `BMI` | Body mass index |
| `heartRate` | Heart rate |
| `glucose` | Glucose level |
| `TenYearCHD` | **Target**: 10-year coronary heart disease risk |

## Project Structure

```
.
├── data/                    # Raw data (not tracked by git)
├── notebooks/
│   ├── 01_eda.ipynb         # Exploratory data analysis
│   ├── 02_preprocessing.ipynb  # Feature engineering & preprocessing
│   ├── 03_bayesian_logistic.ipynb  # Model building & MCMC sampling
│   └── 04_evaluation.ipynb  # Model comparison & evaluation
├── src/
│   ├── preprocess.py        # Data loading & preprocessing utilities
│   ├── model.py             # Bayesian model definitions (PyMC)
│   └── evaluate.py          # Evaluation metrics & plotting
├── report/
│   └── report.md            # Final report
├── requirements.txt
└── README.md
```

## Setup

```bash
# Clone repo
git clone <your-repo-url>
cd bayesian-cvd-diagnosis

# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Download dataset from Kaggle, place framingham.csv in data/

# Launch Jupyter
jupyter notebook
```

## Methodology

1. **EDA**:
   - Explore class imbalance, feature distributions, correlations
   - Diagnose multicollinearity using VIF (for prior/feature strategy)
2. **Leakage-safe preprocessing**:
   - Split raw data into train/test with stratification
   - Fit MICE imputer on train only, transform train/test
   - Fit scaler on imputed train only, transform train/test
3. **Bayesian Modeling**:
   - Logistic and probit likelihoods with weakly informative priors
   - MCMC sampling via NUTS (4 chains for final report)
   - Log-likelihood computation for WAIC/PSIS-LOO comparison
   - Prior sensitivity analysis (e.g., Normal(0, 2.5) vs Normal(0, 0.5))
4. **Posterior predictive inference**:
   - Use `sample_posterior_predictive(var_names=["p"])` to obtain continuous risk
   - Report mean risk and 95% HDI for individual patients
5. **Evaluation under imbalance**:
   - ROC-AUC, PR-AUC, Brier score, calibration curves
6. **Clinical decision analysis**:
   - Asymmetric loss (false negative cost > false positive cost)
   - Bayes-optimal threshold \(p^* = \frac{C_{FP}}{C_{FP}+C_{FN}}\)

## Reproducible Run Order

Run notebooks in this exact order:

1. `notebooks/01_eda.ipynb`
2. `notebooks/02_preprocessing.ipynb`
3. `notebooks/03_bayesian_logistic.ipynb` (long runtime; generates `data/idata_*.nc`)
4. `notebooks/04_evaluation.ipynb`

## Interpretation Notes

- **Odds Ratios (OR)** are reported for **logistic** coefficients only: \(OR=\exp(\beta)\).
- For **probit**, coefficients are interpreted on the latent-normal scale (not OR).
- If PPD is accidentally sampled on `y_obs` instead of `p`, uncertainty intervals collapse to binary outcomes; this project explicitly avoids that pitfall.

## References

- Framingham Heart Study: https://www.framinghamheartstudy.org/
- PyMC documentation: https://www.pymc.io/
- Gelman et al., *Bayesian Data Analysis* (3rd ed.)
- Gelman, A., Jakulin, A., Pittau, M. G., & Su, Y.-S. (2008). *A weakly informative default prior distribution for logistic and other regression models.*
