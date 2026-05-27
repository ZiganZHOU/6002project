# Bayesian Probit/Logistic Regression for Cardiovascular Disease Diagnosis

A graduate course project applying Bayesian machine learning to automated disease diagnosis, using the **Framingham Heart Study** dataset from Kaggle.

## Project Overview

This project constructs a Bayesian probit/logistic regression model to predict the **10-year risk of cardiovascular disease (CVD)** in patients. We compare:

- Classical (frequentist) logistic regression
- Bayesian logistic regression (with informative/non-informative priors)
- Bayesian probit regression

The Bayesian approach provides posterior distributions over model parameters, enabling principled uncertainty quantification in clinical predictions.

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

1. **EDA**: Explore class imbalance, feature distributions, correlations
2. **Preprocessing**: Handle missing values, standardize features, train/test split
3. **Bayesian Modeling**:
   - Prior specification (weakly informative priors on coefficients)
   - MCMC sampling via NUTS (No-U-Turn Sampler) using PyMC
   - Posterior predictive checks
4. **Evaluation**: ROC-AUC, accuracy, calibration plots, comparison with frequentist baseline

## References

- Framingham Heart Study: https://www.framinghamheartstudy.org/
- PyMC documentation: https://www.pymc.io/
- Gelman et al., *Bayesian Data Analysis* (3rd ed.)
