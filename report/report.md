# Project Report: Bayesian Probit/Logistic Regression for CVD Diagnosis

## 1. Introduction

*(TODO: background, motivation, clinical significance)*

## 2. Dataset

*(TODO: describe Framingham dataset, features, target variable)*

## 3. Methodology

### 3.1 Data Preprocessing
*(TODO: missing value strategy, scaling)*

### 3.2 Model Specification

#### Bayesian Logistic Regression
- Link function: logit (sigmoid)
- Priors: α ~ N(0, 1), β_j ~ N(0, 2.5)
- Inference: NUTS via PyMC

#### Bayesian Probit Regression
- Link function: Φ (standard normal CDF)
- Priors: same as above
- Inference: NUTS via PyMC

### 3.3 Prior Justification
*(TODO)*

## 4. Results

*(TODO: tables, plots from notebooks)*

## 5. Discussion

*(TODO: comparison with frequentist baseline, uncertainty quantification, limitations)*

## 6. Conclusion

*(TODO)*

## References

*(TODO)*
