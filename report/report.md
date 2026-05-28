# Project Report: Bayesian Probit/Logistic Regression for CVD Diagnosis

## 1. Introduction

This project applies **Bayesian probit/logistic regression** to automated disease diagnosis, focusing on predicting **10-year coronary heart disease (CHD) risk** using the Framingham Heart Study dataset. Compared to point-estimate classifiers, Bayesian models provide **posterior uncertainty** over coefficients and **predictive uncertainty** over patient-level risk, which is critical for clinical decision support.

## 2. Dataset

We use the **Framingham Heart Study** dataset (Kaggle mirror). The dataset contains demographic, behavioral, and clinical measurements (e.g., age, smoking status, blood pressure, cholesterol, BMI, glucose). The binary target is:

- **`TenYearCHD`**: 1 if CHD occurs within 10 years, else 0.

The target is **class-imbalanced** (positive rate typically around ~15%), therefore probability-focused metrics (PR-AUC, Brier score, calibration) are emphasized alongside ROC-AUC.

## 3. Methodology

### 3.1 Data Preprocessing

- **Missing data (MICE)**: We use **IterativeImputer (MICE-like multiple imputation)** to impute missing feature values. Compared to median imputation, MICE better preserves the multivariate correlation structure and avoids artificially shrinking variance, aligning with the project’s emphasis on uncertainty.
- **Leakage-safe pipeline**: To avoid **data leakage**, we strictly follow *split-first, fit-on-train-only*: (1) stratified train/test split on raw data; (2) fit the MICE imputer on the training set and apply `transform()` to both training and test sets; (3) fit StandardScaler on the imputed training set and transform both sets.
- **Feature scaling**: StandardScaler is applied to features. This stabilizes MCMC geometry and makes weakly-informative Normal priors on coefficients more appropriate.

Implementation reference: `src/preprocess.py`.

### 3.2 Model Specification

#### Bayesian Logistic Regression

The generative model is specified as follows. For patient \(i=1,\dots,n\):

\[
y_i \sim \mathrm{Bernoulli}(p_i)
\]

\[
\mathrm{logit}(p_i) = \alpha + \sum_{j=1}^{k} X_{i,j}\,\beta_j
\]

where \(X\in\mathbb{R}^{n\times k}\) is the standardized design matrix. Weakly-informative priors are:

\[
\alpha \sim \mathcal{N}(0,1), \quad \beta_j \sim \mathcal{N}(0,2.5)
\]

Inference is performed via NUTS in PyMC.

#### Bayesian Probit Regression

The likelihood is the same, \(y_i\sim\mathrm{Bernoulli}(p_i)\), but the link is probit:

\[
p_i = \Phi\Big(\alpha + \sum_{j=1}^{k} X_{i,j}\,\beta_j\Big)
\]

with the same priors \(\alpha\sim\mathcal{N}(0,1)\), \(\beta_j\sim\mathcal{N}(0,2.5)\), and inference via NUTS.

**Latent-variable interpretation (probit):** probit regression can be viewed as introducing a continuous latent health index

\[
Z_i \sim \mathcal{N}(\mu_i, 1), \quad \mu_i = \alpha + \sum_{j=1}^{k} X_{i,j}\,\beta_j
\]

and defining the observed outcome via a threshold:

\[
y_i = \mathbb{I}(Z_i > 0).
\]

This interpretation is often appealing in medical settings, where disease risk may reflect the accumulation of many small effects that approximately aggregate to a normal latent severity score. In contrast, the logistic model’s key practical advantage is the direct clinical interpretation through **odds ratios**.

### 3.3 Prior Justification

Inputs are standardized via StandardScaler. Under standardization, weakly-informative priors such as \(\beta_j \sim \mathcal{N}(0,2.5)\) (or similarly-scaled Cauchy priors) are a common default recommendation for logistic regression to prevent extreme implied probabilities while remaining flexible. This choice is motivated by the practical prior scale guidance in Gelman et al. (2008), and it improves both regularization and sampling stability.

We additionally assess multicollinearity using **VIF (Variance Inflation Factor)**. If severe multicollinearity is present (e.g., VIF \(>10\) for highly correlated physiological variables such as systolic/diastolic blood pressure), we mitigate instability by either (i) removing redundant predictors or (ii) increasing prior regularization (e.g., smaller Normal scale or sparsity-inducing priors), which improves posterior geometry and MCMC convergence.

### 3.4 Inference Details and Diagnostics

- **Sampling**: NUTS with **4 chains** (recommended for reliable \(\hat{R}\) and ESS).
- **Diagnostics**: We report \(\hat{R}\), effective sample sizes (bulk/tail), and divergences. Final runs target **0 divergences**, \(\hat{R} \approx 1.00\), and sufficient ESS.
- **Log-likelihood**: Computed after sampling to support **WAIC/PSIS-LOO** model comparison.
- **Prior sensitivity analysis**: To assess robustness under multicollinearity, we compare baseline priors (\(\beta_j\sim\mathcal{N}(0,2.5)\)) with stronger shrinkage priors (e.g., \(\beta_j\sim\mathcal{N}(0,0.5)\), optionally Laplace). We then compare convergence diagnostics (ESS/\(\hat{R}\)/divergences) and posterior geometry for key correlated predictors.

Implementation reference: `src/model.py` and Notebook 03.

## 4. Results

We compare:

- Frequentist Logistic Regression (sklearn baseline)
- Bayesian Logistic Regression
- Bayesian Probit Regression

### 4.1 MCMC Convergence Validations

Before evaluating predictive performance, we confirmed MCMC convergence for all Bayesian models. Trace plots indicated good mixing, with \(\hat{R} \le 1.01\) and zero divergences across all chains, ensuring the reliability of posterior samples used for downstream analyses.

### 4.2 Discrimination

- **ROC-AUC** is reported for overall ranking performance.
- **PR-AUC** is reported due to class imbalance.

### 4.3 Calibration and Probabilistic Accuracy

- **Calibration curves** assess probability calibration.
- **Brier score** measures mean squared error of predicted probabilities.

### 4.4 Posterior Interpretability

We visualize posterior distributions of coefficients \(\beta\) (forest plot) to identify risk factors with consistently positive/negative effects. For the **logistic model only**, we exponentiate posterior samples of \(\beta\) to obtain full posterior distributions of **Odds Ratios (OR)** and report median OR with 95% HDI for key risk factors.

For the **probit model**, \(\exp(\beta)\) does **not** have an OR interpretation; coefficients are interpreted on the latent-normal scale (or compared approximately via \(\beta_{\text{logistic}}\approx1.6\,\beta_{\text{probit}}\)).

### 4.5 Predictive Uncertainty (Key Bayesian Output)

We compute **posterior predictive distributions** for patient risk \(p(\mathrm{CHD}=1 \mid x)\) and report **mean risk** and **95% HDI** for individual patients, illustrating uncertainty bands that are unavailable under point-estimate approaches.

(See Notebook 04 outputs.)

### 4.6 Bayesian Decision Analysis (Clinical Utility)

In clinical screening, the costs of errors are asymmetric: missing a high-risk patient (false negative) can be far more costly than recommending follow-up for a low-risk patient (false positive). Using a simple asymmetric loss \(C_{FN}\) vs. \(C_{FP}\), the Bayes-optimal decision rule under posterior risk \(p\) is:

\[
\text{predict CHD if } C_{FP}(1-p) < C_{FN}p \quad \Leftrightarrow \quad p > \frac{C_{FP}}{C_{FP}+C_{FN}}.
\]

This threshold follows directly from **expected-loss minimization**. Let action \(a\in\{0,1\}\) (0: no intervention, 1: intervene) and true state \(\theta\in\{0,1\}\) (0: no CHD event, 1: CHD event). With \(\mathcal{L}(a=1,\theta=0)=C_{FP}\) and \(\mathcal{L}(a=0,\theta=1)=C_{FN}\):

\[
\mathbb{E}[\mathcal{L}(a=1)\mid D] = C_{FP}\,P(\theta=0\mid D), \quad
\mathbb{E}[\mathcal{L}(a=0)\mid D] = C_{FN}\,P(\theta=1\mid D).
\]

Choose \(a=1\) when \(\mathbb{E}[\mathcal{L}(a=1)\mid D] < \mathbb{E}[\mathcal{L}(a=0)\mid D]\), which yields:

\[
P(\theta=1\mid D) > \frac{C_{FP}}{C_{FP}+C_{FN}}.
\]

Here, \(P(\theta=1\mid D)\) is estimated by the posterior predictive risk (posterior mean of \(p\)). We visualize test-set decision cost across thresholds and report the cost-based optimal threshold for transparent, decision-oriented deployment.

## 5. Discussion

- **Bayesian vs frequentist**: Bayesian models output full predictive uncertainty (HDI) and often improve calibration (Brier + calibration curve), which is valuable for clinical decision-making.
- **Logit vs probit**: We compare via WAIC/PSIS-LOO and predictive metrics; both are plausible, with differences typically small unless tails matter.
- **Limitations**: Dataset quality, missingness assumptions (MAR for MICE), and linear decision boundary; further work could include hierarchical priors or non-linear models.

## 6. Conclusion

We built Bayesian logistic and probit regression models for 10-year CHD risk prediction, emphasizing **uncertainty-aware prediction** and **probability calibration** under class imbalance. The workflow is reproducible via the provided notebooks.

## References

- Gelman, A., Jakulin, A., Pittau, M. G., & Su, Y.-S. (2008). *A weakly informative default prior distribution for logistic and other regression models.*
- Gelman et al., *Bayesian Data Analysis*.
- PyMC documentation.
- ArviZ documentation for diagnostics and PSIS-LOO/WAIC.
