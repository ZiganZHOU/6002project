# Report: Generalizable Bayesian Optimization for Medical Risk Triage

## 1. Project Aim

This project studies uncertainty-aware medical risk prediction and decision
calibration. The original task was cardiovascular disease diagnosis using the
Framingham Heart Study dataset. The revised project keeps that task as the main
Bayesian case study, but reframes the contribution as a more general framework:

1. estimate patient-level risk with a probabilistic model;
2. represent uncertainty using posterior or bootstrap risk samples;
3. convert risk into `low`, `high`, or `refer` actions through an expected-loss
   triage policy;
4. calibrate that triage policy with Bayesian optimization;
5. evaluate whether the same decision layer transfers across multiple binary
   medical datasets.

The core claim is therefore not that one cardiovascular model solves every
medical diagnosis problem. The claim is that the Bayesian optimization layer can
serve as a reusable decision-calibration mechanism once a task produces risk
probabilities and uncertainty samples.

## 2. Data

The main Bayesian case study uses the Framingham Heart Study dataset:

| Quantity | Value |
|---|---:|
| Patients | 4,240 |
| Predictors | 15 |
| Negative class | 3,596 |
| Positive class | 644 |
| Positive rate | 15.2% |
| Largest missingness | `glucose`, 388 missing values |

The generalization and ablation experiments use additional binary medical
datasets:

| Dataset | Task | Role |
|---|---|---|
| Breast Cancer Wisconsin Diagnostic | malignant vs benign tumor | cross-disease generalization |
| Mammographic Mass | malignant vs benign mammographic mass | triage-style generalization |
| UCI Cleveland Heart Disease | heart disease present vs absent | same-domain generalization |
| CDC Diabetes Health Indicators | diabetes/prediabetes vs healthy | large-data robustness |

The original UCI/Kaggle files are preserved under `data/raw_external/`. Processed
CSV files add explicit column names and a unified binary `target` column so that
the same pipeline can process all auxiliary datasets.

## 3. Preprocessing

All experiments use leakage-safe preprocessing:

1. Split raw data before fitting imputers or scalers.
2. Fit the imputer on the training set only.
3. Transform validation/test sets using the fitted imputer.
4. Fit the scaler on the training set only.
5. Transform validation/test sets using the fitted scaler.

For Framingham, the project also restores simple clinical constraints after
imputation: binary variables are clipped to 0/1, education is clipped to its
ordinal range, and physiologic values are clipped to nonnegative values.

The imputation is best described as MICE-like single imputation. It is not a
full multiple-imputation Bayesian analysis.

## 4. Main Bayesian Models

For the Framingham case study, the project fits:

1. frequentist logistic regression as a sanity-check baseline;
2. Bayesian logistic regression;
3. Bayesian probit regression;
4. grouped hierarchical Bayesian logistic regression.

The Bayesian logistic model is:

\[
y_i \sim \mathrm{Bernoulli}(p_i), \quad
\mathrm{logit}(p_i) = \alpha + x_i^T\beta
\]

\[
\alpha \sim \mathcal{N}(0,1), \quad
\beta_j \sim \mathcal{N}(0,2.5)
\]

The probit model replaces the logistic link with the normal CDF:

\[
p_i = \Phi(\alpha + x_i^T\beta)
\]

The hierarchical logistic model assigns predictors to clinical groups and
learns a shrinkage scale for each group:

\[
\beta_j \sim \mathcal{N}(0,\tau_{g(j)}^2), \quad
\tau_g \sim \mathrm{HalfNormal}(1)
\]

This model lets the posterior decide which feature families require stronger or
weaker regularization.

## 5. Decision Layer

Medical screening often treats false negatives as more costly than false
positives. If \(C_{FP}\) is the cost of unnecessary follow-up and \(C_{FN}\) is
the cost of missing a case, the cost-sensitive threshold is:

\[
p^* = \frac{C_{FP}}{C_{FP}+C_{FN}}
\]

With \(C_{FP}=1\) and \(C_{FN}=5\), the threshold is \(p^*=0.167\), much lower
than the default 0.5 threshold.

The project then extends forced binary classification into a three-action
triage rule:

- `low`: low risk / no immediate intervention;
- `high`: high risk / intervention or follow-up;
- `refer`: uncertain case requiring further testing.

For patient \(i\), posterior or bootstrap risk samples are:

\[
p_i^{(1)}, p_i^{(2)}, \ldots, p_i^{(S)}
\]

The expected losses are:

\[
L_i(low)=C_{FN}\mathbb{E}[p_i]
\]

\[
L_i(high)=C_{FP}\mathbb{E}[1-p_i]
\]

\[
L_i(refer)=C_{REF}
\]

The policy also checks uncertainty around \(p^*\):

\[
\Pr(p_i > p^*) \le q_{low} \Rightarrow low
\]

\[
\Pr(p_i > p^*) \ge q_{high} \Rightarrow high
\]

Otherwise, the patient is referred.

## 6. Bayesian Optimization

The optimized policy parameters are:

| Parameter | Meaning |
|---|---|
| \(q_{low}\) | confidence cutoff for low-risk decisions |
| \(q_{high}\) | confidence cutoff for high-risk decisions |
| \(C_{REF}^{decision}\) | internal referral cost used during policy selection |

The validation objective is:

\[
\text{objective} =
\text{average clinical cost}
+ \lambda \cdot \text{referral rate}
\]

The optimizer fits a Gaussian-process surrogate over this policy space. It uses
a dynamic lower-confidence-bound acquisition rule: the exploration weight is
larger early in the search and smaller later in the search. This connects the
implementation to the proposal's idea of dynamically balancing exploration and
exploitation.

This optimization is intentionally placed downstream of risk modeling. MCMC does
not need to be rerun for every candidate policy; the optimizer reuses saved risk
samples and searches only over decision parameters.

The revised implementation also supports cross-task transfer initialization.
After optimizing one dataset, its best policy parameters can be inserted into
the initial candidate pool for the next dataset. This directly operationalizes
the proposal's idea of using prior optimization knowledge from similar tasks
while keeping the search budget fixed.

## 7. Generalization and Ablation Experiments

The new command:

```powershell
python run_project.py generalize
```

runs the cross-dataset experiment. The alias:

```powershell
python run_project.py ablate
```

runs the same workflow, because generalization and ablation outputs are produced
together.

For auxiliary datasets, the project uses bootstrap logistic models to generate
risk-sample ensembles. This is a lightweight uncertainty proxy used only for
generalization checks. The full Bayesian posterior analysis remains the
Framingham PyMC workflow.

The generated outputs include:

| Output | Meaning |
|---|---|
| `outputs/generalization/dataset_summary.csv` | sample size, feature count, class balance, missingness |
| `outputs/generalization/generalization_summary.csv` | main comparison table across datasets and methods |
| `outputs/generalization/<dataset>/bo_history.csv` | BO search history |
| `outputs/generalization/<dataset>/bo_no_transfer_history.csv` | BO history without transfer initialization |
| `outputs/generalization/<dataset>/random_search_history.csv` | random-search baseline history |
| `outputs/generalization/<dataset>/transfer_initial_points.csv` | transferred policy points used to seed BO |
| `outputs/generalization/<dataset>/policy_best.json` | best policy and final split metrics |
| `outputs/generalization/<dataset>/search_convergence.png` | best-so-far convergence curve for BO and random search |
| `outputs/generalization/generalization_costs.png` | decision-cost comparison plot |
| `outputs/generalization/cost_referral_tradeoff.png` | cost-referral trade-off plot |

The ablations compare:

| Method | Purpose |
|---|---|
| fixed triage | tests whether tuning is useful |
| BO triage | proposed optimized policy with transfer initialization when available |
| BO no-transfer triage | tests whether transferred prior policies help |
| random-search triage | tests whether BO is better than uninformed search |
| mean-only BO triage | tests whether uncertainty samples matter |
| no-refer final | tests whether the reject option matters |
| forced binary cost threshold | tests ordinary low/high decision making |

## 8. Existing Framingham Results

The full Framingham workflow was previously run successfully and produced the
following held-out test performance at the default 0.5 threshold:

| Model | ROC-AUC | PR-AUC | Brier | Accuracy | Sensitivity | Specificity | Precision |
|---|---:|---:|---:|---:|---:|---:|---:|
| Frequentist Logistic | 0.701 | 0.294 | 0.122 | 84.43% | 5.43% | 98.61% | 41.18% |
| Bayesian Logistic | 0.701 | 0.294 | 0.122 | 84.43% | 5.43% | 98.61% | 41.18% |
| Bayesian Probit | 0.673 | 0.267 | 0.133 | 82.43% | 10.08% | 95.41% | 28.26% |
| Hierarchical Logistic | 0.700 | 0.296 | 0.121 | 84.79% | 4.65% | 99.17% | 50.00% |

Accuracy is not the main success measure because the positive class is only
about 15.2%. PR-AUC, Brier score, calibration, sensitivity, and cost-sensitive
decision metrics are more important.

Using \(C_{FP}=1\) and \(C_{FN}=5\), the hierarchical logistic model reached the
following cost-threshold performance:

| Metric | Value |
|---|---:|
| Accuracy | 69.81% |
| Sensitivity | 58.91% |
| Specificity | 71.77% |
| Precision | 27.24% |
| False positives | 203 |
| False negatives | 53 |
| Total observed cost | 468 |

The optimized three-action triage policy classified about 73.82% of final-split
patients directly and referred 26.18% for additional testing. Its average cost
on the final split was 0.538.

## 9. How to Analyze New Results

After running:

```powershell
python run_project.py generalize
```

the main table is `outputs/generalization/generalization_summary.csv`.

The model is accurate if:

- ROC-AUC and PR-AUC are clearly above baseline;
- Brier score is low;
- calibration is reasonable;
- sensitivity improves under cost-sensitive thresholds.

The decision policy is useful if:

- BO triage has lower average cost than fixed triage;
- transfer-initialized BO improves or accelerates BO compared with
  no-transfer BO on later datasets;
- BO triage is better than or more sample-efficient than random search;
- no-refer forced binary decisions create more false negatives or higher cost;
- mean-only triage performs worse than risk-sample triage when uncertainty is
  important.

The BO search itself should be inspected through each dataset's
`search_convergence.png`: the best-so-far objective should decrease quickly and
ideally match or beat random search under the same policy-evaluation budget. The
cost-referral trade-off plot should be used to check whether a lower cost is
achieved by a clinically reasonable referral rate rather than by referring
nearly all cases.

The method generalizes if:

- the same policy optimizer runs without dataset-specific hand tuning;
- BO triage remains competitive across breast cancer, mammographic mass,
  Cleveland heart disease, and CDC diabetes;
- the improvement is not restricted to one dataset.

The method is robust if results remain stable under:

- different random seeds;
- different train/test splits;
- different false-negative costs;
- different referral costs;
- different BO budgets;
- different bootstrap sample counts;
- different row caps for the CDC dataset.

## 10. Limitations

- The Framingham dataset is observational and should not be interpreted
  causally.
- Framingham predicts 10-year CHD risk, not immediate clinical diagnosis.
- Auxiliary datasets use bootstrap logistic uncertainty rather than full MCMC.
- The current triage costs are illustrative, not clinically validated.
- The BO layer calibrates decision policy, not the Bayesian model priors.
- The project is educational and is not suitable for clinical deployment.

## 11. Conclusion

The revised project is best described as a generalizable Bayesian decision
calibration framework for medical risk triage. Framingham provides the main
Bayesian case study with posterior uncertainty and interpretable clinical
effects. The new generalization workflow tests whether the same BO-calibrated
triage layer can transfer to other binary medical datasets, and whether prior
optimized policies can accelerate later BO searches through transfer
initialization.

This structure better matches the proposal's broader ambition while keeping the
implementation realistic: the project demonstrates practical generalization
through multiple datasets and ablation experiments, rather than claiming a
large-scale AutoML system or a formal mathematical proof of universality.

## References

- Framingham Heart Study dataset.
- UCI Machine Learning Repository: Breast Cancer Wisconsin Diagnostic.
- UCI Machine Learning Repository: Mammographic Mass.
- UCI Machine Learning Repository: Heart Disease Cleveland.
- UCI Machine Learning Repository: CDC Diabetes Health Indicators.
- PyMC documentation.
- ArviZ documentation for diagnostics, WAIC, and PSIS-LOO.
- Gelman, Jakulin, Pittau, and Su (2008), weakly informative priors for logistic
  regression.
