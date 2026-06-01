"""
Model evaluation utilities: metrics, calibration, and comparison plots.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    accuracy_score, classification_report,
    precision_recall_curve, average_precision_score,
    brier_score_loss,
    confusion_matrix,
)
from sklearn.calibration import calibration_curve


def metric_summary(y_true, y_pred_prob, threshold: float = 0.5) -> dict:
    """Return thresholded and probability-based metrics in a reusable format."""
    y_pred = (y_pred_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else np.nan
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    precision = tp / (tp + fp) if (tp + fp) else np.nan

    return {
        "roc_auc": roc_auc_score(y_true, y_pred_prob),
        "pr_auc": average_precision_score(y_true, y_pred_prob),
        "brier": brier_score_loss(y_true, y_pred_prob),
        "accuracy": accuracy_score(y_true, y_pred),
        "threshold": threshold,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
    }


def classification_metrics(y_true, y_pred_prob, threshold: float = 0.5):
    summary = metric_summary(y_true, y_pred_prob, threshold=threshold)
    y_pred = (y_pred_prob >= threshold).astype(int)
    print("ROC-AUC  :", summary["roc_auc"])
    print("PR-AUC   :", summary["pr_auc"])
    print("Brier    :", summary["brier"])
    print("Accuracy :", summary["accuracy"])
    print(
        "Confusion:",
        f"TN={summary['tn']} FP={summary['fp']} FN={summary['fn']} TP={summary['tp']}",
    )
    print(classification_report(y_true, y_pred))


def plot_roc_curves(models: dict, y_true, ax=None):
    """
    models: dict of {name: y_prob_array}
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    for name, y_prob in models.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve Comparison")
    ax.legend()
    plt.tight_layout()
    return ax


def plot_calibration(models: dict, y_true, n_bins: int = 10, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    for name, y_prob in models.items():
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_prob, n_bins=n_bins
        )
        ax.plot(mean_predicted_value, fraction_of_positives, "s-", label=name)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration Curves")
    ax.legend()
    plt.tight_layout()
    return ax


def plot_pr_curves(models: dict, y_true, ax=None):
    """
    Precision-Recall curves (better than ROC when classes are imbalanced).
    models: dict of {name: y_prob_array}
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    for name, y_prob in models.items():
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        ax.plot(recall, precision, label=f"{name} (AP={ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve Comparison")
    ax.legend()
    plt.tight_layout()
    return ax


def logistic_odds_ratio_summary(
    idata,
    feature_names: list[str],
    hdi_prob: float = 0.95,
):
    """
    Compute Bayesian odds ratios (OR = exp(beta)) summary for logistic regression.

    Returns a pandas DataFrame with median OR and HDI bounds per feature.
    """
    import pandas as pd
    import arviz as az

    beta = idata.posterior["beta"]
    feature_dim = beta.dims[-1]
    if len(feature_names) != beta.sizes[feature_dim]:
        raise ValueError("feature_names length must match posterior beta features.")

    or_samples = np.exp(beta)
    hdi = az.hdi(or_samples, hdi_prob=hdi_prob)["beta"]
    med = or_samples.median(dim=("chain", "draw"))

    out = pd.DataFrame(
        {
            "feature": feature_names,
            "or_median": med.values,
            f"or_hdi_{int(hdi_prob*100)}_low": hdi.sel(hdi="lower").values,
            f"or_hdi_{int(hdi_prob*100)}_high": hdi.sel(hdi="higher").values,
        }
    )
    out["abs_log_or_median"] = np.abs(np.log(out["or_median"]))
    out = out.sort_values("abs_log_or_median", ascending=False).reset_index(drop=True)
    return out


def expected_loss_threshold(c_fp: float, c_fn: float) -> float:
    """Bayes-optimal threshold for false-positive and false-negative costs."""
    if c_fp <= 0 or c_fn <= 0:
        raise ValueError("Costs must be positive.")
    return c_fp / (c_fp + c_fn)


def threshold_cost_curve(y_true, y_pred_prob, c_fp: float = 1.0, c_fn: float = 5.0):
    """Compute observed test-set cost across thresholds."""
    import pandas as pd

    thresholds = np.linspace(0.01, 0.99, 99)
    rows = []
    for threshold in thresholds:
        y_pred = (y_pred_prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        rows.append(
            {
                "threshold": threshold,
                "cost": c_fp * fp + c_fn * fn,
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            }
        )
    return pd.DataFrame(rows)


def posterior_expected_loss_triage(
    p_samples,
    c_fp: float = 1.0,
    c_fn: float = 5.0,
    c_ref: float = 0.75,
    q_low: float = 0.1,
    q_high: float = 0.9,
):
    """
    Bayesian expected-loss triage with a reject/referral option.

    Actions:
    - low: no intervention
    - high: intervention/follow-up
    - refer: posterior uncertainty or expected loss is high enough that
      additional testing is preferred.

    The rule first minimizes posterior expected loss over low/high/refer, then
    requires posterior confidence around the cost-based threshold before making
    a low/high decision. Ambiguous cases are referred.
    """
    import pandas as pd

    if not (0 <= q_low < q_high <= 1):
        raise ValueError("Require 0 <= q_low < q_high <= 1.")
    if c_ref < 0:
        raise ValueError("Referral cost must be nonnegative.")

    p_samples = np.asarray(p_samples)
    if p_samples.ndim != 2:
        raise ValueError("p_samples must have shape (posterior_samples, n_patients).")

    threshold = expected_loss_threshold(c_fp, c_fn)
    mean_risk = p_samples.mean(axis=0)
    prob_above_threshold = (p_samples > threshold).mean(axis=0)

    loss_low = c_fn * mean_risk
    loss_high = c_fp * (1 - mean_risk)
    loss_refer = np.full_like(mean_risk, c_ref, dtype=float)
    losses = np.vstack([loss_low, loss_high, loss_refer])
    raw_idx = losses.argmin(axis=0)
    raw_action = np.array(["low", "high", "refer"], dtype=object)[raw_idx]

    action = raw_action.copy()
    action[(raw_action == "low") & (prob_above_threshold > q_low)] = "refer"
    action[(raw_action == "high") & (prob_above_threshold < q_high)] = "refer"

    return pd.DataFrame(
        {
            "mean_risk": mean_risk,
            "risk_threshold": threshold,
            "prob_above_threshold": prob_above_threshold,
            "loss_low": loss_low,
            "loss_high": loss_high,
            "loss_refer": loss_refer,
            "raw_expected_loss_action": raw_action,
            "action": action,
        }
    )


def triage_metrics(
    y_true,
    triage_table,
    c_fp: float = 1.0,
    c_fn: float = 5.0,
    c_ref: float = 0.75,
):
    """Summarize clinical utility and decided-case accuracy for triage actions."""
    y_true = np.asarray(y_true).astype(int)
    actions = np.asarray(triage_table["action"], dtype=object)
    if len(actions) != len(y_true):
        raise ValueError("triage_table and y_true must have the same length.")

    refer = actions == "refer"
    high = actions == "high"
    low = actions == "low"
    decided = ~refer

    costs = np.zeros(len(y_true), dtype=float)
    costs[refer] = c_ref
    costs[high & (y_true == 0)] = c_fp
    costs[low & (y_true == 1)] = c_fn

    if decided.any():
        y_pred_decided = high[decided].astype(int)
        tn, fp, fn, tp = confusion_matrix(
            y_true[decided],
            y_pred_decided,
            labels=[0, 1],
        ).ravel()
        decided_accuracy = accuracy_score(y_true[decided], y_pred_decided)
    else:
        tn = fp = fn = tp = 0
        decided_accuracy = np.nan

    return {
        "n": int(len(y_true)),
        "n_low": int(low.sum()),
        "n_high": int(high.sum()),
        "n_refer": int(refer.sum()),
        "coverage": float(decided.mean()),
        "referral_rate": float(refer.mean()),
        "total_cost": float(costs.sum()),
        "average_cost": float(costs.mean()),
        "decided_accuracy": float(decided_accuracy)
        if not np.isnan(decided_accuracy)
        else np.nan,
        "decided_tn": int(tn),
        "decided_fp": int(fp),
        "decided_fn": int(fn),
        "decided_tp": int(tp),
    }


def bayesian_optimize_triage_policy(
    p_samples,
    y_true,
    c_fp: float = 1.0,
    c_fn: float = 5.0,
    eval_c_ref: float = 0.75,
    abstention_penalty: float = 0.1,
    n_initial: int = 12,
    n_iter: int = 30,
    random_state: int = 42,
    n_candidates: int = 2000,
):
    """
    Calibrate triage-policy parameters with Gaussian-process Bayesian optimization.

    Optimized parameters are q_low, q_high, and the internal referral cost used
    by the triage rule. The objective is validation-set clinical cost plus an
    optional abstention-rate penalty. The objective is a gray-box policy score:
    its cost structure is explicit, but the realized value depends on discrete
    low/high/refer decisions over held-out patients. MCMC is not repeated;
    optimization uses saved posterior risk samples.
    """
    import pandas as pd
    import warnings
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
    from sklearn.exceptions import ConvergenceWarning

    rng = np.random.default_rng(random_state)
    p_samples = np.asarray(p_samples)
    y_true = np.asarray(y_true).astype(int)

    bounds = np.array(
        [
            [0.01, 0.40],  # q_low
            [0.60, 0.99],  # q_high
            [0.20, 2.00],  # decision referral cost
        ],
        dtype=float,
    )

    def sample_params(n_rows: int) -> np.ndarray:
        unit = rng.random((n_rows, bounds.shape[0]))
        return bounds[:, 0] + unit * (bounds[:, 1] - bounds[:, 0])

    def objective(theta: np.ndarray) -> tuple[float, dict]:
        q_low, q_high, c_ref_decision = theta
        triage = posterior_expected_loss_triage(
            p_samples,
            c_fp=c_fp,
            c_fn=c_fn,
            c_ref=c_ref_decision,
            q_low=q_low,
            q_high=q_high,
        )
        metrics = triage_metrics(
            y_true,
            triage,
            c_fp=c_fp,
            c_fn=c_fn,
            c_ref=eval_c_ref,
        )
        score = metrics["average_cost"] + abstention_penalty * metrics["referral_rate"]
        return float(score), metrics

    X_obs = sample_params(n_initial)
    rows = []
    y_obs = []
    for iteration, theta in enumerate(X_obs):
        score, metrics = objective(theta)
        y_obs.append(score)
        rows.append(
            {
                "iteration": iteration,
                "q_low": theta[0],
                "q_high": theta[1],
                "decision_referral_cost": theta[2],
                "objective": score,
                **metrics,
            }
        )

    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3))
        * Matern(
            length_scale=np.ones(bounds.shape[0]),
            length_scale_bounds=(1e-3, 1e3),
            nu=2.5,
        )
        + WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-9, 1e-1))
    )
    for step in range(n_iter):
        gp = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            random_state=random_state,
            n_restarts_optimizer=2,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            gp.fit(np.asarray(X_obs), np.asarray(y_obs))

        candidates = sample_params(n_candidates)
        mean, std = gp.predict(candidates, return_std=True)
        acquisition = mean - 1.96 * std
        theta = candidates[int(np.argmin(acquisition))]

        score, metrics = objective(theta)
        X_obs = np.vstack([X_obs, theta])
        y_obs.append(score)
        rows.append(
            {
                "iteration": n_initial + step,
                "q_low": theta[0],
                "q_high": theta[1],
                "decision_referral_cost": theta[2],
                "objective": score,
                **metrics,
            }
        )

    history = pd.DataFrame(rows)
    best = history.loc[history["objective"].idxmin()].to_dict()
    return best, history


def plot_posterior_coefs(idata, feature_names, model_name: str = "bayes_logistic"):
    """Forest plot of posterior beta coefficients."""
    import arviz as az

    plot_idata = idata
    if feature_names is not None:
        beta_dim = idata.posterior["beta"].dims[-1]
        if len(feature_names) == idata.posterior.sizes[beta_dim]:
            plot_idata = idata.copy()
            plot_idata.posterior = plot_idata.posterior.assign_coords(
                {beta_dim: feature_names}
            )

    ax = az.plot_forest(plot_idata, var_names=["beta"], combined=True, figsize=(8, 6))
    plt.title(f"Posterior Coefficients - {model_name}")
    plt.tight_layout()
    return ax
