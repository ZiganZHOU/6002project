"""
Model evaluation utilities: metrics, calibration, and comparison plots.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    accuracy_score, classification_report,
    ConfusionMatrixDisplay,
    precision_recall_curve, average_precision_score,
    brier_score_loss,
)
from sklearn.calibration import calibration_curve


def classification_metrics(y_true, y_pred_prob, threshold: float = 0.5):
    y_pred = (y_pred_prob >= threshold).astype(int)
    print("ROC-AUC  :", roc_auc_score(y_true, y_pred_prob))
    print("PR-AUC   :", average_precision_score(y_true, y_pred_prob))
    print("Brier    :", brier_score_loss(y_true, y_pred_prob))
    print("Accuracy :", accuracy_score(y_true, y_pred))
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

    beta = idata.posterior["beta"].values  # (chain, draw, k)
    beta = beta.reshape(-1, beta.shape[-1])  # (samples, k)
    or_samples = np.exp(beta)  # (samples, k)

    hdi = az.hdi(or_samples, hdi_prob=hdi_prob)  # (k, 2)
    med = np.median(or_samples, axis=0)

    out = pd.DataFrame(
        {
            "feature": feature_names,
            "or_median": med,
            f"or_hdi_{int(hdi_prob*100)}_low": hdi[:, 0],
            f"or_hdi_{int(hdi_prob*100)}_high": hdi[:, 1],
        }
    )
    out["abs_log_or_median"] = np.abs(np.log(out["or_median"]))
    out = out.sort_values("abs_log_or_median", ascending=False).reset_index(drop=True)
    return out


def plot_posterior_coefs(idata, feature_names, model_name: str = "bayes_logistic"):
    """Forest plot of posterior beta coefficients."""
    import arviz as az
    ax = az.plot_forest(idata, var_names=["beta"], combined=True, figsize=(8, 6))
    plt.title(f"Posterior Coefficients — {model_name}")
    plt.tight_layout()
    return ax
