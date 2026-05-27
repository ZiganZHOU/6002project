"""
Model evaluation utilities: metrics, calibration, and comparison plots.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    accuracy_score, classification_report,
    ConfusionMatrixDisplay,
)
from sklearn.calibration import calibration_curve


def classification_metrics(y_true, y_pred_prob, threshold: float = 0.5):
    y_pred = (y_pred_prob >= threshold).astype(int)
    print("Accuracy :", accuracy_score(y_true, y_pred))
    print("ROC-AUC  :", roc_auc_score(y_true, y_pred_prob))
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


def plot_posterior_coefs(idata, feature_names, model_name: str = "bayes_logistic"):
    """Forest plot of posterior beta coefficients."""
    import arviz as az
    ax = az.plot_forest(idata, var_names=["beta"], combined=True, figsize=(8, 6))
    plt.title(f"Posterior Coefficients — {model_name}")
    plt.tight_layout()
    return ax
