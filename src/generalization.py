"""Cross-dataset decision-calibration experiments.

The full Framingham workflow uses PyMC posterior samples. For auxiliary
datasets, this module uses bootstrap logistic models to produce risk-sample
ensembles quickly, then applies the same triage and Bayesian-optimization
policy layer. This keeps the generalization experiments reproducible without
rerunning expensive MCMC for every dataset.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.datasets import DEFAULT_GENERALIZATION_DATASETS, load_binary_dataset
from src.evaluate import (
    bayesian_optimize_triage_policy,
    binary_decision_metrics,
    expected_loss_threshold,
    metric_summary,
    posterior_expected_loss_triage,
    random_search_triage_policy,
    triage_metrics,
)


def run_generalization_experiments(
    data_dir: Path,
    output_dir: Path,
    dataset_names: Iterable[str] = DEFAULT_GENERALIZATION_DATASETS,
    n_bootstrap: int = 40,
    test_size: float = 0.35,
    policy_test_size: float = 0.5,
    c_fp: float = 1.0,
    c_fn: float = 5.0,
    eval_referral_cost: float = 0.75,
    abstention_penalty: float = 0.1,
    bo_initial: int = 8,
    bo_iter: int = 15,
    bo_candidates: int = 1000,
    random_state: int = 42,
    max_rows: int | None = None,
    use_transfer_initialization: bool = True,
    max_transfer_points: int = 3,
) -> pd.DataFrame:
    """Run lightweight cross-dataset BO triage benchmarks."""
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    metadata_rows = []
    transfer_policy_pool: list[np.ndarray] = []

    for dataset_name in dataset_names:
        dataset_dir = output_dir / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        X, y, metadata = load_binary_dataset(
            data_dir,
            dataset_name,
            random_state=random_state,
            max_rows=max_rows,
        )
        metadata_rows.append(metadata)
        _write_json(dataset_dir / "dataset_metadata.json", metadata)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )

        p_samples = bootstrap_logistic_risk_samples(
            X_train,
            y_train,
            X_test,
            n_bootstrap=n_bootstrap,
            random_state=random_state,
        )
        y_prob = p_samples.mean(axis=0)
        np.save(dataset_dir / "bootstrap_risk_samples.npy", p_samples)
        pd.DataFrame({"y_true": y_test, "mean_risk": y_prob}).to_csv(
            dataset_dir / "test_predictions.csv",
            index=False,
        )

        predictive_metrics = metric_summary(y_test, y_prob, threshold=0.5)
        cost_threshold_metrics = binary_decision_metrics(
            y_test,
            y_prob,
            c_fp=c_fp,
            c_fn=c_fn,
            threshold=expected_loss_threshold(c_fp, c_fn),
        )

        validation_idx, final_idx = split_policy_indices(
            y_test,
            policy_test_size=policy_test_size,
            random_state=random_state,
        )
        split_table = pd.DataFrame(
            {
                "test_index": np.arange(len(y_test)),
                "policy_split": np.where(
                    np.isin(np.arange(len(y_test)), validation_idx),
                    "validation",
                    "final",
                ),
            }
        )
        split_table.to_csv(dataset_dir / "policy_split_indices.csv", index=False)

        transfer_initial_points = (
            np.vstack(transfer_policy_pool[-max_transfer_points:])
            if use_transfer_initialization
            and max_transfer_points > 0
            and transfer_policy_pool
            else np.empty((0, 3))
        )
        pd.DataFrame(
            transfer_initial_points,
            columns=["q_low", "q_high", "decision_referral_cost"],
        ).to_csv(dataset_dir / "transfer_initial_points.csv", index=False)

        bo_best, bo_history = bayesian_optimize_triage_policy(
            p_samples[:, validation_idx],
            y_test[validation_idx],
            c_fp=c_fp,
            c_fn=c_fn,
            eval_c_ref=eval_referral_cost,
            abstention_penalty=abstention_penalty,
            n_initial=bo_initial,
            n_iter=bo_iter,
            random_state=random_state,
            n_candidates=bo_candidates,
            initial_points=transfer_initial_points,
        )
        bo_history.to_csv(dataset_dir / "bo_history.csv", index=False)

        bo_no_transfer_best, bo_no_transfer_history = bayesian_optimize_triage_policy(
            p_samples[:, validation_idx],
            y_test[validation_idx],
            c_fp=c_fp,
            c_fn=c_fn,
            eval_c_ref=eval_referral_cost,
            abstention_penalty=abstention_penalty,
            n_initial=bo_initial,
            n_iter=bo_iter,
            random_state=random_state,
            n_candidates=bo_candidates,
            initial_points=None,
        )
        bo_no_transfer_history.to_csv(
            dataset_dir / "bo_no_transfer_history.csv",
            index=False,
        )

        random_best, random_history = random_search_triage_policy(
            p_samples[:, validation_idx],
            y_test[validation_idx],
            c_fp=c_fp,
            c_fn=c_fn,
            eval_c_ref=eval_referral_cost,
            abstention_penalty=abstention_penalty,
            n_trials=bo_initial + bo_iter,
            random_state=random_state,
        )
        random_history.to_csv(dataset_dir / "random_search_history.csv", index=False)
        plot_search_convergence(
            {
                "BO transfer": bo_history,
                "BO no transfer": bo_no_transfer_history,
                "Random search": random_history,
            },
            dataset_dir / "search_convergence.png",
        )

        final_samples = p_samples[:, final_idx]
        final_mean = final_samples.mean(axis=0)
        final_y = y_test[final_idx]

        final_bo = evaluate_policy(
            final_samples,
            final_y,
            params=bo_best,
            c_fp=c_fp,
            c_fn=c_fn,
            eval_referral_cost=eval_referral_cost,
        )
        final_bo_no_transfer = evaluate_policy(
            final_samples,
            final_y,
            params=bo_no_transfer_best,
            c_fp=c_fp,
            c_fn=c_fn,
            eval_referral_cost=eval_referral_cost,
        )
        final_random = evaluate_policy(
            final_samples,
            final_y,
            params=random_best,
            c_fp=c_fp,
            c_fn=c_fn,
            eval_referral_cost=eval_referral_cost,
        )
        final_fixed = evaluate_fixed_policy(
            final_samples,
            final_y,
            c_fp=c_fp,
            c_fn=c_fn,
            eval_referral_cost=eval_referral_cost,
        )
        mean_only_samples = np.repeat(
            final_mean.reshape(1, -1),
            repeats=final_samples.shape[0],
            axis=0,
        )
        final_mean_only = evaluate_policy(
            mean_only_samples,
            final_y,
            params=bo_best,
            c_fp=c_fp,
            c_fn=c_fn,
            eval_referral_cost=eval_referral_cost,
        )
        final_no_refer = binary_decision_metrics(
            final_y,
            final_mean,
            c_fp=c_fp,
            c_fn=c_fn,
            threshold=expected_loss_threshold(c_fp, c_fn),
        )

        dataset_rows = [
            flatten_metrics(dataset_name, "predictive_at_0_5", predictive_metrics),
            flatten_metrics(
                dataset_name,
                "forced_binary_cost_threshold",
                cost_threshold_metrics,
            ),
            flatten_metrics(dataset_name, "fixed_triage", final_fixed),
            flatten_metrics(dataset_name, "bo_triage", final_bo),
            flatten_metrics(
                dataset_name,
                "bo_no_transfer_triage",
                final_bo_no_transfer,
            ),
            flatten_metrics(dataset_name, "random_search_triage", final_random),
            flatten_metrics(dataset_name, "mean_only_bo_triage", final_mean_only),
            flatten_metrics(dataset_name, "no_refer_final", final_no_refer),
        ]
        for row in dataset_rows:
            row.update(
                {
                    "n_rows": metadata["n_rows"],
                    "n_features": metadata["n_features"],
                    "positive_rate": metadata["positive_rate"],
                    "roc_auc": safe_auc(y_test, y_prob),
                    "bo_transfer_validation_objective": bo_best["objective"],
                    "bo_no_transfer_validation_objective": bo_no_transfer_best[
                        "objective"
                    ],
                    "random_validation_objective": random_best["objective"],
                    "n_transfer_initial_points": int(len(transfer_initial_points)),
                }
            )
        all_rows.extend(dataset_rows)

        _write_json(
            dataset_dir / "policy_best.json",
            {
                "bo_best": bo_best,
                "bo_no_transfer_best": bo_no_transfer_best,
                "random_search_best": random_best,
                "final_bo_metrics": final_bo,
                "final_bo_no_transfer_metrics": final_bo_no_transfer,
                "final_random_metrics": final_random,
                "final_fixed_metrics": final_fixed,
                "final_mean_only_metrics": final_mean_only,
                "final_no_refer_metrics": final_no_refer,
                "transfer_initial_points": transfer_initial_points.tolist(),
            },
        )
        transfer_policy_pool.append(policy_point_from_best(bo_best))

    summary = pd.DataFrame(all_rows)
    summary.to_csv(output_dir / "generalization_summary.csv", index=False)
    pd.DataFrame(metadata_rows).to_csv(output_dir / "dataset_summary.csv", index=False)
    plot_generalization_costs(summary, output_dir / "generalization_costs.png")
    plot_cost_referral_tradeoff(summary, output_dir / "cost_referral_tradeoff.png")
    return summary


def bootstrap_logistic_risk_samples(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    n_bootstrap: int,
    random_state: int,
) -> np.ndarray:
    """Fit bootstrap logistic models and return probability samples."""
    rng = np.random.default_rng(random_state)
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = np.asarray(y_train).astype(int)
    n_train = len(y_train)
    samples = []

    for _ in range(n_bootstrap):
        for _attempt in range(100):
            idx = rng.integers(0, n_train, size=n_train)
            if np.unique(y_train[idx]).size == 2:
                break
        else:
            raise ValueError("Could not draw a bootstrap sample with both classes.")

        model = make_logistic_pipeline()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(X_train.iloc[idx], y_train[idx])
        samples.append(model.predict_proba(X_test)[:, 1])

    return np.vstack(samples)


def make_logistic_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(max_iter=2000, solver="lbfgs"),
            ),
        ]
    )


def split_policy_indices(
    y_test: np.ndarray,
    policy_test_size: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(len(y_test))
    stratify = y_test if np.min(np.bincount(np.asarray(y_test).astype(int))) >= 2 else None
    validation_idx, final_idx = train_test_split(
        indices,
        test_size=policy_test_size,
        random_state=random_state,
        stratify=stratify,
    )
    return validation_idx, final_idx


def evaluate_policy(
    p_samples: np.ndarray,
    y_true: np.ndarray,
    params: dict,
    c_fp: float,
    c_fn: float,
    eval_referral_cost: float,
) -> dict:
    triage = posterior_expected_loss_triage(
        p_samples,
        c_fp=c_fp,
        c_fn=c_fn,
        c_ref=float(params["decision_referral_cost"]),
        q_low=float(params["q_low"]),
        q_high=float(params["q_high"]),
    )
    return triage_metrics(
        y_true,
        triage,
        c_fp=c_fp,
        c_fn=c_fn,
        c_ref=eval_referral_cost,
    )


def evaluate_fixed_policy(
    p_samples: np.ndarray,
    y_true: np.ndarray,
    c_fp: float,
    c_fn: float,
    eval_referral_cost: float,
) -> dict:
    triage = posterior_expected_loss_triage(
        p_samples,
        c_fp=c_fp,
        c_fn=c_fn,
        c_ref=eval_referral_cost,
        q_low=0.10,
        q_high=0.90,
    )
    return triage_metrics(
        y_true,
        triage,
        c_fp=c_fp,
        c_fn=c_fn,
        c_ref=eval_referral_cost,
    )


def flatten_metrics(dataset: str, method: str, metrics: dict) -> dict:
    row = {"dataset": dataset, "method": method}
    for key, value in metrics.items():
        if isinstance(value, (np.integer, np.floating)):
            value = value.item()
        row[key] = value
    return row


def safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def policy_point_from_best(best: dict) -> np.ndarray:
    """Convert a best-policy record into a 1 x 3 transfer-initialization row."""
    return np.array(
        [
            [
                float(best["q_low"]),
                float(best["q_high"]),
                float(best["decision_referral_cost"]),
            ]
        ],
        dtype=float,
    )


def plot_search_convergence(histories: dict[str, pd.DataFrame], path: Path) -> None:
    """Plot best-so-far objective for BO and random-search histories."""
    fig, ax = plt.subplots(figsize=(7, 4))
    plotted = False
    for label, history in histories.items():
        if history.empty or "objective" not in history:
            continue
        best_so_far = history["objective"].cummin()
        ax.plot(history["iteration"], best_so_far, marker="o", markersize=3, label=label)
        plotted = True

    if not plotted:
        plt.close(fig)
        return

    ax.set_xlabel("Policy evaluations")
    ax.set_ylabel("Best validation objective so far")
    ax.set_title("Policy Search Convergence")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_generalization_costs(summary: pd.DataFrame, path: Path) -> None:
    plot_methods = [
        "fixed_triage",
        "bo_triage",
        "bo_no_transfer_triage",
        "random_search_triage",
        "mean_only_bo_triage",
        "no_refer_final",
    ]
    plot_df = summary[
        summary["method"].isin(plot_methods) & summary["average_cost"].notna()
    ].copy()
    if plot_df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    pivot = plot_df.pivot(index="dataset", columns="method", values="average_cost")
    pivot[plot_methods].plot(kind="bar", ax=ax)
    ax.set_ylabel("Average cost on final policy split")
    ax.set_xlabel("Dataset")
    ax.set_title("Generalization and Ablation: Triage Decision Cost")
    ax.legend(title="Method", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_cost_referral_tradeoff(summary: pd.DataFrame, path: Path) -> None:
    """Plot final policy average cost against referral rate."""
    plot_methods = [
        "fixed_triage",
        "bo_triage",
        "bo_no_transfer_triage",
        "random_search_triage",
        "mean_only_bo_triage",
    ]
    plot_df = summary[
        summary["method"].isin(plot_methods)
        & summary["average_cost"].notna()
        & summary["referral_rate"].notna()
    ].copy()
    if plot_df.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for method, group in plot_df.groupby("method"):
        ax.scatter(
            group["referral_rate"],
            group["average_cost"],
            label=method,
            s=45,
            alpha=0.85,
        )
        for _, row in group.iterrows():
            ax.annotate(
                row["dataset"],
                (row["referral_rate"], row["average_cost"]),
                fontsize=7,
                alpha=0.8,
                xytext=(3, 3),
                textcoords="offset points",
            )

    ax.set_xlabel("Referral rate")
    ax.set_ylabel("Average cost on final policy split")
    ax.set_title("Cost-Referral Trade-off")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def _json_default(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
