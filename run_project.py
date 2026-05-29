"""
End-to-end and stepwise runner for the Bayesian CHD diagnosis project.

Examples
--------
Run the complete workflow, including Bayesian sampling:
    python run_project.py all

Run only the quick, non-Bayesian checks:
    python run_project.py all --skip-bayes

Run a specific step:
    python run_project.py eda
    python run_project.py baseline
    python run_project.py bayes --draws 1000 --tune 1000 --chains 4
    python run_project.py evaluate
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.evaluate import (
    expected_loss_threshold,
    logistic_odds_ratio_summary,
    metric_summary,
    plot_calibration,
    plot_pr_curves,
    plot_roc_curves,
    threshold_cost_curve,
)
from src.evaluate import plot_posterior_coefs
from src.model import (
    build_bayesian_logistic,
    build_bayesian_probit,
    posterior_predict,
    sample_model,
)
from src.preprocess import (
    RANDOM_STATE,
    TARGET_COL,
    compute_vif,
    full_pipeline,
    load_data,
    make_imputer,
    postprocess_imputed_features,
    summarize_missing,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "data" / "framingham.csv"
DEFAULT_OUTPUTS = ROOT / "outputs"


def clean_number(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: clean_number(val) for key, val in value.items()}
    if isinstance(value, list):
        return [clean_number(item) for item in value]
    if isinstance(value, tuple):
        return [clean_number(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return clean_number(value.tolist())
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean_number(payload), indent=2), encoding="utf-8")


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = ROOT / path
    return path


def ensure_data(data_path: Path) -> None:
    if not data_path.exists():
        raise FileNotFoundError(
            f"Missing dataset: {data_path}. Download framingham.csv and place it in data/."
        )


def prepare_train_imputed(data_path: Path) -> pd.DataFrame:
    df = load_data(str(data_path))
    X_raw = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].values
    X_train_raw, _, _, _ = train_test_split(
        X_raw,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    imputer = make_imputer(sample_posterior=False)
    X_train_imp = pd.DataFrame(
        imputer.fit_transform(X_train_raw),
        columns=X_raw.columns,
        index=X_train_raw.index,
    )
    return postprocess_imputed_features(X_train_imp)


def run_eda(data_path: Path, output_dir: Path) -> dict[str, Any]:
    print("[eda] Loading data and writing exploratory outputs.")
    ensure_data(data_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(str(data_path))
    class_counts = df[TARGET_COL].value_counts().sort_index()
    missing = summarize_missing(df)

    class_counts.to_csv(output_dir / "class_balance.csv", header=["count"])
    missing.to_csv(output_dir / "missing_values.csv", header=["missing_count"])
    df.describe().T.to_csv(output_dir / "raw_feature_summary.csv")

    fig, ax = plt.subplots(figsize=(6, 4))
    class_counts.plot(kind="bar", ax=ax, color=["steelblue", "tomato"])
    ax.set_title("Target Class Distribution")
    ax.set_xlabel("TenYearCHD")
    ax.set_ylabel("Count")
    ax.set_xticklabels(["No CHD", "CHD"], rotation=0)
    ax.text(
        0.5,
        0.9,
        f"Positive rate = {df[TARGET_COL].mean():.3f}",
        transform=ax.transAxes,
        ha="center",
        va="top",
    )
    fig.tight_layout()
    fig.savefig(output_dir / "class_balance.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for label, group in df.groupby(TARGET_COL):
        group["age"].plot(kind="hist", bins=20, alpha=0.55, ax=ax, label=str(label))
    ax.set_title("Age Distribution by CHD Label")
    ax.set_xlabel("Age")
    ax.legend(title=TARGET_COL)
    fig.tight_layout()
    fig.savefig(output_dir / "age_by_target.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Feature Correlation Matrix")
    fig.tight_layout()
    fig.savefig(output_dir / "correlation_heatmap.png", dpi=180)
    plt.close(fig)

    X_train_imp = prepare_train_imputed(data_path)
    vif = compute_vif(X_train_imp)
    vif.to_csv(output_dir / "vif_train_only.csv", index=False)

    summary = {
        "shape": list(df.shape),
        "positive_rate": float(df[TARGET_COL].mean()),
        "class_counts": class_counts.to_dict(),
        "missing_values": missing.to_dict(),
        "top_vif": vif.head(5).to_dict(orient="records"),
    }
    write_json(output_dir / "eda_summary.json", summary)
    print("[eda] Done. Outputs written to", output_dir)
    return summary


def run_preprocess(data_path: Path, output_dir: Path) -> dict[str, Any]:
    print("[preprocess] Running leakage-safe preprocessing checks.")
    ensure_data(data_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test, _, _, feature_names = full_pipeline(str(data_path))
    train_df = pd.DataFrame(X_train, columns=feature_names)
    train_df.describe().T.to_csv(output_dir / "scaled_train_summary.csv")

    summary = {
        "train_shape": list(X_train.shape),
        "test_shape": list(X_test.shape),
        "train_positive_rate": float(np.mean(y_train)),
        "test_positive_rate": float(np.mean(y_test)),
        "x_train_has_nan": bool(np.isnan(X_train).any()),
        "x_test_has_nan": bool(np.isnan(X_test).any()),
        "feature_names": feature_names,
    }
    write_json(output_dir / "preprocess_summary.json", summary)
    print("[preprocess] Done.")
    return summary


def fit_baseline(data_path: Path):
    X_train, X_test, y_train, y_test, _, _, feature_names = full_pipeline(str(data_path))
    model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    return model, y_prob, X_train, X_test, y_train, y_test, feature_names


def run_baseline(data_path: Path, output_dir: Path) -> dict[str, Any]:
    print("[baseline] Fitting frequentist logistic sanity-check model.")
    ensure_data(data_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    _, y_prob, _, _, _, y_test, _ = fit_baseline(data_path)
    metrics_default = metric_summary(y_test, y_prob, threshold=0.5)

    c_fp, c_fn = 1.0, 5.0
    bayes_threshold = expected_loss_threshold(c_fp, c_fn)
    metrics_cost = metric_summary(y_test, y_prob, threshold=bayes_threshold)
    costs = threshold_cost_curve(y_test, y_prob, c_fp=c_fp, c_fn=c_fn)
    costs.to_csv(output_dir / "baseline_cost_curve.csv", index=False)

    best_row = costs.loc[costs["cost"].idxmin()].to_dict()
    predictions = pd.DataFrame(
        {
            "y_true": y_test,
            "p_frequentist_logistic": y_prob,
            "pred_at_0_5": (y_prob >= 0.5).astype(int),
            "pred_at_cost_threshold": (y_prob >= bayes_threshold).astype(int),
        }
    )
    predictions.to_csv(output_dir / "baseline_predictions.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(costs["threshold"], costs["cost"], label="Observed cost")
    ax.axvline(
        bayes_threshold,
        color="tomato",
        linestyle="--",
        label=f"Bayes threshold={bayes_threshold:.3f}",
    )
    ax.axvline(
        best_row["threshold"],
        color="steelblue",
        linestyle="--",
        label=f"Empirical best={best_row['threshold']:.3f}",
    )
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Cost")
    ax.set_title("Baseline Decision Cost")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "baseline_cost_curve.png", dpi=180)
    plt.close(fig)

    summary = {
        "metrics_at_0_5": metrics_default,
        "cost_ratio": {"c_fp": c_fp, "c_fn": c_fn},
        "bayes_threshold": bayes_threshold,
        "metrics_at_cost_threshold": metrics_cost,
        "empirical_best_threshold": best_row,
    }
    write_json(output_dir / "baseline_metrics.json", summary)
    print("[baseline] Done.")
    return summary


def run_bayes(data_path: Path, output_dir: Path, args: argparse.Namespace) -> None:
    print("[bayes] Sampling Bayesian logistic/probit models. This can take a while.")
    ensure_data(data_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    import arviz as az

    X_train, _, y_train, _, _, _, _ = full_pipeline(str(data_path))

    logistic_model = build_bayesian_logistic(
        X_train,
        y_train,
        prior_family="normal",
        beta_scale=2.5,
        model_name="bayes_logistic_baseline",
    )
    idata_logistic = sample_model(
        logistic_model,
        draws=args.draws,
        tune=args.tune,
        chains=args.chains,
        target_accept=args.target_accept,
        random_seed=args.random_seed,
    )
    idata_logistic.to_netcdf(ROOT / "data" / "idata_logistic.nc")
    az.summary(idata_logistic, var_names=["alpha", "beta"]).to_csv(
        output_dir / "bayes_logistic_summary.csv"
    )

    if not args.no_shrinkage:
        logistic_shrink = build_bayesian_logistic(
            X_train,
            y_train,
            prior_family="normal",
            beta_scale=0.5,
            model_name="bayes_logistic_shrink",
        )
        idata_shrink = sample_model(
            logistic_shrink,
            draws=args.shrink_draws,
            tune=args.shrink_tune,
            chains=args.chains,
            target_accept=args.target_accept,
            random_seed=args.random_seed,
        )
        idata_shrink.to_netcdf(ROOT / "data" / "idata_logistic_shrink.nc")
        az.summary(idata_shrink, var_names=["alpha", "beta"]).to_csv(
            output_dir / "bayes_logistic_shrink_summary.csv"
        )

    probit_model = build_bayesian_probit(
        X_train,
        y_train,
        model_name="bayes_probit",
    )
    idata_probit = sample_model(
        probit_model,
        draws=args.draws,
        tune=args.tune,
        chains=args.chains,
        target_accept=args.target_accept,
        random_seed=args.random_seed,
    )
    idata_probit.to_netcdf(ROOT / "data" / "idata_probit.nc")
    az.summary(idata_probit, var_names=["alpha", "beta"]).to_csv(
        output_dir / "bayes_probit_summary.csv"
    )
    print("[bayes] Done. Trace files written to data/*.nc.")


def risk_samples_from_ppc(ppc) -> np.ndarray:
    samples = np.asarray(ppc.posterior_predictive["p"])
    return samples.reshape(-1, samples.shape[-1])


def run_evaluate(data_path: Path, output_dir: Path, args: argparse.Namespace) -> None:
    print("[evaluate] Evaluating Bayesian models and writing comparison outputs.")
    ensure_data(data_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    import arviz as az

    logistic_trace = ROOT / "data" / "idata_logistic.nc"
    probit_trace = ROOT / "data" / "idata_probit.nc"
    if not logistic_trace.exists() or not probit_trace.exists():
        raise FileNotFoundError(
            "Missing Bayesian trace files. Run `python run_project.py bayes` first."
        )

    _, y_prob_freq, X_train, X_test, y_train, y_test, feature_names = fit_baseline(
        data_path
    )
    idata_logistic = az.from_netcdf(logistic_trace)
    idata_probit = az.from_netcdf(probit_trace)

    logistic_model = build_bayesian_logistic(X_train, y_train)
    probit_model = build_bayesian_probit(X_train, y_train)

    ppc_logistic = posterior_predict(
        logistic_model,
        idata_logistic,
        X_test,
        draws=args.prediction_draws,
    )
    ppc_probit = posterior_predict(
        probit_model,
        idata_probit,
        X_test,
        draws=args.prediction_draws,
    )

    p_samples_logistic = risk_samples_from_ppc(ppc_logistic)
    p_samples_probit = risk_samples_from_ppc(ppc_probit)
    y_prob_logistic = p_samples_logistic.mean(axis=0)
    y_prob_probit = p_samples_probit.mean(axis=0)

    models = {
        "Frequentist Logistic": y_prob_freq,
        "Bayesian Logistic": y_prob_logistic,
        "Bayesian Probit": y_prob_probit,
    }
    metrics = {
        name: metric_summary(y_test, y_prob, threshold=0.5)
        for name, y_prob in models.items()
    }
    write_json(output_dir / "model_metrics.json", metrics)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    plot_roc_curves(models, y_test, ax=axes[0])
    plot_pr_curves(models, y_test, ax=axes[1])
    plot_calibration(models, y_test, ax=axes[2])
    fig.tight_layout()
    fig.savefig(output_dir / "model_comparison_curves.png", dpi=180)
    plt.close(fig)

    hdi_logistic = az.hdi(p_samples_logistic.T, hdi_prob=0.95)
    hdi_probit = az.hdi(p_samples_probit.T, hdi_prob=0.95)
    risk_examples = pd.DataFrame(
        {
            "y_true": y_test,
            "bayes_logistic_mean": y_prob_logistic,
            "bayes_logistic_hdi_low": hdi_logistic[:, 0],
            "bayes_logistic_hdi_high": hdi_logistic[:, 1],
            "bayes_probit_mean": y_prob_probit,
            "bayes_probit_hdi_low": hdi_probit[:, 0],
            "bayes_probit_hdi_high": hdi_probit[:, 1],
        }
    )
    risk_examples.to_csv(output_dir / "posterior_risk_predictions.csv", index=False)

    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(len(y_test), size=min(30, len(y_test)), replace=False)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.errorbar(
        x=np.arange(len(idx)),
        y=y_prob_logistic[idx],
        yerr=[
            y_prob_logistic[idx] - hdi_logistic[idx, 0],
            hdi_logistic[idx, 1] - y_prob_logistic[idx],
        ],
        fmt="o",
        capsize=3,
    )
    ax.axhline(0.5, color="k", linestyle="--", alpha=0.3)
    ax.set_title("Bayesian Logistic Individual Risk with 95% HDI")
    ax.set_xlabel("Random test patient subset")
    ax.set_ylabel("Predicted risk")
    fig.tight_layout()
    fig.savefig(output_dir / "bayesian_logistic_risk_hdi.png", dpi=180)
    plt.close(fig)

    or_summary = logistic_odds_ratio_summary(idata_logistic, feature_names)
    or_summary.to_csv(output_dir / "logistic_odds_ratios.csv", index=False)

    plot_posterior_coefs(
        idata_logistic,
        feature_names=feature_names,
        model_name="Bayesian Logistic",
    )
    plt.savefig(output_dir / "bayesian_logistic_forest.png", dpi=180)
    plt.close()

    try:
        az.compare({"logistic": idata_logistic, "probit": idata_probit}, ic="waic").to_csv(
            output_dir / "waic_comparison.csv"
        )
        az.compare({"logistic": idata_logistic, "probit": idata_probit}, ic="loo").to_csv(
            output_dir / "loo_comparison.csv"
        )
    except Exception as exc:
        (output_dir / "model_comparison_warning.txt").write_text(
            f"WAIC/LOO comparison failed: {exc}\n",
            encoding="utf-8",
        )

    c_fp, c_fn = 1.0, 5.0
    threshold = expected_loss_threshold(c_fp, c_fn)
    costs = threshold_cost_curve(y_test, y_prob_logistic, c_fp=c_fp, c_fn=c_fn)
    costs.to_csv(output_dir / "bayesian_logistic_cost_curve.csv", index=False)
    best_row = costs.loc[costs["cost"].idxmin()].to_dict()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(costs["threshold"], costs["cost"], label="Observed cost")
    ax.axvline(
        threshold,
        color="tomato",
        linestyle="--",
        label=f"Bayes threshold={threshold:.3f}",
    )
    ax.axvline(
        best_row["threshold"],
        color="steelblue",
        linestyle="--",
        label=f"Empirical best={best_row['threshold']:.3f}",
    )
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Cost")
    ax.set_title("Bayesian Logistic Decision Cost")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "bayesian_logistic_cost_curve.png", dpi=180)
    plt.close(fig)

    write_json(
        output_dir / "decision_analysis.json",
        {
            "c_fp": c_fp,
            "c_fn": c_fn,
            "bayes_threshold": threshold,
            "empirical_best_threshold": best_row,
            "metrics_at_bayes_threshold": metric_summary(
                y_test,
                y_prob_logistic,
                threshold=threshold,
            ),
        },
    )
    print("[evaluate] Done.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "step",
        nargs="?",
        default="all",
        choices=["all", "eda", "preprocess", "baseline", "bayes", "evaluate"],
        help="Workflow step to run. Defaults to all.",
    )
    parser.add_argument("--data", default=str(DEFAULT_DATA), help="Path to framingham.csv.")
    parser.add_argument("--outputs", default=str(DEFAULT_OUTPUTS), help="Output directory.")
    parser.add_argument("--draws", type=int, default=3000, help="Posterior draws per chain.")
    parser.add_argument("--tune", type=int, default=2000, help="NUTS tuning steps per chain.")
    parser.add_argument("--chains", type=int, default=4, help="Number of MCMC chains.")
    parser.add_argument("--target-accept", type=float, default=0.95)
    parser.add_argument("--random-seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--shrink-draws", type=int, default=2000)
    parser.add_argument("--shrink-tune", type=int, default=1500)
    parser.add_argument("--no-shrinkage", action="store_true")
    parser.add_argument(
        "--prediction-draws",
        type=int,
        default=None,
        help="Optional number of posterior draws to use for posterior prediction.",
    )
    parser.add_argument(
        "--skip-bayes",
        action="store_true",
        help="For `all`, run only EDA/preprocess/baseline steps.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = resolve_path(args.data)
    output_dir = resolve_path(args.outputs)

    if args.step == "eda":
        run_eda(data_path, output_dir)
    elif args.step == "preprocess":
        run_preprocess(data_path, output_dir)
    elif args.step == "baseline":
        run_baseline(data_path, output_dir)
    elif args.step == "bayes":
        run_bayes(data_path, output_dir, args)
    elif args.step == "evaluate":
        run_evaluate(data_path, output_dir, args)
    else:
        run_eda(data_path, output_dir)
        run_preprocess(data_path, output_dir)
        run_baseline(data_path, output_dir)
        if args.skip_bayes:
            print("[all] Skipping Bayesian sampling/evaluation because --skip-bayes was set.")
        else:
            run_bayes(data_path, output_dir, args)
            run_evaluate(data_path, output_dir, args)


if __name__ == "__main__":
    main()
