"""Dataset registry and loaders for cross-dataset generalization experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DatasetSpec:
    """Metadata needed to turn a local CSV into a binary classification task."""

    name: str
    filename: str
    target_col: str = "target"
    drop_cols: tuple[str, ...] = ()
    default_max_rows: int | None = None
    description: str = ""


DATASET_SPECS: dict[str, DatasetSpec] = {
    "framingham": DatasetSpec(
        name="framingham",
        filename="framingham.csv",
        target_col="TenYearCHD",
        default_max_rows=None,
        description="10-year coronary heart disease risk prediction.",
    ),
    "breast_cancer": DatasetSpec(
        name="breast_cancer",
        filename="breast_cancer_wisconsin_diagnostic.csv",
        drop_cols=("id", "diagnosis"),
        description="Wisconsin diagnostic breast cancer: malignant vs benign.",
    ),
    "mammographic_mass": DatasetSpec(
        name="mammographic_mass",
        filename="mammographic_mass.csv",
        description="Mammographic mass severity: malignant vs benign.",
    ),
    "heart_cleveland": DatasetSpec(
        name="heart_cleveland",
        filename="heart_disease_cleveland.csv",
        drop_cols=("target_original",),
        description="Cleveland heart disease: disease present vs absent.",
    ),
    "cdc_diabetes": DatasetSpec(
        name="cdc_diabetes",
        filename="cdc_diabetes_health_indicators.csv",
        drop_cols=("Diabetes_binary",),
        default_max_rows=10000,
        description="CDC diabetes health indicators: diabetes/prediabetes vs healthy.",
    ),
}


DEFAULT_GENERALIZATION_DATASETS = (
    "breast_cancer",
    "mammographic_mass",
    "heart_cleveland",
    "cdc_diabetes",
)


def available_dataset_names() -> list[str]:
    return list(DATASET_SPECS)


def load_binary_dataset(
    data_dir: Path,
    dataset_name: str,
    random_state: int = 42,
    max_rows: int | None = None,
) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """Load a registered binary dataset as numeric features and integer labels."""
    if dataset_name not in DATASET_SPECS:
        names = ", ".join(available_dataset_names())
        raise ValueError(f"Unknown dataset '{dataset_name}'. Available: {names}")

    spec = DATASET_SPECS[dataset_name]
    path = data_dir / spec.filename
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset file: {path}")

    df = pd.read_csv(path)
    if spec.target_col not in df.columns:
        raise ValueError(f"{path.name} is missing target column '{spec.target_col}'.")

    effective_max_rows = max_rows if max_rows is not None else spec.default_max_rows
    if effective_max_rows is not None and len(df) > effective_max_rows:
        df = _stratified_sample(
            df,
            target_col=spec.target_col,
            n_rows=effective_max_rows,
            random_state=random_state,
        )

    y = df[spec.target_col].astype(int).to_numpy()
    drop_cols = [spec.target_col, *spec.drop_cols]
    X = df.drop(columns=[col for col in drop_cols if col in df.columns])
    X = X.apply(pd.to_numeric, errors="coerce")

    metadata = {
        "name": spec.name,
        "description": spec.description,
        "path": str(path),
        "n_rows": int(len(df)),
        "n_features": int(X.shape[1]),
        "positive": int(y.sum()),
        "negative": int(len(y) - y.sum()),
        "positive_rate": float(y.mean()),
        "missing_values": {
            key: int(value)
            for key, value in X.isna().sum().items()
            if int(value) > 0
        },
        "sampled_max_rows": effective_max_rows,
    }
    return X, y, metadata


def _stratified_sample(
    df: pd.DataFrame,
    target_col: str,
    n_rows: int,
    random_state: int,
) -> pd.DataFrame:
    """Sample rows while approximately preserving the class distribution."""
    frac = n_rows / len(df)
    sampled = (
        df.groupby(target_col, group_keys=False)
        .sample(frac=frac, random_state=random_state)
        .reset_index(drop=True)
    )
    if len(sampled) > n_rows:
        sampled = sampled.sample(n=n_rows, random_state=random_state)
    return sampled.reset_index(drop=True)
