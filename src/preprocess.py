"""
Data loading and preprocessing utilities for the Framingham Heart Study dataset.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer


DATA_PATH = "data/framingham.csv"
TARGET_COL = "TenYearCHD"
RANDOM_STATE = 42


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def summarize_missing(df: pd.DataFrame) -> pd.Series:
    missing = df.isnull().sum()
    return missing[missing > 0]


def make_imputer(
    max_iter: int = 20,
    sample_posterior: bool = True,
) -> IterativeImputer:
    return IterativeImputer(
        random_state=RANDOM_STATE,
        max_iter=max_iter,
        sample_posterior=sample_posterior,
        initial_strategy="median",
    )


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values using IterativeImputer (MICE-like).

    Notes:
    - We impute features only (not the target).
    - `sample_posterior=True` introduces stochasticity for multiple imputation.
    """
    df = df.copy()

    if TARGET_COL not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COL}")

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    imputer = make_imputer()
    X_imp = imputer.fit_transform(X)
    X_imp = pd.DataFrame(X_imp, columns=X.columns, index=df.index)

    return pd.concat([X_imp, y], axis=1)


def get_features_target(df: pd.DataFrame):
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return X, y


def split_and_scale(X, y, test_size: float = 0.2):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)
    return X_train_sc, X_test_sc, y_train.values, y_test.values, scaler


def full_pipeline(path: str = DATA_PATH, test_size: float = 0.2):
    """
    Leakage-safe preprocessing pipeline:
    1) Train/test split on raw data (stratified)
    2) Fit MICE imputer on X_train only; transform X_train and X_test
    3) Fit scaler on imputed X_train; transform both sets
    """
    df = load_data(path)
    if TARGET_COL not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COL}")

    X_raw = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].values

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw,
        y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    imputer = make_imputer()
    X_train_imp = imputer.fit_transform(X_train_raw)
    X_test_imp = imputer.transform(X_test_raw)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_imp)
    X_test = scaler.transform(X_test_imp)

    feature_names = X_raw.columns.tolist()
    return X_train, X_test, y_train, y_test, scaler, imputer, feature_names
