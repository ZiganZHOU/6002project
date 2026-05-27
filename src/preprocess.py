"""
Data loading and preprocessing utilities for the Framingham Heart Study dataset.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


DATA_PATH = "data/framingham.csv"
TARGET_COL = "TenYearCHD"
RANDOM_STATE = 42


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def summarize_missing(df: pd.DataFrame) -> pd.Series:
    missing = df.isnull().sum()
    return missing[missing > 0]


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values with column medians (robust to outliers)."""
    df = df.copy()
    for col in df.columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    return df


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


def full_pipeline(path: str = DATA_PATH):
    df = load_data(path)
    df = impute_missing(df)
    X, y = get_features_target(df)
    return split_and_scale(X, y)
