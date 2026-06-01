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
BINARY_COLS = [
    "male",
    "currentSmoker",
    "BPMeds",
    "prevalentStroke",
    "prevalentHyp",
    "diabetes",
]
ORDINAL_BOUNDS = {
    "education": (1, 4),
}
NONNEGATIVE_COLS = [
    "cigsPerDay",
    "totChol",
    "sysBP",
    "diaBP",
    "BMI",
    "heartRate",
    "glucose",
]
FEATURE_GROUPS = {
    "demographic": ["male", "age", "education"],
    "behavioral": ["currentSmoker", "cigsPerDay"],
    "medical_history": ["BPMeds", "prevalentStroke", "prevalentHyp", "diabetes"],
    "vitals": ["sysBP", "diaBP", "BMI", "heartRate"],
    "labs": ["totChol", "glucose"],
}


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def summarize_missing(df: pd.DataFrame) -> pd.Series:
    missing = df.isnull().sum()
    return missing[missing > 0]


def make_imputer(
    max_iter: int = 20,
    sample_posterior: bool = False,
) -> IterativeImputer:
    return IterativeImputer(
        random_state=RANDOM_STATE,
        max_iter=max_iter,
        sample_posterior=sample_posterior,
        initial_strategy="median",
    )


def postprocess_imputed_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Restore simple domain constraints after regression-based imputation.

    IterativeImputer can produce fractional values for binary/ordinal variables.
    Rounding these columns keeps the design matrix clinically interpretable while
    preserving the train-only fitting discipline.
    """
    X = X.copy()

    for col in BINARY_COLS:
        if col in X.columns:
            X[col] = X[col].round().clip(0, 1)

    for col, (lower, upper) in ORDINAL_BOUNDS.items():
        if col in X.columns:
            X[col] = X[col].round().clip(lower, upper)

    for col in NONNEGATIVE_COLS:
        if col in X.columns:
            X[col] = X[col].clip(lower=0)

    return X


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values using IterativeImputer (MICE-like single imputation).

    Notes:
    - We impute features only (not the target).
    - Set `sample_posterior=True` in `make_imputer` for sensitivity checks.
    """
    df = df.copy()

    if TARGET_COL not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COL}")

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    imputer = make_imputer()
    X_imp = imputer.fit_transform(X)
    X_imp = pd.DataFrame(X_imp, columns=X.columns, index=df.index)
    X_imp = postprocess_imputed_features(X_imp)

    return pd.concat([X_imp, y], axis=1)


def get_features_target(df: pd.DataFrame):
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return X, y


def feature_group_index(feature_names: list[str]):
    """
    Map feature names to clinically motivated group indices.

    Returns
    -------
    group_names:
        Ordered group labels.
    group_idx:
        Integer group id for each feature in `feature_names`.
    """
    group_lookup = {
        feature: group_name
        for group_name, features in FEATURE_GROUPS.items()
        for feature in features
    }
    group_names = list(FEATURE_GROUPS)
    group_id = {group_name: idx for idx, group_name in enumerate(group_names)}
    group_idx = []
    for feature in feature_names:
        if feature not in group_lookup:
            raise ValueError(f"Missing feature group assignment for: {feature}")
        group_idx.append(group_id[group_lookup[feature]])
    return group_names, np.asarray(group_idx, dtype="int64")


def split_and_scale(X, y, test_size: float = 0.2):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)
    return X_train_sc, X_test_sc, y_train.values, y_test.values, scaler


def compute_vif(X: pd.DataFrame) -> pd.DataFrame:
    """
    Compute variance inflation factors on centered/scaled predictors.

    VIF should include an intercept and should not be computed on raw,
    uncentered clinical measurements; otherwise variables with large positive
    means can look artificially collinear.
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    import statsmodels.api as sm

    X_scaled = pd.DataFrame(
        StandardScaler().fit_transform(X),
        columns=X.columns,
        index=X.index,
    )
    X_with_const = sm.add_constant(X_scaled, has_constant="add")
    values = X_with_const.values
    rows = []
    for idx, feature in enumerate(X_scaled.columns, start=1):
        rows.append(
            {
                "feature": feature,
                "vif": variance_inflation_factor(values, idx),
            }
        )
    return pd.DataFrame(rows).sort_values("vif", ascending=False).reset_index(drop=True)


def full_pipeline(
    path: str = DATA_PATH,
    test_size: float = 0.2,
    imputer_sample_posterior: bool = False,
):
    """
    Leakage-safe preprocessing pipeline:
    1) Train/test split on raw data (stratified)
    2) Fit IterativeImputer on X_train only; transform X_train and X_test
    3) Restore binary/ordinal/nonnegative domain constraints after imputation
    4) Fit scaler on imputed X_train; transform both sets
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

    imputer = make_imputer(sample_posterior=imputer_sample_posterior)
    X_train_imp = pd.DataFrame(
        imputer.fit_transform(X_train_raw),
        columns=X_raw.columns,
        index=X_train_raw.index,
    )
    X_test_imp = pd.DataFrame(
        imputer.transform(X_test_raw),
        columns=X_raw.columns,
        index=X_test_raw.index,
    )
    X_train_imp = postprocess_imputed_features(X_train_imp)
    X_test_imp = postprocess_imputed_features(X_test_imp)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_imp)
    X_test = scaler.transform(X_test_imp)

    feature_names = X_raw.columns.tolist()
    return X_train, X_test, y_train, y_test, scaler, imputer, feature_names
