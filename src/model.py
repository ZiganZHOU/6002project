"""
Bayesian logistic and probit regression models using PyMC.
"""

import numpy as np


def _require_pymc():
    try:
        import pymc as pm
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PyMC is required for Bayesian model fitting. Install project "
            "dependencies with: python -m pip install -r requirements.txt"
        ) from exc
    return pm


def _make_beta_prior(
    pm,
    name: str,
    n_features: int,
    prior_family: str = "normal",
    beta_scale: float = 2.5,
):
    if prior_family == "normal":
        return pm.Normal(name, mu=0, sigma=beta_scale, shape=n_features)
    if prior_family == "laplace":
        return pm.Laplace(name, mu=0, b=beta_scale, shape=n_features)
    raise ValueError(f"Unsupported prior_family: {prior_family}")


def build_bayesian_logistic(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_name: str = "bayes_logistic",
    prior_family: str = "normal",
    beta_scale: float = 2.5,
):
    """
    Bayesian logistic regression with configurable coefficient priors.

    `model_name` is kept as a reporting label. It is intentionally not passed
    into `pm.Model(name=...)`, because PyMC model names can prefix variables and
    make downstream calls such as `var_names=["beta", "p"]` fragile.
    """
    pm = _require_pymc()
    n_features = X_train.shape[1]
    with pm.Model() as model:
        X = pm.Data("X", X_train)
        # Priors
        alpha = pm.Normal("alpha", mu=0, sigma=1)
        beta = _make_beta_prior(
            pm,
            "beta",
            n_features=n_features,
            prior_family=prior_family,
            beta_scale=beta_scale,
        )

        # Linear predictor
        eta = alpha + pm.math.dot(X, beta)

        # Logistic link likelihood
        p = pm.Deterministic("p", pm.math.sigmoid(eta))
        y_obs = pm.Bernoulli("y_obs", p=p, observed=y_train)

    return model


def build_bayesian_probit(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_name: str = "bayes_probit",
    prior_family: str = "normal",
    beta_scale: float = 2.5,
):
    """
    Bayesian probit regression: uses the normal CDF (Phi) as the link function.

    `model_name` is kept as a reporting label; see `build_bayesian_logistic`.
    """
    pm = _require_pymc()
    n_features = X_train.shape[1]
    with pm.Model() as model:
        X = pm.Data("X", X_train)
        alpha = pm.Normal("alpha", mu=0, sigma=1)
        beta = _make_beta_prior(
            pm,
            "beta",
            n_features=n_features,
            prior_family=prior_family,
            beta_scale=beta_scale,
        )

        eta = alpha + pm.math.dot(X, beta)

        # Probit link: Phi(eta)
        p = pm.Deterministic("p", 0.5 * (1 + pm.math.erf(eta / pm.math.sqrt(2))))
        y_obs = pm.Bernoulli("y_obs", p=p, observed=y_train)

    return model


def build_hierarchical_logistic(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
    group_names: list[str],
    group_idx: np.ndarray,
    model_name: str = "hierarchical_logistic",
    group_scale_prior: float = 1.0,
):
    """
    Bayesian logistic regression with grouped hierarchical shrinkage.

    Coefficients share a learned scale within clinically motivated feature
    groups, e.g. demographics, behavior, medical history, vitals, and labs:

        beta_j ~ Normal(0, tau_group[j])
        tau_g ~ HalfNormal(group_scale_prior)

    This lets the model learn which clinical feature families need stronger or
    weaker shrinkage instead of applying one fixed coefficient prior to every
    predictor.
    """
    pm = _require_pymc()
    n_features = X_train.shape[1]
    if len(feature_names) != n_features:
        raise ValueError("feature_names length must match X_train columns.")
    if len(group_idx) != n_features:
        raise ValueError("group_idx length must match X_train columns.")

    coords = {
        "feature": feature_names,
        "group": group_names,
    }
    group_idx = np.asarray(group_idx, dtype="int64")

    with pm.Model(coords=coords) as model:
        X = pm.Data("X", X_train)
        alpha = pm.Normal("alpha", mu=0, sigma=1)
        group_scale = pm.HalfNormal(
            "group_scale",
            sigma=group_scale_prior,
            dims="group",
        )
        beta = pm.Normal(
            "beta",
            mu=0,
            sigma=group_scale[group_idx],
            dims="feature",
        )

        eta = alpha + pm.math.dot(X, beta)
        p = pm.Deterministic("p", pm.math.sigmoid(eta))
        y_obs = pm.Bernoulli("y_obs", p=p, observed=y_train)

    return model


def sample_model(
    model,
    draws: int = 3000,
    tune: int = 2000,
    chains: int = 4,
    target_accept: float = 0.95,
    random_seed: int = 42,
):
    """Run NUTS sampling and return InferenceData with log-likelihood."""
    pm = _require_pymc()
    with model:
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=random_seed,
            return_inferencedata=True,
            progressbar=True,
        )
        idata = pm.compute_log_likelihood(idata)
    return idata


def posterior_predict(model, idata, X_new: np.ndarray, draws: int | None = None):
    """
    Generate posterior predictive samples for continuous risk p(x).

    Important: we explicitly sample deterministic variable 'p', not 'y_obs'.
    Sampling only y_obs would yield 0/1 draws and cannot provide continuous
    patient-level risk intervals.
    """
    pm = _require_pymc()
    idata_for_prediction = idata
    if draws is not None:
        idata_for_prediction = idata.isel(draw=slice(0, draws))

    with model:
        pm.set_data({"X": X_new})
        ppc = pm.sample_posterior_predictive(
            idata_for_prediction,
            var_names=["p"],
            random_seed=42,
        )
    return ppc
