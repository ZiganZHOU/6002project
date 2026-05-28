"""
Bayesian logistic and probit regression models using PyMC.
"""

import numpy as np
import pymc as pm


def _make_beta_prior(
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
    """
    n_features = X_train.shape[1]
    with pm.Model(name=model_name) as model:
        X = pm.Data("X", X_train)
        # Priors
        alpha = pm.Normal("alpha", mu=0, sigma=1)
        beta = _make_beta_prior(
            "beta",
            n_features=n_features,
            prior_family=prior_family,
            beta_scale=beta_scale,
        )

        # Linear predictor
        eta = alpha + pm.math.dot(X, beta)

        # Likelihood — logistic link
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
    """
    n_features = X_train.shape[1]
    with pm.Model(name=model_name) as model:
        X = pm.Data("X", X_train)
        alpha = pm.Normal("alpha", mu=0, sigma=1)
        beta = _make_beta_prior(
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


def sample_model(
    model,
    draws: int = 3000,
    tune: int = 2000,
    chains: int = 4,
    target_accept: float = 0.95,
):
    """Run NUTS sampling and return InferenceData with log-likelihood."""
    with model:
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
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
    with model:
        pm.set_data({"X": X_new})
        ppc = pm.sample_posterior_predictive(idata, var_names=["p"], random_seed=42, draws=draws)
    return ppc
