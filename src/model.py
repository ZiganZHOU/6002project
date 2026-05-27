"""
Bayesian logistic and probit regression models using PyMC.
"""

import numpy as np
import pymc as pm
import arviz as az


def build_bayesian_logistic(X_train: np.ndarray, y_train: np.ndarray, model_name: str = "bayes_logistic"):
    """
    Bayesian logistic regression with weakly informative Normal(0, 2.5) priors
    on coefficients (following Gelman et al. recommendation for scaled inputs).
    """
    n_features = X_train.shape[1]
    with pm.Model(name=model_name) as model:
        # Priors
        alpha = pm.Normal("alpha", mu=0, sigma=1)
        beta = pm.Normal("beta", mu=0, sigma=2.5, shape=n_features)

        # Linear predictor
        eta = alpha + pm.math.dot(X_train, beta)

        # Likelihood — logistic link
        p = pm.Deterministic("p", pm.math.sigmoid(eta))
        y_obs = pm.Bernoulli("y_obs", p=p, observed=y_train)

    return model


def build_bayesian_probit(X_train: np.ndarray, y_train: np.ndarray, model_name: str = "bayes_probit"):
    """
    Bayesian probit regression: uses the normal CDF (Phi) as the link function.
    """
    import pytensor.tensor as pt
    from scipy.special import ndtr

    n_features = X_train.shape[1]
    with pm.Model(name=model_name) as model:
        alpha = pm.Normal("alpha", mu=0, sigma=1)
        beta = pm.Normal("beta", mu=0, sigma=2.5, shape=n_features)

        eta = alpha + pm.math.dot(X_train, beta)

        # Probit link: Phi(eta)
        p = pm.Deterministic("p", 0.5 * (1 + pm.math.erf(eta / pm.math.sqrt(2))))
        y_obs = pm.Bernoulli("y_obs", p=p, observed=y_train)

    return model


def sample_model(model, draws: int = 2000, tune: int = 1000, chains: int = 2, target_accept: float = 0.9):
    """Run NUTS sampling and return InferenceData."""
    with model:
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            return_inferencedata=True,
            progressbar=True,
        )
    return idata


def posterior_predict(model, idata, X_new: np.ndarray):
    """Generate posterior predictive samples for new data."""
    with model:
        pm.set_data({})  # placeholder; update if using pm.Data containers
        ppc = pm.sample_posterior_predictive(idata)
    return ppc
