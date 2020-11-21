"""Mixture model for matrix completion"""
from typing import Tuple
import numpy as np
from scipy.special import logsumexp
from common import GaussianMixture


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment

    """
    n, _ = X.shape
    mu, var, p = mixture
    K, _ = mu.shape
    log_post = np.zeros((n, K))

    for i in range(n):
        observed_idx = np.nonzero(X[i, :])[0]
        nb_observed = len(observed_idx)
        normalizing = (-nb_observed / 2) * np.log(2 * np.pi * var)
        norm = np.sum((X[i, observed_idx] - mu[:, observed_idx])**2, axis=1) / (2 * var)

        log_post[i, :] = normalizing - norm

    log_post += np.log(p + 1e-16)
    log_p_x = logsumexp(log_post, axis=1).reshape(-1, 1)

    post = np.exp(log_post - log_p_x)
    log_likelihood = np.sum(log_p_x, axis=0).item()

    return post, log_likelihood


def mstep(X: np.ndarray, post: np.ndarray, mixture: GaussianMixture,
          min_variance: float = .25) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        post: (n, K) array holding the soft counts
            for all components for all examples
        mixture: the current gaussian mixture
        min_variance: the minimum variance for each gaussian

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    n, d = X.shape
    mu_hat, sigmas_hat, _ = mixture
    K, _ = post.shape

    p_ji = np.sum(post, axis=0)
    p_hat = p_ji / n

    # Indicator matrix where entry i=1 if i is observed, 0 if i is unobserved (0)
    indicator = X.astype(bool).astype(int)
    observed = np.matmul(post.T, indicator)  # nb points observed (dimensions of features)
    full_points = np.where(observed >= 1)  # indices where the observed value is > 1 (for significance)

    mu_hat[full_points] = np.matmul(post.T, X)[full_points] / observed[full_points]

    normalizing = np.sum(post * np.sum(indicator, axis=1).reshape(-1, 1), axis=0)
    norm = np.sum(X**2, axis=1)[:, None] - 2 * np.matmul(X, mu_hat.T) + np.matmul(indicator, mu_hat.T**2)

    sigmas_hat = np.maximum(np.sum(post * norm, axis=0) / normalizing, min_variance)

    return GaussianMixture(mu_hat, sigmas_hat, p_hat)


def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    old_LL = None
    LL = None

    while old_LL is None or (LL - old_LL >= 1e-6 * np.abs(LL)):
        old_LL = LL
        post, LL = estep(X, mixture)
        mixture = mstep(X, post, mixture)

    return mixture, post, LL


def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model

    Args:
        X: (n, d) array of incomplete data (incomplete entries = 0)
        mixture: a mixture of gaussians

    Returns
        np.ndarray: a (n, d) array with completed data
    """
    n, _ = X.shape
    mu, var, p = mixture
    K = mu.shape[0]

    log_post = np.zeros((n, K))
    X_pred = X.copy()

    #Indices with incomplete entries
    indicator = X_pred.astype(bool).astype(int)
    unobserved = np.where(indicator == 0)

    # post, _ = estep(X, mixture)  # probabilities of user u belonging to each cluster

    # Compute probabilities of user i belonging to each cluster
    for i in range(n):
        observed = np.nonzero(X[i, :])[0]
        normalizing = (-(len(observed)) / 2) * np.log(2 * np.pi * var)
        norm = np.sum((X[i, observed] - mu[:, observed])**2, axis=1) / (2 * var)

        log_post[i, :] = normalizing - norm

    log_post += np.log(p + 1e-16)
    log_p_x = logsumexp(log_post, axis=1).reshape(-1, 1)
    post = np.exp(log_post - log_p_x)

    X_pred[unobserved] = np.matmul(post, mu)[unobserved]  # weighted average of the means

    return X_pred
