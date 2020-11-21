"""Mixture model using EM"""
from typing import Tuple
import numpy as np
from common import GaussianMixture


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment
    """
    n, d = X.shape
    mu, var, p = mixture
    K, _ = mu.shape

    post = np.zeros((n, K), dtype=np.float64)
    for i in range(n):
        post[i, :] = 1 / (2 * np.pi * var)**(d / 2) * np.exp(-np.sum((X[i, :] - mu)**2, axis=1) / (2 * var))

    marginal_prob = p * post
    p_x = np.sum(marginal_prob, axis=1).reshape(-1, 1)

    post = marginal_prob / p_x
    log_likelihood = np.sum(np.log(p_x), axis=0).item()

    return post, log_likelihood


def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    n, d = X.shape
    _, K = post.shape

    p_ji = np.sum(post, axis=0)
    p_hat = p_ji / n

    mu_hat = np.zeros((K, d))
    sigmas_hat = np.zeros(K)

    for j in range(K):
        mu_hat[j, :] = np.matmul(post[:, j], X) / p_ji[j]
        norm = np.sum((mu_hat[j] - X)**2, axis=1)
        sigmas_hat[j] = np.matmul(post[:, j], norm) / (d * p_ji[j])

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
        mixture = mstep(X, post)

    return mixture, post, LL
