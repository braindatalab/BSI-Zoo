import numpy as np
from scipy import linalg
import pytest
import pdb 
from bsi_zoo.estimators import (
    reweighted_lasso,
    iterative_L1,
    iterative_L2,
    iterative_sqrt,
    iterative_L1_typeII,
    iterative_L2_typeII,
)


def _generate_data(n_sensors, n_times, n_sources, nnz):
    rng = np.random.RandomState(42)
    x = np.zeros((n_sources, n_times))
    x[:nnz] = rng.randn(nnz, n_times)
    L = rng.randn(n_sensors, n_sources)  # TODO: add orientation support
    y = L @ x
    cov = rng.randn(n_sensors, n_sensors)
    cov = 1e-3 * (cov @ cov.T)
    # cov = 1e-3 * (cov @ cov.T) / n_times ## devided by the number of time samples for better scalinggit 

    ## initialization of noise covariance with a diagonal matrix 
    # cov = np.diag(np.diag(cov))
    # cov = 1e-2 * np.diag(np.ones(n_sensors))
    noise = rng.multivariate_normal(np.zeros(n_sensors), cov, size=n_times).T
    y += noise
    return y, L, x, cov

# (reweighted_lasso, 0.1, 1e-1, 0, 'diag'),
@pytest.mark.parametrize(
    "solver,alpha,rtol,atol,cov_type", [
        (iterative_L1, 0.1, 1e-1, 5e-1, 'diag'),
        (iterative_L2, 0.01, 1e-1, 0, 'diag'),
        (iterative_sqrt, 0.1, 1e-1, 0, 'diag'),
        (iterative_L1_typeII, 0.1, 1e-1, 5e-1, 'full'),
        (iterative_L2_typeII, 0.3, 1e-1, 1e-1, 'full'),
    ]
)
# def test_estimator(solver, alpha, rtol, atol, cov_type):
#     y, L, x, cov = _generate_data(n_sensors=50, n_times=1, n_sources=200, nnz=1)
#     if cov_type == 'diag':
#         whitener = linalg.inv(linalg.sqrtm(cov))
#         L = whitener @ L
#         y = whitener @ y
#         x_hat = solver(L, y[:, 0], alpha=alpha)
#     else:
#         x_hat = solver(L, y[:, 0], cov, alpha=alpha)
#     x = x[:, 0]
#     np.testing.assert_array_equal(x != 0, x_hat != 0)
#     np.testing.assert_allclose(x, x_hat, rtol=rtol, atol=atol)

def test_estimator(solver, alpha, rtol, atol, cov_type):
    y, L, x, cov = _generate_data(n_sensors=50, n_times=10, n_sources=200, nnz=1)
    if cov_type == 'diag':
        whitener = linalg.inv(linalg.sqrtm(cov))
        L = whitener @ L
        y = whitener @ y
        x_hat = solver(L, y, alpha=alpha)
    else:
        x_hat = solver(L, y, cov, alpha=alpha)
    # x = x[:, 0]
    # x = x.T
    np.testing.assert_array_equal(x != 0, x_hat != 0)
    np.testing.assert_allclose(x, x_hat, rtol=rtol, atol=atol)

# # # Test the performance for single measuremnt vector (SMV) case
# def test_estimator(solver, alpha, rtol, atol, cov_type):
#     y, L, x, cov = _generate_data(n_sensors=50, n_times=1, n_sources=200, nnz=1)
#     if cov_type == 'diag':
#         whitener = linalg.inv(linalg.sqrtm(cov))
#         L = whitener @ L
#         y = whitener @ y
#         x_hat = solver(L, y[:, 0], alpha=alpha)
#     else:
#         x_hat = solver(L, y[:, 0], cov, alpha=alpha)
#     x = x[:, 0]
#     np.testing.assert_array_equal(x != 0, x_hat != 0)
#     np.testing.assert_allclose(x, x_hat, rtol=rtol, atol=atol)
