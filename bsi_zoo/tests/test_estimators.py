import numpy as np
from scipy import linalg
import pytest
import pdb 
from mne.utils import logger, verbose, warn
from bsi_zoo.estimators import (
    iterative_L1,
    iterative_L2,
    iterative_sqrt,
    iterative_L1_typeII,
    iterative_L2_typeII,
    gamma_map,
    champagne,
)


def _generate_data(n_sensors, n_times, n_sources, nnz):
    rng = np.random.RandomState(42)
    x = np.zeros((n_sources, n_times))
    x[:nnz] = rng.randn(nnz, n_times)
    L = rng.randn(n_sensors, n_sources)  # TODO: add orientation support
    y = L @ x
    cov_type = 'full'
    if cov_type == 'diag':
        ## initialization of the noise covariance matrix with a random diagonal matrix 
        cov = rng.randn(n_sensors, n_sensors)
        cov = 1e-3 * (cov @ cov.T)
        cov = np.diag(np.diag(cov))
        ## initialization of the noise covariance with an identity matrix
        cov = 1e-2 * np.diag(np.ones(n_sensors))
    else:
        ## initialization of the noise covariance matrix with a full PSD random matrix
        cov = rng.randn(n_sensors, n_sensors)
        cov = 1e-3 * (cov @ cov.T)
        # cov = 1e-3 * (cov @ cov.T) / n_times ## devided by the number of time samples for better scalinggit 
    noise = rng.multivariate_normal(np.zeros(n_sensors), cov, size=n_times).T
    y += noise
    return y, L, x, cov

@pytest.mark.parametrize(
    "solver,alpha,rtol,atol,cov_type", [
        (iterative_L1, 0.1, 1e-1, 5e-1, 'diag'),
        (iterative_L2, 0.2, 1e-1, 5e-1, 'diag'),
        (iterative_sqrt, 0.1, 1e-1, 5e-1, 'diag'),
        (iterative_L1_typeII, 0.1, 1e-1, 5e-1, 'full'),
        (iterative_L2_typeII, 0.2, 1e-1, 1e-1, 'full'),
        (gamma_map, 0.2, 1e-1, 5e-1, 'full'),
        (champagne, 0.2, 1e-1, 5e-1, 'full'),
    ]
)
def test_estimator(solver, alpha, rtol, atol, cov_type):
    y, L, x, cov = _generate_data(n_sensors=50, n_times=10, n_sources=200, nnz=1)
    if cov_type == 'diag':
        whitener = linalg.inv(linalg.sqrtm(cov))
        L = whitener @ L
        y = whitener @ y
        x_hat = solver(L, y, alpha=alpha)
    else:
        x_hat = solver(L, y, cov, alpha=alpha)
    np.testing.assert_array_equal(x != 0, x_hat != 0)
    np.testing.assert_allclose(x, x_hat, rtol=rtol, atol=atol)