import numpy as np
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
    cov = 1e-2 * np.diag(np.ones(n_sensors))
    noise = rng.multivariate_normal(np.zeros(n_sensors), cov, size=n_times).T
    y += noise
    return y, L, x, cov


def test_reweighted_lasso():
    y, L, x, cov = _generate_data(n_sensors=50, n_times=1, n_sources=200, nnz=1)
    x_hat = reweighted_lasso(L, y[:, 0], cov, alpha_fraction=0.1)
    x = x[:, 0]
    np.testing.assert_array_equal(x != 0, x_hat != 0)
    np.testing.assert_allclose(x, x_hat, rtol=1e-1)


def test_iterative_L1():
    y, L, x, cov = _generate_data(n_sensors=50, n_times=1, n_sources=200, nnz=1)
    x_hat = iterative_L1(L, y[:, 0], cov, alpha=0.1, maxiter=10)
    x = x[:, 0]
    np.testing.assert_array_equal(x != 0, x_hat != 0)
    np.testing.assert_allclose(x, x_hat, atol=1e-1, rtol=5e-1)


def test_iterative_L2():
    y, L, x, cov = _generate_data(n_sensors=50, n_times=1, n_sources=200, nnz=1)
    x_hat = iterative_L2(L, y[:, 0], cov, alpha=0.01, maxiter=10)
    x = x[:, 0]

    np.testing.assert_array_equal(x != 0, x_hat != 0)
    np.testing.assert_allclose(x, x_hat, rtol=1e-1)


def test_iterative_sqrt():
    y, L, x, cov = _generate_data(n_sensors=50, n_times=1, n_sources=200, nnz=1)
    x_hat = iterative_sqrt(L, y[:, 0], cov, alpha=0.1)
    x = x[:, 0]
    np.testing.assert_array_equal(x != 0, x_hat != 0)
    np.testing.assert_allclose(x, x_hat, rtol=1e-1)


def test_iterative_L1_typeII():
    y, L, x, cov = _generate_data(n_sensors=50, n_times=1, n_sources=200, nnz=1)
    x_hat = iterative_L1_typeII(L, y[:, 0], cov, alpha=0.1, maxiter=20)
    x = x[:, 0]

    np.testing.assert_array_equal(x != 0, x_hat != 0)
    np.testing.assert_allclose(x, x_hat, atol=1e-1, rtol=5e-1)


def test_iterative_L2_typeII():
    y, L, x, cov = _generate_data(n_sensors=50, n_times=1, n_sources=200, nnz=1)
    x_hat = iterative_L2_typeII(L, y[:, 0], cov, alpha=0.1)
    x = x[:, 0]

    np.testing.assert_array_equal(x != 0, x_hat != 0)
    np.testing.assert_allclose(x, x_hat, atol=1e1, rtol=1e-1)
