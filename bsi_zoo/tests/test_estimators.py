import numpy as np
import pytest
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


@pytest.mark.parametrize(
    "solver,rtol,atol", [
        (reweighted_lasso, 1e-1, 0),
        (iterative_L1, 1e-1, 5e-1),
        (iterative_L2, 1e-1, 0),
        (iterative_sqrt, 1e-1, 0),
        (iterative_L1_typeII, 1e-1, 5e-1),
        (iterative_L2_typeII, 1e-1, 1e-1),
    ]
)
def test_reweighted_lasso(solver, rtol, atol):
    y, L, x, cov = _generate_data(n_sensors=50, n_times=1, n_sources=200, nnz=1)
    x_hat = solver(L, y[:, 0], cov, alpha=0.1)
    x = x[:, 0]
    np.testing.assert_array_equal(x != 0, x_hat != 0)
    np.testing.assert_allclose(x, x_hat, rtol=rtol, atol=atol)
