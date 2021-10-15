import numpy as np
from scipy import linalg
import pytest
from skimage.feature import peak_local_max

from bsi_zoo.estimators import (
    iterative_L1,
    iterative_L2,
    iterative_sqrt,
    iterative_L1_typeII,
    iterative_L2_typeII,
    gamma_map,
)


def _generate_data(n_sensors, n_times, n_sources, nnz, cov_type, path_to_leadfield):
    rng = np.random.RandomState(42)
    if path_to_leadfield is not None:
        lead_field = np.load(path_to_leadfield, allow_pickle=True)
        L = lead_field["lead_field"]
        n_sensors, n_sources = L.shape
    else:
        L = rng.randn(n_sensors, n_sources)  # TODO: add orientation support

    x = np.zeros((n_sources, n_times))
    x[:nnz] = rng.randn(nnz, n_times)
    y = L @ x

    noise_type = "random"
    if cov_type == "diag":
        if noise_type == "random":
            # initialization of the noise covariance matrix with a random diagonal matrix
            cov = rng.randn(n_sensors, n_sensors)
            cov = 1e-3 * (cov @ cov.T)
            cov = np.diag(np.diag(cov))
        else:
            # initialization of the noise covariance with an identity matrix
            cov = 1e-2 * np.diag(np.ones(n_sensors))
    else:
        # initialization of the noise covariance matrix with a full PSD random matrix
        cov = rng.randn(n_sensors, n_sensors)
        cov = 1e-3 * (cov @ cov.T)
        # cov = 1e-3 * (cov @ cov.T) / n_times ## devided by the number of time samples for better scaling
    noise = rng.multivariate_normal(np.zeros(n_sensors), cov, size=n_times).T
    # snr = 20 * np.log(np.linalg.norm(y, 'fro') / np.linalg.norm(noise, 'fro'))
    y += noise
    if n_times == 1:
        y = y[:, 0]
        x = x[:, 0]

    return y, L, x, cov, noise


@pytest.mark.parametrize("n_times", [1, 10])
@pytest.mark.parametrize(
    "path_to_leadfield", [None, "bsi_zoo/tests/data/lead_field.npz"]
)
@pytest.mark.parametrize(
    "solver,alpha,rtol,atol,cov_type",
    [
        (iterative_L1, 0.1, 1e-1, 5e-1, "diag"),
        (iterative_L2, 0.01, 1e-1, 5e-1, "diag"),
        (iterative_sqrt, 0.1, 1e-1, 5e-1, "diag"),
        (iterative_L1_typeII, 0.1, 1e-1, 5e-1, "full"),
        (iterative_L2_typeII, 0.2, 1e-1, 1e-1, "full"),
        (gamma_map, 0.2, 1e-1, 5e-1, "full"),
    ],
)
def test_estimator(n_times, solver, alpha, rtol, atol, cov_type, path_to_leadfield):
    y, L, x, cov, noise = _generate_data(
        n_sensors=50,
        n_times=n_times,
        n_sources=200,
        nnz=1,
        cov_type=cov_type,
        path_to_leadfield=path_to_leadfield,
    )
    if cov_type == "diag":
        whitener = linalg.inv(linalg.sqrtm(cov))
        L = whitener @ L
        y = whitener @ y
        x_hat = solver(L, y, alpha=alpha)
    else:
        x_hat = solver(L, y, cov, alpha=alpha)

    noise_hat = y - (L @ x_hat)
    if n_times > 1:
        np.testing.assert_array_less(
            np.linalg.norm(
                peak_local_max(np.linalg.norm(x, axis=0), num_peaks=1)
                - peak_local_max(np.linalg.norm(x_hat, axis=0), num_peaks=1), axis=0
            ),
            1.1,
        )
    else:
        np.testing.assert_array_less(
            np.linalg.norm(
                peak_local_max(x, num_peaks=1) - peak_local_max(x_hat, num_peaks=1)
            ),
            1.1,
        )

    if path_to_leadfield is None:
        np.testing.assert_array_equal(x != 0, x_hat != 0)
        np.testing.assert_allclose(x, x_hat, rtol=rtol, atol=atol)

    # residual error check
    if n_times > 1:
        np.testing.assert_allclose(noise, noise_hat, rtol=1, atol=5)
    else:
        np.testing.assert_allclose(noise, noise_hat[:, np.newaxis], rtol=1, atol=5)  # TODO: decide threshold
