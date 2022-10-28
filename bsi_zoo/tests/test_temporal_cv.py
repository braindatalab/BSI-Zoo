import numpy as np
from scipy import linalg

import pytest

from bsi_zoo.data_generator import get_data
from bsi_zoo.estimators import (
    # iterative_L1,
    # iterative_L2,
    # iterative_sqrt,
    # iterative_L1_typeII,
    # iterative_L2_typeII,
    gamma_map,
    TemporalCVSolver,
)


@pytest.mark.parametrize("n_times", [100])
@pytest.mark.parametrize("orientation_type", ["fixed", "free"])
@pytest.mark.parametrize("nnz", [3])
@pytest.mark.parametrize(
    "estimator,rtol,atol,cov_type,extra_params",
    [
        # (iterative_L1, 1e-1, 5e-1, "diag", {}),
        # (iterative_L2, 1e-1, 5e-1, "diag", {}),
        # (iterative_sqrt, 1e-1, 5e-1, "diag", {}),
        # (iterative_L1_typeII, 1e-1, 5e-1, "full", {}),
        # (iterative_L2_typeII, 1e-1, 5e-1, "full", {}),
        (gamma_map, 1e-1, 5e-1, "full", {"update_mode": 1}),
        # (gamma_map, 1e-1, 5e-1, "full", {"update_mode": 2}),
        # (gamma_map, 1e-1, 5e-1, "full", {"update_mode": 3}),
    ],
)
def test_run_temporal_cv(
    n_times,
    estimator,
    rtol,
    atol,
    cov_type,
    nnz,
    orientation_type,
    extra_params,
):
    n_orient = 1 if orientation_type == "fixed" else 3

    y, L, x, cov, noise = get_data(
        n_sensors=50,
        n_times=n_times,
        n_sources=75 // n_orient,
        n_orient=n_orient,
        nnz=nnz,
        cov_type=cov_type,
        path_to_leadfield=None,
        orientation_type=orientation_type,
    )

    if cov_type == "diag":
        whitener = linalg.inv(linalg.sqrtm(cov))
        L = whitener @ L
        y = whitener @ y

    if orientation_type == "fixed":
        alphas = [0.1, 0.01, 0.001, 0.0001, 0.00001]
    else:
        alphas = [0.1, 0.01, 0.001]

    solver = TemporalCVSolver(
        estimator,
        alphas=alphas,
        cov_type=cov_type,
        cov=cov,
        n_orient=n_orient,
        cv=2,
        extra_params=extra_params,
    ).fit(L=L, y=y)
    x_hat = solver.predict(y)

    # check that the estimated noise level is correct
    if orientation_type == "free":
        x_hat = x_hat.reshape(x.shape)
        L = L.reshape(-1, x.shape[0], x.shape[1])
        noise_hat = y - np.einsum("nmr, mrd->nd", L, x_hat)
    elif orientation_type == "fixed":
        noise_hat = y - (L @ x_hat)
        if n_times < 2:
            noise_hat = noise_hat[:, np.newaxis]

    # check that the number of active sources is correct
    np.testing.assert_equal(x != 0, x_hat != 0)

    # residual error check
    if n_times > 1:
        np.testing.assert_allclose(noise, noise_hat, rtol=1, atol=5)
    else:
        np.testing.assert_allclose(noise, noise_hat, rtol=1, atol=5)

    np.testing.assert_allclose(x, x_hat, rtol=rtol, atol=atol)
