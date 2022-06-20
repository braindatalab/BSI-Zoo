import numpy as np
from scipy import linalg
import pytest

from bsi_zoo.config import get_leadfield_path, get_fwd_fname
from bsi_zoo.data_generator import get_data

from bsi_zoo.estimators import (
    iterative_L1,
    iterative_L2,
    iterative_sqrt,
    iterative_L1_typeII,
    iterative_L2_typeII,
    gamma_map,
)


@pytest.mark.parametrize("n_times", [1, 10])
@pytest.mark.parametrize("orientation_type", ["fixed", "free"])
@pytest.mark.parametrize("nnz", [1, 2, 3])
@pytest.mark.parametrize(
    "subject", [None, "CC120166", "CC120264", "CC120309", "CC120313"]
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
def test_estimator(
    n_times,
    solver,
    alpha,
    rtol,
    atol,
    cov_type,
    subject,
    nnz,
    orientation_type,
    save_estimates=True,
):

    if solver != gamma_map and orientation_type == "free":
        pytest.skip("Free orientation support only for Gamma Map solver currently.")

    path_to_leadfield = get_leadfield_path(
        subject, orientation_type
    )  # setting leadfield paths

    y, L, x, cov, noise = get_data(
        n_sensors=50,
        n_times=n_times,
        n_sources=200,
        n_orient=3,
        nnz=nnz,
        cov_type=cov_type,
        path_to_leadfield=path_to_leadfield,
        orientation_type=orientation_type,
    )
    if cov_type == "diag":
        whitener = linalg.inv(linalg.sqrtm(cov))
        L = whitener @ L
        y = whitener @ y
        x_hat = solver(L, y, alpha=alpha)
    else:
        x_hat = solver(L, y, cov, alpha=alpha)

    if orientation_type == "free":
        x_hat = x_hat.reshape(x.shape)
        L = L.reshape(-1, x.shape[0], x.shape[1])
        noise_hat = y - np.einsum("nmr, mrd->nd", L, x_hat)
    elif orientation_type == "fixed":
        noise_hat = y - (L @ x_hat)
        if n_times < 2:
            noise_hat = noise_hat[:, np.newaxis]

    # residual error check
    if n_times > 1:
        np.testing.assert_allclose(noise, noise_hat, rtol=1, atol=5)
    else:
        np.testing.assert_allclose(noise, noise_hat, rtol=1, atol=5)

    if subject is None:
        # dummy data case
        if (
            orientation_type == "fixed"
        ):  # test is too stringent for free orientation setting
            np.testing.assert_array_equal(x != 0, x_hat != 0)
        np.testing.assert_allclose(x, x_hat, rtol=rtol, atol=atol)

    else:
        from mne.inverse_sparse.mxne_inverse import _make_sparse_stc
        from mne import read_forward_solution, convert_forward_solution

        fwd_fname = get_fwd_fname(subject)
        fwd = read_forward_solution(fwd_fname)
        global stc, stc_hat

        if orientation_type == "fixed":
            if n_times > 1:
                fwd = convert_forward_solution(fwd, force_fixed=True)

                active_set = np.linalg.norm(x, axis=1) != 0

                # check if no vertices are estimated
                temp = np.linalg.norm(x_hat, axis=1)
                if len(np.unique(temp)) == 1:
                    print("No vertices estimated!")

                temp_ = np.partition(-temp, nnz)
                max_temp = -temp_[:nnz]  # get n(=nnz) max amplitudes

                # remove 0 from list incase less vertices than nnz were estimated
                max_temp = np.delete(max_temp, np.where(max_temp == 0.0))
                active_set_hat = np.array(list(map(max_temp.__contains__, temp)))

                stc = _make_sparse_stc(
                    x[active_set], active_set, fwd, tmin=1, tstep=1
                )  # ground truth
                stc_hat = _make_sparse_stc(
                    x_hat[active_set_hat], active_set_hat, fwd, tmin=1, tstep=1
                )  # estimate

                # euclidean distance check
                lh_coordinates = fwd["src"][0]["rr"][stc.lh_vertno]
                lh_coordinates_hat = fwd["src"][0]["rr"][stc_hat.lh_vertno]
                rh_coordinates = fwd["src"][1]["rr"][stc.rh_vertno]
                rh_coordinates_hat = fwd["src"][1]["rr"][stc_hat.rh_vertno]
                coordinates = np.concatenate([lh_coordinates, rh_coordinates], axis=0)
                coordinates_hat = np.concatenate(
                    [lh_coordinates_hat, rh_coordinates_hat], axis=0
                )
                euclidean_distance = np.linalg.norm(
                    coordinates - coordinates_hat, axis=1
                )

                np.testing.assert_array_less(np.mean(euclidean_distance), 0.1)
                # TODO: decide threshold for euclidean distance

        else:  # orientation_type == "free":
            if n_times > 1:
                fwd = convert_forward_solution(fwd)

                active_set = np.linalg.norm(x, axis=2) != 0

                temp = np.linalg.norm(x_hat, axis=2)
                temp = np.linalg.norm(temp, axis=1)
                temp_ = np.partition(-temp, nnz)
                max_temp = -temp_[:nnz]  # get n(=nnz) max amplitudes
                max_temp = np.delete(max_temp, np.where(max_temp == 0.0))
                active_set_hat = np.array(list(map(max_temp.__contains__, temp)))
                active_set_hat = np.repeat(active_set_hat, 3).reshape(
                    active_set_hat.shape[0], -1
                )

                stc = _make_sparse_stc(
                    x[active_set], active_set, fwd, tmin=1, tstep=1
                )  # ground truth
                stc_hat = _make_sparse_stc(
                    x_hat[active_set_hat], active_set_hat, fwd, tmin=1, tstep=1
                )  # estimate

                # euclidean distance check
                lh_coordinates = fwd["src"][0]["rr"][stc.lh_vertno]
                lh_coordinates_hat = fwd["src"][0]["rr"][stc_hat.lh_vertno]
                rh_coordinates = fwd["src"][1]["rr"][stc.rh_vertno]
                rh_coordinates_hat = fwd["src"][1]["rr"][stc_hat.rh_vertno]
                coordinates = np.concatenate([lh_coordinates, rh_coordinates], axis=0)
                coordinates_hat = np.concatenate(
                    [lh_coordinates_hat, rh_coordinates_hat], axis=0
                )
                euclidean_distance = np.linalg.norm(
                    coordinates - coordinates_hat, axis=1
                )

                np.testing.assert_array_less(np.mean(euclidean_distance), 0.1)
                # TODO: decide threshold for euclidean distance

        if save_estimates:

            import os

            PATH_TO_SAVE_ESTIMATES = "bsi_zoo/tests/data/estimates/%s/nnz_%d/%s" % (
                orientation_type,
                nnz,
                subject,
            )

            if not os.path.exists(PATH_TO_SAVE_ESTIMATES):
                os.makedirs(PATH_TO_SAVE_ESTIMATES)

            x_name = solver.__name__ + "_x_" + str(n_times) + ".npy"
            x_hat_name = solver.__name__ + "_x_hat_" + str(n_times) + ".npy"
            # save files
            np.save(os.path.join(PATH_TO_SAVE_ESTIMATES, x_name), x)
            np.save(os.path.join(PATH_TO_SAVE_ESTIMATES, x_hat_name), x_hat)
