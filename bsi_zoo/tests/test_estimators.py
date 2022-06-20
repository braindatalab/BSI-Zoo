import numpy as np
from scipy import linalg
import pytest

from config import get_leadfield_path, get_fwd_fname
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
        if orientation_type == "fixed":  # TODO: support for free orientation
            if n_times > 1:
                from mne.inverse_sparse.mxne_inverse import _make_sparse_stc
                from mne import read_forward_solution, convert_forward_solution

                fwd_fname = get_fwd_fname(subject)
                fwd = read_forward_solution(fwd_fname)
                fwd = convert_forward_solution(fwd, force_fixed=True)

                active_set = np.linalg.norm(x, axis=1) != 0
                active_set_hat = np.linalg.norm(x_hat, axis=1) != 0

                stc = _make_sparse_stc(
                    x[active_set], active_set, fwd, tmin=1, tstep=1
                )  # ground truth
                stc_hat = _make_sparse_stc(
                    x_hat[active_set_hat], active_set_hat, fwd, tmin=1, tstep=1
                )  # estimate

                # euclidean distance check
                # supports only nnz=1 case
                # TODO: support for nnz>1
                if nnz == 1:

                    for hemishpere_index, hemi_ in zip(
                        [0, 1], ["lh", "rh"]
                    ):  # 0->lh, 1->rh
                        hemisphere, hemisphere_hat = (
                            stc.vertices[hemishpere_index],
                            stc_hat.vertices[hemishpere_index],
                        )
                        if (
                            hemisphere.any() and hemisphere_hat.any()
                        ):  # if that hemisphere has a source
                            vertice_index = hemisphere[0]
                            # find peak amplitude vertex in estimated
                            peak_vertex, peak_time = stc_hat.get_peak(
                                hemi=hemi_, vert_as_index=True, time_as_index=True
                            )
                            vertice_index_hat = (
                                stc_hat.lh_vertno[peak_vertex]
                                if hemi_ == "lh"
                                else stc_hat.rh_vertno[peak_vertex]
                            )

                            coordinates = fwd["src"][hemishpere_index]["rr"][
                                vertice_index
                            ]
                            coordinates_hat = fwd["src"][hemishpere_index]["rr"][
                                vertice_index_hat
                            ]
                            euclidean_distance = np.linalg.norm(
                                coordinates - coordinates_hat
                            )

                            np.testing.assert_array_less(euclidean_distance, 0.1)
                            # TODO: decide threshold for euclidean distance

            else:
                from mne.inverse_sparse.mxne_inverse import _make_sparse_stc
                from mne import read_forward_solution, convert_forward_solution

                fwd_fname = get_fwd_fname(subject)
                fwd = read_forward_solution(fwd_fname)
                fwd = convert_forward_solution(fwd, force_fixed=True)

                active_set = x != 0
                active_set_hat = x_hat != 0

                stc = _make_sparse_stc(
                    x[active_set], active_set, fwd, tmin=1, tstep=1
                )  # ground truth
                stc_hat = _make_sparse_stc(
                    x_hat[active_set_hat], active_set_hat, fwd, tmin=1, tstep=1
                )  # estimate

                for hemishpere_index, hemi_ in zip(
                    [0, 1], ["lh", "rh"]
                ):  # 0->lh, 1->rh
                    hemisphere, hemisphere_hat = (
                        stc.vertices[hemishpere_index],
                        stc_hat.vertices[hemishpere_index],
                    )
                    if (
                        hemisphere.any() and hemisphere_hat.any()
                    ):  # if that hemisphere has a source
                        vertice_index = hemisphere[0]
                        vertice_index_hat = hemisphere_hat[0]

                        coordinates = fwd["src"][hemishpere_index]["rr"][vertice_index]
                        coordinates_hat = fwd["src"][hemishpere_index]["rr"][
                            vertice_index_hat
                        ]
                        euclidean_distance = np.linalg.norm(
                            coordinates - coordinates_hat
                        )

                        np.testing.assert_array_less(euclidean_distance, 0.1)
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
