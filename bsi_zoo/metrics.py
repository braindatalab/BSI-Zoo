from sklearn.metrics import jaccard_score, mean_squared_error
from bsi_zoo.config import get_fwd_fname
import numpy as np
from mne.inverse_sparse.mxne_inverse import _make_sparse_stc
from mne import read_forward_solution, convert_forward_solution


def jaccard_error(x, x_hat, *args, **kwargs):
    # You can read more on the Jaccard score in Scikit-learn definition https://scikit-learn.org/stable/modules/generated/sklearn.metrics.jaccard_score.html
    return 1 - jaccard_score(x, x_hat, average="samples")


def mse(x, x_hat, *args, **kwargs):
    return mean_squared_error(x, x_hat)


def euclidean_distance(x, x_hat, *args, **kwargs):
    orientation_type = kwargs["orientation_type"]
    subject = kwargs["subject"]
    nnz = kwargs["nnz"]

    fwd_fname = get_fwd_fname(subject)
    fwd = read_forward_solution(fwd_fname)

    if orientation_type == "fixed":
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

    elif orientation_type == "free":
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
    coordinates_hat = np.concatenate([lh_coordinates_hat, rh_coordinates_hat], axis=0)
    euclidean_distance = np.linalg.norm(
        coordinates[: coordinates_hat.shape[0], :] - coordinates_hat, axis=1
    )

    return np.mean(euclidean_distance)
