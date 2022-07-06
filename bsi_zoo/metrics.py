from sklearn.metrics import jaccard_score, mean_squared_error, f1_score
from bsi_zoo.config import get_fwd_fname
import numpy as np
from mne.inverse_sparse.mxne_inverse import _make_sparse_stc
from mne import read_forward_solution, convert_forward_solution
from scipy.spatial.distance import cdist
from ot import emd2


def _get_active_nnz(x, x_hat, orientation_type, subject, nnz):
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

        # temp = np.linalg.norm
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

    return stc, stc_hat, active_set, active_set_hat, fwd


def jaccard_error(x, x_hat, *args, **kwargs):
    # You can read more on the Jaccard score in Scikit-learn definition https://scikit-learn.org/stable/modules/generated/sklearn.metrics.jaccard_score.html
    return 1 - jaccard_score(x, x_hat, average="samples")


def mse(x, x_hat, orientation_type, *args, **kwargs):
    if orientation_type == "free":
        x = np.linalg.norm(x, axis=2)
        x_hat = np.linalg.norm(x_hat, axis=2)

    return mean_squared_error(x, x_hat)


def emd(x, x_hat, orientation_type, subject, *args, **kwargs):

    if orientation_type == "fixed":
        temp = np.linalg.norm(x, axis=1)
        a_mask = temp != 0
        a = temp[a_mask]

        temp = np.linalg.norm(x_hat, axis=1)
        b_mask = temp != 0
        b = temp[b_mask]
        # temp_ = np.partition(-temp, nnz)
        # b = -temp_[:nnz]  # get n(=nnz) max amplitudes
        # b = -temp_[:nnz]  # get n(=nnz) max amplitudes
    elif orientation_type == "free":
        temp = np.linalg.norm(x, axis=2)
        temp = np.linalg.norm(temp, axis=1)
        a_mask = temp != 0
        a = temp[a_mask]

        temp = np.linalg.norm(x_hat, axis=2)
        temp = np.linalg.norm(temp, axis=1)
        b_mask = temp != 0
        b = temp[b_mask]
        # temp_ = np.partition(-temp, nnz)
        # b = -temp_[:nnz]  # get n(=nnz) max amplitudes

    fwd_fname = get_fwd_fname(subject)
    fwd = read_forward_solution(fwd_fname)
    fwd = convert_forward_solution(fwd, force_fixed=True)
    src = fwd["src"]

    stc_a = _make_sparse_stc(a[:, None], a_mask, fwd, tmin=1, tstep=1)
    stc_b = _make_sparse_stc(b[:, None], b_mask, fwd, tmin=1, tstep=1)

    rr_a = np.r_[src[0]["rr"][stc_a.lh_vertno], src[1]["rr"][stc_a.rh_vertno]]
    rr_b = np.r_[src[0]["rr"][stc_b.lh_vertno], src[1]["rr"][stc_b.rh_vertno]]
    M = cdist(rr_a, rr_b, metric="euclidean")

    # Normalize a and b as EMD is defined between probability distributions
    a /= a.sum()
    b /= b.sum()

    return emd2(a, b, M)


def euclidean_distance(x, x_hat, orientation_type, subject, nnz, *args, **kwargs):

    stc, stc_hat, _, _, fwd = _get_active_nnz(x, x_hat, orientation_type, subject, nnz)

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


def nll(x, x_hat, *args, **kwargs):
    y = kwargs["y"]
    L = kwargs["L"]
    cov = kwargs["cov"]
    orientation_type = kwargs["orientation_type"]
    subject = kwargs["subject"]
    nnz = kwargs["nnz"]
    
    # stc, stc_hat, active_set, active_set_hat, fwd = _get_active_nnz(x, x_hat, orientation_type, subject, nnz)#kwargs["active_set"]
    # q = np.zeros(x.shape[0])
    # q[active_set] = 1
    # Marginal NegLogLikelihood score upon estimation of the support:
    # ||(cov + L Q L.T)^-1/2 y||^2_F  + log|cov + L Q L.T| with Q the support matrix
    
    q = np.sum( abs(x_hat) , axis=1) != 0
    
    cov_y = cov + (L * q[:,None])@L.T
    # To take into account the knowledge on nnz you need to add +2log((n_sources-nnz)/nnz)||q||_0
    sign, logdet = np.linalg.slogdet()
    return np.linalg.norm(np.linalg.sqrt(np.linalg.inv(cov_y)),ord='fro')**2 + logdet
    
def f1(x, x_hat, orientation_type, *args, **kwargs):
    if orientation_type == "fixed":
        active_set = np.linalg.norm(x, axis=1) != 0
        active_set_hat = np.linalg.norm(x_hat, axis=1) != 0

    elif orientation_type == "free":
        temp = np.linalg.norm(x, axis=2)
        active_set = np.linalg.norm(temp, axis=1) != 0

        temp = np.linalg.norm(x_hat, axis=2)
        active_set_hat = np.linalg.norm(temp, axis=1) != 0

    return f1_score(active_set, active_set_hat)
