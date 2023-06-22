from mne.utils import logger, warn
from mne.inverse_sparse.mxne_optim import groups_norm2, _mixed_norm_solver_bcd
from numpy.core.fromnumeric import mean
from numpy.lib import diag
from scipy.sparse import spdiags

from scipy import linalg
import numpy as np
from functools import partial
import warnings
from mne.utils import sqrtm_sym, eigh
from mne.fixes import _safe_svd
from sklearn import linear_model
from sklearn.base import BaseEstimator, ClassifierMixin


def _solve_lasso(Lw, y, alpha, max_iter):
    if y.ndim == 1:
        model = linear_model.LassoLars(
            max_iter=max_iter, fit_intercept=False, alpha=alpha
        )
        x = model.fit(Lw, y).coef_.copy()
        x = x.T
    else:
        model = linear_model.MultiTaskLasso(
            max_iter=max_iter, fit_intercept=False, alpha=alpha
        )
        x = model.fit(Lw, y).coef_.copy()
        x = x.T
    return x


def _normalize_R(G, R, G_3, n_nzero, force_equal, n_src, n_orient):
    """Normalize R so that lambda2 is consistent."""
    if n_orient == 1 or force_equal:
        R_Gt = R[:, np.newaxis] * G.T
    else:
        R_Gt = np.matmul(R, G_3).reshape(n_src * 3, -1)
    G_R_Gt = G @ R_Gt
    norm = np.trace(G_R_Gt) / n_nzero
    G_R_Gt /= norm
    R /= norm
    return G_R_Gt


def _get_G_3(G, n_orient):
    if n_orient == 1:
        return None
    else:
        return G.reshape(G.shape[0], -1, n_orient).transpose(1, 2, 0)


def _R_sqrt_mult(other, R_sqrt):
    """Do other @ R ** 0.5."""
    if R_sqrt.ndim == 1:
        assert other.shape[1] == R_sqrt.size
        out = R_sqrt * other
    else:
        assert R_sqrt.shape[1:3] == (3, 3)
        assert other.shape[1] == np.prod(R_sqrt.shape[:2])
        assert other.ndim == 2
        n_src = R_sqrt.shape[0]
        n_chan = other.shape[0]
        out = (
            np.matmul(R_sqrt, other.reshape(n_chan, n_src, 3).transpose(1, 2, 0))
            .reshape(n_src * 3, n_chan)
            .T
        )
    return out


def _compute_reginv2(sing, n_nzero, lambda2):
    """Safely compute reginv from sing."""
    sing = np.array(sing, dtype=np.float64)
    reginv = np.zeros_like(sing)
    sing = sing[:n_nzero]
    with np.errstate(invalid="ignore"):  # if lambda2==0
        reginv[:n_nzero] = np.where(sing > 0, sing / (sing ** 2 + lambda2), 0)
    return reginv


def _compute_orient_prior(G, n_orient, loose=0.9):
    n_sources = G.shape[1]
    orient_prior = np.ones(n_sources, dtype=np.float64)
    if n_orient == 1:
        return orient_prior
    orient_prior[::3] *= loose
    orient_prior[1::3] *= loose
    return orient_prior


def _compute_eloreta_kernel(L, *, lambda2, n_orient, whitener, loose=1.0, max_iter=20):
    """Compute the eLORETA solution."""
    options = dict(eps=1e-6, max_iter=max_iter, force_equal=False)  # taken from mne
    eps, max_iter = options["eps"], options["max_iter"]
    force_equal = bool(options["force_equal"])  # None means False

    G = whitener @ L
    n_nzero = G.shape[0]

    # restore orientation prior
    source_std = np.ones(G.shape[1])

    orient_prior = _compute_orient_prior(G, n_orient, loose=loose)
    source_std *= np.sqrt(orient_prior)

    G *= source_std

    # We do not multiply by the depth prior, as eLORETA should compensate for
    # depth bias.
    _, n_src = G.shape
    n_src //= n_orient

    assert n_orient in (1, 3)

    # src, sens, 3
    G_3 = _get_G_3(G, n_orient)
    if n_orient != 1 and not force_equal:
        # Outer product
        R_prior = source_std.reshape(n_src, 1, 3) * source_std.reshape(n_src, 3, 1)
    else:
        R_prior = source_std ** 2

    # The following was adapted under BSD license by permission of Guido Nolte
    if force_equal or n_orient == 1:
        R_shape = (n_src * n_orient,)
        R = np.ones(R_shape)
    else:
        R_shape = (n_src, n_orient, n_orient)
        R = np.empty(R_shape)
        R[:] = np.eye(n_orient)[np.newaxis]
    R *= R_prior
    _this_normalize_R = partial(
        _normalize_R,
        n_nzero=n_nzero,
        force_equal=force_equal,
        n_src=n_src,
        n_orient=n_orient,
    )
    G_R_Gt = _this_normalize_R(G, R, G_3)
    extra = " (this make take a while)" if n_orient == 3 else ""
    for kk in range(max_iter):
        # 1. Compute inverse of the weights (stabilized) and C
        s, u = eigh(G_R_Gt)
        s = abs(s)
        sidx = np.argsort(s)[::-1][:n_nzero]
        s, u = s[sidx], u[:, sidx]
        with np.errstate(invalid="ignore"):
            s = np.where(s > 0, 1 / (s + lambda2), 0)
        N = np.dot(u * s, u.T)
        del s

        # Update the weights
        R_last = R.copy()
        if n_orient == 1:
            R[:] = 1.0 / np.sqrt((np.dot(N, G) * G).sum(0))
        else:
            M = np.matmul(np.matmul(G_3, N[np.newaxis]), G_3.swapaxes(-2, -1))
            if force_equal:
                _, s = sqrtm_sym(M, inv=True)
                R[:] = np.repeat(1.0 / np.mean(s, axis=-1), 3)
            else:
                R[:], _ = sqrtm_sym(M, inv=True)
        R *= R_prior  # reapply our prior, eLORETA undoes it
        G_R_Gt = _this_normalize_R(G, R, G_3)

        # Check for weight convergence
        delta = np.linalg.norm(R.ravel() - R_last.ravel()) / np.linalg.norm(
            R_last.ravel()
        )
        if delta < eps:
            break
    else:
        warnings.warn("eLORETA weight fitting did not converge (>= %s)" % eps)
    del G_R_Gt
    G /= source_std  # undo our biasing
    G_3 = _get_G_3(G, n_orient)
    _this_normalize_R(G, R, G_3)
    del G_3
    if n_orient == 1 or force_equal:
        R_sqrt = np.sqrt(R)
    else:
        R_sqrt = sqrtm_sym(R)[0]
    assert R_sqrt.shape == R_shape
    A = _R_sqrt_mult(G, R_sqrt)
    # del R, G  # the rest will be done in terms of R_sqrt and A
    eigen_fields, sing, eigen_leads = _safe_svd(A, full_matrices=False)
    # del A
    reginv = _compute_reginv2(sing, n_nzero, lambda2)
    eigen_leads = _R_sqrt_mult(eigen_leads, R_sqrt).T
    trans = np.dot(eigen_fields.T, whitener)
    trans *= reginv[:, None]
    K = np.dot(eigen_leads, trans)
    return K


def _solve_reweighted_lasso(
    L, y, alpha, n_orient, weights, max_iter, max_iter_reweighting, gprime
):
    assert max_iter_reweighting > 0

    for _ in range(max_iter_reweighting):
        mask = weights > 0  # ignore dipoles with zero weights
        L_w = L[:, mask] * weights[np.newaxis, mask]
        assert np.isnan(weights).sum() == 0
        if n_orient > 1:
            n_positions = L_w.shape[1] // n_orient
            lc = np.empty(n_positions)
            for j in range(n_positions):
                L_j = L_w[:, (j * n_orient) : ((j + 1) * n_orient)]
                lc[j] = np.linalg.norm(np.dot(L_j.T, L_j), ord=2)
            coef_, active_set, _ = _mixed_norm_solver_bcd(
                y,
                L_w,
                alpha,
                lipschitz_constant=lc,
                maxit=max_iter,
                tol=1e-8,
                n_orient=n_orient,
                # use_accel=False,
            )
            x = np.zeros((L.shape[1], y.shape[1]))
            mask[mask] = active_set
            if y.ndim == 1:
                x[mask] = coef_ * weights[mask]
            else:
                x[mask] = coef_ * weights[mask, np.newaxis]
            assert np.isnan(x).sum() == 0
        else:
            coef_ = _solve_lasso(L_w, y, alpha, max_iter=max_iter)
            x = np.zeros((L.shape[1], y.shape[1]))
            if y.ndim == 1:
                x[mask] = coef_ * weights
            else:
                x[mask] = coef_ * weights[mask, np.newaxis]
        weights = gprime(x)

    return x


def _gamma_map_opt(
    M,
    G,
    alpha,
    maxit=10000,
    tol=1e-6,
    update_mode=1,
    group_size=1,
    gammas=None,
    verbose=None,
):
    """Hierarchical Bayes (Gamma-MAP).

    Parameters
    ----------
    M : array, shape=(n_sensors, n_times)
        Observation.
    G : array, shape=(n_sensors, n_sources)
        Forward operator.
    alpha : float
        Regularization parameter (noise variance).
    maxit : int
        Maximum number of iterations.
    tol : float
        Tolerance parameter for convergence.
    group_size : int
        Number of consecutive sources which use the same gamma.
    update_mode : int
        Update mode, 1: MacKay update (default), 3: Modified MacKay update.
    gammas : array, shape=(n_sources,)
        Initial values for posterior variances (gammas). If None, a
        variance of 1.0 is used.
    %(verbose)s

    Returns
    -------
    X : array, shape=(n_active, n_times)
        Estimated source time courses.
    active_set : array, shape=(n_active,)
        Indices of active sources.
    """
    from scipy import linalg

    G = G.copy()
    M = M.copy()

    if gammas is None:
        gammas = np.ones(G.shape[1], dtype=np.float64)

    eps = np.finfo(float).eps

    n_sources = G.shape[1]
    n_sensors, n_times = M.shape

    # apply normalization so the numerical values are sane
    M_normalize_constant = np.linalg.norm(np.dot(M, M.T), ord="fro")
    M /= np.sqrt(M_normalize_constant)
    alpha /= M_normalize_constant
    G_normalize_constant = np.linalg.norm(G, ord=np.inf)
    G /= G_normalize_constant

    if n_sources % group_size != 0:
        raise ValueError(
            "Number of sources has to be evenly dividable by the " "group size"
        )

    n_active = n_sources
    active_set = np.arange(n_sources)

    gammas_full_old = gammas.copy()

    if update_mode == 2:
        denom_fun = np.sqrt
    else:
        # do nothing
        def denom_fun(x):
            return x

    last_size = -1
    for itno in range(maxit):
        gammas[np.isnan(gammas)] = 0.0

        gidx = np.abs(gammas) > eps
        active_set = active_set[gidx]
        gammas = gammas[gidx]

        # update only active gammas (once set to zero it stays at zero)
        if n_active > len(active_set):
            n_active = active_set.size
            G = G[:, gidx]

        CM = np.dot(G * gammas[np.newaxis, :], G.T)
        CM.flat[:: n_sensors + 1] += alpha
        # Invert CM keeping symmetry
        U, S, _ = linalg.svd(CM, full_matrices=False)
        S = S[np.newaxis, :]
        del CM
        CMinv = np.dot(U / (S + eps), U.T)
        CMinvG = np.dot(CMinv, G)
        A = np.dot(CMinvG.T, M)  # mult. w. Diag(gamma) in gamma update

        if update_mode == 1:
            # MacKay fixed point update (10) in [1]
            numer = gammas ** 2 * np.mean((A * A.conj()).real, axis=1)
            denom = gammas * np.sum(G * CMinvG, axis=0)
        elif update_mode == 2:
            # modified MacKay fixed point update (11) in [1]
            numer = gammas * np.sqrt(np.mean((A * A.conj()).real, axis=1))
            denom = np.sum(G * CMinvG, axis=0)  # sqrt is applied below
        else:
            raise ValueError("Invalid value for update_mode")

        if group_size == 1:
            if denom is None:
                gammas = numer
            else:
                gammas = numer / np.maximum(denom_fun(denom), np.finfo("float").eps)
        else:
            numer_comb = np.sum(numer.reshape(-1, group_size), axis=1)
            if denom is None:
                gammas_comb = numer_comb
            else:
                denom_comb = np.sum(denom.reshape(-1, group_size), axis=1)
                gammas_comb = numer_comb / denom_fun(denom_comb)

            gammas = np.repeat(gammas_comb / group_size, group_size)

        # compute convergence criterion
        gammas_full = np.zeros(n_sources, dtype=np.float64)
        gammas_full[active_set] = gammas

        err = np.sum(np.abs(gammas_full - gammas_full_old)) / np.sum(
            np.abs(gammas_full_old)
        )

        gammas_full_old = gammas_full

        breaking = err < tol or n_active == 0
        if len(gammas) != last_size or breaking:
            logger.info(
                "Iteration: %d\t active set size: %d\t convergence: "
                "%0.3e" % (itno, len(gammas), err)
            )
            last_size = len(gammas)

        if breaking:
            break

    if itno < maxit - 1:
        logger.info("\nConvergence reached !\n")
    else:
        warn("\nConvergence NOT reached !\n")

    # undo normalization and compute final posterior mean
    n_const = np.sqrt(M_normalize_constant) / G_normalize_constant
    x_active = n_const * gammas[:, None] * A

    return x_active, active_set


class Solver(BaseEstimator, ClassifierMixin):
    def __init__(self, solver, alpha, cov_type, cov, n_orient, extra_params={}):
        self.solver = solver
        self.alpha = alpha
        self.cov = cov
        self.cov_type = cov_type
        self.n_orient = n_orient
        self.extra_params = extra_params

    def fit(self, L, y):
        self.L_ = L
        self.y_ = y

        return self

    def _get_coef(self, y):
        if self.cov_type == "diag":
            coef = self.solver(
                self.L_,
                y,
                alpha=self.alpha,
                n_orient=self.n_orient,
                **self.extra_params
            )
        else:
            coef = self.solver(
                self.L_,
                y,
                self.cov,
                alpha=self.alpha,
                n_orient=self.n_orient,
                **self.extra_params
            )
        return coef

    def predict(self, y):
        return self._get_coef(y)


class SpatialSolver(Solver):
    def fit(self, X, y):
        self.L_ = X
        self.coef_ = self._get_coef(y)
        return self

    def predict(self, X):
        return X @ self.coef_

    def score(self, X, y):
        return -np.mean(self.predict(X) - y) ** 2


def fake_solver(L, y, alpha, n_orient, **kwargs):
    # from sklearn.linear_model import Ridge

    # estimator = Ridge(alpha=alpha, fit_intercept=False)
    # x = estimator.fit(L, y).coef_.T

    depth_prior = 1.0
    depth_scaling = np.linalg.norm(L, axis=0) ** depth_prior
    L = L / depth_scaling[np.newaxis, :]

    K = L.T @ np.linalg.inv(L @ L.T + alpha * np.eye(len(L)))
    x = K @ y

    x /= depth_scaling[:, np.newaxis]

    return x


def iterative_L1(L, y, alpha=0.2, n_orient=1, max_iter=1000, max_iter_reweighting=10):
    """Iterative Type-I estimator with L1 regularizer.

    The optimization objective for iterative estimators in general is::
        x^(k+1) <-- argmin_x ||y - Lx||^2_Fro + alpha * sum_i g(x_i)
    Which in the case of iterative L1, it boils down to::
        x^(k+1) <-- argmin_x ||y - Lx||^2_Fro + alpha * sum_i w_i^(k)|x_i|
    Iterative L1::
        g(x_i) = log(|x_i| + epsilon)
        w_i^(k+1) <-- [|x_i^(k)|+epsilon]

    Parameters
    ----------
    L : array, shape (n_sensors, n_sources)
        lead field matrix modeling the forward operator or dictionary matrix
    y : array, shape (n_sensors,) or (n_sensors, n_times)
        measurement vector, capturing sensor measurements
    alpha : float
        Constant that makes a trade-off between the data fidelity and regularizer.
        Defaults to 0.2.
    n_orient : int
        Number of dipoles per location (typically 1 or 3).
    max_iter : int, optional
        The maximum number of inner loop iterations. Defaults to 1000.
    max_iter_reweighting : int, optional
        Maximum number of reweighting steps i.e outer loop iterations. Defaults to 10.

    Returns
    -------
    y : array, shape (n_sensors,) or (n_sensors, n_times)
        Parameter vector, e.g., source vector in the context of BSI (x in the cost
        function formula).

    References
    ----------
    [1] Candes, Wakin, Boyd, "Enhancing Sparsity by Reweighted l1 Minimization",
    J Fourier Anal Appl (2008) 14: 877–905
    https://web.stanford.edu/~boyd/papers/pdf/rwl1.pdf
    """
    eps = np.finfo(float).eps
    _, n_sources = L.shape
    weights = np.ones(n_sources)

    def gprime(w):
        grp_norms = np.sqrt(groups_norm2(w.copy(), n_orient))
        return np.repeat(grp_norms, n_orient).ravel() + eps

    alpha_max = abs(L.T.dot(y)).max() / len(L)
    alpha = alpha * alpha_max

    x = _solve_reweighted_lasso(
        L, y, alpha, n_orient, weights, max_iter, max_iter_reweighting, gprime
    )

    return x


def iterative_L2(L, y, alpha=0.2, n_orient=1, max_iter=1000, max_iter_reweighting=10):
    """Iterative Type-I estimator with L2 regularizer.

    The optimization objective for iterative estimators in general is::
        x^(k+1) <-- argmin_x ||y - Lx||^2_Fro + alpha * sum_i g(x_i)
    Which in the case of iterative L2, g(x_i) and w_i are defined as follows::
    Iterative L2::
        g(x_i) = log(x_i^2 + epsilon)
        w_i^(k+1) <-- [(x_i^(k))^2+epsilon]
    for solving the following problem:
        x^(k+1) <-- argmin_x ||y - Lx||^2_Fro + alpha * sum_i w_i^(k)|x_i|

    Parameters
    ----------
    L : array, shape (n_sensors, n_sources)
        lead field matrix modeling the forward operator or dictionary matrix
    y : array, shape (n_sensors,) or (n_sensors, n_times)
        measurement vector, capturing sensor measurements
    alpha : float
        Constant that makes a trade-off between the data fidelity and regularizer.
        Defaults to 0.2.
    n_orient : int
        Number of dipoles per location (typically 1 or 3).
    max_iter : int, optional
        The maximum number of inner loop iterations. Defaults to 1000.
    max_iter_reweighting : int, optional
        Maximum number of reweighting steps i.e outer loop iterations. Defaults to 10.

    Returns
    -------
    y : array, shape (n_sensors,) or (n_sensors, n_times)
        Parameter vector, e.g., source vector in the context of BSI (x in the cost
        function formula).

    References
    ----------
    TODO
    """
    eps = np.finfo(float).eps
    _, n_sources = L.shape
    weights = np.ones(n_sources)

    def gprime(w):
        grp_norm2 = groups_norm2(w.copy(), n_orient)
        return np.repeat(grp_norm2, n_orient).ravel() + eps

    alpha_max = abs(L.T.dot(y)).max() / len(L)
    alpha = alpha * alpha_max

    x = _solve_reweighted_lasso(
        L, y, alpha, n_orient, weights, max_iter, max_iter_reweighting, gprime
    )

    return x


def iterative_sqrt(L, y, alpha=0.2, n_orient=1, max_iter=1000, max_iter_reweighting=10):
    """Iterative Type-I estimator with L_0.5 regularizer.

    The optimization objective for iterative estimators in general is::
        x^(k+1) <-- argmin_x ||y - Lx||^2_Fro + alpha * sum_i g(x_i)

    Which in the case of iterative "sqrt", g(x_i) and w_i are define as follows::

    Iterative sqrt (L_0.5)::
        g(x_i) = sqrt(|x_i|)
        w_i^(k+1) <-- [2 sqrt(|x_i|)+epsilon]^-1
    for solving the following problem:
        x^(k+1) <-- argmin_x ||y - Lx||^2_Fro + alpha * sum_i w_i^(k)|x_i|

    Parameters
    ----------
    L : array, shape (n_sensors, n_sources)
        lead field matrix modeling the forward operator or dictionary matrix
    y : array, shape (n_sensors,) or (n_sensors, n_times)
        measurement vector, capturing sensor measurements
    alpha : float
        Constant that makes a trade-off between the data fidelity and regularizer.
        Defaults to 0.2.
    n_orient : XXX
    max_iter : int, optional
        The maximum number of inner loop iterations
    max_iter_reweighting : int, optional
        Maximum number of reweighting steps i.e outer loop iterations

    Returns
    -------
    y : array, shape (n_sensors,) or (n_sensors, n_times)
        Parameter vector, e.g., source vector in the context of BSI (x in the cost function formula).

    References
    ----------
    [1] Strohmeier, D., Bekthi, Y., Haueisen, J., & Gramfort, A. (2016). The iterative
        reweighted Mixed-Norm Estimate for spatio-temporal MEG/EEG source reconstruction.
        IEEE Transactions on Medical Imaging Year : 2016
    """
    _, n_sources = L.shape
    weights = np.ones(n_sources)

    def g(w):
        return np.sqrt(np.sqrt(groups_norm2(w.copy(), n_orient)))

    def gprime(w):
        return 2.0 * np.repeat(g(w), n_orient).ravel()

    alpha_max = abs(L.T.dot(y)).max() / len(L)
    alpha = alpha * alpha_max

    x = _solve_reweighted_lasso(
        L, y, alpha, n_orient, weights, max_iter, max_iter_reweighting, gprime
    )

    return x


def eloreta(y, L, alpha=1 / 9, cov=1, n_orient=1):
    if isinstance(cov, (float, int)):
        cov = alpha * np.eye(L.shape[0])
    # Take care of whitening
    whitener = linalg.inv(linalg.sqrtm(cov))
    y = whitener @ y
    L = whitener @ L

    # alpha is lambda2
    K = _compute_eloreta_kernel(L, lambda2=alpha, n_orient=n_orient, whitener=whitener)
    x = K @ y  # get the source time courses with simple dot product
    return x


def iterative_L1_typeII(
    L, y, cov, alpha=0.2, n_orient=1, max_iter=1000, max_iter_reweighting=10
):
    """Iterative Type-II estimator with L_1 regularizer.

    The optimization objective for iterative Type-II methods is::
        x^(k+1) <-- argmin_x ||y - Lx||^2_Fro + alpha * g_SBl(x)
    Which in the case of iterative L1 Type-II , g_SBl(x) and w_i are define
    as follows::
    Iterative-L1-TypeII::
        g_SBl(x) = min_{gamma >=0} x^T*Gamma^-1*x + log|alpha*Id + L*Gamma*L^T|
        w_i^(k+1) <-- [L_i^T*(lambda*Id + L*hat{W}*hat{X}*L^T)^(-1)*L_i]^(1/2)
    where
        Gamma = diag(gamma) : souce covariance matrix
        hat{W} = diag(W)^-1
        hat{X} = diag(X)^-1
    for solving the following problem:
        x^(k+1) <-- argmin_x ||y - Lx||^2_Fro + alpha * sum_i w_i^(k)|x_i|

    NOTE: Please note that lambda models the noise variance and it is a
    different paramter than regularization paramter alpha. For simplicity,
    we assume lambda = alpha to be consistant with sklearn built-in
    function: "linear_model.LassoLars"

    Parameters
    ----------
    L : array, shape (n_sensors, n_sources)
        lead field matrix modeling the forward operator or dictionary matrix
    y : array, shape (n_sensors,) or (n_sensors, n_times)
        measurement vector, capturing sensor measurements
    cov : array, shape (n_sensors, n_sensors)
        noise covariance matrix. If float it corresponds to the noise variance
        assumed to be diagonal.
    alpha : float
        Constant that makes a trade-off between the data fidelity and regularizer.
        Defaults to 0.2
    n_orient : int
        Number of dipoles per locations (typically 1 or 3).
    max_iter : int, optional
        The maximum number of inner loop iterations. Defaults to 1000.
    max_iter_reweighting : int, optional
        Maximum number of reweighting steps i.e outer loop iterations.
        Defaults to 10.

    Returns
    -------
    y : array, shape (n_sensors,) or (n_sensors, n_times)
        Parameter vector, e.g., source vector in the context of BSI (x in the cost
        function formula).

    References
    ----------
    TODO
    """
    n_sensors, n_sources = L.shape
    weights = np.ones(n_sources)

    alpha_max = abs(L.T.dot(y)).max() / len(L)
    alpha = alpha * alpha_max

    if isinstance(cov, float):
        cov = cov * np.eye(n_sensors)

    def gprime(coef):
        def g(weights):
            return np.sqrt(groups_norm2(weights.copy(), n_orient))

        def w_mat(weights):
            # XXX it should be possible to avoid allocating a big matrix
            # of size n_sources x n_sources
            return np.diag(1.0 / np.repeat(g(weights), n_orient).ravel())

        if coef.ndim < 2:
            x_mat = np.abs(np.diag(coef))
            # X = coef[:, np.newaxis] @ coef[:, np.newaxis].T
            # x_mat = np.diag(np.sqrt(np.diag(X)))
        else:
            X = coef @ coef.T
            x_mat = np.diag(linalg.norm(X, axis=0))
        noise_cov = cov
        proj_source_cov = (L @ np.dot(w_mat(weights), x_mat)) @ L.T
        signal_cov = noise_cov + proj_source_cov
        sigmaY_inv = linalg.inv(signal_cov)

        return 1.0 / np.sqrt(np.sum((L.T @ sigmaY_inv) * L.T, axis=1))
        # return 1.0 / (np.sqrt(np.diag((L_T @ sigmaY_inv) @ L)))

    x = _solve_reweighted_lasso(
        L, y, alpha, n_orient, weights, max_iter, max_iter_reweighting, gprime
    )

    return x


def iterative_L2_typeII(
    L, y, cov=1.0, alpha=0.2, n_orient=1, max_iter=1000, max_iter_reweighting=10
):
    """Iterative Type-II estimator with L_2 regularizer.

    The optimization objective for iterative Type-II methods is::
        x^(k+1) <-- argmin_x ||y - Lx||^2_Fro + alpha * g_SBl(x)
    Which in the case of iterative L2 Type-II , g_SBl(x) and w_i are define
    as follows::
    Iterative-L2-TypeII::
        g_SBl(x) = min_{gamma >=0} x^T*Gamma^-1*x + log|alpha*Id + L*Gamma*L^T|
        w_i^(k+1) <-- [(x_i^(k))^2 + (w_i^(k))^(-1) - (w_i^(k))^(-2) * L_i^T*(lambda*Id + L*hat{W^(k)}*L^T)^(-1)*L_i]^(-1)
    where
        Gamma = diag(gamma) : souce covariance matrix
        hat{W} = diag(W)^-1
    for solving the following problem:
        x^(k+1) <-- argmin_x ||y - Lx||^2_Fro + alpha * sum_i w_i^(k)|x_i|

    Notes
    -----
    Please note that lambda models the noise variance and it is a
    different paramter than regularization paramter alpha. For simplicity,
    we assume lambda = alpha to be consistant with sklearn built-in
    function: "linear_model.LassoLars"
    Given the above assumption, one can see the iterative-L2-TypeII
    as an extension of its Type-I counterpart where eps is tuned adaptively::
    w_i^(k+1) <-- [(x_i^(k))^2+epsilon^(k)]
    where
    epsilon^(k) = (w_i^(k))^(-1) - (w_i^(k))^(-2) * L_i^T*(lambda*Id + L*hat{W^(k)}*L^T)^(-1)*L_i

    Parameters
    ----------
    L : array, shape (n_sensors, n_sources)
        lead field matrix modeling the forward operator or dictionary matrix
    y : array, shape (n_sensors,) or (n_sensors, n_times)
        measurement vector, capturing sensor measurements
    cov : float | array, shape (n_sensors, n_sensors)
        noise covariance matrix. If float it corresponds to the noise variance
        assumed to be diagonal.
    alpha : float
        Constant that makes a trade-off between the data fidelity and regularizer.
        Defaults to 0.2
    n_orient : int
        Number of dipoles per locations (typically 1 or 3).
    max_iter : int, optional
        The maximum number of inner loop iterations
    max_iter_reweighting : int, optional
        Maximum number of reweighting steps i.e outer loop iterations

    Returns
    -------
    x : array, shape (n_sources,) or (n_sources, n_times)
        Parameter vector, e.g., source vector in the context of BSI (x in the cost
        function formula).

    References
    ----------
    XXX
    """
    n_sensors, n_sources = L.shape
    weights = np.ones(n_sources)

    alpha_max = abs(L.T.dot(y)).max() / len(L)
    alpha = alpha * alpha_max

    if isinstance(cov, float):
        cov = cov * np.eye(n_sensors)

    def gprime(coef):
        L_T = L.T
        n_samples, _ = L.shape

        def g(weights):
            return np.sqrt(groups_norm2(weights.copy(), n_orient))

        def w_mat(weights):
            return 1.0 / np.repeat(g(weights), n_orient).ravel()

        def epsilon_update(L, weights, cov):
            noise_cov = cov  # extension of method by importing the noise covariance
            weights_ = w_mat(weights)
            proj_source_cov = (L * weights_[np.newaxis, :]) @ L_T
            signal_cov = noise_cov + proj_source_cov
            sigmaY_inv = linalg.inv(signal_cov)
            # Full computation (slow):
            # np.diag(
            #     w_mat(weights)
            #     - np.multiply(w_mat(weights ** 2), np.diag((L_T @ sigmaY_inv) @ L))
            # )
            return weights_ - (weights_ ** 2) * ((L_T @ sigmaY_inv) * L_T).sum(axis=1)

        def g_coef(coef):
            return groups_norm2(coef.copy(), n_orient)

        def gprime_coef(coef):
            return np.repeat(g_coef(coef), n_orient).ravel()

        return gprime_coef(coef) + epsilon_update(L, weights, cov)

    x = _solve_reweighted_lasso(
        L, y, alpha, n_orient, weights, max_iter, max_iter_reweighting, gprime
    )

    return x


def gamma_map(
    L,
    y,
    cov=1.0,
    alpha=0.2,
    n_orient=1,
    max_iter=1000,
    tol=1e-15,
    update_mode=2,
    # threshold=1e-5,
    gammas=None,
    verbose=True,
):
    if isinstance(cov, float):
        cov = alpha * np.eye(L.shape[0])
    # Take care of whitening
    whitener = linalg.inv(linalg.sqrtm(cov))
    y = whitener @ y
    L = whitener @ L
    x_hat_, active_set = _gamma_map_opt(
        y,
        L,
        alpha=alpha,
        tol=tol,
        maxit=max_iter,
        gammas=gammas,
        update_mode=update_mode,
        group_size=n_orient,
        verbose=verbose,
    )
    x_hat = np.zeros((L.shape[1], y.shape[1]))
    x_hat[active_set] = x_hat_

    return x_hat


def champagne(
    L, y, cov=1.0, alpha=0.2, n_orient=1, max_iter=1000, max_iter_reweighting=10
):
    """Champagne method based on our MATLAB codes

    Parameters
    ----------
    L : array, shape (n_sensors, n_sources)
        lead field matrix modeling the forward operator or dictionary matrix
    y : array, shape (n_sensors,)
        measurement vector, capturing sensor measurements
    cov : float | array, shape (n_sensors, n_sensors)
        noise covariance matrix. If float it corresponds to the noise variance
        assumed to be diagonal.
    alpha : float
        Constant that makes a trade-off between the data fidelity and regularizer.
        Defaults to 0.2
    n_orient : int
        Number of orientations per source. Defaults to 1
    max_iter : int, optional
        The maximum number of inner loop iterations
    max_iter_reweighting : int, optional
        Maximum number of reweighting steps i.e outer loop iterations

    Returns
    -------
    x : array, shape (n_sources,)
        Parameter vector, e.g., source vector in the context of BSI (x in the cost
        function formula).

    References
    ----------
    XXX
    """
    # XXX alpha and max_iter_reweighting are not used
    assert n_orient != 1, "Only 1 orientation is supported"
    _, n_sources = L.shape
    _, n_times = y.shape
    gammas = np.ones(n_sources)
    eps = np.finfo(float).eps
    threshold = 0.2 * mean(diag(cov))
    x = np.zeros((n_sources, n_times))
    n_active = n_sources
    active_set = np.arange(n_sources)
    # H = np.concatenate(L, np.eyes(n_sensors), axis = 1)

    for _ in range(max_iter):
        gammas[np.isnan(gammas)] = 0.0
        gidx = np.abs(gammas) > threshold
        active_set = active_set[gidx]
        gammas = gammas[gidx]

        # update only active gammas (once set to zero it stays at zero)
        if n_active > len(active_set):
            n_active = active_set.size
            L = L[:, gidx]

        Gamma = spdiags(gammas, 0, len(active_set), len(active_set))
        Sigma_y = (L @ Gamma @ L.T) + cov
        U, S, _ = linalg.svd(Sigma_y, full_matrices=False)
        S = S[np.newaxis, :]
        del Sigma_y
        Sigma_y_inv = np.dot(U / (S + eps), U.T)
        # Sigma_y_inv = linalg.inv(Sigma_y)
        x_bar = Gamma @ L.T @ Sigma_y_inv @ y
        gammas = np.sqrt(
            np.diag(x_bar @ x_bar.T / n_times) / np.diag(L.T @ Sigma_y_inv @ L)
        )
        e_bar = y - (L @ x_bar)
        cov = np.sqrt(np.diag(e_bar @ e_bar.T / n_times) / np.diag(Sigma_y_inv))
        threshold = 0.2 * mean(diag(cov))

    x[active_set, :] = x_bar

    return x
