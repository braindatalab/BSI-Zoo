from mne.utils import logger, warn
from numpy.core.fromnumeric import mean
from numpy.lib import diag
from scipy.sparse import spdiags

from scipy import linalg
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, ClassifierMixin

import warnings

warnings.filterwarnings("ignore")


def groups_norm2(A, n_orient):
    """Compute squared L2 norms of groups inplace."""
    n_positions = A.shape[0] // n_orient
    return np.sum(np.power(A, 2, A).reshape(n_positions, -1), axis=1)


def _solve_lasso(Lw, y, alpha, max_iter):
    if y.ndim == 1:
        model = linear_model.LassoLars(
            max_iter=max_iter, normalize=False, fit_intercept=False, alpha=alpha
        )
        x = model.fit(Lw, y).coef_.copy()
        x = x.T
    else:
        model = linear_model.MultiTaskLasso(
            max_iter=max_iter, normalize=False, fit_intercept=False, alpha=alpha
        )
        x = model.fit(Lw, y).coef_.copy()
        x = x.T
    return x


def _solve_reweighted_lasso(
    L, y, alpha, weights, max_iter, max_iter_reweighting, gprime
):
    assert max_iter_reweighting > 0

    for _ in range(max_iter_reweighting):
        L_w = L * weights[np.newaxis, :]
        coef_ = _solve_lasso(L_w, y, alpha, max_iter=max_iter)
        if y.ndim == 1:
            x = coef_ * weights
        else:
            x = coef_ * weights[:, np.newaxis]
        weights = gprime(x)

    return x


class Solver(BaseEstimator, ClassifierMixin):
    def __init__(self, solver, alpha=None, cov=None):
        self.solver = solver
        self.alpha = alpha
        self.cov = cov

    def fit(self, L, y):
        if self.cov is None:
            return self.solver(L, y, alpha=self.alpha)
        else:
            return self.solver(L, y, self.cov, alpha=self.alpha)


def estimator(solver, L, y, cov=None):
    alphas = np.linspace(0.01, 5, 20)
    clf = GridSearchCV(
        estimator=Solver(solver, cov=cov),
        param_grid=dict(alpha=alphas),
        scoring="neg_mean_squared_error",
        cv=5,
        n_jobs=-1,
    )
    clf.fit(L, y)
    if cov is None:
        x = solver(L, y, alpha=clf.best_estimator_.alpha)
    else:
        x = solver(L, y, cov, alpha=clf.best_estimator_.alpha)

    return x


def iterative_L1(L, y, alpha=0.2, max_iter=1000, max_iter_reweighting=10):
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
        Defaults to 1.0
    max_iter : int, optional
        The maximum number of inner loop iterations
    max_iter_reweighting : int, optional
        Maximum number of reweighting steps i.e outer loop iterations
    Returns
    -------
    y : array, shape (n_sensors,) or (n_sensors, n_times)
        Parameter vector, e.g., source vector in the context of BSI (x in the cost
        function formula).
    References
    ----------
    XXX
    """
    eps = np.finfo(float).eps
    _, n_sources = L.shape
    weights = np.ones(n_sources)
    n_orient = 1

    def g(w):
        return np.sqrt(groups_norm2(w.copy(), n_orient))

    def gprime(w):
        return np.repeat(g(w), n_orient).ravel() + eps

    alpha_max = abs(L.T.dot(y)).max() / len(L)
    alpha = alpha * alpha_max

    x = _solve_reweighted_lasso(
        L, y, alpha, weights, max_iter, max_iter_reweighting, gprime
    )

    return x


def iterative_L2(L, y, alpha=0.2, max_iter=1000, max_iter_reweighting=10):
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
    max_iter : int, optional
        The maximum number of inner loop iterations
    max_iter_reweighting : int, optional
        Maximum number of reweighting steps i.e outer loop iterations
    tol : float, optional
        The tolerance for the optimization: if the updates are
        smaller than ``tol``, the optimization code checks the
        dual gap for optimality and continues until it is smaller
        than ``tol``.
    Returns
    -------
    y : array, shape (n_sensors,) or (n_sensors, n_times)
        Parameter vector, e.g., source vector in the context of BSI (x in the cost
        function formula).
    References
    ----------
    TODO
    """
    # XXX : cov is not used
    eps = np.finfo(float).eps
    _, n_sources = L.shape
    weights = np.ones(n_sources)
    n_orient = 1

    def g(w):
        return groups_norm2(w.copy(), n_orient)

    def gprime(w):
        return np.repeat(g(w), n_orient).ravel() + eps

    alpha_max = abs(L.T.dot(y)).max() / len(L)
    alpha = alpha * alpha_max

    x = _solve_reweighted_lasso(
        L, y, alpha, weights, max_iter, max_iter_reweighting, gprime
    )

    return x


def iterative_sqrt(L, y, alpha=0.2, max_iter=1000, max_iter_reweighting=10):
    """Iterative Type-I estimator with L_0.5 regularizer.
    The optimization objective for iterative estimators in general is::
        x^(k+1) <-- argmin_x ||y - Lx||^2_Fro + alpha * sum_i g(x_i)
    Which in the case of iterative "sqrt", g(x_i) and w_i are define as follows::
    Iterative sqrt (L_0.5)::
        g(x_i) = sqrt(|x_i|)
        w_i^(k+1) <-- [2sqrt(|x_i|)+epsilon]^-1
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
    TODO
    """
    _, n_sources = L.shape
    weights = np.ones(n_sources)
    n_orient = 1

    def g(w):
        return np.sqrt(np.sqrt(groups_norm2(w.copy(), n_orient)))

    def gprime(w):
        return 2.0 * np.repeat(g(w), n_orient).ravel()

    alpha_max = abs(L.T.dot(y)).max() / len(L)
    alpha = alpha * alpha_max

    x = _solve_reweighted_lasso(
        L, y, alpha, weights, max_iter, max_iter_reweighting, gprime
    )

    return x


def iterative_L1_typeII(L, y, cov, alpha=0.2, max_iter=1000, max_iter_reweighting=10):
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
    max_iter : int, optional
        The maximum number of inner loop iterations
    max_iter_reweighting : int, optional
        Maximum number of reweighting steps i.e outer loop iterations
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
        n_orient = 1
        L_T = L.T

        def g(weights):
            return np.sqrt(groups_norm2(weights.copy(), n_orient))

        def w_mat(weights):
            return np.diag(1.0 / np.repeat(g(weights), n_orient).ravel())

        if coef.ndim < 2:
            x_mat = np.abs(np.diag(coef))
            # X = coef[:, np.newaxis] @ coef[:, np.newaxis].T
            # x_mat = np.diag(np.sqrt(np.diag(X)))
        else:
            X = coef @ coef.T
            x_mat = np.diag(linalg.norm(X, axis=0))
        noise_cov = cov
        proj_source_cov = (L @ np.dot(w_mat(weights), x_mat)) @ L_T
        signal_cov = noise_cov + proj_source_cov
        sigmaY_inv = linalg.inv(signal_cov)

        return 1.0 / (np.sqrt(np.diag((L_T @ sigmaY_inv) @ L)))

    x = _solve_reweighted_lasso(
        L, y, alpha, weights, max_iter, max_iter_reweighting, gprime
    )

    return x


def iterative_L2_typeII(
    L, y, cov=1.0, alpha=0.2, max_iter=1000, max_iter_reweighting=10
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
        n_orient = 1

        def g(weights):
            return np.sqrt(groups_norm2(weights.copy(), n_orient))

        def w_mat(weights):
            return np.diag(1.0 / np.repeat(g(weights), n_orient).ravel())

        def epsilon_update(L, weights, cov):
            noise_cov = cov  # extension of method by importing the noise covariance
            proj_source_cov = (L @ w_mat(weights)) @ L_T
            signal_cov = noise_cov + proj_source_cov
            sigmaY_inv = linalg.inv(signal_cov)
            return np.diag(
                w_mat(weights)
                - np.multiply(w_mat(weights ** 2), np.diag((L_T @ sigmaY_inv) @ L))
            )

        def g_coef(coef):
            return groups_norm2(coef.copy(), n_orient)

        def gprime_coef(coef):
            return np.repeat(g_coef(coef), n_orient).ravel()

        return gprime_coef(coef) + epsilon_update(L, weights, cov)

    x = _solve_reweighted_lasso(
        L, y, alpha, weights, max_iter, max_iter_reweighting, gprime
    )

    return x


def gamma_map(
    L,
    y,
    cov=1.0,
    alpha=0.2,
    max_iter=1000,
    tol=1e-15,
    update_mode=2,
    threshold=1e-5,
    gammas=None,
    group_size=1,
):
    """Gamma_map method based on MNE package

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
    max_iter : int, optional
        The maximum number iterations
    tol : float
        Tolerance parameter for convergence.
    update_mode : int
        Update mode, 1: MacKay update, 2: Convex-bounding update (defaul),
        3: Expectation-Maximization update
    threshold : float
        A threshold paramter for forcing to zero the small values in
        reconstrcuted gamma in each iteration
    gammas : array, shape=(n_sources,)
        Initial values for posterior variances (gammas). If None, a
        variance of 1.0 is used.
    group_size : int
        Number of consecutive sources which use the same gamma.

    Returns
    -------
    x : array, shape (n_sources,)
        Parameter vector, e.g., source vector in the context of BSI (x in the cost
        function formula).

    References
    ----------
    XXX
    """
    eps = np.finfo(float).eps
    n_sensors, n_sources = L.shape
    if y.ndim < 2:
        y = y[:, np.newaxis]
    n_times = y.shape[1]
    coef = np.zeros((n_sources, n_times))

    if isinstance(cov, float):
        cov = cov * np.eye(n_sensors)

    alpha = mean(np.diag(cov))

    if gammas is None:
        gammas = np.ones(L.shape[1])
        # L_square = np.sum(L ** 2,axis=0)
        # inv_L_square = np.zeros(n_sources)
        # L_nonzero_index = L_square > 0
        # inv_L_square[L_nonzero_index] = 1.0 / L_square[L_nonzero_index]
        # w_filter = spdiags(inv_L_square, 0, n_sources, n_sources) @ L.T
        # vec_init = mean(mean(w_filter @ y) ** 2)
        # gammas = vec_init * np.ones(L.shape[1])

    # # # apply normalization so the numerical values are sane
    # y_normalize_constant = np.linalg.norm(np.dot(y, y.T), ord='fro')
    # y /= np.sqrt(y_normalize_constant)
    # alpha /= y_normalize_constant
    # cov /= y_normalize_constant
    # L_normalize_constant = np.linalg.norm(L, ord=np.inf)
    # L /= L_normalize_constant

    threshold = 0.2 * alpha

    if n_sources % group_size != 0:
        raise ValueError(
            "Number of sources has to be evenly dividable by the " "group size"
        )

    n_active = n_sources
    active_set = np.arange(n_sources)

    gammas_full_old = gammas.copy()
    # x_bar_old = coef

    if update_mode == 2:
        denom_fun = np.sqrt
    elif update_mode == 1:
        # do nothing
        def denom_fun(x):
            return x

    else:
        denom = None

    last_size = -1
    for iter_no in range(max_iter):
        gammas[np.isnan(gammas)] = 0.0
        gidx = np.abs(gammas) > threshold
        active_set = active_set[gidx]
        gammas = gammas[gidx]

        # update only active gammas (once set to zero it stays at zero)
        if n_active > len(active_set):
            n_active = active_set.size
            L = L[:, gidx]

        Sigma_y = np.dot(L * gammas[np.newaxis, :], L.T)
        Sigma_y.flat[:: n_sensors + 1] += alpha
        # Sigma_y += cov

        # Invert CM keeping symmetry
        U, S, _ = linalg.svd(Sigma_y, full_matrices=False)
        S = S[np.newaxis, :]
        del Sigma_y

        Sigma_y_inv = np.dot(U / (S + eps), U.T)
        Sigma_y_invL = np.dot(Sigma_y_inv, L)
        A = np.dot(Sigma_y_invL.T, y)  # mult. w. Diag(gamma) in gamma update

        # heteroscedastic update rule
        W = np.dot(np.diag(gammas), np.dot(L.T, Sigma_y_inv))
        x_bar = np.dot(W, y)
        residual = y - np.dot(L, x_bar)

        M_noise = np.dot(residual, residual.T) / n_times
        # cov = np.diag(np.sqrt((np.diag(M_noise) / np.diag(Sigma_y_inv))))
        alpha = np.mean(np.sqrt((np.diag(M_noise) / np.diag(Sigma_y_inv))))

        # # M_N = linalg.norm(M - np.dot(G, gammas[:, None] * A), ord = 'fro') ** 2 / n_samples
        # # Lambda = np.diag(np.sqrt(np.divide(M_N, CMinv)))
        # # alpha2 = np.mean(np.diag(Lambda))

        # # homoscedastic update rule
        # LW = np.identity(n_sensors)-np.dot(L,W)
        # Cyy = np.dot(y, y.T) / n_times
        # noise_numer = np.mean(np.sum(np.dot(np.dot(LW,Cyy),LW),1))
        # noise_denom = np.mean(np.diag(Sigma_y_inv))
        # alpha = np.sqrt(noise_numer / noise_denom)

        if update_mode == 1:
            # MacKay fixed point update
            numer = gammas ** 2 * np.mean((A * A.conj()).real, axis=1)
            denom = gammas * np.sum(L * Sigma_y_invL, axis=0)
        elif update_mode == 2:
            # convex-bounding update
            numer = gammas * np.sqrt(np.mean((A * A.conj()).real, axis=1))
            denom = np.sum(L * Sigma_y_invL, axis=0)  # sqrt is applied below
        elif update_mode == 3:
            # Expectation Maximization (EM) update
            numer = gammas ** 2 * np.mean((A * A.conj()).real, axis=1) + gammas * (
                1 - gammas * np.sum(L * Sigma_y_invL, axis=0)
            )
        else:
            raise ValueError("Invalid value for update_mode")

        # # learn the regularization parameter (noise variance)

        # if noise_update_mode == 0:
        #     # do nothing : conventinal champagne
        #     pass
        # elif noise_update_mode == 0.5:
        #     # homoscedastic learning
        #     Sigma_X_diag = gammas * (1 - gammas * np.sum(G * CMinvG, axis=0));  # posterior covariance
        #     numer_noise = linalg.norm(M - np.dot(G, gammas[:, None] * A), ord = 'fro') ** 2 / n_times
        #     denom_noise = n_sensors - len(active_set) + np.sum(np.divide(Sigma_X_diag,gammas))
        #     alpha = numer_noise / denom_noise
        # elif noise_update_mode == 1:
        #     # homoscedastic learning (using special case of heteroscedastic learning)
        #     M_N = linalg.norm(M - np.dot(G, gammas[:, None] * A), ord = 'fro') ** 2 / n_times
        #     Lambda = np.diag(np.sqrt(np.divide(M_N, CMinv)))
        #     alpha = np.mean(np.diag(Lambda))
        # elif noise_update_mode == 2:
        #     # heteroscedastic learning
        #     M_N = linalg.norm(M - np.dot(G, gammas[:, None] * A), ord = 'fro') ** 2 / n_times
        #     Lambda = np.diag(np.sqrt(np.divide(M_N, CMinv)))
        # elif noise_update_mode == 3:
        #     # Full-structural noise (FUN) learning
        #     M_N = linalg.norm(M - np.dot(G, gammas[:, None] * A), ord = 'fro') ** 2 / n_times
        #     Lambda = np.matmul(np.matmul(linalg.sqrtm(CM),linalg.sqrtm(np.matmul(np.matmul(linalg.sqrtm(CM),M_N),linalg.sqrtm(CM)))),linalg.sqrtm(CM))
        # elif noise_update_mode == 4:
        #     pass
        #     # Spatial CV
        #     # (hint) using sklearn gridsearch and model selection built-in functions for tuning the hyperparamters.
        # elif noise_update_mode == 5:
        #     pass
        #     # Temporal CV
        #     # (hint) using the type-II loss as a metric for minimizing the Bregman distanct

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

        # compute the noise covariance
        err = np.sum(np.abs(gammas_full - gammas_full_old)) / np.sum(
            np.abs(gammas_full_old)
        )

        # err_x = linalg.norm(x_bar - x_bar_old, ord = 'fro')
        # print(err_x)

        gammas_full_old = gammas_full

        breaking = err < tol or n_active == 0
        if len(gammas) != last_size or breaking:
            logger.info(
                "Iteration: %d\t active set size: %d\t convergence: "
                "%0.3e" % (iter_no, len(gammas), err)
            )
            last_size = len(gammas)

        if breaking:
            break

    if iter_no < max_iter - 1:
        logger.info("\nConvergence reached !\n")
    else:
        warn("\nConvergence NOT reached !\n")

    # undo normalization and compute final posterior mean

    # n_const = np.sqrt(y_normalize_constant) / L_normalize_constant
    n_const = 1
    x_active = n_const * gammas[:, None] * A

    coef[active_set, :] = x_active
    if n_times == 1:
        # x = np.squeeze(coef,axis = 1)
        x = coef[:, 0]
    else:
        x = coef
    return x


def champagne(L, y, cov=1.0, alpha=0.2, max_iter=1000, max_iter_reweighting=10):
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
    n_sensors, n_sources = L.shape
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
