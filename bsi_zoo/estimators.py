import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn import linear_model


def _solve_lasso(Lw, y, alpha, max_iter):
    model = linear_model.LassoLars(
        max_iter=max_iter, normalize=False, fit_intercept=False, alpha=alpha
    )
    return model.fit(Lw, y).coef_.copy()


def _solve_reweighted_lasso(L, y, alpha, weights, max_iter, max_iter_reweighting, gprime):
    _, n_sources = L.shape
    x = np.zeros(n_sources)

    for _ in range(max_iter_reweighting):
        L_w = L / weights[np.newaxis, :]
        coef_ = _solve_lasso(L_w, y, alpha, max_iter=max_iter)
        x = coef_ / weights
        weights[:] = gprime(x)  # modify weights inplace on purpose
    return x


def reweighted_lasso(
    L, y, alpha=0.2, max_iter=2000, max_iter_reweighting=100, tol=1e-4
):
    """Reweighted Lasso estimator with L1 regularizer.

    The optimization objective for Reweighted Lasso is::
        (1 / (2 * n_samples)) * ||y - Lx||^2_Fro + alpha * ||x||_1

    Where::
        ||x||_1 = sum_i sum_j |x_ij|

    Parameters
    ----------
    L : array, shape (n_sensors, n_sources)
        lead field matrix modeling the forward operator or dictionary matrix
    y : array, shape (n_sensors,)
        measurement vector, capturing sensor measurements
    alpha : float
        Constant that makes a trade-off between the data fidelity and
        regularizer. Defaults to 0.2
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
    x : array, shape (n_sources,)
        Parameter vector, e.g., source vector in the context of BSI (x in the
        cost function formula).
    """
    # XXX cov is not used
    n_samples, n_sources = L.shape

    x = np.zeros(n_sources)
    weights = np.ones_like(x)
    x_old = x.copy()

    loss_ = []

    alpha_max = abs(L.T.dot(y)).max() / len(L)
    alpha = alpha * alpha_max

    for i in range(max_iter_reweighting):
        Lw = L * weights
        x = _solve_lasso(Lw, y, alpha, max_iter)
        x = x * weights
        err = abs(x - x_old).max()
        err /= max(abs(x_old).max(), abs(x_old).max(), 1.0)
        x_old = x.copy()
        weights = 2 * (abs(x) ** 0.5 + 1e-10)
        obj = 0.5 * ((L @ x - y) ** 2).sum() / n_samples
        obj += (alpha * abs(x) ** 0.5).sum()
        loss_.append(obj)
        if err < tol and i:
            break

    if i == max_iter_reweighting - 1 and i:
        warnings.warn(
            "Reweighted objective did not converge."
            " You might want to increase "
            "the number of iterations of reweighting."
            " Fitting data with very small alpha"
            " may cause precision problems.",
            ConvergenceWarning,
        )

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
    y : array, shape (n_sensors,)
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
    x : array, shape (n_sources,)
        Parameter vector, e.g., source vector in the context of BSI (x in the cost
        function formula).

    References
    ----------
    XXX
    """
    # XXX cov is not used
    eps = np.finfo(float).eps
    _, n_sources = L.shape
    weights = np.ones(n_sources)

    def gprime(w):
        return 1.0 / (np.abs(w) + eps)

    alpha_max = abs(L.T.dot(y)).max() / len(L)
    alpha = alpha * alpha_max

    x = _solve_reweighted_lasso(L, y, alpha, weights, max_iter, max_iter_reweighting, gprime)

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
    y : array, shape (n_sensors,)
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
    x : array, shape (n_sources,)
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

    def gprime(w):
        return 1.0 / ((w ** 2) + eps)

    alpha_max = abs(L.T.dot(y)).max() / len(L)
    alpha = alpha * alpha_max

    x = _solve_reweighted_lasso(L, y, alpha, weights, max_iter, max_iter_reweighting, gprime)

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
    y : array, shape (n_sensors,)
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
    x : array, shape (n_sources,)
        Parameter vector, e.g., source vector in the context of BSI (x in the cost function formula).

    References
    ----------
    TODO
    """
    # XXX : cov is not used
    eps = np.finfo(float).eps
    _, n_sources = L.shape
    weights = np.ones(n_sources)

    def gprime(w):
        return 1.0 / (2.0 * np.sqrt(np.abs(w)) + eps)

    alpha_max = abs(L.T.dot(y)).max() / len(L)
    alpha = alpha * alpha_max

    x = _solve_reweighted_lasso(L, y, alpha, weights, max_iter, max_iter_reweighting, gprime)

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
    y : array, shape (n_sensors,)
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
    x : array, shape (n_sources,)
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
        L_T = L.T

        def w_mat(weights):
            return np.diag(1.0 / weights)

        x_mat = np.abs(np.diag(coef))
        noise_cov = cov
        proj_source_cov = (L @ np.dot(w_mat(weights), x_mat)) @ L_T
        signal_cov = noise_cov + proj_source_cov
        sigmaY_inv = np.linalg.inv(signal_cov)

        return np.sqrt(np.diag((L_T @ sigmaY_inv) @ L))

    x = _solve_reweighted_lasso(L, y, alpha, weights, max_iter, max_iter_reweighting, gprime)

    return x


def iterative_L2_typeII(L, y, cov=1., alpha=0.2, max_iter=1000, max_iter_reweighting=10):
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
    weights = np.ones(n_sources)

    alpha_max = abs(L.T.dot(y)).max() / len(L)
    alpha = alpha * alpha_max

    if isinstance(cov, float):
        cov = cov * np.eye(n_sensors)

    # def epsilon_update(L, w, alpha, cov):
    #     L_T = L.T
    #     n_samples, _ = L.shape

    #     def w_mat(w):
    #         return np.diag(1 / w)

    #     noise_cov = cov  # extension of method by importing the noise covariance
    #     proj_source_cov = (L @ w_mat(w)) @ L_T
    #     signal_cov = noise_cov + proj_source_cov
    #     sigmaY_inv = np.linalg.inv(signal_cov)
    #     return np.diag(
    #         w_mat(w)
    #         - np.multiply(
    #             w_mat(w ** 2), np.diag((L_T @ sigmaY_inv) @ L)
    #         )
    #     )

    # for _ in range(max_iter_reweighting):
    #     L_w = L / weights[np.newaxis, :]
    #     coef_ = _solve_lasso(L_w, y, alpha, max_iter=max_iter)
    #     x = coef_ / weights
    #     epsilon = epsilon_update(L, weights, alpha, cov)
    #     weights = 1.0 / ((x ** 2) + epsilon)

    def gprime(w):
        L_T = L.T
        n_samples, _ = L.shape

        def w_mat(w):
            return np.diag(1 / w)
        
        def epsilon_update(L, w, alpha, cov):
            noise_cov = cov  # extension of method by importing the noise covariance
            proj_source_cov = (L @ w_mat(w)) @ L_T
            signal_cov = noise_cov + proj_source_cov
            sigmaY_inv = np.linalg.inv(signal_cov)
            return np.diag(
                w_mat(w)
                - np.multiply(
                    w_mat(w ** 2), np.diag((L_T @ sigmaY_inv) @ L)
                )
            )

        return 1.0 / ((w ** 2) + epsilon_update(L, weights, alpha, cov))

    x = _solve_reweighted_lasso(L, y, alpha, weights, max_iter, max_iter_reweighting, gprime)

    return x
