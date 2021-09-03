import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn import linear_model


def solver_lasso(Lw, y, alpha, max_iter):
    model = linear_model.LassoLars(
        max_iter=max_iter, normalize=False, fit_intercept=False, alpha=alpha
    )
    return model.fit(Lw, y).coef_.copy()


def reweighted_lasso(L, y, cov, alpha_fraction=0.01, max_iter=2000,
                     max_iter_reweighting=100, tol=1e-4):
    """Reweighted Lasso estimator with L1 regularizer.

    The optimization objective for Reweighted Lasso is::
        (1 / (2 * n_samples)) * ||y - Lx||^2_Fro + alpha * ||x||_1

    Where::
        ||x||_1 = sum_i sum_j |x_ij|

    Parameters
    ----------
    L: array, shape=(n_sensors, n_sources)
        lead field matrix modeling the forward operator or dictionary matrix
    y: array, shape=(n_sensors,)
        measurement vector, capturing sensor measurements 
    cov : array, shape=(n_sensors, n_sensors)
        noise covariance matrix
    alpha : (float), 
        Constant that makes a trade-off between the data fidelity and regularizer. Defaults to 1.0
    max_iter : int, optional
        The maximum number of inner loop iterations
    max_iter_reweighting : int, optional
        Maximum number of reweighting steps i.e outer loop iterations
    tol : float, optional
        The tolerance for the optimization: if the updates are
        smaller than ``tol``, the optimization code checks the
        dual gap for optimality and continues until it is smaller
        than ``tol``.

    Attributes
    ----------
    x : array, shape=(n_sources,)
        Parameter vector, e.g., source vector in the context of BSI (x in the cost function formula).
    """
    n_samples, n_sources = L.shape

    x = np.zeros(n_sources)
    weights = np.ones_like(x)
    x_old = x.copy()

    loss_ = []

    alpha_max = abs(L.T.dot(y)).max() / len(L)
    alpha = alpha_fraction * alpha_max

    for i in range(max_iter_reweighting):
        Lw = L * weights
        x = solver_lasso(Lw, y, alpha, max_iter)
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


def iterative_L1(L, y, cov, alpha=0.2, maxiter=10):
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
    L: array, shape=(n_sensors, n_sources)
        lead field matrix modeling the forward operator or dictionary matrix
    y: array, shape=(n_sensors,)
        measurement vector, capturing sensor measurements 
    alpha : (float), 
        Constant that makes a trade-off between the data fidelity and regularizer. Defaults to 1.0
    max_iter : int, optional
        The maximum number of inner loop iterations
    cov : noise covariance matrix shape=(n_sensors,n_sensors)
    max_iter_reweighting : int, optional
        Maximum number of reweighting steps i.e outer loop iterations
    tol : float, optional
        The tolerance for the optimization: if the updates are
        smaller than ``tol``, the optimization code checks the
        dual gap for optimality and continues until it is smaller
        than ``tol``.

    Attributes
    ----------
    x : array, shape (n_sources,)
        Parameter vector, e.g., source vector in the context of BSI (x in the cost function formula).
    
    References: 
    """
    n_samples, n_sources = L.shape
    weights = np.ones(n_sources)
    eps = np.finfo(float).eps

    def gprime(w):
        return 1.0 / (np.abs(w) + eps)

    alpha_max = abs(L.T.dot(y)).max() / len(L)
    alpha = alpha * alpha_max

    for k in range(maxiter):
        L_w = L / weights[np.newaxis, :]
        clf = linear_model.LassoLars(alpha=alpha, fit_intercept=False,
                                     normalize=False)
        clf.fit(L_w, y)
        x = clf.coef_ / weights
        weights = gprime(x)

    return x

def iterative_L2(L, y, cov, alpha=0.2, maxiter=10):
    """Iterative Type-I estimator with L2 regularizer.

    The optimization objective for iterative estimators in general is::
        x^(k+1) <-- argmin_x ||y - Lx||^2_Fro + alpha * sum_i g(x_i)
    
    Which in the case of iterative L2, g(x_i) and w_i are define as follows::
    Iterative L2::
        g(x_i) = log(x_i^2 + epsilon)
        w_i^(k+1) <-- [(x_i^(k))^2+epsilon]

    for solving the following problem:
        x^(k+1) <-- argmin_x ||y - Lx||^2_Fro + alpha * sum_i w_i^(k)|x_i|

    Parameters
    ----------
    L: array, shape=(n_sensors, n_sources)
        lead field matrix modeling the forward operator or dictionary matrix
    y: array, shape=(n_sensors,)
        measurement vector, capturing sensor measurements 
    alpha : (float), 
        Constant that makes a trade-off between the data fidelity and regularizer. Defaults to 1.0
    max_iter : int, optional
        The maximum number of inner loop iterations
    cov : noise covariance matrix shape=(n_sensors,n_sensors)
    max_iter_reweighting : int, optional
        Maximum number of reweighting steps i.e outer loop iterations
    tol : float, optional
        The tolerance for the optimization: if the updates are
        smaller than ``tol``, the optimization code checks the
        dual gap for optimality and continues until it is smaller
        than ``tol``.

    Attributes
    ----------
    x : array, shape=(n_sources,)
        Parameter vector, e.g., source vector in the context of BSI (x in the cost function formula).
    
    References: 
    """
    n_samples, n_sources = L.shape
    weights = np.ones(n_sources)
    eps = np.finfo(float).eps

    def gprime(w):
        return 1.0 / ((w ** 2) + eps)

    alpha_max = abs(L.T.dot(y)).max() / len(L)
    alpha = alpha * alpha_max

    for k in range(maxiter):
        L_w = L / weights[np.newaxis, :]
        clf = linear_model.LassoLars(alpha=alpha, fit_intercept=False,
                                     normalize=False)
        clf.fit(L_w, y)
        x = clf.coef_ / weights
        weights = gprime(x)

    return x

def iterative_sqrt(L, y, cov, alpha=0.2, maxiter=10):
    """Iterative type-I estimator with L_0.5 regularizer.
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
    L: array, shape=(n_sensors, n_sources)
        lead field matrix modeling the forward operator or dictionary matrix
    y: array, shape=(n_sensors,)
        measurement vector, capturing sensor measurements 
    alpha : (float), 
        Constant that makes a trade-off between the data fidelity and regularizer. Defaults to 1.0
    max_iter : int, optional
        The maximum number of inner loop iterations
    cov : noise covariance matrix shape=(n_sensors,n_sensors)
    max_iter_reweighting : int, optional
        Maximum number of reweighting steps i.e outer loop iterations
    tol : float, optional
        The tolerance for the optimization: if the updates are
        smaller than ``tol``, the optimization code checks the
        dual gap for optimality and continues until it is smaller
        than ``tol``.
    Attributes
    ----------
    x : array, shape=(n_sources,)
        Parameter vector, e.g., source vector in the context of BSI (x in the cost function formula).
    
    References: 
    """
    n_samples, n_sources = L.shape
    weights = np.ones(n_sources)
    eps = np.finfo(float).eps
    
    def gprime(w):
        return 1. / (2. * np.sqrt(np.abs(w)) + eps)
    
    alpha_max = abs(L.T.dot(y)).max() / len(L)
    alpha = alpha * alpha_max
    
    for k in range(maxiter):
        L_w = L / weights[np.newaxis, :]

        clf = linear_model.LassoLars(alpha=alpha, fit_intercept=False, 
                                     normalize=False)
        clf.fit(L_w, y)
        x = clf.coef_ / weights
        weights = gprime(x)
    
    return x


def iterative_L1_typeII(L, y, cov, alpha=0.2, maxiter=10):
    """Iterative type-II estimator with L_1 regularizer.
    The optimization objective for iterative type-II methods is::
        x^(k+1) <-- argmin_x ||y - Lx||^2_Fro + alpha * g_SBl(x)
    
    Which in the case of iterative L1 type-II , g_SBl(x) and w_i are define 
    as follows::
    
    Iterative-L1-typeII::
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
    L : array, shape=(n_sensors, n_sources)
        lead field matrix modeling the forward operator or dictionary matrix
    y : array, shape=(n_sensors,)
        measurement vector, capturing sensor measurements 
    alpha : (float), 
        Constant that makes a trade-off between the data fidelity and regularizer. Defaults to 1.0
    max_iter : int, optional
        The maximum number of inner loop iterations
    cov : noise covariance matrix shape=(n_sensors,n_sensors)
    max_iter_reweighting : int, optional
        Maximum number of reweighting steps i.e outer loop iterations
    tol : float, optional
        The tolerance for the optimization: if the updates are
        smaller than ``tol``, the optimization code checks the
        dual gap for optimality and continues until it is smaller
        than ``tol``.
    Attributes
    ----------
    x : array, shape=(n_sources,)
        Parameter vector, e.g., source vector in the context of BSI (x in the cost function formula).
    
    References: 
    """
    def gprime(L_, coef, w, alpha):
        L_T = L_.T
        n_samples, _ = L_.shape

        def w_mat(w):
            return np.diag(1. / w)

        x_mat = np.abs(np.diag(coef))
        noise_cov = alpha * np.eye(n_samples)
        ## TODO: Replace matmul with @ for simplicity and efficiency 
        proj_source_cov = np.matmul(np.matmul(L_, np.dot(w_mat(w), x_mat)),
                                    L_T)
        signal_cov = noise_cov + proj_source_cov
        sigmaY_inv = np.linalg.inv(signal_cov)

        return np.sqrt(np.diag(np.matmul(np.matmul(L_T, sigmaY_inv), L_)))

    n_samples, n_sources = L.shape
    weights = np.ones(n_sources)

    alpha_max = abs(L.T.dot(y)).max() / len(L)
    alpha = alpha * alpha_max

    for k in range(maxiter):
        L_w = L / weights[np.newaxis, :]

        clf = linear_model.LassoLars(alpha=alpha, fit_intercept=False,
                                     normalize=False)
        clf.fit(L_w, y)
        x = clf.coef_ / weights
        weights = gprime(L, x, weights, alpha)

    return x


def iterative_L2_typeII(L, y, cov, alpha=0.2, maxiter=10):

    def epsilon_update(L, w, alpha):
        L_T = L.T
        n_samples, _ = L.shape

        def w_mat(w):
            return np.diag(1 / w)

        noise_cov = alpha * np.eye(n_samples)
        proj_source_cov = np.matmul(np.matmul(L, w_mat(w)), L_T)
        signal_cov = noise_cov + proj_source_cov
        sigmaY_inv = np.linalg.inv(signal_cov)
        return np.diag(w_mat(w) - np.multiply((w_mat(w**2)),
                                            np.diag(np.matmul(np.matmul(L_T,
                                                                sigmaY_inv),
                                                                L))))

    n_samples, n_sources = L.shape
    weights = np.ones(n_sources)

    alpha_max = abs(L.T.dot(y)).max() / len(L)
    alpha = alpha * alpha_max

    for k in range(maxiter):
        L_w = L / weights[np.newaxis, :]
        clf = linear_model.LassoLars(alpha=alpha, fit_intercept=False,
                                    normalize=False)
        clf.fit(L_w, y)
        x = clf.coef_ / weights
        epsilon = epsilon_update(L, weights, alpha)
        weights = 1. / ((x ** 2) + epsilon)

    return x