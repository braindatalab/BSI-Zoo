import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn import linear_model


def solver_lasso(Lw, y, alpha, max_iter):
    model = linear_model.LassoLars(max_iter=max_iter, normalize=False,
                                   fit_intercept=False, alpha=alpha)
    return model.fit(Lw, y).coef_.copy()


def reweighted_lasso(L, y, cov, alpha_fraction=0.01, max_iter=2000,
                     max_iter_reweighting=100, tol=1e-4):
    """Reweighted Lasso estimator with L1 regularizer.

    The optimization objective for Reweighted Lasso is::
        (1 / (2 * n_samples)) * ||y - Lx||^2_Fro + alpha * ||x||_0.5

    Where::
        ||x||_0.5 = sum_i sum_j sqrt|x_ij|

    Parameters
    ----------
    alpha : (float or array-like), shape (n_tasks)
        Optional, default ones(n_tasks)
        Constant that multiplies the L0.5 term. Defaults to 1.0
    max_iter : int, optional
        The maximum number of inner loop iterations
    cov : noise covariance matrix
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
        Parameter vector (x in the cost function formula).
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


# class IterativeL1(BaseEstimator, RegressorMixin):
#     """
#     Regression estimator which uses LassoLars algorithm with given alpha
#     normalized for each lead field L and x.
#     """

#     def __init__(self, alpha=0.2, maxiter=10):
#         self.alpha = alpha
#         self.maxiter = maxiter

#     def fit(self, L, x):
#         # eps = 0.01
#         eps = np.finfo(float).eps
#         # L = StandardScaler().fit_transform(L)
#         #   --- Adaptive Lasso for g(|X|) = log(|X| + eps) as a prior (reweithed - \ell_1) ----

#         g = lambda w: np.log(np.abs(w) + eps)
#         gprime = lambda w: 1. / (np.abs(w) + eps)
#         n_samples, n_features = L.shape
#         weights = np.ones(n_features)

#         alpha_max = abs(L.T.dot(x)).max() / len(L)
#         alpha = self.alpha * alpha_max
#         # p_obj = lambda w: 1. / (2 * n_samples) * np.sum((x - np.dot(L, w)) ** 2) \
# #                   + alpha * np.sum(g(w))

#         for k in range(self.maxiter):
#             L_w = L / weights[np.newaxis, :]

#             clf = linear_model.LassoLars(alpha=alpha,
#                                          fit_intercept=False,
#                                          normalize=False)
#             clf.fit(L_w, x)
#             coef_ = clf.coef_ / weights
#             weights = gprime(coef_)
#             # print p_obj(coef_)  # should go down

#         self.coef_ = coef_


def iterative_L2(L, y, cov, alpha=0.2, maxiter=10):

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


# class IterativeSqrt(BaseEstimator, RegressorMixin):
#     def __init__(self, alpha=0.2, maxiter=10):
#         self.alpha = alpha
#         self.maxiter = maxiter

#     def fit(self, L, x):
#         # eps = 0.01
#         eps = np.finfo(float).eps
#         # L = StandardScaler().fit_transform(L)

#         g = lambda w: np.sqrt(np.abs(w))
#         gprime = lambda w: 1. / (2. * np.sqrt(np.abs(w)) + eps)
#         n_samples, n_features = L.shape
#         weights = np.ones(n_features)

#         alpha_max = abs(L.T.dot(x)).max() / len(L)
#         alpha = self.alpha * alpha_max

# #         p_obj = lambda w: 1. / (2 * n_samples) * np.sum((x - np.dot(L, w)) ** 2) \
# #                   + alpha * np.sum(g(w))

#         for k in range(self.maxiter):
#             L_w = L / weights[np.newaxis, :]

#             clf = linear_model.LassoLars(alpha=alpha,
#                                          fit_intercept=False,
#                                          normalize=False)
#             clf.fit(L_w, x)
#             coef_ = clf.coef_ / weights
#             weights = gprime(coef_)
# #             print p_obj(coef_)  # should go down

#         self.coef_ = coef_

# class IterativeL1_TypeII(BaseEstimator, RegressorMixin):
#     def __init__(self, alpha=0.2, maxiter=10):
#         self.alpha = alpha
#         self.maxiter = maxiter

#     def gprime(self, L_, coef, w, alpha):
#         L_T = L_.T
#         n_samples, _ = L_.shape
#         w_mat = lambda w: np.diag(1. /w)

#         x_mat = np.abs(np.diag(coef))
#         noise_cov = alpha*np.eye(n_samples)
#         proj_source_cov = np.matmul(np.matmul(L_, np.dot(w_mat(w),x_mat)),L_T)
#         signal_cov = noise_cov + proj_source_cov
#         sigmaY_inv = np.linalg.inv(signal_cov)

#         return np.sqrt(np.diag(np.matmul(np.matmul(L_T,sigmaY_inv), L_)))

#     def fit(self, L, x):
#         _, n_features = L.shape
#         weights = np.ones(n_features)

#         alpha_max = abs(L.T.dot(x)).max() / len(L)
#         alpha = self.alpha * alpha_max
#         for k in range(self.maxiter):
#             L_w = L / weights[np.newaxis, :]

#             clf = linear_model.LassoLars(alpha=alpha,
#                                          fit_intercept=False,
#                                          normalize=False)
#             clf.fit(L_w, x)
#             coef_ = clf.coef_ / weights
#             weights = self.gprime(L, coef_, weights, alpha)

#         self.coef_ = coef_

# class IterativeL2_TypeII(BaseEstimator, RegressorMixin):
#     def __init__(self, alpha=0.2, maxiter=10):
#         self.alpha = alpha
#         self.maxiter = maxiter

#     def epsilon_update(self, L_, w, alpha):
#         L_T = L_.T
#         n_samples, _ = L_.shape
#         w_mat = lambda w: np.diag(1 /w)
#         noise_cov = alpha*np.eye(n_samples)
#         proj_source_cov = np.matmul(np.matmul(L_,w_mat(w)),L_T)
#         signal_cov = noise_cov + proj_source_cov
#         sigmaY_inv = np.linalg.inv(signal_cov)
#         return np.diag(w_mat(w) - np.multiply((w_mat(w**2)),np.diag(np.matmul(np.matmul(L_T,sigmaY_inv),L_))))

#     def fit(self, L, x):
#         # eps = 0.01
#         eps = np.finfo(float).eps
#         # L = StandardScaler().fit_transform(L)

#         ##  --- Adaptive Lasso for g(|X|) = log(|X^2 + eps|) as a prior (reweithed - \ell_2) ----

#         # g = lambda w: np.log(np.abs((w ** 2) + self.epsilon_update(L, w, alpha)))
#         # gprime = lambda w: 1. / ((w ** 2) + self.epsilon_update(L, w, alpha))

#         g = lambda w: np.log(np.abs((w ** 2) + self.epsilon_update(L, weights, alpha)))
#         gprime = lambda w: 1. / ((w ** 2) + self.epsilon_update(L, weights, alpha))
#         _, n_features = L.shape
#         weights = np.ones(n_features)

#         alpha_max = abs(L.T.dot(x)).max() / len(L)
#         alpha = self.alpha * alpha_max
#         # p_obj = lambda w: 1. / (2 * n_samples) * np.sum((x - np.dot(L, w)) ** 2) \
# #                   + alpha * np.sum(g(w))
#         for k in range(self.maxiter):
#             L_w = L / weights[np.newaxis, :]

#             clf = linear_model.LassoLars(alpha=alpha,
#                                          fit_intercept=False,
#                                          normalize=False)
#             clf.fit(L_w, x)
#             coef_ = clf.coef_ / weights

#             n_samples, _ = L.shape
#             L_T = L.T
#             w_mat = lambda w: np.diag(1 /w)
#             noise_cov = alpha*np.eye(n_samples)
#             proj_source_cov = np.matmul(np.matmul(L,w_mat(weights)),L_T)
#             signal_cov = noise_cov + proj_source_cov
#             sigmaY_inv = np.linalg.inv(signal_cov)
#             epsilon = np.diag(w_mat(weights) - np.multiply((w_mat(weights**2)),np.diag(np.matmul(np.matmul(L_T,sigmaY_inv),L))))
#             weights = 1. / ((coef_ ** 2) + epsilon )
#             # weights = gprime(coef_)
#             #  print p_obj(coef_)  # should go down

#         self.coef_ = coef_

# class GammaMap(BaseEstimator, RegressorMixin):

#     def __init__(self, alpha=0.05, maxiter=100, tol=1e-6, update_mode=2,
#                 gammas=None):
#         self.alpha = alpha
#         self.maxiter = maxiter
#         self.tol = tol
#         self.update_mode = update_mode
#         self.gammas = gammas

#     def fit(self, L, x):
#         # eps = 0.01
#         eps = np.finfo(float).eps
#         gammas = self.gammas
#         L_init = L
#         x_init = x
#         # L = StandardScaler().fit_transform(L)

#         n_sensors, n_sources = L.shape
#         weights = np.zeros(n_sources)
#         if gammas is None:
#             gammas = np.ones(n_sources, dtype=np.float64)

#         # TODO: Initialize it with mathched fileter
#         # MATLAB CODE:
# #       % L_sqaure = sum(L.^2,1);
# #       % inv_L_sqaure = zeros(1,N);
# #       % L_nonzero_index = find(L_sqaure > 0);
# #       % inv_L_sqaure(L_nonzero_index) = 1./L_sqaure(L_nonzero_index);
# #       % w_filter = spdiags(inv_L_sqaure',0,N,N)*L';
# #       % vec_init = mean(mean(w_filter * Y) .^2);
# #       % gamma  = vec_init * ones(N,1);

#         # alpha_max = abs(L.T.dot(x)).max() / len(L)
#         # alpha = self.alpha * alpha_max
#         # alpha = self.alpha * np.cov(x)
#         alpha = self.alpha

#         #tuning alpha using noise covariance
#         import pickle
#         noise_covariance = np.load("./data/test/data_grad_CC120264_450_3/covariance_generate_noise.pickel", allow_pickle=True)

#         print(np.array(noise_covariance))
#         alpha = np.mean(np.diag(noise_covariance))
#         print('Alpha = %f'%alpha)

#         from scipy.linalg import fractional_matrix_power
#         whiten_matrix = fractional_matrix_power(np.linalg.inv(noise_covariance), 0.5)
#         print(whiten_matrix.shape)

#         x = whiten_matrix @ x

#         # apply normalization so the numerical values are sanes
#         x_normalize_constant = linalg.norm(np.dot(x, x.T))
#         x /= np.sqrt(x_normalize_constant)
#         x_init /= np.sqrt(x_normalize_constant)
#         alpha /= x_normalize_constant
#         L_normalize_constant = linalg.norm(L, ord=np.inf)
#         L /= L_normalize_constant

#         n_active = n_sources
#         active_set = np.arange(n_sources)
#         gammas_full_old = gammas.copy()

#         update_mode = self.update_mode
#         if update_mode == 2 or update_mode == 4:
#             denom_fun = np.sqrt
#         else:
#             # do nothing
#             def denom_fun(x):
#                 return x

#         threshold_gammas = eps
#         last_size = -1
#         for k in range(self.maxiter):
#             gammas[np.isnan(gammas)] = 0.0

#             nonzero_idx = (np.abs(gammas) > threshold_gammas)
#             active_set = active_set[nonzero_idx]
#             gammas = gammas[nonzero_idx]

#             # update only active gammas (once set to zero it stays at zero)
#             if n_active > len(active_set):
#                 n_active = active_set.size
#                 L = L[:, nonzero_idx]

#             SigmaY = np.dot(L * gammas[np.newaxis, :], L.T)
#             SigmaY.flat[::n_sensors + 1] += alpha
#             # Invert Sigma_y keeping symmetry
#             U, S, V = np.linalg.svd(SigmaY, full_matrices=False)
#             S = S[np.newaxis, :]
#             del SigmaY
#             SigmaY_inv = np.dot(U / (S + eps), U.T)
#             SigmaY_invL = np.dot(SigmaY_inv, L)
#             A = np.dot(SigmaY_invL.T, x)
# #           x_active = gammas[:, None] * A

#             L_inv = gammas[:, None] * np.dot(L.T,SigmaY_inv)
#             x_active = np.dot(L_inv,x)

# ##            Learn the noise variance accordingto homoscedasstic Champagne. Uncomment when the conventional Champange works well.
# #             Sigma_w_diag = gammas * (1 - gammas * np.sum(L * SigmaY_invL, axis=0));
# #             numer = linalg.norm(x - np.dot(L, x_active)) ** 2
# #             denom = n_sensors - len(active_set) + np.sum(np.divide(Sigma_w_diag,gammas))
# #             alpha = numer / denom;

#             if update_mode == 1:
#                 # Expectation Maximization (EM) update
#                 numer = gammas ** 2 * np.mean((A * A.conj()).real) \
#                         + gammas * (1 - gammas * np.sum(L * SigmaY_invL, axis=0))
#             elif update_mode == 2:
#                 # Convex-bounding update (Champagne)
#                 numer = gammas * np.sqrt(np.mean((A * A.conj()).real))
#                 denom = np.sum(L * SigmaY_invL, axis=0)  # sqrt is applied below
#             elif update_mode == 3:
#                 # MacKay fixed point update
#                 numer = gammas ** 2 * np.mean((A * A.conj()).real)
#                 denom = gammas * np.sum(L * SigmaY_invL, axis=0)
#             elif update_mode == 4:
#                 # LowSNR-BSI update
#                 pass # TODO: Implement LowSNR-BSI that requires whitening the data with noise covariance.
#             else:
#                 raise ValueError('Invalid value for update_mode')


#             if denom is None:
#                 gammas = numer
#             else:
#                 gammas = numer / np.maximum(denom_fun(denom),eps)

#             # compute convergence criterion
#             gammas_full = np.zeros(n_sources, dtype=np.float64)
#             gammas_full[active_set] = gammas

#             err = (np.sum(np.abs(gammas_full - gammas_full_old)) /
#                            np.sum(np.abs(gammas_full_old)))

#             gammas_full_old = gammas_full
#             breaking = (err < self.tol or n_active == 0)
# #             print(k)
#             if len(gammas) != last_size or breaking:
#                 logger.info('Iteration: %d\t active set size: %d\t convergence: '
#                             '%0.3e' % (k, len(gammas), err))
#                 last_size = len(gammas)

#             if breaking:
#                 break

#         if k < self.maxiter - 1:
#             logger.info('\nConvergence reached !\n')
#         else:
#             warn('\nConvergence NOT reached !\n')

#         # undo normalization and compute final posterior mean
# #         n_const = 1
# #         n_const = np.sqrt(x_normalize_constant) / L_normalize_constant
# #         x_active = n_const * gammas[:, None] * A

#         if len(active_set) == 0:
#             raise Exception("No active dipoles found. alpha is too big.")

#         n_const = np.sqrt(x_normalize_constant) / L_normalize_constant
#         weights[active_set] = n_const * x_active

# #         weights[active_set] = x_active
#         coef_ = weights

# #         gammas_diag = spdiags(gammas_full,0,n_sources,n_sources)
# #         L_inv = gammas_diag * np.dot(L_init.T,SigmaY_inv)
# #         coef_ = np.dot(L_inv,x)

# #         gammas_diag = spdiags(gammas_full,0,n_sources,n_sources)
# #         L_inv = gammas_diag * np.dot(L_init.T,SigmaY_inv)
# #         L_inv /= L_normalize_constant
# #         coef_ = np.dot(L_inv,x) * np.sqrt(x_normalize_constant)

#         self.coef_ = coef_

# class GammaMapMNE(BaseEstimator, RegressorMixin):
#     def __init__(self, alpha=0.9, maxiter=500, tol=1e-15, update_mode=2, threshold=1e-5, gammas=None):
#         self.alpha = alpha
#         self.maxiter = maxiter
#         self.tol = tol
#         self.update_mode = update_mode
#         self.gammas = gammas
#         self.group_size = 1
#         self.threshold = threshold

#     def fit(self, L, x):
#         from scipy import linalg
#         G = L.copy()
#         M = x.copy()

#         if self.gammas is None:
#             gammas = np.ones(G.shape[1], dtype=np.float64)

#         eps = np.finfo(float).eps

#         n_sources = G.shape[1]
#         weights = np.zeros(n_sources)
#         n_sensors = M.size

#         import pickle
#         noise_covariance = np.load("./data/test/data_grad_CC120264_450_3/covariance_generate_noise.pickel", allow_pickle=True)
#         noise_covariance = (noise_covariance*1e24)/30

#         # from scipy.linalg import sqrtm
#         # whiten_matrix = np.linalg.inv(sqrtm(noise_covariance))

#         # M = np.real(np.dot(whiten_matrix, M))
#         # noise_covariance = np.real(np.dot(whiten_matrix, noise_covariance))

#         M = M[:, np.newaxis]
#         # # apply normalization so the numerical values are sane
#         M_normalize_constant = np.linalg.norm(np.dot(M, M.T), ord='fro')
#         M /= np.sqrt(M_normalize_constant)
#         self.alpha /= M_normalize_constant
#         noise_covariance /= M_normalize_constant
#         G_normalize_constant = np.linalg.norm(G, ord=np.inf)
#         G /= G_normalize_constant

#         if n_sources % self.group_size != 0:
#             raise ValueError('Number of sources has to be evenly dividable by the '
#                             'group size')

#         n_active = n_sources
#         active_set = np.arange(n_sources)

#         gammas_full_old = gammas.copy()

#         if self.update_mode == 2:
#             denom_fun = np.sqrt
#         else:
#             # do nothing
#             def denom_fun(x):
#                 return x

#         last_size = -1
#         for itno in range(self.maxiter):
#             gammas[np.isnan(gammas)] = 0.0
#             gidx = (np.abs(gammas) > self.threshold)
#             active_set = active_set[gidx]
#             gammas = gammas[gidx]

#             # update only active gammas (once set to zero it stays at zero)
#             if n_active > len(active_set):
#                 n_active = active_set.size
#                 G = G[:, gidx]

#             CM = np.dot(G * gammas[np.newaxis, :], G.T)
#             # CM.flat[::n_sensors + 1] += self.alpha
#             CM += noise_covariance
#             # Invert CM keeping symmetry

#             U, S, _ = linalg.svd(CM, full_matrices=False)
#             S = S[np.newaxis, :]
#             del CM
#             CMinv = np.dot(U / (S + eps), U.T)
#             CMinvG = np.dot(CMinv, G)
#             A = np.dot(CMinvG.T, M)  # mult. w. Diag(gamma) in gamma update

#             if self.update_mode == 1:
#                 # MacKay fixed point update (10) in [1]
#                 numer = gammas ** 2 * np.mean((A * A.conj()).real, axis=1)
#                 denom = gammas * np.sum(G * CMinvG, axis=0)
#             elif self.update_mode == 2:
#                 # modified MacKay fixed point update (11) in [1]
#                 numer = gammas * np.sqrt(np.mean((A * A.conj()).real, axis=1))
#                 denom = np.sum(G * CMinvG, axis=0)  # sqrt is applied below
#             else:
#                 raise ValueError('Invalid value for update_mode')

#             if self.group_size == 1:
#                 if denom is None:
#                     gammas = numer
#                 else:
#                     gammas = numer / np.maximum(denom_fun(denom),
#                                                 np.finfo('float').eps)
#             else:
#                 numer_comb = np.sum(numer.reshape(-1, self.group_size), axis=1)
#                 if denom is None:
#                     gammas_comb = numer_comb
#                 else:
#                     denom_comb = np.sum(denom.reshape(-1, self.group_size), axis=1)
#                     gammas_comb = numer_comb / denom_fun(denom_comb)

#                 gammas = np.repeat(gammas_comb / self.group_size, self.group_size)

#             # compute convergence criterion
#             gammas_full = np.zeros(n_sources, dtype=np.float64)
#             gammas_full[active_set] = gammas

#             err = (np.sum(np.abs(gammas_full - gammas_full_old)) /
#                 np.sum(np.abs(gammas_full_old)))

#             gammas_full_old = gammas_full

#             breaking = (err < self.tol or n_active == 0)
#             if len(gammas) != last_size or breaking:
#                 logger.info('Iteration: %d\t active set size: %d\t convergence: '
#                             '%0.3e' % (itno, len(gammas), err))
#                 last_size = len(gammas)

#             if breaking:
#                 break

#         if itno < self.maxiter - 1:
#             logger.info('\nConvergence reached !\n')
#         else:
#             warn('\nConvergence NOT reached !\n')

#         # undo normalization and compute final posterior mean
#         n_const = np.sqrt(M_normalize_constant) / G_normalize_constant
#         x_active = n_const * gammas[:, None] * A

#         x_active = x_active[:, 0]
#         weights[active_set] = n_const * x_active
#         self.coef_ = weights


# class GammaMapMNENoiseLearning(BaseEstimator, RegressorMixin):
#     def __init__(self, alpha=0.9, maxiter=500, tol=1e-15, update_mode=2, threshold=1e-5, gammas=None):
#         self.alpha = alpha
#         self.maxiter = maxiter
#         self.tol = tol
#         self.update_mode = update_mode
#         self.gammas = gammas
#         self.group_size = 1
#         self.threshold = threshold

#     def fit(self, L, x):
#         from scipy import linalg
#         G = L.copy()
#         M = x.copy()

#         if self.gammas is None:
#             gammas = np.ones(G.shape[1], dtype=np.float64)

#         eps = np.finfo(float).eps

#         n_sources = G.shape[1]
#         # n_samples = M.shape[1] # uncomeent for the time series case
#         n_samples = 1
#         weights = np.zeros(n_sources)
#         n_sensors = M.size

#         import pickle
#         noise_covariance = np.load("./data/test/data_grad_CC120264_450_3/covariance_generate_noise.pickel", allow_pickle=True)
#         noise_covariance = (noise_covariance*1e24)/30

#         # from scipy.linalg import sqrtm
#         # whiten_matrix = np.linalg.inv(sqrtm(noise_covariance))

#         # M = np.real(np.dot(whiten_matrix, M))
#         # noise_covariance = np.real(np.dot(whiten_matrix, noise_covariance))

#         M = M[:, np.newaxis]
#         # # apply normalization so the numerical values are sane
#         M_normalize_constant = np.linalg.norm(np.dot(M, M.T), ord='fro')
#         M /= np.sqrt(M_normalize_constant)
#         self.alpha /= M_normalize_constant
#         noise_covariance /= M_normalize_constant
#         G_normalize_constant = np.linalg.norm(G, ord=np.inf)
#         G /= G_normalize_constant

#         if n_sources % self.group_size != 0:
#             raise ValueError('Number of sources has to be evenly dividable by the '
#                             'group size')

#         n_active = n_sources
#         active_set = np.arange(n_sources)

#         self.alpha = np.mean(np.diag(noise_covariance))
#         gammas_full_old = gammas.copy()
#         x_bar_old = weights

#         if self.update_mode == 2:
#             denom_fun = np.sqrt
#         else:
#             # do nothing
#             def denom_fun(x):
#                 return x

#         last_size = -1
#         for itno in range(self.maxiter):
#             gammas[np.isnan(gammas)] = 0.0
#             gidx = (np.abs(gammas) > self.threshold)
#             active_set = active_set[gidx]
#             gammas = gammas[gidx]

#             # update only active gammas (once set to zero it stays at zero)
#             if n_active > len(active_set):
#                 n_active = active_set.size
#                 G = G[:, gidx]

#             CM = np.dot(G * gammas[np.newaxis, :], G.T)
#             CM.flat[::n_sensors + 1] += self.alpha
#             # CM += noise_covariance

#             # Invert CM keeping symmetry
#             U, S, _ = linalg.svd(CM, full_matrices=False)
#             S = S[np.newaxis, :]
#             del CM

#             CMinv = np.dot(U / (S + eps), U.T)
#             CMinvG = np.dot(CMinv, G)
#             A = np.dot(CMinvG.T, M)  # mult. w. Diag(gamma) in gamma update

#             # heteroscedastic update rule
#             W = np.dot(np.diag(gammas),np.dot(G.T,CMinv))
#             x_bar = np.dot(W,M)
#             residual = M - np.dot(G,x_bar)

#             C_M = np.dot(residual, residual.T) / n_samples
#             # self.alpha = np.mean(np.diag(np.sqrt(np.divide(C_M, CMinv))))

#             # M_N = linalg.norm(M - np.dot(G, gammas[:, None] * A), ord = 'fro') ** 2 / n_samples
#             # Lambda = np.diag(np.sqrt(np.divide(M_N, CMinv)))
#             # alpha2 = np.mean(np.diag(Lambda))

#             # homoscedastic update rule
#             LW = np.identity(n_sensors)-np.dot(G,W)
#             Cyy = np.dot(M, M.T) / n_samples
#             noise_numer = np.mean(np.sum(np.dot(np.dot(LW,Cyy),LW),1))
#             noise_denom = np.mean(np.diag(CMinv))
#             self.alpha = np.sqrt(noise_numer / noise_denom)

#             if self.update_mode == 1:
#                 # MacKay fixed point update (10) in [1]
#                 numer = gammas ** 2 * np.mean((A * A.conj()).real, axis=1)
#                 denom = gammas * np.sum(G * CMinvG, axis=0)
#             elif self.update_mode == 2:
#                 # modified MacKay fixed point update (11) in [1]
#                 numer = gammas * np.sqrt(np.mean((A * A.conj()).real, axis=1))
#                 denom = np.sum(G * CMinvG, axis=0)  # sqrt is applied below
#             else:
#                 raise ValueError('Invalid value for update_mode')

#             if self.group_size == 1:
#                 if denom is None:
#                     gammas = numer
#                 else:
#                     gammas = numer / np.maximum(denom_fun(denom),
#                                                 np.finfo('float').eps)
#             else:
#                 numer_comb = np.sum(numer.reshape(-1, self.group_size), axis=1)
#                 if denom is None:
#                     gammas_comb = numer_comb
#                 else:
#                     denom_comb = np.sum(denom.reshape(-1, self.group_size), axis=1)
#                     gammas_comb = numer_comb / denom_fun(denom_comb)

#                 gammas = np.repeat(gammas_comb / self.group_size, self.group_size)

#             # compute convergence criterion
#             gammas_full = np.zeros(n_sources, dtype=np.float64)
#             gammas_full[active_set] = gammas

#             # compute the noise covariance


#             err = (np.sum(np.abs(gammas_full - gammas_full_old)) /
#                 np.sum(np.abs(gammas_full_old)))

#             # err_x = linalg.norm(x_bar - x_bar_old, ord = 'fro')
#             # print(err_x)

#             gammas_full_old = gammas_full

#             breaking = (err < self.tol or n_active == 0)
#             if len(gammas) != last_size or breaking:
#                 logger.info('Iteration: %d\t active set size: %d\t convergence: '
#                             '%0.3e' % (itno, len(gammas), err))
#                 last_size = len(gammas)

#             if breaking:
#                 break

#         if itno < self.maxiter - 1:
#             logger.info('\nConvergence reached !\n')
#         else:
#             warn('\nConvergence NOT reached !\n')

#         # undo normalization and compute final posterior mean
#         n_const = np.sqrt(M_normalize_constant) / G_normalize_constant
#         x_active = n_const * gammas[:, None] * A

#         x_active = x_active[:, 0]
#         weights[active_set] = n_const * x_active
#         self.coef_ = weights

# def _get_coef(est):
#     if hasattr(est, 'steps'):
#         return est.steps[-1][1].coef_
#     return est.coef_

# class SparseRegressor(BaseEstimator, ClassifierMixin, TransformerMixin):
#     def __init__(self, lead_field, parcel_indices, model, n_jobs=1):
#         self.lead_field = lead_field
#         self.parcel_indices = parcel_indices
#         self.model = model
#         self.n_jobs = n_jobs
#         # self.data_dir = data_dir # this is required only if EMD score would
#         # be used

#     def fit(self, X, y):
#         return self

#     def score(self, X, y):
#         # overwites given score with the EMD score (based on the distance)

#         y_pred = self.predict(X)

#         score = hamming_loss(y, y_pred)
#         '''
#         subjects = np.unique(X['subject'])
#         scores = np.empty(len(subjects))
#         X_used = X.reset_index(drop=True)
#         for idx, subject in enumerate(subjects):
#             subj_idx = X_used[X_used['subject'] == subject].index
#             y_subj = y[subj_idx, :]
#             y_pred_subj = y_pred[subj_idx, :]
#             labels_x = np.load(os.path.join(self.data_dir,
#                                             subject + '_labels.npz'),
#                                allow_pickle=True)['arr_0']

#             score = emd_score(y_subj, y_pred_subj, labels_x)
#             scores[idx] = score * (len(y_subj) / len(y))  # normalize

#         score = np.sum(scores)
#         '''
#         return score

#     def predict(self, X):
#         return (self.decision_function(X) > 0).astype(int)

#     def _run_model(self, model, L, X, fraction_alpha=0.2):
#         norms = np.linalg.norm(L, axis=0)
#         L = L / norms[None, :]

#         est_coefs = np.empty((X.shape[0], L.shape[1]))
#         for idx, idx_used in enumerate(X.index.values):
#             x = X.iloc[idx].values
#             model.fit(L, x)
#             est_coef = np.abs(_get_coef(model))
#             est_coef /= norms
#             est_coefs[idx] = est_coef

#         return est_coefs.T

#     def decision_function(self, X):
#         X = X.reset_index(drop=True)

#         n_parcels = max(max(s) for s in self.parcel_indices)
#         betas = np.empty((len(X), n_parcels))
#         for subj_idx in np.unique(X['subject_id']):
#             l_used = self.lead_field[subj_idx]

#             X_used = X[X['subject_id'] == subj_idx]
#             X_used = X_used.iloc[:, :-2]

#             est_coef = self._run_model(self.model, l_used, X_used)

#             beta = pd.DataFrame(
#                        np.abs(est_coef)
#                    ).groupby(
#                    self.parcel_indices[subj_idx]).max().transpose()
#             betas[X['subject_id'] == subj_idx] = np.array(beta)
#         return betas
