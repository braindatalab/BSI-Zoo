from scipy import linalg
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import check_cv


from .estimators import SpatialSolver


def temporal_cv_metric(y, Sigma_Y):
    """Compute the log-det Bregman divergence between two matrices.

    It is based on the calculation of Gaussian negative log-likelihood.
    Both matrices needs to be squared and have the same size.
    It is given by:

    B_inv = inv(B);
    logdet_distance = trace(A*B_inv) - logdet(A*B_inv) - size(A,1)
    """
    # XXX Need improve w.r.t speed (Perhaps use the following conditions)
    _, logdet = np.linalg.slogdet(Sigma_Y)
    out = (np.linalg.norm(linalg.sqrtm(np.linalg.inv(Sigma_Y) @ y),
                          ord='fro')**2 + logdet)
    return out


def _logdet(A):
    """Compute the logdet of a positive semidefinite matrix."""
    from scipy import linalg
    vals = linalg.eigvalsh(A)
    # avoid negative (numerical errors) or zero (semi-definite matrix) values
    tol = vals.max() * vals.size * np.finfo(np.float64).eps
    vals = np.where(vals > tol, vals, tol)
    return np.sum(np.log(vals))


def logdet_bregman_div_distance_nll(y, Sigma_Y):
    """Compute the log-det Bregman divergence between two matrices."""
    Sigma_Y_inv = np.linalg.inv(Sigma_Y)
    Cov_y = np.cov(y)
    _, n_features = Sigma_Y.shape
    log_like = np.mean(np.sum((y.T @ Sigma_Y_inv) * y.T, axis=1))
    log_like -= _logdet(Cov_y @ Sigma_Y_inv)
    out = log_like - n_features
    return out


class BaseCVSolver(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        solver,
        cov_type,
        cov,
        n_orient,
        alphas=np.linspace(1.4, 0.01, 30),
        cv=5,
        extra_params={},
        n_jobs=1,
    ):
        self.solver = solver
        self.alphas = alphas
        self.cov = cov
        self.cov_type = cov_type
        self.n_orient = n_orient
        self.cv = cv
        self.extra_params = extra_params
        self.n_jobs = n_jobs

    def fit(self, L, y):
        self.L_ = L
        self.y_ = y

        return self

    def predict(self, y):
        self._get_alpha(y)

        if self.cov_type == "diag":
            self.coef_ = self.solver(
                self.L_,
                y,
                alpha=self.alpha_,
                n_orient=self.n_orient,
                **self.extra_params
            )
        else:
            self.coef_ = self.solver(
                self.L_,
                y,
                self.cov,
                alpha=self.alpha_,
                n_orient=self.n_orient,
                **self.extra_params
            )

        return self.coef_


class SpatialCVSolver(BaseCVSolver):
    def _get_alpha(self, y):
        """Sets alpha_ attribute with spatial cross-validation."""
        gs = GridSearchCV(
            estimator=SpatialSolver(
                self.solver,
                cov=self.cov,
                alpha=None,
                cov_type=self.cov_type,
                n_orient=self.n_orient,
            ),
            param_grid=dict(alpha=self.alphas),
            scoring="neg_mean_squared_error",
            cv=self.cv,
            n_jobs=self.n_jobs,
        )
        gs.fit(self.L_, y)
        self.grid_search_ = gs
        self.alpha_ = gs.best_estimator_.alpha


class TemporalCVSolver(BaseCVSolver):
    def _get_alpha(self, y):
        """Sets alpha_ attribute with temporal cross-validation."""
        base_solver = SpatialSolver(
            self.solver,
            cov=self.cov,
            alpha=None,
            cov_type=self.cov_type,
            n_orient=self.n_orient,
        )

        cv = check_cv(self.cv)
        scores = []
        for alpha in self.alphas:
            solver = clone(base_solver)
            solver.set_params(alpha=alpha)
            temporal_cv_scores = []
            for train_idx, test_idx in cv.split(y.T):
                solver.fit(self.L_, y[:, train_idx])
                y_test = y[:, test_idx]
                # X_diag = np.sum(np.abs(solver.coef_), axis=1) != 0
                Cov_X = np.cov(solver.coef_)
                # X_Sqaure = solver.coef_ @ solver.coef_.T
                # Cov_X = np.diag(np.linalg.norm(X_Sqaure, axis=0))
                Sigma_Y = self.cov + ((self.L_ @ Cov_X) @ self.L_.T)
                temporal_cv_scores.append(
                    temporal_cv_metric(y_test, Sigma_Y)
                    # logdet_bregman_div_distance_nll(y_test, Sigma_Y)
                )
            scores.append(
                np.mean(temporal_cv_scores)
            )
        self.alpha_ = self.alphas[np.argmax((scores))]
