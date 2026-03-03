from bsi_zoo.estimators import eloreta, Solver
from bsi_zoo.cross_val import SpatialCVSolver
from bsi_zoo.data_generator import get_data
from bsi_zoo.config import get_leadfield_path
from scipy import linalg

# test eloreta for 1 run

# get data
y, L, x, cov, noise = get_data(
    n_sensors=50,
    n_times=10,
    n_sources=200,
    nnz=1,
    cov_type='diag',
    path_to_leadfield=get_leadfield_path('CC120313'),
    orientation_type='fixed',
    seed=42,
)

whitener = linalg.inv(linalg.sqrtm(cov))
L = whitener @ L
y = whitener @ y


# solver = Solver(
#         eloreta,
#         alpha=0.1,
#         cov_type='diag',
#         cov=cov,
#         n_orient=1,
#     ).fit(L=L, y=y)
# x_hat = solver.predict(y)
# x_hat


estimator_ = SpatialCVSolver(
    eloreta,
    alphas=[0.00001, 1, 0.0001, 1000],
    # [1, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000],
    cov_type='diag',
    cov=cov,
    n_orient=1,
    cv=3,
    # extra_params=extra_params,
    # seed=42,
).fit(L=L, y=y)
x_hat = estimator_.predict(y)
#
print(estimator_.alpha_)