import numpy as np
import itertools
from scipy import linalg
import pandas as pd
from joblib import Memory, Parallel, delayed
from sklearn.utils import check_random_state
from sklearn.model_selection import ParameterGrid


from bsi_zoo.data_generator import get_data
from bsi_zoo.estimators import Solver
from bsi_zoo.cross_val import SpatialCVSolver


def _run_estimator(
    subject,
    estimator,
    metrics,
    this_data_args,
    this_estimator_args,
    seed,
    estimator_name,
    memory,
    do_spatial_cv,
    extra_params={},
):
    print("Benchmarking this data...")
    print(this_data_args)

    y, L, x, cov, _ = memory.cache(get_data)(**this_data_args, seed=seed)

    if this_data_args["cov_type"] == "diag":
        whitener = linalg.inv(linalg.sqrtm(cov))
        L = whitener @ L
        y = whitener @ y

    # estimate x_hat
    if do_spatial_cv:
        estimator_ = SpatialCVSolver(
            estimator,
            alphas=this_estimator_args["alpha"],
            cov_type=this_data_args["cov_type"],
            cov=cov,
            n_orient=this_data_args["n_orient"],
            cv=3,
            extra_params=extra_params,
        ).fit(L=L, y=y)
        x_hat = estimator_.predict(y)
    else:
        estimator_ = Solver(
            estimator,
            alpha=this_estimator_args["alpha"],
            cov_type=this_data_args["cov_type"],
            cov=cov,
            n_orient=this_data_args["n_orient"],
            extra_params=extra_params,
        ).fit(L=L, y=y)
        x_hat = estimator_.predict(y)

    if this_data_args["orientation_type"] == "free":
        x_hat = x_hat.reshape(x.shape)

    this_results = dict(estimator=estimator_name)
    for metric in metrics:
        try:
            metric_score = metric(
                x,
                x_hat,
                subject=subject,
                orientation_type=this_data_args["orientation_type"],
                nnz=this_data_args["nnz"],
                y=y,
                L=L,
                cov=cov,
            )
        except Exception:
            # estimators that predict less vertices for certain parameter combinations; these cannot be evaluated by all current metrics
            metric_score = np.nan
        this_results[metric.__name__] = metric_score
    this_results.update(this_data_args)
    this_results.update({f"estimator__{k}": v for k, v in this_estimator_args.items()})
    if do_spatial_cv:
        this_results.update({"estimator__alpha_cv": estimator_.get_alpha()})

    return this_results


class Benchmark:
    def __init__(
        self,
        estimator,
        subject,
        metrics=[],
        data_args={},
        estimator_args={},
        random_state=None,
        memory=None,
        n_jobs=1,
        do_spatial_cv=False,
        estimator_extra_params={},
    ) -> None:
        self.estimator = estimator
        self.subject = subject
        self.metrics = metrics
        self.data_args = data_args
        self.estimator_args = estimator_args
        self.random_state = random_state
        self.memory = memory if isinstance(memory, Memory) else Memory(memory)
        self.n_jobs = n_jobs
        self.do_spatial_cv = do_spatial_cv
        self.estimator_extra_params = estimator_extra_params
        self.do_spatial_cv = do_spatial_cv

    def run(self, nruns=2):
        rng = check_random_state(self.random_state)
        seeds = rng.randint(low=0, high=2 ** 32, size=nruns)

        estimator = self.memory.cache(self.estimator)

        if self.do_spatial_cv:
            # dont make param grid for estimator args
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(_run_estimator)(
                    self.subject,
                    estimator,
                    self.metrics,
                    this_data_args,
                    self.estimator_args,
                    seed,
                    estimator_name=self.estimator.__name__,
                    memory=self.memory,
                    do_spatial_cv=self.do_spatial_cv,
                    extra_params=self.estimator_extra_params,
                )
                for this_data_args, seed in itertools.product(
                    ParameterGrid(self.data_args), seeds
                )
            )
        else:
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(_run_estimator)(
                    self.subject,
                    estimator,
                    self.metrics,
                    this_data_args,
                    this_estimator_args,
                    seed,
                    estimator_name=self.estimator.__name__,
                    memory=self.memory,
                    do_spatial_cv=self.do_spatial_cv,
                )
                for this_data_args, seed, this_estimator_args in itertools.product(
                    ParameterGrid(self.data_args),
                    seeds,
                    ParameterGrid(self.estimator_args),
                )
            )

        results = pd.DataFrame(results)
        return results
