import numpy as np
import itertools
from pathlib import Path
from scipy import linalg
import pandas as pd
from joblib import Memory, Parallel, delayed
from sklearn.utils import check_random_state
from sklearn.model_selection import ParameterGrid


from bsi_zoo.data_generator import get_data
from bsi_zoo.estimators import (
    iterative_L1,
    iterative_L2,
    # iterative_L1_typeII,
    # iterative_L2_typeII,
    gamma_map,
    iterative_sqrt,
    SpatialCVSolver,
)
from bsi_zoo.metrics import euclidean_distance, mse, emd, f1
from bsi_zoo.config import get_leadfield_path


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

    # estimate x_hat
    if do_spatial_cv:
        estimator_cv = SpatialCVSolver(
            estimator,
            alphas=this_estimator_args["alpha"],
            cov_type=this_data_args["cov_type"],
            cov=cov,
            n_orient=this_data_args["n_orient"],
            cv=3,
            extra_params=extra_params,
        ).fit(L=L, y=y)
        x_hat = estimator_cv.predict(y)
    else:
        if this_data_args["cov_type"] == "diag":
            whitener = linalg.inv(linalg.sqrtm(cov))
            L = whitener @ L
            y = whitener @ y
            x_hat = estimator(L, y, **this_estimator_args)
        else:
            x_hat = estimator(L, y, cov, **this_estimator_args)

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
        this_results.update({"estimator__alpha_cv": estimator_cv.get_alpha()})

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

    def run(self, nruns=2):
        rng = check_random_state(self.random_state)
        seeds = rng.randint(low=0, high=2 ** 32, size=nruns)

        estimator = self.memory.cache(self.estimator)

        if do_spatial_cv:
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


if __name__ == "__main__":
    n_jobs = 10
    do_spatial_cv = True
    metrics = [euclidean_distance, mse, emd, f1]  # list of metric functions here
    nnzs = [1, 2, 3, 5]
    estimator_alphas = [
        0.01,
        0.01544452,
        0.02385332,
        0.03684031,
        0.0568981,
        0.08787639,
        0.13572088,
        0.2096144,
    ]  # logspaced
    memory = Memory(".")

    for subject in ["CC120166", "CC120264", "CC120313", "CC120309"]:
        """ Fixed orientation parameters for the benchmark """

        data_args_I = {
            "n_sensors": [50],
            "n_times": [10],
            "n_sources": [200],
            "n_orient": [3],
            "nnz": nnzs,
            "cov_type": ["diag"],
            "path_to_leadfield": [get_leadfield_path(subject, type="fixed")],
            "orientation_type": ["fixed"],
            "alpha": [0.9, 0.8, 0.6, 0.5, 0.4],  # this is actually SNR
        }

        data_args_II = {
            "n_sensors": [50],
            "n_times": [10],
            "n_sources": [200],
            "n_orient": [3],
            "nnz": nnzs,
            "cov_type": ["full"],
            "path_to_leadfield": [get_leadfield_path(subject, type="fixed")],
            "orientation_type": ["fixed"],
            "alpha": [0.9, 0.8, 0.6, 0.5, 0.4],  # this is actually SNR
        }

        estimators = [
            (iterative_L1, data_args_I, {"alpha": estimator_alphas}),
            (iterative_L2, data_args_I, {"alpha": estimator_alphas}),
            (iterative_sqrt, data_args_I, {"alpha": estimator_alphas}),
            # (iterative_L1_typeII, data_args_II, {"alpha": estimator_alphas}),
            # (iterative_L2_typeII, data_args_II, {"alpha": estimator_alphas}),
            (gamma_map, data_args_II, {"alpha": estimator_alphas}),
        ]

        df_results = []
        for estimator, data_args, estimator_args in estimators:
            benchmark = Benchmark(
                estimator,
                subject,
                metrics,
                data_args,
                estimator_args,
                random_state=42,
                memory=memory,
                n_jobs=n_jobs,
                do_spatial_cv=do_spatial_cv,
            )
            results = benchmark.run(nruns=10)
            df_results.append(results)

        df_results = pd.concat(df_results, axis=0)

        data_path = Path("bsi_zoo/data")
        data_path.mkdir(exist_ok=True)
        df_results.to_pickle(
            data_path
            / f"benchmark_data_{subject}_{data_args['orientation_type'][0]}.pkl"
        )

        print(df_results)

        """ Free orientation parameters for the benchmark """

        data_args_I = {
            "n_sensors": [50],
            "n_times": [10],
            "n_sources": [200],
            "n_orient": [3],
            "nnz": nnzs,
            "cov_type": ["diag"],
            "path_to_leadfield": [get_leadfield_path(subject, type="free")],
            "orientation_type": ["free"],
            "alpha": [0.9, 0.8, 0.7, 0.5, 0.4],  # this is actually SNR
        }

        data_args_II = {
            "n_sensors": [50],
            "n_times": [10],
            "n_sources": [200],
            "n_orient": [3],
            "nnz": nnzs,
            "cov_type": ["full"],
            "path_to_leadfield": [get_leadfield_path(subject, type="free")],
            "orientation_type": ["free"],
            "alpha": [0.9, 0.8, 0.7, 0.5, 0.4],  # this is actually SNR
        }

        estimators = [
            (iterative_L1, data_args_I, {"alpha": estimator_alphas}),
            (iterative_L2, data_args_I, {"alpha": estimator_alphas}),
            (iterative_sqrt, data_args_I, {"alpha": estimator_alphas}),
            # (iterative_L1_typeII, data_args_II, {"alpha": estimator_alphas}),
            # (iterative_L2_typeII, data_args_II, {"alpha": estimator_alphas}),
            (gamma_map, data_args_II, {"alpha": estimator_alphas}),
        ]

        df_results = []
        for estimator, data_args, estimator_args in estimators:
            benchmark = Benchmark(
                estimator,
                subject,
                metrics,
                data_args,
                estimator_args,
                random_state=42,
                memory=memory,
                n_jobs=n_jobs,
                do_spatial_cv=do_spatial_cv,
            )
            results = benchmark.run(nruns=10)
            df_results.append(results)

        df_results = pd.concat(df_results, axis=0)

        data_path = Path("bsi_zoo/data")
        data_path.mkdir(exist_ok=True)
        if do_spatial_cv:
            FILE_NAME = (
                f"benchmark_data_{subject}_{data_args['orientation_type'][0]}.pkl"
            )
        else:
            FILE_NAME = (
                f"benchmark_data_{subject}_{data_args['orientation_type'][0]}.pkl"
            )
        df_results.to_pickle(data_path / FILE_NAME)

        print(df_results)
