import itertools
from pathlib import Path
from scipy import linalg
import pandas as pd
from joblib import Memory, Parallel, delayed
from sklearn.utils import check_random_state
from sklearn.model_selection import ParameterGrid


from bsi_zoo.data_generator import get_data

from bsi_zoo.estimators import gamma_map, iterative_sqrt
from bsi_zoo.metrics import euclidean_distance, mse, emd, f1, nll

from bsi_zoo.estimators import (
    iterative_L1,
    iterative_L2,
    iterative_L1_typeII,
    iterative_L2_typeII,
    gamma_map,
    iterative_sqrt,
)

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
):
    print("Benchmarking this data...")
    print(this_data_args)

    y, L, x, cov, _ = memory.cache(get_data)(**this_data_args, seed=seed)

    # estimate x_hat
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
        this_results[metric.__name__] = metric_score
    this_results.update(this_data_args)
    this_results.update({f"estimator__{k}": v for k, v in this_estimator_args.items()})
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
    ) -> None:
        self.estimator = estimator
        self.subject = subject
        self.metrics = metrics
        self.data_args = data_args
        self.estimator_args = estimator_args
        self.random_state = random_state
        self.memory = memory if isinstance(memory, Memory) else Memory(memory)
        self.n_jobs = n_jobs

    def run(self, nruns=2):
        rng = check_random_state(self.random_state)
        seeds = rng.randint(low=0, high=2 ** 32, size=nruns)

        estimator = self.memory.cache(self.estimator)

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
            )
            for this_data_args, seed, this_estimator_args in itertools.product(
                ParameterGrid(self.data_args), seeds, ParameterGrid(self.estimator_args)
            )
        )

        results = pd.DataFrame(results)
        return results


if __name__ == "__main__":
    n_jobs = 4
    metrics = [euclidean_distance, mse, emd, f1]  # list of metric functions here
    memory = Memory(".")

    """ Fixed orientation parameters for the benchmark """

    subject = "CC120264"
    data_args_I = {
        "n_sensors": [50],
        "n_times": [10],
        "n_sources": [200],
        "n_orient": [3],
        "nnz": [2, 5, 10],
        "cov_type": ["diag"],
        "path_to_leadfield": [get_leadfield_path(subject, type="fixed")],
        "orientation_type": ["fixed"],
        "alpha": [0.9, 0.8, 0.7, 0.5, 0.4],  # this is actually SNR
    }


    metrics = [euclidean_distance, mse, emd, f1, nll]  # list of metric functions here

    data_args_II = {
        "n_sensors": [50],
        "n_times": [10],
        "n_sources": [200],
        "n_orient": [3],
        "nnz": [2, 5, 10],
        "cov_type": ["full"],
        "path_to_leadfield": [get_leadfield_path(subject, type="fixed")],
        "orientation_type": ["fixed"],
        "alpha": [0.9, 0.8, 0.7, 0.5, 0.4],  # this is actually SNR
    }

    estimators = [
        (iterative_L1, data_args_I, {"alpha": [0.2, 0.1, 0.01]}),
        (iterative_L2, data_args_I, {"alpha": [0.2, 0.1, 0.01]}),
        (iterative_sqrt, data_args_I, {"alpha": [0.2, 0.1, 0.01]}),
        (iterative_L1_typeII, data_args_II, {"alpha": [0.2, 0.1, 0.01]}),
        (iterative_L2_typeII, data_args_II, {"alpha": [0.2, 0.1, 0.01]}),
        (gamma_map, data_args_II, {"alpha": [0.2, 0.1, 0.01]}),
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
        )
        results = benchmark.run(nruns=10)
        df_results.append(results)

    df_results = pd.concat(df_results, axis=0)

    data_path = Path("bsi_zoo/data")
    data_path.mkdir(exist_ok=True)
    df_results.to_pickle(
        data_path / f"benchmark_data_{subject}_{data_args['orientation_type'][0]}.pkl"
    )

    print(df_results)

    """ Fixed orientation parameters for the benchmark """

    subject = "CC120264"
    data_args_I = {
        "n_sensors": [50],
        "n_times": [10],
        "n_sources": [200],
        "n_orient": [3],
        "nnz": [2, 5, 10],
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
        "nnz": [2],
        "cov_type": ["full"],
        "path_to_leadfield": [get_leadfield_path(subject, type="free")],
        "orientation_type": ["free"],
        "alpha": [0.9, 0.8, 0.7, 0.5, 0.4],  # this is actually SNR
    }

    estimators = [
        (iterative_L1, data_args_I, {"alpha": [0.2, 0.1, 0.01]}),
        (iterative_L2, data_args_I, {"alpha": [0.2, 0.1, 0.01]}),
        (iterative_sqrt, data_args_I, {"alpha": [0.2, 0.1, 0.01]}),
        (iterative_L1_typeII, data_args_II, {"alpha": [0.2, 0.1, 0.01]}),
        (iterative_L2_typeII, data_args_II, {"alpha": [0.2, 0.1, 0.01]}),
        (gamma_map, data_args_II, {"alpha": [0.2, 0.1, 0.01]}),
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
        )
        results = benchmark.run(nruns=10)
        df_results.append(results)

    df_results = pd.concat(df_results, axis=0)

    data_path = Path("bsi_zoo/data")
    data_path.mkdir(exist_ok=True)
    df_results.to_pickle(
        data_path / f"benchmark_data_{subject}_{data_args['orientation_type'][0]}.pkl"
    )

    print(df_results)
