from joblib import Memory
from pathlib import Path
import pandas as pd

from bsi_zoo.benchmark import Benchmark
from bsi_zoo.estimators import (
    iterative_L1,
    iterative_L2,
    iterative_L1_typeII,
    iterative_L2_typeII,
    gamma_map,
    iterative_sqrt,
)
from bsi_zoo.metrics import euclidean_distance, mse, emd, f1
from bsi_zoo.config import get_leadfield_path

n_jobs = 1  # TODO
nruns = 1
do_spatial_cv = True
subjects = ["CC120166", "CC120264", "CC120313", "CC120309"]
metrics = [euclidean_distance, mse, emd, f1]  # list of metric functions here
nnzs = [1, 2, 3, 5]
alpha_SNR = [0.9, 0.8, 0.6, 0.5, 0.4]
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

for subject in subjects:
    """Fixed orientation parameters for the benchmark"""

    data_args_I = {
        "n_sensors": [50],
        "n_times": [10],
        "n_sources": [200],
        "n_orient": [3],
        "nnz": nnzs,
        "cov_type": ["diag"],
        "path_to_leadfield": [get_leadfield_path(subject, type="fixed")],
        "orientation_type": ["fixed"],
        "alpha": alpha_SNR,  # this is actually SNR
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
        "alpha": alpha_SNR,  # this is actually SNR
    }

    estimators = [
        # (iterative_L1, data_args_I, {"alpha": estimator_alphas}, {}),
        # (iterative_L2, data_args_I, {"alpha": estimator_alphas}, {}),
        # (iterative_sqrt, data_args_I, {"alpha": estimator_alphas}, {}),
        # (iterative_L1_typeII, data_args_II, {"alpha": estimator_alphas}, {}),
        # (iterative_L2_typeII, data_args_II, {"alpha": estimator_alphas}, {}),
        (gamma_map, data_args_II, {"alpha": estimator_alphas}, {"update_mode": 1}),
        (gamma_map, data_args_II, {"alpha": estimator_alphas}, {"update_mode": 2}),
        (gamma_map, data_args_II, {"alpha": estimator_alphas}, {"update_mode": 3}),
    ]

    df_results = []
    for estimator, data_args, estimator_args, estimator_extra_params in estimators:
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
            estimator_extra_params=estimator_extra_params,
        )
        results = benchmark.run(nruns=nruns)
        df_results.append(results)

    df_results = pd.concat(df_results, axis=0)

    data_path = Path("bsi_zoo/data")
    data_path.mkdir(exist_ok=True)
    df_results.to_pickle(
        data_path / f"benchmark_data_{subject}_{data_args['orientation_type'][0]}.pkl"
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
        "alpha": alpha_SNR,  # this is actually SNR
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
        "alpha": alpha_SNR,  # this is actually SNR
    }

    estimators = [
        # (iterative_L1, data_args_I, {"alpha": estimator_alphas}, {}),
        # (iterative_L2, data_args_I, {"alpha": estimator_alphas}, {}),
        # (iterative_sqrt, data_args_I, {"alpha": estimator_alphas}, {}),
        # (iterative_L1_typeII, data_args_II, {"alpha": estimator_alphas}, {}),
        # (iterative_L2_typeII, data_args_II, {"alpha": estimator_alphas}, {}),
        (gamma_map, data_args_II, {"alpha": estimator_alphas}, {"update_mode": 1}),
        (gamma_map, data_args_II, {"alpha": estimator_alphas}, {"update_mode": 2}),
        (gamma_map, data_args_II, {"alpha": estimator_alphas}, {"update_mode": 3}),
    ]

    df_results = []
    for estimator, data_args, estimator_args, estimator_extra_params in estimators:
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
            estimator_extra_params=estimator_extra_params,
        )
        results = benchmark.run(nruns=nruns)
        df_results.append(results)

    df_results = pd.concat(df_results, axis=0)

    data_path = Path("bsi_zoo/data")
    data_path.mkdir(exist_ok=True)
    if do_spatial_cv:
        FILE_NAME = f"benchmark_data_{subject}_{data_args['orientation_type'][0]}.pkl"
    else:
        FILE_NAME = f"benchmark_data_{subject}_{data_args['orientation_type'][0]}.pkl"
    df_results.to_pickle(data_path / FILE_NAME)

    print(df_results)
