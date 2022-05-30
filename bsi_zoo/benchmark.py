from bsi_zoo.data_generator import get_data
from bsi_zoo.estimators import gamma_map
from bsi_zoo.metrics import dummy


class Benchmark:
    def __init__(self, estimator, metrics=[], data_agrs={}) -> None:
        self.estimators = estimator
        self.metrics = metrics
        self.data_agrs = data_agrs

    def benchmark(self, nruns=3):
        y, L, x, cov, noise = get_data(**self.data_agrs)
        profile = {metric.__name__: [] for metric in self.metrics}

        for _ in range(nruns):
            x_hat = self.estimator(L, y, cov, alpha=0.2)
            for metric in self.metrics:
                profile[metric.__name__].append(metric(x, x_hat))

        return profile


if __name__ == "__main__":
    data_agrs = {
        "n_sensors": 50,
        "n_times": 10,
        "n_sources": 200,
        "n_orient": 3,
        "nnz": 3,
        "cov_type": "full",
        "path_to_leadfield": None,
        "orientation_type": "fixed",
    }

    data_agrs_to_benchmark = {"snr": [30, 40, 60]}

    benchmark_gamma_map = Benchmark(gamma_map, [dummy], data_agrs)
    gamma_map_profile = benchmark_gamma_map.benchmark()
    print(gamma_map_profile)
