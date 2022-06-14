from bsi_zoo.data_generator import get_data
from bsi_zoo.estimators import gamma_map
from bsi_zoo.metrics import dummy, jaccard_error
import json


class Benchmark:
    def __init__(
        self,
        estimator,
        metrics=[],
        data_agrs={},
        data_agrs_to_benchmark={},
        save_profile=True,
    ) -> None:
        self.estimator = estimator
        self.metrics = metrics
        self.data_agrs = data_agrs
        self.data_agrs_to_benchmark = data_agrs_to_benchmark
        self.save_profile = save_profile

    def benchmark(self, nruns=2):
        profile = {
            arg_to_benchmark: {} for arg_to_benchmark in self.data_agrs_to_benchmark
        }
        for arg_to_benchmark in self.data_agrs_to_benchmark:
            for arg_value in self.data_agrs_to_benchmark[arg_to_benchmark]:
                self.data_agrs[arg_to_benchmark] = arg_value
                print(data_agrs)
                y, L, x, cov, noise = get_data(**self.data_agrs)

                profile[arg_to_benchmark][arg_value] = {}
                profile_to_store = profile[arg_to_benchmark][arg_value]
                for _ in range(nruns):
                    x_hat = self.estimator(L, y, cov, alpha=0.2)
                    for metric in self.metrics:
                        if metric.__name__ in profile_to_store:
                            profile_to_store[metric.__name__].append(metric(x, x_hat))
                        else:
                            profile_to_store[metric.__name__] = [metric(x, x_hat)]

        if self.save_profile:
            with open(
                "bsi_zoo/data/benchmark_data_%s.json" % self.estimator.__name__, "w"
            ) as fp:
                json.dump(profile, fp)
                print("Profile saved!")

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
        "alpha": 0.99,
    }

    data_agrs_to_benchmark = {"alpha": [0.99, 0.85, 0.8]}
    metrics = [dummy]  # list of metric functions here

    benchmark_gamma_map = Benchmark(
        gamma_map, metrics, data_agrs, data_agrs_to_benchmark
    )
    gamma_map_profile = benchmark_gamma_map.benchmark(nruns=2)
    print(gamma_map_profile)
