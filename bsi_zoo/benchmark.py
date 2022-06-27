from bsi_zoo.data_generator import get_data
from bsi_zoo.estimators import gamma_map
from bsi_zoo.metrics import euclidean_distance, mse
from bsi_zoo.config import get_leadfield_path
import json
from scipy import linalg


class Benchmark:
    def __init__(
        self,
        estimator,
        subject,
        metrics=[],
        data_agrs={},
        data_agrs_to_benchmark={},
        save_profile=True,
    ) -> None:
        self.estimator = estimator
        self.subject = subject
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
                y, L, x, cov, _ = get_data(**self.data_agrs)

                profile[arg_to_benchmark][arg_value] = {}
                profile_to_store = profile[arg_to_benchmark][arg_value]
                for _ in range(nruns):
                    # estimate x_hat
                    if self.data_agrs["cov_type"] == "diag":
                        whitener = linalg.inv(linalg.sqrtm(cov))
                        L = whitener @ L
                        y = whitener @ y
                        x_hat = self.estimator(L, y, alpha=self.data_agrs["alpha"])
                    else:
                        x_hat = self.estimator(L, y, cov, alpha=self.data_agrs["alpha"])

                    for metric in self.metrics:
                        metric_score = metric(
                            x,
                            x_hat,
                            subject=self.subject,
                            orientation_type=self.data_agrs["orientation_type"],
                            nnz=self.data_agrs["nnz"],
                        )
                        if metric.__name__ in profile_to_store:
                            profile_to_store[metric.__name__].append(metric_score)
                        else:
                            profile_to_store[metric.__name__] = [metric_score]

        if self.save_profile:
            with open(
                "bsi_zoo/data/benchmark_data_%s_%s_%s_orient.json"
                % (
                    self.subject,
                    self.estimator.__name__,
                    self.data_agrs["orientation_type"],
                ),
                "w",
            ) as fp:
                json.dump(profile, fp)
                print(
                    "Profile saved for subject %s for %s estimator!"
                    % (self.subject, self.estimator.__name__)
                )

        return profile


if __name__ == "__main__":
    subject = "CC120264"
    data_agrs = {
        "n_sensors": 50,
        "n_times": 10,
        "n_sources": 200,
        "n_orient": 3,
        "nnz": 3,
        "cov_type": "full",
        "path_to_leadfield": get_leadfield_path(subject, type="fixed"),
        "orientation_type": "fixed",
        "alpha": 0.99,
    }

    data_agrs_to_benchmark = {"alpha": [0.99, 0.9, 0.8, 0.5], "nnz": [3, 5, 10]}
    metrics = [euclidean_distance, mse]  # list of metric functions here

    benchmark_gamma_map = Benchmark(
        gamma_map, subject, metrics, data_agrs, data_agrs_to_benchmark
    )

    gamma_map_profile = benchmark_gamma_map.benchmark(nruns=3)
    print(gamma_map_profile)
