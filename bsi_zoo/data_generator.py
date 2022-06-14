import numpy as np


def get_data(
    n_sensors,
    n_times,
    n_sources,
    n_orient,
    nnz,
    cov_type,
    path_to_leadfield,
    orientation_type="fixed",
    alpha=0.99,  # 40dB snr
):
    if orientation_type == "fixed":
        rng = np.random.RandomState(42)
        if path_to_leadfield is not None:
            lead_field = np.load(path_to_leadfield, allow_pickle=True)
            L = lead_field["lead_field"]
            n_sensors, n_sources = L.shape
        else:
            L = rng.randn(n_sensors, n_sources)

        x = np.zeros((n_sources, n_times))
        x[rng.randint(low=0, high=x.shape[0], size=nnz)] = rng.randn(nnz, n_times)
        # x[:nnz] = rng.randn(nnz, n_times)
        y = L @ x

        noise_type = "random"
        if cov_type == "diag":
            if noise_type == "random":
                # initialization of the noise covariance matrix with a random diagonal matrix
                cov = rng.randn(n_sensors, n_sensors)
                cov = 1e-3 * (cov @ cov.T)
                cov = np.diag(np.diag(cov))
            else:
                # initialization of the noise covariance with an identity matrix
                cov = 1e-2 * np.diag(np.ones(n_sensors))
        else:
            # initialization of the noise covariance matrix with a full PSD random matrix
            cov = rng.randn(n_sensors, n_sensors)
            cov = 1e-3 * (cov @ cov.T)
            # cov = 1e-3 * (cov @ cov.T) / n_times ## devided by the number of time samples for better scaling

        signal_norm = np.linalg.norm(y, "fro")
        noise = rng.multivariate_normal(np.zeros(n_sensors), cov, size=n_times).T
        noise_norm = np.linalg.norm(noise, "fro")
        noise_normalised = noise / noise_norm

        noise_scaled = ((1 - alpha) / alpha) * signal_norm * noise_normalised
        cov_scaled = cov * (((1 - alpha) / alpha) * (signal_norm / noise_norm)) ** 2
        y += noise_scaled

        if n_times == 1:
            y = y[:, 0]
            x = x[:, 0]

    elif orientation_type == "free":

        rng = np.random.RandomState(35)
        if path_to_leadfield is not None:
            lead_field = np.load(path_to_leadfield, allow_pickle=True)
            L = lead_field["lead_field"]
            n_sensors, n_sources, _ = L.shape
        else:
            L = rng.randn(n_sensors, n_sources, n_orient)

        x = np.zeros((n_sources, n_orient, n_times))
        x[rng.randint(low=0, high=x.shape[0], size=nnz)] = rng.randn(
            nnz, n_orient, n_times
        )
        y = np.einsum("nmr, mrd->nd", L, x)

        noise_type = "random"
        if cov_type == "diag":
            if noise_type == "random":
                # initialization of the noise covariance matrix with a random diagonal matrix
                cov = rng.randn(n_sensors, n_sensors)
                cov = 1e-3 * (cov @ cov.T)
                cov = np.diag(np.diag(cov))
            else:
                # initialization of the noise covariance with an identity matrix
                cov = 1e-2 * np.diag(np.ones(n_sensors))
        else:
            # initialization of the noise covariance matrix with a full PSD random matrix
            cov = rng.randn(n_sensors, n_sensors)
            cov = 1e-3 * (cov @ cov.T)
            # cov = 1e-3 * (cov @ cov.T) / n_times ## devided by the number of time samples for better scaling

        signal_norm = np.linalg.norm(y, "fro")
        noise = rng.multivariate_normal(np.zeros(n_sensors), cov, size=n_times).T
        noise_norm = np.linalg.norm(noise, "fro")
        noise_normalised = noise / noise_norm

        alpha = 0.99  # 40dB snr
        noise_scaled = ((1 - alpha) / alpha) * signal_norm * noise_normalised
        cov_scaled = cov * (((1 - alpha) / alpha) * (signal_norm / noise_norm)) ** 2
        y += noise_scaled

        # if n_times == 1:
        #     y = y[:, 0]
        #     x = x[:, 0]

        # reshaping L to (n_sensors, n_sources*n_orient)
        L = L.reshape(L.shape[0], -1)

    return y, L, x, cov_scaled, noise_scaled
