import numpy as np
from scipy.stats import wishart


def _add_noise(cov_type, y, alpha, rng, n_sensors, n_times):
    noise_type = "random"
    if cov_type == "diag":
        if noise_type == "random":
            # initialization of the noise covariance matrix with a random diagonal matrix
            rv = wishart(df=n_sensors, scale=1e-3 * np.eye(n_sensors))
            cov = rv.rvs()
            cov = np.diag(np.diag(cov))
        else:
            # initialization of the noise covariance with an identity matrix
            cov = 1e-2 * np.diag(np.ones(n_sensors))
    else:
        # initialization of the noise covariance matrix with a full PSD random matrix
        rv = wishart(df=n_sensors, scale=1e-3 * np.eye(n_sensors))
        cov = rv.rvs()

    signal_norm = np.linalg.norm(y, "fro")
    noise = rng.multivariate_normal(np.zeros(n_sensors), cov, size=n_times).T
    noise_norm = np.linalg.norm(noise, "fro")
    noise_normalised = noise / noise_norm

    noise_scaled = ((1 - alpha) / alpha) * signal_norm * noise_normalised
    cov_scaled = cov * (((1 - alpha) / alpha) * (signal_norm / noise_norm)) ** 2
    y += noise_scaled

    return y, cov_scaled, noise_scaled


def get_data(
    cov_type,
    path_to_leadfield,
    n_sensors=50,
    n_times=10,
    n_sources=200,
    nnz=3,
    orientation_type="fixed",
    alpha=0.99,  # 40dB snr
    seed=None,
):
    n_orient = 3 if orientation_type == "free" else 1
    rng = np.random.RandomState(seed)
    if path_to_leadfield is not None:
        lead_field = np.load(path_to_leadfield, allow_pickle=True)
        L = lead_field["lead_field"]
        if orientation_type == "fixed":
            n_sensors, n_sources = L.shape
        elif orientation_type == "free":
            n_sensors, n_sources, _ = L.shape
    else:
        L = (
            rng.randn(n_sensors, n_sources)
            if orientation_type == "fixed"
            else rng.randn(n_sensors, n_sources, n_orient)
        )

    # generate source locations
    idx = rng.choice(n_sources, size=nnz, replace=False)
    if orientation_type == "fixed":
        x = np.zeros((n_sources, n_times))
        x[idx] = rng.randn(nnz, n_times)
        y = L @ x
    elif orientation_type == "free":
        x = np.zeros((n_sources, n_orient, n_times))
        x[idx] = rng.randn(nnz, n_orient, n_times)
        y = np.einsum("nmr, mrd->nd", L, x)

    # add noise
    y, cov_scaled, noise_scaled = _add_noise(
        cov_type, y, alpha, rng, n_sensors, n_times
    )

    if orientation_type == "free":
        # reshaping L to (n_sensors, n_sources*n_orient)
        L = L.reshape(L.shape[0], -1)

    if n_times == 1 and orientation_type == "fixed":
        y = y[:, 0]
        x = x[:, 0]

    return y, L, x, cov_scaled, noise_scaled
