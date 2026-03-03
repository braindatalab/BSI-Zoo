## Brain Source Imaging (BSI) Zoo

<img src="https://github.com/braindatalab/BSI-Zoo/assets/29161453/96f3ff0e-c167-413a-b86e-3b8c69274739" alt="BSI-Zoo logo" width="300"/>

The **Brain Source Imaging (BSI) Zoo** is a small benchmarking framework for comparing brain source imaging methods under controlled, reproducible conditions. It provides tools to generate data, run estimators, and evaluate performance across a range of simulation settings.

---

## Features

- **Standardised simulations**: configurable control over SNR, number of active sources, covariance structure, and orientation.
- **Random and semi‑real data**: support for both synthetic and semi‑realistic EEG/MEG leadfield data.
- **Multiple estimators included**: a collection of BSI methods (iterative L1/L2, Type‑II, eLORETA, Champagne etc.) implemented in `bsi_zoo.estimators`.
- **Evaluation utilities**: various metrics like EMD, euclidean distance, MSE, F1 score, residual noise. (`bsi_zoo.metrics`).
- **Benchmark scripts**: ready‑to‑run benchmark configuration in `bsi_zoo/run_benchmark.py`.

---

## Installation

### From source

Clone the repository and install in a fresh environment (Python ≥ 3.7):

```bash
git clone https://github.com/braindatalab/BSI-Zoo.git
cd BSI-Zoo
pip install -r requirements.txt
python setup.py install
```

This will install the `bsi_zoo` package and all required scientific Python dependencies.

---

## Basic usage

### Run the default benchmark

The easiest way to get started is to run the benchmark script:

```bash
python -m bsi_zoo/run_benchmark.py
```

This will:

- **generate simulated data** for several subjects and noise levels,
- **run a set of predefined estimators** (see `bsi_zoo.estimators`),
- **evaluate source resconstruction** using metrics from `bsi_zoo.metrics`.

Adjust benchmark settings such as subjects, metrics, or hyperparameters directly in `bsi_zoo/run_benchmark.py`.

### Programmatic use

You can also use the package directly in Python:

```python
from bsi_zoo.benchmark import Benchmark
from bsi_zoo.data_generator import get_data
from bsi_zoo.estimators import iterative_L1
from bsi_zoo.metrics import mse

# Generate a single dataset (example; see code for full options)
X, y, info = get_data(...)

# Define your estimator and metric
estimator = iterative_L1
metric = mse

benchmark = Benchmark(estimators=[(estimator, {}, {}, {})], metrics=[metric])
results = benchmark.run()
```

Check the docstrings in the `bsi_zoo` modules for detailed parameter descriptions.

---



## Citation

If you use this package or any parts of this code in your research, please cite our work:

> Negi, A., Haufe, S., Gramfort, A., & Hashemi, A. (2025). How forgiving are M/EEG inverse solutions to noise level misspecification? An excursion into the BSI-Zoo. *bioRxiv*. [https://www.biorxiv.org/content/10.1101/2025.03.12.642831v1.abstract](https://www.biorxiv.org/content/10.1101/2025.03.12.642831v1.abstract)

BibTeX:
```bibtex
@article{negi2025forgiving,
  title={How forgiving are M/EEG inverse solutions to noise level misspecification? An excursion into the BSI-Zoo},
  author={Negi, Anuja and Haufe, Stefan and Gramfort, Alexandre and Hashemi, Ali},
  journal={bioRxiv},
  pages={2025--03},
  year={2025},
  publisher={Cold Spring Harbor Laboratory}
}
```


## License

This project is released under a **BSD 3‑Clause** license (see `LICENSE` file). Use it freely in your own research or applications, and please consider citing the repository if it is useful in your work.