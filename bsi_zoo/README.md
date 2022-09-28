## Iterative methdos

* Should define a function called "get_estimator" that returns a **scikit-learn type of pipeline**.


$$
\boldsymbol{x}^{(k+1)} \leftarrow \arg \min_{\boldsymbol{x}}\|\boldsymbol{y}-{\bf L} \boldsymbol{x}\|_{2}^{2}+\lambda \sum_{i} g(x_{i})
$$

$$
\boldsymbol{x}^{(k+1)} \leftarrow \arg \min_{\boldsymbol{x}}\|\boldsymbol{y}-{\bf L} \boldsymbol{x}\|_{2}^{2}+\lambda \sum_{i} w_{i}^{(k)}\left|x_{i}\right|
$$

## Single Vector Measurement (SVM) Models

### Iterative $\ell_1$:
\begin{equation}
g(x_{i}) = \log \left(\left|x_{i}\right|+\epsilon\right)
\end{equation}

\begin{equation}
w_{i}^{(k+1)} \leftarrow\left[\left|x_{i}^{(k)}\right|+\epsilon\right]^{-1}
\end{equation}


```python
g = lambda w: np.log(np.abs(w) + eps)
gprime = lambda w: 1. / (np.abs(w) + eps)
```

![](https://i.imgur.com/GJsY3L7.png)

### Iterative $\ell_2$:
\begin{equation}
g(x_{i}) = \log \left(\left|x_{i}^2+\epsilon\right|\right)
\end{equation}

\begin{equation}
w_{i}^{(k+1)} \leftarrow\left[\left(x_{i}^{(k)}\right)^{2}+\epsilon\right]^{-1}
\end{equation}


```python
g = lambda w: np.log(np.abs((w ** 2) + eps))
gprime = lambda w: 1. / ((w ** 2) + eps)
```

![](https://i.imgur.com/AGeUzr6.png)

### Iterative $\ell_{0.5}$:

\begin{equation}
g(x_{i}) =  \sqrt{\left|x_{i}\right|}
\end{equation}

\begin{equation}
w_{i}^{(k+1)} \leftarrow\left[2\sqrt{\left|x_{i}\right|}+\epsilon\right]^{-1}
\end{equation}


```python
g = lambda w: np.sqrt(np.abs(w))
gprime = lambda w: 1. / (2. * np.sqrt(np.abs(w)) + eps)
```

![](https://i.imgur.com/lP6QKEH.png)

## Iterative Methods - Type II Variants
$$
\boldsymbol{x}_{\mathrm{SBL}}=\arg \min _{\boldsymbol{x}}\|\boldsymbol{y}-\Phi \boldsymbol{x}\|_{2}^{2}+\lambda g_{\mathrm{SBL}}(\boldsymbol{x})
$$
where
$$
g_{\mathrm{SBL}}(\boldsymbol{x}) \triangleq \min _{\gamma \geq 0} \boldsymbol{x}^{T} \Gamma^{-1} \boldsymbol{x}+\log \left|\alpha I+{\bf L} \Gamma {\bf L}^{T}\right|
$$

### Iterative $\ell_2$ - Type II:

\begin{equation}
w_{i}^{(k+1)} \leftarrow \left[\left(x_{i}^{(k)}\right)^{2}+\underbrace{\left(w_{i}^{(k)}\right)^{-1}-\left(w_{i}^{(k)}\right)^{-2}{\bf L}_{i}^{T}\left(\alpha I+{\bf L}\widetilde{W}^{(k)}{\bf L}^{T}\right)^{-1}{\bf L}_{i}}\right]^{-1}
\end{equation}

* In comparision to its Type I counterpart, e.g., $w_{i}^{(k+1)} \leftarrow\left[\left(x_{i}^{(k)}\right)^{2}+\epsilon\right]^{-1}$:


\begin{equation}
\epsilon_{i}^{(k)} \leftarrow \left(w_{i}^{(k)}\right)^{-1}-\left(w_{i}^{(k)}\right)^{-2} {\bf L}_{i}^{T}\left(\alpha I+{\bf L} \widetilde{W}^{(k)} {\bf L}^{T}\right)^{-1}{\bf L}_{i}
\end{equation}


```python
def gprime(w):
    L_T = L.T
    n_samples, _ = L.shape

    def w_mat(w):
        return np.diag(1 / w)

    def epsilon_update(L, w, alpha, cov):
        noise_cov = cov  # extension of method by importing the noise covariance
        proj_source_cov = (L @ w_mat(w)) @ L_T
        signal_cov = noise_cov + proj_source_cov
        sigmaY_inv = np.linalg.inv(signal_cov)
        return np.diag(
            w_mat(w)
            - np.multiply(
                w_mat(w ** 2), np.diag((L_T @ sigmaY_inv) @ L)
            )
        )

    return 1.0 / ((w ** 2) + epsilon_update(L, weights, alpha, cov))
```

### Iterative $\ell_1$ - Type II:
\begin{equation}
w_{i}^{(k+1)} \leftarrow\left[{\bf L}_{i}^{T}\left(\alpha I+{\bf L} \widetilde{W}^{(k)} \tilde{X}^{(k)} {\bf L}^{T}\right)^{-1} {\bf L}_{i}\right]^{\frac{1}{2}}
\end{equation}


```python
def gprime(coef):
    L_T = L.T

    def w_mat(weights):
        return np.diag(1.0 / weights)

    x_mat = np.abs(np.diag(coef))
    noise_cov = cov
    proj_source_cov = (L @ np.dot(w_mat(weights), x_mat)) @ L_T
    signal_cov = noise_cov + proj_source_cov
    sigmaY_inv = np.linalg.inv(signal_cov)

    return np.sqrt(np.diag((L_T @ sigmaY_inv) @ L))

```

![](https://i.imgur.com/ptWWQaB.png)

## Methods Comparsision

![](https://i.imgur.com/yoNaj1n.png)

|                              | EMD (validation) | EMD (test) | Jaccard (validation) | Jaccard (test) | Submission Name     |
| ---------------------------- | ---------------- | ---------- | -------------------- | -------------- | ------------------- |
| KNN                          |  1.0             |   1.0      |        1.0           |       1.0      | "knn_resample"      |
| $\ell_1$                     | 0.271            | 0.269      | 0.710                | 0.731          | "lasso_lars"        |
| Iterative $\ell_1$           | 0.238            | 0.241      | 0.639                | 0.669          | "iterative_L1"      |
| Iterative $\ell_2$           | 0.253            | 0.256      | 0.641                | 0.669          | "iterative_L2"      |
| Iterative $\ell_{0.5}$       | 0.237            | 0.237      | 0.646                | 0.677          | "iterative_sqrt"    |
| Iterative $\ell_1$ - Type II | **0.230**        |**0.235**   | **0.636**            | **0.667**      | "iterative_L1_TypeII" |


## How about Champagne and Type-II MM methods ? 

### Quick intro
Model the current activity of the brain sources as Gaussian scale mixtures:
* Source distribution: $p(\mathbf{X} \mid \Gamma)=\prod_{t=1}^{T} \mathcal{N}(0, \Gamma), \quad \Gamma:$ Source covariance with a diagonal structure $\Gamma=\operatorname{diag}(\gamma)=\operatorname{diag}\left(\left[\gamma_{1}, \ldots, \gamma_{N}\right]^{\top}\right)$
* Noise distribution: $\mathbf{E}=[\mathbf{e}(1), \ldots, \mathbf{e}(T)], \mathbf{e}(t) \sim \mathcal{N}(0, \Lambda), t=1, \cdots, T$, where $\Lambda$: Noise covariance with full structure
* Measurement distribution: $p(\mathbf{Y} \mid \mathbf{X})=\prod_{t=1}^{T} \mathcal{N}(\mathbf{L} \mathbf{x}(t), \mathbf{\Lambda})$
*  Posterior source distribution: $p(\mathbf{X} \mid \mathbf{Y}, \mathbf{\Gamma})=\prod_{t=1}^{T} \mathcal{N}\left(\overline{\mathbf{x}}(t), \boldsymbol{\Sigma}_{\mathbf{x}}\right),$ with
$$
\overline{\mathbf{x}}(t)=\Gamma \mathbf{L}^{\top}\left(\boldsymbol{\Sigma}_{\mathbf{y}}\right)^{-1} \mathbf{y}(t) \quad \boldsymbol{\Sigma}_{\mathbf{x}}=\boldsymbol{\Gamma}-\boldsymbol{\Gamma} \mathbf{L}^{\top}\left(\boldsymbol{\Sigma}_{\mathbf{y}}\right)^{-1} \mathbf{L} \boldsymbol{\Gamma} \quad \boldsymbol{\Sigma}_{\mathbf{y}}=\boldsymbol{\Lambda}+\mathbf{L} \Gamma \mathbf{L}^{\top},
$$
as a result of learning $\Gamma$ and $\Lambda$ (hyper-parameters) through minimizing the negative log-likelihood (Type-II) loss, $-\log p(\mathbf{Y} \mid \boldsymbol{\Gamma}, \boldsymbol{\Lambda})$
$$
\text { Type }-\text { II Loss : } \mathcal{L}^{\text {II }}(\boldsymbol{\Gamma}, \boldsymbol{\Lambda})=\log \left|\boldsymbol{\Lambda}+\mathbf{L} \Gamma \mathbf{L}^{\top}\right|+\frac{1}{T} \sum_{t=1}^{T} \mathbf{y}(t)^{\top}\left(\mathbf{\Lambda}+\mathbf{L} \Gamma \mathbf{L}^{\top}\right)^{-1} \mathbf{y}(t)
$$

### MM Unification Framework

![](https://i.imgur.com/V1j5Wj5.png)

Implementation of different cases in Python

```python
if update_mode == 1:
    # MacKay fixed point update
    numer = gammas ** 2 * np.mean((A * A.conj()).real, axis=1)
    denom = gammas * np.sum(L * Sigma_y_invL, axis=0)
elif update_mode == 2:
    # convex-bounding update
    numer = gammas * np.sqrt(np.mean((A * A.conj()).real, axis=1))
    denom = np.sum(L * Sigma_y_invL, axis=0)  # sqrt is applied below
elif update_mode == 3:
    # Expectation Maximization (EM) update
    numer = gammas ** 2 * np.mean((A * A.conj()).real, axis=1) \
        + gammas * (1 - gammas * np.sum(L * Sigma_y_invL, axis=0))
else:
    raise ValueError('Invalid value for update_mode')
```


```python

```
