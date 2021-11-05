BSI Zoo
=========================================

# Brain Source Imaging (BSI) Zoo
The goal of this repository is to efficiently implement all brain source imaging techniques and benchmark their perfromance with respect to different metrics. 
 

## Iterative methdos
* Should define a function called "get_estimator" that returns a **scikit-learn type of pipeline**.
\begin{equation}
\boldsymbol{x}^{(k+1)} \leftarrow \arg \min _{\boldsymbol{x}}\|\boldsymbol{y}-{\bf L} \boldsymbol{x}\|_{2}^{2}+\lambda \sum_{i} g(x_{i})
\end{equation}


\begin{equation}
\boldsymbol{x}^{(k+1)} \leftarrow \arg \min _{\boldsymbol{x}}\|\boldsymbol{y}-{\bf L} \boldsymbol{x}\|_{2}^{2}+\lambda \sum_{i} w_{i}^{(k)}\left|x_{i}\right|
\end{equation}

### Iterative $\ell_1$:
\begin{equation}
g(x_{i}) = \log \left(\left|x_{i}\right|+\epsilon\right)
\end{equation}

\begin{equation}
w_{i}^{(k+1)} \leftarrow\left[\left|x_{i}^{(k)}\right|+\epsilon\right]^{-1}
\end{equation}

```python=
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

```python=
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

```python=
g = lambda w: np.sqrt(np.abs(w))
gprime = lambda w: 1. / (2. * np.sqrt(np.abs(w)) + eps)
```
![](https://i.imgur.com/lP6QKEH.png)


