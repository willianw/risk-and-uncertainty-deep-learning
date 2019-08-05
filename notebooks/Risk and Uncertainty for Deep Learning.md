
    Using TensorFlow backend.
    /home/willian/notebooks_env/risk-and-uncertainty-deep-learning/env/lib/python3.7/site-packages/tensorflow/python/framework/dtypes.py:516: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_qint8 = np.dtype([("qint8", np.int8, 1)])
    /home/willian/notebooks_env/risk-and-uncertainty-deep-learning/env/lib/python3.7/site-packages/tensorflow/python/framework/dtypes.py:517: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_quint8 = np.dtype([("quint8", np.uint8, 1)])
    /home/willian/notebooks_env/risk-and-uncertainty-deep-learning/env/lib/python3.7/site-packages/tensorflow/python/framework/dtypes.py:518: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_qint16 = np.dtype([("qint16", np.int16, 1)])
    /home/willian/notebooks_env/risk-and-uncertainty-deep-learning/env/lib/python3.7/site-packages/tensorflow/python/framework/dtypes.py:519: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_quint16 = np.dtype([("quint16", np.uint16, 1)])
    /home/willian/notebooks_env/risk-and-uncertainty-deep-learning/env/lib/python3.7/site-packages/tensorflow/python/framework/dtypes.py:520: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_qint32 = np.dtype([("qint32", np.int32, 1)])
    /home/willian/notebooks_env/risk-and-uncertainty-deep-learning/env/lib/python3.7/site-packages/tensorflow/python/framework/dtypes.py:525: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      np_resource = np.dtype([("resource", np.ubyte, 1)])
    /home/willian/notebooks_env/risk-and-uncertainty-deep-learning/env/lib/python3.7/site-packages/tensorboard/compat/tensorflow_stub/dtypes.py:541: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_qint8 = np.dtype([("qint8", np.int8, 1)])
    /home/willian/notebooks_env/risk-and-uncertainty-deep-learning/env/lib/python3.7/site-packages/tensorboard/compat/tensorflow_stub/dtypes.py:542: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_quint8 = np.dtype([("quint8", np.uint8, 1)])
    /home/willian/notebooks_env/risk-and-uncertainty-deep-learning/env/lib/python3.7/site-packages/tensorboard/compat/tensorflow_stub/dtypes.py:543: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_qint16 = np.dtype([("qint16", np.int16, 1)])
    /home/willian/notebooks_env/risk-and-uncertainty-deep-learning/env/lib/python3.7/site-packages/tensorboard/compat/tensorflow_stub/dtypes.py:544: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_quint16 = np.dtype([("quint16", np.uint16, 1)])
    /home/willian/notebooks_env/risk-and-uncertainty-deep-learning/env/lib/python3.7/site-packages/tensorboard/compat/tensorflow_stub/dtypes.py:545: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_qint32 = np.dtype([("qint32", np.int32, 1)])
    /home/willian/notebooks_env/risk-and-uncertainty-deep-learning/env/lib/python3.7/site-packages/tensorboard/compat/tensorflow_stub/dtypes.py:550: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      np_resource = np.dtype([("resource", np.ubyte, 1)])
    WARNING: Logging before flag parsing goes to stderr.
    W0804 17:03:11.251269 139710163740416 blas_headers.py:988] Using NumPy C-API based implementation for BLAS functions.


# Risk vs. Uncertainty

There are diverse definitions for the distinction between risk and uncertainty.
The most famous comes from Frank Knight, that states:
> Uncertainty must be taken in a sense radically distinct from the familiar notion of Risk, from which it has never been properly separated. The term "risk," as loosely used in everyday speech and in economic discussion, really covers two things which, functionally at least, in their causal relations to the phenomena of economic organization, are categorically different. ... The essential fact is that "risk" means in some cases a quantity susceptible of measurement, while at other times it is something distinctly not of this character; and there are far-reaching and crucial differences in the bearings of the phenomenon depending on which of the two is really present and operating. ... It will appear that a measurable uncertainty, or "risk" proper, as we shall use the term, is so far different from an unmeasurable one that it is not in effect an uncertainty at all. We ... accordingly restrict the term "uncertainty" to cases of the non-quantitive type.

Here we'll use a variant from Ian Osband:
> [...] We identify risk as inherent stochasticity in a model and uncertainty as the confusion over which model parameters apply. For example, a coin may have a fixed $p = 0.5$ of heads and so the outcome of any single flip holds some risk; a learning agent may also be uncertain of $p$.

# The data

We'll use a simulated example for a heteroskedastic random variable $Y$:

$$
y = 2x + 6\sin\left(2\pi x+\frac{\epsilon}{48}\right) \cdot \mathcal{H}(x+2),\\
\epsilon \sim \mathcal{N}\left(0, \ \frac{\pi}{6} + \arctan\left(1.2x + 1\right)\right), \quad x \sim \mathcal{N}(0, 1)
$$

in which $\mathcal{H}$ stands for [Heaviside function](https://en.wikipedia.org/wiki/Heaviside_step_function)


![png](Risk%20and%20Uncertainty%20for%20Deep%20Learning_files/Risk%20and%20Uncertainty%20for%20Deep%20Learning_7_0.png)


    /home/willian/notebooks_env/risk-and-uncertainty-deep-learning/env/lib/python3.7/site-packages/seaborn/distributions.py:323: MatplotlibDeprecationWarning: Saw kwargs ['c', 'color'] which are all aliases for 'color'.  Kept value from 'color'.  Passing multiple aliases for the same property will raise a TypeError in 3.3.
      ax.plot(x, y, color=color, label=label, **kwargs)



![png](Risk%20and%20Uncertainty%20for%20Deep%20Learning_files/Risk%20and%20Uncertainty%20for%20Deep%20Learning_8_1.png)



![png](Risk%20and%20Uncertainty%20for%20Deep%20Learning_files/Risk%20and%20Uncertainty%20for%20Deep%20Learning_9_0.png)


## Simple Regression

    W0804 17:03:21.341382 139710163740416 deprecation_wrapper.py:119] From /home/willian/notebooks_env/risk-and-uncertainty-deep-learning/env/lib/python3.7/site-packages/keras/backend/tensorflow_backend.py:74: The name tf.get_default_graph is deprecated. Please use tf.compat.v1.get_default_graph instead.
    
    W0804 17:03:21.388681 139710163740416 deprecation_wrapper.py:119] From /home/willian/notebooks_env/risk-and-uncertainty-deep-learning/env/lib/python3.7/site-packages/keras/backend/tensorflow_backend.py:517: The name tf.placeholder is deprecated. Please use tf.compat.v1.placeholder instead.
    
    W0804 17:03:21.423684 139710163740416 deprecation_wrapper.py:119] From /home/willian/notebooks_env/risk-and-uncertainty-deep-learning/env/lib/python3.7/site-packages/keras/backend/tensorflow_backend.py:4138: The name tf.random_uniform is deprecated. Please use tf.random.uniform instead.
    
    W0804 17:03:21.511224 139710163740416 deprecation_wrapper.py:119] From /home/willian/notebooks_env/risk-and-uncertainty-deep-learning/env/lib/python3.7/site-packages/keras/optimizers.py:790: The name tf.train.Optimizer is deprecated. Please use tf.compat.v1.train.Optimizer instead.
    
    W0804 17:03:21.795725 139710163740416 deprecation_wrapper.py:119] From /home/willian/notebooks_env/risk-and-uncertainty-deep-learning/env/lib/python3.7/site-packages/keras/backend/tensorflow_backend.py:986: The name tf.assign_add is deprecated. Please use tf.compat.v1.assign_add instead.
    
    W0804 17:03:22.145732 139710163740416 deprecation_wrapper.py:119] From /home/willian/notebooks_env/risk-and-uncertainty-deep-learning/env/lib/python3.7/site-packages/keras/backend/tensorflow_backend.py:973: The name tf.assign is deprecated. Please use tf.compat.v1.assign instead.
    



![png](Risk%20and%20Uncertainty%20for%20Deep%20Learning_files/Risk%20and%20Uncertainty%20for%20Deep%20Learning_13_0.png)



![png](Risk%20and%20Uncertainty%20for%20Deep%20Learning_files/Risk%20and%20Uncertainty%20for%20Deep%20Learning_15_0.png)


# Methods

- variational inference
- Monte Carlo Dropout
- BOOTStRAP

## Markov chain Monte Carlo – Metropolis-Hastings

    Sampling 3 chains: 100%|██████████| 3000/3000 [00:04<00:00, 668.29draws/s]
    E0804 17:04:13.586716 139710163740416 report.py:143] The gelman-rubin statistic is larger than 1.4 for some parameters. The sampler did not converge.
    E0804 17:04:13.587601 139710163740416 report.py:143] The estimated number of effective samples is smaller than 200 for some parameters.





    array([[<matplotlib.axes._subplots.AxesSubplot object at 0x7f1018c4f510>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7f1018c19d50>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x7f1018bd0dd0>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7f1019b52e50>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x7f101a3841d0>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7f1018b3d610>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x7f1018af48d0>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7f1018aaabd0>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x7f1018ae2e10>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7f1018a97fd0>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x7f1018a51e50>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7f1018a0e690>]],
          dtype=object)




![png](Risk%20and%20Uncertainty%20for%20Deep%20Learning_files/Risk%20and%20Uncertainty%20for%20Deep%20Learning_20_1.png)


## Variational Inference

### Introduction

In a ML problem, we want to aproximate $\hat{f}(X) = Y$. Given that a neural network has weights $w$, we want to maximize the probability $p(Y|w, X)$. During trainning, we adjust $w$ so that $p$ increases. Now for uncertainty we need the posterior probability of weights, i.e., $p(w|Y, X)$. Using Bayes's Theorem:
$$p(w|Y, X) = \frac{p(Y|w, X) \cdot p(w|Y)}{p(X|Y)} = $$
$$= \frac{p(Y|w, X) \cdot p(w|Y)}{p(X|Y)}$$

### The Kullback-Leibler divergence

Given two distributions, $p$ and $q$, we can establish the following similarity [quasimeasure](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence):

$$
\begin{align}
KL(q || p) = \sum_{x}&q(x)\log\left[\frac{q(x)}{p(x)}\right], \qquad\text{Discrete case}\\
KL(q || p) = \int_{-\infty}^\infty &q(x)\log\left[\frac{q(x)}{p(x)}\right]dx, \qquad\text{Continuous case}\\
\end{align}
$$

It's important to tell that it's not completely a distance measure, since $KL(q || p) \neq KL(p || q)$.

## Monte-Carlo Dropout

# References

- https://arxiv.org/pdf/1902.10189.pdf
- https://gdmarmerola.github.io/risk-and-uncertainty-deep-learning/  
- https://arxiv.org/pdf/1905.09638.pdf  
- https://arxiv.org/pdf/1505.05424.pdf  
- https://arxiv.org/pdf/1506.02142.pdf  
- https://arxiv.org/pdf/1602.04621.pdf  
- https://arxiv.org/pdf/1806.03335.pdf  
- https://arxiv.org/pdf/1505.05424.pdf  
- https://arxiv.org/pdf/1601.00670.pdf  
- https://ermongroup.github.io/cs228-notes/inference/variational/  
- https://github.com/ericmjl/website/blob/master/content/blog/variational-inference-with-pymc3-a-lightweight-demo/linreg.ipynb  
- https://medium.com/tensorflow/regression-with-probabilistic-layers-in-tensorflow-probability-e46ff5d37baf  
- https://papers.nips.cc/paper/8080-randomized-prior-functions-for-deep-reinforcement-learning.pdf  
- http://proceedings.mlr.press/v37/salimans15.pdf
- https://www.cs.ubc.ca/~schmidtm/Courses/540-W18/L34.pdf
- http://bayesiandeeplearning.org/2016/papers/BDL_4.pdf  
- https://towardsdatascience.com/uncertainty-estimation-for-neural-network-dropout-as-bayesian-approximation-7d30fc7bc1f2  
