# rlearner: R-learner for Quasi-Oracle Estimation of Heterogeneous Treatment Effects

This package implements the R-learner for estimating
heterogeneous treatment effects, as proposed by Nie and Wager (2017). We consider a
setup where we observe data `(X, W, Y)` generated according
to the following general non-parameteric model

```
X ~ P(X)
W ~ P(W|X) where W is in {0,1}
Y = b(X) + (W-0.5)*tau(X) + epsilon
```

with `E[epsilon | X, W] = 0`. 

The R-learner provides a general framework to estimate the heterogeneous treatment effect `tau(X)`. We first estimate marginal effects and treatment propensities in order to form an objective function that isolates the causal component of the signal. Then, we optimize this data-adaptive objective function. The R-learner is flexible and easy to use: For both steps, we can use any loss-minimization method, e.g., the lasso, random forests, boosting, etc.; moreover, these methods can be fine-tuned by cross validation. 

The package implements the R-learner using various machine learning models. In particular, the function `rlasso` is a lightweight implementation of the R-learner using the lasso (glmnet) and uses `cv.glmnet` for cross-fitting and cross-validation; the function `rboost` is a lightweight implementation of the R-learner using gradient boosting (xgboost), and by default randomly searches over a set of hyper-parameter combinations used in xgboost, while cross-validating on the number of trees with an early stopping option for each of the random searches; the function `rkern` is a lightweight implementation of the R-learner using kernel ridge regression with a Gaussian kernel using a version of the KRLS package. The version of the KRLS package can be found [here](https://github.com/xnie/KRLS). Note the version number we use is 1.1.1. It is adapted from the [KRLS2](https://github.com/lukesonnet/KRLS) package version 1.1.0. 

This package is currently in beta, and we expect to make continual improvements to its performance and usability.

### Authors
This package is written and maintained by Xinkun Nie (xinkun@stanford.edu), Alejandro Schuler, and Stefan Wager.

### Installation
To install this package in R, run the following commands:

```R
library(devtools) 
install_github("xnie/rlearner")
```

### Example usage:

Below is an example of using the function `rlasso`, `rboost`, and `rkern`.

```R
library(rlearner)
n = 100; p = 10

x = matrix(rnorm(n*p), n, p)
w = rbinom(n, 1, 0.5)
y = pmax(x[,1], 0) * w + x[,2] + pmin(x[,3], 0) + rnorm(n)

rlasso_fit = rlasso(x, w, y)
rlasso_est = predict(rlasso_fit, x)

rboost_fit = rboost(x, w, y)
rboost_est = predict(rboost_fit, x)

rkern_fit = rkern(x, w, y)
rkern_est = predict(rkern_fit, x)
```


The package also implements S-, T-, U-, and X-learners. These can be called in a similar fashion. For example,

```R
tlasso_fit = tlasso(data$x, data$w, data$y)
tlasso_tau_hat = predict(tlasso_fit, data$x)

tboost_fit = tboost(data$x, data$w, data$y)
tboost_tau_hat = predict(tboost_fit, data$x)

tkern_fit = tkern(data$x, data$w, data$y)
tkern_tau_hat = predict(tkern_fit, data$x)
```

### Reproducibility
All simulation results in Nie and Wager (2020+) can be reproduced using this package, with the experiments implemented under the directory `/experiments_for_paper`.

### References
Xinkun Nie and Stefan Wager.
<b>Quasi-Oracle Estimation of Heterogeneous Treatment Effects.</b>
Biometrika, forthcoming
[<a href="https://arxiv.org/abs/1712.04912">arxiv</a>]
