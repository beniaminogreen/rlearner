% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/cvboost.R
\name{cvboost}
\alias{cvboost}
\title{Gradient boosting for regression and classification with cross validation to search for hyper-parameters (implemented with xgboost)}
\usage{
cvboost(
  x,
  y,
  weights = NULL,
  k_folds = NULL,
  objective = c("reg:squarederror", "binary:logistic"),
  ntrees_max = 1000,
  num_search_rounds = 10,
  print_every_n = 100,
  early_stopping_rounds = 10,
  nthread = NULL,
  verbose = FALSE
)
}
\arguments{
\item{x}{the input features}

\item{y}{the observed response (real valued)}

\item{weights}{weights for input if doing weighted regression/classification. If set to NULL, no weights are used}

\item{k_folds}{number of folds used in cross validation}

\item{objective}{choose from either "reg:squarederror" for regression or "binary:logistic" for logistic regression}

\item{ntrees_max}{the maximum number of trees to grow for xgboost}

\item{num_search_rounds}{the number of random sampling of hyperparameter combinations for cross validating on xgboost trees}

\item{print_every_n}{the number of iterations (in each iteration, a tree is grown) by which the code prints out information}

\item{early_stopping_rounds}{the number of rounds the test error stops decreasing by which the cross validation in finding the optimal number of trees stops}

\item{nthread}{the number of threads to use. The default is NULL, which uses all available threads. Note that this does not apply to using bayesian optimization to search for hyperparameters.}

\item{verbose}{boolean; whether to print statistic}
}
\value{
a cvboost object
}
\description{
Gradient boosting for regression and classification with cross validation to search for hyper-parameters (implemented with xgboost)
}
\examples{
\dontrun{
n = 100; p = 10

x = matrix(rnorm(n*p), n, p)
y = pmax(x[,1], 0) + x[,2] + pmin(x[,3], 0) + rnorm(n)

fit = cvboost(x, y, objective="reg:squarederror")
est = predict(fit, x)
}

}
