% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/xboost.R
\name{predict.xboost}
\alias{predict.xboost}
\title{predict for xboost}
\usage{
\method{predict}{xboost}(object, newx = NULL, new_p_hat = NULL, ...)
}
\arguments{
\item{object}{a xboost object}

\item{newx}{covariate matrix to make predictions on. If null, return the tau(x) predictions on the training data}

\item{new_p_hat}{propensity score on newx provided by the user. Default to NULL. If the user provided their own propensity p_hat in training, new_p_hat must be provided here.}

\item{...}{additional arguments (currently not used)}
}
\value{
vector of predictions
}
\description{
get estimated tau(x) using the trained xboost model
}
\examples{
\dontrun{
n = 100; p = 10

x = matrix(rnorm(n*p), n, p)
w = rbinom(n, 1, 0.5)
y = pmax(x[,1], 0) * w + x[,2] + pmin(x[,3], 0) + rnorm(n)

xboost_fit = xboost(x, w, y)
xboost_est = predict(xboost_fit, x)
}


}
