% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/tlasso.R
\name{tlasso}
\alias{tlasso}
\title{T-learner}
\usage{
tlasso(X, Y, W, alpha = 1, nfolds.1 = NULL, nfolds.0 = NULL,
  lambda.choice = c("lambda.min", "lambda.1se"))
}
\arguments{
\item{X}{the input features}

\item{Y}{the observed response (real valued)}

\item{W}{the treatment variable (0 or 1)}

\item{alpha}{tuning parameter for the elastic net}

\item{nfolds.1}{number of folds for learning E[Y|X,W=1]}

\item{nfolds.0}{number of folds for learning E[Y|X,W=0]}

\item{lambda.choice}{how to cross-validate}
}
\description{
T-learner
}