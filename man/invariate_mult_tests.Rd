% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/utils.R
\name{invariate_mult_tests}
\alias{invariate_mult_tests}
\title{helper function for testing treatment effect is invariant with a factor of 2 when outcome is multiplied with 2}
\usage{
invariate_mult_tests(tau_hat, tau_hat_2, mean_err = 0.1)
}
\arguments{
\item{tau_hat}{user-supplied treatment effect estimate}

\item{tau_hat_2}{user-supplied treatment effect estimate for the setting with outcome is multiplied with 2}

\item{mean_err}{error tolerance on the mean between 2x tau_hat and tau_hat_2}
}
\description{
helper function to test treatment effect is invariant with a factor of 2 when outcome is multiplied with 2
}
