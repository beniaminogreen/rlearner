#' @include utils.R
#'
#' @title X-learner implemented via random forest
#'
#' @description X-learner as proposed by Kunzel, Sekhon, Bickel, and Yu (2017), implemented via xgboost (boosting)
#'
#' @param x the input features
#' @param w the treatment variable (0 or 1)
#' @param y the observed response (real valued)
#' @param p_hat pre-computed estimates on E[W|X] corresponding to the input x. xboost will compute it internnoney if not provided
#' @param ntrees_max the maximum number of trees to grow for xgboost
#' @param num_search_rounds the number of random sampling of hyperparameter combinations for cross validating on xgboost trees
#' @param print_every_n the number of iterations (in each iteration, a tree is grown) by which the code prints out information
#' @param early_stopping_rounds the number of rounds the test error stops decreasing by which the cross validation in finding the optimal number of trees stops
#' @param nthread the number of threads to use. The default is NULL, which uses none available threads
#' @param verbose boolean; whether to print statistic
#'
#' @export
xrf = function(x, w, y, p_hat, clusters=NULL, tune.parameters = "none") {

  x_1 = x[which(w==1),]
  x_0 = x[which(w==0),]

  y_1 = y[which(w==1)]
  y_0 = y[which(w==0)]

  if (is.null(clusters)) {
    clusters_1 <- NULL
    clusters_0 <- NULL
  } else {
    clusters_1 <- clusters[which(w==1)]
    clusters_0 <- clusters[which(w==0)]
  }


  nobs_1 = nrow(x_1)
  nobs_0 = nrow(x_0)

  nobs = nrow(x)
  pobs = ncol(x)

  t_1_fit <- regression_forest(x_1, y_1, clusters = clusters_1, tune.parameters = tune.parameters)
  mu1_hat <- predict(t_1_fit, newdata = x)$predictions

  t_0_fit <- regression_forest(x_0, y_0, clusters = clusters_0, tune.parameters = tune.parameters)
  mu0_hat <- predict(t_0_fit, newdata = x)$predictions

  d_1 = y_1 - mu0_hat[w==1]
  d_0 = mu1_hat[w==0] - y_0

  x_1_fit <- regression_forest(x_1, d_1, clusters = clusters_1, tune.parameters = tune.parameters)
  x_0_fit <- regression_forest(x_0, d_0, clusters = clusters_1, tune.parameters = tune.parameters)

  tau_1_pred = predict(x_1_fit, newdata = x)$predictions
  tau_0_pred = predict(x_0_fit, newdata = x)$predictions

  tau_hat = tau_1_pred * (1 - p_hat) + tau_0_pred * p_hat

  ret = list(t_1_fit = t_1_fit,
             t_0_fit = t_0_fit,
             x_1_fit = x_1_fit,
             x_0_fit = x_0_fit,
             mu1_hat = mu1_hat,
             mu0_hat = mu0_hat,
             tau_1_pred = tau_1_pred,
             tau_0_pred = tau_0_pred,
             p_hat = p_hat,
             tau_hat = tau_hat)
  class(ret) <- "xrf"
  ret
}

predict.xrf <- function(object, newdata = NULL, new_p_hat = NULL, ...) {
  if (!is.null(newdata)) {
    tau_1_pred = predict(object$x_1_fit, newdata = newdata)
    tau_0_pred = predict(object$x_0_fit, newdata = newdata)
    stopifnot("new_p_hat must be specified"=!is.null(new_p_hat))
    tau_hat = tau_1_pred * (1 - new_p_hat) + tau_0_pred * new_p_hat
  } else {
    tau_hat = object$tau_hat
  }
  return(tau_hat)
}
