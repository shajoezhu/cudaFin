\name{rolling_predict}
\alias{rolling_predict}
\title{Rolling-window lasso forecasts via the rcupp framework}
\usage{
rolling_predict(x, y, window = 50L, lambda = 0.01, n_iter = 100L)
}
\arguments{
  \item{x}{Numeric matrix \code{T x p} of predictors (column-major / R order).}
  \item{y}{Numeric response of length \code{T} (e.g. next-period returns).}
  \item{window}{Number of trailing rows used to fit each window (>= 2, < T).}
  \item{lambda}{Lasso penalty (>= 0).}
  \item{n_iter}{Number of ISTA steps per window (>= 1).}
}
\value{
A list of class \code{"rolling_predict"} with \code{pred} (one-step-ahead
forecasts), \code{actual} (realised \code{y}), \code{coefficients}
(\code{p x (T-window)} path of fitted betas), \code{ynorm} (per-window
response norm), \code{iters}, \code{mse}, \code{dir_acc}, and \code{backend}
(\code{"cuda"} or \code{"cpu-fallback"}).
}
\description{
Refits a lasso by proximal gradient (ISTA) on each rolling window of
\code{x} and forms one-step-ahead forecasts, calling the fixed GPU operators
exposed by \pkg{rcupp} through \code{LinkingTo: rcupp} and
\code{R_GetCCallable()}. This package ships no CUDA code and no CUDA build
logic of its own; it works identically whether \pkg{rcupp} was built with a
CUDA backend or its CPU fallback.
}
\details{
The step size uses the conservative Lipschitz bound \code{1/sum(Xw^2)} and
\code{n_iter} is fixed, so the whole path is deterministic and exactly
reproducible by a pure-R reference (used in the package tests).
}
