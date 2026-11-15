#' Rolling-window lasso forecasts via the rcupp framework
#'
#' Refits a lasso by proximal gradient (ISTA) on each rolling window of
#' \code{x} and forms one-step-ahead forecasts, calling the fixed GPU
#' operators exposed by \pkg{rcupp} through \code{LinkingTo: rcupp} and
#' \code{R_GetCCallable()}. This package ships no CUDA code and no CUDA build
#' logic of its own; it works identically whether \pkg{rcupp} was built with a
#' CUDA backend or its CPU fallback.
#'
#' The step size uses the conservative Lipschitz bound \code{1/sum(Xw^2)} and
#' \code{n_iter} is fixed, so the whole path is deterministic and exactly
#' reproducible by a pure-R reference (used in the package tests).
#'
#' @param x Numeric matrix \code{T x p} of predictors (column-major / R order).
#' @param y Numeric response of length \code{T} (e.g. next-period returns).
#' @param window Number of trailing rows used to fit each window (>= 2, < T).
#' @param lambda Lasso penalty (>= 0).
#' @param n_iter Number of ISTA steps per window (>= 1).
#' @return A list of class \code{"rolling_predict"} with \code{pred}
#'   (one-step-ahead forecasts), \code{actual} (realised \code{y}),
#'   \code{coefficients} (\code{p x (T-window)} path of fitted betas),
#'   \code{ynorm} (per-window response norm), \code{iters}, \code{mse},
#'   \code{dir_acc}, and \code{backend} (\code{"cuda"} or
#'   \code{"cpu-fallback"}).
#' @export
rolling_predict <- function(x, y, window = 50L, lambda = 0.01,
                            n_iter = 100L) {
  x <- as.matrix(x)
  if (!is.numeric(x) || anyNA(x)) {
    stop("'x' must be a numeric matrix without NA", call. = FALSE)
  }
  if (storage.mode(x) != "double") storage.mode(x) <- "double"
  y <- as.numeric(y)
  if (anyNA(y) || length(y) != nrow(x)) {
    stop("'y' must be numeric, without NA, and length nrow(x)", call. = FALSE)
  }

  n <- nrow(x)
  window <- as.integer(window)
  n_iter <- as.integer(n_iter)
  if (length(window) != 1L || is.na(window) || window < 2L || window >= n) {
    stop("'window' must be an integer between 2 and nrow(x) - 1", call. = FALSE)
  }
  if (length(lambda) != 1L || is.na(lambda) || lambda < 0) {
    stop("'lambda' must be a non-negative number", call. = FALSE)
  }
  if (length(n_iter) != 1L || is.na(n_iter) || n_iter < 1L) {
    stop("'n_iter' must be a positive integer", call. = FALSE)
  }

  # cuda_available() is imported from rcupp; referencing it guarantees the
  # rcupp namespace (and its DLL, hence R_init_rcupp registration) is loaded
  # before the C++ R_GetCCallable() lookups below.
  backend <- if (cuda_available()) "cuda" else "cpu-fallback"

  res <- rolling_predict_cpp(x, y, window, as.double(lambda), n_iter)
  res$backend <- backend

  # One-step-ahead forecast diagnostics. The numerical contract is the
  # pred/coef/ynorm path, which tests mirror against pure R.
  res$mse <- mean((res$pred - res$actual)^2)
  res$dir_acc <- mean(sign(res$pred) == sign(res$actual))
  class(res) <- "rolling_predict"
  res
}
