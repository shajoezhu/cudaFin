#' Resident rolling-window lasso forecasts via the rcupp device-context ABI
#'
#' Refits a lasso by proximal gradient (ISTA) on each rolling window of
#' \code{x} and forms one-step-ahead forecasts, keeping ONE full feature panel
#' resident on the device. This is the P3-fin resolution of the "rolling
#' windows change X every window" gap in \code{\link{rolling_predict}}, whose
#' F0 one-shot operators copy \code{Xw} host to device on every call.
#'
#' The solver loop is written downstream in C++ and composes \pkg{rcupp}'s
#' generic Phase-0 resident parts (\code{gemv} / \code{gemv_t} / \code{axpy}
#' / \code{scal} / \code{soft_threshold} / \code{dot} / \code{nrm2}) plus the
#' device-side \code{gather_rows} part through \code{LinkingTo: rcupp} and
#' \code{R_GetCCallable()}. \pkg{rcupp} supplies only device buffers and
#' operators; it never learns about lasso, windows or forecasting. This package
#' ships no CUDA code of its own and works identically on the CUDA backend and
#' the CPU fallback. \code{rcupp_ctx_abi()} stays 1; no new \pkg{rcupp}
#' operator or ABI change is required.
#'
#' With \code{strategy = "resident"} the full \code{T x p} matrix \code{x} and
#' length-\code{T} \code{y} are uploaded ONCE, and each window is obtained by a
#' device-side row gather (no host-to-device traffic beyond the row indices).
#' With \code{strategy = "cold"} each window is reassembled on the host and
#' re-uploaded per fit -- the ROI baseline that isolates the transfer saving.
#' Both strategies share the SAME resident ISTA loop and therefore produce
#' numerically identical results.
#'
#' The step size uses the conservative Lipschitz bound \code{1/sum(Xw^2)} and
#' \code{n_iter} is fixed, so the whole path is deterministic and exactly
#' reproducible by the pure-R reference used in the package tests (identical
#' numerical contract to \code{\link{rolling_predict}}).
#'
#' @param x Numeric matrix \code{T x p} of predictors (column-major / R order).
#' @param y Numeric response of length \code{T} (e.g. next-period returns).
#' @param window Number of trailing rows used to fit each window (>= 2, < T).
#' @param lambda Lasso penalty (>= 0).
#' @param n_iter Number of ISTA steps per window (>= 1).
#' @param strategy Either \code{"resident"} (default; one resident full panel,
#'   device-side row gather per window) or \code{"cold"} (re-upload each window
#'   from the host). Chosen by partial matching (e.g. \code{"r"} / \code{"c"}).
#' @return A list of class \code{c("rolling_resident", "rolling_predict")} with
#'   \code{pred} (one-step-ahead forecasts), \code{actual} (realised \code{y}),
#'   \code{coefficients} (\code{p x (T-window)} path of fitted betas),
#'   \code{ynorm} (per-window response norm), \code{iters}, \code{mse},
#'   \code{dir_acc}, \code{backend} (\code{"cuda"} or \code{"cpu-fallback"}),
#'   \code{cuda}, and \code{strategy}.
#' @seealso \code{\link{rolling_predict}}
#' @export
rolling_resident <- function(x, y, window = 50L, lambda = 0.01,
                             n_iter = 100L,
                             strategy = c("resident", "cold")) {
  strategy <- match.arg(strategy)
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

  # Referencing cuda_available() loads the rcupp namespace / DLL (hence the
  # R_init_rcupp C-callable registrations) before the C++ R_GetCCallable()
  # lookups in rolling_resident_cpp().
  backend <- if (cuda_available()) "cuda" else "cpu-fallback"

  res <- rolling_resident_cpp(x, y, window, as.double(lambda), n_iter, strategy)
  res$backend <- backend
  res$mse <- mean((res$pred - res$actual)^2)
  res$dir_acc <- mean(sign(res$pred) == sign(res$actual))
  class(res) <- c("rolling_resident", "rolling_predict")
  res
}
