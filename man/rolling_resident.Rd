\name{rolling_resident}
\alias{rolling_resident}
\title{Resident rolling-window lasso forecasts via the rcupp device-context ABI}
\usage{
rolling_resident(x, y, window = 50L, lambda = 0.01, n_iter = 100L,
  strategy = c("resident", "cold"))
}
\arguments{
  \item{x}{Numeric matrix \code{T x p} of predictors (column-major / R order).}
  \item{y}{Numeric response of length \code{T} (e.g. next-period returns).}
  \item{window}{Number of trailing rows used to fit each window (>= 2, < T).}
  \item{lambda}{Lasso penalty (>= 0).}
  \item{n_iter}{Number of ISTA steps per window (>= 1).}
  \item{strategy}{Either \code{"resident"} (default; one resident full panel,
    device-side row gather per window) or \code{"cold"} (re-upload each window
    from the host). Chosen by partial matching (e.g. \code{"r"} / \code{"c"}).}
}
\value{
A list of class \code{c("rolling_resident", "rolling_predict")} with
\code{pred} (one-step-ahead forecasts), \code{actual} (realised \code{y}),
\code{coefficients} (\code{p x (T-window)} path of fitted betas),
\code{ynorm} (per-window response norm), \code{iters}, \code{mse},
\code{dir_acc}, \code{backend} (\code{"cuda"} or \code{"cpu-fallback"}),
\code{cuda}, and \code{strategy}.
}
\description{
Refits a lasso by proximal gradient (ISTA) on each rolling window of
\code{x} and forms one-step-ahead forecasts, keeping ONE full feature panel
resident on the device. This is the P3-fin resolution of the "rolling
windows change X every window" gap in \code{\link{rolling_predict}}, whose
F0 one-shot operators copy \code{Xw} host to device on every call.
}
\details{
The solver loop is written downstream in C++ and composes \pkg{rcupp}'s
generic Phase-0 resident parts (\code{gemv} / \code{gemv_t} / \code{axpy}
/ \code{scal} / \code{soft_threshold} / \code{dot} / \code{nrm2}) plus the
device-side \code{gather_rows} part through \code{LinkingTo: rcupp} and
\code{R_GetCCallable()}. \pkg{rcupp} supplies only device buffers and
operators; it never learns about lasso, windows or forecasting. This package
ships no CUDA code of its own and works identically on the CUDA backend and
the CPU fallback. \code{rcupp_ctx_abi()} stays 1; no new \pkg{rcupp}
operator or ABI change is required.

With \code{strategy = "resident"} the full \code{T x p} matrix \code{x} and
length-\code{T} \code{y} are uploaded ONCE, and each window is obtained by a
device-side row gather (no host-to-device traffic beyond the row indices).
With \code{strategy = "cold"} each window is reassembled on the host and
re-uploaded per fit -- the ROI baseline that isolates the transfer saving.
Both strategies share the SAME resident ISTA loop and therefore produce
numerically identical results.

The step size uses the conservative Lipschitz bound \code{1/sum(Xw^2)} and
\code{n_iter} is fixed, so the whole path is deterministic and exactly
reproducible by the pure-R reference used in the package tests (identical
numerical contract to \code{\link{rolling_predict}}).
}
\seealso{
\code{\link{rolling_predict}}
}
