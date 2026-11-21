# cudaFin <a href='https://github.com/shajoezhu/cudaFin'><img src="man/figures/logo.png" align="right" height="160" alt="cudaFin hexsticker" /></a>

<!-- start badges -->

[![Check](https://github.com/shajoezhu/cudaFin/actions/workflows/check.yaml/badge.svg)](https://github.com/shajoezhu/cudaFin/actions/workflows/check.yaml)

A minimal **downstream example of the [rcupp](https://github.com/shajoezhu/rcupp)
CUDA-for-Rcpp framework**, for finance-style workloads. It refits a lasso by
proximal gradient (ISTA) on successive **rolling windows** of a feature matrix
and produces one-step-ahead forecasts, calling only `rcupp`'s fixed GPU
operators from its own C++ through `LinkingTo: rcupp` and `R_GetCCallable()`.

Like its sibling `cudaBio`, this package ships **no `.cu` files, no `configure`,
no `src/Makevars`, and no CUDA detection / CPU-fallback logic** of its own. It
compiles and produces identical results whether `rcupp` was built with a real
CUDA backend or its CPU fallback — a template for rolling-window / portfolio
signal workloads without any CUDA build complexity.

## What it provides

- `rolling_predict(x, y, window = 50, lambda = 0.01, n_iter = 100)` — fits a
  lasso on each trailing window of the `T x p` predictor matrix `x` and forms
  one-step-ahead forecasts of `y`. Returns a `rolling_predict` object with
  `pred` (forecasts), `actual` (realised `y`), `coefficients`
  (`p x (T - window)` path of fitted betas), `ynorm`, `iters`, `mse`,
  `dir_acc`, and `backend` (`"cuda"` or `"cpu-fallback"`).
- `rolling_resident(x, y, window = 50, lambda = 0.01, n_iter = 100,
  strategy = "resident")` — the **P3-fin resident** counterpart built on the
  `rcupp` Phase-0 device-context C ABI. ONE full `X`/`y` panel is uploaded
  once and each window is obtained by a device-side `gather_rows`; the ISTA
  inner step calls `rcupp_d_prox_grad_step()` (one fused CUDA kernel per
  iteration, no per-call host-device copies). `strategy = "cold"` re-uploads
  each window from the host as an ROI baseline.
  Numerically identical to `rolling_predict()` on both backends; adds a
  `strategy` field. See the `rolling-residency` vignette for the demo 3
  measurement.

The step size uses the conservative Lipschitz bound `1/sum(Xw^2)` and `n_iter`
is fixed, so the whole path is deterministic and exactly reproducible by a
pure-R reference (used in the package tests). The C++ uses `rcupp_gemv` /
`rcupp_gemv_t` (the gradient), `rcupp_axpy` / `rcupp_scal` (updates),
`rcupp_dot` (norms and predictions) and `rcupp_soft_threshold` (the prox).

## Installation

`rcupp` must be installed first (any backend):

```r
# install.packages("remotes")
remotes::install_github("shajoezhu/rcupp")
remotes::install_github("shajoezhu/cudaFin")
```

No CUDA Toolkit is needed to build `cudaFin` itself — acceleration comes from
whatever `rcupp` was compiled against.

## Quick start

```r
library(cudaFin)
set.seed(1)
T <- 200; p <- 20
X <- matrix(rnorm(T * p), T, p)
y <- drop(X %*% c(2, -1, rep(0, p - 2))) + rnorm(T)

rp <- rolling_predict(X, y, window = 50, lambda = 0.01, n_iter = 100)
length(rp$pred)    # T - window one-step-ahead forecasts
rp$mse
rp$dir_acc         # directional accuracy
rp$backend         # "cuda" or "cpu-fallback"
```

## How it links to rcupp (template for your own package)

```
# DESCRIPTION
Imports: rcupp, Rcpp
LinkingTo: rcupp, Rcpp
NeedsCompilation: yes
```

```
# NAMESPACE
useDynLib(myPkg, .registration = TRUE)
importFrom(rcupp, cuda_available)
importFrom(Rcpp, evalCpp)
```

```cpp
#include <Rcpp.h>
#include <rcupp.h>

// inside your .Call-able function:
rcupp_gemv(Xw, window, p, beta, Xb);   // Xb = Xw %*% beta
```

`LinkingTo: rcupp` adds `-I<rcupp>/include` automatically (no `src/Makevars`).
The `static inline` wrappers in `<rcupp.h>` call `R_GetCCallable("rcupp", ...)`,
so there is no link-time dependency on `rcupp.so` — just make sure to reference
something from `rcupp` (e.g. `cuda_available()`) in R before the C++ runs, so
its namespace and DLL registrations are loaded. Regenerate exports with
`Rcpp::compileAttributes("myPkg")` after editing — `R CMD build/check` does not
run it for you.

> Note on performance: `rolling_predict()` uses the F0 one-shot operators (each
> call copies `Xw` host-to-device), so a workload where every window changes
> `X` is transfer-bound -- the "demo 3 FIN 0/6" gap. `rolling_resident()`
> resolves this with the Phase-0 device-context ABI: one resident full panel,
> device-side row gather per window, and an ISTA loop whose inner step calls
> `rcupp_d_prox_grad_step()` -- ONE fused CUDA kernel per iteration (the CPU
> fallback composes the same level-1 operators with identical numerics).
> `strategy = "cold"` re-uploads each window as an ROI baseline. No new `rcupp`
> operator is added and `rcupp_ctx_abi()` stays 1; the solver loop lives
> downstream in this package. On an equal-work `n_iter = 400` CUDA run this
> takes demo 3 FIN from `0/6` to **`6/6`** configs with `speedup > 1`
> (`max |dcoef| = 2.44e-15`). See the `rolling-residency` vignette for the
> measured resident-vs-cold-vs-pure-R result.

See the `getting-started` vignette and
[rcupp's public API](https://github.com/shajoezhu/rcupp) for the full operator
and session C ABI.

## Part of the cudaGlmnet family

| Package | Role | Link |
|---|---|---|
| rcupp | low-level Rcpp/CUDA bridge + reusable operator/session framework | [github](https://github.com/shajoezhu/rcupp) |
| cudaGlmnet | glmnet-style GPU lasso front end | [github](https://github.com/shajoezhu/cudaGlmnet) |
| cudaBio | PCA / GWAS power iteration (`p >> n`) downstream example | [github](https://github.com/shajoezhu/cudaBio) |
| **cudaFin** | rolling-window lasso forecasting downstream example | this repo |

## License

Apache License 2.0.
