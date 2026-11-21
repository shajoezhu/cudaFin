#include <Rcpp.h>
#include <rcupp.h>
#include <cstddef>
#include <string>
#include <vector>

/*
 * cudaFin P3-fin: a DOWNSTREAM rolling-window composition that keeps ONE full
 * feature panel RESIDENT on the device and refits every window without ever
 * re-uploading the window matrix.
 *
 * This is the resolution of the demo-3 FIN "0/6" gap documented in the
 * original src/fin.cpp: there, each rcupp_* call is the F0 one-shot ABI
 * (H->D->H per call), so every ISTA iteration re-copies Xw to the device and
 * back. Rolling windows change X each window, so the lasso-specific P1
 * session (one X across many lambdas) cannot help either.
 *
 * Here the solver loop is written DOWNSTREAM in C++ (Rcpp glue) and composes
 * rcupp's generic Phase-0 RESIDENT parts through <rcupp.h> (LinkingTo: rcupp
 * -> -I.../include, R_GetCCallable()). rcupp still knows nothing about lasso,
 * rolling windows or forecasting -- it only supplies device buffers + parts.
 * NO new rcupp operator is added and rcupp_ctx_abi() stays 1. NO CUDA code,
 * NO configure / Makevars / CUDA detection lives in this package: identical
 * numerics on the CUDA backend and the CPU fallback.
 *
 * ONE rcupp_ctx owns every buffer. strategy = "resident" uploads the full
 * T x p X and length-T y ONCE, then each window is obtained by DEVICE-SIDE
 * rcupp_d_gather_rows (rows t-window .. t-1, plus the forecast row t) with no
 * host<->device traffic beyond O(window) index ints. strategy = "cold" instead
 * reassembles each window on the host and re-uploads it per fit -- the ROI
 * baseline that isolates the transfer saving, exactly as in cudaGlmnet's
 * cuda_resident_subsets(). Both strategies share the SAME resident ISTA loop
 * composed only from level-1 resident operators (gemv / copy / axpy / scal /
 * gemv_t / soft_threshold), so no per-operator H->D->H copies occur.
 *
 * Numerical contract is IDENTICAL to src/fin.cpp and the pure-R reference in
 * tests/testthat/test-fin.R:
 *   Lm    = ||Xw||_F^2  (rcupp_d_dot), step = 1/Lm
 *   thresh = lambda * window * step
 *   beta  <- prox(beta + step * Xw^T (yw - Xw beta))  for n_iter steps
 *   pred  = <X[t, ], beta>;  ynorm = ||yw||_2
 * We deliberately compose level-1 operators directly rather than calling the
 * level-2 rcupp_d_prox_grad_step() once per iteration: that shared part
 * allocates/frees scratch buffers inside each call, which is needless churn in
 * a long inner loop. This mirrors cudaGlmnet/src/enet_path.cpp.
 */

namespace {

struct CtxGuard {
  rcupp_ctx c;
  CtxGuard() : c(rcupp_ctx_create()) {}
  ~CtxGuard() { if (c) rcupp_ctx_destroy(c); }
  CtxGuard(const CtxGuard&) = delete;
  CtxGuard& operator=(const CtxGuard&) = delete;
};

// Buffers are owned by the ctx (freed at rcupp_ctx_destroy); guard the one
// genuinely fallible creation path.
inline rcupp_buf need_buf(rcupp_ctx c, int n) {
  rcupp_buf b = rcupp_buf_alloc(c, n);
  if (!b) Rcpp::stop("rcupp: buffer allocation failed");
  return b;
}

}  // namespace

// [[Rcpp::export]]
Rcpp::List rolling_resident_cpp(Rcpp::NumericMatrix X, Rcpp::NumericVector y,
                                 int window, double lambda, int n_iter,
                                 std::string strategy) {
  if (rcupp_ctx_abi() != 1) {
    Rcpp::stop("rcupp >= 0.1.0 (device-context ABI 1) is required");
  }
  const int T = X.nrow();
  const int p = X.ncol();
  const int np = T - window;  // number of one-step-ahead forecasts
  const bool resident = (strategy == "resident");
  if (!resident && strategy != "cold") {
    Rcpp::stop("'strategy' must be \"resident\" or \"cold\"");
  }
  if (window < 1 || window >= T || p < 1 || lambda < 0.0 || n_iter < 1 ||
      y.size() != static_cast<R_xlen_t>(T)) {
    Rcpp::stop("invalid rolling_resident parameters");
  }

  Rcpp::NumericMatrix coef(p, np);  // p x np, column-major
  Rcpp::NumericVector pred(np), actual(np), ynorm(np);
  double total_iters = 0.0;

  CtxGuard g;
  if (!g.c) Rcpp::stop("rcupp: device context creation failed");
  rcupp_ctx c = g.c;

  // Resident working buffers (owned by ctx). Xw is the current window matrix
  // (window x p), yw its response, beta the ISTA iterate, row the forecast
  // feature row; Xb/r are length-window scratch, gg/u length-p scratch.
  rcupp_buf Xw   = need_buf(c, window * p);
  rcupp_buf yw   = need_buf(c, window);
  rcupp_buf beta = need_buf(c, p);
  rcupp_buf Xb   = need_buf(c, window);
  rcupp_buf r    = need_buf(c, window);
  rcupp_buf gg   = need_buf(c, p);
  rcupp_buf u    = need_buf(c, p);
  rcupp_buf row  = need_buf(c, p);

  // Resident strategy: upload the FULL panel once, gather rows device-side.
  // p=1 tensors (yfull, yw) and m=1 gathers (the forecast row) are supported
  // by the same gather_rows kernel (total = m*p).
  rcupp_buf Xfull = NULL, yfull = NULL;
  if (resident) {
    Xfull = need_buf(c, T * p);
    yfull = need_buf(c, T);
    rcupp_buf_upload(c, Xfull, X.begin(), T * p);
    rcupp_buf_upload(c, yfull, y.begin(), T);
  }

  // Cold strategy: host scratch reassembled and uploaded every window.
  std::vector<double> Xw_h, yw_h, row_h;
  if (!resident) {
    Xw_h.resize(static_cast<std::size_t>(window) * p);
    yw_h.resize(window);
    row_h.resize(p);
  }
  // Zero-based row indices of the current training window, reused per fit.
  std::vector<int> idx(window);

  for (int t = window; t < T; ++t) {
    const int slot = t - window;
    for (int k = 0; k < window; ++k) idx[k] = t - window + k;

    if (resident) {
      // Device-side row selection: Xw = X[idx, ], yw = y[idx], row = X[t, ].
      // No host<->device round trip for the window data.
      rcupp_d_gather_rows(c, Xfull, T, p, idx.data(), window, Xw);
      rcupp_d_gather_rows(c, yfull, T, 1, idx.data(), window, yw);
      int ridx = t;  // m = 1 gather of a single forecast feature row
      rcupp_d_gather_rows(c, Xfull, T, p, &ridx, 1, row);
    } else {
      // Cold: assemble this window contiguously on the host, then upload.
      for (int j = 0; j < p; ++j) {
        for (int k = 0; k < window; ++k) {
          Xw_h[static_cast<std::size_t>(k) +
               static_cast<std::size_t>(j) * window] =
              X[(t - window + k) + static_cast<std::size_t>(j) * T];
        }
        row_h[j] = X[t + static_cast<std::size_t>(j) * T];
      }
      for (int k = 0; k < window; ++k) yw_h[k] = y[t - window + k];
      rcupp_buf_upload(c, Xw, Xw_h.data(), window * p);
      rcupp_buf_upload(c, yw, yw_h.data(), window);
      rcupp_buf_upload(c, row, row_h.data(), p);
    }

    // Conservative Lipschitz bound ||Xw||_F^2 and per-window response norm
    // (host-returning scalar reductions, one small sync point each).
    const double Lm = rcupp_d_dot(c, window * p, Xw, Xw);
    const double step = (Lm > 0.0) ? 1.0 / Lm : 0.0;
    const double thresh = lambda * static_cast<double>(window) * step;
    ynorm[slot] = rcupp_d_nrm2(c, window, yw);

    // Downstream ISTA loop: every operator runs on resident buffers, with NO
    // per-call host<->device copies. rcupp never sees the lasso / prox logic.
    rcupp_buf_fill(c, beta, 0.0, p);
    for (int it = 0; it < n_iter; ++it) {
      rcupp_d_gemv(c, Xw, window, p, beta, Xb);        // Xb = Xw beta
      rcupp_d_copy(c, window, yw, r);                  // r = yw
      rcupp_d_axpy(c, window, -1.0, Xb, r);            // r = yw - Xb
      rcupp_d_scal(c, window, step, r);                // r *= step
      rcupp_d_gemv_t(c, Xw, window, p, r, gg);         // gg = Xw^T r
      rcupp_d_copy(c, p, beta, u);                     // u = beta
      rcupp_d_axpy(c, p, 1.0, gg, u);                  // u += gg
      rcupp_d_soft_threshold(c, u, thresh, beta, p);   // beta = prox(u)
      total_iters += 1.0;
    }

    // Forecast at row t (resident dot) and download the fitted betas.
    pred[slot] = rcupp_d_dot(c, p, row, beta);
    actual[slot] = y[t];
    rcupp_buf_download(c, beta,
                       coef.begin() + static_cast<R_xlen_t>(slot) * p, p);
  }

  return Rcpp::List::create(
      Rcpp::Named("pred")         = pred,
      Rcpp::Named("actual")       = actual,
      Rcpp::Named("coefficients") = coef,
      Rcpp::Named("ynorm")        = ynorm,
      Rcpp::Named("iters")        = total_iters,
      Rcpp::Named("cuda")         = static_cast<bool>(rcupp_cuda_available()),
      Rcpp::Named("strategy")     = strategy);
}
