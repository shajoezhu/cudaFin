#include <Rcpp.h>
#include <rcupp.h>
#include <algorithm>
#include <cstddef>
#include <vector>

/*
 * cudaFin: a DOWNSTREAM consumer of the rcupp CUDA-for-Rcpp framework.
 *
 * This file has NO CUDA code and the package has NO configure / Makevars /
 * CUDA detection / CPU-fallback logic. It reaches rcupp's fixed operators
 * purely through <rcupp.h> (supplied by LinkingTo: rcupp -> -I.../include)
 * and R_GetCCallable(). It compiles and produces identical results whether
 * rcupp was built with a CUDA backend or its CPU fallback.
 *
 * Workload: rolling-window lasso forecasting. For each one-step-ahead time
 * point we refit a lasso by proximal gradient (ISTA) on the trailing window
 * and forecast with the newest feature row -- the kernel behind rolling
 * factor / portfolio signal models. All seven public rcupp operators are
 * exercised here, so the pure-R reference in tests doubles as a full ABI
 * check of the second downstream consumer:
 *
 *   gemv (X beta), axpy + scal (build the scaled residual), gemv_t (X^T r),
 *   soft_threshold (the prox), dot (Lipschitz bound and the forecast), and
 *   nrm2 (per-window response norm).
 *
 * Each rcupp_* call is the F0 one-shot ABI (H->D->H per call); persistent
 * device residency for repeated windows is a later rcupp session API (P1).
 */

// [[Rcpp::export]]
Rcpp::List rolling_predict_cpp(Rcpp::NumericMatrix X, Rcpp::NumericVector y,
                               int window, double lambda, int n_iter) {
  const int T = X.nrow();
  const int p = X.ncol();
  const int np = T - window;  /* number of one-step-ahead forecasts */

  /* The R wrapper validates these; guard them here too so a direct
   * cudaFin:::rolling_predict_cpp() call cannot index out of bounds. */
  if (window < 1 || window >= T || p < 1 || lambda < 0.0 || n_iter < 1) {
    Rcpp::stop("invalid rolling_predict parameters");
  }

  Rcpp::NumericMatrix coef(p, np);          /* p x np, column-major */
  Rcpp::NumericVector pred(np), actual(np), ynorm(np);
  double total_iters = 0.0;

  /* Scratch buffers reused across windows. */
  std::vector<double> Xw(static_cast<std::size_t>(window) * p);
  std::vector<double> yw(window);
  std::vector<double> beta(p), Xb(window), r(window), g(p), u(p), row(p);

  for (int t = window; t < T; ++t) {
    const int slot = t - window;  /* 0 .. np-1 */

    /* Copy this window's training rows into a contiguous column-major Xw,
     * with leading dimension `window`. Training rows are t-window .. t-1. */
    for (int j = 0; j < p; ++j) {
      for (int k = 0; k < window; ++k) {
        Xw[static_cast<std::size_t>(k) +
           static_cast<std::size_t>(j) * window] =
            X[(t - window + k) + static_cast<std::size_t>(j) * T];
      }
    }
    for (int k = 0; k < window; ++k) yw[k] = y[t - window + k];

    /* Conservative Lipschitz bound ||X||_F^2 >= lambda_max(X^T X), so
     * step = 1/||X||_F^2 is a valid ISTA step and the prox threshold is
     * lambda / L = lambda * window / ||X||_F^2 = lambda * window * step. */
    const double Lm = rcupp_dot(window * p, Xw.data(), Xw.data());
    const double step = (Lm > 0.0) ? 1.0 / Lm : 0.0;
    const double thresh = lambda * static_cast<double>(window) * step;

    /* nrm2: a per-window diagnostic, mirrorable in the pure-R reference. */
    ynorm[slot] = rcupp_nrm2(window, yw.data());

    std::fill(beta.begin(), beta.end(), 0.0);
    for (int it = 0; it < n_iter; ++it) {
      rcupp_gemv(Xw.data(), window, p, beta.data(), Xb.data());   /* Xb=Xw beta */
      std::copy(yw.begin(), yw.end(), r.begin());                 /* r = yw    */
      rcupp_axpy(window, -1.0, Xb.data(), r.data());             /* r=yw-Xb   */
      rcupp_scal(window, step, r.data());                        /* r *= step */
      rcupp_gemv_t(Xw.data(), window, p, r.data(), g.data());    /* g=Xw^T r  */
      std::copy(beta.begin(), beta.end(), u.begin());            /* u = beta  */
      rcupp_axpy(p, 1.0, g.data(), u.data());                    /* u+=g      */
      rcupp_soft_threshold(u.data(), thresh, beta.data(), p);    /* beta=prox */
      total_iters += 1.0;
    }

    /* Forecast at row t: gather the (strided) feature row and dot with beta. */
    for (int j = 0; j < p; ++j)
      row[j] = X[t + static_cast<std::size_t>(j) * T];
    pred[slot] = rcupp_dot(p, row.data(), beta.data());
    actual[slot] = y[t];

    for (int j = 0; j < p; ++j)
      coef[j + static_cast<std::size_t>(slot) * p] = beta[j];
  }

  return Rcpp::List::create(
      Rcpp::Named("pred")         = pred,
      Rcpp::Named("actual")       = actual,
      Rcpp::Named("coefficients") = coef,
      Rcpp::Named("ynorm")        = ynorm,
      Rcpp::Named("iters")        = total_iters,
      Rcpp::Named("cuda")         = static_cast<bool>(rcupp_cuda_available()));
}
