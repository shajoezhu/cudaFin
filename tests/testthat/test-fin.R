# Pure-R reference of the exact ISTA contract implemented in src/fin.cpp.
# If these match, every rcupp operator used by cudaFin (gemv, gemv_t, axpy,
# scal, dot, nrm2, soft_threshold) is reachable and numerically correct --
# an independent ABI check of this second downstream consumer.
ista_ref <- function(Xw, yw, lambda, n_iter) {
  n <- nrow(Xw)
  p <- ncol(Xw)
  Lm <- sum(Xw^2)                       # rcupp_dot
  step <- if (Lm > 0) 1 / Lm else 0
  thresh <- lambda * n * step
  beta <- rep(0, p)
  for (it in seq_len(n_iter)) {
    Xb <- as.vector(Xw %*% beta)        # rcupp_gemv
    r <- yw - Xb                        # rcupp_axpy
    r <- r * step                       # rcupp_scal
    g <- as.vector(crossprod(Xw, r))    # rcupp_gemv_t
    u <- beta + g                       # rcupp_axpy
    beta <- sign(u) * pmax(abs(u) - thresh, 0)  # rcupp_soft_threshold
  }
  beta
}

rolling_ref <- function(X, y, window, lambda, n_iter) {
  n <- nrow(X)
  p <- ncol(X)
  np <- n - window
  pred <- actual <- ynorm <- numeric(np)
  coef <- matrix(0, p, np)
  for (tt in (window + 1L):n) {
    slot <- tt - window
    idx <- (tt - window):(tt - 1L)
    Xw <- X[idx, , drop = FALSE]
    yw <- y[idx]
    beta <- ista_ref(Xw, yw, lambda, n_iter)
    pred[slot] <- sum(X[tt, ] * beta)  # rcupp_dot
    actual[slot] <- y[tt]
    ynorm[slot] <- sqrt(sum(yw^2))     # rcupp_nrm2
    coef[, slot] <- beta
  }
  list(pred = pred, actual = actual, ynorm = ynorm, coefficients = coef)
}

test_that("rolling-window forecasts match the pure-R ISTA reference", {
  set.seed(11)
  n <- 60L
  p <- 5L
  X <- matrix(rnorm(n * p), n, p)
  beta_true <- c(0.8, -0.5, 0, 0, 0.3)
  y <- drop(X %*% beta_true) + rnorm(n, sd = 0.5)
  window <- 30L
  lambda <- 0.05
  n_iter <- 120L

  fit <- rolling_predict(X, y, window, lambda, n_iter)
  ref <- rolling_ref(X, y, window, lambda, n_iter)

  expect_s3_class(fit, "rolling_predict")
  expect_equal(dim(fit$coefficients), c(p, n - window))
  expect_equal(fit$pred, ref$pred, tolerance = 1e-8)
  expect_equal(fit$coefficients, ref$coefficients, tolerance = 1e-8)
  expect_equal(fit$ynorm, ref$ynorm, tolerance = 1e-8)
  expect_equal(fit$actual, ref$actual)
  expect_equal(fit$iters, n_iter * (n - window))
  expect_equal(fit$backend,
               if (rcupp::cuda_available()) "cuda" else "cpu-fallback")
  expect_equal(fit$cuda, rcupp::cuda_available())
  # Forecast diagnostics are finite and in range.
  expect_true(is.finite(fit$mse) && fit$mse >= 0)
  expect_true(is.finite(fit$dir_acc) && fit$dir_acc >= 0 && fit$dir_acc <= 1)
})

test_that("rolling_predict works on CPU fallback and CUDA backends alike", {
  set.seed(3)
  X <- matrix(rnorm(40 * 3), 40, 3)
  y <- rnorm(40)
  # The result must not depend on which backend rcupp was built with; checking
  # against the pure-R reference covers both the GPU and CPU code paths.
  fit <- rolling_predict(X, y, window = 20L, lambda = 0.1, n_iter = 50L)
  ref <- rolling_ref(X, y, 20L, 0.1, 50L)
  expect_equal(fit$pred, ref$pred, tolerance = 1e-8)
  expect_equal(fit$coefficients, ref$coefficients, tolerance = 1e-8)
})

test_that("rolling_predict validates inputs", {
  X <- matrix(rnorm(60 * 5), 60, 5)
  y <- rnorm(60)
  expect_error(rolling_predict(matrix("a", 2, 2), rnorm(2)), "numeric matrix")
  expect_error(rolling_predict(X, rnorm(10)), "length nrow")
  expect_error(rolling_predict(X, y, window = 1), "between 2")
  expect_error(rolling_predict(X, y, window = 60), "between 2")
  expect_error(rolling_predict(X, y, lambda = -1), "non-negative")
  expect_error(rolling_predict(X, y, n_iter = 0), "positive integer")
})

# P3-fin: the resident device-context path must reproduce the exact ISTA
# numerical contract (rolling_predict / pure-R reference) on BOTH backends.
test_that("rolling_resident resident/cold match the pure-R ISTA reference", {
  set.seed(11)
  n <- 60L
  p <- 5L
  X <- matrix(rnorm(n * p), n, p)
  beta_true <- c(0.8, -0.5, 0, 0, 0.3)
  y <- drop(X %*% beta_true) + rnorm(n, sd = 0.5)
  window <- 30L
  lambda <- 0.05
  n_iter <- 120L

  ref <- rolling_ref(X, y, window, lambda, n_iter)
  f0 <- rolling_predict(X, y, window, lambda, n_iter)

  res <- rolling_resident(X, y, window, lambda, n_iter, strategy = "resident")
  cold <- rolling_resident(X, y, window, lambda, n_iter, strategy = "cold")

  expect_s3_class(res, "rolling_predict")
  expect_s3_class(res, "rolling_resident")
  expect_equal(res$strategy, "resident")
  expect_equal(cold$strategy, "cold")
  expect_equal(dim(res$coefficients), c(p, n - window))

  # Both strategies reproduce the pure-R oracle to the same tolerance as the
  # original F0 path.
  expect_equal(res$pred, ref$pred, tolerance = 1e-8)
  expect_equal(res$coefficients, ref$coefficients, tolerance = 1e-8)
  expect_equal(res$ynorm, ref$ynorm, tolerance = 1e-8)
  expect_equal(cold$pred, ref$pred, tolerance = 1e-8)
  expect_equal(cold$coefficients, ref$coefficients, tolerance = 1e-8)

  # resident, cold and the F0 one-shot path all use the SAME op sequence and
  # kernel order, so their coefficients are identical up to rounding.
  expect_equal(res$coefficients, f0$coefficients, tolerance = 1e-12)
  expect_equal(cold$coefficients, f0$coefficients, tolerance = 1e-12)
  expect_equal(res$pred, f0$pred, tolerance = 1e-12)

  expect_equal(res$iters, n_iter * (n - window))
  expect_equal(cold$iters, n_iter * (n - window))
  expect_equal(res$backend,
               if (rcupp::cuda_available()) "cuda" else "cpu-fallback")
  expect_equal(res$cuda, rcupp::cuda_available())
  expect_true(is.finite(res$mse) && res$mse >= 0)
  expect_true(is.finite(res$dir_acc) && res$dir_acc >= 0 && res$dir_acc <= 1)
})

test_that("rolling_resident validates strategy and inputs", {
  set.seed(3)
  X <- matrix(rnorm(40 * 3), 40, 3)
  y <- rnorm(40)
  # Partial matching resolves to the two documented strategies.
  expect_equal(rolling_resident(X, y, 20L, 0.1, 20L, "r")$strategy, "resident")
  expect_equal(rolling_resident(X, y, 20L, 0.1, 20L, "c")$strategy, "cold")
  expect_error(
    rolling_resident(X, y, 20L, 0.1, 20L, "bogus"), "should be one of"
  )
  # Input validation mirrors rolling_predict.
  expect_error(rolling_resident(X, rnorm(10)), "length nrow")
  expect_error(rolling_resident(X, y, window = 1), "between 2")
  expect_error(rolling_resident(X, y, window = 20L, lambda = -1),
               "non-negative")
  expect_error(rolling_resident(X, y, window = 20L, n_iter = 0),
               "positive integer")
})

test_that("rolling_resident resident == cold on both backends", {
  set.seed(21)
  X <- matrix(rnorm(50 * 6), 50, 6)
  y <- rnorm(50)
  res <- rolling_resident(X, y, 25L, 0.02, 60L, strategy = "resident")
  cold <- rolling_resident(X, y, 25L, 0.02, 60L, strategy = "cold")
  expect_equal(res$pred, cold$pred)
  expect_equal(res$coefficients, cold$coefficients)
  expect_equal(res$ynorm, cold$ynorm)
  expect_lt(max(abs(res$coefficients - cold$coefficients)), 1e-12)
})
