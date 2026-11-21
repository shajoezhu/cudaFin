# cudaFin 0.0.0.9007

* Add `rolling_resident()`: a P3-fin resident rolling-window lasso path built
  on the `rcupp` Phase-0 device-context C ABI. ONE full `X`/`y` panel is
  uploaded once and each window is obtained by a device-side `gather_rows`,
  with the ISTA loop composed only from resident level-1 operators (no
  per-call host-to-device copies). `strategy = "resident"` vs `"cold"`
  isolates the transfer saving. It is numerically identical to
  `rolling_predict()` and the pure-R reference (`max |dcoef| ~ 1e-16`) on
  both the CUDA and CPU-fallback backends. No new `rcupp` operator is added
  and `rcupp_ctx_abi()` stays 1; the solver loop lives downstream in this
  package. See the `rolling-residency` vignette for the demo 3 measurement.

* Development version bumped from 0.0.0.9006.

# cudaFin 0.0.0.9006

* Development version bumped from 0.0.0.9005.

# cudaFin 0.0.0.9005

* Development version bumped from 0.0.0.9004.

# cudaFin 0.0.0.9004

* Initial development version.
