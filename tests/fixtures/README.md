# Numerical test-case identifiers

The identifiers below name logical cases in the C++ tests; they do not imply a separate downloaded or generated data file. Expected values use analytic constructions, exact integer/fraction cases where practical, and fixed constants reviewed independently of the implementation. Property tests use deterministic seeds recorded in their source. Tests do not fetch a calculator or dataset at run time.

## Stable family cases

- `TRI-01`–`TRI-05`: triangular solves, orientation, diagonal policy, multiple right-hand sides, and validation boundaries.
- `LU-01`–`LU-05`: pivoted factorization, solve, determinant diagnostics, singularity, and conditioning.
- `QR-01`–`QR-06`: reconstruction, least squares, permutation, rank, conditioning and shape failures.
- `CHO-01`–`CHO-06`: positive-definite factorization, solves, whitening, log determinant, and invalid inputs.
- `EIG-01`–`EIG-06`: ordered symmetric eigenvalues, invariant subspaces, rank and definiteness boundaries.
- `COV-01`–`COV-08`: covariance analysis, correlation conversion, propagation and diagnostic outcomes.
- `DIF-01`–`DIF-07`: gradient/Jacobian formulas, bounds, callback order and failure handling.
- `STA-01`–`STA-06`: descriptive statistics, sample-size distinctions, finite input and variance boundaries.
- `PRO-01`–`PRO-07`: chi-square survival and log-survival domains, accuracy and extreme-tail behavior.

## Supplemental boundaries

- `SUP-TRI-01`: equality at a triangular pivot threshold.
- `SUP-LU-01`: deterministic pivot tie.
- `SUP-QR-01`–`SUP-QR-05`: norm tie, non-negative diagonal convention, zero trailing norm, maximum finite scale and rank-threshold equality.
- `SUP-EIG-01`–`SUP-EIG-02`: near-maximum finite rotation and sweep-limit failure.
- `SUP-DIF-01`: representable adjacent-value step at a subnormal scale.
- `SUP-STA-01`–`SUP-STA-02`: roundoff clamp boundary and overflow-safe large variance.
- `SUP-PRO-01`: probability-range clamp boundary.

The C++ test sources are authoritative for each case's inputs and assertions. The requirement map connects public `NUM-*` contracts to these identifiers and the actual CTest names. A case name or passing test is evidence only for the asserted behavior, not universal numerical correctness.
