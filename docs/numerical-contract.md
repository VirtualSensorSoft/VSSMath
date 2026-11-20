# Numerical contract

This document describes the behavior the implementation and tests are intended
to preserve. It does not claim universal correctness or cross-platform
bitwise-identical results.

## Scalar, storage, and shapes

The scalar is `double`. The assessed initial profile requires an 8-byte
IEC 60559 binary floating-point representation with radix 2. Vectors own a
contiguous sequence; matrices own row-major elements. Extents and indices use
`std::size_t`; public operations check required shape and extent arithmetic
before accessing or allocating where their result contract permits a failure.
Zero-area matrix shapes preserve their row and column extents.

## Ownership and views

Owning containers do not borrow constructor input. Views are non-owning and
become invalid when their source is destroyed, moved, or resized. Callers must
keep view storage valid for each operation and must synchronize writes that
could alias concurrent access.

## Validation and failures

Algorithms validate options, dimensions, and values in documented order.
Non-finite inputs are rejected when the relevant operation policy requires
finite values; containers themselves may store them. Failed `Result<T>` values
contain no partial result. Completed factorizations can carry singular or
rank-deficient diagnostics where specified; solves reject unusable factors
unless a documented option allows a value with an explicit condition report.

## Allocation and exceptions

Expected input and numerical failures use the documented `Result<T>` status
contract. Allocation failures and container size failures may propagate as
standard C++ exceptions (including `std::bad_alloc` and `std::length_error`);
they are not converted into numerical error codes. Exceptions thrown by a
user-supplied finite-difference callback propagate unchanged. If an exception
escapes an operation, no partial owning result is returned; input views passed
to numerical operations are not modified by the library.

## Tolerances and accuracy

Tolerance is operation-specific and scale-aware. Rank, condition, symmetry,
residual, and probability boundaries are distinct contracts. A tolerance-based
equality is not a claim of bitwise equality. Tests use analytic cases,
fixed-seed properties, boundary cases, and independently reviewed constants.
Fixture provenance and expected bounds are documented alongside the test map.

## Algorithm families

- Triangular solves support lower/upper systems and explicit/unit diagonals.
- LU uses deterministic partial row pivoting and reports singularity and
  condition diagnostics.
- QR uses deterministic column-pivoted Householder factorization and provides
  least-squares solutions only when the documented rank conditions hold.
- Full QR also supports original-column order or deterministic optional
  pivoting for tall, square, and wide matrices. It returns an owning square
  orthogonal factor and upper-trapezoidal factor; numerical rank is certified
  from the shared SVD cutoff rather than inferred from the triangular diagonal.
- Cholesky validates symmetry and positive definiteness; it does not repair an
  input by adding jitter.
- Symmetric eigendecomposition orders eigenvalues and applies documented
  deterministic conventions for isolated eigenvectors; clustered eigenspaces
  are compared as subspaces.
- Covariance operations distinguish positive-definite, semidefinite, and
  non-semidefinite results, and do not silently repair input.
- Schur complements use a bounded symmetric spectrum and the selected rank
  policy to apply the signed pseudoinverse on retained directions. Conditional
  covariance first validates the complete joint covariance as positive
  semidefinite, preserves requested coordinate order, and reports the
  conditional spectrum without projecting its result.
- Covariance whitening constructs the symmetric inverse square root on the
  retained positive support. It reports the discarded-space residual and
  rejects vectors outside that support; Mahalanobis distance does not silently
  project an incompatible residual.
- Truncated low-rank approximation retains the requested leading SVD modes and
  returns factors, reconstruction, and the Frobenius residual. PCA treats rows
  as observations, centers each column, reports sample explained variance, and
  whitens only retained components with positive sample variance.
- Known-covariance generalized least squares whitens admitted stochastic
  observation directions and enforces excluded covariance directions as exact
  constraints. It returns parameter covariance only when every parameter
  direction is identified; otherwise the covariance is disengaged and the
  identified subspace is explicit.
- Covariance diagnostics validate the complete joint matrix before extracting
  ordered marginals or constructing the supported symmetric inverse square
  root. Cross-covariance transforms preserve their rectangular source ordering
  and evaluate the full two-map contraction without requiring a PSD test on the
  cross block.
- Determinants use scaled pivoted LU with numerical bounds and exact dyadic
  elimination when those bounds cannot establish the sign, singularity, or
  output range. Log magnitudes are evaluated from the scaled result, including
  for nonzero determinants that overflow or underflow as scalars.
- Finite-difference operations use deterministic formulas and callback order;
  invalid steps or callback results fail without publishing partial output.
- Descriptive statistics and chi-square tail functions follow explicit domain,
  finite-value, and accuracy checks.

Exact APIs and error codes are defined by the public headers. Sparse matrices,
other scalar types, implicit units, application workflows, automatic repair,
and unsupported platforms are not part of the current contract.
