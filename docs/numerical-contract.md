# Numerical contract

This document describes the behavior the implementation and tests are intended
to preserve. It does not claim universal correctness or cross-platform
bitwise-identical results.

## Scalar, storage, and shapes

The scalar is `double`. The planned initial profile requires an 8-byte
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
- Cholesky validates symmetry and positive definiteness; it does not repair an
  input by adding jitter.
- Symmetric eigendecomposition orders eigenvalues and applies documented
  deterministic conventions for isolated eigenvectors; clustered eigenspaces
  are compared as subspaces.
- Covariance operations distinguish positive-definite, semidefinite, and
  non-semidefinite results, and do not silently repair input.
- Finite-difference operations use deterministic formulas and callback order;
  invalid steps or callback results fail without publishing partial output.
- Descriptive statistics and chi-square tail functions follow explicit domain,
  finite-value, and accuracy checks.

Exact APIs and error codes are defined by the public headers. Sparse matrices,
other scalar types, implicit units, application workflows, automatic repair,
and unsupported platforms are not part of the current contract.
