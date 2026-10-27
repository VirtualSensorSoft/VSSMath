# Architecture

VirtualSensorMath is a compiled C++20 static library with a CMake package
export. Its public namespace is `vss::math`; the installed target is
`VirtualSensorSoft::math`.

## Components

- `Vector` and `Matrix` own contiguous `double` storage. Matrices use
  row-major layout.
- `VectorView`, `ConstVectorView`, `MatrixView`, and `ConstMatrixView` borrow
  caller-owned storage; their validity is bounded by that storage's lifetime.
- Core operations provide vector and matrix arithmetic.
- `linear` contains triangular solves, LU, pivoted QR, Cholesky, and symmetric
  eigendecomposition.
- `statistics`, `covariance`, `differentiation`, and `probability` provide
  descriptive statistics, covariance diagnostics/transforms, finite
  differences, and chi-square tail functions.
- `Result<T>` and `Status` represent operation failures without publishing a
  partial owning result. Some factorization results intentionally represent a
  completed but singular or rank-deficient diagnostic state.

Algorithms use standard-library storage and do not delegate numerical work to
an external runtime. Build and test dependencies are development tools only.

## Ownership and concurrency

Owning values copy their data and move ownership. Views never extend the
lifetime of their backing storage. Independent objects can be used
concurrently. Concurrent const access is supported where documented; any
write that aliases another access requires caller synchronization. There is no
mutable global numerical configuration or shared scratch cache.

## Build boundary

CMake exports one target and generated version header. Tests install into a
task-owned prefix to avoid collisions with other packages named
`VirtualSensorMath`. Doxygen output is generated during documentation builds;
it is not committed as source.
