#ifndef VSS_MATH_LINEAR_COMPLEX_HPP
#define VSS_MATH_LINEAR_COMPLEX_HPP

#include <complex>
#include <span>
#include <vss/math/Array.hpp>
#include <vss/math/Core.hpp>
#include <vss/math/Limits.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/Tolerance.hpp>
#include <vss/math/Vector.hpp>
#include <vss/math/linear/SolvePolicy.hpp>

namespace vss::math {

/// @brief Complex scalar with binary64 real and imaginary components.
using Complex = std::complex<Scalar>;

} // namespace vss::math

namespace vss::math::linear {

/// @brief Immutable borrowed matrix view with independent non-negative strides.
/// @details The span and its elements must remain alive and unchanged for the
///          duration of the call. The last reachable offset is checked before
///          access; zero strides are permitted for repeated read-only values.
struct ComplexMatrixView final {
  /// @brief Borrowed storage; each value is read-only for the call duration.
  std::span<const Complex> values;
  /// @brief Matrix row count.
  Extent rows;
  /// @brief Matrix column count.
  Extent columns;
  /// @brief Non-negative element offset between row starts.
  Extent row_stride;
  /// @brief Non-negative element offset between column starts.
  Extent column_stride;
};

/// @brief Owning row-major complex matrix result.
struct ComplexMatrix final {
  /// @brief Number of matrix rows.
  Extent rows;
  /// @brief Number of matrix columns.
  Extent columns;
  /// @brief Immutable owned row-major values; size is rows times columns.
  Array<Complex> values;
};

/// @brief Hermitian eigensystem with ascending real eigenvalues.
struct ComplexEigen final {
  /// @brief Real eigenvalues in ascending order.
  Vector values;
  /// @brief Corresponding orthonormal eigenvectors stored by columns.
  ComplexMatrix vectors;
  /// @brief Safely evaluated residual norm for each eigenpair.
  Vector residual_norms;
};

/// @brief Thin complex singular value decomposition.
struct ComplexSVD final {
  /// @brief Left singular vectors, shape m by min(m,n).
  ComplexMatrix left;
  /// @brief Nonnegative singular values in nonincreasing order.
  Vector singular;
  /// @brief Adjoint of the right singular-vector matrix, shape min(m,n) by n.
  ComplexMatrix right_adjoint;
  /// @brief Count of singular values strictly above the policy cutoff.
  Extent rank;
};

/// @brief Computes the dense complex matrix product without conjugation.
[[nodiscard]] Result<ComplexMatrix>
multiply(ComplexMatrixView a, ComplexMatrixView b, Limits limits);

/// @brief Returns an owned conjugate transpose.
[[nodiscard]] Result<ComplexMatrix> conjugateTranspose(ComplexMatrixView matrix,
                                                       Limits limits);

/// @brief Solves a square complex system for one or more right-hand sides.
[[nodiscard]] Result<ComplexMatrix> solveComplex(ComplexMatrixView a,
                                                 ComplexMatrixView b,
                                                 SolvePolicy policy,
                                                 Limits limits);

/// @brief Computes a Hermitian eigendecomposition using the default admission
///        tolerance of zero absolute and 64 binary64 epsilons relative.
[[nodiscard]] Result<ComplexEigen>
eigenHermitian(ComplexMatrixView matrix, SolvePolicy policy, Limits limits);

/// @brief Computes a Hermitian eigendecomposition with explicit admission
///        tolerance; solver residual tolerances remain in SolvePolicy.
[[nodiscard]] Result<ComplexEigen> eigenHermitian(ComplexMatrixView matrix,
                                                  Tolerance hermitian_tolerance,
                                                  SolvePolicy policy,
                                                  Limits limits);

/// @brief Computes the thin complex singular value decomposition.
[[nodiscard]] Result<ComplexSVD>
factorComplexSVD(ComplexMatrixView matrix, SolvePolicy policy, Limits limits);

} // namespace vss::math::linear

#endif // VSS_MATH_LINEAR_COMPLEX_HPP
