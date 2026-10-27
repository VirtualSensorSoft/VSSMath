#ifndef VSS_MATH_LINEAR_SYMMETRIC_EIGEN_HPP
#define VSS_MATH_LINEAR_SYMMETRIC_EIGEN_HPP

#include <cstdint>
#include <vss/math/Matrix.hpp>
#include <vss/math/MatrixView.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/Tolerance.hpp>
#include <vss/math/Vector.hpp>
#include <vss/math/VectorView.hpp>
#include <vss/math/linear/FactorizationOptions.hpp>

namespace vss::math::linear {

/// @brief Classifies a symmetric matrix using its computed eigenvalues.
enum class PositiveDefiniteness : std::uint8_t {
  /// @brief Every eigenvalue is strictly above the rank threshold.
  PositiveDefinite,
  /// @brief Every eigenvalue is at least the negative rank threshold.
  PositiveSemidefinite,
  /// @brief At least one eigenvalue is below the negative rank threshold.
  NotPositiveSemidefinite
};

/// @brief Policies for symmetric eigendecomposition.
/// @details The tolerances are passive values. They are validated by
///          `decomposeSymmetricEigen()` before any matrix inspection.
struct SymmetricEigenOptions final {
  /// @brief Absolute and relative threshold used for rank classification.
  RankTolerance rankTolerance{};
  /// @brief Tolerance used to validate mirrored matrix entries.
  Tolerance symmetryTolerance{.absolute = 0.0, .relative = 1.0e-12};
  /// @brief Maximum number of complete cyclic Jacobi sweeps.
  Extent maximumSweeps{64U};
};

/// @brief Owns an ordered eigendecomposition of a real symmetric matrix.
/// @details The successful result represents `A = Q D Q^T`, where `D` is
///          ascending and `Q` is orthonormal. Input is copied and never
///          modified; the lower triangle is authoritative after mirrored
///          entries pass the configured symmetry check. Repeated eigenspaces
///          are represented by a stable orthonormal basis, and callers should
///          compare their projectors rather than individual basis vectors.
///          The cyclic Jacobi algorithm uses O(n^3) time per sweep and O(n^2)
///          owned storage for an n by n matrix.
class SymmetricEigenDecomposition final {
public:
  SymmetricEigenDecomposition() = delete;

  /// @brief Performs a deep copy of the eigendecomposition.
  SymmetricEigenDecomposition(const SymmetricEigenDecomposition&) = default;

  /// @brief Replaces this object with a deep copy.
  SymmetricEigenDecomposition&
  operator=(const SymmetricEigenDecomposition&) = default;

  /// @brief Transfers storage and resets the source to an empty state.
  SymmetricEigenDecomposition(SymmetricEigenDecomposition&& other) noexcept;

  /// @brief Replaces this object by transferring storage.
  SymmetricEigenDecomposition&
  operator=(SymmetricEigenDecomposition&& other) noexcept;

  ~SymmetricEigenDecomposition() = default;

  /// @brief Returns the represented matrix extent.
  [[nodiscard]] Extent size() const noexcept;

  /// @brief Returns the number of eigenvalues strictly above the rank
  ///        threshold.
  [[nodiscard]] Extent rank() const noexcept;

  /// @brief Returns the strict threshold used for rank decisions.
  [[nodiscard]] Scalar rankThreshold() const noexcept;

  /// @brief Returns the ascending eigenvalue vector as a borrowed view.
  [[nodiscard]] ConstVectorView eigenvalues() const noexcept;

  /// @brief Returns the orthonormal eigenvector columns as a borrowed view.
  [[nodiscard]] ConstMatrixView eigenvectors() const noexcept;

  /// @brief Returns the smallest ordered eigenvalue, or zero when moved-from.
  [[nodiscard]] Scalar minimumEigenvalue() const noexcept;

  /// @brief Returns the largest ordered eigenvalue, or zero when moved-from.
  [[nodiscard]] Scalar maximumEigenvalue() const noexcept;

  /// @brief Returns the positive-definiteness classification.
  [[nodiscard]] PositiveDefiniteness definiteness() const noexcept;

  /// @brief Returns the number of complete cyclic sweeps performed.
  [[nodiscard]] Extent completedSweeps() const noexcept;

  /// @brief Returns the off-diagonal convergence threshold.
  [[nodiscard]] Scalar convergenceThreshold() const noexcept;

  /// @brief Returns the final maximum absolute off-diagonal entry.
  [[nodiscard]] Scalar finalMaximumOffDiagonal() const noexcept;

private:
  friend Result<SymmetricEigenDecomposition>
  decomposeSymmetricEigen(ConstMatrixView matrix,
                          SymmetricEigenOptions options);

  SymmetricEigenDecomposition(Extent size, Extent rank, Scalar rankThreshold,
                              Vector eigenvalues, Matrix eigenvectors,
                              PositiveDefiniteness definiteness,
                              Extent completedSweeps,
                              Scalar convergenceThreshold,
                              Scalar finalMaximumOffDiagonal) noexcept;

  void resetMovedFrom() noexcept;

  Extent size_;
  Extent rank_;
  Scalar rankThreshold_;
  Vector eigenvalues_;
  Matrix eigenvectors_;
  PositiveDefiniteness definiteness_;
  Extent completedSweeps_;
  Scalar convergenceThreshold_;
  Scalar finalMaximumOffDiagonal_;
};

/// @brief Computes an ordered eigendecomposition of a non-empty symmetric
///        matrix.
/// @param matrix Borrowed square input; mirrored entries are checked first.
/// @param options Rank, symmetry, and convergence policies.
/// @return A complete eigendecomposition or a validation/numerical failure.
/// @note The lower triangle is authoritative after symmetry validation.
///       Complexity is O(n^3 * maximumSweeps) in the worst case and O(n^2)
///       owned storage.
[[nodiscard]] Result<SymmetricEigenDecomposition>
decomposeSymmetricEigen(ConstMatrixView matrix,
                        SymmetricEigenOptions options = {});

} // namespace vss::math::linear

#endif // VSS_MATH_LINEAR_SYMMETRIC_EIGEN_HPP
