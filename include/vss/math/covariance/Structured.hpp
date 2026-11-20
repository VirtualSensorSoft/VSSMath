#ifndef VSS_MATH_COVARIANCE_STRUCTURED_HPP
#define VSS_MATH_COVARIANCE_STRUCTURED_HPP

#include <optional>
#include <span>
#include <vss/math/Array.hpp>
#include <vss/math/Core.hpp>
#include <vss/math/Limits.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/MatrixView.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/Vector.hpp>
#include <vss/math/VectorView.hpp>
#include <vss/math/linear/RankAware.hpp>
#include <vss/math/linear/SolvePolicy.hpp>

namespace vss::math::covariance {

/// @brief Storage representation of a real symmetric covariance operator.
enum class StructureKind : std::uint8_t {
  diagonal_low_rank, ///< Nonnegative diagonal plus a factor times its
                     ///< transpose.

  toeplitz, ///< Symmetric matrix described by its first row.

  block, ///< Direct sum of square covariance factors.

  kronecker, ///< Kronecker product with the rightmost factor varying fastest.

  gaussian_kernel ///< Gaussian covariance evaluated from rows of point
                  ///< coordinates.
};
/// @brief Explicit algorithm selection for structured covariance solves.
enum class StructuredSolveKind : std::uint8_t {
  factorized, ///< Compute the retained spectral solution.
  iterative ///< Use Jacobi preconditioned CG with a full-rank SPD certificate.
};

/// @brief Borrowed inputs selecting exactly one structured PSD representation.
struct StructuredSpec final {
  /// @brief Selected representation.
  StructureKind kind;
  /// @brief Borrowed diagonal entries or Toeplitz lags.
  ConstVectorView diagonal_or_lags;
  /// @brief Borrowed update factor or Gaussian coordinates, where applicable.
  std::optional<ConstMatrixView> low_rank_or_points;
  /// @brief Borrowed direct-sum or Kronecker factors, in caller order.
  std::span<const Matrix> factors;
  /// @brief Gaussian marginal variance, present only for a kernel.
  std::optional<Scalar> kernel_scale;
  /// @brief Positive Gaussian length scale, present only for a kernel.
  std::optional<Scalar> kernel_length;
};

/// @brief Owned structured covariance; the represented dense matrix is not
/// kept.
struct StructuredCovariance final {
  /// @brief Selected representation.
  StructureKind kind;
  /// @brief Order of the represented square operator.
  Index dimension;
  /// @brief Owned diagonal entries or Toeplitz lags.
  Vector diagonal_or_lags;
  /// @brief Owned update factor or Gaussian point coordinates.
  std::optional<Matrix> low_rank_or_points;
  /// @brief Owned direct-sum or Kronecker factors, in caller order.
  Array<Matrix> factors;
  /// @brief Gaussian marginal variance, present only for a kernel.
  std::optional<Scalar> kernel_scale;
  /// @brief Positive Gaussian length scale, present only for a kernel.
  std::optional<Scalar> kernel_length;
};

/// @brief Validates a representation and copies its active payload.
/// @details Inputs are borrowed only during the call. Inactive fields must be
/// empty. Construction checks symmetry and positive semidefiniteness where
/// these do not follow from the representation.
[[nodiscard]] Result<StructuredCovariance>
structuredCovariance(const StructuredSpec& spec, SolvePolicy policy,
                     Limits limits);
/// @brief Constructs a covariance from square PSD Kronecker factors.
/// @details An empty sequence represents the scalar identity.
[[nodiscard]] Result<StructuredCovariance>
kroneckerCovariance(std::span<const Matrix> psd_factors, SolvePolicy policy,
                    Limits limits);
/// @brief Solves a structured covariance system for each borrowed RHS column.
/// @details Factorized mode uses the retained spectrum and reports residuals
/// in original coordinates. Iterative mode requires a sufficient full-rank
/// certificate and never switches to factorized mode. Both obey work limits.
[[nodiscard]] Result<linear::LinearEstimate>
solveStructuredCovariance(const StructuredCovariance& covariance,
                          ConstMatrixView rhs, StructuredSolveKind mode,
                          SolvePolicy policy, Limits limits);
/// @brief Applies the representation to a borrowed vector without dense
/// expansion.
/// @details Revalidates payload shape and finiteness. Spectral validation is
/// performed during construction. The returned vector owns its storage.
[[nodiscard]] Result<Vector>
applyStructuredCovariance(const StructuredCovariance& covariance,
                          ConstVectorView value, Limits limits);

} // namespace vss::math::covariance
#endif
