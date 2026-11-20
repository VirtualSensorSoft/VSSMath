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

enum class StructureKind : std::uint8_t {
  diagonal_low_rank,
  toeplitz,
  block,
  kronecker,
  gaussian_kernel
};
enum class StructuredSolveKind : std::uint8_t { factorized, iterative };

/// @brief Borrowed inputs selecting exactly one structured PSD representation.
struct StructuredSpec final {
  StructureKind kind;
  ConstVectorView diagonal_or_lags;
  std::optional<ConstMatrixView> low_rank_or_points;
  std::span<const Matrix> factors;
  std::optional<Scalar> kernel_scale;
  std::optional<Scalar> kernel_length;
};

/// @brief Owned structured covariance; the represented dense matrix is not
/// kept.
struct StructuredCovariance final {
  StructureKind kind;
  Index dimension;
  Vector diagonal_or_lags;
  std::optional<Matrix> low_rank_or_points;
  Array<Matrix> factors;
  std::optional<Scalar> kernel_scale;
  std::optional<Scalar> kernel_length;
};

[[nodiscard]] Result<StructuredCovariance>
structuredCovariance(const StructuredSpec& spec, SolvePolicy policy,
                     Limits limits);
[[nodiscard]] Result<StructuredCovariance>
kroneckerCovariance(std::span<const Matrix> psd_factors, SolvePolicy policy,
                    Limits limits);
[[nodiscard]] Result<linear::LinearEstimate>
solveStructuredCovariance(const StructuredCovariance& covariance,
                          ConstMatrixView rhs, StructuredSolveKind mode,
                          SolvePolicy policy, Limits limits);
[[nodiscard]] Result<Vector>
applyStructuredCovariance(const StructuredCovariance& covariance,
                          ConstVectorView value, Limits limits);

} // namespace vss::math::covariance
#endif
