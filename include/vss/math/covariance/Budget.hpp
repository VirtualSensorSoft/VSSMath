#ifndef VSS_MATH_COVARIANCE_BUDGET_HPP
#define VSS_MATH_COVARIANCE_BUDGET_HPP

#include <cstdint>
#include <span>
#include <vss/math/Array.hpp>
#include <vss/math/Limits.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/MatrixView.hpp>
#include <vss/math/Result.hpp>

namespace vss::math {

/// @brief Specifies whether source cross-covariances are complete or declared
///        absent pairs are independent.
enum class DependencePolicy : std::uint8_t {
  /// @brief Require an explicit cross block for every unordered source pair.
  exact_joint,
  /// @brief Treat absent pairs as independent and require supplied blocks zero.
  declared_independent_blocks
};

/// @brief One input covariance block mapped from listed Jacobian coordinates.
struct SourceBlock final {
  /// @brief Stable nonzero identifier for this covariance source.
  std::uint64_t id;
  /// @brief Disjoint input-coordinate positions owned by this source.
  std::span<const Index> coordinates;
  /// @brief Covariance in the order listed by @p coordinates.
  ConstMatrixView covariance;
};

/// @brief Cross-covariance ordered by the declared left and right source IDs.
struct SourceCrossBlock final {
  /// @brief Identifier of the source supplying the row coordinates.
  std::uint64_t left;
  /// @brief Identifier of the source supplying the column coordinates.
  std::uint64_t right;
  /// @brief Cross-covariance with rows in left and columns in right order.
  ConstMatrixView covariance;
};

/// @brief Signed covariance contribution for one canonical source pair.
struct BudgetEntry final {
  /// @brief Lower numeric source identifier in this canonical pair.
  std::uint64_t left;
  /// @brief Higher numeric source identifier, or equal to @p left for a
  ///        diagonal contribution.
  std::uint64_t right;
  /// @brief Signed propagated contribution, including both orientations for a
  ///        cross-source pair.
  Matrix contribution;
};

/// @brief Total propagated covariance and its diagonal/cross-source terms.
struct UncertaintyBudget final {
  /// @brief Sum of all entries in @p terms.
  Matrix covariance;
  /// @brief Source-pair contributions sorted by canonical numeric identifiers.
  Array<BudgetEntry> terms;
};

namespace covariance {

/// @brief Decomposes a propagated covariance into source-pair contributions.
/// @param jacobian Map from the complete ordered input coordinates to outputs.
/// @param sources Unique source IDs with disjoint coordinates partitioning the
///        Jacobian columns and their local covariance blocks.
/// @param cross Covariance blocks in the coordinate order declared by each ID.
/// @param dependence Explicit completeness or independence policy.
/// @param limits Maximum result and operation workspace storage.
/// @return Owned total covariance and all signed diagonal/cross contributions,
///         sorted by numeric source IDs.
/// @retval ErrorCode::InvalidDomain Invalid partition, duplicate pair, missing
///         exact-joint pair, or a dependency under the independent policy.
/// @retval ErrorCode::DimensionMismatch A covariance block has the wrong shape.
/// @retval ErrorCode::NotPositiveSemidefinite The assembled joint covariance is
///         materially indefinite.
/// @retval ErrorCode::ResourceLimit Checked storage exceeds either ceiling.
[[nodiscard]] Result<UncertaintyBudget>
uncertaintyContributionBudget(ConstMatrixView jacobian,
                              std::span<const SourceBlock> sources,
                              std::span<const SourceCrossBlock> cross,
                              DependencePolicy dependence, Limits limits);

} // namespace covariance
} // namespace vss::math

#endif // VSS_MATH_COVARIANCE_BUDGET_HPP
