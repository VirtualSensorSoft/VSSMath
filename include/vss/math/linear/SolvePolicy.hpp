#ifndef VSS_MATH_LINEAR_SOLVE_POLICY_HPP
#define VSS_MATH_LINEAR_SOLVE_POLICY_HPP

#include <cstdint>
#include <optional>
#include <vss/math/Core.hpp>

namespace vss::math {

/// @brief Closed interval for algorithms accepting a search bracket.
struct Interval final {
  /// @brief Lower endpoint.
  Scalar lower{};
  /// @brief Upper endpoint.
  Scalar upper{};
};

/// @brief Iteration tolerances and work ceilings.
struct IterPolicy final {
  /// @brief Absolute stopping tolerance.
  Scalar abs{};
  /// @brief Relative stopping tolerance.
  Scalar rel{};
  /// @brief Maximum attempted iteration steps.
  Index max_iterations{};
  /// @brief Maximum attempted callback evaluations.
  Index max_evaluations{};
  /// @brief Optional bracket when supported by the operation.
  std::optional<Interval> bracket;
};

/// @brief Determines how a solve handles excluded numerical directions.
enum class RankPolicy : std::uint8_t {
  reject,             ///< Fail when a required direction is excluded.
  supported_subspace, ///< Solve over the admitted directions.
  minimum_norm        ///< Return a minimum-norm representative.
};

/// @brief Numerical rank, condition and iteration policy for additive solvers.
struct SolvePolicy final {
  /// @brief Iteration tolerances and work ceilings.
  IterPolicy iteration{};
  /// @brief Absolute cutoff in the units of the input spectrum.
  Scalar rank_abs{};
  /// @brief Relative cutoff with respect to the largest spectral magnitude.
  Scalar rank_rel{};
  /// @brief Lower bound for the reciprocal condition of a retained solve.
  Scalar reciprocal_condition_floor{};
  /// @brief Rule for excluded numerical directions.
  RankPolicy rank{};
  /// @brief Restart size for operations supporting restarts.
  Index restart{};
};

} // namespace vss::math
#endif
