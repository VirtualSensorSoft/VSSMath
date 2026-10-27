#ifndef VSS_MATH_DETAIL_SOLVE_OUTCOME_ACCESS_HPP
#define VSS_MATH_DETAIL_SOLVE_OUTCOME_ACCESS_HPP

#include <utility>
#include <vss/math/linear/LinearSolve.hpp>

namespace vss::math::linear::detail {

/// @brief Unsupported private construction access for algorithm translation
///        units and build-only contract tests.
/// @details This header is not installed or exported. Consumers must not use
///          this helper as a public factory.
template <class Solution> struct SolveOutcomeAccess final {
  /// @brief Constructs an outcome through its private constructor.
  /// @param solution Owning solution to transfer into the outcome.
  /// @param conditionReport Optional condition diagnostic to transfer.
  /// @return A completed solve outcome.
  // NOLINTBEGIN(performance-move-const-arg): the helper preserves the exact
  // by-value construction contract for the private outcome boundary.
  [[nodiscard]] static SolveOutcome<Solution>
  make(Solution solution, std::optional<ConditionReport> conditionReport) {
    return SolveOutcome<Solution>(std::move(solution),
                                  std::move(conditionReport));
  }
  // NOLINTEND(performance-move-const-arg)
};

} // namespace vss::math::linear::detail

#endif // VSS_MATH_DETAIL_SOLVE_OUTCOME_ACCESS_HPP
