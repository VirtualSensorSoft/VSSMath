#ifndef VSS_MATH_LINEAR_LINEAR_SOLVE_HPP
#define VSS_MATH_LINEAR_LINEAR_SOLVE_HPP

#include <optional>
#include <utility>
#include <vss/math/linear/FactorizationOptions.hpp>

namespace vss::math::linear {

/// @brief Policy for a solve whose condition estimate meets or exceeds its
///        limit.
enum class IllConditionedSolvePolicy : std::uint8_t {
  /// @brief Return an error instead of a solution.
  Reject,
  /// @brief Return a solution together with its condition diagnostic.
  AllowWithDiagnostic
};

/// @brief Controls how a linear solve handles an ill-conditioned factorization.
struct SolveOptions final {
  /// @brief Action to take when the condition estimate meets or exceeds its
  ///        limit.
  IllConditionedSolvePolicy illConditionedPolicy{
      IllConditionedSolvePolicy::Reject};
};

namespace detail {

/// @brief Unsupported private construction access for `SolveOutcome`.
/// @details The definition is supplied by the non-installed source detail
///          header. It is declared here only so the public outcome can grant
///          narrowly scoped friendship without exposing a consumer factory.
template <class Solution> struct SolveOutcomeAccess;

} // namespace detail

/// @brief Owns a solve result and an optional condition diagnostic.
/// @tparam Solution Owning solution type returned by a linear algorithm.
/// @details An outcome has no meaningful empty state and therefore cannot be
///          default constructed. Copies are deep when `Solution` is deep-copy
///          able; moves are `noexcept` when the solution's move is `noexcept`.
template <class Solution> class SolveOutcome final {
public:
  /// @brief No empty outcome exists.
  SolveOutcome() = delete;

  /// @brief Copies the solution and optional report.
  SolveOutcome(const SolveOutcome&) = default;

  /// @brief Replaces this outcome with a copy.
  SolveOutcome& operator=(const SolveOutcome&) = default;

  /// @brief Transfers the solution and optional report.
  SolveOutcome(SolveOutcome&&) noexcept = default;

  /// @brief Replaces this outcome by moving another outcome.
  SolveOutcome& operator=(SolveOutcome&&) noexcept = default;

  /// @brief Destroys the owned solution and report.
  ~SolveOutcome() = default;

  /// @brief Returns the owned solution without copying it.
  /// @return A const reference valid while this outcome is alive and unchanged.
  [[nodiscard]] const Solution& solution() const noexcept { return solution_; }

  /// @brief Returns the optional condition diagnostic without copying it.
  /// @return A const reference to the optional report.
  [[nodiscard]] const std::optional<ConditionReport>&
  conditionReport() const noexcept {
    return conditionReport_;
  }

private:
  // NOLINTBEGIN(performance-move-const-arg): the report is intentionally
  // transferred through the exact by-value construction contract.
  SolveOutcome(Solution solution,
               std::optional<ConditionReport> conditionReport)
      : solution_(std::move(solution)),
        conditionReport_(std::move(conditionReport)) {}
  // NOLINTEND(performance-move-const-arg)

  template <class> friend struct detail::SolveOutcomeAccess;

  Solution solution_;
  std::optional<ConditionReport> conditionReport_;
};

} // namespace vss::math::linear

#endif // VSS_MATH_LINEAR_LINEAR_SOLVE_HPP
