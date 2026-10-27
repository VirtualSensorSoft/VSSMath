#ifndef VSS_MATH_DETAIL_SOLVE_VALIDATION_HPP
#define VSS_MATH_DETAIL_SOLVE_VALIDATION_HPP

#include <vss/math/Error.hpp>
#include <vss/math/linear/LinearSolve.hpp>

namespace vss::math::linear::detail {

inline Status validateSolveOptions(const SolveOptions& options) {
  switch (options.illConditionedPolicy) {
  case IllConditionedSolvePolicy::Reject:
  case IllConditionedSolvePolicy::AllowWithDiagnostic:
    return Status::success();
  }
  return Status::failure(ErrorCode::InvalidDomain,
                         "solve policy enumerator is invalid");
}

} // namespace vss::math::linear::detail

#endif // VSS_MATH_DETAIL_SOLVE_VALIDATION_HPP
