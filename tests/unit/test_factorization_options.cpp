#include "Test.hpp"

#include <cstdint>
#include <type_traits>
#include <vss/math/Core.hpp>
#include <vss/math/Tolerance.hpp>
#include <vss/math/linear/FactorizationOptions.hpp>
#include <vss/math/linear/LinearSolve.hpp>

namespace {

using vss::math::Scalar;
using vss::math::Tolerance;
using vss::math::linear::ConditionClassification;
using vss::math::linear::FactorizationOptions;
using vss::math::linear::IllConditionedSolvePolicy;
using vss::math::linear::RankTolerance;
using vss::math::linear::SolveOptions;
using vss::math::linear::SymmetricFactorizationOptions;

static_assert(std::is_same_v<
              std::remove_cvref_t<decltype(RankTolerance{}.absolute)>, Scalar>);
static_assert(std::is_same_v<
              std::remove_cvref_t<decltype(RankTolerance{}.relative)>, Scalar>);
static_assert(
    std::is_same_v<
        std::remove_cvref_t<decltype(FactorizationOptions{}.rankTolerance)>,
        RankTolerance>);
static_assert(
    std::is_same_v<
        std::remove_cvref_t<decltype(FactorizationOptions{}.conditionLimit)>,
        Scalar>);
static_assert(
    std::is_same_v<std::remove_cvref_t<
                       decltype(SymmetricFactorizationOptions{}.factorization)>,
                   FactorizationOptions>);
static_assert(
    std::is_same_v<std::remove_cvref_t<decltype(SymmetricFactorizationOptions{}
                                                    .symmetryTolerance)>,
                   Tolerance>);
static_assert(
    std::is_same_v<
        std::remove_cvref_t<decltype(SolveOptions{}.illConditionedPolicy)>,
        IllConditionedSolvePolicy>);

} // namespace

int main() {
  const RankTolerance rankTolerance{};
  VSS_CHECK(rankTolerance.absolute == 0.0);
  VSS_CHECK(rankTolerance.relative == 1.0e-12);

  const FactorizationOptions factorization{};
  VSS_CHECK(factorization.rankTolerance.absolute == 0.0);
  VSS_CHECK(factorization.rankTolerance.relative == 1.0e-12);
  VSS_CHECK(factorization.conditionLimit == 1.0e10);

  const SymmetricFactorizationOptions symmetric{};
  VSS_CHECK(symmetric.factorization.rankTolerance.absolute == 0.0);
  VSS_CHECK(symmetric.factorization.rankTolerance.relative == 1.0e-12);
  VSS_CHECK(symmetric.factorization.conditionLimit == 1.0e10);
  VSS_CHECK(symmetric.symmetryTolerance.absolute == 0.0);
  VSS_CHECK(symmetric.symmetryTolerance.relative == 1.0e-12);

  const SolveOptions solve{};
  VSS_CHECK(solve.illConditionedPolicy == IllConditionedSolvePolicy::Reject);
  VSS_CHECK(static_cast<std::uint8_t>(ConditionClassification::WithinLimit) ==
            0U);
  VSS_CHECK(
      static_cast<std::uint8_t>(ConditionClassification::IllConditioned) == 1U);
}
