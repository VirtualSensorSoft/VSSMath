#include "Test.hpp"

#include <array>
#include <cmath>
#include <initializer_list>
#include <limits>
#include <vss/math/Error.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/covariance/Diagnostics.hpp>
#include <vss/math/linear/SolvePolicy.hpp>

namespace {

using namespace vss::math;

Matrix matrix(std::initializer_list<std::initializer_list<Scalar>> rows) {
  const Index rowCount = rows.size();
  const Index columnCount = rowCount == 0U ? 0U : rows.begin()->size();
  Matrix result(rowCount, columnCount, 0.0);
  Index row = 0U;
  for (const auto& values : rows) {
    VSS_CHECK(values.size() == columnCount);
    Index column = 0U;
    for (const Scalar value : values) {
      result.at(row, column++) = value;
    }
    ++row;
  }
  return result;
}

SolvePolicy policy() {
  SolvePolicy result{};
  result.iteration.max_iterations = 128U;
  result.rank = RankPolicy::supported_subspace;
  return result;
}

Limits limits() { return {4096U, 262144U, 0U}; }

void marginalCovariance() {
  const Matrix joint = matrix({{4.0, 3.0}, {3.0, 9.0}});
  const std::array<Index, 1U> second{1U};
  const auto marginal =
      covariance::marginalCovariance(joint.view(), second, policy(), limits());
  VSS_CHECK(marginal.hasValue());
  VSS_CHECK(marginal.value().rows() == 1U && marginal.value().columns() == 1U);
  VSS_CHECK(marginal.value().at(0U, 0U) == 9.0);

  const std::array<Index, 2U> reverse{1U, 0U};
  const auto ordered =
      covariance::marginalCovariance(joint.view(), reverse, policy(), limits());
  VSS_CHECK(ordered.hasValue());
  VSS_CHECK(ordered.value().at(0U, 0U) == 9.0);
  VSS_CHECK(ordered.value().at(0U, 1U) == 3.0);
  VSS_CHECK(ordered.value().at(1U, 0U) == 3.0);
  VSS_CHECK(ordered.value().at(1U, 1U) == 4.0);

  SolvePolicy strict = policy();
  strict.rank = RankPolicy::reject;
  const Matrix singular = matrix({{1.0, 1.0}, {1.0, 1.0}});
  const auto singularMarginal =
      covariance::marginalCovariance(singular.view(), second, strict, limits());
  VSS_CHECK(singularMarginal.hasValue());

  const auto empty = covariance::marginalCovariance(
      joint.view(), std::span<const Index>{}, policy(), limits());
  VSS_CHECK(!empty.hasValue());
  VSS_CHECK(empty.status().code() == ErrorCode::InsufficientSamples);
  const std::array<Index, 2U> duplicate{0U, 0U};
  const auto repeated = covariance::marginalCovariance(joint.view(), duplicate,
                                                       policy(), limits());
  VSS_CHECK(!repeated.hasValue());
  VSS_CHECK(repeated.status().code() == ErrorCode::InvalidDomain);
  const std::array<Index, 1U> outOfRange{2U};
  const auto invalidIndex = covariance::marginalCovariance(
      joint.view(), outOfRange, policy(), limits());
  VSS_CHECK(!invalidIndex.hasValue());
  VSS_CHECK(invalidIndex.status().code() == ErrorCode::IndexOutOfRange);

  const Matrix indefinite = matrix({{1.0, 2.0}, {2.0, 1.0}});
  const auto invalidJoint = covariance::marginalCovariance(
      indefinite.view(), second, policy(), limits());
  VSS_CHECK(!invalidJoint.hasValue());
  VSS_CHECK(invalidJoint.status().code() == ErrorCode::NotPositiveSemidefinite);
  const auto denied = covariance::marginalCovariance(joint.view(), second,
                                                     policy(), {0U, 0U, 0U});
  VSS_CHECK(!denied.hasValue());
  VSS_CHECK(denied.status().code() == ErrorCode::ResourceLimit);
}

void inverseSquareRoot() {
  const Matrix diagonal = matrix({{4.0, 0.0}, {0.0, 9.0}});
  const auto root =
      covariance::inverseSquareRoot(diagonal.view(), policy(), limits());
  VSS_CHECK(root.hasValue());
  VSS_CHECK(root.value().classification ==
            covariance::CovarianceKind::positive_definite);
  VSS_CHECK(root.value().rank == 2U);
  VSS_CHECK(root.value().reciprocal_condition.has_value());
  VSS_CHECK(std::abs(root.value().covariance.at(0U, 0U) - 0.5) < 1.0e-12);
  VSS_CHECK(std::abs(root.value().covariance.at(1U, 1U) - 1.0 / 3.0) < 1.0e-12);
  VSS_CHECK(std::abs(*root.value().reciprocal_condition - 4.0 / 9.0) < 1.0e-12);

  const Matrix rankOne = matrix({{2.0, 2.0}, {2.0, 2.0}});
  const auto supportedRoot =
      covariance::inverseSquareRoot(rankOne.view(), policy(), limits());
  VSS_CHECK(supportedRoot.hasValue());
  VSS_CHECK(supportedRoot.value().rank == 1U);
  VSS_CHECK(supportedRoot.value().classification ==
            covariance::CovarianceKind::semidefinite_singular);
  VSS_CHECK(!supportedRoot.value().reciprocal_condition.has_value());
  for (Index row = 0U; row < 2U; ++row) {
    for (Index column = 0U; column < 2U; ++column) {
      VSS_CHECK(std::abs(supportedRoot.value().covariance.at(row, column) -
                         0.25) < 1.0e-12);
    }
  }

  SolvePolicy strict = policy();
  strict.rank = RankPolicy::reject;
  const auto rejected =
      covariance::inverseSquareRoot(rankOne.view(), strict, limits());
  VSS_CHECK(!rejected.hasValue());
  VSS_CHECK(rejected.status().code() == ErrorCode::RankDeficient);

  const Matrix zero(2U, 2U, 0.0);
  const auto zeroRoot =
      covariance::inverseSquareRoot(zero.view(), policy(), limits());
  VSS_CHECK(zeroRoot.hasValue());
  VSS_CHECK(zeroRoot.value().rank == 0U);
  VSS_CHECK(zeroRoot.value().covariance.at(0U, 0U) == 0.0);

  const Matrix indefinite = matrix({{1.0, 2.0}, {2.0, 1.0}});
  const auto invalid =
      covariance::inverseSquareRoot(indefinite.view(), policy(), limits());
  VSS_CHECK(!invalid.hasValue());
  VSS_CHECK(invalid.status().code() == ErrorCode::NotPositiveSemidefinite);
}

void determinantAndLogMagnitude() {
  using covariance::determinant;
  using covariance::logAbsoluteDeterminant;
  const SolvePolicy solve = policy();
  const Matrix exchange = matrix({{0.0, 2.0}, {3.0, 0.0}});
  VSS_CHECK(determinant(exchange.view(), solve, limits()).value() == -6.0);
  VSS_CHECK(
      std::abs(
          logAbsoluteDeterminant(exchange.view(), solve, limits()).value() -
          std::log(6.0)) < 1.0e-12);
  const Matrix empty;
  VSS_CHECK(determinant(empty.view(), solve, {}).value() == 1.0);
  VSS_CHECK(logAbsoluteDeterminant(empty.view(), solve, {}).value() == 0.0);

  const Matrix singular(2U, 2U, 1.0);
  const auto zero = determinant(singular.view(), solve, limits()).value();
  VSS_CHECK(zero == 0.0 && !std::signbit(zero));
  const auto logZero =
      logAbsoluteDeterminant(singular.view(), solve, limits()).value();
  VSS_CHECK(std::isinf(logZero) && logZero < 0.0);

  // The rounded elimination pivot can vanish even though the integer
  // determinant is exactly one.
  const Scalar adjacent = std::ldexp(1.0, 52);
  const Matrix cancellation =
      matrix({{adjacent, adjacent - 1.0}, {adjacent + 1.0, adjacent}});
  VSS_CHECK(determinant(cancellation.view(), solve, limits()).value() == 1.0);
  VSS_CHECK(
      logAbsoluteDeterminant(cancellation.view(), solve, limits()).value() ==
      0.0);
  VSS_CHECK(
      determinant(cancellation.view(), solve, {4U, 128U, 0U}).status().code() ==
      ErrorCode::ResourceLimit);
  const Matrix permuted = matrix({{0.0, adjacent, adjacent - 1.0},
                                  {1.0, 0.0, 0.0},
                                  {0.0, adjacent + 1.0, adjacent}});
  VSS_CHECK(determinant(permuted.view(), solve, limits()).value() == -1.0);

  const Matrix division = matrix({{adjacent, adjacent - 1.0, 0.0},
                                  {adjacent + 1.0, adjacent, 0.0},
                                  {0.0, 0.0, 7.0}});
  VSS_CHECK(determinant(division.view(), solve, limits()).value() == 7.0);

  const Scalar subnormal = std::numeric_limits<Scalar>::denorm_min();
  const Matrix tieToZero = matrix({{-subnormal, 0.0}, {0.0, 0.5}});
  const auto tie = determinant(tieToZero.view(), solve, limits()).value();
  VSS_CHECK(tie == 0.0 && std::signbit(tie));
  const Matrix tieToEven = matrix({{subnormal, 0.0}, {0.0, 1.5}});
  VSS_CHECK(determinant(tieToEven.view(), solve, limits()).value() ==
            2.0 * subnormal);
  Matrix near = singular;
  near.at(1U, 1U) = std::nextafter(1.0, 2.0);
  VSS_CHECK(determinant(near.view(), solve, limits()).value() ==
            std::numeric_limits<Scalar>::epsilon());

  const Scalar large = std::ldexp(1.0, 800);
  const Scalar small = std::ldexp(1.0, -800);
  Matrix balanced(4U, 4U);
  balanced.at(0U, 0U) = balanced.at(1U, 1U) = large;
  balanced.at(2U, 2U) = balanced.at(3U, 3U) = small;
  VSS_CHECK(determinant(balanced.view(), solve, limits()).value() == 1.0);
  VSS_CHECK(logAbsoluteDeterminant(balanced.view(), solve, limits()).value() ==
            0.0);
  const Matrix overflow = matrix({{large, 0.0}, {0.0, large}});
  VSS_CHECK(determinant(overflow.view(), solve, limits()).status().code() ==
            ErrorCode::NonFiniteResult);
  VSS_CHECK(
      std::abs(
          logAbsoluteDeterminant(overflow.view(), solve, limits()).value() -
          1600.0 * std::log(2.0)) < 1.0e-12);
  const Matrix underflow = matrix({{-small, 0.0}, {0.0, small}});
  const auto rounded = determinant(underflow.view(), solve, limits()).value();
  VSS_CHECK(rounded == 0.0 && std::signbit(rounded));
  VSS_CHECK(
      std::abs(
          logAbsoluteDeterminant(underflow.view(), solve, limits()).value() +
          1600.0 * std::log(2.0)) < 1.0e-12);

  const Matrix nonfinite(1U, 1U, std::numeric_limits<Scalar>::infinity());
  VSS_CHECK(logAbsoluteDeterminant(nonfinite.view(), solve, limits())
                .status()
                .code() == ErrorCode::NonFiniteInput);
  auto exhausted = solve;
  exhausted.iteration.max_iterations = 0;
  exhausted.reciprocal_condition_floor = 0.1;
  const Matrix general = matrix({{1.0, 1.0}, {0.0, 1.0}});
  VSS_CHECK(determinant(general.view(), exhausted, limits()).status().code() ==
            ErrorCode::NoConvergence);
  exhausted.iteration.max_iterations = 128U;
  VSS_CHECK(determinant(general.view(), exhausted, limits()).value() == 1.0);
  exhausted.reciprocal_condition_floor = 0.5;
  VSS_CHECK(determinant(general.view(), exhausted, limits()).status().code() ==
            ErrorCode::IllConditioned);
  auto widePolicy = solve;
  widePolicy.rank = RankPolicy::reject;
  widePolicy.rank_rel = 0.25;
  widePolicy.reciprocal_condition_floor = 0.9;
  const Scalar wide = 0.75 * std::numeric_limits<Scalar>::max();
  const Matrix rotation = matrix({{wide, wide}, {-wide, wide}});
  const auto logWide =
      logAbsoluteDeterminant(rotation.view(), widePolicy, limits());
  VSS_CHECK(logWide.hasValue());
  VSS_CHECK(std::abs(logWide.value() - (2.0 * std::log(wide) + std::log(2.0))) <
            1.0e-12);
  const Matrix unequal = matrix({{1.0, 0.0}, {0.0, 0.25}});
  auto strict = solve;
  strict.rank = RankPolicy::reject;
  strict.rank_rel = 0.25;
  VSS_CHECK(determinant(unequal.view(), strict, limits()).status().code() ==
            ErrorCode::RankDeficient);
  strict.rank = RankPolicy::minimum_norm;
  VSS_CHECK(determinant(unequal.view(), strict, limits()).value() == 0.25);
  strict.reciprocal_condition_floor = 0.5;
  VSS_CHECK(logAbsoluteDeterminant(unequal.view(), strict, limits())
                .status()
                .code() == ErrorCode::IllConditioned);
  strict.rank_abs = -1.0;
  VSS_CHECK(determinant(exchange.view(), strict, limits()).status().code() ==
            ErrorCode::InvalidTolerance);
}

} // namespace

int main() {
  marginalCovariance();
  inverseSquareRoot();
  determinantAndLogMagnitude();
  return 0;
}
