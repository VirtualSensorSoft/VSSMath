#include "Test.hpp"

#include <array>
#include <cmath>
#include <initializer_list>
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

} // namespace

int main() {
  marginalCovariance();
  inverseSquareRoot();
  return 0;
}
