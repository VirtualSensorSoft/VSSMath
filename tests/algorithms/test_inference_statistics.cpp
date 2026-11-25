#include "Test.hpp"

#include <cmath>
#include <limits>
#include <vss/math/Error.hpp>
#include <vss/math/statistics/Inference.hpp>

using namespace vss::math;
using namespace vss::math::statistics;

namespace {

void varianceIntervalUsesExactNormalIidChiSquareQuantiles() {
  const auto interval =
      varianceConfidenceInterval(1.0, 3U, 0.5, NormalIID{true});
  VSS_CHECK(interval.hasValue());
  VSS_CHECK(std::abs(interval.value().lower - 1.0 / std::log(4.0)) < 1e-12);
  VSS_CHECK(std::abs(interval.value().upper - 1.0 / std::log(4.0 / 3.0)) <
            1e-12);

  const auto zero = varianceConfidenceInterval(0.0, 3U, 0.95, NormalIID{true});
  VSS_CHECK(zero.hasValue());
  VSS_CHECK(zero.value().lower == 0.0);
  VSS_CHECK(zero.value().upper == 0.0);
}

void varianceIntervalRejectsInvalidPremisesAndUnrepresentableEndpoints() {
  const auto premise =
      varianceConfidenceInterval(1.0, 3U, 0.5, NormalIID{false});
  VSS_CHECK(!premise.hasValue());
  VSS_CHECK(premise.status().code() == ErrorCode::InvalidDomain);

  const auto count = varianceConfidenceInterval(1.0, 1U, 0.5, NormalIID{true});
  VSS_CHECK(!count.hasValue());
  VSS_CHECK(count.status().code() == ErrorCode::InsufficientSamples);

  const auto confidence =
      varianceConfidenceInterval(1.0, 3U, 1.0, NormalIID{true});
  VSS_CHECK(!confidence.hasValue());
  VSS_CHECK(confidence.status().code() == ErrorCode::InvalidDomain);

  const auto negative =
      varianceConfidenceInterval(-1.0, 3U, 0.5, NormalIID{true});
  VSS_CHECK(!negative.hasValue());
  VSS_CHECK(negative.status().code() == ErrorCode::InvalidDomain);

  const auto overflow = varianceConfidenceInterval(
      std::numeric_limits<Scalar>::max(), 4U, 0.95, NormalIID{true});
  VSS_CHECK(!overflow.hasValue());
  VSS_CHECK(overflow.status().code() == ErrorCode::NonFiniteResult);
}

} // namespace

int main() {
  varianceIntervalUsesExactNormalIidChiSquareQuantiles();
  varianceIntervalRejectsInvalidPremisesAndUnrepresentableEndpoints();
}
