#include "Test.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>
#include <vss/math/Core.hpp>
#include <vss/math/Error.hpp>
#include <vss/math/Limits.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/SampleView.hpp>
#include <vss/math/Vector.hpp>
#include <vss/math/probability/Distribution.hpp>
#include <vss/math/statistics/Bootstrap.hpp>

using namespace vss::math;
using namespace vss::math::statistics;

namespace {

Limits ample() {
  return {.elements = 100000U, .bytes = 800000U, .evaluations = 10000U};
}

SampleView sample(const Matrix& rows) {
  return {.rows = rows.view(), .observation_shape = {}, .valid = {}};
}

Result<Vector> mean(const SampleView& rows) {
  if (rows.rows.rows() == 0U || rows.rows.columns() != 1U) {
    return Result<Vector>::failure(ErrorCode::InvalidDomain,
                                   "mean expects nonempty scalar rows");
  }
  Vector result(1U, 0.0);
  for (Index row = 0U; row < rows.rows.rows(); ++row) {
    result.at(0U) += rows.rows.at(row, 0U);
  }
  result.at(0U) /= static_cast<Scalar>(rows.rows.rows());
  return Result<Vector>::success(std::move(result));
}

Index unbiasedIndex(probability::Rng& rng, Index bound) {
  if (bound == 1U) {
    static_cast<void>(rng.next());
    return 0U;
  }
  const auto wide = static_cast<std::uint64_t>(bound);
  const std::uint64_t threshold = (std::uint64_t{0} - wide) % wide;
  for (Index attempt = 0U; attempt < 128U; ++attempt) {
    const auto word = rng.next();
    if (word >= threshold) {
      return static_cast<Index>(word % wide);
    }
  }
  return bound;
}

Scalar linearQuantile(std::vector<Scalar> values, Scalar probabilityValue) {
  std::ranges::sort(values);
  const Scalar position =
      static_cast<Scalar>(values.size() - 1U) * probabilityValue;
  const auto lower = static_cast<Index>(std::floor(position));
  const Index upper = std::min(lower + 1U, values.size() - 1U);
  return std::lerp(values[lower], values[upper],
                   position - static_cast<Scalar>(lower));
}

void bootstrapReturnsPercentileAndBasicIntervals() {
  const Matrix rows(4U, 1U, 2.0);
  const BootstrapPolicy percentile{.kind = BootstrapKind::iid,
                                   .block_length = 1U,
                                   .draws = 32U,
                                   .confidence = 0.9,
                                   .interval = QuantileInterval::percentile,
                                   .invalid = InvalidDraw::reject};
  probability::Rng percentileRng(17U);
  const auto result =
      bootstrapEstimate(sample(rows), mean, percentile, percentileRng, ample());
  VSS_CHECK(result.hasValue());
  VSS_CHECK(result.value().estimates.rows() == percentile.draws);
  VSS_CHECK(result.value().estimates.columns() == 1U);
  VSS_CHECK(result.value().intervals.rows() == 1U);
  VSS_CHECK(result.value().intervals.columns() == 2U);
  VSS_CHECK(result.value().intervals.at(0U, 0U) == 2.0);
  VSS_CHECK(result.value().intervals.at(0U, 1U) == 2.0);
  VSS_CHECK(result.value().monte_carlo_error.has_value());
  const auto monteCarloError = result.value().monte_carlo_error;
  VSS_CHECK(monteCarloError.has_value());
  if (monteCarloError.has_value()) {
    VSS_CHECK(monteCarloError.value().at(0U) == 0.0);
  }

  BootstrapPolicy basic = percentile;
  basic.interval = QuantileInterval::basic;
  probability::Rng basicRng(17U);
  const auto reflected =
      bootstrapEstimate(sample(rows), mean, basic, basicRng, ample());
  VSS_CHECK(reflected.hasValue());
  VSS_CHECK(reflected.value().intervals.at(0U, 0U) == 2.0);
  VSS_CHECK(reflected.value().intervals.at(0U, 1U) == 2.0);
}

void bootstrapValidatesSamplingAndEvaluationBudgetsBeforeCallingStatistic() {
  const Matrix rows(3U, 1U, 1.0);
  Index calls = 0U;
  const Statistic counted = [&calls](const SampleView& input) {
    ++calls;
    return mean(input);
  };
  BootstrapPolicy policy{.kind = BootstrapKind::iid,
                         .block_length = 1U,
                         .draws = 4U,
                         .confidence = 0.9,
                         .interval = QuantileInterval::percentile,
                         .invalid = InvalidDraw::reject};
  Limits capped = ample();
  capped.evaluations = 3U;
  probability::Rng rng(1U);
  const auto denied =
      bootstrapEstimate(sample(rows), counted, policy, rng, capped);
  VSS_CHECK(!denied.hasValue());
  VSS_CHECK(denied.status().code() == ErrorCode::ResourceLimit);
  VSS_CHECK(calls == 0U);

  policy.block_length = 4U;
  probability::Rng validRng(1U);
  const auto badBlock =
      bootstrapEstimate(sample(rows), counted, policy, validRng, ample());
  VSS_CHECK(!badBlock.hasValue());
  VSS_CHECK(badBlock.status().code() == ErrorCode::InvalidDomain);
  VSS_CHECK(calls == 0U);
}

void bootstrapUsesTheSpecifiedSeedStreamAndBasicIntervalFormula() {
  Matrix rows(5U, 1U, 0.0);
  for (Index row = 0U; row < rows.rows(); ++row) {
    rows.at(row, 0U) = static_cast<Scalar>(row + 1U);
  }
  Index calls = 0U;
  const Statistic counted = [&calls](const SampleView& input) {
    ++calls;
    return mean(input);
  };
  const BootstrapPolicy policy{.kind = BootstrapKind::iid,
                               .block_length = 1U,
                               .draws = 5U,
                               .confidence = 0.6,
                               .interval = QuantileInterval::basic,
                               .invalid = InvalidDraw::reject};
  probability::Rng rng(42U);
  const auto result =
      bootstrapEstimate(sample(rows), counted, policy, rng, ample());
  VSS_CHECK(result.hasValue());
  VSS_CHECK(calls == 6U);

  probability::Rng referenceRng(42U);
  std::vector<Scalar> means;
  for (Index draw = 0U; draw < policy.draws; ++draw) {
    Scalar total = 0.0;
    for (Index row = 0U; row < rows.rows(); ++row) {
      total +=
          static_cast<Scalar>(unbiasedIndex(referenceRng, rows.rows()) + 1U);
    }
    const Scalar expectedMean = total / static_cast<Scalar>(rows.rows());
    means.push_back(expectedMean);
    VSS_CHECK(result.value().estimates.at(draw, 0U) == expectedMean);
  }
  const Scalar lowerQuantile = linearQuantile(means, 0.2);
  const Scalar upperQuantile = linearQuantile(means, 0.8);
  VSS_CHECK(std::abs(result.value().intervals.at(0U, 0U) -
                     (6.0 - upperQuantile)) < 1e-12);
  VSS_CHECK(std::abs(result.value().intervals.at(0U, 1U) -
                     (6.0 - lowerQuantile)) < 1e-12);
}

void bootstrapHonorsInvalidDrawPolicyWithoutRetrying() {
  const Matrix rows(2U, 1U, 1.0);
  Index calls = 0U;
  const Statistic invalid = [&calls](const SampleView&) {
    ++calls;
    return Result<Vector>::failure(ErrorCode::InvalidDomain,
                                   "fixture invalid replicate");
  };
  BootstrapPolicy policy{.kind = BootstrapKind::iid,
                         .block_length = 1U,
                         .draws = 3U,
                         .confidence = 0.8,
                         .interval = QuantileInterval::percentile,
                         .invalid = InvalidDraw::reject};
  probability::Rng rejectRng(11U);
  const auto rejected =
      bootstrapEstimate(sample(rows), invalid, policy, rejectRng, ample());
  VSS_CHECK(!rejected.hasValue());
  VSS_CHECK(rejected.status().code() == ErrorCode::InvalidDomain);
  VSS_CHECK(calls == 1U);

  calls = 0U;
  policy.invalid = InvalidDraw::report_conditional;
  probability::Rng conditionalRng(11U);
  const auto conditional =
      bootstrapEstimate(sample(rows), invalid, policy, conditionalRng, ample());
  VSS_CHECK(!conditional.hasValue());
  VSS_CHECK(conditional.status().code() == ErrorCode::InsufficientSamples);
  VSS_CHECK(calls == policy.draws);
}

} // namespace

int main() {
  bootstrapReturnsPercentileAndBasicIntervals();
  bootstrapValidatesSamplingAndEvaluationBudgetsBeforeCallingStatistic();
  bootstrapUsesTheSpecifiedSeedStreamAndBasicIntervalFormula();
  bootstrapHonorsInvalidDrawPolicyWithoutRetrying();
}
