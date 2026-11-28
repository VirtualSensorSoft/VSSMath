#include "Test.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <numbers>
#include <random>
#include <utility>
#include <vector>
#include <vss/math/Core.hpp>
#include <vss/math/Error.hpp>
#include <vss/math/Limits.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/SampleView.hpp>
#include <vss/math/Vector.hpp>
#include <vss/math/covariance/Conditioning.hpp>
#include <vss/math/covariance/PropagationResult.hpp>
#include <vss/math/probability/Distribution.hpp>
#include <vss/math/statistics/Bootstrap.hpp>
#include <vss/math/statistics/DependentInference.hpp>

using namespace vss::math;
using namespace vss::math::statistics;

namespace {

Limits ample() {
  return {.elements = 100000U, .bytes = 800000U, .evaluations = 10000U};
}

Matrix matrix(std::initializer_list<std::initializer_list<Scalar>> rows) {
  const Extent columns = rows.size() == 0U ? 0U : rows.begin()->size();
  Matrix output(rows.size(), columns);
  Extent i = 0U;
  for (const auto& row : rows) {
    VSS_CHECK(row.size() == columns);
    Extent j = 0U;
    for (const Scalar value : row) {
      output.at(i, j++) = value;
    }
    ++i;
  }
  return output;
}

SampleView sample(const Matrix& rows) {
  return {.rows = rows.view(), .observation_shape = {}, .valid = {}};
}

Result<Vector> mean(const SampleView& rows) {
  if (rows.rows.rows() == 0U || rows.rows.columns() != 1U) {
    return Result<Vector>::failure(ErrorCode::InvalidDomain,
                                   "mean expects nonempty scalar rows");
  }
  Vector output(1U, 0.0);
  for (Index i = 0U; i < rows.rows.rows(); ++i) {
    output.at(0U) += rows.rows.at(i, 0U);
  }
  output.at(0U) /= static_cast<Scalar>(rows.rows.rows());
  return Result<Vector>::success(std::move(output));
}

void checkNear(Scalar actual, Scalar expected) {
  VSS_CHECK(std::abs(actual - expected) <=
            1.0e-12 +
                (1.0e-10 * std::max(std::abs(actual), std::abs(expected))));
}

Index referenceIndex(probability::Rng& rng, Index bound) {
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

Scalar referenceQuantile(std::vector<Scalar> values, Scalar probabilityValue) {
  std::ranges::sort(values);
  const Scalar position =
      static_cast<Scalar>(values.size() - 1U) * probabilityValue;
  const auto lower = static_cast<Index>(std::floor(position));
  const Index upper = std::min(lower + 1U, values.size() - 1U);
  return std::lerp(values[lower], values[upper],
                   position - static_cast<Scalar>(lower));
}

Scalar independentBlockMean(const Matrix& rows, BootstrapKind kind,
                            Index blockLength, probability::Rng& rng) {
  const Index count = rows.rows();
  Scalar total = 0.0;
  Index output = 0U;
  while (output < count) {
    const Index startCount =
        kind == BootstrapKind::moving_block ? count - blockLength + 1U : count;
    Index source = referenceIndex(rng, startCount);
    for (Index offset = 0U; offset < blockLength && output < count; ++offset) {
      total += rows.at(source, 0U);
      ++output;
      if (kind == BootstrapKind::circular_block) {
        source = source + 1U == count ? 0U : source + 1U;
      } else if (offset + 1U < blockLength && output < count) {
        ++source;
      }
    }
  }
  return total / static_cast<Scalar>(count);
}

Scalar normalFrom(std::mt19937_64& engine) {
  const Scalar first =
      (static_cast<Scalar>(engine() >> 11U) + 0.5) / 9007199254740992.0;
  const Scalar second =
      (static_cast<Scalar>(engine() >> 11U) + 0.5) / 9007199254740992.0;
  return std::sqrt(-2.0 * std::log(first)) *
         std::cos(2.0 * std::numbers::pi * second);
}

Matrix stationaryAr1(Index count, Scalar rho, std::uint64_t seed) {
  std::mt19937_64 engine(seed);
  Matrix rows(count, 1U);
  Scalar previous = normalFrom(engine) / std::sqrt(1.0 - (rho * rho));
  rows.at(0U, 0U) = previous;
  for (Index index = 1U; index < count; ++index) {
    previous = (rho * previous) + normalFrom(engine);
    rows.at(index, 0U) = previous;
  }
  return rows;
}

void dependentBootstrapUsesCircularBlocksAndPreservesConstantStatistics() {
  const Matrix rows(5U, 1U, 4.0);
  const BootstrapPolicy policy{.kind = BootstrapKind::circular_block,
                               .block_length = 2U,
                               .draws = 16U,
                               .confidence = 0.8,
                               .interval = QuantileInterval::percentile,
                               .invalid = InvalidDraw::reject};
  probability::Rng rng(719U);
  const auto result =
      dependentBootstrap(sample(rows), mean, policy, rng, ample());
  VSS_CHECK(result.hasValue());
  if (result.hasValue()) {
    VSS_CHECK(result.value().intervals.at(0U, 0U) == 4.0);
    VSS_CHECK(result.value().intervals.at(0U, 1U) == 4.0);
  }
}

void dependentBootstrapRejectsIidAndInvalidBlockLength() {
  const Matrix rows(3U, 1U, 1.0);
  BootstrapPolicy policy{.kind = BootstrapKind::iid,
                         .block_length = 1U,
                         .draws = 4U,
                         .confidence = 0.8,
                         .interval = QuantileInterval::percentile,
                         .invalid = InvalidDraw::reject};
  probability::Rng iidRng(1U);
  auto result = dependentBootstrap(sample(rows), mean, policy, iidRng, ample());
  VSS_CHECK(!result.hasValue());
  if (!result.hasValue()) {
    VSS_CHECK(result.status().code() == ErrorCode::InvalidDomain);
  }
  policy.kind = BootstrapKind::moving_block;
  policy.block_length = 4U;
  probability::Rng blockRng(1U);
  result = dependentBootstrap(sample(rows), mean, policy, blockRng, ample());
  VSS_CHECK(!result.hasValue());
  if (!result.hasValue()) {
    VSS_CHECK(result.status().code() == ErrorCode::InvalidDomain);
  }
}

void dependentBootstrapLengthOneMatchesIidBootstrapStream() {
  Matrix rows(4U, 1U);
  rows.at(0U, 0U) = 0.0;
  rows.at(1U, 0U) = 2.0;
  rows.at(2U, 0U) = 4.0;
  rows.at(3U, 0U) = 6.0;
  BootstrapPolicy policy{.kind = BootstrapKind::moving_block,
                         .block_length = 1U,
                         .draws = 17U,
                         .confidence = 0.95,
                         .interval = QuantileInterval::basic,
                         .invalid = InvalidDraw::reject};
  probability::Rng blockRng(804U);
  const auto block =
      dependentBootstrap(sample(rows), mean, policy, blockRng, ample());
  policy.kind = BootstrapKind::iid;
  probability::Rng iidRng(804U);
  const auto iid =
      bootstrapEstimate(sample(rows), mean, policy, iidRng, ample());
  VSS_CHECK(block.hasValue());
  VSS_CHECK(iid.hasValue());
  if (block.hasValue() && iid.hasValue()) {
    for (Index i = 0U; i < policy.draws; ++i) {
      checkNear(block.value().estimates.at(i, 0U),
                iid.value().estimates.at(i, 0U));
    }
    checkNear(block.value().intervals.at(0U, 0U),
              iid.value().intervals.at(0U, 0U));
    checkNear(block.value().intervals.at(0U, 1U),
              iid.value().intervals.at(0U, 1U));
  }
}

void dependentBootstrapMatchesIndependentAr1ReplayAndReferenceVariance() {
  constexpr Scalar normal975 = 1.959963984540054;
  for (const Index count : {32U, 128U}) {
    const Index blockLength = count == 32U ? 3U : 5U;
    for (const Scalar rho : {-0.5, 0.0, 0.5}) {
      const std::uint64_t profile =
          static_cast<std::uint64_t>(count) +
          static_cast<std::uint64_t>((rho + 0.5) * 100.0);
      const Matrix rows = stationaryAr1(count, rho, 91007U + profile);
      Scalar exactVariance = 0.0;
      for (Index left = 0U; left < count; ++left) {
        for (Index right = 0U; right < count; ++right) {
          const Index distance = left > right ? left - right : right - left;
          exactVariance +=
              std::pow(rho, static_cast<Scalar>(distance)) / (1.0 - rho * rho);
        }
      }
      exactVariance /= static_cast<Scalar>(count * count);
      Scalar formulaVariance = 0.0;
      for (Index lag = 1U; lag < count; ++lag) {
        formulaVariance += 2.0 * static_cast<Scalar>(count - lag) *
                           std::pow(rho, static_cast<Scalar>(lag));
      }
      formulaVariance =
          (static_cast<Scalar>(count) + formulaVariance) /
          (static_cast<Scalar>(count * count) * (1.0 - rho * rho));
      checkNear(exactVariance, formulaVariance);
      checkNear(1.0 - std::erfc(normal975 / std::numbers::sqrt2), 0.95);
      const Scalar referenceHalfWidth = normal975 * std::sqrt(exactVariance);
      VSS_CHECK(referenceHalfWidth > 0.0);

      for (const BootstrapKind kind :
           {BootstrapKind::moving_block, BootstrapKind::circular_block}) {
        for (const QuantileInterval interval :
             {QuantileInterval::percentile, QuantileInterval::basic}) {
          const BootstrapPolicy policy{.kind = kind,
                                       .block_length = blockLength,
                                       .draws = 17U,
                                       .confidence = 0.95,
                                       .interval = interval,
                                       .invalid = InvalidDraw::reject};
          const std::uint64_t seed = 513009U + (profile * 4U) +
                                     (static_cast<std::uint64_t>(kind) * 2U) +
                                     static_cast<std::uint64_t>(interval);
          probability::Rng actualRng(seed);
          const auto actual = dependentBootstrap(sample(rows), mean, policy,
                                                 actualRng, ample());
          VSS_CHECK(actual.hasValue());
          if (!actual.hasValue()) {
            continue;
          }
          probability::Rng referenceRng(seed);
          std::vector<Scalar> estimates;
          estimates.reserve(policy.draws);
          for (Index draw = 0U; draw < policy.draws; ++draw) {
            estimates.push_back(
                independentBlockMean(rows, kind, blockLength, referenceRng));
            checkNear(actual.value().estimates.at(draw, 0U), estimates.back());
          }
          const Scalar alpha = 1.0 - policy.confidence;
          const Scalar lowerQ = referenceQuantile(estimates, alpha * 0.5);
          const Scalar upperQ =
              referenceQuantile(estimates, 1.0 - (alpha * 0.5));
          Scalar expectedLower = lowerQ;
          Scalar expectedUpper = upperQ;
          if (interval == QuantileInterval::basic) {
            const Scalar observed = mean(sample(rows)).value().at(0U);
            expectedLower = (2.0 * observed) - upperQ;
            expectedUpper = (2.0 * observed) - lowerQ;
          }
          checkNear(actual.value().intervals.at(0U, 0U), expectedLower);
          checkNear(actual.value().intervals.at(0U, 1U), expectedUpper);
        }
      }

      const BootstrapPolicy fullBlock{.kind = BootstrapKind::circular_block,
                                      .block_length = count,
                                      .draws = 17U,
                                      .confidence = 0.95,
                                      .interval = QuantileInterval::percentile,
                                      .invalid = InvalidDraw::reject};
      probability::Rng fullRng(81377U + profile);
      const auto collapsed =
          dependentBootstrap(sample(rows), mean, fullBlock, fullRng, ample());
      VSS_CHECK(collapsed.hasValue());
      if (collapsed.hasValue()) {
        const Scalar observed = mean(sample(rows)).value().at(0U);
        checkNear(collapsed.value().intervals.at(0U, 0U), observed);
        checkNear(collapsed.value().intervals.at(0U, 1U), observed);
      }
    }
  }
}

void knownCovarianceAr1ReferencePassesPredeclaredCoverageExperiment() {
  constexpr Index trials = 131072U;
  constexpr Scalar confidence = 0.95;
  constexpr Scalar intervalAllowance = 0.0077927;
  constexpr Scalar normal975 = 1.959963984540054;
  constexpr Scalar alphaPerAssertion = 0.001 / 4096.0;
  const Scalar calculatedAllowance = std::sqrt(
      std::log(2.0 / alphaPerAssertion) / (2.0 * static_cast<Scalar>(trials)));
  const Scalar roundedAllowance =
      std::ceil(calculatedAllowance * 1.0e7) / 1.0e7;
  checkNear(roundedAllowance, intervalAllowance);
  const Scalar alternativeMargin = 0.02 - intervalAllowance;
  VSS_CHECK(std::exp(-2.0 * static_cast<Scalar>(trials) * alternativeMargin *
                     alternativeMargin) < 0.01);

  Index assertion = 0U;
  for (const Index count : {32U, 128U}) {
    for (const Scalar rho : {-0.5, 0.0, 0.5}) {
      auto varianceNumerator = static_cast<Scalar>(count);
      for (Index lag = 1U; lag < count; ++lag) {
        varianceNumerator += 2.0 * static_cast<Scalar>(count - lag) *
                             std::pow(rho, static_cast<Scalar>(lag));
      }
      const Scalar meanVariance =
          varianceNumerator /
          (static_cast<Scalar>(count * count) * (1.0 - (rho * rho)));
      const Scalar halfWidth = normal975 * std::sqrt(meanVariance);
      const Scalar exactCoverage =
          1.0 - std::erfc(normal975 / std::numbers::sqrt2);
      checkNear(exactCoverage, confidence);

      std::mt19937_64 engine(820391U + assertion);
      Index covered = 0U;
      const Scalar initialScale = 1.0 / std::sqrt(1.0 - (rho * rho));
      for (Index trial = 0U; trial < trials; ++trial) {
        Scalar previous = normalFrom(engine) * initialScale;
        Scalar total = previous;
        for (Index row = 1U; row < count; ++row) {
          previous = (rho * previous) + normalFrom(engine);
          total += previous;
        }
        const Scalar trajectoryMean = total / static_cast<Scalar>(count);
        covered += std::abs(trajectoryMean) <= halfWidth ? 1U : 0U;
      }
      const Scalar observed =
          static_cast<Scalar>(covered) / static_cast<Scalar>(trials);
      VSS_CHECK(std::abs(observed - exactCoverage) <= intervalAllowance);
      ++assertion;
    }
  }
}

void circularBootstrapRetainsTheFiniteMarkovCoverageReference() {
  constexpr Index trials = 2048U;
  for (Index regime = 0U; regime < 2U; ++regime) {
    const Scalar stayProbability = regime == 0U ? 0.5 : 0.75;
    Scalar exactCoverage = 0.0;
    for (Index pattern = 0U; pattern < 16U; ++pattern) {
      std::array<Scalar, 4U> observations{};
      Index matches = 0U;
      for (Index index = 0U; index < 4U; ++index) {
        observations.at(index) = static_cast<Scalar>((pattern >> index) & 1U);
        if (index != 0U &&
            observations.at(index) == observations.at(index - 1U)) {
          ++matches;
        }
      }
      const Scalar dataMass =
          0.5 * std::pow(stayProbability, static_cast<Scalar>(matches)) *
          std::pow(1.0 - stayProbability, static_cast<Scalar>(3U - matches));
      Index below = 0U;
      Index above = 0U;
      for (Index firstStart = 0U; firstStart < 4U; ++firstStart) {
        for (Index secondStart = 0U; secondStart < 4U; ++secondStart) {
          const Scalar replicateMean =
              (observations.at(firstStart) +
               observations.at((firstStart + 1U) % 4U) +
               observations.at(secondStart) +
               observations.at((secondStart + 1U) % 4U)) /
              4.0;
          below += replicateMean < 0.5 ? 1U : 0U;
          above += replicateMean > 0.5 ? 1U : 0U;
        }
      }
      const Scalar belowProbability = static_cast<Scalar>(below) / 16.0;
      const Scalar aboveProbability = static_cast<Scalar>(above) / 16.0;
      const auto miss = [](Scalar probability) {
        return (5.0 * std::pow(probability, 4.0) * (1.0 - probability)) +
               std::pow(probability, 5.0);
      };
      exactCoverage +=
          dataMass * (1.0 - miss(belowProbability) - miss(aboveProbability));
    }
    const Scalar reference =
        regime == 0U ? 283489.0 / 524288.0 : 686627.0 / 2097152.0;
    checkNear(exactCoverage, reference);

    std::mt19937_64 data(734277U + regime);
    probability::Rng rng(516209U + regime);
    Index covered = 0U;
    for (Index trial = 0U; trial < trials; ++trial) {
      Matrix rows(4U, 1U);
      std::uint64_t word = data();
      std::uint64_t state = word & 1U;
      rows.at(0U, 0U) = static_cast<Scalar>(state);
      for (Index index = 1U; index < 4U; ++index) {
        word = data();
        const bool stay = regime == 0U ? (word & 1U) != 0U : (word & 3U) != 0U;
        if (!stay) {
          state = 1U - state;
        }
        rows.at(index, 0U) = static_cast<Scalar>(state);
      }
      const BootstrapPolicy policy{.kind = BootstrapKind::circular_block,
                                   .block_length = 2U,
                                   .draws = 5U,
                                   .confidence = 0.5,
                                   .interval = QuantileInterval::percentile,
                                   .invalid = InvalidDraw::reject};
      const auto result =
          dependentBootstrap(sample(rows), mean, policy, rng, ample());
      VSS_CHECK(result.hasValue());
      if (result.hasValue() && result.value().intervals.at(0U, 0U) <= 0.5 &&
          result.value().intervals.at(0U, 1U) >= 0.5) {
        ++covered;
      }
    }
    const Scalar rate = static_cast<Scalar>(covered) / trials;
    VSS_CHECK(std::abs(rate - reference) <= 0.06);
  }
}

void fullLengthCircularBlocksCollapseMeanInterval() {
  const BootstrapPolicy policy{.kind = BootstrapKind::circular_block,
                               .block_length = 4U,
                               .draws = 16U,
                               .confidence = 0.5,
                               .interval = QuantileInterval::percentile,
                               .invalid = InvalidDraw::reject};
  Index covered = 0U;
  for (Index first = 0U; first < 2U; ++first) {
    for (Index second = 0U; second < 2U; ++second) {
      const Matrix rows = matrix({{static_cast<Scalar>(first)},
                                  {static_cast<Scalar>(first)},
                                  {static_cast<Scalar>(second)},
                                  {static_cast<Scalar>(second)}});
      probability::Rng rng(481U + (first * 2U) + second);
      const auto result =
          dependentBootstrap(sample(rows), mean, policy, rng, ample());
      VSS_CHECK(result.hasValue());
      if (result.hasValue()) {
        const Scalar observedMean = static_cast<Scalar>(first + second) / 2.0;
        checkNear(result.value().intervals.at(0U, 0U), observedMean);
        checkNear(result.value().intervals.at(0U, 1U), observedMean);
        if (result.value().intervals.at(0U, 0U) <= 0.5 &&
            result.value().intervals.at(0U, 1U) >= 0.5) {
          ++covered;
        }
      }
    }
  }
  VSS_CHECK(covered == 2U);
}

void neweyWestMatchesScalarAndJointBartlettSums() {
  const Matrix scalar = matrix({{1.0}, {0.0}, {-1.0}});
  const auto scalarResult =
      neweyWest(sample(scalar), 1U, CenterRule::estimated, ample());
  VSS_CHECK(scalarResult.hasValue());
  if (scalarResult.hasValue()) {
    checkNear(scalarResult.value().covariance.at(0U, 0U), 2.0 / 3.0);
    VSS_CHECK(scalarResult.value().classification ==
              covariance::CovarianceKind::positive_definite);
  }

  const Matrix joint = matrix({{1.0, 0.0}, {0.0, 1.0}, {-1.0, -1.0}});
  const std::array<Index, 1U> vectorShape{2U};
  const SampleView jointScores{
      .rows = joint.view(), .observation_shape = vectorShape, .valid = {}};
  const auto jointResult =
      neweyWest(jointScores, 1U, CenterRule::known, ample());
  VSS_CHECK(jointResult.hasValue());
  if (jointResult.hasValue()) {
    checkNear(jointResult.value().covariance.at(0U, 0U), 2.0 / 3.0);
    checkNear(jointResult.value().covariance.at(0U, 1U), 1.0 / 3.0);
    checkNear(jointResult.value().covariance.at(1U, 0U), 1.0 / 3.0);
    checkNear(jointResult.value().covariance.at(1U, 1U), 1.0 / 3.0);
  }
}

void neweyWestMatchesIndependentLagPairFormulaAcrossLags() {
  const Matrix rows = matrix({{1.0, -2.0},
                              {0.5, 1.0},
                              {-1.0, 3.0},
                              {2.0, 0.25},
                              {-0.5, -1.5},
                              {1.5, 2.0}});
  const std::array<Index, 1U> shape{2U};
  const SampleView scores{
      .rows = rows.view(), .observation_shape = shape, .valid = {}};
  constexpr Index count = 6U;
  constexpr Index dimension = 2U;
  std::array<Scalar, dimension> meanValue{};
  for (Index coordinate = 0U; coordinate < dimension; ++coordinate) {
    for (Index row = 0U; row < count; ++row) {
      meanValue.at(coordinate) +=
          rows.at(row, coordinate) / static_cast<Scalar>(count);
    }
  }
  for (const CenterRule center : {CenterRule::known, CenterRule::estimated}) {
    for (Index lag = 0U; lag < count; ++lag) {
      const auto actual = neweyWest(scores, lag, center, ample());
      VSS_CHECK(actual.hasValue());
      if (!actual.hasValue()) {
        continue;
      }
      for (Index left = 0U; left < dimension; ++left) {
        for (Index right = 0U; right < dimension; ++right) {
          Scalar expected = 0.0;
          for (Index time = 0U; time < count; ++time) {
            const Scalar leftNow =
                rows.at(time, left) -
                (center == CenterRule::estimated ? meanValue.at(left) : 0.0);
            const Scalar rightNow =
                rows.at(time, right) -
                (center == CenterRule::estimated ? meanValue.at(right) : 0.0);
            expected += leftNow * rightNow / count;
          }
          for (Index offset = 1U; offset <= lag; ++offset) {
            Scalar forward = 0.0;
            Scalar reverse = 0.0;
            for (Index time = offset; time < count; ++time) {
              const Scalar leftNow =
                  rows.at(time, left) -
                  (center == CenterRule::estimated ? meanValue.at(left) : 0.0);
              const Scalar rightPrevious =
                  rows.at(time - offset, right) -
                  (center == CenterRule::estimated ? meanValue.at(right) : 0.0);
              const Scalar rightNow =
                  rows.at(time, right) -
                  (center == CenterRule::estimated ? meanValue.at(right) : 0.0);
              const Scalar leftPrevious =
                  rows.at(time - offset, left) -
                  (center == CenterRule::estimated ? meanValue.at(left) : 0.0);
              forward += leftNow * rightPrevious / count;
              reverse += rightNow * leftPrevious / count;
            }
            const Scalar weight = 1.0 - (static_cast<Scalar>(offset) /
                                         static_cast<Scalar>(lag + 1U));
            expected += weight * (forward + reverse);
          }
          checkNear(actual.value().covariance.at(left, right), expected);
        }
      }
    }
  }
}

void neweyWestDistinguishesKnownAndEstimatedCentersAndAcceptsShiftArithmetic() {
  const Matrix constant(3U, 1U, 2.0);
  const auto estimated =
      neweyWest(sample(constant), 0U, CenterRule::estimated, ample());
  const auto known =
      neweyWest(sample(constant), 0U, CenterRule::known, ample());
  VSS_CHECK(estimated.hasValue());
  VSS_CHECK(known.hasValue());
  if (estimated.hasValue()) {
    checkNear(estimated.value().covariance.at(0U, 0U), 0.0);
    VSS_CHECK(estimated.value().classification ==
              covariance::CovarianceKind::semidefinite_singular);
    VSS_CHECK(estimated.value().rank == 0U);
    VSS_CHECK(!estimated.value().reciprocal_condition.has_value());
  }
  if (known.hasValue()) {
    checkNear(known.value().covariance.at(0U, 0U), 4.0);
  }

  const Matrix shifted =
      matrix({{0.0}, {0.0}, {0.0}, {0.0}, {4.0}, {4.0}, {4.0}, {4.0}});
  const auto finite =
      neweyWest(sample(shifted), 1U, CenterRule::estimated, ample());
  VSS_CHECK(finite.hasValue());
  if (finite.hasValue()) {
    VSS_CHECK(std::isfinite(finite.value().covariance.at(0U, 0U)));
  }
}

void neweyWestRejectsInvalidLagMaskAndInsufficientLimits() {
  const Matrix rows = matrix({{1.0}, {2.0}, {3.0}});
  auto result = neweyWest(sample(rows), 3U, CenterRule::estimated, ample());
  VSS_CHECK(!result.hasValue());
  if (!result.hasValue()) {
    VSS_CHECK(result.status().code() == ErrorCode::InvalidDomain);
  }

  std::vector<std::uint8_t> mask{1U, 0U, 1U};
  const SampleView incomplete{
      .rows = rows.view(), .observation_shape = {}, .valid = mask};
  result = neweyWest(incomplete, 1U, CenterRule::estimated, ample());
  VSS_CHECK(!result.hasValue());
  if (!result.hasValue()) {
    VSS_CHECK(result.status().code() == ErrorCode::InvalidDomain);
  }

  const Limits tiny{.elements = 0U, .bytes = 0U, .evaluations = 0U};
  result = neweyWest(sample(rows), 1U, CenterRule::estimated, tiny);
  VSS_CHECK(!result.hasValue());
  if (!result.hasValue()) {
    VSS_CHECK(result.status().code() == ErrorCode::ResourceLimit);
  }
}

void neweyWestValidatesFiniteValuesAndPreservesTensorCoordinateShape() {
  Matrix nonfinite(2U, 1U);
  nonfinite.at(0U, 0U) = 1.0;
  nonfinite.at(1U, 0U) = std::numeric_limits<Scalar>::infinity();
  auto result = neweyWest(sample(nonfinite), 0U, CenterRule::known, ample());
  VSS_CHECK(!result.hasValue());
  if (!result.hasValue()) {
    VSS_CHECK(result.status().code() == ErrorCode::NonFiniteInput);
  }

  Matrix wide(2U, 1U);
  wide.at(0U, 0U) = std::numeric_limits<Scalar>::max();
  wide.at(1U, 0U) = -std::numeric_limits<Scalar>::max();
  result = neweyWest(sample(wide), 0U, CenterRule::estimated, ample());
  VSS_CHECK(!result.hasValue());
  if (!result.hasValue()) {
    VSS_CHECK(result.status().code() == ErrorCode::NonFiniteResult);
  }

  const Matrix tensorRows =
      matrix({{1.0, 2.0, 3.0, 4.0}, {-1.0, 0.5, 2.0, 3.0}});
  const std::array<Index, 2U> tensorShape{2U, 2U};
  const SampleView tensorScores{
      .rows = tensorRows.view(), .observation_shape = tensorShape, .valid = {}};
  result = neweyWest(tensorScores, 0U, CenterRule::known, ample());
  VSS_CHECK(result.hasValue());
  if (result.hasValue()) {
    VSS_CHECK(result.value().covariance.rows() == 4U);
    VSS_CHECK(result.value().covariance.columns() == 4U);
  }
}

} // namespace

int main() {
  dependentBootstrapUsesCircularBlocksAndPreservesConstantStatistics();
  dependentBootstrapRejectsIidAndInvalidBlockLength();
  dependentBootstrapLengthOneMatchesIidBootstrapStream();
  dependentBootstrapMatchesIndependentAr1ReplayAndReferenceVariance();
  knownCovarianceAr1ReferencePassesPredeclaredCoverageExperiment();
  circularBootstrapRetainsTheFiniteMarkovCoverageReference();
  fullLengthCircularBlocksCollapseMeanInterval();
  neweyWestMatchesScalarAndJointBartlettSums();
  neweyWestMatchesIndependentLagPairFormulaAcrossLags();
  neweyWestDistinguishesKnownAndEstimatedCentersAndAcceptsShiftArithmetic();
  neweyWestRejectsInvalidLagMaskAndInsufficientLimits();
  neweyWestValidatesFiniteValuesAndPreservesTensorCoordinateShape();
}
