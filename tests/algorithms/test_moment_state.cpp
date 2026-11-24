#include "Test.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <limits>
#include <new>
#include <optional>
#include <span>
#include <utility>
#include <vss/math/Core.hpp>
#include <vss/math/Error.hpp>
#include <vss/math/Limits.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/SampleView.hpp>
#include <vss/math/Vector.hpp>
#include <vss/math/covariance/PropagationResult.hpp>
#include <vss/math/statistics/MomentState.hpp>
#include <vss/math/statistics/Moments.hpp>
#include <vss/math/statistics/Population.hpp>
#include <vss/math/statistics/Weighted.hpp>

using namespace vss::math;
using namespace vss::math::statistics;

namespace {
// Test-only state for verifying the accumulator's allocation-free contract.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
std::atomic<bool> countAllocations{false};
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
std::atomic<std::size_t> observedAllocations{0U};
} // namespace

void* operator new(std::size_t size) {
  if (countAllocations.load(std::memory_order_relaxed)) {
    observedAllocations.fetch_add(1U, std::memory_order_relaxed);
  }
  // Global allocation replacement is isolated to this test executable.
  // Mutable storage is required by the operator new contract.
  // NOLINTNEXTLINE(misc-const-correctness,cppcoreguidelines-no-malloc,cppcoreguidelines-owning-memory)
  if (void* const allocation = std::malloc(size == 0U ? 1U : size)) {
    return allocation;
  }
  throw std::bad_alloc{};
}

void* operator new[](std::size_t size) { return ::operator new(size); }

void operator delete(void* allocation) noexcept {
  // NOLINTNEXTLINE(cppcoreguidelines-no-malloc,cppcoreguidelines-owning-memory)
  std::free(allocation);
}
void operator delete[](void* allocation) noexcept {
  // NOLINTNEXTLINE(cppcoreguidelines-no-malloc,cppcoreguidelines-owning-memory)
  std::free(allocation);
}
void operator delete(void* allocation, std::size_t size) noexcept {
  static_cast<void>(size);
  // NOLINTNEXTLINE(cppcoreguidelines-no-malloc,cppcoreguidelines-owning-memory)
  std::free(allocation);
}
void operator delete[](void* allocation, std::size_t size) noexcept {
  static_cast<void>(size);
  // NOLINTNEXTLINE(cppcoreguidelines-no-malloc,cppcoreguidelines-owning-memory)
  std::free(allocation);
}

namespace {

bool close(Scalar actual, Scalar expected) {
  return std::abs(actual - expected) <=
         1.0e-12 + (1.0e-10 * std::max(std::abs(actual), std::abs(expected)));
}

void checkPopulationMean(const Result<MomentEstimate>& result,
                         Scalar expected) {
  VSS_CHECK(result);
  if (!result) {
    return;
  }
  const auto& mean = result.value().population_mean;
  VSS_CHECK(mean.has_value());
  if (mean) {
    VSS_CHECK(close(mean.value().at(0U), expected));
  }
}

void emptyAndMergedOnlineMoments() {
  constexpr Limits limits{
      .elements = 4096U, .bytes = 262144U, .evaluations = 0U};
  const MomentRequest request{.orders = 0x0eU,
                              .representation = MomentRepresentation::component,
                              .selected = {},
                              .contraction = std::nullopt};
  auto empty = makeOnlineMoments({}, request, limits);
  VSS_CHECK(empty);
  if (!empty) {
    return;
  }
  VSS_CHECK(
      finalize(empty.value(), WeightRule::empirical, limits).status().code() ==
      ErrorCode::InsufficientSamples);

  const MomentRequest secondOnly{.orders = 0x02U,
                                 .representation =
                                     MomentRepresentation::component,
                                 .selected = {},
                                 .contraction = std::nullopt};
  auto scalarVariance = makeOnlineMoments({}, secondOnly, limits);
  VSS_CHECK(scalarVariance);
  if (scalarVariance) {
    VSS_CHECK(scalarVariance.value().input_dimension == 1U);
    VSS_CHECK(scalarVariance.value().shape.view().empty());
    const Index fixedStorage = scalarVariance.value().centered_sums.size();
    const Vector sample{std::array<Scalar, 1U>{3.0}};
    for (Index i = 0U; i < 32U; ++i) {
      VSS_CHECK(add(scalarVariance.value(), sample.view(), 1.0).ok());
    }
    VSS_CHECK(scalarVariance.value().centered_sums.size() == fixedStorage);
  }

  auto left = makeOnlineMoments({}, request, limits);
  auto right = makeOnlineMoments({}, request, limits);
  VSS_CHECK(left && right);
  if (!left || !right) {
    return;
  }
  const Vector one{std::array<Scalar, 1U>{1.0}};
  const Vector two{std::array<Scalar, 1U>{2.0}};
  const Vector four{std::array<Scalar, 1U>{4.0}};
  VSS_CHECK(add(left.value(), one.view(), 1.0).ok());
  VSS_CHECK(add(right.value(), two.view(), 1.0).ok());
  VSS_CHECK(add(right.value(), four.view(), 1.0).ok());
  VSS_CHECK(merge(left.value(), right.value()).ok());

  const auto estimate = finalize(left.value(), WeightRule::empirical, limits);
  VSS_CHECK(estimate);
  if (estimate) {
    VSS_CHECK(estimate.value().terms.view().size() == 3U);
    VSS_CHECK(
        close(estimate.value().terms.view()[0].values.at(0U), 14.0 / 9.0));
    VSS_CHECK(
        close(estimate.value().terms.view()[1].values.at(0U), 20.0 / 27.0));
    VSS_CHECK(
        close(estimate.value().terms.view()[2].values.at(0U), 98.0 / 27.0));
  }
}

void weightedOffsetAndReset() {
  constexpr Limits limits{
      .elements = 4096U, .bytes = 262144U, .evaluations = 0U};
  const MomentRequest request{.orders = 0x0eU,
                              .representation = MomentRepresentation::component,
                              .selected = {},
                              .contraction = std::nullopt};
  const MomentRequest secondOnly{.orders = 0x02U,
                                 .representation =
                                     MomentRepresentation::component,
                                 .selected = {},
                                 .contraction = std::nullopt};
  auto weighted = makeOnlineMoments({}, request, limits);
  VSS_CHECK(weighted);
  if (weighted) {
    const Vector seven{std::array<Scalar, 1U>{7.0}};
    const Vector one{std::array<Scalar, 1U>{1.0}};
    VSS_CHECK(add(weighted.value(), seven.view(), 1.0).ok());
    VSS_CHECK(add(weighted.value(), one.view(), 3.0).ok());
    const auto estimate =
        finalize(weighted.value(), WeightRule::empirical, limits);
    VSS_CHECK(estimate);
    if (estimate) {
      VSS_CHECK(
          close(estimate.value().terms.view()[0].values.at(0U), 27.0 / 4.0));
    }
    observedAllocations = 0U;
    countAllocations = true;
    const Status onlineReset = reset(weighted.value());
    countAllocations = false;
    VSS_CHECK(onlineReset.ok());
    VSS_CHECK(observedAllocations == 0U);
    VSS_CHECK(finalize(weighted.value(), WeightRule::empirical, limits)
                  .status()
                  .code() == ErrorCode::InsufficientSamples);
    VSS_CHECK(add(weighted.value(), one.view(), 1.0).ok());
    VSS_CHECK(add(weighted.value(), seven.view(), 3.0).ok());
    const auto fresh =
        finalize(weighted.value(), WeightRule::empirical, limits);
    VSS_CHECK(fresh &&
              close(fresh.value().terms.view()[0].values.at(0U), 27.0 / 4.0));
  }

  const MomentRequest varianceOnly{.orders = 0x02U,
                                   .representation =
                                       MomentRepresentation::component,
                                   .selected = {},
                                   .contraction = std::nullopt};
  auto highDynamicRange = makeOnlineMoments({}, varianceOnly, limits);
  VSS_CHECK(highDynamicRange);
  if (highDynamicRange) {
    const Vector negative{std::array<Scalar, 1U>{-1.0e150}};
    const Vector positive{std::array<Scalar, 1U>{1.0e150}};
    VSS_CHECK(add(highDynamicRange.value(), negative.view(), 1.0e10).ok());
    VSS_CHECK(add(highDynamicRange.value(), positive.view(), 1.0e10).ok());
    const auto estimate =
        finalize(highDynamicRange.value(), WeightRule::empirical, limits);
    VSS_CHECK(estimate);
    if (estimate) {
      VSS_CHECK(close(estimate.value().terms.view()[0].values.at(0U), 1.0e300));
    }
  }

  auto downweightedOutlier = makeOnlineMoments({}, varianceOnly, limits);
  VSS_CHECK(downweightedOutlier);
  if (downweightedOutlier) {
    const Vector zero{std::array<Scalar, 1U>{0.0}};
    const Vector extreme{std::array<Scalar, 1U>{1.0e200}};
    VSS_CHECK(add(downweightedOutlier.value(), zero.view(), 1.0e200).ok());
    VSS_CHECK(add(downweightedOutlier.value(), extreme.view(), 1.0).ok());
    const auto estimate =
        finalize(downweightedOutlier.value(), WeightRule::empirical, limits);
    VSS_CHECK(estimate);
    if (estimate) {
      VSS_CHECK(close(estimate.value().terms.view()[0].values.at(0U), 1.0e200));
    }
  }
  auto dominantBlock = makeOnlineMoments({}, varianceOnly, limits);
  auto outlierBlock = makeOnlineMoments({}, varianceOnly, limits);
  VSS_CHECK(dominantBlock && outlierBlock);
  if (dominantBlock && outlierBlock) {
    const Vector zero{std::array<Scalar, 1U>{0.0}};
    const Vector extreme{std::array<Scalar, 1U>{1.0e200}};
    VSS_CHECK(add(dominantBlock.value(), zero.view(), 1.0e200).ok());
    VSS_CHECK(add(outlierBlock.value(), extreme.view(), 1.0).ok());
    VSS_CHECK(merge(dominantBlock.value(), outlierBlock.value()).ok());
    const auto estimate =
        finalize(dominantBlock.value(), WeightRule::empirical, limits);
    VSS_CHECK(estimate);
    if (estimate) {
      VSS_CHECK(close(estimate.value().terms.view()[0].values.at(0U), 1.0e200));
    }
  }

  auto overflowRollback = makeOnlineMoments({}, varianceOnly, limits);
  VSS_CHECK(overflowRollback);
  if (overflowRollback) {
    const Vector first{std::array<Scalar, 1U>{1.0}};
    const Vector second{std::array<Scalar, 1U>{2.0}};
    VSS_CHECK(add(overflowRollback.value(), first.view(),
                  std::numeric_limits<Scalar>::max())
                  .ok());
    VSS_CHECK(add(overflowRollback.value(), second.view(),
                  std::numeric_limits<Scalar>::max())
                  .code() == ErrorCode::NonFiniteResult);
    const auto unchanged =
        finalize(overflowRollback.value(), WeightRule::empirical, limits);
    checkPopulationMean(unchanged, 1.0);
    VSS_CHECK(unchanged &&
              unchanged.value().terms.view()[0].values.at(0U) == 0.0);
  }

  auto offset = makeOnlineMoments({}, request, limits);
  VSS_CHECK(offset);
  if (offset) {
    constexpr Scalar anchor = 1099511627776.0;
    for (const Scalar delta : std::array<Scalar, 4U>{-4.0, -2.0, 1.0, 5.0}) {
      const Vector sample{std::array<Scalar, 1U>{anchor + delta}};
      VSS_CHECK(add(offset.value(), sample.view(), 1.0).ok());
    }
    const auto estimate =
        finalize(offset.value(), WeightRule::empirical, limits);
    VSS_CHECK(estimate);
    if (estimate) {
      checkPopulationMean(estimate, anchor + 0.0);
      VSS_CHECK(
          close(estimate.value().terms.view()[0].values.at(0U), 23.0 / 2.0));
      VSS_CHECK(
          close(estimate.value().terms.view()[1].values.at(0U), 27.0 / 2.0));
      VSS_CHECK(
          close(estimate.value().terms.view()[2].values.at(0U), 449.0 / 2.0));
    }
  }

  auto left = makeOnlineMoments({}, request, limits);
  auto right = makeOnlineMoments({}, request, limits);
  VSS_CHECK(left && right);
  if (left && right) {
    constexpr Scalar largeAnchor = 1099511627776.0;
    for (const Scalar delta : std::array<Scalar, 4U>{-4.0, -2.0, 1.0, 5.0}) {
      const Vector sample{std::array<Scalar, 1U>{largeAnchor + delta}};
      auto& block = delta < 0.0 ? left.value() : right.value();
      VSS_CHECK(add(block, sample.view(), 1.0).ok());
    }
    VSS_CHECK(merge(left.value(), right.value()).ok());
    const auto merged = finalize(left.value(), WeightRule::empirical, limits);
    VSS_CHECK(merged);
    if (merged) {
      VSS_CHECK(
          close(merged.value().terms.view()[0].values.at(0U), 23.0 / 2.0));
      VSS_CHECK(
          close(merged.value().terms.view()[1].values.at(0U), 27.0 / 2.0));
      VSS_CHECK(
          close(merged.value().terms.view()[2].values.at(0U), 449.0 / 2.0));
    }
  }

  auto replayA = makeOnlineMoments({}, request, limits);
  auto replayA2 = makeOnlineMoments({}, request, limits);
  auto replayB = makeOnlineMoments({}, request, limits);
  auto replayC = makeOnlineMoments({}, request, limits);
  VSS_CHECK(replayA && replayA2 && replayB && replayC);
  if (replayA && replayA2 && replayB && replayC) {
    const std::array<Scalar, 6U> values{1.0, 2.0, 4.0, 8.0, 16.0, 32.0};
    for (Index i = 0U; i < values.size(); ++i) {
      const Vector sample{std::array<Scalar, 1U>{values.at(i)}};
      VSS_CHECK(add(replayA.value(), sample.view(), 1.0).ok());
      VSS_CHECK(add(replayA2.value(), sample.view(), 1.0).ok());
      VSS_CHECK(
          add((i < 3U ? replayB.value() : replayC.value()), sample.view(), 1.0)
              .ok());
    }
    const auto replayDirect =
        finalize(replayA.value(), WeightRule::empirical, limits);
    const auto replayRepeat =
        finalize(replayA2.value(), WeightRule::empirical, limits);
    VSS_CHECK(merge(replayB.value(), replayC.value()).ok());
    const auto replayMerged =
        finalize(replayB.value(), WeightRule::empirical, limits);
    VSS_CHECK(replayDirect && replayRepeat && replayMerged);
    if (replayDirect && replayRepeat && replayMerged) {
      for (Index term = 0U; term < 3U; ++term) {
        const Scalar actual =
            replayDirect.value().terms.view()[term].values.at(0U);
        const Scalar sameTree =
            replayRepeat.value().terms.view()[term].values.at(0U);
        const Scalar alternate =
            replayMerged.value().terms.view()[term].values.at(0U);
        VSS_CHECK(std::bit_cast<std::uint64_t>(actual) ==
                  std::bit_cast<std::uint64_t>(sameTree));
        VSS_CHECK(close(actual, alternate));
      }
    }
  }

  auto unitDestination = makeOnlineMoments({}, request, limits);
  auto zeroOnly = makeOnlineMoments({}, request, limits);
  VSS_CHECK(unitDestination && zeroOnly);
  if (unitDestination && zeroOnly) {
    const Vector sample{std::array<Scalar, 1U>{3.0}};
    VSS_CHECK(add(unitDestination.value(), sample.view(), 1.0).ok());
    VSS_CHECK(add(zeroOnly.value(), sample.view(), 0.0).ok());
    VSS_CHECK(merge(unitDestination.value(), zeroOnly.value()).ok());
    VSS_CHECK(finalize(unitDestination.value(), WeightRule::unweighted, limits)
                  .status()
                  .code() == ErrorCode::InvalidDomain);
  }

  auto compatible = makeOnlineMoments({}, request, limits);
  auto incompatible = makeOnlineMoments({}, secondOnly, limits);
  VSS_CHECK(compatible && incompatible);
  if (compatible && incompatible) {
    const Vector sample{std::array<Scalar, 1U>{9.0}};
    VSS_CHECK(add(compatible.value(), sample.view(), 1.0).ok());
    VSS_CHECK(add(incompatible.value(), sample.view(), 1.0).ok());
    VSS_CHECK(merge(compatible.value(), incompatible.value()).code() ==
              ErrorCode::DimensionMismatch);
    const auto unchanged =
        finalize(compatible.value(), WeightRule::empirical, limits);
    VSS_CHECK(unchanged &&
              unchanged.value().terms.view()[0].values.at(0U) == 0.0);
  }

  auto bounded = makeOnlineMoments({}, request, limits);
  VSS_CHECK(bounded);
  if (bounded) {
    const std::array<Index, 8U> slots{
        bounded.value().centered_sums.size(),
        bounded.value().scratch_sums.size(),
        bounded.value().mean.size(),
        bounded.value().anchor.size(),
        bounded.value().mean_offset.size(),
        bounded.value().tuple_orders.view().size(),
        bounded.value().tuple_indices.view().size(),
        bounded.value().stored_request.selected.view().size()};
    const Vector sample{std::array<Scalar, 1U>{1.0}};
    observedAllocations = 0U;
    bool accepted = true;
    countAllocations = true;
    for (Index i = 0U; i < 16U; ++i) {
      accepted = accepted && add(bounded.value(), sample.view(), 1.0).ok();
    }
    countAllocations = false;
    const std::size_t afterSixteen = observedAllocations;
    countAllocations = true;
    for (Index i = 16U; i < 4096U; ++i) {
      accepted = accepted && add(bounded.value(), sample.view(), 1.0).ok();
    }
    countAllocations = false;
    VSS_CHECK(accepted);
    VSS_CHECK(afterSixteen == 0U && observedAllocations == 0U);
    VSS_CHECK(bounded.value().centered_sums.size() == slots[0U]);
    VSS_CHECK(bounded.value().scratch_sums.size() == slots[1U]);
    VSS_CHECK(bounded.value().mean.size() == slots[2U]);
    VSS_CHECK(bounded.value().anchor.size() == slots[3U]);
    VSS_CHECK(bounded.value().mean_offset.size() == slots[4U]);
    VSS_CHECK(bounded.value().tuple_orders.view().size() == slots[5U]);
    VSS_CHECK(bounded.value().tuple_indices.view().size() == slots[6U]);
    VSS_CHECK(bounded.value().stored_request.selected.view().size() ==
              slots[7U]);
  }
}

void rollingCapacityEvictionAndReset() {
  constexpr Limits limits{
      .elements = 4096U, .bytes = 262144U, .evaluations = 0U};
  const MomentRequest request{.orders = 0x0fU,
                              .representation = MomentRepresentation::component,
                              .selected = {},
                              .contraction = std::nullopt};
  auto rolling = makeRollingMoments({}, 2U, request, limits);
  VSS_CHECK(rolling);
  if (!rolling) {
    return;
  }
  VSS_CHECK(evict(rolling.value()).code() == ErrorCode::InsufficientSamples);
  const std::array<Scalar, 3U> inputs{1.0, 2.0, 4.0};
  for (Index i = 0U; i < inputs.size(); ++i) {
    const Scalar value = inputs.at(i);
    const Vector observation{std::array<Scalar, 1U>{value}};
    VSS_CHECK(add(rolling.value(), observation.view(), 1.0).ok());
    const auto current =
        finalize(rolling.value(), WeightRule::empirical, limits);
    VSS_CHECK(current);
    if (current) {
      constexpr std::array<Scalar, 3U> expectedMeans{1.0, 1.5, 3.0};
      constexpr std::array<Scalar, 3U> expectedVariances{0.0, 0.25, 1.0};
      const Scalar expectedMean = expectedMeans.at(i);
      const Scalar expectedVariance = expectedVariances.at(i);
      checkPopulationMean(current, expectedMean);
      VSS_CHECK(close(current.value().terms.view()[1].values.at(0U),
                      expectedVariance));
    }
  }
  auto estimate = finalize(rolling.value(), WeightRule::empirical, limits);
  VSS_CHECK(estimate);
  if (estimate) {
    checkPopulationMean(estimate, 3.0);
    VSS_CHECK(close(estimate.value().terms.view()[1].values.at(0U), 1.0));
    VSS_CHECK(close(estimate.value().terms.view()[2].values.at(0U), 0.0));
    VSS_CHECK(close(estimate.value().terms.view()[3].values.at(0U), 1.0));
  }
  const Vector invalid{std::array<Scalar, 2U>{9.0, 10.0}};
  VSS_CHECK(add(rolling.value(), invalid.view(), 1.0).code() ==
            ErrorCode::DimensionMismatch);
  VSS_CHECK(add(rolling.value(), invalid.view(), -1.0).code() ==
            ErrorCode::DimensionMismatch);
  estimate = finalize(rolling.value(), WeightRule::empirical, limits);
  checkPopulationMean(estimate, 3.0);
  VSS_CHECK(estimate &&
            close(estimate.value().terms.view()[1].values.at(0U), 1.0));
  VSS_CHECK(evict(rolling.value()).ok());
  estimate = finalize(rolling.value(), WeightRule::empirical, limits);
  checkPopulationMean(estimate, 4.0);
  VSS_CHECK(close(estimate.value().terms.view()[1].values.at(0U), 0.0));
  observedAllocations = 0U;
  countAllocations = true;
  const Status rollingReset = reset(rolling.value());
  countAllocations = false;
  VSS_CHECK(rollingReset.ok());
  VSS_CHECK(observedAllocations == 0U);
  VSS_CHECK(finalize(rolling.value(), WeightRule::empirical, limits)
                .status()
                .code() == ErrorCode::InsufficientSamples);
  const Vector seven{std::array<Scalar, 1U>{7.0}};
  const Vector one{std::array<Scalar, 1U>{1.0}};
  VSS_CHECK(add(rolling.value(), seven.view(), 1.0).ok());
  VSS_CHECK(add(rolling.value(), one.view(), 3.0).ok());
  estimate = finalize(rolling.value(), WeightRule::empirical, limits);
  VSS_CHECK(estimate &&
            close(estimate.value().terms.view()[1].values.at(0U), 27.0 / 4.0));

  VSS_CHECK(makeRollingMoments({}, 0U, request, limits).status().code() ==
            ErrorCode::InvalidExtent);
  VSS_CHECK(
      makeRollingMoments({}, 2U, request, Limits{0U, 0U, 0U}).status().code() ==
      ErrorCode::ResourceLimit);
  const Vector scalar{std::array<Scalar, 1U>{5.0}};
  VSS_CHECK(add(rolling.value(), scalar.view(), -1.0).code() ==
            ErrorCode::InvalidDomain);

  auto wrapped = makeRollingMoments({}, 2U, request, limits);
  VSS_CHECK(wrapped);
  if (wrapped) {
    for (const Scalar value : std::array<Scalar, 4U>{1.0, 2.0, 4.0, 8.0}) {
      const Vector observation{std::array<Scalar, 1U>{value}};
      VSS_CHECK(add(wrapped.value(), observation.view(), 1.0).ok());
    }
    estimate = finalize(wrapped.value(), WeightRule::empirical, limits);
    checkPopulationMean(estimate, 6.0);
    VSS_CHECK(estimate &&
              close(estimate.value().terms.view()[1].values.at(0U), 4.0));
    VSS_CHECK(evict(wrapped.value()).ok());
    estimate = finalize(wrapped.value(), WeightRule::empirical, limits);
    checkPopulationMean(estimate, 8.0);
  }

  auto negativeWeight = makeRollingMoments({}, 2U, request, limits);
  VSS_CHECK(negativeWeight);
  if (negativeWeight) {
    const Vector observation{std::array<Scalar, 1U>{5.0}};
    VSS_CHECK(add(negativeWeight.value(), observation.view(), -1.0).code() ==
              ErrorCode::InvalidDomain);
  }

  auto zeroWeight = makeRollingMoments({}, 2U, request, limits);
  VSS_CHECK(zeroWeight);
  if (zeroWeight) {
    const Vector observation{std::array<Scalar, 1U>{2.0}};
    VSS_CHECK(add(zeroWeight.value(), observation.view(), 0.0).ok());
    VSS_CHECK(finalize(zeroWeight.value(), WeightRule::empirical, limits)
                  .status()
                  .code() == ErrorCode::InsufficientSamples);
    VSS_CHECK(evict(zeroWeight.value()).ok());
    VSS_CHECK(evict(zeroWeight.value()).code() ==
              ErrorCode::InsufficientSamples);
  }

  auto recovery = makeRollingMoments({}, 2U, request, limits);
  VSS_CHECK(recovery);
  if (recovery) {
    for (const auto& [value, weight] :
         std::array<std::pair<Scalar, Scalar>, 3U>{
             std::pair<Scalar, Scalar>{1.0e12, 1.0e12},
             std::pair<Scalar, Scalar>{1.0, 1.0},
             std::pair<Scalar, Scalar>{2.0, 1.0}}) {
      const Vector observation{std::array<Scalar, 1U>{value}};
      VSS_CHECK(add(recovery.value(), observation.view(), weight).ok());
    }
    estimate = finalize(recovery.value(), WeightRule::empirical, limits);
    checkPopulationMean(estimate, 1.5);
    VSS_CHECK(estimate &&
              close(estimate.value().terms.view()[1].values.at(0U), 0.25));
  }
}

void shapedOnlineMatchesBatch() {
  constexpr Limits limits{
      .elements = 4096U, .bytes = 262144U, .evaluations = 0U};
  const std::array<Index, 1U> shape{2U};
  Matrix sample{3U, 2U};
  const std::array<Scalar, 6U> cells{1.0, 2.0, 2.0, 0.0, 4.0, 3.0};
  for (Index row = 0U; row < 3U; ++row) {
    for (Index column = 0U; column < 2U; ++column) {
      sample.at(row, column) = cells.at((row * 2U) + column);
    }
  }
  const MomentRequest request{.orders = 0x0eU,
                              .representation = MomentRepresentation::dense,
                              .selected = {},
                              .contraction = std::nullopt};
  auto online = makeOnlineMoments(shape, request, limits);
  VSS_CHECK(online);
  if (!online) {
    return;
  }
  for (Index row = 0U; row < sample.rows(); ++row) {
    const Vector values{
        std::array<Scalar, 2U>{sample.at(row, 0U), sample.at(row, 1U)}};
    VSS_CHECK(add(online.value(), values.view(), 1.0).ok());
  }
  const auto streamed = finalize(online.value(), WeightRule::empirical, limits);
  const auto batch = estimateCentralMoments(
      SampleView{
          .rows = sample.view(), .observation_shape = shape, .valid = {}},
      WeightSpec{.rule = WeightRule::unweighted,
                 .rowWeights = std::nullopt,
                 .componentWeights = std::nullopt,
                 .frequencyCounts = {}},
      CenterSpec{}, request, limits);
  VSS_CHECK(streamed && batch);
  if (streamed && batch) {
    VSS_CHECK(streamed.value().observation_shape.view().size() == 1U);
    VSS_CHECK(streamed.value().observation_shape.view()[0U] == 2U);
    VSS_CHECK(streamed.value().terms.view().size() ==
              batch.value().terms.view().size());
    VSS_CHECK(streamed.value().diagnostics.global.input ==
              batch.value().diagnostics.global.input);
    VSS_CHECK(streamed.value().diagnostics.global.valid ==
              batch.value().diagnostics.global.valid);
    VSS_CHECK(streamed.value().diagnostics.global.positive ==
              batch.value().diagnostics.global.positive);
    VSS_CHECK(streamed.value().diagnostics.global.weight_mass ==
              batch.value().diagnostics.global.weight_mass);
    for (Index power = 0U; power < 3U; ++power) {
      const auto actual =
          streamed.value().diagnostics.global.normalized_weight_powers.at(
              power);
      const auto expected =
          batch.value().diagnostics.global.normalized_weight_powers.at(power);
      VSS_CHECK(actual.has_value() == expected.has_value());
      if (actual && expected) {
        VSS_CHECK(close(*actual, *expected));
      }
    }
    for (Index term = 0U; term < streamed.value().terms.view().size(); ++term) {
      const auto actual = streamed.value().terms.view()[term].values.view();
      const auto expected = batch.value().terms.view()[term].values.view();
      VSS_CHECK(actual.size() == expected.size());
      for (Index value = 0U; value < actual.size(); ++value) {
        VSS_CHECK(close(actual.at(value), expected.at(value)));
      }
    }
  }

  auto projected = [&]() {
    Matrix map{1U, 2U};
    map.at(0U, 0U) = 1.0;
    map.at(0U, 1U) = 1.0;
    const MomentRequest projectedRequest{.orders = 0x02U,
                                         .representation =
                                             MomentRepresentation::contraction,
                                         .selected = {},
                                         .contraction = map.view()};
    return makeOnlineMoments(shape, projectedRequest, limits);
  }();
  VSS_CHECK(projected);
  if (!projected) {
    return;
  }
  for (Index row = 0U; row < sample.rows(); ++row) {
    const Vector values{
        std::array<Scalar, 2U>{sample.at(row, 0U), sample.at(row, 1U)}};
    VSS_CHECK(add(projected.value(), values.view(), 1.0).ok());
  }
  const auto projectedEstimate =
      finalize(projected.value(), WeightRule::empirical, limits);
  VSS_CHECK(projectedEstimate);
  if (projectedEstimate) {
    VSS_CHECK(projectedEstimate.value().observation_shape.view().size() == 1U);
    VSS_CHECK(projectedEstimate.value().observation_shape.view()[0U] == 1U);
    VSS_CHECK(projectedEstimate.value().terms.view()[0].layout ==
              MomentRepresentation::dense);
    VSS_CHECK(close(projectedEstimate.value().terms.view()[0].values.at(0U),
                    14.0 / 3.0));
  }

  auto selected = [&]() {
    const std::array<Index, 2U> tuple{0U, 1U};
    const std::array<std::span<const Index>, 1U> tuples{tuple};
    const MomentRequest selectedRequest{.orders = 0x02U,
                                        .representation =
                                            MomentRepresentation::selected,
                                        .selected = tuples,
                                        .contraction = std::nullopt};
    return makeOnlineMoments(shape, selectedRequest, limits);
  }();
  VSS_CHECK(selected);
  if (selected) {
    for (Index row = 0U; row < sample.rows(); ++row) {
      const Vector values{
          std::array<Scalar, 2U>{sample.at(row, 0U), sample.at(row, 1U)}};
      VSS_CHECK(add(selected.value(), values.view(), 1.0).ok());
    }
    const auto selectedEstimate =
        finalize(selected.value(), WeightRule::empirical, limits);
    VSS_CHECK(selectedEstimate);
    if (selectedEstimate) {
      VSS_CHECK(selectedEstimate.value().terms.view().size() == 1U);
      VSS_CHECK(selectedEstimate.value()
                    .terms.view()[0]
                    .flattened_tuples.view()
                    .size() == 2U);
      VSS_CHECK(close(selectedEstimate.value().terms.view()[0].values.at(0U),
                      7.0 / 9.0));
    }
  }
}

} // namespace

int main() {
  try {
    emptyAndMergedOnlineMoments();
    weightedOffsetAndReset();
    rollingCapacityEvictionAndReset();
    shapedOnlineMatchesBatch();
    return 0;
  } catch (const std::exception&) {
    return 1;
  }
}
