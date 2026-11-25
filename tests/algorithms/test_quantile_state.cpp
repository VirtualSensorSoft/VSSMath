#include "Test.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <span>
#include <utility>
#include <vector>
#include <vss/math/Core.hpp>
#include <vss/math/Error.hpp>
#include <vss/math/Limits.hpp>
#include <vss/math/Vector.hpp>
#include <vss/math/statistics/Order.hpp>
#include <vss/math/statistics/QuantileState.hpp>

using namespace vss::math;
using namespace vss::math::statistics;

namespace {

Vector vector(std::initializer_list<Scalar> values) {
  return Vector(std::span<const Scalar>(values.begin(), values.size()));
}

Limits ample() {
  return {.elements = std::numeric_limits<std::size_t>::max(),
          .bytes = std::numeric_limits<std::size_t>::max(),
          .evaluations = std::numeric_limits<std::uint64_t>::max()};
}

void checkState(const QuantileState& state, const std::vector<Scalar>& values,
                Scalar epsilon) {
  VSS_CHECK(state.count == values.size());
  VSS_CHECK(state.epsilon == epsilon);
  VSS_CHECK(state.markers.view().size() <= state.capacity);
  const auto markers = state.markers.view();
  Index total = 0U;
  for (Index i = 0U; i < markers.size(); ++i) {
    const auto& marker = markers[i];
    VSS_CHECK(marker.count > 0U);
    VSS_CHECK(std::isfinite(marker.value));
    if (i != 0U) {
      VSS_CHECK(markers[i - 1U].value <= marker.value);
    }
    total += marker.count;
  }
  VSS_CHECK(total == state.count);
  if (!values.empty()) {
    VSS_CHECK(markers.front().value ==
              *std::min_element(values.begin(), values.end()));
    VSS_CHECK(markers.back().value ==
              *std::max_element(values.begin(), values.end()));
  }
}

void checkEveryRank(const QuantileState& state,
                    const std::vector<Scalar>& values, Scalar epsilon) {
  std::vector<Scalar> sorted = values;
  std::stable_sort(sorted.begin(), sorted.end());
  const Index n = sorted.size();
  for (Index target = 1U; target <= n; ++target) {
    const Scalar p =
        (static_cast<Scalar>(target) - 0.5) / static_cast<Scalar>(n);
    const auto result = streamingQuantile(state, p);
    VSS_CHECK(result.hasValue());
    const auto bound = result.value();
    VSS_CHECK(bound.lower_rank >= 1U);
    VSS_CHECK(bound.lower_rank <= bound.upper_rank);
    VSS_CHECK(bound.upper_rank <= n);
    const Index k = p == 1.0 ? n : target;
    const auto tolerance =
        static_cast<Index>(std::floor(epsilon * static_cast<Scalar>(n)));
    bool certified = false;
    for (Index rank = 1U; rank <= n; ++rank) {
      if (sorted[rank - 1U] == bound.value && rank >= bound.lower_rank &&
          rank <= bound.upper_rank &&
          (rank > k ? rank - k : k - rank) <= tolerance) {
        certified = true;
        break;
      }
    }
    VSS_CHECK(certified);
  }
  const auto minimum = streamingQuantile(state, 0.0);
  const auto maximum = streamingQuantile(state, 1.0);
  VSS_CHECK(minimum.hasValue());
  VSS_CHECK(maximum.hasValue());
  VSS_CHECK(minimum.value().value == sorted.front());
  VSS_CHECK(minimum.value().lower_rank == 1U);
  VSS_CHECK(minimum.value().upper_rank == 1U);
  VSS_CHECK(maximum.value().value == sorted.back());
  VSS_CHECK(maximum.value().lower_rank == n);
  VSS_CHECK(maximum.value().upper_rank == n);
}

void deterministicSketchHonorsRankBoundsForAdversarialPrefixes() {
  std::array<std::vector<Scalar>, 5U> streams;
  for (Index i = 0U; i < 80U; ++i) {
    streams[0U].push_back(static_cast<Scalar>(i));
    streams[1U].push_back(static_cast<Scalar>(79U - i));
    streams[2U].push_back(i % 2U == 0U ? static_cast<Scalar>(i)
                                       : -static_cast<Scalar>(i));
    streams[3U].push_back(static_cast<Scalar>(i % 3U));
    Scalar value = 0.0;
    if (i % 4U == 0U) {
      value = 1.0;
    } else if (i % 4U == 1U) {
      value = -1.0;
    }
    streams[4U].push_back(value);
  }
  for (const Scalar epsilon : {0.01, 0.1, 0.49}) {
    for (const auto& stream : streams) {
      auto created = makeStreamingQuantile(epsilon, 256U, ample());
      VSS_CHECK(created.hasValue());
      QuantileState state = std::move(created).value();
      std::vector<Scalar> values;
      for (const Scalar next : stream) {
        values.push_back(next);
        const Status added = add(state, next);
        VSS_CHECK(added.ok());
        checkState(state, values, epsilon);
        checkEveryRank(state, values, epsilon);
      }
      VSS_CHECK(state.markers.view().size() <= state.count);
    }
  }
}

void exactSmallStreamsAndRollingQueriesMatchTheOrderStatisticApi() {
  auto created = makeStreamingQuantile(0.1, 32U, ample());
  VSS_CHECK(created.hasValue());
  QuantileState state = std::move(created).value();
  const std::array<Scalar, 5U> stream{3.0, 1.0, 2.0, 2.0, -1.0};
  for (const Scalar value : stream) {
    VSS_CHECK(add(state, value).ok());
  }
  checkEveryRank(state, std::vector<Scalar>(stream.begin(), stream.end()), 0.1);
  const Vector window = vector({3.0, 1.0, 2.0, 2.0, -1.0});
  for (const QuantileRule rule :
       {QuantileRule::linear, QuantileRule::inverse_empirical}) {
    const auto rolling = rollingQuantile(window.view(), 0.25, rule, ample());
    const auto exact = quantile(window.view(), 0.25, rule, ample());
    VSS_CHECK(rolling.hasValue());
    VSS_CHECK(exact.hasValue());
    VSS_CHECK(rolling.value() == exact.value());
  }
}

void invalidQueriesAndCapacityFailuresAreTransactional() {
  auto created = makeStreamingQuantile(0.1, 1U, ample());
  VSS_CHECK(created.hasValue());
  QuantileState state = std::move(created).value();
  VSS_CHECK(add(state, 4.0).ok());
  const auto before = state.markers.view();
  const Scalar firstValue = before.front().value;
  const Index firstCount = state.count;
  const Status exhausted = add(state, 5.0);
  VSS_CHECK(!exhausted.ok());
  VSS_CHECK(exhausted.code() == ErrorCode::ResourceLimit);
  VSS_CHECK(state.count == firstCount);
  VSS_CHECK(state.markers.view().size() == 1U);
  VSS_CHECK(state.markers.view().front().value == firstValue);

  const auto invalidProbability = streamingQuantile(state, 1.1);
  VSS_CHECK(!invalidProbability.hasValue());
  VSS_CHECK(invalidProbability.status().code() == ErrorCode::InvalidDomain);
  const Status nonfinite = add(state, std::numeric_limits<Scalar>::infinity());
  VSS_CHECK(!nonfinite.ok());
  VSS_CHECK(nonfinite.code() == ErrorCode::NonFiniteInput);
  VSS_CHECK(state.count == firstCount);

  auto badEpsilon = makeStreamingQuantile(0.0, 2U, ample());
  VSS_CHECK(!badEpsilon.hasValue());
  VSS_CHECK(badEpsilon.status().code() == ErrorCode::InvalidDomain);
}

} // namespace

int main() {
  deterministicSketchHonorsRankBoundsForAdversarialPrefixes();
  exactSmallStreamsAndRollingQueriesMatchTheOrderStatisticApi();
  invalidQueriesAndCapacityFailuresAreTransactional();
}
