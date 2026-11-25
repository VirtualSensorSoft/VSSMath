#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <string>
#include <vector>
#include <vss/math/Error.hpp>
#include <vss/math/statistics/Order.hpp>
#include <vss/math/statistics/QuantileState.hpp>

namespace vss::math::statistics {
namespace {

Status invalid(const char* message) {
  return Status::failure(ErrorCode::InvalidDomain, message);
}

bool thresholdFor(Scalar epsilon, Index count, Index& threshold) {
  const long double value = 2.0L * static_cast<long double>(epsilon) *
                            static_cast<long double>(count);
  if (!std::isfinite(value) ||
      value > static_cast<long double>(std::numeric_limits<Index>::max())) {
    return false;
  }
  threshold = static_cast<Index>(std::floor(value));
  return true;
}

Status validateState(const QuantileState& state) {
  if (!std::isfinite(state.epsilon) || state.epsilon <= 0.0 ||
      state.epsilon >= 0.5 || state.capacity == 0U ||
      (state.count == 0U && !state.markers.view().empty()) ||
      state.capacity > state.limits.elements ||
      state.capacity > state.limits.bytes / sizeof(QuantileMarker)) {
    return invalid("quantile sketch configuration or empty state is malformed");
  }
  const auto markers = state.markers.view();
  if (markers.size() > state.capacity ||
      markers.size() > state.limits.elements ||
      markers.size() > state.limits.bytes / sizeof(QuantileMarker) ||
      (state.count == 0U) != markers.empty()) {
    return invalid("quantile sketch marker storage is inconsistent");
  }
  if (state.count == 0U) {
    return Status::success();
  }
  Index total = 0U;
  for (Index i = 0U; i < markers.size(); ++i) {
    const QuantileMarker& marker = markers[i];
    if (!std::isfinite(marker.value) || marker.count == 0U ||
        marker.count > std::numeric_limits<Index>::max() - total ||
        (i != 0U && markers[i - 1U].value > marker.value)) {
      return invalid(
          "quantile sketch tuple ordering or rank counts are malformed");
    }
    total += marker.count;
  }
  if (total != state.count || markers.front().slack != 0U ||
      markers.back().slack != 0U || markers.front().count != 1U ||
      markers.back().count != 1U) {
    return invalid("quantile sketch extrema or total rank count is malformed");
  }
  if (static_cast<long double>(state.epsilon) * state.count < 1.0L) {
    if (markers.size() != state.count) {
      return invalid("exact-mode quantile sketch discarded source ranks");
    }
    for (const QuantileMarker& marker : markers) {
      if (marker.count != 1U || marker.slack != 0U) {
        return invalid(
            "exact-mode quantile markers must have unit exact ranks");
      }
    }
    return Status::success();
  }
  Index threshold = 0U;
  if (!thresholdFor(state.epsilon, state.count, threshold)) {
    return invalid("quantile sketch rank threshold is not representable");
  }
  for (Index i = 1U; i + 1U < markers.size(); ++i) {
    if (markers[i].slack >
            std::numeric_limits<Index>::max() - markers[i].count ||
        markers[i].count + markers[i].slack > threshold) {
      return invalid("quantile sketch tuple violates its rank invariant");
    }
  }
  return Status::success();
}

Index targetRank(Scalar probability, Index count) {
  if (probability <= 0.0) {
    return 1U;
  }
  if (probability >= 1.0) {
    return count;
  }
  const long double raw =
      std::ceil(static_cast<long double>(probability) * count);
  return std::max<Index>(1U, static_cast<Index>(raw));
}

} // namespace

Result<QuantileState> makeStreamingQuantile(Scalar epsilon, Index capacity,
                                            Limits limits) {
  if (!std::isfinite(epsilon) || epsilon <= 0.0 || epsilon >= 0.5 ||
      capacity == 0U) {
    return Result<QuantileState>::failure(
        ErrorCode::InvalidDomain,
        "quantile sketch requires 0<epsilon<0.5 and positive capacity");
  }
  if (capacity > limits.elements ||
      capacity > limits.bytes / sizeof(QuantileMarker)) {
    return Result<QuantileState>::failure(
        ErrorCode::ResourceLimit,
        "quantile sketch capacity exceeds configured storage limits");
  }
  return Result<QuantileState>::success(QuantileState{.markers = {},
                                                      .epsilon = epsilon,
                                                      .capacity = capacity,
                                                      .count = 0U,
                                                      .limits = limits});
}

Status add(QuantileState& state, Scalar observation) {
  const Status valid = validateState(state);
  if (!valid.ok()) {
    return valid;
  }
  if (!std::isfinite(observation)) {
    return Status::failure(ErrorCode::NonFiniteInput,
                           "quantile sketch observations must be finite");
  }
  if (state.count == std::numeric_limits<Index>::max()) {
    return Status::failure(ErrorCode::SizeOverflow,
                           "quantile sketch observation count overflows");
  }
  const auto current = state.markers.view();
  const Index candidateSlots = current.size() + 1U;
  if (candidateSlots > state.limits.bytes / (2U * sizeof(QuantileMarker))) {
    return Status::failure(
        ErrorCode::ResourceLimit,
        "transactional quantile update workspace exceeds limits");
  }

  std::vector<QuantileMarker> candidate(current.begin(), current.end());
  const auto insertion =
      std::upper_bound(candidate.begin(), candidate.end(), observation,
                       [](Scalar value, const QuantileMarker& marker) {
                         return value < marker.value;
                       });
  const Index position = static_cast<Index>(insertion - candidate.begin());
  const Index newCount = state.count + 1U;
  const bool exact = static_cast<long double>(state.epsilon) * newCount < 1.0L;
  Index slack = 0U;
  if (!exact && position != 0U && position != candidate.size()) {
    Index oldThreshold = 0U;
    if (!thresholdFor(state.epsilon, state.count, oldThreshold)) {
      return Status::failure(ErrorCode::SizeOverflow,
                             "quantile sketch insertion threshold overflows");
    }
    slack = oldThreshold > 0U ? oldThreshold - 1U : 0U;
  }
  candidate.insert(
      candidate.begin() + static_cast<std::ptrdiff_t>(position),
      QuantileMarker{.value = observation, .count = 1U, .slack = slack});

  if (!exact && candidate.size() > 2U) {
    Index threshold = 0U;
    if (!thresholdFor(state.epsilon, newCount, threshold)) {
      return Status::failure(ErrorCode::SizeOverflow,
                             "quantile sketch compression threshold overflows");
    }
    for (Index i = candidate.size() - 2U; i > 0U; --i) {
      if (i + 1U >= candidate.size() - 1U) {
        continue;
      }
      const Index next = i + 1U;
      const auto& left = candidate[i];
      const auto& right = candidate[next];
      if (left.count <= std::numeric_limits<Index>::max() - right.count &&
          left.count + right.count <=
              std::numeric_limits<Index>::max() - right.slack &&
          left.count + right.count + right.slack <= threshold) {
        candidate[next].count += left.count;
        candidate.erase(candidate.begin() + static_cast<std::ptrdiff_t>(i));
      }
    }
  }
  if (candidate.size() > state.capacity ||
      candidate.size() > state.limits.elements ||
      candidate.size() > state.limits.bytes / sizeof(QuantileMarker)) {
    return Status::failure(
        ErrorCode::ResourceLimit,
        "quantile sketch capacity cannot preserve its rank guarantee");
  }
  auto owned = Array<QuantileMarker>::copy(candidate, state.limits);
  if (!owned) {
    return owned.status();
  }
  QuantileState updated{.markers = std::move(owned).value(),
                        .epsilon = state.epsilon,
                        .capacity = state.capacity,
                        .count = newCount,
                        .limits = state.limits};
  const Status invariant = validateState(updated);
  if (!invariant.ok()) {
    return invariant;
  }
  state = std::move(updated);
  return Status::success();
}

Result<QuantileBound> streamingQuantile(const QuantileState& state,
                                        Scalar probability) {
  const Status valid = validateState(state);
  if (!valid.ok()) {
    return Result<QuantileBound>::failure(valid.code(),
                                          std::string(valid.message()));
  }
  if (!std::isfinite(probability) || probability < 0.0 || probability > 1.0) {
    return Result<QuantileBound>::failure(
        ErrorCode::InvalidDomain,
        "quantile probability must be finite and within [0,1]");
  }
  if (state.count == 0U) {
    return Result<QuantileBound>::failure(
        ErrorCode::InsufficientSamples,
        "quantile sketch has no observations to query");
  }
  const auto markers = state.markers.view();
  if (probability == 0.0) {
    return Result<QuantileBound>::success(QuantileBound{
        .value = markers.front().value, .lower_rank = 1U, .upper_rank = 1U});
  }
  if (probability == 1.0) {
    return Result<QuantileBound>::success(
        QuantileBound{.value = markers.back().value,
                      .lower_rank = state.count,
                      .upper_rank = state.count});
  }
  const Index target = targetRank(probability, state.count);
  if (static_cast<long double>(state.epsilon) * state.count < 1.0L) {
    Index rank = 0U;
    for (const QuantileMarker& marker : markers) {
      rank += marker.count;
      if (rank >= target) {
        return Result<QuantileBound>::success(QuantileBound{
            .value = marker.value, .lower_rank = rank, .upper_rank = rank});
      }
    }
    return Result<QuantileBound>::failure(
        ErrorCode::InvalidDomain,
        "exact quantile sketch ranks do not cover the target");
  }
  const long double error =
      static_cast<long double>(state.epsilon) * state.count;
  Index minimumRank = 0U;
  for (Index i = 0U; i < markers.size(); ++i) {
    minimumRank += markers[i].count;
    if (static_cast<long double>(minimumRank) +
            static_cast<long double>(markers[i].slack) >
        static_cast<long double>(target) + error) {
      const Index selected = i - 1U;
      Index lowerRank = 0U;
      for (Index j = 0U; j <= selected; ++j) {
        lowerRank += markers[j].count;
      }
      const Index upperRank = markers[selected].slack > state.count - lowerRank
                                  ? state.count
                                  : lowerRank + markers[selected].slack;
      return Result<QuantileBound>::success(
          QuantileBound{.value = markers[selected].value,
                        .lower_rank = lowerRank,
                        .upper_rank = upperRank});
    }
  }
  return Result<QuantileBound>::success(
      QuantileBound{.value = markers.back().value,
                    .lower_rank = state.count,
                    .upper_rank = state.count});
}

Result<Scalar> rollingQuantile(ConstVectorView finiteWindow, Scalar probability,
                               QuantileRule rule, Limits limits) {
  return quantile(finiteWindow, probability, rule, limits);
}

} // namespace vss::math::statistics
