#include <algorithm>
#include <cmath>
#include <cstddef>
#include <detail/CompensatedSum.hpp>
#include <limits>
#include <new>
#include <string>
#include <vector>
#include <vss/math/Error.hpp>
#include <vss/math/statistics/Order.hpp>

namespace vss::math::statistics {
namespace {

template <class T> Result<T> failure(ErrorCode code, const char* message) {
  return Result<T>::failure(code, message);
}

bool validRule(QuantileRule rule) noexcept {
  return rule == QuantileRule::linear ||
         rule == QuantileRule::inverse_empirical;
}

Status validateExtent(ConstVectorView values) {
  if (values.empty()) {
    return Status::failure(ErrorCode::InsufficientSamples,
                           "order statistic requires at least one value");
  }
  return Status::success();
}

Status validateFinite(ConstVectorView values) {
  for (Index index = 0U; index < values.size(); ++index) {
    if (!std::isfinite(values.at(index))) {
      return Status::failure(ErrorCode::NonFiniteInput,
                             "order statistic input is non-finite at index " +
                                 std::to_string(index));
    }
  }
  return Status::success();
}

Status preflightScratch(std::size_t count, Limits limits) {
  constexpr std::size_t scalarBytes = sizeof(Scalar);
  if (count > (std::numeric_limits<std::size_t>::max() - 1U) / 2U) {
    return Status::failure(ErrorCode::SizeOverflow,
                           "order statistic scratch extent overflows");
  }
  const std::size_t elements = (2U * count) + 1U;
  if (elements > std::numeric_limits<std::size_t>::max() / scalarBytes) {
    return Status::failure(ErrorCode::SizeOverflow,
                           "order statistic scratch byte count overflows");
  }
  const std::size_t bytes = elements * scalarBytes;
  if (elements > limits.elements || bytes > limits.bytes) {
    return Status::failure(ErrorCode::ResourceLimit,
                           "order statistic exceeds the supplied limits");
  }
  return Status::success();
}

Result<std::vector<Scalar>> sortedCopy(ConstVectorView values, Limits limits) {
  const Status preflight = preflightScratch(values.size(), limits);
  if (!preflight.ok()) {
    return Result<std::vector<Scalar>>::failure(
        preflight.code(), std::string(preflight.message()));
  }
  try {
    std::vector<Scalar> sorted;
    sorted.reserve(values.size());
    for (Index index = 0U; index < values.size(); ++index) {
      sorted.push_back(values.at(index));
    }
    std::stable_sort(sorted.begin(), sorted.end());
    return Result<std::vector<Scalar>>::success(std::move(sorted));
  } catch (const std::bad_alloc&) {
    return Result<std::vector<Scalar>>::failure(
        ErrorCode::ResourceLimit,
        "order statistic could not allocate its bounded scratch");
  }
}

Scalar linearQuantile(const std::vector<Scalar>& sorted, Scalar probability) {
  if (probability == 0.0) {
    return sorted.front();
  }
  if (probability == 1.0) {
    return sorted.back();
  }
  const Scalar rank = static_cast<Scalar>(sorted.size() - 1U) * probability;
  const auto lower = static_cast<std::size_t>(std::floor(rank));
  const auto upper = std::min(lower + 1U, sorted.size() - 1U);
  return std::lerp(sorted[lower], sorted[upper],
                   rank - static_cast<Scalar>(lower));
}

Scalar inverseEmpiricalQuantile(const std::vector<Scalar>& sorted,
                                Scalar probability) {
  if (probability == 0.0) {
    return sorted.front();
  }
  const Scalar cdfRank = probability * static_cast<Scalar>(sorted.size());
  if (cdfRank <= 1.0) {
    return sorted.front();
  }
  const auto oneBasedRank = static_cast<std::size_t>(std::ceil(cdfRank));
  return sorted[std::min(oneBasedRank - 1U, sorted.size() - 1U)];
}

Scalar meanOfSorted(const std::vector<Scalar>& sorted, std::size_t first,
                    std::size_t last) {
  Scalar scale = 0.0;
  for (std::size_t index = first; index < last; ++index) {
    scale = std::max(scale, std::abs(sorted[index]));
  }
  if (scale == 0.0) {
    return 0.0;
  }
  detail::CompensatedSum sum;
  for (std::size_t index = first; index < last; ++index) {
    sum.add(sorted[index] / scale);
  }
  return scale * (sum.total() / static_cast<Scalar>(last - first));
}

Result<Scalar> sortedMean(ConstVectorView values, Scalar fraction,
                          Limits limits, bool winsorize) {
  const Status extent = validateExtent(values);
  if (!extent.ok()) {
    return Result<Scalar>::failure(extent.code(),
                                   std::string(extent.message()));
  }
  if (!std::isfinite(fraction) || fraction < 0.0 || fraction >= 0.5) {
    return failure<Scalar>(ErrorCode::InvalidDomain,
                           "tail fraction must be finite and in [0, 1/2)");
  }
  const Status validation = validateFinite(values);
  if (!validation.ok()) {
    return Result<Scalar>::failure(validation.code(),
                                   std::string(validation.message()));
  }
  auto sortedResult = sortedCopy(values, limits);
  if (!sortedResult.hasValue()) {
    return Result<Scalar>::failure(
        sortedResult.status().code(),
        std::string(sortedResult.status().message()));
  }
  const auto& sorted = sortedResult.value();
  const auto trimCount = static_cast<std::size_t>(
      std::floor(static_cast<long double>(sorted.size()) *
                 static_cast<long double>(fraction)));
  if (!winsorize) {
    const Scalar result =
        meanOfSorted(sorted, trimCount, sorted.size() - trimCount);
    if (!std::isfinite(result)) {
      return failure<Scalar>(ErrorCode::NonFiniteResult,
                             "trimmed mean is not representable");
    }
    return Result<Scalar>::success(result);
  }

  Scalar scale = 0.0;
  for (std::size_t index = 0U; index < sorted.size(); ++index) {
    const Scalar value = index < trimCount
                             ? sorted[trimCount]
                             : (index >= sorted.size() - trimCount
                                    ? sorted[sorted.size() - trimCount - 1U]
                                    : sorted[index]);
    scale = std::max(scale, std::abs(value));
  }
  if (scale == 0.0) {
    return Result<Scalar>::success(0.0);
  }
  detail::CompensatedSum sum;
  for (std::size_t index = 0U; index < sorted.size(); ++index) {
    const Scalar value = index < trimCount
                             ? sorted[trimCount]
                             : (index >= sorted.size() - trimCount
                                    ? sorted[sorted.size() - trimCount - 1U]
                                    : sorted[index]);
    sum.add(value / scale);
  }
  const Scalar result =
      scale * (sum.total() / static_cast<Scalar>(sorted.size()));
  if (!std::isfinite(result)) {
    return failure<Scalar>(ErrorCode::NonFiniteResult,
                           "winsorized mean is not representable");
  }
  return Result<Scalar>::success(result);
}

} // namespace

Result<Scalar> quantile(ConstVectorView values, Scalar probability,
                        QuantileRule rule, Limits limits) {
  const Status extent = validateExtent(values);
  if (!extent.ok()) {
    return Result<Scalar>::failure(extent.code(),
                                   std::string(extent.message()));
  }
  if (!std::isfinite(probability) || probability < 0.0 || probability > 1.0 ||
      !validRule(rule)) {
    return failure<Scalar>(ErrorCode::InvalidDomain,
                           "quantile probability or rule is invalid");
  }
  const Status validation = validateFinite(values);
  if (!validation.ok()) {
    return Result<Scalar>::failure(validation.code(),
                                   std::string(validation.message()));
  }
  auto sortedResult = sortedCopy(values, limits);
  if (!sortedResult.hasValue()) {
    return Result<Scalar>::failure(
        sortedResult.status().code(),
        std::string(sortedResult.status().message()));
  }
  const Scalar result =
      rule == QuantileRule::linear
          ? linearQuantile(sortedResult.value(), probability)
          : inverseEmpiricalQuantile(sortedResult.value(), probability);
  if (!std::isfinite(result)) {
    return failure<Scalar>(ErrorCode::NonFiniteResult,
                           "quantile result is not representable");
  }
  return Result<Scalar>::success(result);
}

Result<Scalar> median(ConstVectorView values, Limits limits) {
  return quantile(values, 0.5, QuantileRule::linear, limits);
}

Result<Scalar> trimmedMean(ConstVectorView values, Scalar fraction,
                           Limits limits) {
  return sortedMean(values, fraction, limits, false);
}

Result<Scalar> winsorizedMean(ConstVectorView values, Scalar fraction,
                              Limits limits) {
  return sortedMean(values, fraction, limits, true);
}

} // namespace vss::math::statistics
