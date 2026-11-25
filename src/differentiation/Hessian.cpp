#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <span>
#include <string>
#include <utility>
#include <vector>
#include <vss/math/Core.hpp>
#include <vss/math/Error.hpp>
#include <vss/math/Limits.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/Tensor.hpp>
#include <vss/math/TensorLimits.hpp>
#include <vss/math/TensorShape.hpp>
#include <vss/math/Vector.hpp>
#include <vss/math/VectorView.hpp>
#include <vss/math/differentiation/FiniteDifference.hpp>
#include <vss/math/differentiation/Hessian.hpp>

namespace vss::math::differentiation {
namespace {

using Values = std::vector<Scalar>;
using Sample = std::function<Result<Values>(ConstVectorView)>;

Result<Values> callbackValues(const ScalarFunction& function,
                              ConstVectorView point) {
  try {
    auto result = function(point);
    if (!result) {
      return Result<Values>::failure(result.status().code(),
                                     std::string(result.status().message()));
    }
    if (!std::isfinite(result.value())) {
      return Result<Values>::failure(
          ErrorCode::NonFiniteInput,
          "Hessian callback returned a non-finite scalar");
    }
    return Result<Values>::success(Values{result.value()});
  } catch (...) {
    return Result<Values>::failure(ErrorCode::CallbackFailure,
                                   "Hessian callback threw an exception");
  }
}

Result<Values> callbackValues(const VectorFunction& function,
                              ConstVectorView point) {
  try {
    auto result = function(point);
    if (!result) {
      return Result<Values>::failure(result.status().code(),
                                     std::string(result.status().message()));
    }
    if (result.value().empty()) {
      return Result<Values>::success(Values{});
    }
    Values values;
    values.reserve(result.value().size());
    for (Index i = 0U; i < result.value().size(); ++i) {
      values.push_back(result.value().at(i));
    }
    if (!std::ranges::all_of(
            values, [](Scalar value) { return std::isfinite(value); })) {
      return Result<Values>::failure(
          ErrorCode::NonFiniteInput,
          "Hessian callback returned non-finite values");
    }
    return Result<Values>::success(std::move(values));
  } catch (...) {
    return Result<Values>::failure(ErrorCode::CallbackFailure,
                                   "Hessian callback threw an exception");
  }
}

bool finite(ConstVectorView values) {
  for (Index i = 0U; i < values.size(); ++i) {
    if (!std::isfinite(values.at(i))) {
      return false;
    }
  }
  return true;
}

bool productFits(Extent a, Extent b, Extent& product) {
  if (a != 0U && b > std::numeric_limits<Extent>::max() / a) {
    return false;
  }
  product = a * b;
  return true;
}

Status validate(ConstVectorView x, const HessianPolicy& policy) {
  if (!finite(x)) {
    return Status::failure(ErrorCode::NonFiniteInput,
                           "Hessian parameters must be finite");
  }
  if (policy.steps.size() != x.size()) {
    return Status::failure(
        ErrorCode::DimensionMismatch,
        "Hessian requires one step configuration per parameter");
  }
  if (!std::isfinite(policy.check_tolerance) || policy.check_tolerance < 0.0 ||
      policy.halvings == 0U) {
    return Status::failure(ErrorCode::InvalidDomain,
                           "Hessian refinement policy is invalid");
  }
  for (Index i = 0U; i < x.size(); ++i) {
    const auto& step = policy.steps[i];
    if (!std::isfinite(step.scale) || step.scale <= 0.0 ||
        (step.explicitStep &&
         (!std::isfinite(*step.explicitStep) || *step.explicitStep <= 0.0)) ||
        (step.lowerBound && !std::isfinite(*step.lowerBound)) ||
        (step.upperBound && !std::isfinite(*step.upperBound)) ||
        (step.lowerBound && step.upperBound &&
         *step.lowerBound >= *step.upperBound) ||
        (step.lowerBound && x.at(i) < *step.lowerBound) ||
        (step.upperBound && x.at(i) > *step.upperBound)) {
      return Status::failure(ErrorCode::InvalidDomain,
                             "Hessian step or bounds are invalid");
    }
  }
  return Status::success();
}

struct Stencil final {
  std::vector<Scalar> offsets;
  std::vector<Scalar> weights;
};

std::vector<Scalar> derivativeWeights(std::span<const Scalar> offsets,
                                      Index order, Scalar scale) {
  std::vector<Scalar> weights(offsets.size(), 0.0);
  for (Index i = 0U; i < offsets.size(); ++i) {
    std::vector<long double> polynomial{1.0L};
    long double denominator = 1.0L;
    for (Index j = 0U; j < offsets.size(); ++j) {
      if (j == i) {
        continue;
      }
      const auto node = static_cast<long double>(offsets[j] / scale);
      const long double difference =
          static_cast<long double>(offsets[i] / scale) - node;
      if (difference == 0.0L) {
        return {};
      }
      std::vector<long double> next(polynomial.size() + 1U, 0.0L);
      for (Index k = 0U; k < polynomial.size(); ++k) {
        next[k] -= node * polynomial[k];
        next[k + 1U] += polynomial[k];
      }
      polynomial = std::move(next);
      denominator *= difference;
    }
    long double coefficient = polynomial[order] / denominator;
    if (order == 2U) {
      coefficient *= 2.0L;
    }
    coefficient /=
        std::pow(static_cast<long double>(scale), static_cast<int>(order));
    weights[i] = static_cast<Scalar>(coefficient);
  }
  return weights;
}

Result<Stencil> makeStencil(Scalar base, const ParameterStep& config,
                            Index derivativeOrder, bool oneSided,
                            Scalar levelStep) {
  const Scalar defaultStep =
      std::pow(std::numeric_limits<Scalar>::epsilon(), 0.25) *
      std::max(std::abs(base), config.scale);
  Scalar h = config.explicitStep.value_or(defaultStep) * levelStep;
  const Scalar lower =
      config.lowerBound.value_or(-std::numeric_limits<Scalar>::infinity());
  const Scalar upper =
      config.upperBound.value_or(std::numeric_limits<Scalar>::infinity());
  auto make = [&](int direction, bool central) -> Result<Stencil> {
    const Index count = central ? derivativeOrder + 1U : derivativeOrder + 2U;
    Scalar allowed = h;
    if (central) {
      allowed = std::min({allowed, base - lower, upper - base});
    } else {
      const Scalar room = direction > 0 ? upper - base : base - lower;
      allowed = std::min(allowed, room / static_cast<Scalar>(count - 1U));
    }
    if (!(allowed > 0.0) || !std::isfinite(allowed)) {
      return Result<Stencil>::failure(
          ErrorCode::NoConvergence,
          "no finite Hessian stencil fits the bounds");
    }
    Stencil stencil;
    stencil.offsets.reserve(count);
    if (central) {
      if (derivativeOrder == 2U) {
        stencil.offsets = {-allowed, 0.0, allowed};
      } else {
        stencil.offsets = {-allowed, allowed};
      }
    } else {
      for (Index i = 0U; i < count; ++i) {
        stencil.offsets.push_back(static_cast<Scalar>(direction) *
                                  static_cast<Scalar>(i) * allowed);
      }
    }
    Scalar prior = std::numeric_limits<Scalar>::quiet_NaN();
    for (Scalar& offset : stencil.offsets) {
      const Scalar point = base + offset;
      if (!std::isfinite(point) || point < lower || point > upper ||
          (std::isfinite(prior) && point == prior)) {
        return Result<Stencil>::failure(
            ErrorCode::NoConvergence,
            "Hessian stencil points are not representable");
      }
      const Scalar actualOffset = point - base;
      if ((offset != 0.0 && actualOffset == 0.0) ||
          !std::isfinite(actualOffset)) {
        return Result<Stencil>::failure(
            ErrorCode::NoConvergence,
            "Hessian stencil displacement is not representable");
      }
      offset = actualOffset;
      prior = point;
    }
    stencil.weights =
        derivativeWeights(stencil.offsets, derivativeOrder, allowed);
    if (stencil.weights.size() != count) {
      return Result<Stencil>::failure(ErrorCode::NoConvergence,
                                      "Hessian stencil weights are singular");
    }
    return Result<Stencil>::success(std::move(stencil));
  };

  auto central = make(1, true);
  if (central) {
    return central;
  }
  if (!oneSided) {
    return central;
  }
  auto forward = make(1, false);
  if (forward) {
    return forward;
  }
  return make(-1, false);
}

using Cache = std::map<std::vector<std::uint64_t>, Values>;

bool cacheEntryFits(Index inputCount, Index outputCount,
                    Extent reservedElements, Extent reservedBytes,
                    Limits limits) {
  constexpr Extent overhead = sizeof(Cache::value_type) + (3U * sizeof(void*));
  if (reservedElements > limits.elements ||
      outputCount > limits.elements - reservedElements ||
      reservedBytes > limits.bytes ||
      outputCount > std::numeric_limits<Extent>::max() / sizeof(Scalar)) {
    return false;
  }
  const Extent valuesBytes = outputCount * sizeof(Scalar);
  if (valuesBytes > std::numeric_limits<Extent>::max() - overhead ||
      inputCount >
          (std::numeric_limits<Extent>::max() - valuesBytes - overhead) /
              sizeof(std::uint64_t)) {
    return false;
  }
  const Extent entryBytes =
      valuesBytes + (inputCount * sizeof(std::uint64_t)) + overhead;
  return entryBytes <= limits.bytes - reservedBytes;
}

std::vector<std::uint64_t> keyFor(ConstVectorView point) {
  std::vector<std::uint64_t> key;
  key.reserve(point.size());
  for (Index i = 0U; i < point.size(); ++i) {
    key.push_back(std::bit_cast<std::uint64_t>(point.at(i)));
  }
  return key;
}

Result<const Values*> sampleAt(ConstVectorView x, const Sample& sample,
                               Cache& cache, std::uint64_t& evaluations,
                               Index outputCount, Extent reservedElements,
                               Extent reservedBytes, Limits limits) {
  const auto key = keyFor(x);
  if (const auto found = cache.find(key); found != cache.end()) {
    return Result<const Values*>::success(&found->second);
  }
  if (evaluations >= limits.evaluations) {
    return Result<const Values*>::failure(
        ErrorCode::ResourceLimit, "Hessian callback evaluation limit exceeded");
  }
  const Extent availableElements = reservedElements <= limits.elements
                                       ? limits.elements - reservedElements
                                       : 0U;
  if (reservedElements > limits.elements ||
      (outputCount != 0U && cache.size() >= availableElements / outputCount)) {
    return Result<const Values*>::failure(
        ErrorCode::ResourceLimit, "Hessian cache exceeds configured limits");
  }
  if (outputCount > std::numeric_limits<Extent>::max() / sizeof(Scalar)) {
    return Result<const Values*>::failure(ErrorCode::ResourceLimit,
                                          "Hessian cache size overflows");
  }
  const Extent valueBytes = outputCount * sizeof(Scalar);
  constexpr Extent nodeOverhead =
      sizeof(Cache::value_type) + (3U * sizeof(void*));
  if (valueBytes > std::numeric_limits<Extent>::max() - nodeOverhead ||
      x.size() >
          (std::numeric_limits<Extent>::max() - valueBytes - nodeOverhead) /
              sizeof(std::uint64_t)) {
    return Result<const Values*>::failure(ErrorCode::ResourceLimit,
                                          "Hessian cache size overflows");
  }
  const Extent perSampleBytes =
      valueBytes + (x.size() * sizeof(std::uint64_t)) + nodeOverhead;
  const Extent availableBytes =
      reservedBytes <= limits.bytes ? limits.bytes - reservedBytes : 0U;
  if (reservedBytes > limits.bytes ||
      (perSampleBytes != 0U &&
       cache.size() >= availableBytes / perSampleBytes)) {
    return Result<const Values*>::failure(
        ErrorCode::ResourceLimit,
        "Hessian cache exceeds configured byte limit");
  }
  ++evaluations;
  auto result = sample(x);
  if (!result) {
    return Result<const Values*>::failure(
        result.status().code(), std::string(result.status().message()));
  }
  if (result.value().size() != outputCount) {
    return Result<const Values*>::failure(
        ErrorCode::DimensionMismatch, "Hessian callback output shape changed");
  }
  auto inserted = cache.emplace(key, std::move(result).value());
  return Result<const Values*>::success(&inserted.first->second);
}

Result<Values> estimate(ConstVectorView x, const HessianPolicy& policy,
                        const Sample& sample, const Values& baseline,
                        Scalar levelStep, Limits limits,
                        std::uint64_t& evaluations) {
  const Index n = x.size();
  const Index m = baseline.size();
  Extent count = 0U;
  if (!productFits(m, n, count) || !productFits(count, n, count) ||
      count > limits.elements || count > limits.bytes / sizeof(Scalar)) {
    return Result<Values>::failure(ErrorCode::ResourceLimit,
                                   "Hessian output exceeds configured limits");
  }
  if (count > limits.elements / 2U ||
      count > limits.bytes / (2U * sizeof(Scalar))) {
    return Result<Values>::failure(
        ErrorCode::ResourceLimit,
        "Hessian refinement outputs exceed configured limits");
  }
  const Extent reservedElements = count * 2U;
  const Extent outputBytes = count * 2U * sizeof(Scalar);
  // Account for the current and previous estimates, compensated accumulators,
  // the copied baseline and the bounded per-coordinate stencil workspace.
  if (count > limits.bytes / (2U * sizeof(long double)) ||
      m > std::numeric_limits<Extent>::max() / sizeof(Scalar)) {
    return Result<Values>::failure(ErrorCode::ResourceLimit,
                                   "Hessian scratch size overflows");
  }
  const Extent accumulatorBytes = 2U * count * sizeof(long double);
  const Extent baselineBytes = m * sizeof(Scalar);
  constexpr Extent perCoordinateBytes =
      (16U * sizeof(Scalar)) + (2U * sizeof(Stencil));
  if (outputBytes > limits.bytes ||
      accumulatorBytes > limits.bytes - outputBytes ||
      baselineBytes > limits.bytes - outputBytes - accumulatorBytes ||
      n > (limits.bytes - outputBytes - accumulatorBytes - baselineBytes) /
              perCoordinateBytes) {
    return Result<Values>::failure(ErrorCode::ResourceLimit,
                                   "Hessian scratch exceeds configured limits");
  }
  const Extent reservedBytes =
      outputBytes + accumulatorBytes + baselineBytes + (n * perCoordinateBytes);
  Values output(count, 0.0);
  std::vector<long double> sums(count, 0.0L);
  std::vector<long double> corrections(count, 0.0L);
  std::vector<Stencil> first(n);
  std::vector<Stencil> second(n);
  for (Index i = 0U; i < n; ++i) {
    auto a =
        makeStencil(x.at(i), policy.steps[i], 1U, policy.one_sided, levelStep);
    auto b =
        makeStencil(x.at(i), policy.steps[i], 2U, policy.one_sided, levelStep);
    if (!a || !b) {
      return Result<Values>::failure(ErrorCode::NoConvergence,
                                     "no valid Hessian stencil is available");
    }
    first[i] = std::move(a).value();
    second[i] = std::move(b).value();
  }
  for (Index i = 0U; i < n; ++i) {
    for (Index j = i; j < n; ++j) {
      // Keep cache lifetime to one matrix entry, as specified. A whole-level
      // cache retains O(n^2) point keys, each of length n.
      if (!cacheEntryFits(n, m, reservedElements, reservedBytes, limits)) {
        return Result<Values>::failure(
            ErrorCode::ResourceLimit,
            "Hessian baseline cache exceeds configured limits");
      }
      Cache cache;
      cache.emplace(keyFor(x), baseline);
      const Stencil& si = (i == j) ? second[i] : first[i];
      const Stencil& sj = first[j];
      for (Index a = 0U; a < si.offsets.size(); ++a) {
        const Index bCount = i == j ? 1U : sj.offsets.size();
        for (Index b = 0U; b < bCount; ++b) {
          Vector point(x.size());
          for (Index k = 0U; k < x.size(); ++k) {
            point.at(k) = x.at(k);
          }
          point.at(i) += si.offsets[a];
          if (i != j) {
            point.at(j) += sj.offsets[b];
          }
          auto values = sampleAt(point.view(), sample, cache, evaluations, m,
                                 reservedElements, reservedBytes, limits);
          if (!values) {
            return Result<Values>::failure(
                values.status().code(), std::string(values.status().message()));
          }
          const long double weight =
              i == j ? static_cast<long double>(si.weights[a])
                     : static_cast<long double>(si.weights[a]) * sj.weights[b];
          for (Index row = 0U; row < m; ++row) {
            const long double contribution =
                weight * static_cast<long double>(values.value()->at(row));
            const auto index = ((row * n + i) * n) + j;
            const long double updated = sums[index] + contribution;
            if (std::abs(sums[index]) >= std::abs(contribution)) {
              corrections[index] += (sums[index] - updated) + contribution;
            } else {
              corrections[index] += (contribution - updated) + sums[index];
            }
            sums[index] = updated;
          }
        }
      }
    }
  }
  for (Index row = 0U; row < m; ++row) {
    for (Index i = 0U; i < n; ++i) {
      for (Index j = i; j < n; ++j) {
        const Index upper = ((row * n + i) * n) + j;
        const long double accumulated = sums[upper] + corrections[upper];
        const auto value = static_cast<Scalar>(accumulated);
        if (!std::isfinite(accumulated) || !std::isfinite(value)) {
          return Result<Values>::failure(ErrorCode::NonFiniteResult,
                                         "Hessian contraction is not finite");
        }
        output[upper] = value;
        output[((row * n + j) * n) + i] = value;
      }
    }
  }
  return Result<Values>::success(std::move(output));
}

Result<Values> estimateProduct(ConstVectorView x, ConstVectorView direction,
                               const HessianPolicy& policy,
                               const Sample& sample, Scalar directionNorm,
                               Scalar levelFactor, const Values& baseline,
                               Limits limits, std::uint64_t& evaluations) {
  const Index n = x.size();
  Scalar directionStep = std::numeric_limits<Scalar>::infinity();
  Scalar directionLower = -std::numeric_limits<Scalar>::infinity();
  Scalar directionUpper = std::numeric_limits<Scalar>::infinity();
  std::vector<Scalar> unit(n);
  for (Index i = 0U; i < n; ++i) {
    unit[i] = direction.at(i) / directionNorm;
  }
  for (Index i = 0U; i < n; ++i) {
    if (unit[i] == 0.0) {
      continue;
    }
    const auto& step = policy.steps[i];
    const Scalar nominal = step.explicitStep.value_or(
        std::pow(std::numeric_limits<Scalar>::epsilon(), 0.25) *
        std::max(std::abs(x.at(i)), step.scale));
    directionStep = std::min(directionStep, nominal / std::abs(unit[i]));
    const Scalar lower =
        step.lowerBound.value_or(-std::numeric_limits<Scalar>::infinity());
    const Scalar upper =
        step.upperBound.value_or(std::numeric_limits<Scalar>::infinity());
    const Scalar a = (lower - x.at(i)) / unit[i];
    const Scalar b = (upper - x.at(i)) / unit[i];
    directionLower = std::max(directionLower, std::min(a, b));
    directionUpper = std::min(directionUpper, std::max(a, b));
  }
  if (!std::isfinite(directionStep) || directionStep <= 0.0) {
    return Result<Values>::failure(
        ErrorCode::NoConvergence,
        "no representable direction step is available");
  }
  ParameterStep directionalConfig(directionStep);
  directionalConfig.explicitStep = directionStep;
  directionalConfig.lowerBound = directionLower;
  directionalConfig.upperBound = directionUpper;
  auto directional =
      makeStencil(0.0, directionalConfig, 1U, policy.one_sided, levelFactor);
  if (!directional) {
    return Result<Values>::failure(
        ErrorCode::NoConvergence,
        "no admissible directional stencil is available");
  }

  Values product(n, 0.0);
  for (Index i = 0U; i < n; ++i) {
    // The product is assembled by coordinate; do not retain every sample
    // from all coordinates, which would turn O(n) workspace into O(n^2).
    constexpr Extent fixedScratchElements = 3U;
    if (!cacheEntryFits(n, 1U, (n * 5U) + fixedScratchElements,
                        (n * 6U * sizeof(Scalar)) + sizeof(Scalar), limits)) {
      return Result<Values>::failure(
          ErrorCode::ResourceLimit,
          "Hessian-vector baseline cache exceeds configured limits");
    }
    Cache cache;
    cache.emplace(keyFor(x), baseline);
    Scalar aLower = -std::numeric_limits<Scalar>::infinity();
    Scalar aUpper = std::numeric_limits<Scalar>::infinity();
    const auto& coordinate = policy.steps[i];
    for (const Scalar b : directional.value().offsets) {
      const Scalar shifted = x.at(i) + (b * unit[i]);
      const Scalar lower = coordinate.lowerBound.value_or(
          -std::numeric_limits<Scalar>::infinity());
      const Scalar upper = coordinate.upperBound.value_or(
          std::numeric_limits<Scalar>::infinity());
      aLower = std::max(aLower, lower - shifted);
      aUpper = std::min(aUpper, upper - shifted);
    }
    ParameterStep axis = coordinate;
    axis.lowerBound = aLower;
    axis.upperBound = aUpper;
    const Scalar axisScale = coordinate.explicitStep.value_or(
        std::pow(std::numeric_limits<Scalar>::epsilon(), 0.25) *
        std::max(std::abs(x.at(i)), coordinate.scale));
    axis.explicitStep = axisScale;
    auto axisStencil =
        makeStencil(0.0, axis, 1U, policy.one_sided, levelFactor);
    if (!axisStencil) {
      return Result<Values>::failure(
          ErrorCode::NoConvergence,
          "no admissible coordinate-direction rectangle exists");
    }
    long double sum = 0.0L;
    for (Index a = 0U; a < axisStencil.value().offsets.size(); ++a) {
      for (Index b = 0U; b < directional.value().offsets.size(); ++b) {
        Vector point(n);
        for (Index k = 0U; k < n; ++k) {
          point.at(k) = x.at(k) + (directional.value().offsets[b] * unit[k]);
        }
        point.at(i) += axisStencil.value().offsets[a];
        for (Index k = 0U; k < n; ++k) {
          const auto& bound = policy.steps[k];
          if (!std::isfinite(point.at(k)) ||
              (bound.lowerBound && point.at(k) < *bound.lowerBound) ||
              (bound.upperBound && point.at(k) > *bound.upperBound)) {
            return Result<Values>::failure(
                ErrorCode::NoConvergence,
                "Hessian-vector stencil left its bounds");
          }
        }
        const Extent reservedElements = (n * 5U) + fixedScratchElements;
        const Extent reservedBytes = (n * 6U * sizeof(Scalar)) + sizeof(Scalar);
        auto values = sampleAt(point.view(), sample, cache, evaluations, 1U,
                               reservedElements, reservedBytes, limits);
        if (!values) {
          return Result<Values>::failure(
              values.status().code(), std::string(values.status().message()));
        }
        sum += static_cast<long double>(axisStencil.value().weights[a]) *
               directional.value().weights[b] * values.value()->at(0U);
      }
    }
    product[i] =
        static_cast<Scalar>(sum * static_cast<long double>(directionNorm));
    if (!std::isfinite(product[i])) {
      return Result<Values>::failure(ErrorCode::NonFiniteResult,
                                     "Hessian-vector component is not finite");
    }
  }
  return Result<Values>::success(std::move(product));
}

template <class Function>
Result<Values> run(Function&& callback, ConstVectorView x,
                   const HessianPolicy& policy, Limits limits) {
  const Status valid = validate(x, policy);
  if (!valid.ok()) {
    return Result<Values>::failure(valid.code(), std::string(valid.message()));
  }
  if (limits.evaluations == 0U || x.size() > limits.bytes / sizeof(Scalar)) {
    return Result<Values>::failure(
        ErrorCode::ResourceLimit,
        "Hessian preflight exceeds configured limits");
  }
  const Sample sample = std::forward<Function>(callback);
  Vector base(x.size());
  for (Index i = 0U; i < x.size(); ++i) {
    base.at(i) = x.at(i);
  }
  std::uint64_t evaluations = 1U;
  if (evaluations > limits.evaluations) {
    return Result<Values>::failure(
        ErrorCode::ResourceLimit, "Hessian callback evaluation limit exceeded");
  }
  auto baseline = sample(base.view());
  if (!baseline) {
    return baseline;
  }
  Extent outputSize = 0U;
  if (!productFits(baseline.value().size(), x.size(), outputSize) ||
      !productFits(outputSize, x.size(), outputSize) ||
      outputSize > limits.elements ||
      outputSize > limits.bytes / sizeof(Scalar) ||
      baseline.value().size() > limits.elements ||
      baseline.value().size() > limits.bytes / sizeof(Scalar)) {
    return Result<Values>::failure(ErrorCode::ResourceLimit,
                                   "Hessian output exceeds configured limits");
  }
  if (baseline.value().empty()) {
    return Result<Values>::success(Values{});
  }
  Result<Values> previous = Result<Values>::failure(
      ErrorCode::NoConvergence, "Hessian refinement has no prior estimate");
  Scalar factor = 1.0;
  for (Index level = 0U;; ++level) {
    auto current = estimate(x, policy, sample, baseline.value(), factor, limits,
                            evaluations);
    if (!current) {
      return current;
    }
    if (previous) {
      bool stable = current.value().size() == previous.value().size();
      for (Index i = 0U; stable && i < current.value().size(); ++i) {
        const Scalar a = previous.value()[i];
        const Scalar b = current.value()[i];
        if (std::abs(b - a) > policy.check_tolerance *
                                  std::max({1.0, std::abs(a), std::abs(b)})) {
          stable = false;
        }
      }
      if (stable) {
        return current;
      }
    }
    previous = std::move(current);
    if (level == policy.halvings) {
      break;
    }
    factor *= 0.5;
    if (factor == 0.0) {
      return Result<Values>::failure(
          ErrorCode::NoConvergence,
          "Hessian refinement step underflowed before convergence");
    }
  }
  return Result<Values>::failure(
      ErrorCode::NoConvergence,
      "Hessian estimates did not stabilize under refinement");
}

} // namespace

Result<Matrix> numericalHessian(const ScalarFunction& function,
                                ConstVectorView parameters,
                                const HessianPolicy& policy, Limits limits) {
  if (!function) {
    return Result<Matrix>::failure(ErrorCode::CallbackFailure,
                                   "Hessian callback is empty");
  }
  auto result = run(
      [&function](ConstVectorView x) { return callbackValues(function, x); },
      parameters, policy, limits);
  if (!result) {
    return Result<Matrix>::failure(result.status().code(),
                                   std::string(result.status().message()));
  }
  const Index n = parameters.size();
  Matrix matrix(n, n);
  for (Index i = 0U; i < n; ++i) {
    for (Index j = 0U; j < n; ++j) {
      matrix.at(i, j) = result.value()[(i * n) + j];
    }
  }
  return Result<Matrix>::success(std::move(matrix));
}

Result<Tensor> numericalOutputHessians(const VectorFunction& function,
                                       ConstVectorView parameters,
                                       const HessianPolicy& policy,
                                       Limits limits) {
  if (!function) {
    return Result<Tensor>::failure(ErrorCode::CallbackFailure,
                                   "Hessian callback is empty");
  }
  if (parameters.empty()) {
    const Status valid = validate(parameters, policy);
    if (!valid.ok()) {
      return Result<Tensor>::failure(valid.code(),
                                     std::string(valid.message()));
    }
    if (limits.evaluations == 0U) {
      return Result<Tensor>::failure(
          ErrorCode::ResourceLimit,
          "Hessian callback evaluation limit exceeded");
    }
    Vector empty;
    auto baseline = callbackValues(function, empty.view());
    if (!baseline) {
      return Result<Tensor>::failure(baseline.status().code(),
                                     std::string(baseline.status().message()));
    }
    if (baseline.value().size() > limits.elements ||
        baseline.value().size() > limits.bytes / sizeof(Scalar)) {
      return Result<Tensor>::failure(
          ErrorCode::ResourceLimit, "Hessian output exceeds configured limits");
    }
    const std::array<Extent, 3> emptyExtents{baseline.value().size(), 0U, 0U};
    auto emptyShape = TensorShape::create(emptyExtents);
    if (!emptyShape) {
      return Result<Tensor>::failure(
          emptyShape.status().code(),
          std::string(emptyShape.status().message()));
    }
    return Tensor::create(emptyShape.value(), 0.0,
                          TensorLimits{limits.elements});
  }
  auto result = run(
      [&function](ConstVectorView x) { return callbackValues(function, x); },
      parameters, policy, limits);
  if (!result) {
    return Result<Tensor>::failure(result.status().code(),
                                   std::string(result.status().message()));
  }
  const Index n = parameters.size();
  const Index m = result.value().size() / (n * n);
  const std::array<Extent, 3> extents{m, n, n};
  auto shape = TensorShape::create(extents);
  if (!shape) {
    return Result<Tensor>::failure(shape.status().code(),
                                   std::string(shape.status().message()));
  }
  auto tensor =
      Tensor::create(shape.value(), 0.0, TensorLimits{limits.elements});
  if (!tensor) {
    return tensor;
  }
  for (Index output = 0U; output < m; ++output) {
    for (Index row = 0U; row < n; ++row) {
      for (Index column = 0U; column < n; ++column) {
        const std::array<Index, 3> indices{output, row, column};
        const Index linear = ((output * n + row) * n) + column;
        tensor.value().at(indices) = result.value().at(linear);
      }
    }
  }
  return tensor;
}

Result<Vector> hessianVectorProduct(const ScalarFunction& function,
                                    ConstVectorView parameters,
                                    ConstVectorView direction,
                                    const HessianPolicy& policy,
                                    Limits limits) {
  const Status valid = validate(parameters, policy);
  if (!valid.ok()) {
    return Result<Vector>::failure(valid.code(), std::string(valid.message()));
  }
  if (direction.size() != parameters.size()) {
    return Result<Vector>::failure(
        ErrorCode::DimensionMismatch,
        "Hessian direction extent does not match parameters");
  }
  if (!finite(direction)) {
    return Result<Vector>::failure(ErrorCode::NonFiniteInput,
                                   "Hessian direction must be finite");
  }
  if (!function) {
    return Result<Vector>::failure(ErrorCode::CallbackFailure,
                                   "Hessian callback is empty");
  }
  if (parameters.size() > limits.elements ||
      parameters.size() > limits.bytes / sizeof(Scalar)) {
    return Result<Vector>::failure(ErrorCode::ResourceLimit,
                                   "Hessian-vector output exceeds limits");
  }
  Scalar scale = 0.0;
  for (Index i = 0U; i < direction.size(); ++i) {
    scale = std::max(scale, std::abs(direction.at(i)));
  }
  if (scale == 0.0) {
    return Result<Vector>::success(Vector(parameters.size()));
  }
  if (limits.evaluations == 0U || limits.elements < 3U ||
      parameters.size() > (limits.elements - 3U) / 5U ||
      limits.bytes < sizeof(Scalar) ||
      parameters.size() >
          (limits.bytes - sizeof(Scalar)) / (6U * sizeof(Scalar)) ||
      parameters.size() >
          std::numeric_limits<Extent>::max() / (6U * sizeof(Scalar))) {
    return Result<Vector>::failure(
        ErrorCode::ResourceLimit,
        "Hessian-vector preflight exceeds configured limits");
  }
  Vector base(parameters.size());
  for (Index i = 0U; i < parameters.size(); ++i) {
    base.at(i) = parameters.at(i);
  }
  auto baseline = callbackValues(function, base.view());
  if (!baseline) {
    return Result<Vector>::failure(baseline.status().code(),
                                   std::string(baseline.status().message()));
  }
  if (baseline.value().size() != 1U) {
    return Result<Vector>::failure(
        ErrorCode::DimensionMismatch,
        "Hessian-vector callback must be scalar-valued");
  }
  std::uint64_t evaluations = 1U;
  Result<Values> previous = Result<Values>::failure(
      ErrorCode::NoConvergence,
      "Hessian-vector refinement has no prior estimate");
  const Sample sample = [&function](ConstVectorView point) {
    return callbackValues(function, point);
  };
  Scalar factor = 1.0;
  for (Index level = 0U;; ++level) {
    auto current =
        estimateProduct(parameters, direction, policy, sample, scale, factor,
                        baseline.value(), limits, evaluations);
    if (!current) {
      return Result<Vector>::failure(current.status().code(),
                                     std::string(current.status().message()));
    }
    if (previous) {
      bool stable = current.value().size() == previous.value().size();
      for (Index i = 0U; stable && i < current.value().size(); ++i) {
        const Scalar a = previous.value()[i];
        const Scalar b = current.value()[i];
        if (std::abs(b - a) > policy.check_tolerance *
                                  std::max({1.0, std::abs(a), std::abs(b)})) {
          stable = false;
        }
      }
      if (stable) {
        Vector result(current.value());
        return Result<Vector>::success(std::move(result));
      }
    }
    previous = std::move(current);
    if (level == policy.halvings) {
      break;
    }
    factor *= 0.5;
    if (factor == 0.0) {
      return Result<Vector>::failure(
          ErrorCode::NoConvergence,
          "Hessian-vector refinement step underflowed before convergence");
    }
  }
  return Result<Vector>::failure(ErrorCode::NoConvergence,
                                 "Hessian-vector estimates did not stabilize");
}

} // namespace vss::math::differentiation
