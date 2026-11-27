#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <new>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>
#include <vss/math/Core.hpp>
#include <vss/math/Error.hpp>
#include <vss/math/Limits.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/Vector.hpp>
#include <vss/math/VectorView.hpp>
#include <vss/math/signal/Dependence.hpp>

namespace vss::math::signal {
namespace {

using Wide = long double;

template <class T> Result<T> fail(ErrorCode code, const char* message) {
  return Result<T>::failure(code, message);
}

class CompensatedSum final {
public:
  void add(Wide value) noexcept {
    const Wide next = sum_ + value;
    if (std::abs(sum_) >= std::abs(value)) {
      correction_ += (sum_ - next) + value;
    } else {
      correction_ += (value - next) + sum_;
    }
    sum_ = next;
  }
  [[nodiscard]] Wide value() const noexcept { return sum_ + correction_; }

private:
  Wide sum_{};
  Wide correction_{};
};

bool supported(LagNormalization normalization) noexcept {
  return normalization == LagNormalization::full_count ||
         normalization == LagNormalization::overlap_count;
}

Result<Index> outputBytes(Index count) {
  constexpr Index perLag =
      sizeof(std::int64_t) + sizeof(Scalar) + sizeof(Index);
  if (count > std::numeric_limits<Index>::max() / perLag) {
    return fail<Index>(ErrorCode::SizeOverflow,
                       "lag result storage size overflows");
  }
  return Result<Index>::success(count * perLag);
}

} // namespace

Result<LagSeries> LagSeries::compute(ConstVectorView samples,
                                     std::span<const Index> lags,
                                     LagNormalization normalization,
                                     Limits limits, bool correlation) {
  const Index n = samples.size();
  if (!supported(normalization)) {
    return fail<LagSeries>(ErrorCode::InvalidDomain,
                           "unknown lag normalization policy");
  }
  if (n == 0U) {
    return fail<LagSeries>(ErrorCode::InsufficientSamples,
                           "autodependence requires at least one sample");
  }
  Scalar reference = samples.at(0U);
  if (!std::isfinite(reference)) {
    return fail<LagSeries>(ErrorCode::NonFiniteInput,
                           "autodependence samples must be finite");
  }
  bool halfDifference = false;
  for (Index i = 0U; i < n; ++i) {
    const Scalar value = samples.at(i);
    if (!std::isfinite(value)) {
      return fail<LagSeries>(ErrorCode::NonFiniteInput,
                             "autodependence samples must be finite");
    }
    if constexpr (std::numeric_limits<Wide>::max_exponent <=
                  std::numeric_limits<Scalar>::max_exponent) {
      halfDifference = halfDifference || !std::isfinite(value - reference);
    }
  }
  for (const Index lag : lags) {
    if (lag >= n) {
      return fail<LagSeries>(ErrorCode::IndexOutOfRange,
                             "requested lag must be less than sample count");
    }
    if (lag > static_cast<Index>(std::numeric_limits<std::int64_t>::max())) {
      return fail<LagSeries>(ErrorCode::SizeOverflow,
                             "requested lag cannot be represented in result");
    }
  }
  const auto bytes = outputBytes(lags.size());
  if (!bytes) {
    return fail<LagSeries>(bytes.status().code(),
                           "lag result storage size overflows");
  }
  if (lags.size() > limits.elements / 3U || bytes.value() > limits.bytes) {
    return fail<LagSeries>(ErrorCode::ResourceLimit,
                           "lag result exceeds configured resource limits");
  }

  try {
    std::vector<std::int64_t> ownedLags;
    std::vector<Index> counts;
    ownedLags.reserve(lags.size());
    counts.reserve(lags.size());
    for (const Index lag : lags) {
      ownedLags.push_back(static_cast<std::int64_t>(lag));
      counts.push_back(n - lag);
    }
    Vector values(lags.size(), 0.0);

    const auto shifted = [&](Index index) -> Wide {
      const Scalar value = samples.at(index);
      if constexpr (std::numeric_limits<Wide>::max_exponent >
                    std::numeric_limits<Scalar>::max_exponent) {
        return static_cast<Wide>(value) - static_cast<Wide>(reference);
      } else {
        if (halfDifference) {
          return (static_cast<Wide>(value) * 0.5L) -
                 (static_cast<Wide>(reference) * 0.5L);
        }
        return static_cast<Wide>(value - reference);
      }
    };
    Wide scale = 0.0L;
    for (Index i = 0U; i < n; ++i) {
      scale = std::max(scale, std::abs(shifted(i)));
    }

    Wide mean = 0.0L;
    if (scale != 0.0L) {
      CompensatedSum meanSum;
      for (Index i = 0U; i < n; ++i) {
        meanSum.add(shifted(i) / scale);
      }
      mean = meanSum.value() / static_cast<Wide>(n);
    }
    const auto centered = [&](Index index) {
      return scale == 0.0L ? 0.0L : (shifted(index) / scale) - mean;
    };

    CompensatedSum fullVarianceSum;
    for (Index i = 0U; i < n; ++i) {
      const Wide deviation = centered(i);
      fullVarianceSum.add(deviation * deviation);
    }
    const Wide varianceSum = fullVarianceSum.value();
    if (correlation && !(varianceSum > 0.0L)) {
      return fail<LagSeries>(ErrorCode::InvalidDomain,
                             "autocorrelation is undefined for zero variance");
    }

    int scaleExponent = 0;
    const Wide scaleFraction =
        scale == 0.0L ? 0.0L : std::frexp(scale, &scaleExponent);
    if (halfDifference) {
      ++scaleExponent;
    }
    for (Index output = 0U; output < lags.size(); ++output) {
      const Index lag = lags[output];
      CompensatedSum products;
      for (Index t = lag; t < n; ++t) {
        products.add(centered(t) * centered(t - lag));
      }
      const Wide sum = products.value();
      const Index denominator =
          normalization == LagNormalization::full_count ? n : n - lag;
      Wide value = 0.0L;
      if (correlation) {
        if (lag == 0U) {
          value = 1.0L;
        } else {
          value = (sum / varianceSum) *
                  (normalization == LagNormalization::full_count
                       ? 1.0L
                       : static_cast<Wide>(n) / denominator);
        }
      } else if (sum != 0.0L && scale != 0.0L) {
        int sumExponent = 0;
        const Wide sumFraction = std::frexp(sum / denominator, &sumExponent);
        const std::int64_t exponent =
            static_cast<std::int64_t>(sumExponent) + (2LL * scaleExponent);
        if (exponent > std::numeric_limits<int>::max() ||
            exponent < std::numeric_limits<int>::min()) {
          return fail<LagSeries>(ErrorCode::NonFiniteResult,
                                 "autocovariance result is not representable");
        }
        const Wide fraction = sumFraction * scaleFraction * scaleFraction;
        value = std::scalbn(fraction, static_cast<int>(exponent));
      }
      const auto result = static_cast<Scalar>(value);
      if (!std::isfinite(result)) {
        return fail<LagSeries>(ErrorCode::NonFiniteResult,
                               "autodependence result is not representable");
      }
      values.at(output) = result;
    }
    return Result<LagSeries>::success(
        LagSeries(std::move(ownedLags), std::move(values), std::move(counts)));
  } catch (const std::bad_alloc&) {
    return fail<LagSeries>(ErrorCode::ResourceLimit,
                           "lag result allocation failed");
  } catch (const std::length_error&) {
    return fail<LagSeries>(ErrorCode::ResourceLimit,
                           "lag result extent is unsupported");
  }
}

Result<LagSeries> LagSeries::computeCross(ConstVectorView x, ConstVectorView y,
                                          std::span<const std::int64_t> lags,
                                          LagNormalization normalization,
                                          Limits limits, bool correlation) {
  const Index n = x.size();
  if (!supported(normalization)) {
    return fail<LagSeries>(ErrorCode::InvalidDomain,
                           "unknown lag normalization policy");
  }
  if (x.size() != y.size()) {
    return fail<LagSeries>(ErrorCode::DimensionMismatch,
                           "cross-dependence sequences must have equal size");
  }
  if (n == 0U) {
    return fail<LagSeries>(ErrorCode::InsufficientSamples,
                           "cross-dependence requires at least one sample");
  }

  const Scalar referenceX = x.at(0U);
  const Scalar referenceY = y.at(0U);
  if (!std::isfinite(referenceX) || !std::isfinite(referenceY)) {
    return fail<LagSeries>(ErrorCode::NonFiniteInput,
                           "cross-dependence samples must be finite");
  }
  bool halfDifferenceX = false;
  bool halfDifferenceY = false;
  for (Index i = 0U; i < n; ++i) {
    const Scalar valueX = x.at(i);
    const Scalar valueY = y.at(i);
    if (!std::isfinite(valueX) || !std::isfinite(valueY)) {
      return fail<LagSeries>(ErrorCode::NonFiniteInput,
                             "cross-dependence samples must be finite");
    }
    if constexpr (std::numeric_limits<Wide>::max_exponent <=
                  std::numeric_limits<Scalar>::max_exponent) {
      halfDifferenceX = halfDifferenceX || !std::isfinite(valueX - referenceX);
      halfDifferenceY = halfDifferenceY || !std::isfinite(valueY - referenceY);
    }
  }

  const auto lagMagnitude = [](std::int64_t lag) {
    return lag < 0 ? static_cast<std::uint64_t>(-(lag + 1)) + 1U
                   : static_cast<std::uint64_t>(lag);
  };
  for (const std::int64_t lag : lags) {
    if (lagMagnitude(lag) >= static_cast<std::uint64_t>(n)) {
      return fail<LagSeries>(ErrorCode::IndexOutOfRange,
                             "signed lag has no matched sample pair");
    }
  }

  const auto bytes = outputBytes(lags.size());
  if (!bytes) {
    return fail<LagSeries>(bytes.status().code(),
                           "cross-lag result storage size overflows");
  }
  if (lags.size() > limits.elements / 3U || bytes.value() > limits.bytes) {
    return fail<LagSeries>(ErrorCode::ResourceLimit,
                           "cross-lag result exceeds configured limits");
  }

  try {
    std::vector<std::int64_t> ownedLags(lags.begin(), lags.end());
    std::vector<Index> counts;
    counts.reserve(lags.size());
    for (const std::int64_t lag : lags) {
      counts.push_back(n - static_cast<Index>(lagMagnitude(lag)));
    }
    Vector values(lags.size(), 0.0);

    const auto shifted = [](ConstVectorView series, Index index,
                            Scalar reference, bool halfDifference) -> Wide {
      const Scalar value = series.at(index);
      if constexpr (std::numeric_limits<Wide>::max_exponent >
                    std::numeric_limits<Scalar>::max_exponent) {
        return static_cast<Wide>(value) - static_cast<Wide>(reference);
      } else {
        if (halfDifference) {
          return (static_cast<Wide>(value) * 0.5L) -
                 (static_cast<Wide>(reference) * 0.5L);
        }
        return static_cast<Wide>(value - reference);
      }
    };
    Wide scaleX = 0.0L;
    Wide scaleY = 0.0L;
    for (Index i = 0U; i < n; ++i) {
      scaleX = std::max(scaleX,
                        std::abs(shifted(x, i, referenceX, halfDifferenceX)));
      scaleY = std::max(scaleY,
                        std::abs(shifted(y, i, referenceY, halfDifferenceY)));
    }

    Wide meanX = 0.0L;
    Wide meanY = 0.0L;
    if (scaleX != 0.0L) {
      CompensatedSum meanSum;
      for (Index i = 0U; i < n; ++i) {
        meanSum.add(shifted(x, i, referenceX, halfDifferenceX) / scaleX);
      }
      meanX = meanSum.value() / static_cast<Wide>(n);
    }
    if (scaleY != 0.0L) {
      CompensatedSum meanSum;
      for (Index i = 0U; i < n; ++i) {
        meanSum.add(shifted(y, i, referenceY, halfDifferenceY) / scaleY);
      }
      meanY = meanSum.value() / static_cast<Wide>(n);
    }
    const auto centeredX = [&](Index index) {
      return scaleX == 0.0L
                 ? 0.0L
                 : shifted(x, index, referenceX, halfDifferenceX) / scaleX -
                       meanX;
    };
    const auto centeredY = [&](Index index) {
      return scaleY == 0.0L
                 ? 0.0L
                 : shifted(y, index, referenceY, halfDifferenceY) / scaleY -
                       meanY;
    };

    CompensatedSum varianceXSum;
    CompensatedSum varianceYSum;
    for (Index i = 0U; i < n; ++i) {
      const Wide deviationX = centeredX(i);
      const Wide deviationY = centeredY(i);
      varianceXSum.add(deviationX * deviationX);
      varianceYSum.add(deviationY * deviationY);
    }
    const Wide varianceX = varianceXSum.value();
    const Wide varianceY = varianceYSum.value();
    if (correlation && (!(varianceX > 0.0L) || !(varianceY > 0.0L))) {
      return fail<LagSeries>(ErrorCode::InvalidDomain,
                             "cross-correlation requires positive variance");
    }

    int scaleExponentX = 0;
    int scaleExponentY = 0;
    const Wide scaleFractionX =
        scaleX == 0.0L ? 0.0L : std::frexp(scaleX, &scaleExponentX);
    const Wide scaleFractionY =
        scaleY == 0.0L ? 0.0L : std::frexp(scaleY, &scaleExponentY);
    if (halfDifferenceX) {
      ++scaleExponentX;
    }
    if (halfDifferenceY) {
      ++scaleExponentY;
    }
    for (Index output = 0U; output < lags.size(); ++output) {
      const auto magnitude = static_cast<Index>(lagMagnitude(lags[output]));
      const Index count = counts[output];
      CompensatedSum products;
      for (Index t = 0U; t < count; ++t) {
        const Index xIndex = lags[output] < 0 ? t : t + magnitude;
        const Index yIndex = lags[output] < 0 ? t + magnitude : t;
        products.add(centeredX(xIndex) * centeredY(yIndex));
      }
      const Wide sum = products.value();
      const Index denominator =
          normalization == LagNormalization::full_count ? n : count;
      Wide value = 0.0L;
      if (correlation) {
        const Wide covarianceScale =
            std::sqrt(varianceX) * std::sqrt(varianceY);
        value = (sum / covarianceScale) *
                (normalization == LagNormalization::full_count
                     ? 1.0L
                     : static_cast<Wide>(n) / denominator);
      } else if (sum != 0.0L && scaleX != 0.0L && scaleY != 0.0L) {
        int sumExponent = 0;
        const Wide sumFraction = std::frexp(sum / denominator, &sumExponent);
        const std::int64_t exponent = static_cast<std::int64_t>(sumExponent) +
                                      scaleExponentX + scaleExponentY;
        if (exponent > std::numeric_limits<int>::max() ||
            exponent < std::numeric_limits<int>::min()) {
          return fail<LagSeries>(
              ErrorCode::NonFiniteResult,
              "cross-covariance result is not representable");
        }
        value = std::scalbn(sumFraction * scaleFractionX * scaleFractionY,
                            static_cast<int>(exponent));
      }
      const auto result = static_cast<Scalar>(value);
      if (!std::isfinite(result)) {
        return fail<LagSeries>(ErrorCode::NonFiniteResult,
                               "cross-dependence result is not representable");
      }
      values.at(output) = result;
    }
    return Result<LagSeries>::success(
        LagSeries(std::move(ownedLags), std::move(values), std::move(counts)));
  } catch (const std::bad_alloc&) {
    return fail<LagSeries>(ErrorCode::ResourceLimit,
                           "cross-lag result allocation failed");
  } catch (const std::length_error&) {
    return fail<LagSeries>(ErrorCode::ResourceLimit,
                           "cross-lag result extent is unsupported");
  }
}

Result<LagSeries> autocovariance(ConstVectorView samples,
                                 std::span<const Index> lags,
                                 LagNormalization normalization,
                                 Limits limits) {
  return LagSeries::compute(samples, lags, normalization, limits, false);
}

Result<LagSeries> autocorrelation(ConstVectorView samples,
                                  std::span<const Index> lags,
                                  LagNormalization normalization,
                                  Limits limits) {
  return LagSeries::compute(samples, lags, normalization, limits, true);
}

Result<LagSeries> crossCovariance(ConstVectorView x, ConstVectorView y,
                                  std::span<const std::int64_t> lags,
                                  LagNormalization normalization,
                                  Limits limits) {
  return LagSeries::computeCross(x, y, lags, normalization, limits, false);
}

Result<LagSeries> crossCorrelation(ConstVectorView x, ConstVectorView y,
                                   std::span<const std::int64_t> lags,
                                   LagNormalization normalization,
                                   Limits limits) {
  return LagSeries::computeCross(x, y, lags, normalization, limits, true);
}

} // namespace vss::math::signal
