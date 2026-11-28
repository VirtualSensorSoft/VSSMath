#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <new>
#include <stdexcept>
#include <vss/math/Matrix.hpp>
#include <vss/math/linear/QR.hpp>
#include <vss/math/linear/TriangularSolve.hpp>
#include <vss/math/signal/Smoothing.hpp>

namespace vss::math::signal {
namespace {

template <class T> Result<T> failure(ErrorCode code, const char* message) {
  return Result<T>::failure(code, message);
}

bool checkedAdd(Extent lhs, Extent rhs, Extent& result) noexcept {
  if (rhs > std::numeric_limits<Extent>::max() - lhs) {
    return false;
  }
  result = lhs + rhs;
  return true;
}

bool checkedMultiply(Extent lhs, Extent rhs, Extent& result) noexcept {
  if (lhs != 0U && rhs > std::numeric_limits<Extent>::max() / lhs) {
    return false;
  }
  result = lhs * rhs;
  return true;
}

bool addExponent(std::int64_t& value, std::int64_t increment) noexcept {
  if ((increment > 0 &&
       value > std::numeric_limits<std::int64_t>::max() - increment) ||
      (increment < 0 &&
       value < std::numeric_limits<std::int64_t>::min() - increment)) {
    return false;
  }
  value += increment;
  return true;
}

struct BinaryScale final {
  Scalar fraction{0.5};
  std::int64_t exponent{1};
};

bool normalize(BinaryScale& scale) noexcept {
  int adjustment = 0;
  scale.fraction = std::frexp(scale.fraction, &adjustment);
  return addExponent(scale.exponent, adjustment);
}

bool derivativeScale(Index derivative, Scalar coordinateScale, Scalar step,
                     BinaryScale& result) noexcept {
  for (Index order = 0U; order < derivative; ++order) {
    const Scalar factorialTerm = static_cast<Scalar>(order + 1U);
    int factorialExponent = 0;
    const Scalar factorialFraction =
        std::frexp(factorialTerm, &factorialExponent);
    result.fraction *= factorialFraction;
    if (!addExponent(result.exponent, factorialExponent) ||
        !normalize(result)) {
      return false;
    }

    int coordinateExponent = 0;
    int stepExponent = 0;
    const Scalar coordinateFraction =
        std::frexp(coordinateScale, &coordinateExponent);
    const Scalar stepFraction = std::frexp(step, &stepExponent);
    result.fraction /= coordinateFraction;
    if (!addExponent(result.exponent,
                     -static_cast<std::int64_t>(coordinateExponent)) ||
        !normalize(result)) {
      return false;
    }
    result.fraction /= stepFraction;
    if (!addExponent(result.exponent,
                     -static_cast<std::int64_t>(stepExponent)) ||
        !normalize(result)) {
      return false;
    }
  }
  return true;
}

bool scaledProduct(Scalar sum, Scalar maxWeight, Scalar maxSample,
                   const BinaryScale& scale, Scalar& result) noexcept {
  if (sum == 0.0 || maxWeight == 0.0 || maxSample == 0.0) {
    result = 0.0;
    return true;
  }
  int sumExponent = 0;
  int weightExponent = 0;
  int sampleExponent = 0;
  Scalar fraction = std::frexp(sum, &sumExponent) *
                    std::frexp(maxWeight, &weightExponent) *
                    std::frexp(maxSample, &sampleExponent) * scale.fraction;
  std::int64_t exponent = scale.exponent;
  if (!addExponent(exponent, sumExponent) ||
      !addExponent(exponent, weightExponent) ||
      !addExponent(exponent, sampleExponent)) {
    return false;
  }
  int adjustment = 0;
  fraction = std::frexp(fraction, &adjustment);
  if (!addExponent(exponent, adjustment)) {
    return false;
  }

  if (exponent > std::numeric_limits<Scalar>::max_exponent ||
      exponent > std::numeric_limits<int>::max()) {
    return false;
  }
  if (exponent < std::numeric_limits<Scalar>::min_exponent -
                     std::numeric_limits<Scalar>::digits - 1) {
    result = std::copysign(0.0, sum);
    return true;
  }
  result = std::scalbn(fraction, static_cast<int>(exponent));
  return std::isfinite(result);
}

ErrorCode preflight(Extent sampleCount, Extent window, Extent columns,
                    Limits limits, bool singleton) noexcept {
  if (sampleCount > limits.elements) {
    return ErrorCode::ResourceLimit;
  }
  Extent scalarCount = sampleCount;
  if (!singleton) {
    Extent designElements = 0U;
    Extent triangularElements = 0U;
    Extent threeDesigns = 0U;
    Extent auxiliary = 0U;
    if (!checkedMultiply(window, columns, designElements) ||
        !checkedMultiply(columns, columns, triangularElements) ||
        !checkedMultiply(designElements, 3U, threeDesigns) ||
        !checkedAdd(columns, columns, auxiliary) ||
        !checkedAdd(auxiliary, columns, auxiliary) ||
        !checkedAdd(auxiliary, columns, auxiliary) ||
        !checkedAdd(auxiliary, window, auxiliary) ||
        !checkedAdd(scalarCount, threeDesigns, scalarCount) ||
        !checkedAdd(scalarCount, triangularElements, scalarCount) ||
        !checkedAdd(scalarCount, auxiliary, scalarCount)) {
      return ErrorCode::SizeOverflow;
    }
    Extent permutationBytes = 0U;
    if (!checkedMultiply(columns, sizeof(Index), permutationBytes)) {
      return ErrorCode::SizeOverflow;
    }
    if (scalarCount > limits.bytes / sizeof(Scalar) ||
        permutationBytes > limits.bytes - scalarCount * sizeof(Scalar)) {
      return ErrorCode::ResourceLimit;
    }
    return ErrorCode::None;
  }
  return scalarCount <= limits.bytes / sizeof(Scalar)
             ? ErrorCode::None
             : ErrorCode::ResourceLimit;
}

Extent windowStart(Index target, Extent sampleCount, Extent window,
                   Extent halfWindow) noexcept {
  const Extent maximumStart = sampleCount - window;
  if (target < halfWindow) {
    return 0U;
  }
  return std::min(target - halfWindow, maximumStart);
}

} // namespace

Result<Vector> savitzkyGolay(ConstVectorView samples, Index window,
                             Index degree, Index derivative, Scalar step,
                             Limits limits) {
  const Extent sampleCount = samples.size();
  if (!std::isfinite(step)) {
    return failure<Vector>(ErrorCode::NonFiniteInput,
                           "Savitzky-Golay step is not finite");
  }
  for (Index i = 0U; i < sampleCount; ++i) {
    if (!std::isfinite(samples.at(i))) {
      return failure<Vector>(ErrorCode::NonFiniteInput,
                             "Savitzky-Golay samples must be finite");
    }
  }
  if (step <= 0.0 || window == 0U || window % 2U == 0U || degree >= window ||
      derivative > degree) {
    return failure<Vector>(
        ErrorCode::InvalidDomain,
        "invalid Savitzky-Golay window, degree, derivative, or step");
  }
  if (sampleCount == 0U || window > sampleCount) {
    return failure<Vector>(ErrorCode::InsufficientSamples,
                           "Savitzky-Golay window exceeds available samples");
  }

  const Extent columns = degree + 1U;
  const Extent halfWindow = window / 2U;
  const Scalar coordinateScale =
      static_cast<Scalar>(std::max<Extent>(1U, halfWindow));
  const ErrorCode preflightError =
      preflight(sampleCount, window, columns, limits, window == 1U);
  if (preflightError != ErrorCode::None) {
    return failure<Vector>(
        preflightError,
        preflightError == ErrorCode::SizeOverflow
            ? "Savitzky-Golay workspace size overflows"
            : "Savitzky-Golay output or workspace exceeds limits");
  }

  try {
    Vector output(sampleCount);
    if (window == 1U) {
      for (Index i = 0U; i < sampleCount; ++i) {
        output.at(i) = samples.at(i);
      }
      return Result<Vector>::success(std::move(output));
    }

    BinaryScale scale;
    if (!derivativeScale(derivative, coordinateScale, step, scale)) {
      return failure<Vector>(
          ErrorCode::NonFiniteResult,
          "Savitzky-Golay derivative scale is unrepresentable");
    }

    for (Index targetOffset = 0U; targetOffset < window; ++targetOffset) {
      Matrix design(window, columns);
      for (Index row = 0U; row < window; ++row) {
        const Scalar coordinate =
            (row >= targetOffset ? static_cast<Scalar>(row - targetOffset)
                                 : -static_cast<Scalar>(targetOffset - row)) /
            coordinateScale;
        Scalar power = 1.0;
        for (Index column = 0U; column < columns; ++column) {
          design.at(row, column) = power;
          power *= coordinate;
        }
      }

      auto factorization = linear::factorizeQR(design.view());
      if (!factorization) {
        return failure<Vector>(factorization.status().code(),
                               "Savitzky-Golay design QR factorization failed");
      }
      if (!factorization.value().hasFullColumnRank()) {
        return failure<Vector>(
            ErrorCode::RankDeficient,
            "Savitzky-Golay local polynomial design is rank deficient");
      }

      const auto permutation = factorization.value().permutation();
      Index pivotPosition = columns;
      for (Index column = 0U; column < columns; ++column) {
        if (permutation[column] == derivative) {
          pivotPosition = column;
          break;
        }
      }
      if (pivotPosition == columns) {
        return failure<Vector>(
            ErrorCode::RankDeficient,
            "Savitzky-Golay QR permutation omitted a coefficient");
      }

      Vector rightHandSide(columns, 0.0);
      rightHandSide.at(pivotPosition) = 1.0;
      linear::TriangularSolveOptions solveOptions{linear::Triangle::Upper};
      solveOptions.transpose = linear::TransposeMode::Transpose;
      const auto solved =
          linear::solveTriangular(factorization.value().upperFactor().view(),
                                  rightHandSide.view(), solveOptions);
      if (!solved) {
        return failure<Vector>(solved.status().code(),
                               "Savitzky-Golay triangular solve failed");
      }

      const Matrix q = factorization.value().economyQ();
      Vector weights(window, 0.0);
      Scalar maximumWeight = 0.0;
      for (Index row = 0U; row < window; ++row) {
        Scalar sum = 0.0;
        Scalar correction = 0.0;
        for (Index column = 0U; column < columns; ++column) {
          const Scalar term =
              q.at(row, column) * solved.value().solution().at(column);
          const Scalar adjusted = term - correction;
          const Scalar next = sum + adjusted;
          correction = (next - sum) - adjusted;
          sum = next;
        }
        if (!std::isfinite(sum)) {
          return failure<Vector>(
              ErrorCode::NonFiniteResult,
              "Savitzky-Golay filter coefficients are non-finite");
        }
        weights.at(row) = sum;
        maximumWeight = std::max(maximumWeight, std::abs(sum));
      }

      for (Index target = 0U; target < sampleCount; ++target) {
        const Extent start =
            windowStart(target, sampleCount, window, halfWindow);
        if (target - start != targetOffset) {
          continue;
        }
        Scalar maximumSample = 0.0;
        for (Index j = 0U; j < window; ++j) {
          maximumSample =
              std::max(maximumSample, std::abs(samples.at(start + j)));
        }
        Scalar sum = 0.0;
        Scalar correction = 0.0;
        if (maximumWeight != 0.0 && maximumSample != 0.0) {
          for (Index j = 0U; j < window; ++j) {
            const Scalar term = (weights.at(j) / maximumWeight) *
                                (samples.at(start + j) / maximumSample);
            const Scalar adjusted = term - correction;
            const Scalar next = sum + adjusted;
            correction = (next - sum) - adjusted;
            sum = next;
          }
        }
        Scalar value = 0.0;
        if (!scaledProduct(sum, maximumWeight, maximumSample, scale, value)) {
          return failure<Vector>(ErrorCode::NonFiniteResult,
                                 "Savitzky-Golay result is not representable");
        }
        output.at(target) = value;
      }
    }
    return Result<Vector>::success(std::move(output));
  } catch (const std::bad_alloc&) {
    return failure<Vector>(ErrorCode::ResourceLimit,
                           "Savitzky-Golay allocation failed");
  } catch (const std::length_error&) {
    return failure<Vector>(ErrorCode::ResourceLimit,
                           "Savitzky-Golay storage size is not representable");
  }
}

} // namespace vss::math::signal
