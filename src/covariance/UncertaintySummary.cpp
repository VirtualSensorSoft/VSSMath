#include <algorithm>
#include <cmath>
#include <limits>
#include <new>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <vss/math/Core.hpp>
#include <vss/math/Error.hpp>
#include <vss/math/Limits.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/MatrixView.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/Vector.hpp>
#include <vss/math/VectorView.hpp>
#include <vss/math/covariance/PropagationResult.hpp>
#include <vss/math/covariance/UncertaintySummary.hpp>
#include <vss/math/linear/SymmetricEigen.hpp>

namespace vss::math::covariance {
namespace {

template <class T> Result<T> failure(ErrorCode code, const char* message) {
  return Result<T>::failure(code, message);
}

bool finite(ConstMatrixView matrix) noexcept {
  for (Index row = 0U; row < matrix.rows(); ++row) {
    for (Index column = 0U; column < matrix.columns(); ++column) {
      if (!std::isfinite(matrix.at(row, column))) {
        return false;
      }
    }
  }
  return true;
}

bool representable(long double value) noexcept {
  return std::isfinite(value) &&
         std::abs(value) <= std::numeric_limits<Scalar>::max();
}

bool checkedMultiply(Index lhs, Index rhs, Index& result) noexcept {
  if (lhs != 0U && rhs > std::numeric_limits<Index>::max() / lhs) {
    return false;
  }
  result = lhs * rhs;
  return true;
}

bool checkedAdd(Index lhs, Index rhs, Index& result) noexcept {
  if (rhs > std::numeric_limits<Index>::max() - lhs) {
    return false;
  }
  result = lhs + rhs;
  return true;
}

bool finite(ConstVectorView vector) noexcept {
  for (Index index = 0U; index < vector.size(); ++index) {
    if (!std::isfinite(vector.at(index))) {
      return false;
    }
  }
  return true;
}

bool summaryBudget(Index dimension, bool hasBounds, Limits limits) noexcept {
  Index square = 0U;
  Index spectralWorkspace = 0U;
  Index workspace = 0U;
  Index output = 0U;
  Index boundedOutput = 0U;
  if (!checkedMultiply(dimension, dimension, square) ||
      !checkedMultiply(square, 3U, spectralWorkspace) ||
      !checkedAdd(spectralWorkspace, dimension, workspace) ||
      !checkedMultiply(dimension, hasBounds ? 5U : 3U, output)) {
    return false;
  }
  boundedOutput = std::max(workspace, output);
  return boundedOutput <= limits.elements &&
         boundedOutput <= limits.bytes / sizeof(Scalar);
}

Result<bool> covarianceIsAdmissible(ConstMatrixView covariance) {
  const Index dimension = covariance.rows();
  Scalar scale = 0.0;
  for (Index row = 0U; row < dimension; ++row) {
    for (Index column = 0U; column < dimension; ++column) {
      scale = std::max(scale, std::abs(covariance.at(row, column)));
    }
  }
  const long double allowance =
      64.0L * static_cast<long double>(std::max<Index>(1U, dimension)) *
      std::numeric_limits<Scalar>::epsilon();
  for (Index first = 0U; first < dimension; ++first) {
    for (Index second = first + 1U; second < dimension; ++second) {
      const Scalar upper = covariance.at(first, second);
      const Scalar lower = covariance.at(second, first);
      const Scalar difference =
          scale == 0.0 ? 0.0 : std::abs((upper / scale) - (lower / scale));
      if (static_cast<long double>(difference) > allowance) {
        return Result<bool>::failure(ErrorCode::NotSymmetric,
                                     "covariance failed scaled symmetry check");
      }
    }
  }
  for (Index row = 0U; row < dimension; ++row) {
    if (covariance.at(row, row) < 0.0) {
      return Result<bool>::success(false);
    }
  }
  if (scale == 0.0 || dimension == 1U) {
    return Result<bool>::success(true);
  }

  Matrix normalized(dimension, dimension, 0.0);
  for (Index first = 0U; first < dimension; ++first) {
    normalized.at(first, first) = covariance.at(first, first) / scale;
    for (Index second = first + 1U; second < dimension; ++second) {
      const Scalar symmetric = std::midpoint(covariance.at(first, second),
                                             covariance.at(second, first));
      const Scalar value = symmetric / scale;
      normalized.at(first, second) = value;
      normalized.at(second, first) = value;
    }
  }
  linear::SymmetricEigenOptions options{};
  options.rankTolerance = {.absolute = 0.0, .relative = 0.0};
  options.symmetryTolerance = {.absolute = 0.0, .relative = 0.0};
  options.maximumSweeps = 64U;
  const auto eigen =
      linear::decomposeSymmetricEigen(normalized.view(), options);
  if (!eigen.hasValue()) {
    return Result<bool>::failure(eigen.status().code(),
                                 std::string(eigen.status().message()));
  }
  for (Index index = 0U; index < dimension; ++index) {
    if (static_cast<long double>(eigen.value().eigenvalues().at(index)) <
        -allowance) {
      return Result<bool>::success(false);
    }
  }
  return Result<bool>::success(true);
}

struct ScaledValue final {
  long double mantissa{};
  int exponent{};
};

class ScaledAccumulator final {
public:
  void addProduct(Scalar left, Scalar right) noexcept {
    if (left == 0.0 || right == 0.0) {
      return;
    }
    int leftExponent = 0;
    int rightExponent = 0;
    const long double leftMantissa =
        std::frexp(static_cast<long double>(left), &leftExponent);
    const long double rightMantissa =
        std::frexp(static_cast<long double>(right), &rightExponent);
    addTerm(leftMantissa * rightMantissa, leftExponent + rightExponent);
  }

  void addProduct(ScaledValue left, Scalar right) noexcept {
    if (left.mantissa == 0.0L || right == 0.0) {
      return;
    }
    int rightExponent = 0;
    const long double rightMantissa =
        std::frexp(static_cast<long double>(right), &rightExponent);
    addTerm(left.mantissa * rightMantissa, left.exponent + rightExponent);
  }

  [[nodiscard]] ScaledValue scaledValue() const noexcept {
    if (!hasValue_ || sum_ == 0.0L) {
      return {};
    }
    int adjustment = 0;
    const long double mantissa = std::frexp(sum_, &adjustment);
    return {.mantissa = mantissa, .exponent = exponent_ + adjustment};
  }

  [[nodiscard]] bool scalarValue(Scalar& output) const noexcept {
    const ScaledValue scaled = scaledValue();
    if (scaled.mantissa == 0.0L) {
      output = 0.0;
      return true;
    }
    if (scaled.exponent > std::numeric_limits<Scalar>::max_exponent) {
      return false;
    }
    output = std::scalbn(static_cast<Scalar>(scaled.mantissa), scaled.exponent);
    return std::isfinite(output);
  }

private:
  void addTerm(long double mantissa, int exponent) noexcept {
    if (!hasValue_) {
      exponent_ = exponent;
      sum_ = mantissa;
      hasValue_ = true;
      return;
    }
    if (exponent > exponent_) {
      const int difference = exponent_ - exponent;
      sum_ = std::scalbn(sum_, difference);
      compensation_ = std::scalbn(compensation_, difference);
      exponent_ = exponent;
    }
    const long double term = std::scalbn(mantissa, exponent - exponent_);
    const long double adjusted = term - compensation_;
    const long double next = sum_ + adjusted;
    compensation_ = (next - sum_) - adjusted;
    sum_ = next;
  }
  long double sum_{};
  long double compensation_{};
  int exponent_{};
  bool hasValue_{};
};

} // namespace

Result<UncertaintySummary>
summarizeUncertainty(const PropagationResult& value, Scalar coverageFactor,
                     std::optional<ConstVectorView> lowerQuantile,
                     std::optional<ConstVectorView> upperQuantile,
                     Limits limits) {
  try {
    const Index dimension = value.mean.size();
    if (value.covariance.rows() != dimension ||
        value.covariance.columns() != dimension) {
      return failure<UncertaintySummary>(
          ErrorCode::DimensionMismatch,
          "propagated mean and covariance dimensions do not match");
    }
    if (lowerQuantile.has_value() != upperQuantile.has_value()) {
      return failure<UncertaintySummary>(
          ErrorCode::InvalidDomain,
          "lower and upper uncertainty bounds must be supplied together");
    }
    const bool hasBounds = lowerQuantile.has_value();
    if (hasBounds && (lowerQuantile->size() != dimension ||
                      upperQuantile->size() != dimension)) {
      return failure<UncertaintySummary>(
          ErrorCode::DimensionMismatch,
          "uncertainty bounds have the wrong size");
    }
    if (dimension == 0U) {
      return failure<UncertaintySummary>(
          ErrorCode::InsufficientSamples,
          "uncertainty summary needs a nonempty mean");
    }
    if (!summaryBudget(dimension, hasBounds, limits)) {
      return failure<UncertaintySummary>(
          ErrorCode::ResourceLimit,
          "uncertainty summary exceeds supplied limits");
    }
    if (!finite(value.mean.view()) || !finite(value.covariance.view()) ||
        !std::isfinite(coverageFactor) ||
        (hasBounds && (!finite(*lowerQuantile) || !finite(*upperQuantile)))) {
      return failure<UncertaintySummary>(
          ErrorCode::NonFiniteInput,
          "uncertainty summary inputs must be finite");
    }
    if (coverageFactor < 0.0) {
      return failure<UncertaintySummary>(ErrorCode::InvalidDomain,
                                         "coverage factor must be nonnegative");
    }
    if (hasBounds) {
      for (Index index = 0U; index < dimension; ++index) {
        if (lowerQuantile->at(index) > upperQuantile->at(index)) {
          return failure<UncertaintySummary>(ErrorCode::InvalidDomain,
                                             "uncertainty bounds are reversed");
        }
      }
    }
    const auto admissible = covarianceIsAdmissible(value.covariance.view());
    if (!admissible.hasValue()) {
      return Result<UncertaintySummary>::failure(
          admissible.status().code(),
          std::string(admissible.status().message()));
    }
    if (!admissible.value()) {
      return failure<UncertaintySummary>(
          ErrorCode::NotPositiveSemidefinite,
          "propagated covariance is not positive semidefinite");
    }

    Vector standard(dimension, 0.0);
    Vector expanded(dimension, 0.0);
    for (Index index = 0U; index < dimension; ++index) {
      const Scalar diagonal = value.covariance.at(index, index);
      const Scalar deviation = diagonal == 0.0 ? 0.0 : std::sqrt(diagonal);
      const long double expandedValue =
          static_cast<long double>(coverageFactor) * deviation;
      if (!representable(expandedValue)) {
        return failure<UncertaintySummary>(
            ErrorCode::NonFiniteResult,
            "expanded uncertainty is not representable");
      }
      standard.at(index) = deviation;
      expanded.at(index) = static_cast<Scalar>(expandedValue);
    }
    std::optional<Vector> lower;
    std::optional<Vector> upper;
    if (hasBounds) {
      Vector lowerValues(dimension);
      Vector upperValues(dimension);
      for (Index index = 0U; index < dimension; ++index) {
        lowerValues.at(index) = lowerQuantile->at(index);
        upperValues.at(index) = upperQuantile->at(index);
      }
      lower.emplace(std::move(lowerValues));
      upper.emplace(std::move(upperValues));
    }
    return Result<UncertaintySummary>::success(
        UncertaintySummary{.nominal = value.mean,
                           .standard = std::move(standard),
                           .expanded = std::move(expanded),
                           .lower = std::move(lower),
                           .upper = std::move(upper),
                           .coverage_factor = coverageFactor});
  } catch (const std::bad_alloc&) {
    return failure<UncertaintySummary>(ErrorCode::ResourceLimit,
                                       "uncertainty summary allocation failed");
  } catch (const std::length_error&) {
    return failure<UncertaintySummary>(
        ErrorCode::ResourceLimit,
        "uncertainty summary exceeds container limits");
  }
}

Result<Matrix> crossCovarianceFromMaps(ConstMatrixView left,
                                       ConstMatrixView crossCovariance,
                                       ConstMatrixView right, Limits limits) {
  try {
    if (left.columns() != crossCovariance.rows() ||
        crossCovariance.columns() != right.columns()) {
      return failure<Matrix>(
          ErrorCode::DimensionMismatch,
          "cross-covariance maps and source block do not conform");
    }
    const Index outputRows = left.rows();
    const Index outputColumns = right.rows();
    if (outputRows != 0U &&
        outputColumns > std::numeric_limits<Index>::max() / outputRows) {
      return failure<Matrix>(ErrorCode::ResourceLimit,
                             "cross-covariance output size overflows");
    }
    const Index outputElements = outputRows * outputColumns;
    if (outputElements > std::numeric_limits<Index>::max() / sizeof(Scalar)) {
      return failure<Matrix>(ErrorCode::ResourceLimit,
                             "cross-covariance byte size overflows");
    }
    const Index outputBytes = outputElements * sizeof(Scalar);
    const bool needsIntermediate = outputRows != 0U && outputColumns != 0U &&
                                   left.columns() != 0U &&
                                   crossCovariance.columns() != 0U;
    Index intermediateElements = 0U;
    Index intermediateBytes = 0U;
    if (needsIntermediate &&
        (!checkedMultiply(outputRows, crossCovariance.columns(),
                          intermediateElements) ||
         intermediateElements >
             std::numeric_limits<Index>::max() / sizeof(ScaledValue))) {
      return failure<Matrix>(ErrorCode::ResourceLimit,
                             "cross-covariance workspace size overflows");
    }
    intermediateBytes = intermediateElements * sizeof(ScaledValue);
    Index totalElements = 0U;
    if (!checkedAdd(outputElements, intermediateElements, totalElements) ||
        intermediateBytes > std::numeric_limits<Index>::max() - outputBytes ||
        totalElements > limits.elements ||
        outputBytes + intermediateBytes > limits.bytes) {
      return failure<Matrix>(
          ErrorCode::ResourceLimit,
          "cross-covariance workspace exceeds supplied limits");
    }
    if (!finite(left) || !finite(crossCovariance) || !finite(right)) {
      return failure<Matrix>(ErrorCode::NonFiniteInput,
                             "cross-covariance inputs must be finite");
    }

    Matrix result(outputRows, outputColumns, 0.0);
    if (!needsIntermediate) {
      return Result<Matrix>::success(std::move(result));
    }

    std::vector<ScaledValue> intermediate(intermediateElements);
    for (Index outputRow = 0U; outputRow < outputRows; ++outputRow) {
      for (Index rightCoordinate = 0U;
           rightCoordinate < crossCovariance.columns(); ++rightCoordinate) {
        ScaledAccumulator sum;
        for (Index leftCoordinate = 0U; leftCoordinate < left.columns();
             ++leftCoordinate) {
          sum.addProduct(left.at(outputRow, leftCoordinate),
                         crossCovariance.at(leftCoordinate, rightCoordinate));
        }
        intermediate.at((outputRow * crossCovariance.columns()) +
                        rightCoordinate) = sum.scaledValue();
      }
    }
    for (Index outputRow = 0U; outputRow < outputRows; ++outputRow) {
      for (Index outputComponent = 0U; outputComponent < outputColumns;
           ++outputComponent) {
        ScaledAccumulator sum;
        for (Index rightCoordinate = 0U;
             rightCoordinate < crossCovariance.columns(); ++rightCoordinate) {
          sum.addProduct(
              intermediate.at((outputRow * crossCovariance.columns()) +
                              rightCoordinate),
              right.at(outputComponent, rightCoordinate));
        }
        Scalar value = 0.0;
        if (!sum.scalarValue(value)) {
          return failure<Matrix>(
              ErrorCode::NonFiniteResult,
              "cross-covariance result is not representable");
        }
        result.at(outputRow, outputComponent) = value;
      }
    }
    return Result<Matrix>::success(std::move(result));
  } catch (const std::bad_alloc&) {
    return failure<Matrix>(ErrorCode::ResourceLimit,
                           "cross-covariance allocation failed");
  } catch (const std::length_error&) {
    return failure<Matrix>(ErrorCode::ResourceLimit,
                           "cross-covariance exceeds container limits");
  }
}

} // namespace vss::math::covariance
