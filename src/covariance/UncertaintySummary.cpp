#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <string>
#include <utility>
#include <vss/math/Error.hpp>
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
  for (Index row = 0U; row < dimension; ++row) {
    for (Index column = row + 1U; column < dimension; ++column) {
      const Scalar upper = covariance.at(row, column);
      const Scalar lower = covariance.at(column, row);
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
  for (Index row = 0U; row < dimension; ++row) {
    normalized.at(row, row) = covariance.at(row, row) / scale;
    for (Index column = row + 1U; column < dimension; ++column) {
      const Scalar symmetric =
          std::midpoint(covariance.at(row, column), covariance.at(column, row));
      const Scalar value = symmetric / scale;
      normalized.at(row, column) = value;
      normalized.at(column, row) = value;
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

class ScaledAccumulator final {
public:
  void add(Scalar left, Scalar cross, Scalar right) noexcept {
    if (left == 0.0 || cross == 0.0 || right == 0.0) {
      return;
    }
    int leftExponent = 0;
    int crossExponent = 0;
    int rightExponent = 0;
    const long double leftMantissa =
        std::frexp(static_cast<long double>(left), &leftExponent);
    const long double crossMantissa =
        std::frexp(static_cast<long double>(cross), &crossExponent);
    const long double rightMantissa =
        std::frexp(static_cast<long double>(right), &rightExponent);
    const int exponent = leftExponent + crossExponent + rightExponent;
    const long double mantissa = leftMantissa * crossMantissa * rightMantissa;
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

  [[nodiscard]] long double value() const noexcept {
    return hasValue_ ? std::scalbn(sum_, exponent_) : 0.0L;
  }

private:
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
        ErrorCode::DimensionMismatch, "uncertainty bounds have the wrong size");
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
        ErrorCode::NonFiniteInput, "uncertainty summary inputs must be finite");
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
        admissible.status().code(), std::string(admissible.status().message()));
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
    lower.emplace(dimension);
    upper.emplace(dimension);
    for (Index index = 0U; index < dimension; ++index) {
      lower->at(index) = lowerQuantile->at(index);
      upper->at(index) = upperQuantile->at(index);
    }
  }
  return Result<UncertaintySummary>::success(
      UncertaintySummary{value.mean, std::move(standard), std::move(expanded),
                         std::move(lower), std::move(upper), coverageFactor});
}

Result<Matrix> crossCovarianceFromMaps(ConstMatrixView left,
                                       ConstMatrixView crossCovariance,
                                       ConstMatrixView right, Limits limits) {
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
  const Index workspaceBytes =
      outputElements == 0U ? 0U : sizeof(ScaledAccumulator);
  if (outputElements > limits.elements ||
      workspaceBytes > std::numeric_limits<Index>::max() - outputBytes ||
      outputBytes + workspaceBytes > limits.bytes) {
    return failure<Matrix>(ErrorCode::ResourceLimit,
                           "cross-covariance output exceeds supplied limits");
  }
  if (!finite(left) || !finite(crossCovariance) || !finite(right)) {
    return failure<Matrix>(ErrorCode::NonFiniteInput,
                           "cross-covariance inputs must be finite");
  }

  Matrix result(outputRows, outputColumns, 0.0);
  for (Index row = 0U; row < outputRows; ++row) {
    for (Index column = 0U; column < outputColumns; ++column) {
      ScaledAccumulator sum;
      for (Index leftCoordinate = 0U; leftCoordinate < left.columns();
           ++leftCoordinate) {
        for (Index rightCoordinate = 0U; rightCoordinate < right.columns();
             ++rightCoordinate) {
          sum.add(left.at(row, leftCoordinate),
                  crossCovariance.at(leftCoordinate, rightCoordinate),
                  right.at(column, rightCoordinate));
        }
      }
      const long double value = sum.value();
      if (!representable(value)) {
        return failure<Matrix>(ErrorCode::NonFiniteResult,
                               "cross-covariance result is not representable");
      }
      result.at(row, column) = static_cast<Scalar>(value);
    }
  }
  return Result<Matrix>::success(std::move(result));
}

} // namespace vss::math::covariance
