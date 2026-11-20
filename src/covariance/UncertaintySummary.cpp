#include <cmath>
#include <limits>
#include <vss/math/Error.hpp>
#include <vss/math/covariance/UncertaintySummary.hpp>

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
