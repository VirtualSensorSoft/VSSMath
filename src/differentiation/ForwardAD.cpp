#include <algorithm>
#include <cmath>
#include <limits>
#include <span>
#include <string>
#include <utility>
#include <vector>
#include <vss/math/Array.hpp>
#include <vss/math/Core.hpp>
#include <vss/math/Error.hpp>
#include <vss/math/Limits.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/VectorView.hpp>
#include <vss/math/differentiation/ForwardAD.hpp>

namespace vss::math::differentiation {
namespace {

bool finite(const Dual& value) {
  if (!std::isfinite(value.primal)) {
    return false;
  }
  for (Index i = 0U; i < value.tangent.size(); ++i) {
    if (!std::isfinite(value.tangent.at(i))) {
      return false;
    }
  }
  return true;
}

Result<Dual> failure(ErrorCode code, const char* message) {
  return Result<Dual>::failure(code, message);
}

bool tangentWithinLimits(Index width, Limits limits) {
  return width < limits.elements && width < limits.bytes / sizeof(Scalar);
}

bool addProductWithin(Extent left, Extent right, Extent& total, Extent limit) {
  if (total > limit || (left != 0U && right > (limit - total) / left)) {
    return false;
  }
  total += left * right;
  return true;
}

} // namespace

Result<Dual> dualUnary(DualUnaryOp operation, const Dual& value,
                       Limits limits) {
  if (!finite(value)) {
    return failure(ErrorCode::NonFiniteInput,
                   "dual input contains a non-finite value");
  }
  if (!tangentWithinLimits(value.tangent.size(), limits)) {
    return failure(ErrorCode::ResourceLimit,
                   "dual tangent exceeds configured limits");
  }

  Scalar primal = 0.0;
  Scalar multiplier = 0.0;
  const Scalar a = value.primal;
  switch (operation) {
  case DualUnaryOp::negate:
    primal = -a;
    multiplier = -1.0;
    break;
  case DualUnaryOp::exp:
    primal = std::exp(a);
    multiplier = primal;
    break;
  case DualUnaryOp::log:
    if (!(a > 0.0)) {
      return failure(ErrorCode::InvalidDomain,
                     "dual logarithm requires a positive primal");
    }
    primal = std::log(a);
    multiplier = 1.0 / a;
    break;
  case DualUnaryOp::log1p:
    if (!(a > -1.0)) {
      return failure(ErrorCode::InvalidDomain,
                     "dual log1p requires primal greater than -1");
    }
    primal = std::log1p(a);
    multiplier = 1.0 / (1.0 + a);
    break;
  case DualUnaryOp::expm1:
    primal = std::expm1(a);
    multiplier = std::exp(a);
    break;
  case DualUnaryOp::sqrt:
    if (!(a > 0.0)) {
      return failure(ErrorCode::InvalidDomain,
                     "dual square root derivative requires positive primal");
    }
    primal = std::sqrt(a);
    multiplier = 0.5 / primal;
    break;
  case DualUnaryOp::sin:
    primal = std::sin(a);
    multiplier = std::cos(a);
    break;
  case DualUnaryOp::cos:
    primal = std::cos(a);
    multiplier = -std::sin(a);
    break;
  case DualUnaryOp::tanh: {
    primal = std::tanh(a);
    const Scalar tail = std::exp(-std::abs(a));
    const Scalar denominator = 1.0 + (tail * tail);
    multiplier = (4.0 * tail * tail) / (denominator * denominator);
    break;
  }
  case DualUnaryOp::abs:
    if (a == 0.0) {
      return failure(ErrorCode::InvalidDomain,
                     "dual absolute value is not differentiable at zero");
    }
    primal = std::abs(a);
    multiplier = std::signbit(a) ? -1.0 : 1.0;
    break;
  default:
    return failure(ErrorCode::InvalidDomain, "unknown dual unary operation");
  }
  if (!std::isfinite(primal) || !std::isfinite(multiplier)) {
    return failure(ErrorCode::NonFiniteResult,
                   "dual unary operation is not representable");
  }
  Vector tangent(value.tangent.size());
  for (Index i = 0U; i < tangent.size(); ++i) {
    tangent.at(i) = multiplier * value.tangent.at(i);
    if (!std::isfinite(tangent.at(i))) {
      return failure(ErrorCode::NonFiniteResult,
                     "dual unary tangent is not representable");
    }
  }
  return Result<Dual>::success(
      Dual{.primal = primal, .tangent = std::move(tangent)});
}

Result<Dual> dualBinary(DualBinaryOp operation, const Dual& left,
                        const Dual& right, Limits limits) {
  if (left.tangent.size() != right.tangent.size()) {
    return failure(ErrorCode::DimensionMismatch,
                   "dual binary operands have different tangent widths");
  }
  if (!finite(left) || !finite(right)) {
    return failure(ErrorCode::NonFiniteInput,
                   "dual binary input contains a non-finite value");
  }
  if (!tangentWithinLimits(left.tangent.size(), limits)) {
    return failure(ErrorCode::ResourceLimit,
                   "dual tangent exceeds configured limits");
  }

  const Scalar a = left.primal;
  const Scalar b = right.primal;
  long double primalWide = 0.0L;
  switch (operation) {
  case DualBinaryOp::add:
    primalWide = static_cast<long double>(a) + b;
    break;
  case DualBinaryOp::subtract:
    primalWide = static_cast<long double>(a) - b;
    break;
  case DualBinaryOp::multiply:
    primalWide = static_cast<long double>(a) * b;
    break;
  case DualBinaryOp::divide:
    if (b == 0.0) {
      return failure(ErrorCode::InvalidDomain,
                     "dual division requires a nonzero divisor");
    }
    primalWide = static_cast<long double>(a) / b;
    break;
  case DualBinaryOp::power:
    if (!(a > 0.0)) {
      return failure(ErrorCode::InvalidDomain,
                     "dual real power requires a positive base");
    }
    primalWide =
        std::pow(static_cast<long double>(a), static_cast<long double>(b));
    break;
  default:
    return failure(ErrorCode::InvalidDomain, "unknown dual binary operation");
  }
  const auto primal = static_cast<Scalar>(primalWide);
  if (!std::isfinite(primalWide) || !std::isfinite(primal)) {
    return failure(ErrorCode::NonFiniteResult,
                   "dual binary primal is not representable");
  }
  Vector tangent(left.tangent.size());
  for (Index i = 0U; i < tangent.size(); ++i) {
    const Scalar u = left.tangent.at(i);
    const Scalar v = right.tangent.at(i);
    long double derivative = 0.0L;
    switch (operation) {
    case DualBinaryOp::add:
      derivative = static_cast<long double>(u) + v;
      break;
    case DualBinaryOp::subtract:
      derivative = static_cast<long double>(u) - v;
      break;
    case DualBinaryOp::multiply:
      derivative =
          (static_cast<long double>(b) * u) + (static_cast<long double>(a) * v);
      break;
    case DualBinaryOp::divide:
      derivative = (static_cast<long double>(u) -
                    ((static_cast<long double>(a) / b) * v)) /
                   b;
      break;
    case DualBinaryOp::power:
      derivative = primalWide * (((static_cast<long double>(b) / a) * u) +
                                 (std::log(static_cast<long double>(a)) * v));
      break;
    default:
      return failure(ErrorCode::InvalidDomain, "unknown dual binary operation");
    }
    tangent.at(i) = static_cast<Scalar>(derivative);
    if (!std::isfinite(tangent.at(i))) {
      return failure(ErrorCode::NonFiniteResult,
                     "dual binary tangent is not representable");
    }
  }
  return Result<Dual>::success(
      Dual{.primal = primal, .tangent = std::move(tangent)});
}

Result<ADResult> forwardAutomaticDerivative(const DualFunction& function,
                                            ConstVectorView parameters,
                                            Index blockWidth, Limits limits) {
  if (!function) {
    return Result<ADResult>::failure(ErrorCode::CallbackFailure,
                                     "forward AD callback is empty");
  }
  if (blockWidth == 0U) {
    return Result<ADResult>::failure(ErrorCode::InvalidDomain,
                                     "forward AD block width must be positive");
  }
  for (Index i = 0U; i < parameters.size(); ++i) {
    if (!std::isfinite(parameters.at(i))) {
      return Result<ADResult>::failure(ErrorCode::NonFiniteInput,
                                       "forward AD parameters must be finite");
    }
  }
  const Index n = parameters.size();
  if (n == std::numeric_limits<Index>::max()) {
    return Result<ADResult>::failure(ErrorCode::ResourceLimit,
                                     "forward AD input size overflows");
  }
  const Index evaluations =
      n == 0U ? 1U
              : (n / blockWidth) + static_cast<Index>(n % blockWidth != 0U);
  const Index actualBlockWidth = std::min(blockWidth, n);
  Extent inputElements = 0U;
  Extent inputBytes = 0U;
  if (evaluations > limits.evaluations ||
      !addProductWithin(n, actualBlockWidth + 1U, inputElements,
                        limits.elements) ||
      !addProductWithin(n, sizeof(Dual) + (actualBlockWidth * sizeof(Scalar)),
                        inputBytes, limits.bytes)) {
    return Result<ADResult>::failure(
        ErrorCode::ResourceLimit,
        "forward AD preflight exceeds configured limits");
  }

  Vector values;
  Matrix jacobian;
  bool initialized = false;
  for (Index start = 0U; start < evaluations; ++start) {
    const Index first = n == 0U ? 0U : start * blockWidth;
    const Index width = n == 0U ? 0U : std::min(blockWidth, n - first);
    std::vector<Dual> inputs;
    inputs.reserve(n);
    for (Index i = 0U; i < n; ++i) {
      Vector tangent(width);
      if (i >= first && i < first + width) {
        tangent.at(i - first) = 1.0;
      }
      inputs.push_back(
          Dual{.primal = parameters.at(i), .tangent = std::move(tangent)});
    }
    Result<Array<Dual>> output = Result<Array<Dual>>::failure(
        ErrorCode::CallbackFailure, "forward AD callback did not return");
    try {
      output = function(std::span<const Dual>{inputs});
    } catch (...) {
      return Result<ADResult>::failure(
          ErrorCode::CallbackFailure, "forward AD callback threw an exception");
    }
    if (!output) {
      return Result<ADResult>::failure(output.status().code(),
                                       std::string(output.status().message()));
    }
    const auto entries = output.value().view();
    if (initialized && entries.size() != values.size()) {
      return Result<ADResult>::failure(
          ErrorCode::DimensionMismatch,
          "forward AD output count changed between blocks");
    }
    for (Index row = 0U; row < entries.size(); ++row) {
      const Dual& item = entries[row];
      if (!std::isfinite(item.primal)) {
        return Result<ADResult>::failure(
            ErrorCode::NonFiniteResult,
            "forward AD callback returned a non-finite primal");
      }
      if (item.tangent.size() != width) {
        return Result<ADResult>::failure(
            ErrorCode::DimensionMismatch,
            "forward AD output tangent width is incorrect");
      }
      for (Index j = 0U; j < width; ++j) {
        if (!std::isfinite(item.tangent.at(j))) {
          return Result<ADResult>::failure(
              ErrorCode::NonFiniteResult,
              "forward AD callback returned a non-finite tangent");
        }
      }
      if (initialized && start != 0U && item.primal != values.at(row)) {
        return Result<ADResult>::failure(
            ErrorCode::InvalidDomain,
            "forward AD primal changed between seed blocks");
      }
    }
    Extent liveElements = 0U;
    Extent liveBytes = 0U;
    if (!addProductWithin(n, width + 1U, liveElements, limits.elements) ||
        !addProductWithin(n, sizeof(Dual) + (width * sizeof(Scalar)), liveBytes,
                          limits.bytes) ||
        !addProductWithin(entries.size(), width + 1U, liveElements,
                          limits.elements) ||
        !addProductWithin(entries.size(),
                          sizeof(Dual) + (width * sizeof(Scalar)), liveBytes,
                          limits.bytes)) {
      return Result<ADResult>::failure(
          ErrorCode::ResourceLimit,
          "forward AD callback storage exceeds limits");
    }
    if (n == std::numeric_limits<Index>::max() ||
        entries.size() > std::numeric_limits<Extent>::max() / (n + 1U) ||
        entries.size() * (n + 1U) >
            std::numeric_limits<Extent>::max() / sizeof(Scalar)) {
      return Result<ADResult>::failure(ErrorCode::ResourceLimit,
                                       "forward AD output size overflows");
    }
    const Extent resultElements = entries.size() * (n + 1U);
    const Extent resultBytes = resultElements * sizeof(Scalar);
    if (initialized) {
      if (!addProductWithin(resultElements, 1U, liveElements,
                            limits.elements) ||
          !addProductWithin(resultBytes, 1U, liveBytes, limits.bytes)) {
        return Result<ADResult>::failure(
            ErrorCode::ResourceLimit, "forward AD live result exceeds limits");
      }
    } else {
      if (!addProductWithin(resultElements, 1U, liveElements,
                            limits.elements) ||
          !addProductWithin(resultBytes, 1U, liveBytes, limits.bytes)) {
        return Result<ADResult>::failure(ErrorCode::ResourceLimit,
                                         "forward AD result exceeds limits");
      }
      values = Vector(entries.size());
      jacobian = Matrix(entries.size(), n);
      initialized = true;
    }
    for (Index row = 0U; row < entries.size(); ++row) {
      if (start == 0U) {
        values.at(row) = entries[row].primal;
      }
      for (Index j = 0U; j < width; ++j) {
        jacobian.at(row, first + j) = entries[row].tangent.at(j);
      }
    }
  }
  if (!initialized) {
    // For an empty input the loop still executes once and establishes output.
    return Result<ADResult>::failure(
        ErrorCode::CallbackFailure, "forward AD did not evaluate its callback");
  }
  return Result<ADResult>::success(
      ADResult{.value = std::move(values), .jacobian = std::move(jacobian)});
}

} // namespace vss::math::differentiation
