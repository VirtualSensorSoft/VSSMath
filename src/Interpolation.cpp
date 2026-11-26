#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <vss/math/Core.hpp>
#include <vss/math/Error.hpp>
#include <vss/math/Interpolation.hpp>
#include <vss/math/Limits.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/Vector.hpp>
#include <vss/math/VectorView.hpp>

namespace vss::math {
namespace {

using Wide = long double;

Status validateSamples(ConstVectorView nodes, ConstVectorView values) {
  if (nodes.size() != values.size()) {
    return Status::failure(ErrorCode::DimensionMismatch,
                           "interpolation nodes and values differ in extent");
  }
  if (nodes.size() < 2U) {
    return Status::failure(ErrorCode::InsufficientSamples,
                           "interpolation requires at least two knots");
  }
  for (Index i = 0U; i < nodes.size(); ++i) {
    if (!std::isfinite(nodes.at(i)) || !std::isfinite(values.at(i))) {
      return Status::failure(ErrorCode::NonFiniteInput,
                             "interpolation input contains a non-finite value");
    }
    if (i > 0U && !(nodes.at(i) > nodes.at(i - 1U))) {
      return Status::failure(ErrorCode::InvalidDomain,
                             "interpolation knots must be strictly increasing");
    }
  }
  return Status::success();
}

bool validExtrapolation(Extrapolation value) noexcept {
  return value == Extrapolation::reject || value == Extrapolation::hold ||
         value == Extrapolation::linear;
}

bool finiteScalar(Wide value, Scalar& result) noexcept {
  if (!std::isfinite(value) ||
      std::abs(value) > static_cast<Wide>(std::numeric_limits<Scalar>::max())) {
    return false;
  }
  result = static_cast<Scalar>(value);
  return std::isfinite(result);
}

bool checkedAdd(std::size_t left, std::size_t right,
                std::size_t& result) noexcept {
  if (right > std::numeric_limits<std::size_t>::max() - left) {
    return false;
  }
  result = left + right;
  return true;
}

bool checkedMultiply(std::size_t left, std::size_t right,
                     std::size_t& result) noexcept {
  if (left != 0U && right > std::numeric_limits<std::size_t>::max() / left) {
    return false;
  }
  result = left * right;
  return true;
}

bool splineLimitsAllow(Extent n, Limits limits) noexcept {
  std::size_t coefficientCount = 0U;
  std::size_t nodeCount = 0U;
  std::size_t outputCount = 0U;
  std::size_t scratchCount = 0U;
  std::size_t outputBytes = 0U;
  std::size_t scratchBytes = 0U;
  std::size_t totalBytes = 0U;
  if (!checkedMultiply(n - 1U, 4U, coefficientCount) ||
      !checkedAdd(coefficientCount, n, outputCount) ||
      outputCount > limits.elements ||
      !checkedMultiply(n, sizeof(Scalar), nodeCount) ||
      !checkedMultiply(coefficientCount, sizeof(Scalar), outputBytes) ||
      !checkedAdd(nodeCount, outputBytes, outputBytes) ||
      !checkedAdd(outputBytes, 2U * sizeof(Extent), outputBytes) ||
      !checkedMultiply(n, 9U, scratchCount) ||
      !checkedMultiply(scratchCount, sizeof(Wide), scratchBytes) ||
      !checkedAdd(outputBytes, scratchBytes, totalBytes)) {
    return false;
  }
  return totalBytes <= limits.bytes;
}

Result<Scalar> invalidScalar(ErrorCode code, const char* message) {
  return Result<Scalar>::failure(code, message);
}

bool validBoundary(const SplineSpec& policy) noexcept {
  if (policy.boundary == SplineBoundary::natural) {
    return !policy.left_derivative.has_value() &&
           !policy.right_derivative.has_value();
  }
  if (policy.boundary == SplineBoundary::clamped) {
    return policy.left_derivative.has_value() &&
           policy.right_derivative.has_value();
  }
  return false;
}

Wide polynomial(const Matrix& coefficients, Index row, Wide u) {
  Wide value = static_cast<Wide>(coefficients.at(row, 3U));
  value = value * u + static_cast<Wide>(coefficients.at(row, 2U));
  value = value * u + static_cast<Wide>(coefficients.at(row, 1U));
  return value * u + static_cast<Wide>(coefficients.at(row, 0U));
}

} // namespace

Result<Scalar> interpolate(ConstVectorView nodes, ConstVectorView values,
                           Scalar x, const InterpolateSpec& policy) {
  const Status samplesStatus = validateSamples(nodes, values);
  if (!samplesStatus.ok()) {
    return Result<Scalar>::failure(samplesStatus.code(),
                                   std::string(samplesStatus.message()));
  }
  if (!std::isfinite(x)) {
    return invalidScalar(ErrorCode::NonFiniteInput,
                         "interpolation query must be finite");
  }
  if (!validExtrapolation(policy.extrapolation)) {
    return invalidScalar(ErrorCode::InvalidDomain,
                         "unknown interpolation extrapolation policy");
  }
  if (policy.maximum_gap.has_value() &&
      (!std::isfinite(*policy.maximum_gap) || *policy.maximum_gap <= 0.0)) {
    return invalidScalar(
        ErrorCode::InvalidTolerance,
        "maximum interpolation gap must be finite and positive");
  }

  Index lower = 0U;
  Index upper = nodes.size() - 1U;
  while (lower < upper) {
    const Index middle = lower + (upper - lower) / 2U;
    if (nodes.at(middle) < x) {
      lower = middle + 1U;
    } else {
      upper = middle;
    }
  }
  if (nodes.at(lower) == x) {
    return Result<Scalar>::success(values.at(lower));
  }

  Index left = 0U;
  if (x < nodes.at(0U)) {
    if (policy.extrapolation == Extrapolation::reject) {
      return invalidScalar(ErrorCode::InvalidDomain,
                           "interpolation query precedes the knot domain");
    }
    if (policy.extrapolation == Extrapolation::hold) {
      return Result<Scalar>::success(values.at(0U));
    }
  } else if (x > nodes.at(nodes.size() - 1U)) {
    if (policy.extrapolation == Extrapolation::reject) {
      return invalidScalar(ErrorCode::InvalidDomain,
                           "interpolation query follows the knot domain");
    }
    if (policy.extrapolation == Extrapolation::hold) {
      return Result<Scalar>::success(values.at(values.size() - 1U));
    }
    left = nodes.size() - 2U;
  } else {
    left = lower - 1U;
  }

  const Wide x0 = static_cast<Wide>(nodes.at(left));
  const Wide x1 = static_cast<Wide>(nodes.at(left + 1U));
  const Wide width = x1 - x0;
  if (policy.maximum_gap.has_value() &&
      width > static_cast<Wide>(*policy.maximum_gap)) {
    return invalidScalar(ErrorCode::InvalidDomain,
                         "interpolation segment exceeds maximum gap");
  }
  const Wide fraction = (static_cast<Wide>(x) - x0) / width;
  const Wide y0 = static_cast<Wide>(values.at(left));
  const Wide y1 = static_cast<Wide>(values.at(left + 1U));
  Scalar result = 0.0;
  if (!finiteScalar(y0 + fraction * (y1 - y0), result)) {
    return invalidScalar(ErrorCode::NonFiniteResult,
                         "interpolated value is not representable");
  }
  return Result<Scalar>::success(result);
}

Result<Spline> makeSpline(ConstVectorView nodes, ConstVectorView values,
                          const SplineSpec& policy, Limits limits) {
  const Status samplesStatus = validateSamples(nodes, values);
  if (!samplesStatus.ok()) {
    return Result<Spline>::failure(samplesStatus.code(),
                                   std::string(samplesStatus.message()));
  }
  if (!validBoundary(policy)) {
    return Result<Spline>::failure(
        ErrorCode::InvalidDomain,
        "spline boundary kind and endpoint derivatives are inconsistent");
  }
  if ((policy.left_derivative.has_value() &&
       !std::isfinite(*policy.left_derivative)) ||
      (policy.right_derivative.has_value() &&
       !std::isfinite(*policy.right_derivative))) {
    return Result<Spline>::failure(ErrorCode::NonFiniteInput,
                                   "spline endpoint derivative is non-finite");
  }
  if (!splineLimitsAllow(nodes.size(), limits)) {
    return Result<Spline>::failure(ErrorCode::ResourceLimit,
                                   "spline output or workspace exceeds limits");
  }

  const Extent n = nodes.size();
  try {
    std::vector<Wide> lower(n, 0.0L);
    std::vector<Wide> diagonal(n, 0.0L);
    std::vector<Wide> upper(n, 0.0L);
    std::vector<Wide> rhs(n, 0.0L);
    std::vector<Wide> solution(n, 0.0L);
    std::vector<Wide> widths(n - 1U, 0.0L);
    for (Index i = 0U; i + 1U < n; ++i) {
      widths[i] =
          static_cast<Wide>(nodes.at(i + 1U)) - static_cast<Wide>(nodes.at(i));
    }

    if (policy.boundary == SplineBoundary::natural) {
      diagonal[0] = 1.0L;
      diagonal[n - 1U] = 1.0L;
      for (Index i = 1U; i + 1U < n; ++i) {
        lower[i] = widths[i - 1U];
        diagonal[i] = 2.0L * (widths[i - 1U] + widths[i]);
        upper[i] = widths[i];
        const Wide previousSlope =
            (static_cast<Wide>(values.at(i)) - values.at(i - 1U)) /
            widths[i - 1U];
        const Wide nextSlope =
            (static_cast<Wide>(values.at(i + 1U)) - values.at(i)) / widths[i];
        rhs[i] = 6.0L * (nextSlope - previousSlope);
      }
    } else {
      const Wide firstSlope =
          (static_cast<Wide>(values.at(1U)) - values.at(0U)) / widths[0];
      diagonal[0] = 2.0L * widths[0];
      upper[0] = widths[0];
      rhs[0] = 6.0L * (firstSlope - *policy.left_derivative);
      const Index last = n - 1U;
      const Wide lastSlope =
          (static_cast<Wide>(values.at(last)) - values.at(last - 1U)) /
          widths[last - 1U];
      lower[last] = widths[last - 1U];
      diagonal[last] = 2.0L * widths[last - 1U];
      rhs[last] = 6.0L * (*policy.right_derivative - lastSlope);
      for (Index i = 1U; i < last; ++i) {
        lower[i] = widths[i - 1U];
        diagonal[i] = 2.0L * (widths[i - 1U] + widths[i]);
        upper[i] = widths[i];
        const Wide previousSlope =
            (static_cast<Wide>(values.at(i)) - values.at(i - 1U)) /
            widths[i - 1U];
        const Wide nextSlope =
            (static_cast<Wide>(values.at(i + 1U)) - values.at(i)) / widths[i];
        rhs[i] = 6.0L * (nextSlope - previousSlope);
      }
    }

    std::vector<Wide> pivots(n, 0.0L);
    std::vector<Wide> factors(n, 0.0L);
    pivots[0] = diagonal[0];
    if (!(pivots[0] > 0.0L) || !std::isfinite(pivots[0])) {
      return Result<Spline>::failure(ErrorCode::NoConvergence,
                                     "spline LDLT encountered a bad pivot");
    }
    for (Index i = 1U; i < n; ++i) {
      factors[i] = lower[i] / pivots[i - 1U];
      pivots[i] = diagonal[i] - factors[i] * upper[i - 1U];
      if (!(pivots[i] > 0.0L) || !std::isfinite(pivots[i])) {
        return Result<Spline>::failure(ErrorCode::NoConvergence,
                                       "spline LDLT encountered a bad pivot");
      }
    }
    std::vector<Wide> forward(n, 0.0L);
    forward[0] = rhs[0];
    for (Index i = 1U; i < n; ++i) {
      forward[i] = rhs[i] - factors[i] * forward[i - 1U];
    }
    solution[n - 1U] = forward[n - 1U] / pivots[n - 1U];
    for (Index i = n - 1U; i > 0U; --i) {
      solution[i - 1U] = forward[i - 1U] / pivots[i - 1U] -
                         (upper[i - 1U] / pivots[i - 1U]) * solution[i];
    }

    const Wide residualTolerance =
        64.0L * std::numeric_limits<Scalar>::epsilon();
    for (Index i = 0U; i < n; ++i) {
      const Wide lhs = lower[i] * solution[i == 0U ? 0U : i - 1U] +
                       diagonal[i] * solution[i] +
                       upper[i] * solution[i + 1U < n ? i + 1U : i];
      const Wide scale =
          std::abs(lower[i] * solution[i == 0U ? 0U : i - 1U]) +
          std::abs(diagonal[i] * solution[i]) +
          std::abs(upper[i] * solution[i + 1U < n ? i + 1U : i]) +
          std::abs(rhs[i]);
      if (!std::isfinite(lhs) ||
          std::abs(lhs - rhs[i]) > residualTolerance * scale) {
        return Result<Spline>::failure(ErrorCode::NoConvergence,
                                       "spline residual certification failed");
      }
    }

    Spline spline{Vector(n), Matrix(n - 1U, 4U, 0.0)};
    for (Index i = 0U; i < n; ++i) {
      spline.nodes.at(i) = nodes.at(i);
    }
    for (Index i = 0U; i + 1U < n; ++i) {
      const Wide h = widths[i];
      const Wide dy = static_cast<Wide>(values.at(i + 1U)) - values.at(i);
      const Wide h2 = h * h;
      const std::array<Wide, 4U> coefficient{
          static_cast<Wide>(values.at(i)),
          dy - h2 * (2.0L * solution[i] + solution[i + 1U]) / 6.0L,
          h2 * solution[i] / 2.0L,
          h2 * (solution[i + 1U] - solution[i]) / 6.0L};
      for (Index j = 0U; j < 4U; ++j) {
        Scalar stored = 0.0;
        if (!finiteScalar(coefficient.at(j), stored)) {
          return Result<Spline>::failure(
              ErrorCode::NonFiniteResult,
              "spline coefficient is outside finite Scalar range");
        }
        spline.coefficients.at(i, j) = stored;
      }
    }
    return Result<Spline>::success(std::move(spline));
  } catch (const std::bad_alloc&) {
    return Result<Spline>::failure(ErrorCode::ResourceLimit,
                                   "spline workspace allocation failed");
  } catch (const std::length_error&) {
    return Result<Spline>::failure(ErrorCode::ResourceLimit,
                                   "spline workspace extent is unsupported");
  }
}

Result<Scalar> evaluate(const Spline& spline, Scalar x,
                        Extrapolation extrapolation) {
  if (spline.nodes.size() < 2U) {
    return invalidScalar(ErrorCode::InsufficientSamples,
                         "spline requires at least two knots");
  }
  if (spline.coefficients.rows() != spline.nodes.size() - 1U ||
      spline.coefficients.columns() != 4U) {
    return invalidScalar(ErrorCode::DimensionMismatch,
                         "spline coefficient matrix has the wrong shape");
  }
  if (!std::isfinite(x)) {
    return invalidScalar(ErrorCode::NonFiniteInput,
                         "spline query must be finite");
  }
  if (!validExtrapolation(extrapolation)) {
    return invalidScalar(ErrorCode::InvalidDomain,
                         "unknown spline extrapolation policy");
  }
  for (Index i = 0U; i < spline.nodes.size(); ++i) {
    if (!std::isfinite(spline.nodes.at(i))) {
      return invalidScalar(ErrorCode::NonFiniteInput,
                           "spline contains a non-finite knot");
    }
    if (i > 0U && !(spline.nodes.at(i) > spline.nodes.at(i - 1U))) {
      return invalidScalar(ErrorCode::InvalidDomain,
                           "spline knots must be strictly increasing");
    }
  }
  for (Index i = 0U; i < spline.coefficients.rows(); ++i) {
    for (Index j = 0U; j < 4U; ++j) {
      if (!std::isfinite(spline.coefficients.at(i, j))) {
        return invalidScalar(ErrorCode::NonFiniteInput,
                             "spline contains a non-finite coefficient");
      }
    }
  }

  const bool before = x < spline.nodes.at(0U);
  const bool after = x > spline.nodes.at(spline.nodes.size() - 1U);
  if ((before || after) && extrapolation == Extrapolation::reject) {
    return invalidScalar(ErrorCode::InvalidDomain,
                         "spline query lies outside its knot domain");
  }
  Index row = 0U;
  Wide u = 0.0L;
  if (before) {
    row = 0U;
    u = 0.0L;
  } else if (after) {
    row = spline.nodes.size() - 2U;
    u = 1.0L;
  } else {
    Index low = 0U;
    Index high = spline.nodes.size();
    while (low < high) {
      const Index middle = low + (high - low) / 2U;
      if (spline.nodes.at(middle) < x) {
        low = middle + 1U;
      } else {
        high = middle;
      }
    }
    if (low == spline.nodes.size()) {
      row = spline.nodes.size() - 2U;
      u = 1.0L;
    } else if (spline.nodes.at(low) == x) {
      row = low == spline.nodes.size() - 1U ? low - 1U : low;
      u = low == spline.nodes.size() - 1U ? 1.0L : 0.0L;
    } else {
      row = low - 1U;
      u = (static_cast<Wide>(x) - spline.nodes.at(row)) /
          (static_cast<Wide>(spline.nodes.at(row + 1U)) - spline.nodes.at(row));
    }
  }

  const Wide y = polynomial(spline.coefficients, row, u);
  Wide interpolatedValue = y;
  if ((before || after) && extrapolation == Extrapolation::linear) {
    const Wide h =
        static_cast<Wide>(spline.nodes.at(row + 1U)) - spline.nodes.at(row);
    const Wide derivativeU =
        static_cast<Wide>(spline.coefficients.at(row, 1U)) +
        2.0L * spline.coefficients.at(row, 2U) * u +
        3.0L * spline.coefficients.at(row, 3U) * u * u;
    const Wide endpointX = before ? spline.nodes.at(0U)
                                  : spline.nodes.at(spline.nodes.size() - 1U);
    interpolatedValue += (derivativeU / h) * (static_cast<Wide>(x) - endpointX);
  }
  Scalar converted = 0.0;
  if (!finiteScalar(interpolatedValue, converted)) {
    return invalidScalar(ErrorCode::NonFiniteResult,
                         "spline value is outside finite Scalar range");
  }
  return Result<Scalar>::success(converted);
}

} // namespace vss::math
