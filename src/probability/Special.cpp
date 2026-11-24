#include <algorithm>
#include <cmath>
#include <iterator>
#include <limits>
#include <string>
#include <vss/math/Error.hpp>
#include <vss/math/probability/Special.hpp>

namespace vss::math::probability {
namespace {

constexpr Scalar lanczosG = 7.0;
constexpr Scalar lanczosCoefficients[] = {
    0.99999999999980993,  676.5203681218851,     -1259.1392167224028,
    771.32342877765313,   -176.61502916214059,   12.507343278686905,
    -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7};
constexpr Scalar logSqrtTwoPi = 0.91893853320467274178032973640562;
constexpr Scalar logTwo = 0.69314718055994530941723212145818;
constexpr Scalar tiny =
    std::numeric_limits<Scalar>::min() / std::numeric_limits<Scalar>::epsilon();

template <class T> Result<T> failure(ErrorCode code, const char* message) {
  return Result<T>::failure(code, message);
}

Status validatePolicy(const ProbabilityPolicy& policy, bool iterative) {
  const auto& iteration = policy.iteration;
  if (!std::isfinite(iteration.abs) || !std::isfinite(iteration.rel) ||
      !std::isfinite(policy.abs_log_error) || iteration.abs <= 0.0 ||
      iteration.rel <= 0.0 || policy.abs_log_error <= 0.0) {
    return Status::failure(
        ErrorCode::InvalidTolerance,
        "probability tolerances must be finite and positive");
  }
  if (iterative &&
      (iteration.max_iterations == 0U || iteration.max_evaluations == 0U)) {
    return Status::failure(
        ErrorCode::InvalidDomain,
        "iterative probability work limits must be positive");
  }
  return Status::success();
}

Scalar logGammaLanczos(Scalar x) {
  Scalar sum = lanczosCoefficients[0];
  const Scalar shifted = x - 1.0;
  for (Index index = 1U; index < std::size(lanczosCoefficients); ++index) {
    sum += lanczosCoefficients[index] / (shifted + static_cast<Scalar>(index));
  }
  const Scalar t = shifted + lanczosG + 0.5;
  return logSqrtTwoPi + (shifted + 0.5) * std::log(t) - t + std::log(sum);
}

Scalar logOneMinusExp(Scalar value) {
  return value < -logTwo ? std::log1p(-std::exp(value))
                         : std::log(-std::expm1(value));
}

Result<Scalar> gammaSeriesLogLower(Scalar shape, Scalar x,
                                   const ProbabilityPolicy& policy) {
  Scalar term = 1.0 / shape;
  Scalar sum = term;
  Scalar current = shape;
  const Index limit = std::min(policy.iteration.max_iterations,
                               policy.iteration.max_evaluations);
  bool converged = false;
  for (Index iteration = 0U; iteration < limit; ++iteration) {
    current += 1.0;
    term *= x / current;
    sum += term;
    if (!std::isfinite(term) || !std::isfinite(sum)) {
      return failure<Scalar>(ErrorCode::NonFiniteResult,
                             "incomplete-gamma series overflowed");
    }
    const Scalar tolerance =
        policy.iteration.abs + policy.iteration.rel * std::abs(sum);
    if (std::abs(term) <= tolerance) {
      converged = true;
      break;
    }
  }
  if (!converged) {
    return failure<Scalar>(ErrorCode::NoConvergence,
                           "incomplete-gamma series reached its work limit");
  }
  const Scalar value =
      shape * std::log(x) - x - logGammaLanczos(shape) + std::log(sum);
  if (!std::isfinite(value) || value > 0.0) {
    return failure<Scalar>(ErrorCode::NonFiniteResult,
                           "incomplete-gamma lower tail is not representable");
  }
  return Result<Scalar>::success(value);
}

Result<Scalar> gammaFractionLogUpper(Scalar shape, Scalar x,
                                     const ProbabilityPolicy& policy) {
  Scalar b = x + 1.0 - shape;
  Scalar c = 1.0 / tiny;
  Scalar d = 1.0 / std::max(std::abs(b), tiny);
  if (b < 0.0) {
    d = -d;
  }
  Scalar h = d;
  const Index limit = std::min(policy.iteration.max_iterations,
                               policy.iteration.max_evaluations);
  bool converged = false;
  for (Index index = 1U; index <= limit; ++index) {
    const Scalar i = static_cast<Scalar>(index);
    const Scalar an = -i * (i - shape);
    b += 2.0;
    d = an * d + b;
    if (std::abs(d) < tiny) {
      d = tiny;
    }
    c = b + an / c;
    if (std::abs(c) < tiny) {
      c = tiny;
    }
    d = 1.0 / d;
    const Scalar delta = d * c;
    h *= delta;
    if (!std::isfinite(h) || !std::isfinite(delta)) {
      return failure<Scalar>(ErrorCode::NonFiniteResult,
                             "incomplete-gamma fraction overflowed");
    }
    if (std::abs(delta - 1.0) <= policy.iteration.abs + policy.iteration.rel) {
      converged = true;
      break;
    }
  }
  if (!converged || h <= 0.0) {
    return failure<Scalar>(ErrorCode::NoConvergence,
                           "incomplete-gamma fraction reached its work limit");
  }
  const Scalar value =
      shape * std::log(x) - x - logGammaLanczos(shape) + std::log(h);
  if (!std::isfinite(value) || value > 0.0) {
    return failure<Scalar>(ErrorCode::NonFiniteResult,
                           "incomplete-gamma upper tail is not representable");
  }
  return Result<Scalar>::success(value);
}

Scalar betaFraction(Scalar a, Scalar b, Scalar x, Index maximumIterations,
                    Scalar tolerance, bool& converged) {
  const Scalar qab = a + b;
  const Scalar qap = a + 1.0;
  const Scalar qam = a - 1.0;
  Scalar c = 1.0;
  Scalar d = 1.0 - qab * x / qap;
  if (std::abs(d) < tiny) {
    d = tiny;
  }
  d = 1.0 / d;
  Scalar h = d;
  converged = false;
  for (Index index = 1U; index <= maximumIterations; ++index) {
    const Scalar m = static_cast<Scalar>(index);
    const Scalar m2 = 2.0 * m;
    Scalar aa = m * (b - m) * x / ((qam + m2) * (a + m2));
    d = 1.0 + aa * d;
    if (std::abs(d) < tiny) {
      d = tiny;
    }
    c = 1.0 + aa / c;
    if (std::abs(c) < tiny) {
      c = tiny;
    }
    d = 1.0 / d;
    h *= d * c;
    aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
    d = 1.0 + aa * d;
    if (std::abs(d) < tiny) {
      d = tiny;
    }
    c = 1.0 + aa / c;
    if (std::abs(c) < tiny) {
      c = tiny;
    }
    d = 1.0 / d;
    const Scalar delta = d * c;
    h *= delta;
    if (!std::isfinite(h) || !std::isfinite(delta)) {
      return std::numeric_limits<Scalar>::quiet_NaN();
    }
    if (std::abs(delta - 1.0) <= tolerance) {
      converged = true;
      break;
    }
  }
  return h;
}

Result<Scalar> betaLogLower(Scalar a, Scalar b, Scalar x,
                            const ProbabilityPolicy& policy) {
  bool converged = false;
  const Scalar h =
      betaFraction(a, b, x,
                   std::min(policy.iteration.max_iterations,
                            policy.iteration.max_evaluations),
                   policy.iteration.abs + policy.iteration.rel, converged);
  if (!std::isfinite(h) || h <= 0.0) {
    return failure<Scalar>(ErrorCode::NonFiniteResult,
                           "incomplete-beta fraction is not finite");
  }
  if (!converged) {
    return failure<Scalar>(ErrorCode::NoConvergence,
                           "incomplete-beta fraction reached its work limit");
  }
  const Scalar logBeta =
      logGammaLanczos(a) + logGammaLanczos(b) - logGammaLanczos(a + b);
  const Scalar value =
      a * std::log(x) + b * std::log1p(-x) - logBeta + std::log(h / a);
  if (!std::isfinite(value) || value > 0.0) {
    return failure<Scalar>(ErrorCode::NonFiniteResult,
                           "incomplete-beta lower tail is not representable");
  }
  return Result<Scalar>::success(value);
}

} // namespace

Result<Scalar> logGamma(Scalar x, ProbabilityPolicy policy) {
  const auto validPolicy = validatePolicy(policy, false);
  if (!validPolicy.ok()) {
    return Result<Scalar>::failure(validPolicy.code(),
                                   std::string(validPolicy.message()));
  }
  if (!std::isfinite(x)) {
    return failure<Scalar>(ErrorCode::NonFiniteInput,
                           "log-gamma argument must be finite");
  }
  if (x <= 0.0) {
    return failure<Scalar>(ErrorCode::InvalidDomain,
                           "log-gamma argument must be positive");
  }
  const Scalar result = logGammaLanczos(x);
  if (!std::isfinite(result)) {
    return failure<Scalar>(ErrorCode::NonFiniteResult,
                           "log-gamma result is not finite");
  }
  return Result<Scalar>::success(result);
}

Result<LogTailPair> incompleteGamma(Scalar shape, Scalar x,
                                    ProbabilityPolicy policy) {
  const auto validPolicy = validatePolicy(policy, true);
  if (!validPolicy.ok()) {
    return Result<LogTailPair>::failure(validPolicy.code(),
                                        std::string(validPolicy.message()));
  }
  if (!std::isfinite(shape) || std::isnan(x) ||
      x == -std::numeric_limits<Scalar>::infinity()) {
    return failure<LogTailPair>(ErrorCode::NonFiniteInput,
                                "incomplete-gamma inputs are non-finite");
  }
  if (shape <= 0.0 || x < 0.0) {
    return failure<LogTailPair>(ErrorCode::InvalidDomain,
                                "incomplete-gamma requires shape>0 and x>=0");
  }
  const Scalar negativeInfinity = -std::numeric_limits<Scalar>::infinity();
  if (x == 0.0) {
    return Result<LogTailPair>::success({negativeInfinity, 0.0});
  }
  if (x == std::numeric_limits<Scalar>::infinity()) {
    return Result<LogTailPair>::success({0.0, negativeInfinity});
  }
  if (x < shape + 1.0) {
    auto lower = gammaSeriesLogLower(shape, x, policy);
    if (!lower) {
      return Result<LogTailPair>::failure(
          lower.status().code(), std::string(lower.status().message()));
    }
    return Result<LogTailPair>::success(
        {lower.value(), logOneMinusExp(lower.value())});
  }
  auto upper = gammaFractionLogUpper(shape, x, policy);
  if (!upper) {
    return Result<LogTailPair>::failure(upper.status().code(),
                                        std::string(upper.status().message()));
  }
  return Result<LogTailPair>::success(
      {logOneMinusExp(upper.value()), upper.value()});
}

Result<LogTailPair> incompleteBeta(Scalar a, Scalar b, Scalar x,
                                   ProbabilityPolicy policy) {
  const auto validPolicy = validatePolicy(policy, true);
  if (!validPolicy.ok()) {
    return Result<LogTailPair>::failure(validPolicy.code(),
                                        std::string(validPolicy.message()));
  }
  if (!std::isfinite(a) || !std::isfinite(b) || !std::isfinite(x)) {
    return failure<LogTailPair>(ErrorCode::NonFiniteInput,
                                "incomplete-beta inputs must be finite");
  }
  if (a <= 0.0 || b <= 0.0 || x < 0.0 || x > 1.0) {
    return failure<LogTailPair>(
        ErrorCode::InvalidDomain,
        "incomplete-beta requires a,b>0 and x in [0,1]");
  }
  const Scalar negativeInfinity = -std::numeric_limits<Scalar>::infinity();
  if (x == 0.0) {
    return Result<LogTailPair>::success({negativeInfinity, 0.0});
  }
  if (x == 1.0) {
    return Result<LogTailPair>::success({0.0, negativeInfinity});
  }
  if (!std::isfinite(a + b)) {
    return failure<LogTailPair>(
        ErrorCode::NonFiniteResult,
        "incomplete-beta shape sum is not representable");
  }
  const Scalar threshold = (a + 1.0) / (a + b + 2.0);
  if (x < threshold) {
    auto lower = betaLogLower(a, b, x, policy);
    if (!lower) {
      return Result<LogTailPair>::failure(
          lower.status().code(), std::string(lower.status().message()));
    }
    return Result<LogTailPair>::success(
        {lower.value(), logOneMinusExp(lower.value())});
  }
  auto reflected = betaLogLower(b, a, 1.0 - x, policy);
  if (!reflected) {
    return Result<LogTailPair>::failure(
        reflected.status().code(), std::string(reflected.status().message()));
  }
  return Result<LogTailPair>::success(
      {logOneMinusExp(reflected.value()), reflected.value()});
}

Result<Scalar> logSumExp(ConstVectorView terms) {
  Scalar maximum = -std::numeric_limits<Scalar>::infinity();
  for (Index index = 0U; index < terms.size(); ++index) {
    const Scalar value = terms.at(index);
    if (std::isnan(value) || value == std::numeric_limits<Scalar>::infinity()) {
      return failure<Scalar>(ErrorCode::NonFiniteInput,
                             "log-sum-exp accepts finite values or -infinity");
    }
    maximum = std::max(maximum, value);
  }
  if (maximum == -std::numeric_limits<Scalar>::infinity()) {
    return Result<Scalar>::success(maximum);
  }
  Scalar sum = 0.0;
  Scalar correction = 0.0;
  for (Index index = 0U; index < terms.size(); ++index) {
    const Scalar value = terms.at(index);
    if (value == -std::numeric_limits<Scalar>::infinity()) {
      continue;
    }
    const Scalar adjusted = std::exp(value - maximum) - correction;
    const Scalar next = sum + adjusted;
    correction = (next - sum) - adjusted;
    sum = next;
  }
  const Scalar result = maximum + std::log(sum);
  if (!std::isfinite(result)) {
    return failure<Scalar>(ErrorCode::NonFiniteResult,
                           "log-sum-exp result is not finite");
  }
  return Result<Scalar>::success(result);
}

} // namespace vss::math::probability
