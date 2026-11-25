#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>
#include <vss/math/Error.hpp>
#include <vss/math/Roots.hpp>
#include <vss/math/linear/FactorizationOptions.hpp>
#include <vss/math/linear/QR.hpp>

namespace vss::math {
namespace {

using Wide = long double;

bool opposite(Scalar left, Scalar right) noexcept {
  return std::signbit(left) != std::signbit(right);
}

bool validTolerance(const IterPolicy& policy) noexcept {
  return std::isfinite(policy.abs) && std::isfinite(policy.rel) &&
         policy.abs >= 0.0 && policy.rel >= 0.0;
}

Result<Scalar> sample(const UnaryFunction& function, Scalar x,
                      Index& evaluations, Index maximum) {
  if (evaluations >= maximum) {
    return Result<Scalar>::failure(ErrorCode::ResourceLimit,
                                   "root callback evaluation limit reached");
  }
  ++evaluations;
  try {
    const Result<Scalar> result = function(x);
    if (!result) {
      return Result<Scalar>::failure(result.status().code(),
                                     std::string(result.status().message()));
    }
    if (!std::isfinite(result.value())) {
      return Result<Scalar>::failure(ErrorCode::NonFiniteInput,
                                     "root callback returned a non-finite value");
    }
    return result;
  } catch (...) {
    return Result<Scalar>::failure(ErrorCode::CallbackFailure,
                                   "root callback threw an exception");
  }
}

Result<RootResult> scalarFailure(const Status& status) {
  return Result<RootResult>::failure(status.code(),
                                     std::string(status.message()));
}

RootResult rootResult(Scalar x, Scalar fx, std::optional<Interval> bracket,
                      Index functionEvaluations,
                      Index derivativeEvaluations, Index iterations,
                      Termination termination) {
  return {x, fx, bracket, functionEvaluations, derivativeEvaluations,
          iterations, termination};
}

Wide widthTolerance(Scalar lower, Scalar upper,
                    const IterPolicy& policy) noexcept {
  const Wide a = static_cast<Wide>(lower);
  const Wide b = static_cast<Wide>(upper);
  return static_cast<Wide>(policy.abs) +
         static_cast<Wide>(policy.rel) * std::max(std::abs(a), std::abs(b));
}

bool widthConverged(Scalar lower, Scalar upper,
                    const IterPolicy& policy) noexcept {
  return static_cast<Wide>(upper) - static_cast<Wide>(lower) <=
         widthTolerance(lower, upper, policy);
}

Scalar midpoint(Scalar lower, Scalar upper) noexcept {
  return static_cast<Scalar>((static_cast<Wide>(lower) +
                              static_cast<Wide>(upper)) /
                             2.0L);
}

Scalar bestEndpoint(Scalar lower, Scalar fLower, Scalar upper,
                    Scalar fUpper) noexcept {
  if (std::abs(fLower) < std::abs(fUpper) ||
      (std::abs(fLower) == std::abs(fUpper) && lower < upper)) {
    return lower;
  }
  return upper;
}

Result<RootResult> bracketedRoot(const UnaryFunction& function,
                                Interval bracket, const IterPolicy& policy,
                                bool useBrent) {
  if (!validTolerance(policy)) {
    return Result<RootResult>::failure(ErrorCode::InvalidTolerance,
                                       "root tolerances must be finite and non-negative");
  }
  if (!std::isfinite(bracket.lower) || !std::isfinite(bracket.upper)) {
    return Result<RootResult>::failure(ErrorCode::NonFiniteInput,
                                       "root bracket endpoints must be finite");
  }
  if (!(bracket.lower < bracket.upper) || policy.bracket.has_value()) {
    return Result<RootResult>::failure(ErrorCode::InvalidDomain,
                                       "root bracket or policy is invalid");
  }
  if (!function) {
    return Result<RootResult>::failure(ErrorCode::CallbackFailure,
                                       "root function is empty");
  }

  Index evaluations = 0U;
  auto lowerResult = sample(function, bracket.lower, evaluations,
                            policy.max_evaluations);
  if (!lowerResult) {
    return scalarFailure(lowerResult.status());
  }
  Scalar lower = bracket.lower;
  Scalar fLower = lowerResult.value();
  if (fLower == 0.0) {
    return Result<RootResult>::success(rootResult(
        lower, fLower, Interval{lower, lower}, evaluations, 0U, 0U,
        Termination::converged));
  }

  auto upperResult = sample(function, bracket.upper, evaluations,
                            policy.max_evaluations);
  if (!upperResult) {
    return scalarFailure(upperResult.status());
  }
  Scalar upper = bracket.upper;
  Scalar fUpper = upperResult.value();
  if (fUpper == 0.0) {
    return Result<RootResult>::success(rootResult(
        upper, fUpper, Interval{upper, upper}, evaluations, 0U, 0U,
        Termination::converged));
  }
  if (!opposite(fLower, fUpper)) {
    return Result<RootResult>::failure(ErrorCode::InvalidDomain,
                                       "root bracket does not change sign");
  }
  Scalar a = lower;
  Scalar fa = fLower;
  Scalar b = upper;
  Scalar fb = fUpper;
  Scalar c = a;
  Scalar fc = fa;
  Scalar d = c;
  bool previousWasBisection = true;
  Index iterations = 0U;

  while (true) {
    const Scalar candidate = bestEndpoint(lower, fLower, upper, fUpper);
    const Scalar residual = candidate == lower ? fLower : fUpper;
    if (widthConverged(lower, upper, policy)) {
      return Result<RootResult>::success(rootResult(
          candidate, residual, Interval{lower, upper}, evaluations, 0U,
          iterations, Termination::converged));
    }
    if (iterations >= policy.max_iterations) {
      return Result<RootResult>::success(rootResult(
          candidate, residual, Interval{lower, upper}, evaluations, 0U,
          iterations, Termination::cap));
    }
    if (std::nextafter(lower, upper) == upper) {
      return Result<RootResult>::failure(
          ErrorCode::NoConvergence,
          "root bracket has no representable interior point");
    }

    Scalar trial = midpoint(lower, upper);
    bool bisection = true;
    if (useBrent) {
      Wide proposal = std::numeric_limits<Wide>::quiet_NaN();
      const bool distinct = fa != fb && fa != fc && fb != fc;
      if (distinct) {
        const Wide wa = static_cast<Wide>(a);
        const Wide wb = static_cast<Wide>(b);
        const Wide wc = static_cast<Wide>(c);
        const Wide wfa = static_cast<Wide>(fa);
        const Wide wfb = static_cast<Wide>(fb);
        const Wide wfc = static_cast<Wide>(fc);
        proposal = wa * wfb * wfc / ((wfa - wfb) * (wfa - wfc)) +
                   wb * wfa * wfc / ((wfb - wfa) * (wfb - wfc)) +
                   wc * wfa * wfb / ((wfc - wfa) * (wfc - wfb));
      } else if (fb != fa) {
        proposal = static_cast<Wide>(b) - static_cast<Wide>(fb) *
                                               (static_cast<Wide>(b) - a) /
                                               (static_cast<Wide>(fb) - fa);
      }
      const Wide midpointGuard =
          (3.0L * static_cast<Wide>(a) + static_cast<Wide>(b)) / 4.0L;
      const Wide lowGuard = std::min(static_cast<Wide>(b), midpointGuard);
      const Wide highGuard = std::max(static_cast<Wide>(b), midpointGuard);
      const Wide previousDistance = previousWasBisection
                                        ? std::abs(static_cast<Wide>(b) - c)
                                        : std::abs(static_cast<Wide>(c) - d);
      const Wide distance = std::abs(proposal - static_cast<Wide>(b));
      if (std::isfinite(proposal) && proposal > lowGuard &&
          proposal < highGuard && previousDistance >
              widthTolerance(lower, upper, policy) &&
          distance < previousDistance / 2.0L &&
          proposal >= static_cast<Wide>(lower) &&
          proposal <= static_cast<Wide>(upper)) {
        const Scalar converted = static_cast<Scalar>(proposal);
        if (converted > lower && converted < upper) {
          trial = converted;
          bisection = false;
        }
      }
    }

    auto trialResult = sample(function, trial, evaluations,
                              policy.max_evaluations);
    if (!trialResult) {
      return scalarFailure(trialResult.status());
    }
    const Scalar fTrial = trialResult.value();
    ++iterations;
    if (fTrial == 0.0) {
      return Result<RootResult>::success(rootResult(
          trial, fTrial, Interval{trial, trial}, evaluations, 0U, iterations,
          Termination::converged));
    }
    d = c;
    c = b;
    fc = fb;
    if (opposite(fLower, fTrial)) {
      upper = trial;
      fUpper = fTrial;
    } else {
      lower = trial;
      fLower = fTrial;
    }
    a = lower;
    fa = fLower;
    b = upper;
    fb = fUpper;
    if (std::abs(fa) < std::abs(fb)) {
      std::swap(a, b);
      std::swap(fa, fb);
    }
    previousWasBisection = bisection;
  }
}

} // namespace

Result<RootResult> rootBisection(const UnaryFunction& function,
                                 Interval bracket,
                                 const IterPolicy& policy) {
  return bracketedRoot(function, bracket, policy, false);
}

Result<RootResult> rootBrent(const UnaryFunction& function, Interval bracket,
                             const IterPolicy& policy) {
  return bracketedRoot(function, bracket, policy, true);
}

Result<RootResult> rootNewton(const UnaryFunction& function,
                              const UnaryFunction& derivative, Scalar initial,
                              const IterPolicy& policy) {
  if (!validTolerance(policy)) {
    return Result<RootResult>::failure(ErrorCode::InvalidTolerance,
                                       "root tolerances must be finite and non-negative");
  }
  if (!std::isfinite(initial)) {
    return Result<RootResult>::failure(ErrorCode::NonFiniteInput,
                                       "Newton initial value must be finite");
  }
  if (!function || !derivative) {
    return Result<RootResult>::failure(ErrorCode::CallbackFailure,
                                       "Newton callback is empty");
  }
  Scalar lower = 0.0;
  Scalar upper = 0.0;
  Scalar fLower = 0.0;
  Scalar fUpper = 0.0;
  bool bracketed = policy.bracket.has_value();
  Index evaluations = 0U;
  Index derivativeEvaluations = 0U;
  Index iterations = 0U;
  if (bracketed) {
    const Interval bracket = *policy.bracket;
    if (!std::isfinite(bracket.lower) || !std::isfinite(bracket.upper)) {
      return Result<RootResult>::failure(ErrorCode::NonFiniteInput,
                                         "Newton bracket must be finite");
    }
    if (!(bracket.lower < bracket.upper) || initial < bracket.lower ||
        initial > bracket.upper) {
      return Result<RootResult>::failure(ErrorCode::InvalidDomain,
                                         "Newton bracket or initial point is invalid");
    }
    lower = bracket.lower;
    upper = bracket.upper;
    auto lo = sample(function, lower, evaluations, policy.max_evaluations);
    if (!lo) {
      return scalarFailure(lo.status());
    }
    fLower = lo.value();
    if (fLower == 0.0) {
      return Result<RootResult>::success(rootResult(
          lower, fLower, Interval{lower, lower}, evaluations, 0U, 0U,
          Termination::converged));
    }
    auto hi = sample(function, upper, evaluations, policy.max_evaluations);
    if (!hi) {
      return scalarFailure(hi.status());
    }
    fUpper = hi.value();
    if (fUpper == 0.0) {
      return Result<RootResult>::success(rootResult(
          upper, fUpper, Interval{upper, upper}, evaluations, 0U, 0U,
          Termination::converged));
    }
    if (!opposite(fLower, fUpper)) {
      return Result<RootResult>::failure(ErrorCode::InvalidDomain,
                                         "Newton bracket does not change sign");
    }
  }
  Scalar x = initial;
  Scalar fx = 0.0;
  if (bracketed && x == lower) {
    fx = fLower;
  } else if (bracketed && x == upper) {
    fx = fUpper;
  } else {
    auto initialResult = sample(function, x, evaluations,
                                policy.max_evaluations);
    if (!initialResult) {
      return scalarFailure(initialResult.status());
    }
    fx = initialResult.value();
  }
  const Scalar initialResidual = fx;
  auto residualConverged = [&](Scalar value) {
    return std::abs(static_cast<Wide>(value)) <=
           static_cast<Wide>(policy.abs) +
               static_cast<Wide>(policy.rel) *
                   std::abs(static_cast<Wide>(initialResidual));
  };
  if (fx == 0.0 || (!bracketed && residualConverged(fx))) {
    return Result<RootResult>::success(rootResult(
        x, fx, bracketed ? std::optional<Interval>{Interval{lower, upper}}
                         : std::nullopt,
        evaluations, derivativeEvaluations, iterations, Termination::converged));
  }

  while (iterations < policy.max_iterations) {
    if (evaluations >= policy.max_evaluations - derivativeEvaluations) {
      return Result<RootResult>::failure(ErrorCode::ResourceLimit,
                                         "Newton derivative evaluation limit reached");
    }
    ++derivativeEvaluations;
    Result<Scalar> derivativeResult = Result<Scalar>::failure(
        ErrorCode::CallbackFailure, "Newton derivative callback threw");
    try {
      derivativeResult = derivative(x);
    } catch (...) {
      return scalarFailure(derivativeResult.status());
    }
    if (!derivativeResult) {
      return scalarFailure(derivativeResult.status());
    }
    const Scalar slope = derivativeResult.value();
    if (!std::isfinite(slope)) {
      return Result<RootResult>::failure(ErrorCode::NonFiniteInput,
                                         "Newton derivative is non-finite");
    }
    Scalar trial = x;
    bool midpointStep = false;
    if (slope == 0.0) {
      if (!bracketed) {
        return Result<RootResult>::failure(ErrorCode::RankDeficient,
                                           "Newton derivative is zero");
      }
      midpointStep = true;
    } else {
      const Wide proposed = static_cast<Wide>(x) -
                            static_cast<Wide>(fx) / slope;
      if (!std::isfinite(proposed) ||
          std::abs(proposed) > std::numeric_limits<Scalar>::max()) {
        return Result<RootResult>::failure(ErrorCode::NonFiniteResult,
                                           "Newton trial is not representable");
      }
      trial = static_cast<Scalar>(proposed);
      if (bracketed) {
        const Wide quarter = (static_cast<Wide>(upper) - lower) / 4.0L;
        const Wide centralLower = static_cast<Wide>(lower) + quarter;
        const Wide centralUpper = static_cast<Wide>(upper) - quarter;
        if (trial == x || static_cast<Wide>(trial) < centralLower ||
            static_cast<Wide>(trial) > centralUpper) {
          midpointStep = true;
        }
      }
    }
    if (midpointStep) {
      trial = midpoint(lower, upper);
    }
    if (trial == x || (bracketed && !(trial > lower && trial < upper))) {
      if (bracketed && widthConverged(lower, upper, policy)) {
        const Scalar best = bestEndpoint(lower, fLower, upper, fUpper);
        return Result<RootResult>::success(rootResult(
            best, best == lower ? fLower : fUpper, Interval{lower, upper},
            evaluations, derivativeEvaluations, iterations,
            Termination::converged));
      }
      return Result<RootResult>::failure(ErrorCode::NoConvergence,
                                         "Newton has no distinct admissible trial");
    }
    auto trialResult = sample(function, trial, evaluations,
                              policy.max_evaluations);
    if (!trialResult) {
      return scalarFailure(trialResult.status());
    }
    const Scalar trialValue = trialResult.value();
    ++iterations;
    if (bracketed) {
      if (trialValue == 0.0) {
        return Result<RootResult>::success(rootResult(
            trial, trialValue, Interval{trial, trial}, evaluations,
            derivativeEvaluations, iterations, Termination::converged));
      }
      if (opposite(fLower, trialValue)) {
        upper = trial;
        fUpper = trialValue;
      } else {
        lower = trial;
        fLower = trialValue;
      }
    }
    x = trial;
    fx = trialValue;
    if (bracketed && widthConverged(lower, upper, policy)) {
      const Scalar best = bestEndpoint(lower, fLower, upper, fUpper);
      return Result<RootResult>::success(rootResult(
          best, best == lower ? fLower : fUpper, Interval{lower, upper},
          evaluations, derivativeEvaluations, iterations,
          Termination::converged));
    }
    if (!bracketed && (fx == 0.0 || residualConverged(fx))) {
      return Result<RootResult>::success(rootResult(
          x, fx, bracketed ? std::optional<Interval>{Interval{lower, upper}}
                           : std::nullopt,
          evaluations, derivativeEvaluations, iterations,
          Termination::converged));
    }
  }
  return Result<RootResult>::success(rootResult(
      x, fx, bracketed ? std::optional<Interval>{Interval{lower, upper}}
                       : std::nullopt,
      evaluations, derivativeEvaluations, iterations, Termination::cap));
}

} // namespace vss::math
