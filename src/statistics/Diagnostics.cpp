#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>
#include <vss/math/Core.hpp>
#include <vss/math/Error.hpp>
#include <vss/math/Limits.hpp>
#include <vss/math/ProbabilityPolicy.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/SampleView.hpp>
#include <vss/math/Vector.hpp>
#include <vss/math/VectorView.hpp>
#include <vss/math/probability/Distribution.hpp>
#include <vss/math/statistics/Diagnostics.hpp>
#include <vss/math/statistics/Inference.hpp>

namespace vss::math::statistics {
namespace {

using probability::Distribution;

template <class T> Result<T> propagate(const Status& status) {
  return Result<T>::failure(status.code(), std::string(status.message()));
}

ProbabilityPolicy goodnessPolicy(Index draws) {
  return ProbabilityPolicy{.iteration = {.abs = 1.0e-13,
                                         .rel = 1.0e-11,
                                         .max_iterations = 4096U,
                                         .max_evaluations = 4096U,
                                         .bracket = std::nullopt},
                           .max_draws = static_cast<std::uint64_t>(draws),
                           .abs_log_error = 1.0e-11};
}

Result<Vector> scalarValues(const SampleView& sample, Limits limits) {
  const Index n = sample.rows.rows();
  if (sample.rows.columns() != 1U ||
      (!sample.observation_shape.empty() &&
       (sample.observation_shape.size() != 1U ||
        sample.observation_shape.front() != 1U))) {
    return Result<Vector>::failure(
        ErrorCode::DimensionMismatch,
        "goodness-of-fit tests require scalar observations");
  }
  if (n == 0U) {
    return Result<Vector>::failure(ErrorCode::InsufficientSamples,
                                   "goodness-of-fit needs at least one row");
  }
  if (!sample.valid.empty()) {
    return Result<Vector>::failure(ErrorCode::InvalidDomain,
                                   "goodness-of-fit requires complete rows");
  }
  if (n > limits.elements || n > limits.bytes / sizeof(Scalar)) {
    return Result<Vector>::failure(ErrorCode::ResourceLimit,
                                   "goodness-of-fit sample exceeds limits");
  }
  Vector values(n);
  for (Index i = 0U; i < n; ++i) {
    const Scalar value = sample.rows.at(i, 0U);
    if (!std::isfinite(value)) {
      return Result<Vector>::failure(ErrorCode::NonFiniteInput,
                                     "goodness-of-fit sample is non-finite");
    }
    values.at(i) = value;
  }
  return Result<Vector>::success(std::move(values));
}

bool discrete(const Distribution& law) {
  return law.measure() != probability::Measure::real_density;
}

Result<Scalar> ksStatistic(ConstVectorView values, const Distribution& law,
                           Limits /*limits*/) {
  std::vector<Scalar> sorted(values.size());
  for (Index i = 0U; i < values.size(); ++i) {
    sorted[i] = values.at(i);
  }
  std::ranges::stable_sort(sorted);
  if (!discrete(law)) {
    Scalar statistic = 0.0;
    Index first = 0U;
    while (first < sorted.size()) {
      Index last = first + 1U;
      while (last < sorted.size() && sorted[last] == sorted[first]) {
        ++last;
      }
      const auto tails =
          probability::evaluate(law, sorted[first], goodnessPolicy(1U));
      if (!tails.hasValue()) {
        return propagate<Scalar>(tails.status());
      }
      if (!tails.value().on_support) {
        return Result<Scalar>::success(std::numeric_limits<Scalar>::infinity());
      }
      const auto sampleCount = static_cast<Scalar>(sorted.size());
      const auto left = static_cast<Scalar>(first) / sampleCount;
      const auto right = static_cast<Scalar>(last) / sampleCount;
      statistic = std::max(
          {statistic, tails.value().cdf - left, right - tails.value().cdf});
      first = last;
    }
    return Result<Scalar>::success(statistic);
  }

  Scalar statistic = 0.0;
  const auto sampleCount = static_cast<Scalar>(sorted.size());
  Index first = 0U;
  while (first < sorted.size()) {
    Index last = first + 1U;
    while (last < sorted.size() && sorted[last] == sorted[first]) {
      ++last;
    }
    const auto tails =
        probability::evaluate(law, sorted[first], goodnessPolicy(1U));
    if (!tails.hasValue()) {
      return propagate<Scalar>(tails.status());
    }
    if (!tails.value().on_support) {
      return Result<Scalar>::success(std::numeric_limits<Scalar>::infinity());
    }
    const Scalar empiricalLeft = static_cast<Scalar>(first) / sampleCount;
    const Scalar empiricalRight = static_cast<Scalar>(last) / sampleCount;
    const Scalar modelLeft = tails.value().cdf - tails.value().density_or_mass;
    statistic = std::max({statistic, std::abs(empiricalLeft - modelLeft),
                          std::abs(empiricalRight - tails.value().cdf)});
    first = last;
  }
  return Result<Scalar>::success(statistic);
}

Result<Scalar> andersonDarlingStatistic(ConstVectorView values,
                                        const Distribution& law,
                                        Limits limits) {
  std::vector<Scalar> sorted(values.size());
  for (Index i = 0U; i < values.size(); ++i) {
    sorted[i] = values.at(i);
  }
  std::ranges::stable_sort(sorted);
  const Index n = sorted.size();
  if (!discrete(law)) {
    long double sum = 0.0L;
    for (Index i = 0U; i < n; ++i) {
      const auto lower =
          probability::evaluate(law, sorted[i], goodnessPolicy(1U));
      const auto upper =
          probability::evaluate(law, sorted[n - 1U - i], goodnessPolicy(1U));
      if (!lower.hasValue()) {
        return propagate<Scalar>(lower.status());
      }
      if (!upper.hasValue()) {
        return propagate<Scalar>(upper.status());
      }
      if (!lower.value().on_support || !upper.value().on_support ||
          lower.value().log_tails.log_cdf ==
              -std::numeric_limits<Scalar>::infinity() ||
          upper.value().log_tails.log_survival ==
              -std::numeric_limits<Scalar>::infinity()) {
        return Result<Scalar>::success(std::numeric_limits<Scalar>::infinity());
      }
      const long double weight = ((2.0L * static_cast<long double>(i)) + 1.0L) /
                                 static_cast<long double>(n);
      sum += weight * (lower.value().log_tails.log_cdf +
                       upper.value().log_tails.log_survival);
    }
    const long double statistic = -static_cast<long double>(n) - sum;
    if (!std::isfinite(statistic) ||
        statistic > std::numeric_limits<Scalar>::max()) {
      return Result<Scalar>::failure(
          ErrorCode::NonFiniteResult,
          "Anderson-Darling result is unrepresentable");
    }
    return Result<Scalar>::success(static_cast<Scalar>(statistic));
  }

  for (const Scalar value : sorted) {
    const auto atValue = probability::evaluate(law, value, goodnessPolicy(1U));
    if (!atValue.hasValue()) {
      return propagate<Scalar>(atValue.status());
    }
    if (!atValue.value().on_support) {
      return Result<Scalar>::success(std::numeric_limits<Scalar>::infinity());
    }
  }

  const auto atoms = law.finiteAtoms();
  if (!atoms.locations.empty()) {
    if (atoms.locations.size() != atoms.masses.size()) {
      return Result<Scalar>::failure(
          ErrorCode::InvalidDomain,
          "finite atom locations and masses have inconsistent extents");
    }
    long double statistic = 0.0L;
    const auto sampleCount = static_cast<long double>(n);
    for (Index i = 0U; i < atoms.locations.size(); ++i) {
      const auto tails =
          probability::evaluate(law, atoms.locations[i], goodnessPolicy(1U));
      if (!tails.hasValue()) {
        return propagate<Scalar>(tails.status());
      }
      const Scalar f = tails.value().cdf;
      if (f <= 0.0 || f >= 1.0 || atoms.masses[i] == 0.0) {
        continue;
      }
      // NOLINTNEXTLINE(misc-include-cleaner): Clang 18 header mapping bug.
      const auto count = std::ranges::upper_bound(sorted, atoms.locations[i]);
      const auto empirical =
          static_cast<long double>(count - sorted.begin()) / sampleCount;
      const long double difference = empirical - static_cast<long double>(f);
      statistic += sampleCount * atoms.masses[i] * difference * difference /
                   (static_cast<long double>(f) * (1.0L - f));
    }
    if (!std::isfinite(statistic) ||
        statistic > std::numeric_limits<Scalar>::max()) {
      return Result<Scalar>::failure(
          ErrorCode::NonFiniteResult,
          "finite-atom AD statistic is unrepresentable");
    }
    return Result<Scalar>::success(static_cast<Scalar>(statistic));
  }
  if (law.measure() == probability::Measure::atom_mass) {
    return Result<Scalar>::failure(
        ErrorCode::InvalidDomain,
        "discrete AD requires enumerable finite atoms or integer support");
  }

  const Scalar maximum = sorted.back();
  if (maximum < 0.0 || std::floor(maximum) != maximum) {
    return Result<Scalar>::success(std::numeric_limits<Scalar>::infinity());
  }
  if (maximum > static_cast<Scalar>(limits.evaluations)) {
    return Result<Scalar>::failure(
        ErrorCode::ResourceLimit, "discrete AD support exceeds evaluation cap");
  }
  const auto observedMaximum = static_cast<Index>(maximum);
  Index end = observedMaximum;
  const auto parameters = law.parameters();
  if (law.family() == probability::Family::bernoulli) {
    end = 1U;
  } else if (law.family() == probability::Family::binomial &&
             !parameters.empty() &&
             parameters.front() <= static_cast<Scalar>(limits.evaluations)) {
    end = static_cast<Index>(parameters.front());
  }
  if (end >= limits.evaluations) {
    return Result<Scalar>::failure(
        ErrorCode::ResourceLimit, "discrete AD support exceeds evaluation cap");
  }
  long double statistic = 0.0L;
  bool converged = false;
  for (Index k = 0U; k <= limits.evaluations; ++k) {
    if ((law.family() == probability::Family::bernoulli ||
         law.family() == probability::Family::binomial) &&
        k > end) {
      converged = true;
      break;
    }
    const auto tails =
        probability::evaluate(law, static_cast<Scalar>(k), goodnessPolicy(1U));
    if (!tails.hasValue()) {
      return propagate<Scalar>(tails.status());
    }
    const Scalar f = tails.value().cdf;
    if (tails.value().density_or_mass > 0.0 && f > 0.0 && f < 1.0) {
      // NOLINTNEXTLINE(misc-include-cleaner): Clang 18 header mapping bug.
      const auto count =
          std::ranges::upper_bound(sorted, static_cast<Scalar>(k));
      const long double empirical =
          static_cast<long double>(count - sorted.begin()) / n;
      const long double difference = empirical - f;
      statistic += static_cast<long double>(n) * tails.value().density_or_mass *
                   difference * difference / (f * (1.0L - f));
    }
    if (k > observedMaximum && law.family() != probability::Family::bernoulli &&
        law.family() != probability::Family::binomial) {
      const Scalar tailCdf = tails.value().cdf;
      const Scalar sf = tails.value().survival;
      const long double bound =
          tailCdf > 0.0 ? static_cast<long double>(n) * sf * sf / tailCdf
                        : std::numeric_limits<long double>::infinity();
      if (bound <= 1.0e-12L + (1.0e-10L * statistic)) {
        converged = true;
        break;
      }
    }
  }
  if (law.family() != probability::Family::bernoulli &&
      law.family() != probability::Family::binomial && !converged) {
    return Result<Scalar>::failure(ErrorCode::NoConvergence,
                                   "discrete AD tail bound did not converge");
  }
  if (!std::isfinite(statistic) ||
      statistic > std::numeric_limits<Scalar>::max()) {
    return Result<Scalar>::failure(ErrorCode::NonFiniteResult,
                                   "discrete AD statistic is unrepresentable");
  }
  return Result<Scalar>::success(static_cast<Scalar>(statistic));
}

Result<Distribution> lawFromParameters(const Distribution& original,
                                       ConstVectorView fitted) {
  const auto p = fitted;
  const auto family = original.family();
  const auto exact = [&](Index count) { return p.size() == count; };
  switch (family) {
  case probability::Family::normal:
    return exact(2U) ? probability::normal(p.at(0U), p.at(1U))
                     : Result<Distribution>::failure(
                           ErrorCode::DimensionMismatch,
                           "normal refit requires mean and scale");
  case probability::Family::student_t:
    return exact(3U) ? probability::studentT(p.at(0U), p.at(1U), p.at(2U))
                     : Result<Distribution>::failure(
                           ErrorCode::DimensionMismatch,
                           "Student refit requires three parameters");
  case probability::Family::laplace:
    return exact(2U) ? probability::laplace(p.at(0U), p.at(1U))
                     : Result<Distribution>::failure(
                           ErrorCode::DimensionMismatch,
                           "Laplace refit requires two parameters");
  case probability::Family::logistic:
    return exact(2U) ? probability::logistic(p.at(0U), p.at(1U))
                     : Result<Distribution>::failure(
                           ErrorCode::DimensionMismatch,
                           "logistic refit requires two parameters");
  case probability::Family::lognormal:
    return exact(2U) ? probability::lognormal(p.at(0U), p.at(1U))
                     : Result<Distribution>::failure(
                           ErrorCode::DimensionMismatch,
                           "lognormal refit requires two parameters");
  case probability::Family::gamma:
    return exact(2U) ? probability::gamma(p.at(0U), p.at(1U))
                     : Result<Distribution>::failure(
                           ErrorCode::DimensionMismatch,
                           "gamma refit requires two parameters");
  case probability::Family::beta:
    return exact(2U) ? probability::beta(p.at(0U), p.at(1U))
                     : Result<Distribution>::failure(
                           ErrorCode::DimensionMismatch,
                           "beta refit requires two parameters");
  case probability::Family::weibull:
    return exact(2U) ? probability::weibull(p.at(0U), p.at(1U))
                     : Result<Distribution>::failure(
                           ErrorCode::DimensionMismatch,
                           "Weibull refit requires two parameters");
  case probability::Family::skew_normal:
    return exact(3U) ? probability::skewNormal(p.at(0U), p.at(1U), p.at(2U))
                     : Result<Distribution>::failure(
                           ErrorCode::DimensionMismatch,
                           "skew-normal refit requires three parameters");
  case probability::Family::uniform:
    return exact(2U) ? probability::uniform(p.at(0U), p.at(1U))
                     : Result<Distribution>::failure(
                           ErrorCode::DimensionMismatch,
                           "uniform refit requires two parameters");
  case probability::Family::triangular:
    return exact(3U) ? probability::triangular(p.at(0U), p.at(1U), p.at(2U))
                     : Result<Distribution>::failure(
                           ErrorCode::DimensionMismatch,
                           "triangular refit requires three parameters");
  case probability::Family::truncated_normal:
    return exact(4U) ? probability::truncatedNormal(p.at(0U), p.at(1U),
                                                    p.at(2U), p.at(3U))
                     : Result<Distribution>::failure(
                           ErrorCode::DimensionMismatch,
                           "truncated-normal refit requires four parameters");
  case probability::Family::bernoulli:
    return exact(1U) ? probability::bernoulli(p.at(0U))
                     : Result<Distribution>::failure(
                           ErrorCode::DimensionMismatch,
                           "Bernoulli refit requires success probability");
  case probability::Family::binomial:
    if (!exact(1U) || original.parameters().empty()) {
      return Result<Distribution>::failure(
          ErrorCode::DimensionMismatch,
          "binomial refit requires success probability");
    }
    return probability::binomial(
        static_cast<std::uint64_t>(original.parameters()[0]), p.at(0U));
  case probability::Family::poisson:
    return exact(1U)
               ? probability::poisson(p.at(0U))
               : Result<Distribution>::failure(ErrorCode::DimensionMismatch,
                                               "Poisson refit requires rate");
  case probability::Family::negative_binomial:
    if (!exact(1U) || original.parameters().empty()) {
      return Result<Distribution>::failure(
          ErrorCode::DimensionMismatch,
          "negative-binomial refit requires success probability");
    }
    return probability::negativeBinomial(original.parameters()[0], p.at(0U));
  case probability::Family::chi_square:
    return exact(1U) ? probability::chiSquare(p.at(0U))
                     : Result<Distribution>::failure(
                           ErrorCode::DimensionMismatch,
                           "chi-square refit requires degrees of freedom");
  case probability::Family::fisher_f:
    return exact(2U) ? probability::fisherF(p.at(0U), p.at(1U))
                     : Result<Distribution>::failure(
                           ErrorCode::DimensionMismatch,
                           "F refit requires two degrees of freedom");
  default:
    return Result<Distribution>::failure(
        ErrorCode::InvalidDomain,
        "legacy parameter refit is undefined for this distribution family");
  }
}

template <class StatisticFunction>
Result<TestResult>
goodness(const SampleView& sample, const Distribution& law, bool statisticOnly,
         Index replicates,
         const std::function<Result<Distribution>(ConstVectorView)>& refit,
         probability::Rng& rng, Limits limits,
         StatisticFunction statisticFunction) {
  auto observed = scalarValues(sample, limits);
  if (!observed.hasValue()) {
    return propagate<TestResult>(observed.status());
  }
  const Index n = observed.value().size();
  if (n > limits.elements / 3U || n > limits.bytes / (3U * sizeof(Scalar))) {
    return Result<TestResult>::failure(
        ErrorCode::ResourceLimit,
        "goodness-of-fit sample and sorting workspace exceed limits");
  }
  Index draws = 0U;
  if (!statisticOnly) {
    if (replicates == 0U) {
      return Result<TestResult>::failure(
          ErrorCode::InvalidDomain,
          "calibrated goodness-of-fit requires positive replicates");
    }
    if (replicates == std::numeric_limits<Index>::max() ||
        replicates + 1U > std::numeric_limits<Index>::max() / n) {
      return Result<TestResult>::failure(
          ErrorCode::SizeOverflow, "goodness calibration draw count overflows");
    }
    draws = (replicates + 1U) * n;
    const Index callbackCalls = refit ? replicates + 1U : 0U;
    if (draws > limits.evaluations ||
        callbackCalls > limits.evaluations - draws) {
      return Result<TestResult>::failure(
          ErrorCode::ResourceLimit,
          "goodness calibration exceeds the draw/evaluation limit");
    }
  }
  Distribution observedLaw = law;
  if (refit) {
    auto fitted = refit(observed.value().view());
    if (!fitted.hasValue()) {
      return Result<TestResult>::failure(
          ErrorCode::CallbackFailure,
          "goodness-of-fit observed-data refit callback failed");
    }
    if (fitted.value().measure() != law.measure()) {
      return Result<TestResult>::failure(
          ErrorCode::DimensionMismatch,
          "refitted goodness-of-fit law changed its measure");
    }
    observedLaw = std::move(fitted).value();
  }
  const auto observedStatistic =
      statisticFunction(observed.value().view(), observedLaw, limits);
  if (!observedStatistic.hasValue()) {
    return propagate<TestResult>(observedStatistic.status());
  }
  TestResult result{.statistic = observedStatistic.value(),
                    .p_value = std::nullopt,
                    .effective_df = std::nullopt,
                    .sampling_error = std::nullopt,
                    .calibration_method = {}};
  if (statisticOnly) {
    return Result<TestResult>::success(std::move(result));
  }
  Index exceedances = 0U;
  auto samplingPolicy = goodnessPolicy(draws);
  for (Index b = 0U; b < replicates; ++b) {
    Vector generated(observed.value().size());
    for (Index i = 0U; i < generated.size(); ++i) {
      const auto draw = probability::sample(observedLaw, rng, samplingPolicy);
      if (!draw.hasValue()) {
        return propagate<TestResult>(draw.status());
      }
      generated.at(i) = draw.value();
    }
    Distribution replicateLaw = observedLaw;
    if (refit) {
      auto refitted = refit(generated.view());
      if (!refitted.hasValue()) {
        return Result<TestResult>::failure(
            ErrorCode::CallbackFailure,
            "goodness-of-fit replicate refit callback failed");
      }
      if (refitted.value().measure() != law.measure()) {
        return Result<TestResult>::failure(
            ErrorCode::DimensionMismatch,
            "replicate refit changed the declared measure");
      }
      replicateLaw = std::move(refitted).value();
    }
    const auto replicateStatistic =
        statisticFunction(generated.view(), replicateLaw, limits);
    if (!replicateStatistic.hasValue()) {
      return propagate<TestResult>(replicateStatistic.status());
    }
    exceedances +=
        replicateStatistic.value() >= observedStatistic.value() ? 1U : 0U;
  }
  const Scalar p = static_cast<Scalar>(exceedances + 1U) /
                   static_cast<Scalar>(replicates + 1U);
  result.p_value = p;
  result.sampling_error =
      std::sqrt(p * (1.0 - p) / static_cast<Scalar>(replicates + 1U));
  result.calibration_method = refit ? "approximate_composite_bootstrap"
                                    : "simple-null parametric Monte Carlo";
  return Result<TestResult>::success(std::move(result));
}

} // namespace

Result<Scalar> aic(Scalar logLikelihood, Index parameters) {
  if (!std::isfinite(logLikelihood)) {
    return Result<Scalar>::failure(ErrorCode::NonFiniteInput,
                                   "AIC log likelihood must be finite");
  }
  const long double value =
      (-2.0L * logLikelihood) + (2.0L * static_cast<long double>(parameters));
  if (!std::isfinite(value) ||
      std::abs(value) > std::numeric_limits<Scalar>::max()) {
    return Result<Scalar>::failure(ErrorCode::NonFiniteResult,
                                   "AIC result is not representable");
  }
  return Result<Scalar>::success(static_cast<Scalar>(value));
}

Result<Scalar> bic(Scalar logLikelihood, Index parameters,
                   Index independentCount) {
  if (!std::isfinite(logLikelihood)) {
    return Result<Scalar>::failure(ErrorCode::NonFiniteInput,
                                   "BIC log likelihood must be finite");
  }
  if (independentCount == 0U) {
    return Result<Scalar>::failure(ErrorCode::InvalidDomain,
                                   "BIC independent count must be positive");
  }
  const long double value =
      (-2.0L * logLikelihood) +
      (static_cast<long double>(parameters) *
       std::log(static_cast<long double>(independentCount)));
  if (!std::isfinite(value) ||
      std::abs(value) > std::numeric_limits<Scalar>::max()) {
    return Result<Scalar>::failure(ErrorCode::NonFiniteResult,
                                   "BIC result is not representable");
  }
  return Result<Scalar>::success(static_cast<Scalar>(value));
}

Result<TestResult> ks(const SampleView& sample, const Distribution& law,
                      const GoodnessCalibration& calibration,
                      probability::Rng& rng, Limits limits) {
  const auto refit = calibration.refit_law;
  return goodness(sample, law, calibration.statistic_only,
                  calibration.replicates, refit, rng, limits, ksStatistic);
}

Result<TestResult> andersonDarling(const SampleView& sample,
                                   const Distribution& law,
                                   const GoodnessCalibration& calibration,
                                   probability::Rng& rng, Limits limits) {
  const auto refit = calibration.refit_law;
  return goodness(sample, law, calibration.statistic_only,
                  calibration.replicates, refit, rng, limits,
                  andersonDarlingStatistic);
}

Result<TestResult> ks(const SampleView& sample, const Distribution& law,
                      const GoodnessSpec& specification, probability::Rng& rng,
                      Limits limits) {
  std::function<Result<Distribution>(ConstVectorView)> refit;
  if (specification.refit_parameters) {
    const auto callback = specification.refit_parameters;
    refit = [&law, callback](ConstVectorView values) {
      const auto fitted = callback(values);
      if (!fitted.hasValue()) {
        return propagate<Distribution>(fitted.status());
      }
      return lawFromParameters(law, fitted.value().view());
    };
  }
  if (!specification.statistic_only &&
      specification.calibration.draws != specification.replicates) {
    return Result<TestResult>::failure(
        ErrorCode::InvalidDomain,
        "GoodnessSpec calibration draws must equal replicates");
  }
  return goodness(sample, law, specification.statistic_only,
                  specification.replicates, refit, rng, limits, ksStatistic);
}

Result<TestResult> andersonDarling(const SampleView& sample,
                                   const Distribution& law,
                                   const GoodnessSpec& specification,
                                   probability::Rng& rng, Limits limits) {
  std::function<Result<Distribution>(ConstVectorView)> refit;
  if (specification.refit_parameters) {
    const auto callback = specification.refit_parameters;
    refit = [&law, callback](ConstVectorView values) {
      const auto fitted = callback(values);
      if (!fitted.hasValue()) {
        return propagate<Distribution>(fitted.status());
      }
      return lawFromParameters(law, fitted.value().view());
    };
  }
  if (!specification.statistic_only &&
      specification.calibration.draws != specification.replicates) {
    return Result<TestResult>::failure(
        ErrorCode::InvalidDomain,
        "GoodnessSpec calibration draws must equal replicates");
  }
  return goodness(sample, law, specification.statistic_only,
                  specification.replicates, refit, rng, limits,
                  andersonDarlingStatistic);
}

} // namespace vss::math::statistics
