#include <cmath>
#include <limits>
#include <string>
#include <vss/math/Error.hpp>
#include <vss/math/probability/Composite.hpp>

namespace vss::math::probability {
namespace {

Result<ObservationLikelihood> invalidEvent(const char* message) {
  return Result<ObservationLikelihood>::failure(ErrorCode::InvalidDomain,
                                                message);
}

Result<ObservationLikelihood> intervalLikelihood(const Distribution& law,
                                                 IntervalSupport support,
                                                 ProbabilityPolicy policy,
                                                 bool quantized) {
  const auto likelihood = quantized
                              ? quantizedLikelihood(law, support, policy)
                              : censoredLogProbability(law, support, policy);
  if (!likelihood) {
    return Result<ObservationLikelihood>::failure(
        likelihood.status().code(), std::string(likelihood.status().message()));
  }
  return Result<ObservationLikelihood>::success(
      ObservationLikelihood{likelihood.value()});
}

} // namespace

Result<ObservationLikelihood>
nonidealObservationLogProbability(const Distribution& law,
                                  const ObservationEvent& event,
                                  ProbabilityPolicy policy) {
  switch (event.kind) {
  case ObservationKind::exact: {
    if (!event.value || event.support || event.omit_unavailable) {
      return invalidEvent("exact observations require only a finite value");
    }
    if (!std::isfinite(*event.value)) {
      return Result<ObservationLikelihood>::failure(
          ErrorCode::NonFiniteInput, "exact observation is not finite");
    }
    const auto likelihood = evaluate(law, *event.value, policy);
    if (!likelihood) {
      return Result<ObservationLikelihood>::failure(
          likelihood.status().code(),
          std::string(likelihood.status().message()));
    }
    return Result<ObservationLikelihood>::success(
        ObservationLikelihood{likelihood.value().log_density_or_mass});
  }
  case ObservationKind::interval:
  case ObservationKind::below_threshold:
  case ObservationKind::above_threshold:
  case ObservationKind::quantized: {
    if (event.value || !event.support || event.omit_unavailable) {
      return invalidEvent(
          "reported interval events require only their support");
    }
    const auto support = *event.support;
    if (event.kind == ObservationKind::below_threshold &&
        support.lower != -std::numeric_limits<Scalar>::infinity()) {
      return invalidEvent("below-threshold support must be lower-unbounded");
    }
    if (event.kind == ObservationKind::above_threshold &&
        support.upper != std::numeric_limits<Scalar>::infinity()) {
      return invalidEvent("above-threshold support must be upper-unbounded");
    }
    if (event.kind == ObservationKind::quantized &&
        (!std::isfinite(support.lower) || !std::isfinite(support.upper))) {
      return invalidEvent("quantized bins must have finite endpoints");
    }
    return intervalLikelihood(law, support, policy,
                              event.kind == ObservationKind::quantized);
  }
  case ObservationKind::unavailable:
    if (event.value || event.support || !event.omit_unavailable) {
      return invalidEvent(
          "unavailable observations require explicit omission and no value");
    }
    return Result<ObservationLikelihood>::success(ObservationLikelihood{});
  }
  return invalidEvent("observation kind is invalid");
}

} // namespace vss::math::probability
