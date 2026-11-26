#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <new>
#include <queue>
#include <string>
#include <vector>
#include <vss/math/Core.hpp>
#include <vss/math/Error.hpp>
#include <vss/math/Integration.hpp>
#include <vss/math/Limits.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/TensorElementwise.hpp>
#include <vss/math/VectorView.hpp>
#include <vss/math/linear/Iterative.hpp>
#include <vss/math/linear/SolvePolicy.hpp>

namespace vss::math {
namespace {

using Wide = long double;

constexpr std::array<Scalar, 8> kronrodNodes{
    0.991455371120812639206854697526329, 0.949107912342758524526189684047851,
    0.864864423359769072789712788640926, 0.741531185599394439863864773280788,
    0.586087235467691130294144838258730, 0.405845151377397166906606412076961,
    0.207784955007898467600689403773245, 0.0};
constexpr std::array<Scalar, 8> kronrodWeights{
    0.022935322010529224963732008058970, 0.063092092629978553290700663189204,
    0.104790010322250183839876322541518, 0.140653259715525918745189590510238,
    0.169004726639267902826583426598550, 0.190350578064785409913256402421014,
    0.204432940075298892414161999234649, 0.209482141084727828012999174891714};
constexpr std::array<Scalar, 8> gaussWeights{
    0.0, 0.129484966168869693270611432679082,
    0.0, 0.279705391489276667901467771423780,
    0.0, 0.381830050505118944950369775488975,
    0.0, 0.417959183673469387755102040816327};

struct Panel final {
  Scalar lower{};
  Scalar upper{};
  Scalar value{};
  Scalar error{};
};

struct LargerErrorFirst final {
  bool operator()(const Panel& left, const Panel& right) const noexcept {
    if (left.error != right.error) {
      return left.error < right.error;
    }
    return left.lower > right.lower;
  }
};

Scalar intervalMidpoint(Scalar lower, Scalar upper) noexcept {
  return static_cast<Scalar>((static_cast<Wide>(lower) + upper) / 2.0L);
}

void compensatedAdd(Wide value, Wide& sum, Wide& correction) noexcept {
  const Wide adjusted = value - correction;
  const Wide next = sum + adjusted;
  correction = (next - sum) - adjusted;
  sum = next;
}

bool finiteScalar(Wide value, Scalar& output) noexcept {
  if (!std::isfinite(value) ||
      std::abs(value) > static_cast<Wide>(std::numeric_limits<Scalar>::max())) {
    return false;
  }
  output = static_cast<Scalar>(value);
  return std::isfinite(output);
}

Result<Scalar> callbackValue(const UnaryFunction& function, Scalar x,
                             Index& evaluations, Index maximum) {
  if (evaluations >= maximum) {
    return Result<Scalar>::failure(ErrorCode::ResourceLimit,
                                   "quadrature evaluation budget exhausted");
  }
  ++evaluations;
  try {
    Result<Scalar> result = function(x);
    if (!result) {
      return Result<Scalar>::failure(result.status().code(),
                                     std::string(result.status().message()));
    }
    if (!std::isfinite(result.value())) {
      return Result<Scalar>::failure(ErrorCode::NonFiniteInput,
                                     "integrand returned a non-finite sample");
    }
    return result;
  } catch (...) {
    return Result<Scalar>::failure(ErrorCode::CallbackFailure,
                                   "integrand threw an exception");
  }
}

Result<Panel> evaluatePanel(const UnaryFunction& function, Scalar lower,
                            Scalar upper, Index& evaluations, Index maximum) {
  const Wide a = static_cast<Wide>(lower);
  const Wide b = static_cast<Wide>(upper);
  const Wide centerWide = a + (b - a) / 2.0L;
  const Wide halfWide = (b - a) / 2.0L;
  const auto center = static_cast<Scalar>(centerWide);
  auto centerValue = callbackValue(function, center, evaluations, maximum);
  if (!centerValue) {
    return Result<Panel>::failure(centerValue.status().code(),
                                  std::string(centerValue.status().message()));
  }

  std::array<Scalar, 15> values{};
  values[14] = centerValue.value();
  Index slot = 0U;
  for (int node = 6; node >= 0; --node) {
    const Wide offset =
        halfWide * kronrodNodes.at(static_cast<std::size_t>(node));
    const auto left = static_cast<Scalar>(centerWide - offset);
    const auto right = static_cast<Scalar>(centerWide + offset);
    if (left <= lower || left >= center || right <= center || right >= upper ||
        left == right) {
      return Result<Panel>::failure(
          ErrorCode::NoConvergence,
          "quadrature panel has no distinct representable interior nodes");
    }
    auto leftValue = callbackValue(function, left, evaluations, maximum);
    if (!leftValue) {
      return Result<Panel>::failure(leftValue.status().code(),
                                    std::string(leftValue.status().message()));
    }
    auto rightValue = callbackValue(function, right, evaluations, maximum);
    if (!rightValue) {
      return Result<Panel>::failure(rightValue.status().code(),
                                    std::string(rightValue.status().message()));
    }
    values.at(slot++) = leftValue.value();
    values.at(slot++) = rightValue.value();
  }

  Wide kronrod = 0.0L;
  Wide kronrodCorrection = 0.0L;
  Wide gauss = 0.0L;
  Wide gaussCorrection = 0.0L;
  Wide absolute = 0.0L;
  Wide absoluteCorrection = 0.0L;
  for (std::size_t i = 0U; i < 7U; ++i) {
    const std::size_t left = i * 2U;
    const std::size_t right = left + 1U;
    const std::size_t node = 6U - i;
    const Wide pair = static_cast<Wide>(values.at(left)) + values.at(right);
    compensatedAdd(static_cast<Wide>(kronrodWeights.at(node)) * pair, kronrod,
                   kronrodCorrection);
    compensatedAdd(static_cast<Wide>(kronrodWeights.at(node)) *
                       (std::abs(static_cast<Wide>(values.at(left))) +
                        std::abs(static_cast<Wide>(values.at(right)))),
                   absolute, absoluteCorrection);
    if (node == 1U || node == 3U || node == 5U) {
      compensatedAdd(static_cast<Wide>(gaussWeights.at(node)) * pair, gauss,
                     gaussCorrection);
    }
  }
  compensatedAdd(static_cast<Wide>(kronrodWeights[7]) * values[14], kronrod,
                 kronrodCorrection);
  compensatedAdd(static_cast<Wide>(kronrodWeights[7]) *
                     std::abs(static_cast<Wide>(values[14])),
                 absolute, absoluteCorrection);
  compensatedAdd(static_cast<Wide>(gaussWeights[7]) * values[14], gauss,
                 gaussCorrection);

  const Wide mean = kronrod / 2.0L;
  Wide variation = 0.0L;
  Wide variationCorrection = 0.0L;
  for (std::size_t i = 0U; i < 7U; ++i) {
    const std::size_t node = 6U - i;
    compensatedAdd(
        static_cast<Wide>(kronrodWeights.at(node)) *
            (std::abs(static_cast<Wide>(values.at(i * 2U)) - mean) +
             std::abs(static_cast<Wide>(values.at(i * 2U + 1U)) - mean)),
        variation, variationCorrection);
  }
  compensatedAdd(static_cast<Wide>(kronrodWeights[7]) *
                     std::abs(static_cast<Wide>(values[14]) - mean),
                 variation, variationCorrection);
  const Wide kValue = kronrod * halfWide;
  const Wide gValue = gauss * halfWide;
  const Wide aValue = absolute * halfWide;
  const Wide vValue = variation * halfWide;
  const Wide discrepancy = std::abs(kValue - gValue);
  Wide error = discrepancy;
  if (vValue > 0.0L && discrepancy > 0.0L) {
    const Wide ratio = 200.0L * discrepancy / vValue;
    error = vValue * std::min(1.0L, std::pow(ratio, 1.5L));
  }
  error =
      std::max(error, 50.0L * std::numeric_limits<Scalar>::epsilon() * aValue);
  Scalar convertedValue = 0.0;
  Scalar convertedError = 0.0;
  if (error < 0.0L || !finiteScalar(kValue, convertedValue) ||
      !finiteScalar(error, convertedError)) {
    return Result<Panel>::failure(
        ErrorCode::NonFiniteResult,
        "quadrature panel result is not representable");
  }
  return Result<Panel>::success(
      Panel{lower, upper, convertedValue, convertedError});
}

} // namespace

Result<IntegralResult> quadratureAdaptive(const UnaryFunction& function,
                                          Interval limits,
                                          const IterPolicy& policy,
                                          Limits resources) {
  if (!std::isfinite(limits.lower) || !std::isfinite(limits.upper)) {
    return Result<IntegralResult>::failure(
        ErrorCode::NonFiniteInput, "quadrature endpoints must be finite");
  }
  if (!std::isfinite(policy.abs) || !std::isfinite(policy.rel) ||
      policy.abs < 0.0 || policy.rel < 0.0) {
    return Result<IntegralResult>::failure(ErrorCode::InvalidTolerance,
                                           "quadrature tolerances are invalid");
  }
  if (policy.bracket.has_value()) {
    return Result<IntegralResult>::failure(
        ErrorCode::InvalidDomain, "quadrature does not accept a root bracket");
  }
  if (limits.lower == limits.upper) {
    return Result<IntegralResult>::success(
        {0.0, 0.0, 0U, 0U, Termination::converged});
  }
  if (!function) {
    return Result<IntegralResult>::failure(ErrorCode::CallbackFailure,
                                           "quadrature integrand is empty");
  }
  const Index maxEvaluations =
      std::min(policy.max_evaluations, resources.evaluations);
  if (maxEvaluations < 15U || resources.elements < 1U ||
      resources.bytes < sizeof(Panel) + sizeof(IntegralResult)) {
    return Result<IntegralResult>::failure(
        ErrorCode::ResourceLimit, "quadrature initial panel exceeds limits");
  }
  const bool reversed = limits.upper < limits.lower;
  const Scalar lower = reversed ? limits.upper : limits.lower;
  const Scalar upper = reversed ? limits.lower : limits.upper;
  Index evaluations = 0U;
  auto initial =
      evaluatePanel(function, lower, upper, evaluations, maxEvaluations);
  if (!initial) {
    return Result<IntegralResult>::failure(
        initial.status().code(), std::string(initial.status().message()));
  }
  std::priority_queue<Panel, std::vector<Panel>, LargerErrorFirst> panels;
  try {
    panels.push(initial.value());
  } catch (const std::bad_alloc&) {
    return Result<IntegralResult>::failure(
        ErrorCode::ResourceLimit, "quadrature panel heap allocation failed");
  }
  Wide total = initial.value().value;
  Wide totalCorrection = 0.0L;
  Wide totalError = initial.value().error;
  Wide errorCorrection = 0.0L;
  evaluations = 15U;
  Index leaves = 1U;
  Index iterations = 0U;
  auto converged = [&]() {
    const Wide tolerance = static_cast<Wide>(policy.abs) +
                           static_cast<Wide>(policy.rel) * std::abs(total);
    const Wide outward = 64.0L * std::numeric_limits<Scalar>::epsilon() *
                         (std::abs(total) + std::abs(totalError));
    return totalError + outward <= tolerance;
  };
  while (!converged()) {
    if (iterations >= policy.max_iterations) {
      Scalar value = 0.0;
      Scalar error = 0.0;
      if (!finiteScalar(reversed ? -total : total, value) ||
          !finiteScalar(totalError, error)) {
        return Result<IntegralResult>::failure(
            ErrorCode::NonFiniteResult,
            "quadrature total is not representable");
      }
      return Result<IntegralResult>::success(
          {value, error, evaluations, leaves, Termination::cap});
    }
    if (panels.empty() || maxEvaluations - evaluations < 30U ||
        leaves >
            resources.elements - std::min<Extent>(resources.elements, 2U)) {
      return Result<IntegralResult>::failure(
          ErrorCode::ResourceLimit, "quadrature split exceeds work limits");
    }
    const std::size_t simultaneousPanels =
        static_cast<std::size_t>(leaves) + 2U;
    if (simultaneousPanels >
        (resources.bytes - sizeof(IntegralResult)) / sizeof(Panel)) {
      return Result<IntegralResult>::failure(
          ErrorCode::ResourceLimit, "quadrature panel storage exceeds limits");
    }
    const Panel parent = panels.top();
    panels.pop();
    const Scalar middle = intervalMidpoint(parent.lower, parent.upper);
    if (middle <= parent.lower || middle >= parent.upper) {
      return Result<IntegralResult>::failure(
          ErrorCode::NoConvergence, "quadrature interval cannot be split");
    }
    auto left = evaluatePanel(function, parent.lower, middle, evaluations,
                              maxEvaluations);
    if (!left) {
      return Result<IntegralResult>::failure(
          left.status().code(), std::string(left.status().message()));
    }
    auto right = evaluatePanel(function, middle, parent.upper, evaluations,
                               maxEvaluations);
    if (!right) {
      return Result<IntegralResult>::failure(
          right.status().code(), std::string(right.status().message()));
    }
    compensatedAdd(-static_cast<Wide>(parent.value), total, totalCorrection);
    compensatedAdd(left.value().value, total, totalCorrection);
    compensatedAdd(right.value().value, total, totalCorrection);
    compensatedAdd(-static_cast<Wide>(parent.error), totalError,
                   errorCorrection);
    compensatedAdd(left.value().error, totalError, errorCorrection);
    compensatedAdd(right.value().error, totalError, errorCorrection);
    try {
      panels.push(left.value());
      panels.push(right.value());
    } catch (const std::bad_alloc&) {
      return Result<IntegralResult>::failure(
          ErrorCode::ResourceLimit, "quadrature panel heap allocation failed");
    }
    ++leaves;
    ++iterations;
  }
  Scalar value = 0.0;
  Scalar error = 0.0;
  if (!finiteScalar(reversed ? -total : total, value) ||
      !finiteScalar(std::max(0.0L, totalError), error)) {
    return Result<IntegralResult>::failure(
        ErrorCode::NonFiniteResult, "quadrature total is not representable");
  }
  return Result<IntegralResult>::success(
      {value, error, evaluations, leaves, Termination::converged});
}

Result<Scalar> integrateSamples(ConstVectorView nodes, ConstVectorView values,
                                SampleRule rule) {
  if (nodes.size() != values.size()) {
    return Result<Scalar>::failure(ErrorCode::DimensionMismatch,
                                   "sample nodes and values differ in extent");
  }
  if (rule != SampleRule::trapezoid && rule != SampleRule::simpson) {
    return Result<Scalar>::failure(ErrorCode::InvalidDomain,
                                   "unknown sampled integration rule");
  }
  const Extent minimum = rule == SampleRule::trapezoid ? 2U : 3U;
  if (nodes.size() < minimum) {
    return Result<Scalar>::failure(ErrorCode::InsufficientSamples,
                                   "sampled integration has too few values");
  }
  for (Index i = 0U; i < nodes.size(); ++i) {
    if (!std::isfinite(nodes.at(i)) || !std::isfinite(values.at(i))) {
      return Result<Scalar>::failure(
          ErrorCode::NonFiniteInput,
          "sampled integral contains non-finite data");
    }
    if (i > 0U && !(nodes.at(i) > nodes.at(i - 1U))) {
      return Result<Scalar>::failure(ErrorCode::InvalidDomain,
                                     "sample nodes must strictly increase");
    }
  }

  Wide sum = 0.0L;
  Wide correction = 0.0L;
  if (rule == SampleRule::trapezoid) {
    for (Index i = 0U; i + 1U < nodes.size(); ++i) {
      const Wide width = static_cast<Wide>(nodes.at(i + 1U)) - nodes.at(i);
      const Wide area =
          width * (static_cast<Wide>(values.at(i)) + values.at(i + 1U)) / 2.0L;
      compensatedAdd(area, sum, correction);
    }
  } else {
    const Extent panels = nodes.size() - 1U;
    if (panels % 2U != 0U) {
      return Result<Scalar>::failure(
          ErrorCode::InvalidDomain,
          "Simpson integration requires an even panel count");
    }
    const Wide width =
        static_cast<Wide>(nodes.at(nodes.size() - 1U)) - nodes.at(0U);
    const Wide delta = width / static_cast<Wide>(panels);
    for (Index i = 1U; i < panels; ++i) {
      const Wide expected =
          static_cast<Wide>(nodes.at(0U)) + static_cast<Wide>(i) * delta;
      const Wide allowed = 8.0L * std::numeric_limits<Scalar>::epsilon() *
                           std::max(delta, static_cast<Wide>(i) * delta);
      if (std::abs(static_cast<Wide>(nodes.at(i)) - expected) > allowed) {
        return Result<Scalar>::failure(
            ErrorCode::InvalidDomain,
            "Simpson samples are not uniformly spaced");
      }
    }
    compensatedAdd(values.at(0U), sum, correction);
    compensatedAdd(values.at(nodes.size() - 1U), sum, correction);
    for (Index i = 1U; i < panels; ++i) {
      compensatedAdd((i % 2U == 0U ? 2.0L : 4.0L) * values.at(i), sum,
                     correction);
    }
    sum *= delta / 3.0L;
  }
  Scalar result = 0.0;
  if (!finiteScalar(sum, result)) {
    return Result<Scalar>::failure(ErrorCode::NonFiniteResult,
                                   "sampled integral is not representable");
  }
  return Result<Scalar>::success(result);
}

} // namespace vss::math
