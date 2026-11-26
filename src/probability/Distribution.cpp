#include "DistributionInternal.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <limits>
#include <map>
#include <memory>
#include <numbers>
#include <numeric>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>
#include <vss/math/Core.hpp>
#include <vss/math/Error.hpp>
#include <vss/math/Integration.hpp>
#include <vss/math/Limits.hpp>
#include <vss/math/ProbabilityPolicy.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/VectorView.hpp>
#include <vss/math/probability/Composite.hpp>
#include <vss/math/probability/Distribution.hpp>
#include <vss/math/probability/LogTailPair.hpp>
#include <vss/math/probability/Special.hpp>

namespace vss::math::probability {
namespace {
constexpr std::size_t distributionWorkspaceElements = 1'000'000U;
constexpr std::size_t distributionWorkspaceBytes = 64U * 1024U * 1024U;
} // namespace

struct Distribution::State final {
  Family family{};
  Measure measure{};
  std::vector<Scalar> parameters;
  std::uint64_t integerParameter{};
  std::vector<std::shared_ptr<const State>> components;
  std::vector<Scalar> weights;
  std::vector<Scalar> atoms;
  std::vector<Scalar> masses;
  std::vector<Scalar> cumulativeMasses;
  std::vector<Scalar> survivalMasses;
  Index depth{1U};
};

namespace detail {

class DistributionFactory final {
public:
  static Result<Distribution> create(Family family, Measure measure,
                                     std::initializer_list<Scalar> parameters,
                                     std::uint64_t integer = 0U) {
    auto state = std::make_shared<Distribution::State>();
    state->family = family;
    state->measure = measure;
    state->parameters.assign(parameters.begin(), parameters.end());
    state->integerParameter = integer;
    return Result<Distribution>::success(Distribution(
        std::shared_ptr<const Distribution::State>(std::move(state))));
  }

  static Result<Distribution> empirical(std::vector<Scalar> atoms,
                                        std::vector<Scalar> masses,
                                        Limits limits,
                                        Family family = Family::empirical);

  static Result<Distribution>
  composite(Family family,
            std::vector<std::shared_ptr<const Distribution::State>> components,
            std::vector<Scalar> weights, Measure measure, Limits limits);

  static Result<Distribution> convolution(
      std::vector<std::shared_ptr<const Distribution::State>> components,
      Measure measure, Limits limits);
};

} // namespace detail

Result<Distribution>
detail::DistributionFactory::empirical(std::vector<Scalar> atoms,
                                       std::vector<Scalar> masses,
                                       Limits limits, Family family) {
  constexpr std::size_t retainedCap = 1024U;
  if (atoms.empty() || atoms.size() != masses.size()) {
    return Result<Distribution>::failure(
        ErrorCode::InvalidExtent,
        "empirical observations and masses must have equal nonzero length");
  }
  if (atoms.size() > retainedCap || atoms.size() > limits.elements / 4U ||
      atoms.size() > limits.bytes / (4U * sizeof(Scalar))) {
    return Result<Distribution>::failure(
        ErrorCode::ResourceLimit, "empirical law exceeds its storage limit");
  }
  Scalar maximumMass = 0.0;
  for (Index index = 0U; index < atoms.size(); ++index) {
    if (!std::isfinite(atoms[index]) || !std::isfinite(masses[index])) {
      return Result<Distribution>::failure(
          ErrorCode::NonFiniteInput,
          "empirical observations and masses must be finite");
    }
    if (masses[index] < 0.0) {
      return Result<Distribution>::failure(
          ErrorCode::InvalidDomain, "empirical masses must be nonnegative");
    }
    maximumMass = std::max(maximumMass, masses[index]);
  }
  if (maximumMass == 0.0) {
    return Result<Distribution>::failure(
        ErrorCode::InvalidDomain, "empirical law requires positive total mass");
  }
  std::vector<std::pair<Scalar, Scalar>> weighted;
  weighted.reserve(atoms.size());
  for (Index index = 0U; index < atoms.size(); ++index) {
    const Scalar scaled = masses[index] / maximumMass;
    if (masses[index] > 0.0 && scaled == 0.0) {
      return Result<Distribution>::failure(
          ErrorCode::NonFiniteResult,
          "empirical positive mass is below representable normalization range");
    }
    if (scaled > 0.0) {
      weighted.emplace_back(atoms[index], scaled);
    }
  }
  std::vector<Scalar>().swap(atoms);
  std::vector<Scalar>().swap(masses);
  std::sort(weighted.begin(), weighted.end(),
            [](const auto& left, const auto& right) {
              return left.first < right.first;
            });
  std::vector<Scalar> uniqueAtoms;
  std::vector<Scalar> uniqueMasses;
  uniqueAtoms.reserve(weighted.size());
  uniqueMasses.reserve(weighted.size());
  for (const auto& [atom, mass] : weighted) {
    if (!uniqueAtoms.empty() && uniqueAtoms.back() == atom) {
      uniqueMasses.back() += mass;
    } else {
      uniqueAtoms.push_back(atom);
      uniqueMasses.push_back(mass);
    }
  }
  std::vector<std::pair<Scalar, Scalar>>().swap(weighted);
  long double total = 0.0L;
  long double massCorrection = 0.0L;
  for (const Scalar mass : uniqueMasses) {
    const long double adjusted =
        static_cast<long double>(mass) - massCorrection;
    const long double next = total + adjusted;
    massCorrection = (next - total) - adjusted;
    total = next;
  }
  if (!std::isfinite(total) || total <= 0.0) {
    return Result<Distribution>::failure(
        ErrorCode::NonFiniteResult, "empirical normalized mass is not finite");
  }
  for (Scalar& mass : uniqueMasses) {
    mass = static_cast<Scalar>(static_cast<long double>(mass) / total);
    if (mass == 0.0) {
      return Result<Distribution>::failure(
          ErrorCode::NonFiniteResult,
          "empirical positive normalized mass is not representable");
    }
  }
  auto state = std::make_shared<Distribution::State>();
  state->family = family;
  state->measure = Measure::atom_mass;
  state->atoms = std::move(uniqueAtoms);
  state->masses = std::move(uniqueMasses);
  state->cumulativeMasses.resize(state->masses.size());
  state->survivalMasses.resize(state->masses.size());
  long double cumulative = 0.0L;
  for (Index index = 0U; index < state->masses.size(); ++index) {
    cumulative += state->masses[index];
    state->cumulativeMasses[index] = static_cast<Scalar>(cumulative);
  }
  long double survival = 0.0L;
  for (Index index = state->masses.size(); index-- > 0U;) {
    survival += state->masses[index];
    state->survivalMasses[index] = static_cast<Scalar>(survival);
  }
  state->cumulativeMasses.back() = 1.0;
  state->survivalMasses.front() = 1.0;
  return Result<Distribution>::success(Distribution(std::move(state)));
}

Result<Distribution> detail::DistributionFactory::composite(
    Family family,
    std::vector<std::shared_ptr<const Distribution::State>> components,
    std::vector<Scalar> weights, Measure measure, Limits limits) {
  constexpr std::size_t retainedCap = 1024U;
  constexpr std::size_t byteCap = std::size_t{16U} * 1024U * 1024U;
  if (components.empty() || components.size() != weights.size()) {
    return Result<Distribution>::failure(
        ErrorCode::InvalidExtent,
        "composite law requires matching nonempty components and weights");
  }
  const std::size_t effectiveBytes = std::min(limits.bytes, byteCap);
  if (components.size() > retainedCap || components.size() > limits.elements ||
      components.size() >
          effectiveBytes / (sizeof(std::shared_ptr<const Distribution::State>) +
                            sizeof(Scalar))) {
    return Result<Distribution>::failure(
        ErrorCode::ResourceLimit, "composite law exceeds its storage limit");
  }
  Scalar maximumWeight = 0.0;
  Index maximumDepth = 0U;
  for (Index index = 0U; index < components.size(); ++index) {
    if (!components[index] || !std::isfinite(weights[index]) ||
        weights[index] < 0.0) {
      return Result<Distribution>::failure(
          ErrorCode::InvalidDomain,
          "composite weights or components are invalid");
    }
    if (components[index]->measure != measure && weights[index] > 0.0) {
      return Result<Distribution>::failure(
          ErrorCode::InvalidDomain,
          "positive composite components must share the same measure");
    }
    maximumWeight = std::max(maximumWeight, weights[index]);
    maximumDepth = std::max(maximumDepth, components[index]->depth);
  }
  if (maximumWeight == 0.0) {
    return Result<Distribution>::failure(
        ErrorCode::InvalidDomain, "composite law requires positive mass");
  }
  if (maximumDepth >= 32U) {
    return Result<Distribution>::failure(ErrorCode::ResourceLimit,
                                         "composite nesting exceeds depth 32");
  }
  long double scaledTotal = 0.0L;
  long double weightCorrection = 0.0L;
  for (Scalar& weight : weights) {
    const bool positive = weight > 0.0;
    weight /= maximumWeight;
    if (positive && weight == 0.0) {
      return Result<Distribution>::failure(ErrorCode::NonFiniteResult,
                                           "positive composite weight is below "
                                           "representable normalization range");
    }
    const long double adjusted =
        static_cast<long double>(weight) - weightCorrection;
    const long double next = scaledTotal + adjusted;
    weightCorrection = (next - scaledTotal) - adjusted;
    scaledTotal = next;
  }
  if (!std::isfinite(scaledTotal) || scaledTotal <= 0.0) {
    return Result<Distribution>::failure(
        ErrorCode::NonFiniteResult, "composite weight total is not finite");
  }
  for (Scalar& weight : weights) {
    weight =
        static_cast<Scalar>(static_cast<long double>(weight) / scaledTotal);
  }
  auto state = std::make_shared<Distribution::State>();
  state->family = family;
  state->measure = measure;
  state->components = std::move(components);
  state->weights = std::move(weights);
  state->depth = maximumDepth + 1U;
  return Result<Distribution>::success(Distribution(std::move(state)));
}

Result<Distribution> detail::DistributionFactory::convolution(
    std::vector<std::shared_ptr<const Distribution::State>> components,
    Measure measure, Limits limits) {
  constexpr std::size_t depthCap = 32U;
  constexpr std::size_t componentCap = 1024U;
  const std::size_t bytesPerComponent =
      sizeof(std::shared_ptr<const Distribution::State>);
  if (components.size() > depthCap || components.size() > componentCap ||
      components.size() > limits.elements ||
      components.size() > limits.bytes / bytesPerComponent) {
    return Result<Distribution>::failure(
        ErrorCode::ResourceLimit, "convolution expression exceeds limits");
  }
  Index depth = 1U;
  for (const auto& component : components) {
    if (!component) {
      return Result<Distribution>::failure(
          ErrorCode::InvalidDomain, "convolution contains a moved-from law");
    }
    depth = std::max(depth, component->depth + 1U);
  }
  if (depth > depthCap) {
    return Result<Distribution>::failure(
        ErrorCode::ResourceLimit, "convolution nesting exceeds depth 32");
  }
  auto state = std::make_shared<Distribution::State>();
  state->family = Family::finite_convolution;
  state->measure = measure;
  state->components = std::move(components);
  state->depth = depth;
  return Result<Distribution>::success(Distribution(std::move(state)));
}

Result<Distribution>
detail::DistributionEngine::mixture(std::span<const WeightedLaw> terms,
                                    ProbabilityPolicy policy, Limits limits) {
  if (!std::isfinite(policy.iteration.abs) ||
      !std::isfinite(policy.iteration.rel) ||
      !std::isfinite(policy.abs_log_error) || policy.iteration.abs <= 0.0 ||
      policy.iteration.rel <= 0.0 || policy.abs_log_error <= 0.0) {
    return Result<Distribution>::failure(
        ErrorCode::InvalidTolerance,
        "probability tolerances must be positive and finite");
  }
  if (policy.iteration.max_iterations == 0U ||
      policy.iteration.max_evaluations == 0U) {
    return Result<Distribution>::failure(
        ErrorCode::InvalidDomain, "probability work limits must be positive");
  }
  if (terms.empty()) {
    return Result<Distribution>::failure(ErrorCode::InvalidExtent,
                                         "mixture requires at least one term");
  }
  if (terms.size() > 1024U || terms.size() > limits.elements ||
      terms.size() >
          limits.bytes / (sizeof(std::shared_ptr<const Distribution::State>) +
                          sizeof(Scalar)) ||
      terms.size() > limits.evaluations) {
    return Result<Distribution>::failure(
        ErrorCode::ResourceLimit,
        "mixture input exceeds its configured storage or work limit");
  }
  std::vector<std::shared_ptr<const Distribution::State>> components;
  std::vector<Scalar> weights;
  components.reserve(terms.size());
  weights.reserve(terms.size());
  Measure measure = Measure::real_density;
  bool foundMeasure = false;
  for (const WeightedLaw& term : terms) {
    if (!std::isfinite(term.weight) || term.weight < 0.0) {
      return Result<Distribution>::failure(
          ErrorCode::InvalidDomain,
          "mixture weights must be finite and nonnegative");
    }
    if (term.weight == 0.0) {
      continue;
    }
    auto state = share(term.law);
    if (!state) {
      return Result<Distribution>::failure(ErrorCode::InvalidDomain,
                                           "mixture contains a moved-from law");
    }
    if (foundMeasure && state->measure != measure) {
      return Result<Distribution>::failure(
          ErrorCode::InvalidDomain,
          "positive mixture components must use the same measure");
    }
    measure = state->measure;
    foundMeasure = true;
    components.push_back(std::move(state));
    weights.push_back(term.weight);
  }
  if (!foundMeasure) {
    return Result<Distribution>::failure(
        ErrorCode::InvalidDomain, "mixture requires positive total weight");
  }
  return DistributionFactory::composite(Family::mixture, std::move(components),
                                        std::move(weights), measure, limits);
}

Distribution::Distribution(std::shared_ptr<const State> state) noexcept
    : state_(std::move(state)) {}

Family Distribution::family() const noexcept {
  return state_ ? state_->family : Family::normal;
}

Measure Distribution::measure() const noexcept {
  return state_ ? state_->measure : Measure::real_density;
}

std::span<const Scalar> Distribution::parameters() const noexcept {
  return state_ ? std::span<const Scalar>{state_->parameters}
                : std::span<const Scalar>{};
}

FiniteAtomView Distribution::finiteAtoms() const noexcept {
  if (!state_ ||
      (state_->family != Family::empirical &&
       state_->family != Family::finite_convolution) ||
      state_->atoms.empty() || state_->atoms.size() != state_->masses.size()) {
    return {};
  }
  return {state_->atoms, state_->masses};
}

std::shared_ptr<const Distribution::State>
detail::DistributionEngine::share(const Distribution& law) noexcept {
  return law.state_;
}

Result<detail::DistributionSnapshot>
detail::DistributionEngine::snapshot(const Distribution& law, Limits limits) {
  if (!law.state_) {
    return Result<DistributionSnapshot>::failure(ErrorCode::InvalidDomain,
                                                 "distribution is moved from");
  }
  std::size_t elements = 0U;
  std::size_t bytes = 0U;
  const auto reserve = [&](std::size_t count, std::size_t width) {
    if (count > limits.elements - std::min(elements, limits.elements) ||
        (width != 0U &&
         count > std::numeric_limits<std::size_t>::max() / width)) {
      return false;
    }
    const std::size_t added = count * width;
    if (added > limits.bytes - std::min(bytes, limits.bytes))
      return false;
    elements += count;
    bytes += added;
    return true;
  };
  const auto copy = [&](const auto& self,
                        const std::shared_ptr<const Distribution::State>& state,
                        Index depth) -> Result<DistributionSnapshot> {
    if (!state || depth > 32U || !reserve(1U, sizeof(DistributionSnapshot)) ||
        !reserve(state->parameters.size(), sizeof(Scalar)) ||
        !reserve(state->weights.size(), sizeof(Scalar)) ||
        !reserve(state->atoms.size(), sizeof(Scalar)) ||
        !reserve(state->masses.size(), sizeof(Scalar)) ||
        !reserve(state->components.size(), sizeof(DistributionSnapshot))) {
      return Result<DistributionSnapshot>::failure(
          ErrorCode::ResourceLimit,
          "distribution snapshot exceeds its depth or storage budget");
    }
    DistributionSnapshot result;
    result.family = state->family;
    result.measure = state->measure;
    result.parameters = state->parameters;
    result.integer_parameter = state->integerParameter;
    result.weights = state->weights;
    result.atoms = state->atoms;
    result.masses = state->masses;
    result.components.reserve(state->components.size());
    for (const auto& child : state->components) {
      auto nested = self(self, child, depth + 1U);
      if (!nested)
        return nested;
      result.components.push_back(std::move(nested).value());
    }
    return Result<DistributionSnapshot>::success(std::move(result));
  };
  try {
    return copy(copy, law.state_, 1U);
  } catch (const std::bad_alloc&) {
    return Result<DistributionSnapshot>::failure(
        ErrorCode::ResourceLimit, "distribution snapshot allocation failed");
  } catch (const std::length_error&) {
    return Result<DistributionSnapshot>::failure(
        ErrorCode::ResourceLimit,
        "distribution snapshot extent is unsupported");
  }
}

// Composite support traversal follows the stored, explicitly depth-capped
// expression tree; this recursion is the representation's intended walk.
Result<std::vector<std::pair<Scalar, Scalar>>>
// NOLINTNEXTLINE(misc-no-recursion)
detail::DistributionEngine::finiteAtoms(const Distribution& law,
                                        Limits limits) {
  if (!law.state_) {
    return Result<std::vector<std::pair<Scalar, Scalar>>>::failure(
        ErrorCode::InvalidDomain, "distribution is moved from");
  }
  const auto& state = *law.state_;
  if (state.family == Family::empirical ||
      state.family == Family::finite_convolution) {
    if (state.atoms.size() > limits.elements ||
        state.atoms.size() > limits.bytes / (2U * sizeof(Scalar))) {
      return Result<std::vector<std::pair<Scalar, Scalar>>>::failure(
          ErrorCode::ResourceLimit, "finite support exceeds configured limits");
    }
    std::vector<std::pair<Scalar, Scalar>> result;
    result.reserve(state.atoms.size());
    for (Index index = 0U; index < state.atoms.size(); ++index) {
      result.emplace_back(state.atoms[index], state.masses[index]);
    }
    return Result<std::vector<std::pair<Scalar, Scalar>>>::success(
        std::move(result));
  }
  if (state.family == Family::bernoulli) {
    const Scalar p = state.parameters[0];
    std::vector<std::pair<Scalar, Scalar>> result;
    if (p < 1.0) {
      result.emplace_back(0.0, 1.0 - p);
    }
    if (p > 0.0) {
      result.emplace_back(1.0, p);
    }
    return Result<std::vector<std::pair<Scalar, Scalar>>>::success(
        std::move(result));
  }
  if (state.family == Family::binomial) {
    constexpr std::uint64_t supportCap = 1024U;
    const auto trials = state.integerParameter;
    if (trials >= supportCap || trials + 1U > limits.elements ||
        (trials + 1U) > limits.bytes / (2U * sizeof(Scalar))) {
      return Result<std::vector<std::pair<Scalar, Scalar>>>::failure(
          ErrorCode::ResourceLimit,
          "binomial finite support exceeds configured limits");
    }
    const Scalar p = state.parameters[1];
    std::vector<std::pair<Scalar, Scalar>> result;
    if (p == 0.0) {
      result.emplace_back(0.0, 1.0);
      return Result<std::vector<std::pair<Scalar, Scalar>>>::success(
          std::move(result));
    }
    if (p == 1.0) {
      result.emplace_back(static_cast<Scalar>(trials), 1.0);
      return Result<std::vector<std::pair<Scalar, Scalar>>>::success(
          std::move(result));
    }
    Scalar mass = std::pow(1.0 - p, static_cast<Scalar>(trials));
    for (std::uint64_t index = 0U; index <= trials; ++index) {
      if (mass == 0.0 || !std::isfinite(mass)) {
        return Result<std::vector<std::pair<Scalar, Scalar>>>::failure(
            ErrorCode::NonFiniteResult,
            "binomial support probability is not representable");
      }
      result.emplace_back(static_cast<Scalar>(index), mass);
      if (index < trials) {
        mass *= (static_cast<Scalar>(trials - index) /
                 static_cast<Scalar>(index + 1U)) *
                (p / (1.0 - p));
      }
    }
    return Result<std::vector<std::pair<Scalar, Scalar>>>::success(
        std::move(result));
  }
  if (state.family == Family::truncated &&
      state.measure == Measure::atom_mass) {
    const Distribution base(state.components.front());
    auto support = finiteAtoms(base, limits);
    if (!support) {
      return Result<std::vector<std::pair<Scalar, Scalar>>>::failure(
          support.status().code(), std::string(support.status().message()));
    }
    const Scalar lower = state.parameters[0];
    const Scalar upper = state.parameters[1];
    const bool includeLower = (state.integerParameter & 1U) != 0U;
    const bool includeUpper = (state.integerParameter & 2U) != 0U;
    std::vector<std::pair<Scalar, Scalar>> result;
    for (const auto& [atom, mass] : support.value()) {
      const bool aboveLower = atom > lower || (includeLower && atom == lower);
      const bool belowUpper = atom < upper || (includeUpper && atom == upper);
      if (aboveLower && belowUpper && mass > 0.0) {
        result.emplace_back(atom, static_cast<Scalar>(std::exp(
                                      std::log(mass) - state.parameters[2])));
      }
    }
    if (result.empty()) {
      return Result<std::vector<std::pair<Scalar, Scalar>>>::failure(
          ErrorCode::InvalidDomain, "truncated atom law has empty support");
    }
    return Result<std::vector<std::pair<Scalar, Scalar>>>::success(
        std::move(result));
  }
  if (state.family == Family::mixture) {
    std::vector<std::pair<Scalar, Scalar>> result;
    for (Index index = 0U; index < state.components.size(); ++index) {
      if (state.weights[index] == 0.0) {
        continue;
      }
      const Distribution component(state.components[index]);
      auto atoms = finiteAtoms(component, limits);
      if (!atoms) {
        return Result<std::vector<std::pair<Scalar, Scalar>>>::failure(
            atoms.status().code(), std::string(atoms.status().message()));
      }
      for (const auto& [atom, mass] : atoms.value()) {
        result.emplace_back(atom, state.weights[index] * mass);
        if (result.size() > limits.elements || result.size() > 1024U) {
          return Result<std::vector<std::pair<Scalar, Scalar>>>::failure(
              ErrorCode::ResourceLimit,
              "mixture support exceeds configured limits");
        }
      }
    }
    std::sort(result.begin(), result.end(),
              [](const auto& left, const auto& right) {
                return left.first < right.first;
              });
    std::vector<std::pair<Scalar, Scalar>> merged;
    for (const auto& value : result) {
      if (!merged.empty() && merged.back().first == value.first) {
        merged.back().second += value.second;
      } else {
        merged.push_back(value);
      }
    }
    return Result<std::vector<std::pair<Scalar, Scalar>>>::success(
        std::move(merged));
  }
  return Result<std::vector<std::pair<Scalar, Scalar>>>::failure(
      ErrorCode::InvalidDomain,
      "finite convolution requires laws with finite discrete support");
}

Result<Distribution>
detail::DistributionEngine::convolution(std::span<const Distribution> terms,
                                        ProbabilityPolicy policy,
                                        Limits limits) {
  if (!std::isfinite(policy.iteration.abs) ||
      !std::isfinite(policy.iteration.rel) ||
      !std::isfinite(policy.abs_log_error) || policy.iteration.abs <= 0.0 ||
      policy.iteration.rel <= 0.0 || policy.abs_log_error <= 0.0) {
    return Result<Distribution>::failure(
        ErrorCode::InvalidTolerance,
        "convolution policy tolerances are invalid");
  }
  if (policy.iteration.max_iterations == 0U ||
      policy.iteration.max_evaluations == 0U || limits.evaluations == 0U) {
    return Result<Distribution>::failure(
        ErrorCode::InvalidDomain, "convolution work limits must be positive");
  }
  if (terms.size() > 32U) {
    return Result<Distribution>::failure(ErrorCode::ResourceLimit,
                                         "convolution depth exceeds 32 terms");
  }
  if (terms.empty()) {
    return DistributionFactory::empirical({0.0}, {1.0}, limits,
                                          Family::finite_convolution);
  }
  if (terms.size() == 1U) {
    if (!share(terms.front())) {
      return Result<Distribution>::failure(
          ErrorCode::InvalidDomain, "convolution contains a moved-from law");
    }
    return Result<Distribution>::success(terms.front());
  }
  bool allNormal = true;
  bool allPoisson = true;
  bool allBinomial = true;
  bool allNegativeBinomial = true;
  bool allDiscrete = true;
  Index normalCount = 0U;
  Index poissonCount = 0U;
  Index gammaCount = 0U;
  long double gammaShape = 0.0L;
  Scalar commonGammaScale = 0.0;
  bool commonGammaScaleSet = false;
  Scalar commonCountProbability = 0.0;
  bool commonCountProbabilitySet = false;
  std::uint64_t binomialTrials = 0U;
  long double poissonRate = 0.0L;
  long double negativeBinomialSuccesses = 0.0L;
  std::vector<std::shared_ptr<const Distribution::State>> components;
  components.reserve(terms.size());
  long double normalMean = 0.0L;
  long double normalVariance = 0.0L;
  for (const Distribution& term : terms) {
    const auto state = share(term);
    if (!state) {
      return Result<Distribution>::failure(
          ErrorCode::InvalidDomain, "convolution contains a moved-from law");
    }
    components.push_back(state);
    allNormal = allNormal && state->family == Family::normal;
    allPoisson = allPoisson && state->family == Family::poisson;
    allBinomial = allBinomial && state->family == Family::binomial;
    allNegativeBinomial =
        allNegativeBinomial && state->family == Family::negative_binomial;
    if (state->family == Family::normal) {
      ++normalCount;
      normalMean += state->parameters[0];
      normalVariance +=
          static_cast<long double>(state->parameters[1]) * state->parameters[1];
    }
    poissonCount += state->family == Family::poisson ? 1U : 0U;
    if (state->family == Family::poisson) {
      poissonRate += state->parameters[0];
    }
    if (state->family == Family::binomial) {
      const Scalar probability = state->parameters[1];
      if (!commonCountProbabilitySet) {
        commonCountProbability = probability;
        commonCountProbabilitySet = true;
      }
      allBinomial = allBinomial && probability == commonCountProbability;
      if (state->integerParameter >
          std::numeric_limits<std::uint64_t>::max() - binomialTrials) {
        return Result<Distribution>::failure(
            ErrorCode::SizeOverflow,
            "binomial convolution trial count overflows");
      }
      binomialTrials += state->integerParameter;
    }
    if (state->family == Family::negative_binomial) {
      const Scalar probability = state->parameters[1];
      if (!commonCountProbabilitySet) {
        commonCountProbability = probability;
        commonCountProbabilitySet = true;
      }
      allNegativeBinomial =
          allNegativeBinomial && probability == commonCountProbability;
      negativeBinomialSuccesses += state->parameters[0];
    }
    if (state->family == Family::gamma || state->family == Family::chi_square) {
      const Scalar scale =
          state->family == Family::gamma ? state->parameters[1] : 2.0;
      const Scalar shape = state->family == Family::gamma
                               ? state->parameters[0]
                               : state->parameters[0] / 2.0;
      if (!commonGammaScaleSet) {
        commonGammaScale = scale;
        commonGammaScaleSet = true;
      }
      if (scale == commonGammaScale) {
        ++gammaCount;
        gammaShape += shape;
      }
    }
    allDiscrete = allDiscrete && (state->measure == Measure::integer_mass ||
                                  state->measure == Measure::atom_mass);
  }
  if (allNormal) {
    if (!std::isfinite(normalMean) || !std::isfinite(normalVariance) ||
        std::abs(normalMean) > std::numeric_limits<Scalar>::max() ||
        normalVariance >
            static_cast<long double>(std::numeric_limits<Scalar>::max()) *
                std::numeric_limits<Scalar>::max()) {
      return Result<Distribution>::failure(
          ErrorCode::NonFiniteResult, "normal convolution parameters overflow");
    }
    return normal(static_cast<Scalar>(normalMean),
                  static_cast<Scalar>(std::sqrt(normalVariance)));
  }
  if (allPoisson) {
    if (!std::isfinite(poissonRate) ||
        poissonRate > std::numeric_limits<Scalar>::max()) {
      return Result<Distribution>::failure(
          ErrorCode::NonFiniteResult, "Poisson convolution rate overflows");
    }
    return poisson(static_cast<Scalar>(poissonRate));
  }
  if (allBinomial) {
    return binomial(binomialTrials, commonCountProbability);
  }
  if (allNegativeBinomial) {
    if (!std::isfinite(negativeBinomialSuccesses) ||
        negativeBinomialSuccesses > std::numeric_limits<Scalar>::max()) {
      return Result<Distribution>::failure(
          ErrorCode::NonFiniteResult,
          "negative-binomial convolution shape overflows");
    }
    return negativeBinomial(static_cast<Scalar>(negativeBinomialSuccesses),
                            commonCountProbability);
  }
  if (gammaCount == terms.size() && gammaCount > 1U) {
    if (!std::isfinite(gammaShape) ||
        gammaShape > std::numeric_limits<Scalar>::max()) {
      return Result<Distribution>::failure(
          ErrorCode::NonFiniteResult,
          "common-scale gamma convolution shape overflows");
    }
    return gamma(static_cast<Scalar>(gammaShape), commonGammaScale);
  }
  const bool poissonWithGamma =
      terms.size() == 2U && poissonCount == 1U && gammaCount == 1U &&
      std::all_of(terms.begin(), terms.end(), [](const Distribution& term) {
        return term.state_->family == Family::gamma ||
               term.state_->family == Family::poisson;
      });
  if (poissonWithGamma) {
    return DistributionFactory::convolution(std::move(components),
                                            Measure::real_density, limits);
  }
  const bool betaOneWithGamma =
      terms.size() == 2U &&
      std::all_of(terms.begin(), terms.end(),
                  [](const Distribution& term) {
                    return term.state_->family == Family::beta ||
                           term.state_->family == Family::gamma;
                  }) &&
      std::any_of(terms.begin(), terms.end(),
                  [](const Distribution& term) {
                    return term.state_->family == Family::beta &&
                           term.state_->parameters[0] == 1.0;
                  }) &&
      std::any_of(terms.begin(), terms.end(), [](const Distribution& term) {
        return term.state_->family == Family::gamma &&
               term.state_->parameters[1] == 1.0;
      });
  if (betaOneWithGamma) {
    return DistributionFactory::convolution(std::move(components),
                                            Measure::real_density, limits);
  }
  const bool normalWithPoisson =
      normalCount > 0U && poissonCount == 1U &&
      terms.size() == normalCount + poissonCount &&
      std::all_of(terms.begin(), terms.end(), [](const Distribution& term) {
        return term.state_->family == Family::normal ||
               term.state_->family == Family::poisson;
      });
  if (normalWithPoisson) {
    if (!std::isfinite(normalMean) || !std::isfinite(normalVariance) ||
        std::abs(normalMean) > std::numeric_limits<Scalar>::max() ||
        normalVariance >
            static_cast<long double>(std::numeric_limits<Scalar>::max()) *
                std::numeric_limits<Scalar>::max()) {
      return Result<Distribution>::failure(
          ErrorCode::NonFiniteResult,
          "normal and Poisson convolution parameters overflow");
    }
    auto combinedNormal =
        normal(static_cast<Scalar>(normalMean),
               static_cast<Scalar>(std::sqrt(normalVariance)));
    if (!combinedNormal) {
      return Result<Distribution>::failure(
          combinedNormal.status().code(),
          std::string(combinedNormal.status().message()));
    }
    const auto poissonState = std::find_if(
        components.begin(), components.end(),
        [](const auto& state) { return state->family == Family::poisson; });
    std::vector<std::shared_ptr<const Distribution::State>> expression{
        share(combinedNormal.value()), *poissonState};
    return DistributionFactory::convolution(std::move(expression),
                                            Measure::real_density, limits);
  }
  if (allDiscrete) {
    std::map<Scalar, Scalar> aggregate{{0.0, 1.0}};
    std::uint64_t operations = 0U;
    const Limits supportLimits{
        std::min<std::size_t>(limits.elements, 1024U),
        std::min<std::size_t>(limits.bytes, std::size_t{16U} * 1024U * 1024U),
        limits.evaluations};
    for (const Distribution& term : terms) {
      auto support = finiteAtoms(term, supportLimits);
      if (!support) {
        return Result<Distribution>::failure(
            support.status().code(), std::string(support.status().message()));
      }
      if (support.value().empty()) {
        return Result<Distribution>::failure(ErrorCode::InvalidDomain,
                                             "convolution support is empty");
      }
      if (aggregate.size() >
          limits.elements / std::max<std::size_t>(support.value().size(), 1U)) {
        return Result<Distribution>::failure(
            ErrorCode::ResourceLimit, "convolution output exceeds limits");
      }
      std::map<Scalar, Scalar> next;
      for (const auto& [lhs, lhsMass] : aggregate) {
        for (const auto& [rhs, rhsMass] : support.value()) {
          ++operations;
          if (operations > limits.evaluations ||
              operations > policy.iteration.max_evaluations) {
            return Result<Distribution>::failure(
                ErrorCode::ResourceLimit,
                "convolution evaluation cap exceeded");
          }
          const Scalar atom = lhs + rhs;
          const Scalar mass = lhsMass * rhsMass;
          if (!std::isfinite(atom) ||
              (mass == 0.0 && lhsMass > 0.0 && rhsMass > 0.0)) {
            return Result<Distribution>::failure(
                ErrorCode::NonFiniteResult,
                "convolution atom or positive mass is not representable");
          }
          next[atom] += mass;
        }
      }
      if (next.size() > limits.elements || next.size() > 1024U ||
          next.size() > limits.bytes / (2U * sizeof(Scalar))) {
        return Result<Distribution>::failure(
            ErrorCode::ResourceLimit, "convolution support exceeds limits");
      }
      aggregate = std::move(next);
    }
    std::vector<Scalar> atoms;
    std::vector<Scalar> masses;
    atoms.reserve(aggregate.size());
    masses.reserve(aggregate.size());
    for (const auto& [atom, mass] : aggregate) {
      atoms.push_back(atom);
      masses.push_back(mass);
    }
    return DistributionFactory::empirical(std::move(atoms), std::move(masses),
                                          limits, Family::finite_convolution);
  }
  const bool normalWithDiscrete =
      normalCount > 0U && normalCount < terms.size() &&
      std::all_of(
          terms.begin(), terms.end(),
          [](const Distribution& term) {
            return term.state_->family == Family::normal ||
                   term.state_->measure == Measure::integer_mass ||
                   term.state_->measure == Measure::atom_mass;
          });
  const bool twoUniforms = terms.size() == 2U &&
                           terms[0].state_->family == Family::uniform &&
                           terms[1].state_->family == Family::uniform;
  const bool normalWithUniform =
      terms.size() == 2U && ((terms[0].state_->family == Family::normal &&
                              terms[1].state_->family == Family::uniform) ||
                             (terms[1].state_->family == Family::normal &&
                              terms[0].state_->family == Family::uniform));
  const bool poissonWithTruncatedBase =
      terms.size() == 2U && ((terms[0].state_->family == Family::poisson &&
                              terms[1].state_->family == Family::truncated) ||
                             (terms[1].state_->family == Family::poisson &&
                              terms[0].state_->family == Family::truncated));
  const bool gammaWithTruncatedBase =
      terms.size() == 2U && ((terms[0].state_->family == Family::gamma &&
                              terms[1].state_->family == Family::truncated) ||
                             (terms[1].state_->family == Family::gamma &&
                              terms[0].state_->family == Family::truncated));
  if (normalWithDiscrete) {
    if (!std::isfinite(normalMean) || !std::isfinite(normalVariance) ||
        std::abs(normalMean) > std::numeric_limits<Scalar>::max() ||
        normalVariance >
            static_cast<long double>(std::numeric_limits<Scalar>::max()) *
                std::numeric_limits<Scalar>::max()) {
      return Result<Distribution>::failure(
          ErrorCode::NonFiniteResult,
          "normal and discrete convolution parameters overflow");
    }
    std::map<Scalar, Scalar> aggregate{{0.0, 1.0}};
    std::uint64_t operations = 0U;
    const Limits supportLimits{
        std::min<std::size_t>(limits.elements, 1024U),
        std::min<std::size_t>(limits.bytes, std::size_t{16U} * 1024U * 1024U),
        limits.evaluations};
    for (const Distribution& term : terms) {
      if (term.state_->family == Family::normal) {
        continue;
      }
      auto support = finiteAtoms(term, supportLimits);
      if (!support) {
        return Result<Distribution>::failure(
            support.status().code(), std::string(support.status().message()));
      }
      if (aggregate.size() >
          limits.elements / std::max<std::size_t>(support.value().size(), 1U)) {
        return Result<Distribution>::failure(
            ErrorCode::ResourceLimit,
            "normal and discrete convolution support exceeds limits");
      }
      std::map<Scalar, Scalar> next;
      for (const auto& [lhs, lhsMass] : aggregate) {
        for (const auto& [rhs, rhsMass] : support.value()) {
          if (operations >= limits.evaluations ||
              operations >= policy.iteration.max_evaluations) {
            return Result<Distribution>::failure(
                ErrorCode::ResourceLimit,
                "normal and discrete convolution evaluation cap exceeded");
          }
          ++operations;
          const Scalar atom = lhs + rhs;
          const Scalar mass = lhsMass * rhsMass;
          if (!std::isfinite(atom) ||
              (mass == 0.0 && lhsMass > 0.0 && rhsMass > 0.0)) {
            return Result<Distribution>::failure(
                ErrorCode::NonFiniteResult,
                "normal and discrete convolution support is not representable");
          }
          next[atom] += mass;
        }
      }
      if (next.size() > supportLimits.elements ||
          next.size() > supportLimits.bytes / (2U * sizeof(Scalar))) {
        return Result<Distribution>::failure(
            ErrorCode::ResourceLimit,
            "normal and discrete convolution support exceeds limits");
      }
      aggregate = std::move(next);
    }
    std::vector<Scalar> atoms;
    std::vector<Scalar> masses;
    atoms.reserve(aggregate.size());
    masses.reserve(aggregate.size());
    for (const auto& [atom, mass] : aggregate) {
      atoms.push_back(atom);
      masses.push_back(mass);
    }
    auto discrete = DistributionFactory::empirical(
        std::move(atoms), std::move(masses), supportLimits);
    if (!discrete) {
      return Result<Distribution>::failure(
          discrete.status().code(), std::string(discrete.status().message()));
    }
    auto combinedNormal =
        normal(static_cast<Scalar>(normalMean),
               static_cast<Scalar>(std::sqrt(normalVariance)));
    if (!combinedNormal) {
      return Result<Distribution>::failure(
          combinedNormal.status().code(),
          std::string(combinedNormal.status().message()));
    }
    std::vector<std::shared_ptr<const Distribution::State>> expression{
        share(combinedNormal.value()), share(discrete.value())};
    return DistributionFactory::convolution(std::move(expression),
                                            Measure::real_density, limits);
  }
  if (twoUniforms || normalWithUniform || poissonWithTruncatedBase ||
      gammaWithTruncatedBase) {
    const Measure measure = Measure::real_density;
    return DistributionFactory::convolution(std::move(components), measure,
                                            limits);
  }
  return Result<Distribution>::failure(
      ErrorCode::InvalidDomain,
      "this convolution family combination has no qualified numerical path");
}

Rng::Rng(std::uint64_t seed) noexcept {
  auto splitMix = [&seed]() {
    std::uint64_t value = (seed += 0x9e3779b97f4a7c15ULL);
    value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
  };
  for (auto& item : state_) {
    item = splitMix();
  }
}

std::uint64_t Rng::next() noexcept {
  const std::uint64_t result = std::rotl(state_[1] * 5U, 7) * 9U;
  const std::uint64_t temporary = state_[1] << 17U;
  state_[2] ^= state_[0];
  state_[3] ^= state_[1];
  state_[1] ^= state_[2];
  state_[0] ^= state_[3];
  state_[2] ^= temporary;
  state_[3] = std::rotl(state_[3], 45);
  return result;
}

namespace {

constexpr Scalar logTwo = std::numbers::ln2;

Result<Distribution> make(Family family, Measure measure,
                          std::initializer_list<Scalar> parameters,
                          std::uint64_t integer = 0U) {
  for (const Scalar value : parameters) {
    if (!std::isfinite(value)) {
      return Result<Distribution>::failure(
          ErrorCode::NonFiniteInput, "distribution parameters must be finite");
    }
  }
  return detail::DistributionFactory::create(family, measure, parameters,
                                             integer);
}

bool inClosedUnitInterval(Scalar value) { return value >= 0.0 && value <= 1.0; }

Scalar logOneMinusExp(Scalar value) {
  return value < -logTwo ? std::log1p(-std::exp(value))
                         : std::log(-std::expm1(value));
}

LogTailPair lowerTail(Scalar logLower) {
  return {logLower, logOneMinusExp(logLower)};
}

LogTailPair upperTail(Scalar logUpper) {
  return {logOneMinusExp(logUpper), logUpper};
}

Scalar softplus(Scalar value) {
  return std::max(value, 0.0) + std::log1p(std::exp(-std::abs(value)));
}

Scalar logBetaValue(Scalar a, Scalar b, ProbabilityPolicy policy) {
  const auto ga = logGamma(a, policy);
  const auto gb = logGamma(b, policy);
  const auto gab = logGamma(a + b, policy);
  if (!ga || !gb || !gab) {
    return std::numeric_limits<Scalar>::quiet_NaN();
  }
  return ga.value() + gb.value() - gab.value();
}

Result<LogTailPair> normalLogTails(Scalar z, ProbabilityPolicy policy) {
  auto pair = incompleteGamma(0.5, 0.5 * z * z, policy);
  if (!pair) {
    return Result<LogTailPair>::failure(pair.status().code(),
                                        std::string(pair.status().message()));
  }
  constexpr Scalar logHalf = -std::numbers::ln2;
  const Scalar smaller = pair.value().log_survival + logHalf;
  return Result<LogTailPair>::success(
      z < 0.0 ? LogTailPair{smaller, logOneMinusExp(smaller)}
              : LogTailPair{logOneMinusExp(smaller), smaller});
}

Scalar logDifference(Scalar logLarge, Scalar logSmall) {
  if (logSmall == -std::numeric_limits<Scalar>::infinity()) {
    return logLarge;
  }
  if (!(logLarge > logSmall)) {
    return -std::numeric_limits<Scalar>::infinity();
  }
  return logLarge + logOneMinusExp(logSmall - logLarge);
}

Scalar logAddExp(Scalar left, Scalar right) {
  if (left == -std::numeric_limits<Scalar>::infinity()) {
    return right;
  }
  if (right == -std::numeric_limits<Scalar>::infinity()) {
    return left;
  }
  const Scalar maximum = std::max(left, right);
  return maximum + std::log1p(std::exp(std::min(left, right) - maximum));
}

ProbabilityPolicy defaultPolicy() {
  return {{1.0e-13, 1.0e-13, 1024U, 100000U, std::nullopt}, 100000U, 2.0e-9};
}

Result<Scalar> owenT(Scalar h, Scalar a, const ProbabilityPolicy& policy) {
  if (a == 0.0) {
    return Result<Scalar>::success(0.0);
  }
  const Scalar upper = std::atan(std::abs(a));
  const auto integrand = [h](Scalar theta) {
    const Scalar cosine = std::cos(theta);
    const Scalar ratio = h / cosine;
    return std::exp(-0.5 * ratio * ratio) / (2.0 * std::numbers::pi);
  };
  const Scalar left = 0.0;
  const Scalar right = upper;
  const Scalar middle = 0.5 * (left + right);
  const Scalar fLeft = integrand(left);
  const Scalar fMiddle = integrand(middle);
  const Scalar fRight = integrand(right);
  Scalar estimate = (right - left) * (fLeft + 4.0 * fMiddle + fRight) / 6.0;
  struct Interval {
    Scalar left;
    Scalar right;
    Scalar fLeft;
    Scalar fMiddle;
    Scalar fRight;
    Scalar estimate;
    Index depth;
  };
  std::vector<Interval> stack{
      {left, right, fLeft, fMiddle, fRight, estimate, 0U}};
  Scalar total = 0.0;
  std::uint64_t evaluations = 3U;
  const Scalar tolerance =
      std::min(policy.iteration.abs, policy.abs_log_error * 0.25);
  while (!stack.empty()) {
    const Interval interval = stack.back();
    stack.pop_back();
    const Scalar mid = 0.5 * (interval.left + interval.right);
    const Scalar lm = 0.5 * (interval.left + mid);
    const Scalar rm = 0.5 * (mid + interval.right);
    if (evaluations + 2U > policy.iteration.max_evaluations) {
      return Result<Scalar>::failure(
          ErrorCode::NoConvergence,
          "Owen-T quadrature reached its evaluation limit");
    }
    const Scalar flm = integrand(lm);
    const Scalar frm = integrand(rm);
    evaluations += 2U;
    const Scalar leftEstimate =
        (mid - interval.left) *
        (interval.fLeft + 4.0 * flm + interval.fMiddle) / 6.0;
    const Scalar rightEstimate =
        (interval.right - mid) *
        (interval.fMiddle + 4.0 * frm + interval.fRight) / 6.0;
    const Scalar refined = leftEstimate + rightEstimate;
    const Scalar error = std::abs(refined - interval.estimate) / 15.0;
    if (error <=
        tolerance * (interval.right - interval.left) +
            8.0 * std::numeric_limits<Scalar>::epsilon() * std::abs(refined)) {
      total += refined + (refined - interval.estimate) / 15.0;
      estimate = 0.0;
      continue;
    }
    if (interval.depth >= policy.iteration.max_iterations ||
        interval.depth >= 64U) {
      return Result<Scalar>::failure(
          ErrorCode::NoConvergence,
          "Owen-T quadrature reached its depth limit");
    }
    stack.push_back({mid, interval.right, interval.fMiddle, frm,
                     interval.fRight, rightEstimate, interval.depth + 1U});
    stack.push_back({interval.left, mid, interval.fLeft, flm, interval.fMiddle,
                     leftEstimate, interval.depth + 1U});
  }
  static_cast<void>(estimate);
  return Result<Scalar>::success(std::copysign(total, a));
}

Result<TailValues> finiteTailValues(Scalar logDensity, LogTailPair tails,
                                    bool onSupport) {
  const Scalar negativeInfinity = -std::numeric_limits<Scalar>::infinity();
  const Scalar logMaximum = std::log(std::numeric_limits<Scalar>::max());
  if (std::isnan(logDensity) || std::isnan(tails.log_cdf) ||
      std::isnan(tails.log_survival)) {
    return Result<TailValues>::failure(ErrorCode::NonFiniteResult,
                                       "probability evaluation produced NaN");
  }
  if (logDensity > logMaximum) {
    return Result<TailValues>::failure(
        ErrorCode::NonFiniteResult,
        "finite probability density exceeds the scalar range");
  }
  const Scalar density =
      logDensity == negativeInfinity ? 0.0 : std::exp(logDensity);
  return Result<TailValues>::success(
      {density, logDensity, std::exp(tails.log_cdf),
       std::exp(tails.log_survival), tails, onSupport});
}

} // namespace

Result<Distribution> normal(Scalar meanValue, Scalar sd) {
  if (!std::isfinite(meanValue) || !std::isfinite(sd)) {
    return Result<Distribution>::failure(ErrorCode::NonFiniteInput,
                                         "normal parameters must be finite");
  }
  if (sd <= 0.0) {
    return Result<Distribution>::failure(
        ErrorCode::InvalidDomain, "normal standard deviation must be positive");
  }
  return make(Family::normal, Measure::real_density, {meanValue, sd});
}

Result<Distribution> studentT(Scalar nu, Scalar location, Scalar scale) {
  if (!std::isfinite(nu) || !std::isfinite(location) || !std::isfinite(scale)) {
    return Result<Distribution>::failure(ErrorCode::NonFiniteInput,
                                         "Student-t parameters must be finite");
  }
  if (nu <= 0.0 || scale <= 0.0) {
    return Result<Distribution>::failure(
        ErrorCode::InvalidDomain,
        "Student-t degrees of freedom and scale must be positive");
  }
  return make(Family::student_t, Measure::real_density, {nu, location, scale});
}

Result<Distribution> laplace(Scalar location, Scalar scale) {
  if (!std::isfinite(location) || !std::isfinite(scale)) {
    return Result<Distribution>::failure(ErrorCode::NonFiniteInput,
                                         "Laplace parameters must be finite");
  }
  if (scale <= 0.0) {
    return Result<Distribution>::failure(ErrorCode::InvalidDomain,
                                         "Laplace scale must be positive");
  }
  return make(Family::laplace, Measure::real_density, {location, scale});
}

Result<Distribution> logistic(Scalar location, Scalar scale) {
  if (!std::isfinite(location) || !std::isfinite(scale)) {
    return Result<Distribution>::failure(ErrorCode::NonFiniteInput,
                                         "logistic parameters must be finite");
  }
  if (scale <= 0.0) {
    return Result<Distribution>::failure(ErrorCode::InvalidDomain,
                                         "logistic scale must be positive");
  }
  return make(Family::logistic, Measure::real_density, {location, scale});
}

Result<Distribution> lognormal(Scalar logMean, Scalar logSd) {
  if (!std::isfinite(logMean) || !std::isfinite(logSd)) {
    return Result<Distribution>::failure(ErrorCode::NonFiniteInput,
                                         "lognormal parameters must be finite");
  }
  if (logSd <= 0.0) {
    return Result<Distribution>::failure(
        ErrorCode::InvalidDomain,
        "lognormal log standard deviation must be positive");
  }
  return make(Family::lognormal, Measure::real_density, {logMean, logSd});
}

Result<Distribution> gamma(Scalar shape, Scalar scale) {
  if (!std::isfinite(shape) || !std::isfinite(scale)) {
    return Result<Distribution>::failure(ErrorCode::NonFiniteInput,
                                         "gamma parameters must be finite");
  }
  if (shape <= 0.0 || scale <= 0.0) {
    return Result<Distribution>::failure(
        ErrorCode::InvalidDomain, "gamma shape and scale must be positive");
  }
  return make(Family::gamma, Measure::real_density, {shape, scale});
}

Result<Distribution> beta(Scalar shapeA, Scalar shapeB) {
  if (!std::isfinite(shapeA) || !std::isfinite(shapeB)) {
    return Result<Distribution>::failure(ErrorCode::NonFiniteInput,
                                         "beta parameters must be finite");
  }
  if (shapeA <= 0.0 || shapeB <= 0.0) {
    return Result<Distribution>::failure(ErrorCode::InvalidDomain,
                                         "beta shapes must be positive");
  }
  return make(Family::beta, Measure::real_density, {shapeA, shapeB});
}

Result<Distribution> weibull(Scalar shape, Scalar scale) {
  if (!std::isfinite(shape) || !std::isfinite(scale)) {
    return Result<Distribution>::failure(ErrorCode::NonFiniteInput,
                                         "Weibull parameters must be finite");
  }
  if (shape <= 0.0 || scale <= 0.0) {
    return Result<Distribution>::failure(
        ErrorCode::InvalidDomain, "Weibull shape and scale must be positive");
  }
  return make(Family::weibull, Measure::real_density, {shape, scale});
}

Result<Distribution> skewNormal(Scalar location, Scalar scale, Scalar shape) {
  if (!std::isfinite(location) || !std::isfinite(scale) ||
      !std::isfinite(shape)) {
    return Result<Distribution>::failure(
        ErrorCode::NonFiniteInput, "skew-normal parameters must be finite");
  }
  if (scale <= 0.0) {
    return Result<Distribution>::failure(ErrorCode::InvalidDomain,
                                         "skew-normal scale must be positive");
  }
  return make(Family::skew_normal, Measure::real_density,
              {location, scale, shape});
}

Result<Distribution> uniform(Scalar lower, Scalar upper) {
  if (!std::isfinite(lower) || !std::isfinite(upper)) {
    return Result<Distribution>::failure(ErrorCode::NonFiniteInput,
                                         "uniform bounds must be finite");
  }
  if (!(lower < upper)) {
    return Result<Distribution>::failure(ErrorCode::InvalidDomain,
                                         "uniform bounds must be ordered");
  }
  return make(Family::uniform, Measure::real_density, {lower, upper});
}

Result<Distribution> triangular(Scalar lower, Scalar mode, Scalar upper) {
  if (!std::isfinite(lower) || !std::isfinite(mode) || !std::isfinite(upper)) {
    return Result<Distribution>::failure(
        ErrorCode::NonFiniteInput, "triangular parameters must be finite");
  }
  if (!(lower < upper) || mode < lower || mode > upper) {
    return Result<Distribution>::failure(
        ErrorCode::InvalidDomain, "triangular bounds and mode are invalid");
  }
  return make(Family::triangular, Measure::real_density, {lower, mode, upper});
}

Result<Distribution> truncatedNormal(Scalar meanValue, Scalar sd, Scalar lower,
                                     Scalar upper) {
  if (!std::isfinite(meanValue) || !std::isfinite(sd) || std::isnan(lower) ||
      std::isnan(upper)) {
    return Result<Distribution>::failure(
        ErrorCode::NonFiniteInput, "truncated-normal parameters are invalid");
  }
  if (sd <= 0.0 || !(lower < upper)) {
    return Result<Distribution>::failure(
        ErrorCode::InvalidDomain,
        "truncated-normal scale and interval are invalid");
  }
  const ProbabilityPolicy policy = defaultPolicy();
  const Scalar lowerZ = (lower - meanValue) / sd;
  const Scalar upperZ = (upper - meanValue) / sd;
  auto lowerPair = normalLogTails(lowerZ, policy);
  auto upperPair = normalLogTails(upperZ, policy);
  if (!lowerPair || !upperPair) {
    const Status& status = !lowerPair ? lowerPair.status() : upperPair.status();
    return Result<Distribution>::failure(status.code(),
                                         std::string(status.message()));
  }
  Scalar logNormalizer = -std::numeric_limits<Scalar>::infinity();
  if (lowerZ >= 0.0) {
    logNormalizer = logDifference(lowerPair.value().log_survival,
                                  upperPair.value().log_survival);
  } else if (upperZ <= 0.0) {
    logNormalizer =
        logDifference(upperPair.value().log_cdf, lowerPair.value().log_cdf);
  } else {
    const Scalar excluded = std::exp(lowerPair.value().log_cdf) +
                            std::exp(upperPair.value().log_survival);
    logNormalizer = std::log1p(-excluded);
  }
  if (!std::isfinite(logNormalizer)) {
    return Result<Distribution>::failure(
        ErrorCode::InvalidDomain,
        "truncated-normal interval has no representable probability");
  }
  return detail::DistributionFactory::create(Family::truncated_normal,
                                             Measure::real_density,
                                             {meanValue, sd, lower, upper});
}

Result<Distribution> bernoulli(Scalar success) {
  if (!std::isfinite(success)) {
    return Result<Distribution>::failure(
        ErrorCode::NonFiniteInput, "Bernoulli probability must be finite");
  }
  if (!inClosedUnitInterval(success)) {
    return Result<Distribution>::failure(
        ErrorCode::InvalidDomain, "Bernoulli probability must lie in [0,1]");
  }
  return make(Family::bernoulli, Measure::integer_mass, {success});
}

Result<Distribution> binomial(std::uint64_t trials, Scalar success) {
  if (!std::isfinite(success)) {
    return Result<Distribution>::failure(ErrorCode::NonFiniteInput,
                                         "binomial probability must be finite");
  }
  if (!inClosedUnitInterval(success)) {
    return Result<Distribution>::failure(
        ErrorCode::InvalidDomain, "binomial probability must lie in [0,1]");
  }
  return make(Family::binomial, Measure::integer_mass,
              {static_cast<Scalar>(trials), success}, trials);
}

Result<Distribution> poisson(Scalar rate) {
  if (!std::isfinite(rate)) {
    return Result<Distribution>::failure(ErrorCode::NonFiniteInput,
                                         "Poisson rate must be finite");
  }
  if (rate < 0.0) {
    return Result<Distribution>::failure(ErrorCode::InvalidDomain,
                                         "Poisson rate must be non-negative");
  }
  return make(Family::poisson, Measure::integer_mass, {rate});
}

Result<Distribution> negativeBinomial(Scalar successes,
                                      Scalar successProbability) {
  if (!std::isfinite(successes) || !std::isfinite(successProbability)) {
    return Result<Distribution>::failure(
        ErrorCode::NonFiniteInput,
        "negative-binomial parameters must be finite");
  }
  if (successes <= 0.0 || successProbability <= 0.0 ||
      successProbability > 1.0) {
    return Result<Distribution>::failure(ErrorCode::InvalidDomain,
                                         "negative-binomial successes must be "
                                         "positive and probability in (0,1]");
  }
  return make(Family::negative_binomial, Measure::integer_mass,
              {successes, successProbability});
}

Result<Distribution> chiSquare(Scalar degreesOfFreedom) {
  if (!std::isfinite(degreesOfFreedom)) {
    return Result<Distribution>::failure(
        ErrorCode::NonFiniteInput,
        "chi-square degrees of freedom must be finite");
  }
  if (degreesOfFreedom <= 0.0) {
    return Result<Distribution>::failure(
        ErrorCode::InvalidDomain,
        "chi-square degrees of freedom must be positive");
  }
  return make(Family::chi_square, Measure::real_density, {degreesOfFreedom});
}

Result<Distribution> fisherF(Scalar numeratorDof, Scalar denominatorDof) {
  if (!std::isfinite(numeratorDof) || !std::isfinite(denominatorDof)) {
    return Result<Distribution>::failure(ErrorCode::NonFiniteInput,
                                         "F degrees of freedom must be finite");
  }
  if (numeratorDof <= 0.0 || denominatorDof <= 0.0) {
    return Result<Distribution>::failure(
        ErrorCode::InvalidDomain, "F degrees of freedom must be positive");
  }
  return make(Family::fisher_f, Measure::real_density,
              {numeratorDof, denominatorDof});
}

Result<Distribution> empirical(ConstVectorView observations,
                               ConstVectorView masses, Limits limits) {
  if (observations.size() != masses.size()) {
    return Result<Distribution>::failure(
        ErrorCode::DimensionMismatch,
        "empirical observations and masses must have equal length");
  }
  if (observations.size() > 1024U ||
      observations.size() > limits.elements / 4U ||
      observations.size() > limits.bytes / (4U * sizeof(Scalar))) {
    return Result<Distribution>::failure(
        ErrorCode::ResourceLimit,
        "empirical input exceeds its configured storage limit");
  }
  std::vector<Scalar> values;
  std::vector<Scalar> weights;
  values.reserve(observations.size());
  weights.reserve(masses.size());
  for (Index index = 0U; index < observations.size(); ++index) {
    values.push_back(observations.at(index));
    weights.push_back(masses.at(index));
  }
  return detail::DistributionFactory::empirical(std::move(values),
                                                std::move(weights), limits);
}

Result<Distribution> mixture(std::span<const WeightedLaw> terms,
                             ProbabilityPolicy policy) {
  return mixture(terms, policy,
                 {1024U, std::size_t{16U} * 1024U * 1024U, 100000U});
}

Result<Distribution> mixture(std::span<const WeightedLaw> terms,
                             ProbabilityPolicy policy, Limits limits) {
  return detail::DistributionEngine::mixture(terms, policy, limits);
}

Result<Distribution>
finiteConvolution(std::span<const Distribution> independentTerms,
                  ProbabilityPolicy policy) {
  return finiteConvolution(independentTerms, policy,
                           {1024U, std::size_t{16U} * 1024U * 1024U, 100000U});
}

Result<Distribution>
finiteConvolution(std::span<const Distribution> independentTerms,
                  ProbabilityPolicy policy, Limits limits) {
  return detail::DistributionEngine::convolution(independentTerms, policy,
                                                 limits);
}

namespace detail {

// This exhaustive family dispatcher centralizes each law's support and tail
// contracts; composite cases recurse through the depth-limited law tree.
// NOLINTNEXTLINE(readability-function-size,misc-no-recursion)
Result<TailValues> DistributionEngine::evaluate(const Distribution& law,
                                                Scalar x,
                                                ProbabilityPolicy policy) {
  if (!law.state_) {
    return Result<TailValues>::failure(ErrorCode::InvalidDomain,
                                       "distribution is moved from");
  }
  if (std::isnan(x)) {
    return Result<TailValues>::failure(ErrorCode::NonFiniteInput,
                                       "distribution coordinate is NaN");
  }
  if (x == -std::numeric_limits<Scalar>::infinity()) {
    return Result<TailValues>::success(
        {0.0,
         -std::numeric_limits<Scalar>::infinity(),
         0.0,
         1.0,
         {-std::numeric_limits<Scalar>::infinity(), 0.0},
         false});
  }
  if (x == std::numeric_limits<Scalar>::infinity()) {
    return Result<TailValues>::success(
        {0.0,
         -std::numeric_limits<Scalar>::infinity(),
         1.0,
         0.0,
         {0.0, -std::numeric_limits<Scalar>::infinity()},
         false});
  }
  if (!std::isfinite(policy.iteration.abs) ||
      !std::isfinite(policy.iteration.rel) ||
      !std::isfinite(policy.abs_log_error) || policy.iteration.abs <= 0.0 ||
      policy.iteration.rel <= 0.0 || policy.abs_log_error <= 0.0) {
    return Result<TailValues>::failure(
        ErrorCode::InvalidTolerance, "probability tolerances must be positive");
  }
  if (policy.iteration.max_iterations == 0U ||
      policy.iteration.max_evaluations == 0U) {
    return Result<TailValues>::failure(
        ErrorCode::InvalidDomain, "probability work limits must be positive");
  }

  const Distribution::State& state = *law.state_;
  const auto& p = state.parameters;
  const Scalar negativeInfinity = -std::numeric_limits<Scalar>::infinity();
  const Scalar positiveInfinity = std::numeric_limits<Scalar>::infinity();
  Scalar logDensity = negativeInfinity;
  LogTailPair tails{};
  bool onSupport = true;

  auto gaussianTails = [&](Scalar z) -> Result<LogTailPair> {
    return normalLogTails(z, policy);
  };

  switch (state.family) {
  case Family::normal: {
    const Scalar z = (x - p[0]) / p[1];
    logDensity = -0.5 * z * z - std::log(p[1]) - 0.9189385332046727;
    auto pair = gaussianTails(z);
    if (!pair) {
      return Result<TailValues>::failure(pair.status().code(),
                                         std::string(pair.status().message()));
    }
    tails = pair.value();
    onSupport = std::isfinite(x);
    break;
  }
  case Family::student_t: {
    const Scalar nu = p[0];
    const Scalar z = (x - p[1]) / p[2];
    const auto gammaNumerator = logGamma((nu + 1.0) / 2.0, policy);
    const auto gammaDenominator = logGamma(nu / 2.0, policy);
    if (!gammaNumerator || !gammaDenominator) {
      const Status& status =
          gammaNumerator ? gammaDenominator.status() : gammaNumerator.status();
      return Result<TailValues>::failure(status.code(),
                                         std::string(status.message()));
    }
    const Scalar logNorm = gammaNumerator.value() - gammaDenominator.value() -
                           0.5 * std::log(nu * std::numbers::pi) -
                           std::log(p[2]);
    logDensity = logNorm - ((nu + 1.0) / 2.0) * std::log1p(z * z / nu);
    const Scalar ratio = nu / (nu + z * z);
    auto betaTail = incompleteBeta(nu / 2.0, 0.5, ratio, policy);
    if (!betaTail) {
      return Result<TailValues>::failure(
          betaTail.status().code(), std::string(betaTail.status().message()));
    }
    constexpr Scalar logHalf = -std::numbers::ln2;
    const Scalar smaller = betaTail.value().log_cdf + logHalf;
    tails = z < 0.0 ? LogTailPair{smaller, logOneMinusExp(smaller)}
                    : LogTailPair{logOneMinusExp(smaller), smaller};
    onSupport = std::isfinite(x);
    break;
  }
  case Family::skew_normal: {
    const Scalar z = (x - p[0]) / p[1];
    logDensity = std::numbers::ln2 - 0.5 * z * z - 0.9189385332046727 -
                 std::log(p[1]) +
                 std::log(0.5 * std::erfc(-p[2] * z / std::numbers::sqrt2));
    auto normalPair = gaussianTails(z);
    if (!normalPair) {
      return Result<TailValues>::failure(
          normalPair.status().code(),
          std::string(normalPair.status().message()));
    }
    if (p[2] == 0.0) {
      tails = normalPair.value();
    } else {
      auto owen = owenT(z, p[2], policy);
      if (!owen) {
        return Result<TailValues>::failure(
            owen.status().code(), std::string(owen.status().message()));
      }
      Scalar cdf = std::exp(normalPair.value().log_cdf) - 2.0 * owen.value();
      const Scalar roundoff = 16.0 * std::numeric_limits<Scalar>::epsilon();
      if (cdf < 0.0 && cdf >= -roundoff) {
        cdf = 0.0;
      }
      if (cdf > 1.0 && cdf <= 1.0 + roundoff) {
        cdf = 1.0;
      }
      if (cdf < 0.0 || cdf > 1.0 || !std::isfinite(cdf)) {
        return Result<TailValues>::failure(
            ErrorCode::NonFiniteResult,
            "skew-normal CDF is outside the unit interval");
      }
      if (cdf == 0.0) {
        tails = {negativeInfinity, 0.0};
      } else if (cdf == 1.0) {
        tails = {0.0, negativeInfinity};
      } else {
        tails = lowerTail(std::log(cdf));
      }
    }
    onSupport = std::isfinite(x);
    break;
  }
  case Family::laplace: {
    const Scalar z = (x - p[0]) / p[1];
    logDensity = -std::abs(z) - std::log(2.0 * p[1]);
    tails = z <= 0.0 ? LogTailPair{-logTwo - std::abs(z),
                                   logOneMinusExp(-logTwo - std::abs(z))}
                     : LogTailPair{logOneMinusExp(-logTwo - z), -logTwo - z};
    onSupport = std::isfinite(x);
    break;
  }
  case Family::logistic: {
    const Scalar z = (x - p[0]) / p[1];
    tails = {-softplus(-z), -softplus(z)};
    logDensity = tails.log_cdf + tails.log_survival - std::log(p[1]);
    onSupport = std::isfinite(x);
    break;
  }
  case Family::lognormal: {
    if (x < 0.0) {
      return Result<TailValues>::success(
          {0.0, negativeInfinity, 0.0, 1.0, {negativeInfinity, 0.0}, false});
    }
    if (x == 0.0) {
      return Result<TailValues>::success(
          {0.0, negativeInfinity, 0.0, 1.0, {negativeInfinity, 0.0}, true});
    }
    const Scalar z = (std::log(x) - p[0]) / p[1];
    auto pair = gaussianTails(z);
    if (!pair) {
      return Result<TailValues>::failure(pair.status().code(),
                                         std::string(pair.status().message()));
    }
    tails = pair.value();
    logDensity =
        -0.5 * z * z - 0.9189385332046727 - std::log(p[1]) - std::log(x);
    onSupport = true;
    break;
  }
  case Family::gamma: {
    if (x < 0.0) {
      return Result<TailValues>::success(
          {0.0, negativeInfinity, 0.0, 1.0, {negativeInfinity, 0.0}, false});
    }
    const Scalar shape = p[0];
    if (x == 0.0) {
      auto pair = incompleteGamma(shape, 0.0, policy);
      if (!pair) {
        return Result<TailValues>::failure(
            pair.status().code(), std::string(pair.status().message()));
      }
      if (shape < 1.0) {
        return Result<TailValues>::success(
            {positiveInfinity, positiveInfinity, 0.0, 1.0, pair.value(), true});
      }
      logDensity = shape == 1.0 ? -std::log(p[1]) : negativeInfinity;
      tails = pair.value();
      break;
    }
    const Scalar y = x / p[1];
    auto lg = logGamma(shape, policy);
    if (!lg) {
      return Result<TailValues>::failure(lg.status().code(),
                                         std::string(lg.status().message()));
    }
    logDensity = (shape - 1.0) * std::log(y) - y - lg.value() - std::log(p[1]);
    auto pair = incompleteGamma(shape, y, policy);
    if (!pair) {
      return Result<TailValues>::failure(pair.status().code(),
                                         std::string(pair.status().message()));
    }
    tails = {pair.value().log_cdf, pair.value().log_survival};
    onSupport = true;
    break;
  }
  case Family::beta: {
    if (x < 0.0 || x > 1.0) {
      return Result<TailValues>::success(
          x < 0.0 ? TailValues{0.0,
                               negativeInfinity,
                               0.0,
                               1.0,
                               {negativeInfinity, 0.0},
                               false}
                  : TailValues{0.0,
                               negativeInfinity,
                               1.0,
                               0.0,
                               {0.0, negativeInfinity},
                               false});
    }
    const Scalar a = p[0];
    const Scalar b = p[1];
    auto pair = incompleteBeta(a, b, x, policy);
    if (!pair) {
      return Result<TailValues>::failure(pair.status().code(),
                                         std::string(pair.status().message()));
    }
    tails = pair.value();
    if (x == 0.0) {
      if (a < 1.0) {
        return Result<TailValues>::success(
            {positiveInfinity, positiveInfinity, 0.0, 1.0, tails, true});
      }
      logDensity = a == 1.0 ? -logBetaValue(a, b, policy) : negativeInfinity;
    } else if (x == 1.0) {
      if (b < 1.0) {
        return Result<TailValues>::success(
            {positiveInfinity, positiveInfinity, 1.0, 0.0, tails, true});
      }
      logDensity = b == 1.0 ? -logBetaValue(a, b, policy) : negativeInfinity;
    } else {
      logDensity = (a - 1.0) * std::log(x) + (b - 1.0) * std::log1p(-x) -
                   logBetaValue(a, b, policy);
    }
    onSupport = true;
    break;
  }
  case Family::weibull: {
    const Scalar shape = p[0];
    const Scalar scale = p[1];
    if (x < 0.0) {
      return Result<TailValues>::success(
          {0.0, negativeInfinity, 0.0, 1.0, {negativeInfinity, 0.0}, false});
    }
    if (x == 0.0) {
      tails = {negativeInfinity, 0.0};
      if (shape < 1.0) {
        return Result<TailValues>::success(
            {positiveInfinity, positiveInfinity, 0.0, 1.0, tails, true});
      }
      logDensity = shape == 1.0 ? -std::log(scale) : negativeInfinity;
    } else {
      const Scalar logRatio = std::log(x) - std::log(scale);
      const Scalar power = std::exp(shape * logRatio);
      const Scalar logSurvival = -power;
      tails = upperTail(logSurvival);
      logDensity =
          std::log(shape) - std::log(scale) + (shape - 1.0) * logRatio - power;
    }
    onSupport = true;
    break;
  }
  case Family::uniform: {
    const Scalar lower = p[0];
    const Scalar upper = p[1];
    const Scalar width = upper - lower;
    if (x < lower) {
      return Result<TailValues>::success(
          {0.0, negativeInfinity, 0.0, 1.0, {negativeInfinity, 0.0}, false});
    }
    if (x > upper) {
      return Result<TailValues>::success(
          {0.0, negativeInfinity, 1.0, 0.0, {0.0, negativeInfinity}, false});
    }
    const Scalar cdf = (x - lower) / width;
    if (cdf == 0.0) {
      tails = {negativeInfinity, 0.0};
    } else if (cdf == 1.0) {
      tails = {0.0, negativeInfinity};
    } else {
      tails = lowerTail(std::log(cdf));
    }
    logDensity = -std::log(width);
    onSupport = true;
    break;
  }
  case Family::triangular: {
    const Scalar lower = p[0];
    const Scalar mode = p[1];
    const Scalar upper = p[2];
    const Scalar width = upper - lower;
    if (x < lower || x > upper) {
      return Result<TailValues>::success(
          x < lower ? TailValues{0.0,
                                 negativeInfinity,
                                 0.0,
                                 1.0,
                                 {negativeInfinity, 0.0},
                                 false}
                    : TailValues{0.0,
                                 negativeInfinity,
                                 1.0,
                                 0.0,
                                 {0.0, negativeInfinity},
                                 false});
    }
    Scalar cdf = 0.0;
    Scalar density = 0.0;
    if (x <= mode && mode > lower) {
      const Scalar delta = x - lower;
      cdf = delta * delta / (width * (mode - lower));
      density = 2.0 * delta / (width * (mode - lower));
    } else if (x > mode && upper > mode) {
      const Scalar delta = upper - x;
      const Scalar remaining = delta * delta / (width * (upper - mode));
      cdf = 1.0 - remaining;
      density = 2.0 * delta / (width * (upper - mode));
    } else if (mode == lower) {
      const Scalar delta = upper - x;
      cdf = 1.0 - (delta * delta) / (width * width);
      density = 2.0 * delta / (width * width);
    } else {
      const Scalar delta = x - lower;
      cdf = (delta * delta) / (width * width);
      density = 2.0 * delta / (width * width);
    }
    if (cdf <= 0.0) {
      tails = {negativeInfinity, 0.0};
    } else if (cdf >= 1.0) {
      tails = {0.0, negativeInfinity};
    } else {
      tails = lowerTail(std::log(cdf));
    }
    logDensity = density == 0.0 ? negativeInfinity : std::log(density);
    onSupport = true;
    break;
  }
  case Family::truncated_normal: {
    const Scalar meanValue = p[0];
    const Scalar sd = p[1];
    const Scalar lowerZ = (p[2] - meanValue) / sd;
    const Scalar upperZ = (p[3] - meanValue) / sd;
    const Scalar z = (x - meanValue) / sd;
    auto lowerPair = gaussianTails(lowerZ);
    auto upperPair = gaussianTails(upperZ);
    auto atX = gaussianTails(z);
    if (!lowerPair || !upperPair || !atX) {
      const Status* status = nullptr;
      if (!lowerPair) {
        status = &lowerPair.status();
      } else if (!upperPair) {
        status = &upperPair.status();
      } else {
        status = &atX.status();
      }
      return Result<TailValues>::failure(status->code(),
                                         std::string(status->message()));
    }
    Scalar logNormalizer = negativeInfinity;
    if (lowerZ >= 0.0) {
      logNormalizer = logDifference(lowerPair.value().log_survival,
                                    upperPair.value().log_survival);
    } else if (upperZ <= 0.0) {
      logNormalizer =
          logDifference(upperPair.value().log_cdf, lowerPair.value().log_cdf);
    } else {
      const Scalar excluded = std::exp(lowerPair.value().log_cdf) +
                              std::exp(upperPair.value().log_survival);
      logNormalizer = std::log1p(-excluded);
    }
    if (!std::isfinite(logNormalizer)) {
      return Result<TailValues>::failure(
          ErrorCode::InvalidDomain,
          "truncated-normal interval has no representable probability");
    }
    const Scalar standardLogDensity = -0.5 * z * z - 0.9189385332046727;
    logDensity = standardLogDensity - std::log(sd) - logNormalizer;
    if (x <= p[2]) {
      tails = {negativeInfinity, 0.0};
    } else if (x >= p[3]) {
      tails = {0.0, negativeInfinity};
    } else {
      const Scalar logLower =
          logDifference(atX.value().log_cdf, lowerPair.value().log_cdf) -
          logNormalizer;
      const Scalar logUpper = logDifference(atX.value().log_survival,
                                            upperPair.value().log_survival);
      tails = {logLower, logUpper - logNormalizer};
    }
    onSupport = x >= p[2] && x <= p[3];
    break;
  }
  case Family::truncated: {
    const Scalar lower = p[0];
    const Scalar upper = p[1];
    const Scalar logNormalizer = p[2];
    const bool includeLower = (state.integerParameter & 1U) != 0U;
    const bool includeUpper = (state.integerParameter & 2U) != 0U;
    if (x < lower || x > upper) {
      return Result<TailValues>::success(
          {0.0, negativeInfinity, x < lower ? 0.0 : 1.0, x < lower ? 1.0 : 0.0,
           x < lower ? LogTailPair{negativeInfinity, 0.0}
                     : LogTailPair{0.0, negativeInfinity},
           false});
    }
    const Distribution base(state.components.front());
    auto baseValue = evaluate(base, x, policy);
    if (!baseValue) {
      return Result<TailValues>::failure(
          baseValue.status().code(), std::string(baseValue.status().message()));
    }
    const bool discrete = state.measure == Measure::integer_mass ||
                          state.measure == Measure::atom_mass;
    const bool excludedAtom = discrete && ((x == lower && !includeLower) ||
                                           (x == upper && !includeUpper));
    logDensity = excludedAtom
                     ? negativeInfinity
                     : baseValue.value().log_density_or_mass - logNormalizer;

    const auto lowerMass = intervalLogProbability(
        base, IntervalSupport{lower, x, includeLower, true}, policy);
    const auto upperMass = intervalLogProbability(
        base, IntervalSupport{x, upper, false, includeUpper}, policy);
    if (!lowerMass || !upperMass) {
      const Status& status =
          !lowerMass ? lowerMass.status() : upperMass.status();
      return Result<TailValues>::failure(status.code(),
                                         std::string(status.message()));
    }
    const Scalar logCdf = lowerMass.value() - logNormalizer;
    const Scalar logSurvival = upperMass.value() - logNormalizer;
    const Scalar logTotal = logAddExp(logCdf, logSurvival);
    tails = {std::min(0.0, logCdf - logTotal),
             std::min(0.0, logSurvival - logTotal)};
    onSupport = baseValue.value().on_support && x >= lower && x <= upper &&
                !excludedAtom;
    break;
  }
  case Family::bernoulli: {
    const Scalar probability = p[0];
    const bool atZero = x == 0.0;
    const bool atOne = x == 1.0;
    Scalar mass = 0.0;
    if (atZero) {
      mass = 1.0 - probability;
    } else if (atOne) {
      mass = probability;
    }
    logDensity = mass > 0.0 ? std::log(mass) : negativeInfinity;
    Scalar cdf = 1.0;
    Scalar survival = 0.0;
    if (x < 0.0) {
      cdf = 0.0;
      survival = 1.0;
    } else if (x < 1.0) {
      cdf = 1.0 - probability;
      survival = probability;
    }
    tails = {cdf == 0.0 ? negativeInfinity : std::log(cdf),
             survival == 0.0 ? negativeInfinity : std::log(survival)};
    onSupport = atZero || atOne;
    return Result<TailValues>::success(
        {mass, logDensity, cdf, survival, tails, onSupport});
  }
  case Family::binomial: {
    const std::uint64_t trials = state.integerParameter;
    const Scalar probability = p[1];
    if (x < 0.0) {
      return Result<TailValues>::success(
          {0.0, negativeInfinity, 0.0, 1.0, {negativeInfinity, 0.0}, false});
    }
    const auto n = static_cast<Scalar>(trials);
    if (x >= n || probability == 0.0 || probability == 1.0) {
      const Scalar atom = probability == 1.0 ? n : 0.0;
      const Scalar mass = (x == atom) ? 1.0 : 0.0;
      const Scalar cdf = x < atom ? 0.0 : 1.0;
      const Scalar survival = 1.0 - cdf;
      tails = cdf == 0.0 ? LogTailPair{negativeInfinity, 0.0}
                         : LogTailPair{0.0, negativeInfinity};
      return Result<TailValues>::success({mass,
                                          mass == 0.0 ? negativeInfinity : 0.0,
                                          cdf, survival, tails, x == atom});
    }
    const Scalar k = std::floor(x);
    const bool integerCoordinate = x == k;
    Scalar cdf = 0.0;
    Scalar survival = 1.0;
    if (k >= n) {
      cdf = 1.0;
      survival = 0.0;
    } else {
      auto lower = incompleteBeta(n - k, k + 1.0, 1.0 - probability, policy);
      if (!lower) {
        return Result<TailValues>::failure(
            lower.status().code(), std::string(lower.status().message()));
      }
      auto upper = incompleteBeta(k + 1.0, n - k, probability, policy);
      if (!upper) {
        return Result<TailValues>::failure(
            upper.status().code(), std::string(upper.status().message()));
      }
      tails = {lower.value().log_cdf, upper.value().log_cdf};
      cdf = std::exp(tails.log_cdf);
      survival = std::exp(tails.log_survival);
    }
    Scalar mass = 0.0;
    if (integerCoordinate && k <= n) {
      const auto gN = logGamma(n + 1.0, policy);
      const auto gK = logGamma(k + 1.0, policy);
      const auto gR = logGamma(n - k + 1.0, policy);
      if (!gN || !gK || !gR) {
        const Status* status = nullptr;
        if (!gN) {
          status = &gN.status();
        } else if (!gK) {
          status = &gK.status();
        } else {
          status = &gR.status();
        }
        return Result<TailValues>::failure(status->code(),
                                           std::string(status->message()));
      }
      logDensity = gN.value() - gK.value() - gR.value();
      if (k > 0.0) {
        logDensity += k * std::log(probability);
      }
      if (n > k) {
        logDensity += (n - k) * std::log1p(-probability);
      }
      mass = std::exp(logDensity);
    } else {
      logDensity = negativeInfinity;
    }
    if (!integerCoordinate) {
      onSupport = false;
    } else {
      onSupport = x >= 0.0 && x <= n;
    }
    return Result<TailValues>::success(
        {mass, logDensity, cdf, survival, tails, onSupport});
  }
  case Family::poisson: {
    const Scalar rate = p[0];
    if (x < 0.0) {
      return Result<TailValues>::success(
          {0.0, negativeInfinity, 0.0, 1.0, {negativeInfinity, 0.0}, false});
    }
    if (rate == 0.0) {
      const Scalar mass = x == 0.0 ? 1.0 : 0.0;
      const Scalar cdf = x < 0.0 ? 0.0 : 1.0;
      return Result<TailValues>::success({mass,
                                          mass == 0.0 ? negativeInfinity : 0.0,
                                          cdf,
                                          0.0,
                                          {0.0, negativeInfinity},
                                          x == 0.0});
    }
    const Scalar k = std::floor(x);
    const bool integerCoordinate = x == k;
    auto gammaPair = incompleteGamma(k + 1.0, rate, policy);
    if (!gammaPair) {
      return Result<TailValues>::failure(
          gammaPair.status().code(), std::string(gammaPair.status().message()));
    }
    tails = {gammaPair.value().log_survival, gammaPair.value().log_cdf};
    Scalar mass = 0.0;
    if (integerCoordinate) {
      auto factorial = logGamma(k + 1.0, policy);
      if (!factorial) {
        return Result<TailValues>::failure(
            factorial.status().code(),
            std::string(factorial.status().message()));
      }
      logDensity = -rate + k * std::log(rate) - factorial.value();
      mass = std::exp(logDensity);
    } else {
      logDensity = negativeInfinity;
    }
    onSupport = integerCoordinate;
    return Result<TailValues>::success(
        {mass, logDensity, std::exp(tails.log_cdf),
         std::exp(tails.log_survival), tails, onSupport});
  }
  case Family::negative_binomial: {
    const Scalar successes = p[0];
    const Scalar probability = p[1];
    if (x < 0.0) {
      return Result<TailValues>::success(
          {0.0, negativeInfinity, 0.0, 1.0, {negativeInfinity, 0.0}, false});
    }
    if (probability == 1.0) {
      const Scalar mass = x == 0.0 ? 1.0 : 0.0;
      const Scalar cdf = x < 0.0 ? 0.0 : 1.0;
      return Result<TailValues>::success({mass,
                                          mass == 0.0 ? negativeInfinity : 0.0,
                                          cdf,
                                          0.0,
                                          {0.0, negativeInfinity},
                                          x == 0.0});
    }
    const Scalar k = std::floor(x);
    const bool integerCoordinate = x == k;
    auto lower = incompleteBeta(successes, k + 1.0, probability, policy);
    auto upper = incompleteBeta(k + 1.0, successes, 1.0 - probability, policy);
    if (!lower || !upper) {
      const Status& status = !lower ? lower.status() : upper.status();
      return Result<TailValues>::failure(status.code(),
                                         std::string(status.message()));
    }
    tails = {lower.value().log_cdf, upper.value().log_cdf};
    if (integerCoordinate) {
      auto first = logGamma(successes + k, policy);
      auto second = logGamma(k + 1.0, policy);
      auto third = logGamma(successes, policy);
      if (!first || !second || !third) {
        const Status* status = nullptr;
        if (!first) {
          status = &first.status();
        } else if (!second) {
          status = &second.status();
        } else {
          status = &third.status();
        }
        return Result<TailValues>::failure(status->code(),
                                           std::string(status->message()));
      }
      logDensity = first.value() - second.value() - third.value() +
                   successes * std::log(probability);
      if (k > 0.0) {
        logDensity += k * std::log1p(-probability);
      }
    } else {
      logDensity = negativeInfinity;
    }
    return finiteTailValues(logDensity, tails, integerCoordinate);
  }
  case Family::fisher_f: {
    const Scalar numerator = p[0];
    const Scalar denominator = p[1];
    if (x < 0.0) {
      return Result<TailValues>::success(
          {0.0, negativeInfinity, 0.0, 1.0, {negativeInfinity, 0.0}, false});
    }
    if (x == 0.0) {
      auto pair =
          incompleteBeta(numerator / 2.0, denominator / 2.0, 0.0, policy);
      if (!pair) {
        return Result<TailValues>::failure(
            pair.status().code(), std::string(pair.status().message()));
      }
      if (numerator < 2.0) {
        return Result<TailValues>::success(
            {positiveInfinity, positiveInfinity, 0.0, 1.0, pair.value(), true});
      }
      logDensity =
          numerator == 2.0
              ? (numerator / 2.0) * std::log(numerator / denominator) -
                    logBetaValue(numerator / 2.0, denominator / 2.0, policy)
              : negativeInfinity;
      tails = pair.value();
      break;
    }
    const Scalar logScale = std::log(numerator) - std::log(denominator);
    const Scalar logRatio = logScale + std::log(x);
    const Scalar transform =
        logRatio > 0.0 ? 1.0 / (1.0 + std::exp(-logRatio))
                       : std::exp(logRatio) / (1.0 + std::exp(logRatio));
    auto pair =
        incompleteBeta(numerator / 2.0, denominator / 2.0, transform, policy);
    if (!pair) {
      return Result<TailValues>::failure(pair.status().code(),
                                         std::string(pair.status().message()));
    }
    tails = pair.value();
    const Scalar logDenom = softplus(logRatio);
    logDensity = (numerator / 2.0) * logScale +
                 (numerator / 2.0 - 1.0) * std::log(x) -
                 ((numerator + denominator) / 2.0) * logDenom -
                 logBetaValue(numerator / 2.0, denominator / 2.0, policy);
    onSupport = true;
    break;
  }
  case Family::chi_square: {
    if (x < 0.0) {
      return Result<TailValues>::success(
          {0.0, negativeInfinity, 0.0, 1.0, {negativeInfinity, 0.0}, false});
    }
    const Scalar shape = p[0] / 2.0;
    const Scalar y = x / 2.0;
    if (x == 0.0 && shape < 1.0) {
      auto pair = incompleteGamma(shape, 0.0, policy);
      if (!pair) {
        return Result<TailValues>::failure(
            pair.status().code(), std::string(pair.status().message()));
      }
      return Result<TailValues>::success(
          {positiveInfinity, positiveInfinity, 0.0, 1.0, pair.value(), true});
    }
    if (x == 0.0) {
      tails = {negativeInfinity, 0.0};
      logDensity = shape == 1.0 ? -std::numbers::ln2 : negativeInfinity;
      break;
    }
    auto lg = logGamma(shape, policy);
    if (!lg) {
      return Result<TailValues>::failure(lg.status().code(),
                                         std::string(lg.status().message()));
    }
    logDensity =
        (shape - 1.0) * std::log(y) - y - lg.value() - std::numbers::ln2;
    auto pair = incompleteGamma(shape, y, policy);
    if (!pair) {
      return Result<TailValues>::failure(pair.status().code(),
                                         std::string(pair.status().message()));
    }
    tails = {pair.value().log_cdf, pair.value().log_survival};
    break;
  }
  case Family::finite_convolution: {
    if (!state.components.empty()) {
      if (state.components.size() == 2U) {
        std::shared_ptr<const Distribution::State> normalState;
        std::shared_ptr<const Distribution::State> uniformState;
        for (const auto& component : state.components) {
          if (component->family == Family::normal)
            normalState = component;
          if (component->family == Family::uniform)
            uniformState = component;
        }
        if (normalState && uniformState) {
          const Scalar mean = normalState->parameters[0];
          const Scalar deviation = normalState->parameters[1];
          const Scalar lower = uniformState->parameters[0];
          const Scalar upper = uniformState->parameters[1];
          const Scalar width = upper - lower;
          const Scalar zLower = (x - lower - mean) / deviation;
          const Scalar zUpper = (x - upper - mean) / deviation;
          auto lowerTails = gaussianTails(zLower);
          auto upperTails = gaussianTails(zUpper);
          if (!lowerTails || !upperTails) {
            const Status& status =
                lowerTails ? upperTails.status() : lowerTails.status();
            return Result<TailValues>::failure(status.code(),
                                               std::string(status.message()));
          }
          logDensity = logDifference(lowerTails.value().log_cdf,
                                     upperTails.value().log_cdf) -
                       std::log(width);
          const auto cdfPrimitive = [](Scalar z) {
            const Scalar density =
                std::exp(-0.5 * z * z) / std::sqrt(2.0 * std::numbers::pi);
            const Scalar cdf = 0.5 * std::erfc(-z / std::numbers::sqrt2);
            return z * cdf + density;
          };
          const auto survivalPrimitive = [](Scalar z) {
            const Scalar density =
                std::exp(-0.5 * z * z) / std::sqrt(2.0 * std::numbers::pi);
            const Scalar survival = 0.5 * std::erfc(z / std::numbers::sqrt2);
            return z * survival - density;
          };
          const Scalar middle = (zLower + zUpper) / 2.0;
          Scalar cdf = 0.0;
          Scalar survival = 0.0;
          if (middle <= 0.0) {
            cdf = deviation * (cdfPrimitive(zLower) - cdfPrimitive(zUpper)) /
                  width;
            cdf = std::clamp(cdf, 0.0, 1.0);
            survival = 1.0 - cdf;
          } else {
            survival = deviation *
                       (survivalPrimitive(zLower) - survivalPrimitive(zUpper)) /
                       width;
            survival = std::clamp(survival, 0.0, 1.0);
            cdf = 1.0 - survival;
          }
          tails = {cdf > 0.0 ? std::log(cdf) : negativeInfinity,
                   survival > 0.0 ? std::log(survival) : negativeInfinity};
          onSupport = true;
          return finiteTailValues(logDensity, tails, onSupport);
        }
        std::shared_ptr<const Distribution::State> poissonState;
        std::shared_ptr<const Distribution::State> truncatedState;
        for (const auto& component : state.components) {
          if (component->family == Family::poisson)
            poissonState = component;
          if (component->family == Family::truncated &&
              component->measure == Measure::real_density) {
            truncatedState = component;
          }
        }
        if (poissonState && truncatedState) {
          const Scalar lower = truncatedState->parameters[0];
          const Scalar upper = truncatedState->parameters[1];
          const Scalar rate = poissonState->parameters[0];
          const Distribution base(truncatedState);
          if (rate == 0.0)
            return evaluate(base, x, policy);
          if (x < lower) {
            return Result<TailValues>::success({0.0,
                                                negativeInfinity,
                                                0.0,
                                                1.0,
                                                {negativeInfinity, 0.0},
                                                false});
          }
          const long double maximumCountValue =
              std::floor(static_cast<long double>(x) - lower);
          if (maximumCountValue >
                  static_cast<long double>(
                      std::numeric_limits<std::uint64_t>::max()) ||
              maximumCountValue >=
                  static_cast<long double>(policy.iteration.max_evaluations)) {
            return Result<TailValues>::failure(
                ErrorCode::ResourceLimit,
                "Poisson and bounded convolution exceeds evaluation limit");
          }
          const auto maximumCount =
              static_cast<std::uint64_t>(std::max(0.0L, maximumCountValue));
          const long double minimumCountValue =
              std::ceil(static_cast<long double>(x) - upper);
          const auto minimumCount = static_cast<std::uint64_t>(
              std::max(0.0L, std::min(minimumCountValue,
                                      static_cast<long double>(maximumCount))));
          Scalar logMass = -rate;
          Scalar logCdf = negativeInfinity;
          Scalar logSurvival = negativeInfinity;
          Scalar logDensitySum = negativeInfinity;
          std::uint64_t evaluations = 0U;
          for (std::uint64_t k = 0U; k <= maximumCount; ++k) {
            if (++evaluations > policy.iteration.max_evaluations) {
              return Result<TailValues>::failure(
                  ErrorCode::ResourceLimit,
                  "Poisson and bounded convolution exceeds evaluation limit");
            }
            auto value = evaluate(base, x - static_cast<Scalar>(k), policy);
            if (!value)
              return Result<TailValues>::failure(
                  value.status().code(), std::string(value.status().message()));
            logCdf =
                logAddExp(logCdf, logMass + value.value().log_tails.log_cdf);
            logSurvival = logAddExp(
                logSurvival, logMass + value.value().log_tails.log_survival);
            if (k >= minimumCount &&
                value.value().log_density_or_mass != negativeInfinity) {
              logDensitySum = logAddExp(
                  logDensitySum, logMass + value.value().log_density_or_mass);
            }
            if (k < maximumCount) {
              logMass += std::log(rate) - std::log(static_cast<Scalar>(k + 1U));
            }
          }
          Distribution poissonLaw(poissonState);
          auto omittedCounts =
              evaluate(poissonLaw, static_cast<Scalar>(maximumCount), policy);
          if (!omittedCounts)
            return Result<TailValues>::failure(
                omittedCounts.status().code(),
                std::string(omittedCounts.status().message()));
          logSurvival = logAddExp(logSurvival,
                                  omittedCounts.value().log_tails.log_survival);
          return finiteTailValues(logDensitySum, {logCdf, logSurvival}, true);
        }
        std::shared_ptr<const Distribution::State> gammaOuterState;
        for (const auto& component : state.components) {
          if (component->family == Family::gamma)
            gammaOuterState = component;
          if (component->family == Family::truncated &&
              component->measure == Measure::real_density) {
            truncatedState = component;
          }
        }
        if (gammaOuterState && truncatedState) {
          const Scalar lower = truncatedState->parameters[0];
          const Scalar upper = truncatedState->parameters[1];
          const Scalar shape = gammaOuterState->parameters[0];
          const Distribution base(truncatedState);
          const Distribution gammaLaw(gammaOuterState);
          if (x < lower) {
            return Result<TailValues>::success({0.0,
                                                negativeInfinity,
                                                0.0,
                                                1.0,
                                                {negativeInfinity, 0.0},
                                                false});
          }
          if (policy.iteration.max_evaluations < 15U ||
              policy.iteration.max_iterations == 0U) {
            return Result<TailValues>::failure(
                ErrorCode::ResourceLimit,
                "Gamma and bounded convolution work budget is too small");
          }
          IterPolicy innerPolicy = policy.iteration;
          innerPolicy.abs =
              std::min(policy.iteration.abs, policy.abs_log_error);
          innerPolicy.rel =
              std::min(policy.iteration.rel, policy.abs_log_error);
          const Limits innerLimits{distributionWorkspaceElements,
                                   distributionWorkspaceBytes,
                                   policy.iteration.max_evaluations};
          Scalar density = 0.0;
          const Scalar densityLower = std::max(0.0, x - upper);
          const Scalar densityUpper = x - lower;
          if (densityUpper > densityLower) {
            const Scalar width = densityUpper - densityLower;
            const Index power =
                shape < 2.0
                    ? static_cast<Index>(std::max(2.0, std::ceil(2.0 / shape)))
                    : 1U;
            const UnaryFunction densityKernel =
                [&](Scalar coordinate) -> Result<Scalar> {
              const Scalar powered =
                  std::pow(coordinate, static_cast<Scalar>(power));
              const Scalar v = densityLower + width * powered;
              const Scalar jacobian =
                  width * static_cast<Scalar>(power) *
                  std::pow(coordinate, static_cast<Scalar>(power - 1U));
              auto baseValue = evaluate(base, x - v, policy);
              if (!baseValue)
                return Result<Scalar>::failure(
                    baseValue.status().code(),
                    std::string(baseValue.status().message()));
              auto gammaValue = evaluate(gammaLaw, v, policy);
              if (!gammaValue)
                return Result<Scalar>::failure(
                    gammaValue.status().code(),
                    std::string(gammaValue.status().message()));
              if (baseValue.value().log_density_or_mass == negativeInfinity ||
                  gammaValue.value().log_density_or_mass == negativeInfinity) {
                return Result<Scalar>::success(0.0);
              }
              const long double value =
                  std::exp(static_cast<long double>(
                               baseValue.value().log_density_or_mass) +
                           gammaValue.value().log_density_or_mass) *
                  jacobian;
              if (!std::isfinite(value) ||
                  value > std::numeric_limits<Scalar>::max()) {
                return Result<Scalar>::failure(
                    ErrorCode::NonFiniteResult,
                    "Gamma and bounded density is not representable");
              }
              return Result<Scalar>::success(static_cast<Scalar>(value));
            };
            auto integral = quadratureAdaptive(densityKernel, {0.0, 1.0},
                                               innerPolicy, innerLimits);
            if (!integral)
              return Result<TailValues>::failure(
                  integral.status().code(),
                  std::string(integral.status().message()));
            if (integral.value().termination != Termination::converged) {
              return Result<TailValues>::failure(
                  ErrorCode::NoConvergence,
                  "Gamma and bounded density quadrature did not converge");
            }
            density = integral.value().value;
          } else if (x == lower && shape < 1.0) {
            density = positiveInfinity;
          }
          Scalar cdf = 0.0;
          const Scalar cdfUpper = std::min(upper, x);
          if (cdfUpper > lower) {
            const Scalar width = cdfUpper - lower;
            const Index power =
                (x <= upper && shape < 2.0)
                    ? static_cast<Index>(std::max(2.0, std::ceil(2.0 / shape)))
                    : 1U;
            const UnaryFunction cdfKernel =
                [&](Scalar coordinate) -> Result<Scalar> {
              const Scalar powered =
                  std::pow(coordinate, static_cast<Scalar>(power));
              const Scalar t = cdfUpper - width * powered;
              const Scalar jacobian =
                  width * static_cast<Scalar>(power) *
                  std::pow(coordinate, static_cast<Scalar>(power - 1U));
              auto baseValue = evaluate(base, t, policy);
              if (!baseValue)
                return Result<Scalar>::failure(
                    baseValue.status().code(),
                    std::string(baseValue.status().message()));
              auto gammaValue = evaluate(gammaLaw, x - t, policy);
              if (!gammaValue)
                return Result<Scalar>::failure(
                    gammaValue.status().code(),
                    std::string(gammaValue.status().message()));
              const long double value =
                  static_cast<long double>(baseValue.value().density_or_mass) *
                  gammaValue.value().cdf * jacobian;
              if (!std::isfinite(value) ||
                  value > std::numeric_limits<Scalar>::max()) {
                return Result<Scalar>::failure(
                    ErrorCode::NonFiniteResult,
                    "Gamma and bounded CDF is not representable");
              }
              return Result<Scalar>::success(static_cast<Scalar>(value));
            };
            auto integral = quadratureAdaptive(cdfKernel, {0.0, 1.0},
                                               innerPolicy, innerLimits);
            if (!integral)
              return Result<TailValues>::failure(
                  integral.status().code(),
                  std::string(integral.status().message()));
            if (integral.value().termination != Termination::converged) {
              return Result<TailValues>::failure(
                  ErrorCode::NoConvergence,
                  "Gamma and bounded CDF quadrature did not converge");
            }
            cdf = std::clamp(integral.value().value, 0.0, 1.0);
          }
          const Scalar survival = 1.0 - cdf;
          logDensity = density > 0.0 ? std::log(density) : negativeInfinity;
          tails = {cdf > 0.0 ? std::log(cdf) : negativeInfinity,
                   survival > 0.0 ? std::log(survival) : negativeInfinity};
          onSupport = true;
          return finiteTailValues(logDensity, tails, onSupport);
        }
        std::shared_ptr<const Distribution::State> betaState;
        std::shared_ptr<const Distribution::State> gammaState;
        for (const auto& component : state.components) {
          if (component->family == Family::beta &&
              component->parameters[0] == 1.0) {
            betaState = component;
          } else if (component->family == Family::gamma &&
                     component->parameters[1] == 1.0) {
            gammaState = component;
          }
        }
        if (betaState && gammaState) {
          if (x < 0.0) {
            return Result<TailValues>::success({0.0,
                                                negativeInfinity,
                                                0.0,
                                                1.0,
                                                {negativeInfinity, 0.0},
                                                false});
          }
          const Scalar betaShape = betaState->parameters[1];
          const Scalar gammaShape = gammaState->parameters[0];
          const Scalar probabilitySlack =
              8.0 * (policy.iteration.abs + policy.iteration.rel);
          const Scalar collisionExponent = betaShape + gammaShape - 1.0;
          const bool densityPole = x == 1.0 && collisionExponent <= 0.0;
          auto gammaLogNormalizer = logGamma(gammaShape, policy);
          if (!gammaLogNormalizer) {
            return Result<TailValues>::failure(
                gammaLogNormalizer.status().code(),
                std::string(gammaLogNormalizer.status().message()));
          }
          if (x == 0.0) {
            if (collisionExponent < 0.0) {
              return Result<TailValues>::success({positiveInfinity,
                                                  positiveInfinity,
                                                  0.0,
                                                  1.0,
                                                  {negativeInfinity, 0.0},
                                                  true});
            }
            if (collisionExponent == 0.0) {
              auto numerator = logGamma(betaShape + 1.0, policy);
              auto denominator = logGamma(betaShape + gammaShape, policy);
              if (!numerator || !denominator) {
                const Status& status =
                    numerator ? denominator.status() : numerator.status();
                return Result<TailValues>::failure(
                    status.code(), std::string(status.message()));
              }
              return finiteTailValues(numerator.value() - denominator.value(),
                                      {negativeInfinity, 0.0}, true);
            }
            return finiteTailValues(negativeInfinity, {negativeInfinity, 0.0},
                                    true);
          }
          // Accumulate scaled probability integrals in extended precision;
          // ordinary tails may underflow after their logarithmic offsets
          // return.
          using IntegralValues = std::array<long double, 3>;
          // NOLINTBEGIN(misc-non-private-member-variables-in-classes): this
          // private quadrature work record has no public API or invariants.
          struct QuadratureCell final {
            Scalar left;
            Scalar right;
            IntegralValues estimate;
            IntegralValues error;
            Index depth;

            QuadratureCell(Scalar leftBound, Scalar rightBound,
                           IntegralValues estimateValue,
                           IntegralValues errorValue, Index level)
                : left(leftBound), right(rightBound), estimate(estimateValue),
                  error(errorValue), depth(level) {}
          };
          // NOLINTEND(misc-non-private-member-variables-in-classes)
          std::uint64_t evaluations = 0U;
          const auto integrate =
              [&](Scalar lower, Scalar upper, Index power,
                  const std::function<Result<IntegralValues>(Scalar, Scalar)>&
                      kernel) -> Result<IntegralValues> {
            if (!(upper > lower)) {
              return Result<IntegralValues>::success({0.0, 0.0, 0.0});
            }
            const Scalar width = upper - lower;
            const auto sample =
                [&](Scalar coordinate) -> Result<IntegralValues> {
              if (evaluations >= policy.iteration.max_evaluations) {
                return Result<IntegralValues>::failure(
                    ErrorCode::NoConvergence,
                    "beta-gamma convolution reached its evaluation limit");
              }
              ++evaluations;
              const Scalar powered = std::pow(coordinate, power);
              const Scalar value = lower + (width * powered);
              const Scalar jacobian = width * static_cast<Scalar>(power) *
                                      std::pow(coordinate, power - 1U);
              return kernel(value, jacobian);
            };
            constexpr std::array<Scalar, 16> kronrodNodes{
                0.998002298693397060285172840152271209,
                0.987992518020485428489565718586612581,
                0.967739075679139134257347978784337225,
                0.937273392400705904307758947710209471,
                0.897264532344081900882509656454495883,
                0.848206583410427216200648320774216851,
                0.790418501442465932967649294817947347,
                0.724417731360170047416186054613938010,
                0.650996741297416970533735895313274693,
                0.570972172608538847537226737253910641,
                0.485081863640239680693655740232350613,
                0.394151347077563369897207370981045468,
                0.299180007153168812166780024266388963,
                0.201194093997434522300628303394596208,
                0.101142066918717499027074231447392339,
                0.0};
            // Embedded 15-point Gauss / 31-point Kronrod rule (QUADPACK DQK31).
            constexpr std::array<Scalar, 16> kronrodWeights{
                0.005377479872923348987792051430127650,
                0.015007947329316122538374763075807268,
                0.025460847326715320186874001019653359,
                0.035346360791375846222037948478360048,
                0.044589751324764876608227299373279690,
                0.053481524690928087265343147239430297,
                0.062009567800670640285139230960802932,
                0.069854121318728258709520077099147476,
                0.076849680757720378894432777482659007,
                0.083080502823133021038289247286103790,
                0.088564443056211770647275443693774303,
                0.093126598170825321225486872747345719,
                0.096642726983623678505179907627589335,
                0.099173598721791959332393173484603131,
                0.100769845523875595044946662617569722,
                0.101330007014791549017374792767492547};
            constexpr std::array<Scalar, 8> gaussWeights{
                0.030753241996117268354628393577204418,
                0.070366047488108124709267416450667338,
                0.107159220467171935011869546685869303,
                0.139570677926154314447804794511028323,
                0.166269205816993933553200860481208811,
                0.186161000015562211026800561866422825,
                0.198431485327111576456118326443839325,
                0.202578241925561272880620199967519315};
            const auto evaluateCell =
                [&](Scalar left, Scalar right,
                    Index depth) -> Result<QuadratureCell> {
              const Scalar center = 0.5 * (left + right);
              const Scalar halfWidth = 0.5 * (right - left);
              std::array<IntegralValues, 31> values{};
              IntegralValues kronrod{};
              IntegralValues gauss{};
              IntegralValues absolute{};
              const auto accumulate = [&](const IntegralValues& value,
                                          long double weight,
                                          IntegralValues& total) {
                for (Index channel = 0U; channel < total.size(); ++channel) {
                  total.at(channel) += weight * value.at(channel);
                }
              };
              auto centerValue = sample(center);
              if (!centerValue) {
                return Result<QuadratureCell>::failure(
                    centerValue.status().code(),
                    std::string(centerValue.status().message()));
              }
              values[30] = centerValue.value();
              accumulate(values[30], kronrodWeights[15], kronrod);
              accumulate(values[30], std::abs(kronrodWeights[15]), absolute);
              accumulate(values[30], gaussWeights[7], gauss);
              for (Index node = 0U; node < 15U; ++node) {
                const Scalar offset = halfWidth * kronrodNodes.at(node);
                auto lowerValue = sample(center - offset);
                auto upperValue = sample(center + offset);
                if (!lowerValue || !upperValue) {
                  const Status& status =
                      lowerValue ? upperValue.status() : lowerValue.status();
                  return Result<QuadratureCell>::failure(
                      status.code(), std::string(status.message()));
                }
                values.at(2U * node) = lowerValue.value();
                values.at((2U * node) + 1U) = upperValue.value();
                IntegralValues pair{};
                for (Index channel = 0U; channel < pair.size(); ++channel) {
                  pair.at(channel) = lowerValue.value().at(channel) +
                                     upperValue.value().at(channel);
                }
                accumulate(pair, kronrodWeights.at(node), kronrod);
                accumulate({std::abs(lowerValue.value()[0]),
                            std::abs(lowerValue.value()[1]),
                            std::abs(lowerValue.value()[2])},
                           kronrodWeights.at(node), absolute);
                accumulate({std::abs(upperValue.value()[0]),
                            std::abs(upperValue.value()[1]),
                            std::abs(upperValue.value()[2])},
                           kronrodWeights.at(node), absolute);
                if (node % 2U == 1U) {
                  accumulate(pair, gaussWeights.at(node / 2U), gauss);
                }
              }
              IntegralValues estimate{};
              IntegralValues error{};
              IntegralValues mean{};
              IntegralValues asc{};
              for (Index channel = 0U; channel < estimate.size(); ++channel) {
                estimate.at(channel) = kronrod.at(channel) * halfWidth;
                mean.at(channel) = 0.5 * kronrod.at(channel);
                asc.at(channel) =
                    kronrodWeights[15] *
                    std::abs(values[30].at(channel) - mean.at(channel));
                for (Index node = 0U; node < 15U; ++node) {
                  asc.at(channel) +=
                      kronrodWeights.at(node) *
                      (std::abs(values.at(2U * node).at(channel) -
                                mean.at(channel)) +
                       std::abs(values.at((2U * node) + 1U).at(channel) -
                                mean.at(channel)));
                }
                const long double resabs = absolute.at(channel) * halfWidth;
                asc.at(channel) *= halfWidth;
                error.at(channel) = std::abs(
                    (kronrod.at(channel) - gauss.at(channel)) * halfWidth);
                if (asc.at(channel) != 0.0 && error.at(channel) != 0.0) {
                  error.at(channel) =
                      asc.at(channel) *
                      std::min(1.0L, std::pow(200.0L * error.at(channel) /
                                                  asc.at(channel),
                                              1.5L));
                }
                const long double roundoffFloor =
                    50.0L * std::numeric_limits<Scalar>::epsilon() * resabs;
                error.at(channel) = std::max(error.at(channel), roundoffFloor);
              }
              return Result<QuadratureCell>::success(
                  {left, right, estimate, error, depth});
            };
            auto initial = evaluateCell(0.0, 1.0, 0U);
            if (!initial) {
              return Result<IntegralValues>::failure(
                  initial.status().code(),
                  std::string(initial.status().message()));
            }
            std::vector<QuadratureCell> cells{initial.value()};
            const auto aggregateCells = [&]() {
              std::array<IntegralValues, 2> sums{};
              std::array<IntegralValues, 2> corrections{};
              for (const QuadratureCell& cell : cells) {
                for (Index channel = 0U; channel < cell.estimate.size();
                     ++channel) {
                  const std::array<long double, 2> values{
                      cell.estimate.at(channel), cell.error.at(channel)};
                  for (Index kind = 0U; kind < values.size(); ++kind) {
                    const long double adjusted =
                        values.at(kind) - corrections.at(kind).at(channel);
                    const long double next =
                        sums.at(kind).at(channel) + adjusted;
                    corrections.at(kind).at(channel) =
                        (next - sums.at(kind).at(channel)) - adjusted;
                    sums.at(kind).at(channel) = next;
                  }
                }
              }
              return sums;
            };
            auto totals = aggregateCells();
            IntegralValues total = totals[0];
            IntegralValues localError = totals[1];
            IntegralValues previous = total;
            Index subdivisions = 0U;
            constexpr long double segmentToleranceFraction = 0.125L;
            while (true) {
              bool converged = subdivisions > 0U;
              for (Index channel = 0U; channel < total.size(); ++channel) {
                long double tolerance =
                    static_cast<long double>(policy.iteration.abs) +
                    (static_cast<long double>(policy.iteration.rel) *
                     std::abs(total.at(channel)));
                if (channel != 0U && total.at(channel) > 0.0) {
                  tolerance =
                      std::min(tolerance,
                               static_cast<long double>(policy.abs_log_error) *
                                   total.at(channel));
                }
                converged =
                    converged &&
                    localError.at(channel) <=
                        segmentToleranceFraction * tolerance &&
                    std::abs(total.at(channel) - previous.at(channel)) <=
                        segmentToleranceFraction * tolerance;
              }
              if (converged) {
                return Result<IntegralValues>::success(total);
              }
              if (subdivisions >= policy.iteration.max_iterations) {
                return Result<IntegralValues>::failure(
                    ErrorCode::NoConvergence,
                    "beta-gamma convolution reached its subdivision limit");
              }
              Index selected = 0U;
              long double largestScore = -1.0L;
              for (Index cellIndex = 0U; cellIndex < cells.size();
                   ++cellIndex) {
                long double score = 0.0L;
                for (Index channel = 0U; channel < total.size(); ++channel) {
                  long double tolerance =
                      static_cast<long double>(policy.iteration.abs) +
                      (static_cast<long double>(policy.iteration.rel) *
                       std::abs(total.at(channel)));
                  if (channel != 0U && total.at(channel) > 0.0) {
                    tolerance = std::min(tolerance, static_cast<long double>(
                                                        policy.abs_log_error) *
                                                        total.at(channel));
                  }
                  score = std::max(
                      score,
                      cells.at(cellIndex).error.at(channel) /
                          std::max(tolerance,
                                   std::numeric_limits<long double>::min()));
                }
                if (score > largestScore) {
                  largestScore = score;
                  selected = cellIndex;
                }
              }
              const QuadratureCell parent = cells.at(selected);
              if (parent.depth >= policy.iteration.max_iterations ||
                  parent.depth >= 64U) {
                return Result<IntegralValues>::failure(
                    ErrorCode::NoConvergence, "beta-gamma convolution reached "
                                              "its subdivision depth limit");
              }
              const Scalar middle = 0.5 * (parent.left + parent.right);
              if (!(middle > parent.left) || !(middle < parent.right)) {
                return Result<IntegralValues>::failure(
                    ErrorCode::NoConvergence, "beta-gamma convolution interval "
                                              "cannot be refined further");
              }
              previous = total;
              const QuadratureCell oldCell = cells.at(selected);
              cells.erase(
                  cells.begin() +
                  static_cast<std::vector<QuadratureCell>::difference_type>(
                      selected));
              auto leftCell =
                  evaluateCell(oldCell.left, middle, oldCell.depth + 1U);
              auto rightCell =
                  evaluateCell(middle, oldCell.right, oldCell.depth + 1U);
              if (!leftCell || !rightCell) {
                const Status& status =
                    leftCell ? rightCell.status() : leftCell.status();
                return Result<IntegralValues>::failure(
                    status.code(), std::string(status.message()));
              }
              cells.push_back(leftCell.value());
              cells.push_back(rightCell.value());
              totals = aggregateCells();
              total = totals[0];
              localError = totals[1];
              ++subdivisions;
            }
          };
          const auto powerFor = [&](Scalar exponent) -> Result<Index> {
            if (!(exponent > 0.0)) {
              return Result<Index>::failure(
                  ErrorCode::InvalidDomain,
                  "convolution singularity is not integrable");
            }
            const Scalar required = std::ceil(2.0 / exponent);
            if (!std::isfinite(required) || required > 64.0 ||
                required >
                    static_cast<Scalar>(policy.iteration.max_iterations)) {
              return Result<Index>::failure(
                  ErrorCode::NoConvergence,
                  "convolution endpoint regularization exceeds its work cap");
            }
            return Result<Index>::success(
                std::max<Index>(2U, static_cast<Index>(required)));
          };
          Scalar densityLogScale = 0.0;
          Scalar cdfLogScale = 0.0;
          Scalar survivalLogScale = 0.0;
          Scalar gammaBaseline = 0.0;
          bool useStableLargeGammaRatio = false;
          if (x < 0.5) {
            auto betaNumerator = logGamma(betaShape + 1.0, policy);
            auto densityDenominator = logGamma(betaShape + gammaShape, policy);
            auto cdfDenominator =
                logGamma(betaShape + gammaShape + 1.0, policy);
            if (!betaNumerator || !densityDenominator || !cdfDenominator) {
              const Status* status = nullptr;
              if (!betaNumerator) {
                status = &betaNumerator.status();
              } else if (!densityDenominator) {
                status = &densityDenominator.status();
              } else {
                status = &cdfDenominator.status();
              }
              return Result<TailValues>::failure(
                  status->code(), std::string(status->message()));
            }
            const Scalar logX = std::log(x);
            densityLogScale = betaNumerator.value() -
                              densityDenominator.value() +
                              (collisionExponent * logX);
            cdfLogScale = betaNumerator.value() - cdfDenominator.value() +
                          ((betaShape + gammaShape) * logX);
          } else if (x >= 2.0) {
            gammaBaseline = x - 1.0;
            auto baselineTails =
                incompleteGamma(gammaShape, gammaBaseline, policy);
            if (!baselineTails) {
              return Result<TailValues>::failure(
                  baselineTails.status().code(),
                  std::string(baselineTails.status().message()));
            }
            densityLogScale = (gammaShape - 1.0) * std::log(gammaBaseline) -
                              gammaBaseline - gammaLogNormalizer.value();
            survivalLogScale = baselineTails.value().log_survival;
            // Keep the upper-gamma ratio in a range where its asymptotic
            // correction can be evaluated without subtracting O(x) logs.
            const Scalar asymptoticThreshold =
                std::max(64.0, 64.0 + 4.0 * gammaShape * gammaShape);
            useStableLargeGammaRatio = gammaBaseline >= asymptoticThreshold;
          }
          if (!std::isfinite(densityLogScale) || !std::isfinite(cdfLogScale) ||
              !std::isfinite(survivalLogScale)) {
            return Result<TailValues>::failure(
                ErrorCode::NoConvergence,
                "beta-gamma log-integral scaling is not representable");
          }
          // Relative to x^(a-1)e^-x, the upper incomplete gamma has the
          // asymptotic series 1 + (a-1)/x + (a-1)(a-2)/x^2 + ... .
          const auto upperGammaCorrection = [&](Scalar value) {
            long double sum = 1.0L;
            long double term = 1.0L;
            long double previousMagnitude = 1.0L;
            for (Index order = 1U; order <= 32U; ++order) {
              term *= (static_cast<long double>(gammaShape) - order) /
                      static_cast<long double>(value);
              const long double magnitude = std::abs(term);
              if (magnitude >= previousMagnitude) {
                break;
              }
              sum += term;
              if (magnitude <=
                  std::numeric_limits<Scalar>::epsilon() * std::abs(sum)) {
                break;
              }
              previousMagnitude = magnitude;
            }
            return static_cast<Scalar>(std::log(sum));
          };
          const auto kernel = [&](Scalar coordinate, Scalar jacobian,
                                  bool belowOne) -> Result<IntegralValues> {
            const Scalar betaGap =
                belowOne ? (1.0 - x) + coordinate : coordinate;
            const Scalar gammaCoordinate =
                belowOne ? coordinate : (x - 1.0) + coordinate;
            auto gammaTails =
                incompleteGamma(gammaShape, gammaCoordinate, policy);
            if (!gammaTails) {
              return Result<IntegralValues>::failure(
                  gammaTails.status().code(),
                  std::string(gammaTails.status().message()));
            }
            const Scalar logScale = std::log(betaShape) +
                                    ((betaShape - 1.0) * std::log(betaGap)) +
                                    std::log(jacobian);
            const Scalar logDensityValue =
                ((gammaShape - 1.0) * std::log(gammaCoordinate)) -
                gammaCoordinate - gammaLogNormalizer.value() + logScale;
            Scalar densityLogRatio = logDensityValue - densityLogScale;
            Scalar survivalLogRatio =
                logScale + gammaTails.value().log_survival - survivalLogScale;
            if (useStableLargeGammaRatio && !belowOne) {
              const Scalar delta = coordinate;
              const Scalar gammaRatio =
                  (gammaShape - 1.0) * std::log1p(delta / gammaBaseline) -
                  delta;
              densityLogRatio = logScale + gammaRatio;
              survivalLogRatio = logScale + gammaRatio +
                                 upperGammaCorrection(gammaBaseline + delta) -
                                 upperGammaCorrection(gammaBaseline);
            }
            const long double densityValue =
                densityPole
                    ? 0.0L
                    : std::exp(static_cast<long double>(densityLogRatio));
            const long double cdfValue = std::exp(static_cast<long double>(
                logScale + gammaTails.value().log_cdf - cdfLogScale));
            const long double survivalValue =
                std::exp(static_cast<long double>(survivalLogRatio));
            if ((!densityPole && !std::isfinite(densityValue)) ||
                !std::isfinite(cdfValue) || !std::isfinite(survivalValue)) {
              return Result<IntegralValues>::failure(
                  ErrorCode::NonFiniteResult,
                  "regularized convolution integrand is not finite");
            }
            return Result<IntegralValues>::success(
                {densityValue, cdfValue, survivalValue});
          };
          IntegralValues integrated{};
          const auto addSegment = [&](Scalar lower, Scalar upper, Index power,
                                      bool belowOne) -> Status {
            auto segment = integrate(
                lower, upper, power, [&](Scalar coordinate, Scalar jacobian) {
                  return kernel(coordinate, jacobian, belowOne);
                });
            if (!segment) {
              return segment.status();
            }
            for (Index index = 0U; index < integrated.size(); ++index) {
              integrated.at(index) += segment.value().at(index);
            }
            return Status::success();
          };
          if (x < 1.0 && x > 0.0) {
            const Scalar split = std::min(x, 1.0 - x);
            auto power = powerFor(gammaShape);
            if (!power) {
              return Result<TailValues>::failure(
                  power.status().code(), std::string(power.status().message()));
            }
            Status status = addSegment(0.0, split, power.value(), true);
            if (!status.ok()) {
              return Result<TailValues>::failure(status.code(),
                                                 std::string(status.message()));
            }
            status = addSegment(split, x, 1U, true);
            if (!status.ok()) {
              return Result<TailValues>::failure(status.code(),
                                                 std::string(status.message()));
            }
          } else if (x >= 1.0) {
            const Scalar offset = x - 1.0;
            const Scalar split = std::min(1.0, offset);
            const Scalar regularizingExponent =
                x == 1.0 && !densityPole
                    ? std::min(betaShape, collisionExponent)
                    : betaShape;
            auto power = powerFor(regularizingExponent);
            if (!power) {
              return Result<TailValues>::failure(
                  power.status().code(), std::string(power.status().message()));
            }
            Status status = addSegment(0.0, split == 0.0 ? 1.0 : split,
                                       power.value(), false);
            if (!status.ok()) {
              return Result<TailValues>::failure(status.code(),
                                                 std::string(status.message()));
            }
            if (split > 0.0 && split < 1.0) {
              status = addSegment(split, 1.0, 1U, false);
              if (!status.ok()) {
                return Result<TailValues>::failure(
                    status.code(), std::string(status.message()));
              }
            }
          }
          const long double densityValue =
              densityPole ? std::numeric_limits<long double>::infinity()
                          : integrated[0];
          const Scalar logDensityValue = [&] {
            if (densityPole) {
              return positiveInfinity;
            }
            if (integrated[0] == 0.0L) {
              return negativeInfinity;
            }
            return static_cast<Scalar>(std::log(densityValue)) +
                   densityLogScale;
          }();
          Scalar const logCdf =
              integrated[1] == 0.0L
                  ? negativeInfinity
                  : static_cast<Scalar>(std::log(integrated[1])) + cdfLogScale;
          Scalar logSurvival =
              integrated[2] == 0.0L
                  ? negativeInfinity
                  : static_cast<Scalar>(std::log(integrated[2])) +
                        survivalLogScale;
          if (x < 1.0) {
            logSurvival = logAddExp(logSurvival, betaShape * std::log1p(-x));
          }
          if ((!densityPole && !std::isfinite(densityValue)) ||
              (x > 0.0 && (logDensityValue == negativeInfinity ||
                           logCdf == negativeInfinity ||
                           logSurvival == negativeInfinity))) {
            return Result<TailValues>::failure(
                ErrorCode::NoConvergence,
                "beta-gamma log integral is outside the scaled range");
          }
          const Scalar cdf = std::exp(logCdf);
          const Scalar survival = std::exp(logSurvival);
          if (std::isnan(cdf) || std::isnan(survival) || cdf < 0.0 ||
              survival < 0.0 || cdf > 1.0 + probabilitySlack ||
              survival > 1.0 + probabilitySlack) {
            return Result<TailValues>::failure(
                ErrorCode::NonFiniteResult, "beta-gamma convolution result is "
                                            "outside its probability range");
          }
          const LogTailPair logTails{logCdf, logSurvival};
          if (densityPole) {
            return Result<TailValues>::success(
                {.density_or_mass = positiveInfinity,
                 .log_density_or_mass = positiveInfinity,
                 .cdf = static_cast<Scalar>(cdf),
                 .survival = static_cast<Scalar>(survival),
                 .log_tails = logTails,
                 .on_support = true});
          }
          return finiteTailValues(logDensityValue, logTails, true);
        }
      }
      std::shared_ptr<const Distribution::State> continuousState;
      std::shared_ptr<const Distribution::State> discreteState;
      for (const auto& component : state.components) {
        if (component->family == Family::uniform) {
          continue;
        }
        if (component->measure == Measure::real_density) {
          continuousState = component;
        } else {
          discreteState = component;
        }
      }
      if (continuousState && discreteState) {
        Distribution const continuousLaw(std::move(continuousState));
        Distribution const discreteLaw(std::move(discreteState));
        if (discreteLaw.state_->family == Family::poisson &&
            continuousLaw.state_->family == Family::gamma) {
          if (x < 0.0) {
            return Result<TailValues>::success({0.0,
                                                negativeInfinity,
                                                0.0,
                                                1.0,
                                                {negativeInfinity, 0.0},
                                                false});
          }
          constexpr Scalar exactIntegerLimit = 9007199254740992.0;
          if (std::floor(x) > exactIntegerLimit) {
            return Result<TailValues>::failure(
                ErrorCode::ResourceLimit,
                "Poisson-gamma convolution coordinate exceeds exact count "
                "range");
          }
          const Scalar rate = discreteLaw.state_->parameters[0];
          Scalar logMass = rate == 0.0 ? 0.0 : -rate;
          Scalar logDensitySum = negativeInfinity;
          Scalar logCdf = negativeInfinity;
          Scalar logSurvival = negativeInfinity;
          std::uint64_t evaluations = 0U;
          for (std::uint64_t count = 0U;; ++count) {
            if (static_cast<Scalar>(count) > x) {
              break;
            }
            if (++evaluations > policy.iteration.max_evaluations) {
              return Result<TailValues>::failure(
                  ErrorCode::ResourceLimit,
                  "Poisson-gamma convolution evaluation cap exceeded");
            }
            const Scalar residual = x - static_cast<Scalar>(count);
            auto shifted = evaluate(continuousLaw, residual, policy);
            if (!shifted) {
              return Result<TailValues>::failure(
                  shifted.status().code(),
                  std::string(shifted.status().message()));
            }
            if (shifted.value().density_or_mass == positiveInfinity) {
              logDensitySum = positiveInfinity;
            } else if (logDensitySum != positiveInfinity &&
                       shifted.value().log_density_or_mass !=
                           negativeInfinity) {
              logDensitySum = logAddExp(
                  logDensitySum, logMass + shifted.value().log_density_or_mass);
            }
            if (shifted.value().log_tails.log_cdf != negativeInfinity) {
              logCdf = logAddExp(logCdf,
                                 logMass + shifted.value().log_tails.log_cdf);
            }
            if (shifted.value().log_tails.log_survival != negativeInfinity) {
              logSurvival =
                  logAddExp(logSurvival,
                            logMass + shifted.value().log_tails.log_survival);
            }
            if (rate == 0.0) {
              break;
            }
            logMass +=
                std::log(rate) - std::log(static_cast<Scalar>(count + 1U));
          }
          if (++evaluations > policy.iteration.max_evaluations) {
            return Result<TailValues>::failure(
                ErrorCode::ResourceLimit,
                "Poisson-gamma convolution evaluation cap exceeded");
          }
          auto countTail = evaluate(discreteLaw, std::floor(x), policy);
          if (!countTail) {
            return Result<TailValues>::failure(
                countTail.status().code(),
                std::string(countTail.status().message()));
          }
          logSurvival =
              logAddExp(logSurvival, countTail.value().log_tails.log_survival);
          if (logDensitySum == positiveInfinity) {
            return Result<TailValues>::success({positiveInfinity,
                                                positiveInfinity,
                                                std::exp(logCdf),
                                                std::exp(logSurvival),
                                                {logCdf, logSurvival},
                                                true});
          }
          return finiteTailValues(logDensitySum, {logCdf, logSurvival}, true);
        }
        if (discreteLaw.state_->family == Family::poisson &&
            continuousLaw.state_->family == Family::normal) {
          const Scalar rate = discreteLaw.state_->parameters[0];
          const Scalar logMaximumDensity =
              -std::log(continuousLaw.state_->parameters[1]) -
              0.5 * std::log(2.0 * std::numbers::pi_v<Scalar>);
          Scalar logDensitySum = negativeInfinity;
          Scalar logCdf = negativeInfinity;
          Scalar logSurvival = negativeInfinity;
          Scalar logMass = -rate;
          std::uint64_t evaluations = 0U;
          for (std::uint64_t count = 0U;; ++count) {
            if (count > 9007199254740992ULL ||
                ++evaluations > policy.iteration.max_evaluations) {
              return Result<TailValues>::failure(
                  ErrorCode::ResourceLimit,
                  "Poisson-normal convolution evaluation cap exceeded");
            }
            auto shifted =
                evaluate(continuousLaw, x - static_cast<Scalar>(count), policy);
            if (!shifted) {
              return Result<TailValues>::failure(
                  shifted.status().code(),
                  std::string(shifted.status().message()));
            }
            if (shifted.value().log_density_or_mass != negativeInfinity) {
              logDensitySum = logAddExp(
                  logDensitySum, logMass + shifted.value().log_density_or_mass);
            }
            logCdf =
                logAddExp(logCdf, logMass + shifted.value().log_tails.log_cdf);
            logSurvival = logAddExp(
                logSurvival, logMass + shifted.value().log_tails.log_survival);

            if (++evaluations > policy.iteration.max_evaluations) {
              return Result<TailValues>::failure(
                  ErrorCode::ResourceLimit,
                  "Poisson-normal convolution evaluation cap exceeded");
            }
            auto remaining =
                evaluate(discreteLaw, static_cast<Scalar>(count), policy);
            if (!remaining) {
              return Result<TailValues>::failure(
                  remaining.status().code(),
                  std::string(remaining.status().message()));
            }
            const Scalar logRemainder =
                remaining.value().log_tails.log_survival;
            const auto logTarget = [&](Scalar current) {
              const Scalar absoluteRelative =
                  logAddExp(std::log(policy.iteration.abs),
                            std::log(policy.iteration.rel) + current);
              const Scalar relativeLog =
                  std::log(policy.abs_log_error) + current;
              return std::min(absoluteRelative, relativeLog);
            };
            const bool densityResolved =
                logRemainder + logMaximumDensity <= logTarget(logDensitySum);
            const bool cdfResolved = logRemainder <= logTarget(logCdf);
            const bool survivalResolved =
                logRemainder <= logTarget(logSurvival);
            if (densityResolved && cdfResolved && survivalResolved) {
              return finiteTailValues(logDensitySum, {logCdf, logSurvival},
                                      true);
            }
            if (rate == 0.0) {
              return finiteTailValues(logDensitySum, {logCdf, logSurvival},
                                      true);
            }
            logMass +=
                std::log(rate) - std::log(static_cast<Scalar>(count + 1U));
          }
        }
        auto atoms =
            finiteAtoms(discreteLaw, {1024U, std::size_t{16U} * 1024U * 1024U,
                                      policy.iteration.max_evaluations});
        if (!atoms) {
          return Result<TailValues>::failure(
              atoms.status().code(), std::string(atoms.status().message()));
        }
        Scalar logDensitySum = negativeInfinity;
        Scalar logCdf = negativeInfinity;
        Scalar logSurvival = negativeInfinity;
        bool supported = false;
        std::uint64_t evaluations = 0U;
        for (const auto& [atom, mass] : atoms.value()) {
          if (++evaluations > policy.iteration.max_evaluations) {
            return Result<TailValues>::failure(
                ErrorCode::ResourceLimit,
                "convolution evaluation cap exceeded");
          }
          auto shifted = evaluate(continuousLaw, x - atom, policy);
          if (!shifted) {
            return Result<TailValues>::failure(
                shifted.status().code(),
                std::string(shifted.status().message()));
          }
          const Scalar logMass = std::log(mass);
          if (shifted.value().density_or_mass == positiveInfinity) {
            logDensitySum = positiveInfinity;
          } else if (logDensitySum != positiveInfinity) {
            logDensitySum = logAddExp(
                logDensitySum, logMass + shifted.value().log_density_or_mass);
          }
          logCdf =
              logAddExp(logCdf, logMass + shifted.value().log_tails.log_cdf);
          logSurvival = logAddExp(
              logSurvival, logMass + shifted.value().log_tails.log_survival);
          supported = supported || shifted.value().on_support;
        }
        tails = {logCdf, logSurvival};
        if (logDensitySum == positiveInfinity) {
          return Result<TailValues>::success(
              {positiveInfinity, positiveInfinity, std::exp(logCdf),
               std::exp(logSurvival), tails, supported});
        }
        return finiteTailValues(logDensitySum, tails, supported);
      }
      if (state.components.size() == 2U &&
          state.components[0]->family == Family::uniform &&
          state.components[1]->family == Family::uniform) {
        const auto& left = state.components[0]->parameters;
        const auto& right = state.components[1]->parameters;
        const long double a = left[0];
        const long double b = left[1];
        const long double c = right[0];
        const long double d = right[1];
        const auto cdfOfUniformSum =
            [](long double firstLower, long double firstUpper,
               long double secondLower, long double secondUpper,
               long double coordinate) {
              const auto positiveSquare = [](long double value) {
                return value > 0.0L ? value * value : 0.0L;
              };
              const long double numerator =
                  positiveSquare(coordinate - firstLower - secondLower) -
                  positiveSquare(coordinate - firstUpper - secondLower) -
                  positiveSquare(coordinate - firstLower - secondUpper) +
                  positiveSquare(coordinate - firstUpper - secondUpper);
              const long double denominator = 2.0L * (firstUpper - firstLower) *
                                              (secondUpper - secondLower);
              return std::clamp(numerator / denominator, 0.0L, 1.0L);
            };
        const long double lower = a + c;
        const long double upper = b + d;
        const long double coordinate = x;
        const long double cdf = cdfOfUniformSum(a, b, c, d, coordinate);
        const long double survival =
            cdfOfUniformSum(-b, -a, -d, -c, -coordinate);
        const long double overlap = std::max(
            0.0L, std::min(b, coordinate - c) - std::max(a, coordinate - d));
        const long double density = overlap / ((b - a) * (d - c));
        tails = {cdf == 0.0L ? negativeInfinity
                             : static_cast<Scalar>(std::log(cdf)),
                 survival == 0.0L ? negativeInfinity
                                  : static_cast<Scalar>(std::log(survival))};
        logDensity = density == 0.0L ? negativeInfinity
                                     : static_cast<Scalar>(std::log(density));
        onSupport = coordinate >= lower && coordinate <= upper;
        return finiteTailValues(logDensity, tails, onSupport);
      }
      return Result<TailValues>::failure(
          ErrorCode::InvalidDomain,
          "convolution expression has no qualified evaluator");
    }
    [[fallthrough]];
  }
  case Family::empirical: {
    const auto upper =
        std::upper_bound(state.atoms.begin(), state.atoms.end(), x);
    const Index upperIndex = static_cast<Index>(upper - state.atoms.begin());
    const Scalar cdf =
        upperIndex == 0U ? 0.0 : state.cumulativeMasses[upperIndex - 1U];
    const Scalar survival = upperIndex == state.atoms.size()
                                ? 0.0
                                : state.survivalMasses[upperIndex];
    Scalar mass = 0.0;
    const auto exact =
        std::lower_bound(state.atoms.begin(), state.atoms.end(), x);
    if (exact != state.atoms.end() && *exact == x) {
      mass = state.masses[static_cast<Index>(exact - state.atoms.begin())];
    }
    tails = {cdf == 0.0 ? negativeInfinity : std::log(cdf),
             survival == 0.0 ? negativeInfinity : std::log(survival)};
    logDensity = mass == 0.0 ? negativeInfinity : std::log(mass);
    return Result<TailValues>::success(
        {mass, logDensity, cdf, survival, tails, mass > 0.0});
  }
  case Family::mixture: {
    Scalar logDensitySum = negativeInfinity;
    Scalar logCdf = negativeInfinity;
    Scalar logSurvival = negativeInfinity;
    bool supported = false;
    std::uint64_t evaluations = 0U;
    for (Index index = 0U; index < state.components.size(); ++index) {
      if (state.weights[index] == 0.0) {
        continue;
      }
      if (++evaluations > policy.iteration.max_evaluations) {
        return Result<TailValues>::failure(ErrorCode::ResourceLimit,
                                           "mixture evaluation cap exceeded");
      }
      Distribution const component(state.components[index]);
      auto value = evaluate(component, x, policy);
      if (!value) {
        return Result<TailValues>::failure(
            value.status().code(), std::string(value.status().message()));
      }
      const Scalar logWeight = std::log(state.weights[index]);
      if (value.value().density_or_mass == positiveInfinity) {
        logDensitySum = positiveInfinity;
      } else if (logDensitySum != positiveInfinity) {
        logDensitySum = logAddExp(
            logDensitySum, logWeight + value.value().log_density_or_mass);
      }
      logCdf = logAddExp(logCdf, logWeight + value.value().log_tails.log_cdf);
      logSurvival = logAddExp(logSurvival,
                              logWeight + value.value().log_tails.log_survival);
      supported = supported || value.value().on_support;
    }
    tails = {logCdf, logSurvival};
    if (logDensitySum == positiveInfinity) {
      return Result<TailValues>::success(
          {positiveInfinity, positiveInfinity, std::exp(logCdf),
           std::exp(logSurvival), tails, supported});
    }
    return finiteTailValues(logDensitySum, tails, supported);
  }
  default:
    return Result<TailValues>::failure(
        ErrorCode::InvalidDomain,
        "evaluation for this distribution family is not implemented yet");
  }

  return finiteTailValues(logDensity, tails, onSupport);
}

Result<Scalar>
// NOLINTNEXTLINE(misc-no-recursion): nested truncated laws have a depth cap.
DistributionEngine::intervalLogProbability(const Distribution& law,
                                           IntervalSupport interval,
                                           ProbabilityPolicy policy) {
  const Scalar negativeInfinity = -std::numeric_limits<Scalar>::infinity();
  if (std::isnan(interval.lower) || std::isnan(interval.upper)) {
    return Result<Scalar>::failure(ErrorCode::NonFiniteInput,
                                   "interval endpoints cannot be NaN");
  }
  if (interval.lower > interval.upper ||
      interval.lower == std::numeric_limits<Scalar>::infinity() ||
      interval.upper == -std::numeric_limits<Scalar>::infinity()) {
    return Result<Scalar>::failure(ErrorCode::InvalidDomain,
                                   "interval endpoints are reversed");
  }
  if (!std::isfinite(policy.iteration.abs) ||
      !std::isfinite(policy.iteration.rel) ||
      !std::isfinite(policy.abs_log_error) || policy.iteration.abs <= 0.0 ||
      policy.iteration.rel <= 0.0 || policy.abs_log_error <= 0.0) {
    return Result<Scalar>::failure(ErrorCode::InvalidTolerance,
                                   "probability tolerances must be positive");
  }
  if (policy.iteration.max_iterations == 0U ||
      policy.iteration.max_evaluations == 0U) {
    return Result<Scalar>::failure(ErrorCode::InvalidDomain,
                                   "probability work limits must be positive");
  }
  if (interval.lower == interval.upper &&
      !(interval.include_lower && interval.include_upper)) {
    return Result<Scalar>::success(negativeInfinity);
  }
  auto lower = evaluate(law, interval.lower, policy);
  auto upper = evaluate(law, interval.upper, policy);
  if (!lower || !upper) {
    const Status& status = !lower ? lower.status() : upper.status();
    return Result<Scalar>::failure(status.code(),
                                   std::string(status.message()));
  }
  const bool discrete = law.measure() == Measure::integer_mass ||
                        law.measure() == Measure::atom_mass;
  const Scalar lowerMass =
      discrete ? lower.value().log_density_or_mass : negativeInfinity;
  const Scalar upperMass =
      discrete ? upper.value().log_density_or_mass : negativeInfinity;
  const Scalar leftBelow =
      interval.include_lower && discrete
          ? logDifference(lower.value().log_tails.log_cdf, lowerMass)
          : lower.value().log_tails.log_cdf;
  const Scalar leftAbove =
      interval.include_upper || !discrete
          ? upper.value().log_tails.log_cdf
          : logDifference(upper.value().log_tails.log_cdf, upperMass);
  const Scalar rightBelow =
      interval.include_lower && discrete
          ? logAddExp(lower.value().log_tails.log_survival, lowerMass)
          : lower.value().log_tails.log_survival;
  const Scalar rightAbove =
      interval.include_upper || !discrete
          ? upper.value().log_tails.log_survival
          : logAddExp(upper.value().log_tails.log_survival, upperMass);
  const Scalar fromLowerTail = logDifference(leftAbove, leftBelow);
  const Scalar fromUpperTail = logDifference(rightBelow, rightAbove);
  if (leftAbove <= rightBelow) {
    return Result<Scalar>::success(
        fromLowerTail != negativeInfinity ? fromLowerTail : fromUpperTail);
  }
  return Result<Scalar>::success(
      fromUpperTail != negativeInfinity ? fromUpperTail : fromLowerTail);
}

Result<Distribution> DistributionEngine::truncated(const Distribution& base,
                                                   IntervalSupport interval,
                                                   ProbabilityPolicy policy,
                                                   Limits limits) {
  if (!base.state_) {
    return Result<Distribution>::failure(ErrorCode::InvalidDomain,
                                         "base distribution is moved from");
  }
  constexpr std::size_t byteCap = std::size_t{16U} * 1024U * 1024U;
  const std::size_t stateBytes =
      sizeof(Distribution::State) +
      sizeof(std::shared_ptr<const Distribution::State>) + 3U * sizeof(Scalar);
  if (limits.elements < 4U || std::min(limits.bytes, byteCap) < stateBytes ||
      limits.evaluations == 0U) {
    return Result<Distribution>::failure(
        ErrorCode::ResourceLimit, "truncated law exceeds configured limits");
  }
  if (base.state_->depth >= 32U) {
    return Result<Distribution>::failure(
        ErrorCode::ResourceLimit, "truncated law nesting exceeds depth 32");
  }
  policy.iteration.max_evaluations = static_cast<Index>(std::min<std::uint64_t>(
      policy.iteration.max_evaluations, limits.evaluations));
  auto logMass = intervalLogProbability(base, interval, policy);
  if (!logMass) {
    return Result<Distribution>::failure(
        logMass.status().code(), std::string(logMass.status().message()));
  }
  if (logMass.value() == -std::numeric_limits<Scalar>::infinity()) {
    return Result<Distribution>::failure(
        ErrorCode::InvalidDomain,
        "truncation interval has zero probability under the base law");
  }
  auto state = std::make_shared<Distribution::State>();
  state->family = Family::truncated;
  state->measure = base.state_->measure;
  state->components.push_back(base.state_);
  state->parameters = {interval.lower, interval.upper, logMass.value()};
  state->integerParameter =
      (interval.include_lower ? 1U : 0U) | (interval.include_upper ? 2U : 0U);
  state->depth = base.state_->depth + 1U;
  return Result<Distribution>::success(Distribution(std::move(state)));
}

// Composite moments recurse only through the explicitly depth-capped law tree.
// NOLINTNEXTLINE(misc-no-recursion)
Result<Scalar> DistributionEngine::rawMoment(const Distribution& law,
                                             Index order,
                                             ProbabilityPolicy policy) {
  if (!law.state_) {
    return Result<Scalar>::failure(ErrorCode::InvalidDomain,
                                   "distribution is moved from");
  }
  if (!std::isfinite(policy.iteration.abs) ||
      !std::isfinite(policy.iteration.rel) ||
      !std::isfinite(policy.abs_log_error) || policy.iteration.abs <= 0.0 ||
      policy.iteration.rel <= 0.0 || policy.abs_log_error <= 0.0) {
    return Result<Scalar>::failure(ErrorCode::InvalidTolerance,
                                   "probability tolerances must be positive");
  }
  if (policy.iteration.max_evaluations == 0U ||
      order > policy.iteration.max_evaluations) {
    return Result<Scalar>::failure(ErrorCode::ResourceLimit,
                                   "raw-moment order exceeds its work limit");
  }
  if (order == 0U) {
    return Result<Scalar>::success(1.0);
  }
  const Distribution::State& state = *law.state_;
  const auto& p = state.parameters;
  const auto k = static_cast<long double>(order);
  const long double pi = std::numbers::pi_v<long double>;
  const auto maxValue =
      static_cast<long double>(std::numeric_limits<Scalar>::max());
  auto finish = [&](long double value) -> Result<Scalar> {
    if (!std::isfinite(value) || std::abs(value) > maxValue) {
      return Result<Scalar>::failure(ErrorCode::NonFiniteResult,
                                     "raw moment is outside scalar range");
    }
    const auto converted = static_cast<Scalar>(value);
    if (!std::isfinite(converted)) {
      return Result<Scalar>::failure(ErrorCode::NonFiniteResult,
                                     "raw moment conversion is not finite");
    }
    return Result<Scalar>::success(converted);
  };
  auto binomialCoefficient = [](Index n, Index r) {
    if (r > n) {
      return 0.0L;
    }
    r = std::min(r, n - r);
    long double value = 1.0L;
    for (Index index = 1U; index <= r; ++index) {
      value *= static_cast<long double>(n - r + index) /
               static_cast<long double>(index);
    }
    return value;
  };
  auto finiteSupportMoment = [&]() -> Result<Scalar> {
    const Limits limits{
        std::min<Index>(1024U, policy.iteration.max_evaluations),
        std::size_t{16U} * 1024U * 1024U, policy.iteration.max_evaluations};
    auto support = finiteAtoms(law, limits);
    if (!support) {
      return Result<Scalar>::failure(support.status().code(),
                                     std::string(support.status().message()));
    }
    long double total = 0.0L;
    for (const auto& [atom, mass] : support.value()) {
      total += static_cast<long double>(mass) *
               std::pow(static_cast<long double>(atom), k);
    }
    return finish(total);
  };
  auto factorialMomentSum =
      [&](const std::function<long double(Index)>& factorialMoment)
      -> Result<Scalar> {
    std::vector<long double> stirling(order + 1U, 0.0L);
    stirling[0] = 1.0L;
    for (Index row = 1U; row <= order; ++row) {
      for (Index column = row; column > 0U; --column) {
        stirling[column] = stirling[column] * static_cast<long double>(column) +
                           stirling[column - 1U];
      }
      stirling[0] = 0.0L;
    }
    long double value = 0.0L;
    for (Index column = 1U; column <= order; ++column) {
      value += stirling[column] * factorialMoment(column);
    }
    return finish(value);
  };

  switch (state.family) {
  case Family::normal: {
    std::vector<long double> moments(order + 1U, 0.0L);
    moments[0] = 1.0L;
    moments[1] = static_cast<long double>(p[0]);
    const long double meanValue = p[0];
    const long double varianceValue =
        static_cast<long double>(p[1]) * static_cast<long double>(p[1]);
    for (Index index = 2U; index <= order; ++index) {
      moments[index] = meanValue * moments[index - 1U] +
                       static_cast<long double>(index - 1U) * varianceValue *
                           moments[index - 2U];
    }
    return finish(moments[order]);
  }
  case Family::student_t: {
    const long double nu = p[0];
    if (k >= nu) {
      return Result<Scalar>::failure(
          ErrorCode::InvalidDomain,
          "requested Student-t raw moment does not exist absolutely");
    }
    const long double location = p[1];
    const long double scale = p[2];
    long double raw = 0.0L;
    for (Index central = 0U; central <= order; central += 2U) {
      const long double halfOrder = static_cast<long double>(central) / 2.0L;
      const long double centralMoment =
          std::pow(scale, static_cast<int>(central)) * std::pow(nu, halfOrder) *
          std::tgamma(halfOrder + 0.5L) *
          std::tgamma((nu - static_cast<long double>(central)) / 2.0L) /
          (std::sqrt(pi) * std::tgamma(nu / 2.0L));
      raw += binomialCoefficient(order, central) *
             std::pow(location, static_cast<int>(order - central)) *
             centralMoment;
    }
    return finish(raw);
  }
  case Family::laplace: {
    long double raw = 0.0L;
    const long double scale = p[1];
    for (Index central = 0U; central <= order; central += 2U) {
      long double factorial = 1.0L;
      for (Index index = 2U; index <= central; ++index) {
        factorial *= static_cast<long double>(index);
      }
      raw += binomialCoefficient(order, central) *
             std::pow(static_cast<long double>(p[0]),
                      static_cast<int>(order - central)) *
             factorial * std::pow(scale, static_cast<int>(central));
    }
    return finish(raw);
  }
  case Family::logistic: {
    long double raw = 0.0L;
    const long double scale = p[1];
    for (Index central = 0U; central <= order; central += 2U) {
      const Index halfOrder = central / 2U;
      long double factorial = 1.0L;
      for (Index index = 2U; index <= central; ++index) {
        factorial *= static_cast<long double>(index);
      }
      long double zeta = 1.0L;
      if (central >= 4U) {
        for (Index index = 2U; index <= 100U; ++index) {
          zeta += std::pow(static_cast<long double>(index),
                           -static_cast<int>(central));
        }
      } else if (central == 2U) {
        zeta = pi * pi / 6.0L;
      }
      const long double centralMoment =
          central == 0U
              ? 1.0L
              : 2.0L * (1.0L - std::pow(2.0L, 1.0L - 2.0L * halfOrder)) *
                    factorial * zeta *
                    std::pow(scale, static_cast<int>(central));
      raw += binomialCoefficient(order, central) *
             std::pow(static_cast<long double>(p[0]),
                      static_cast<int>(order - central)) *
             centralMoment;
    }
    return finish(raw);
  }
  case Family::lognormal:
    return finish(
        std::exp(k * static_cast<long double>(p[0]) +
                 0.5L * k * k * static_cast<long double>(p[1]) * p[1]));
  case Family::gamma:
  case Family::chi_square: {
    const long double shape = state.family == Family::gamma
                                  ? static_cast<long double>(p[0])
                                  : static_cast<long double>(p[0]) / 2.0L;
    const long double scale = state.family == Family::gamma ? p[1] : 2.0L;
    long double value = std::pow(scale, static_cast<int>(order));
    for (Index index = 0U; index < order; ++index) {
      value *= shape + static_cast<long double>(index);
    }
    return finish(value);
  }
  case Family::beta: {
    const long double a = p[0];
    const long double b = p[1];
    long double value = 1.0L;
    for (Index index = 0U; index < order; ++index) {
      value *= (a + static_cast<long double>(index)) /
               (a + b + static_cast<long double>(index));
    }
    return finish(value);
  }
  case Family::weibull: {
    const long double shape = p[0];
    const long double scale = p[1];
    return finish(std::pow(scale, static_cast<int>(order)) *
                  std::tgamma(1.0L + k / shape));
  }
  case Family::skew_normal: {
    const long double delta = static_cast<long double>(p[2]) /
                              std::hypot(1.0L, static_cast<long double>(p[2]));
    const long double normalScale = std::sqrt(1.0L - delta * delta);
    long double raw = 0.0L;
    for (Index standardizedOrder = 0U; standardizedOrder <= order;
         ++standardizedOrder) {
      long double standardizedMoment = 0.0L;
      for (Index absoluteOrder = 0U; absoluteOrder <= standardizedOrder;
           ++absoluteOrder) {
        const Index gaussianOrder = standardizedOrder - absoluteOrder;
        if (gaussianOrder % 2U != 0U) {
          continue;
        }
        const long double absoluteMoment =
            std::pow(2.0L, static_cast<long double>(absoluteOrder) / 2.0L) *
            std::tgamma((static_cast<long double>(absoluteOrder) + 1.0L) /
                        2.0L) /
            std::sqrt(pi);
        long double gaussianMoment = 1.0L;
        for (Index index = 2U; index <= gaussianOrder; index += 2U) {
          gaussianMoment *= static_cast<long double>(index - 1U);
        }
        standardizedMoment +=
            binomialCoefficient(standardizedOrder, absoluteOrder) *
            std::pow(delta, static_cast<int>(absoluteOrder)) * absoluteMoment *
            std::pow(normalScale, static_cast<int>(gaussianOrder)) *
            gaussianMoment;
      }
      raw += binomialCoefficient(order, standardizedOrder) *
             std::pow(static_cast<long double>(p[0]),
                      static_cast<int>(order - standardizedOrder)) *
             std::pow(static_cast<long double>(p[1]),
                      static_cast<int>(standardizedOrder)) *
             standardizedMoment;
    }
    return finish(raw);
  }
  case Family::uniform: {
    const long double lower = p[0];
    const long double width = static_cast<long double>(p[1]) - p[0];
    long double result = 0.0L;
    for (Index term = 0U; term <= order; ++term) {
      result += binomialCoefficient(order, term) *
                std::pow(lower, static_cast<int>(order - term)) *
                std::pow(width, static_cast<int>(term)) /
                static_cast<long double>(term + 1U);
    }
    return finish(result);
  }
  case Family::triangular: {
    const long double lower = p[0];
    const long double width = static_cast<long double>(p[2]) - p[0];
    const long double modeFraction =
        (static_cast<long double>(p[1]) - p[0]) / width;
    long double result = 0.0L;
    for (Index power = 0U; power <= order; ++power) {
      long double tMoment = 0.0L;
      if (modeFraction > 0.0L) {
        tMoment += 2.0L * std::pow(modeFraction, static_cast<int>(power + 1U)) /
                   static_cast<long double>(power + 2U);
      }
      if (modeFraction < 1.0L) {
        tMoment +=
            2.0L / (1.0L - modeFraction) *
            ((1.0L - std::pow(modeFraction, static_cast<int>(power + 1U))) /
                 static_cast<long double>(power + 1U) -
             (1.0L - std::pow(modeFraction, static_cast<int>(power + 2U))) /
                 static_cast<long double>(power + 2U));
      }
      result += binomialCoefficient(order, power) *
                std::pow(lower, static_cast<int>(order - power)) *
                std::pow(width, static_cast<int>(power)) * tMoment;
    }
    return finish(result);
  }
  case Family::truncated_normal: {
    const long double meanValue = p[0];
    const long double sd = p[1];
    const long double lower = (static_cast<long double>(p[2]) - meanValue) / sd;
    const long double upper = (static_cast<long double>(p[3]) - meanValue) / sd;
    auto low = normalLogTails(static_cast<Scalar>(lower), policy);
    auto high = normalLogTails(static_cast<Scalar>(upper), policy);
    if (!low || !high) {
      const Status& status = !low ? low.status() : high.status();
      return Result<Scalar>::failure(status.code(),
                                     std::string(status.message()));
    }
    Scalar logZ = -std::numeric_limits<Scalar>::infinity();
    if (lower >= 0.0L) {
      logZ = logDifference(low.value().log_survival, high.value().log_survival);
    } else if (upper <= 0.0L) {
      logZ = logDifference(high.value().log_cdf, low.value().log_cdf);
    } else {
      const long double outside =
          std::exp(static_cast<long double>(low.value().log_cdf)) +
          std::exp(static_cast<long double>(high.value().log_survival));
      logZ = static_cast<Scalar>(std::log1p(-outside));
    }
    const long double z = std::exp(static_cast<long double>(logZ));
    std::vector<long double> integrals(order + 1U, 0.0L);
    const auto phi = [](long double value) {
      return std::isfinite(value)
                 ? std::exp(-0.5L * value * value) /
                       std::sqrt(2.0L * std::numbers::pi_v<long double>)
                 : 0.0L;
    };
    integrals[0] = z;
    if (order >= 1U) {
      integrals[1] = phi(lower) - phi(upper);
    }
    for (Index index = 2U; index <= order; ++index) {
      const long double lowerTerm =
          std::isfinite(lower)
              ? std::pow(lower, static_cast<int>(index - 1U)) * phi(lower)
              : 0.0L;
      const long double upperTerm =
          std::isfinite(upper)
              ? std::pow(upper, static_cast<int>(index - 1U)) * phi(upper)
              : 0.0L;
      integrals[index] =
          lowerTerm - upperTerm +
          static_cast<long double>(index - 1U) * integrals[index - 2U];
    }
    long double raw = 0.0L;
    for (Index zOrder = 0U; zOrder <= order; ++zOrder) {
      raw += binomialCoefficient(order, zOrder) *
             std::pow(meanValue, static_cast<int>(order - zOrder)) *
             std::pow(sd, static_cast<int>(zOrder)) * integrals[zOrder] / z;
    }
    return finish(raw);
  }
  case Family::bernoulli:
  case Family::empirical:
    return finiteSupportMoment();
  case Family::finite_convolution: {
    if (state.components.empty()) {
      return finiteSupportMoment();
    }
    std::vector<long double> moments(order + 1U, 0.0L);
    moments[0] = 1.0L;
    std::uint64_t operations = 0U;
    for (const auto& componentState : state.components) {
      Distribution const component(componentState);
      std::vector<long double> next(order + 1U, 0.0L);
      for (Index degree = 0U; degree <= order; ++degree) {
        auto componentMoment = rawMoment(component, degree, policy);
        if (!componentMoment) {
          return Result<Scalar>::failure(
              componentMoment.status().code(),
              std::string(componentMoment.status().message()));
        }
        for (Index existingOrder = 0U; existingOrder <= order - degree;
             ++existingOrder) {
          if (++operations > policy.iteration.max_evaluations) {
            return Result<Scalar>::failure(
                ErrorCode::ResourceLimit,
                "convolution moment evaluation cap exceeded");
          }
          next[degree + existingOrder] +=
              binomialCoefficient(degree + existingOrder, degree) *
              static_cast<long double>(componentMoment.value()) *
              moments[existingOrder];
        }
      }
      moments = std::move(next);
    }
    return finish(moments[order]);
  }
  case Family::binomial: {
    const auto n = static_cast<long double>(state.integerParameter);
    const long double probability = p[1];
    return factorialMomentSum([&](Index degree) {
      if (degree > state.integerParameter) {
        return 0.0L;
      }
      long double falling = 1.0L;
      for (Index offset = 0U; offset < degree; ++offset) {
        falling *= n - static_cast<long double>(offset);
      }
      return falling * std::pow(probability, static_cast<int>(degree));
    });
  }
  case Family::poisson: {
    const long double rate = p[0];
    return factorialMomentSum(
        [&](Index degree) { return std::pow(rate, static_cast<int>(degree)); });
  }
  case Family::negative_binomial: {
    const long double successes = p[0];
    const long double probability = p[1];
    return factorialMomentSum([&](Index degree) {
      long double const rising =
          std::tgamma(successes + static_cast<long double>(degree)) /
          std::tgamma(successes);
      return rising * std::pow((1.0L - probability) / probability,
                               static_cast<int>(degree));
    });
  }
  case Family::fisher_f: {
    const long double numerator = p[0];
    const long double denominator = p[1];
    if (denominator <= 2.0L * k) {
      return Result<Scalar>::failure(ErrorCode::InvalidDomain,
                                     "requested F moment does not exist");
    }
    const long double ratio = denominator / numerator;
    return finish(
        std::pow(ratio, static_cast<int>(order)) *
        std::tgamma(numerator / 2.0L + k) / std::tgamma(numerator / 2.0L) *
        std::tgamma(denominator / 2.0L - k) / std::tgamma(denominator / 2.0L));
  }
  case Family::truncated: {
    const Scalar lower = p[0];
    const Scalar upper = p[1];
    const Scalar logNormalizer = p[2];
    const bool includeLower = (state.integerParameter & 1U) != 0U;
    const bool includeUpper = (state.integerParameter & 2U) != 0U;
    const Distribution base(state.components.front());
    if (state.measure == Measure::integer_mass ||
        state.measure == Measure::atom_mass) {
      const Limits supportLimits{
          std::min<std::size_t>(1024U, policy.iteration.max_evaluations),
          std::size_t{16U} * 1024U * 1024U, policy.iteration.max_evaluations};
      auto support = finiteAtoms(base, supportLimits);
      if (!support) {
        return Result<Scalar>::failure(support.status().code(),
                                       std::string(support.status().message()));
      }
      long double total = 0.0L;
      for (const auto& [atom, mass] : support.value()) {
        const bool aboveLower = atom > lower || (includeLower && atom == lower);
        const bool belowUpper = atom < upper || (includeUpper && atom == upper);
        if (!aboveLower || !belowUpper || mass <= 0.0) {
          continue;
        }
        total += std::exp(std::log(static_cast<long double>(mass)) -
                          static_cast<long double>(logNormalizer)) *
                 std::pow(static_cast<long double>(atom), k);
      }
      return finish(total);
    }
    if (base.family() == Family::truncated_normal) {
      const auto& baseParameters = base.state_->parameters;
      const Scalar clippedLower = std::max(lower, baseParameters[2]);
      const Scalar clippedUpper = std::min(upper, baseParameters[3]);
      auto clipped = truncatedNormal(baseParameters[0], baseParameters[1],
                                     clippedLower, clippedUpper);
      if (!clipped) {
        return Result<Scalar>::failure(clipped.status().code(),
                                       std::string(clipped.status().message()));
      }
      return rawMoment(clipped.value(), order, policy);
    }
    Scalar integrationLower = lower;
    Scalar integrationUpper = upper;
    switch (base.family()) {
    case Family::uniform:
      integrationLower = std::max(integrationLower, base.state_->parameters[0]);
      integrationUpper = std::min(integrationUpper, base.state_->parameters[1]);
      break;
    case Family::triangular:
      integrationLower = std::max(integrationLower, base.state_->parameters[0]);
      integrationUpper = std::min(integrationUpper, base.state_->parameters[2]);
      break;
    case Family::beta:
      integrationLower = std::max(integrationLower, 0.0);
      integrationUpper = std::min(integrationUpper, 1.0);
      break;
    default:
      break;
    }
    if (!std::isfinite(integrationLower) || !std::isfinite(integrationUpper)) {
      return Result<Scalar>::failure(
          ErrorCode::NoConvergence,
          "unbounded truncated raw moment needs a qualified tail enclosure");
    }
    if (integrationLower >= integrationUpper) {
      return Result<Scalar>::failure(
          ErrorCode::InvalidDomain,
          "truncated law has no representable continuous integration range");
    }
    auto integrand = [&](Scalar coordinate) -> Result<long double> {
      Scalar evaluationPoint = coordinate;
      if (coordinate == integrationLower) {
        evaluationPoint = std::nextafter(integrationLower, integrationUpper);
      } else if (coordinate == integrationUpper) {
        evaluationPoint = std::nextafter(integrationUpper, integrationLower);
      }
      auto value = evaluate(base, evaluationPoint, policy);
      if (!value) {
        return Result<long double>::failure(
            value.status().code(), std::string(value.status().message()));
      }
      if (value.value().log_density_or_mass ==
          -std::numeric_limits<Scalar>::infinity()) {
        return Result<long double>::success(0.0L);
      }
      const long double density =
          std::exp(static_cast<long double>(value.value().log_density_or_mass) -
                   static_cast<long double>(logNormalizer));
      const long double moment =
          std::pow(static_cast<long double>(coordinate), k);
      const long double result = density * moment;
      if (!std::isfinite(result)) {
        return Result<long double>::failure(
            ErrorCode::NonFiniteResult,
            "truncated raw-moment integrand is outside range");
      }
      return Result<long double>::success(result);
    };
    auto leftValue = integrand(integrationLower);
    auto middleValue = integrand(integrationLower +
                                 0.5 * (integrationUpper - integrationLower));
    auto rightValue = integrand(integrationUpper);
    if (!leftValue) {
      return Result<Scalar>::failure(leftValue.status().code(),
                                     std::string(leftValue.status().message()));
    }
    if (!middleValue) {
      return Result<Scalar>::failure(
          middleValue.status().code(),
          std::string(middleValue.status().message()));
    }
    if (!rightValue) {
      return Result<Scalar>::failure(
          rightValue.status().code(),
          std::string(rightValue.status().message()));
    }
    std::uint64_t evaluations = 3U;
    const long double whole =
        static_cast<long double>(integrationUpper - integrationLower) *
        (leftValue.value() + 4.0L * middleValue.value() + rightValue.value()) /
        6.0L;
    std::function<Result<long double>(Scalar, Scalar, long double, long double,
                                      long double, long double, Scalar, Index)>
        integrate;
    integrate = [&](Scalar a, Scalar b, long double fa, long double fm,
                    long double fb, long double estimate, Scalar tolerance,
                    Index depth) -> Result<long double> {
      if (depth >= policy.iteration.max_iterations ||
          evaluations + 2U > policy.iteration.max_evaluations) {
        return Result<long double>::failure(
            depth >= policy.iteration.max_iterations ? ErrorCode::NoConvergence
                                                     : ErrorCode::ResourceLimit,
            "truncated raw-moment quadrature reached its work limit");
      }
      const Scalar middle = 0.5 * (a + b);
      const Scalar leftMiddle = 0.5 * (a + middle);
      const Scalar rightMiddle = 0.5 * (middle + b);
      auto flm = integrand(leftMiddle);
      auto frm = integrand(rightMiddle);
      evaluations += 2U;
      if (!flm || !frm) {
        const Status& status = !flm ? flm.status() : frm.status();
        return Result<long double>::failure(status.code(),
                                            std::string(status.message()));
      }
      const long double leftEstimate = static_cast<long double>(middle - a) *
                                       (fa + 4.0L * flm.value() + fm) / 6.0L;
      const long double rightEstimate = static_cast<long double>(b - middle) *
                                        (fm + 4.0L * frm.value() + fb) / 6.0L;
      const long double refined = leftEstimate + rightEstimate;
      const long double error = std::abs(refined - estimate) / 15.0L;
      const long double allowed =
          static_cast<long double>(tolerance) +
          static_cast<long double>(policy.iteration.rel) * std::abs(refined);
      if (error <= allowed) {
        return Result<long double>::success(refined +
                                            (refined - estimate) / 15.0L);
      }
      auto left = integrate(a, middle, fa, flm.value(), fm, leftEstimate,
                            tolerance * 0.5, depth + 1U);
      if (!left) {
        return left;
      }
      auto right = integrate(middle, b, fm, frm.value(), fb, rightEstimate,
                             tolerance * 0.5, depth + 1U);
      if (!right) {
        return right;
      }
      return Result<long double>::success(left.value() + right.value());
    };
    auto integral =
        integrate(integrationLower, integrationUpper, leftValue.value(),
                  middleValue.value(), rightValue.value(), whole,
                  policy.iteration.abs, 0U);
    if (!integral) {
      return Result<Scalar>::failure(integral.status().code(),
                                     std::string(integral.status().message()));
    }
    return finish(integral.value());
  }
  case Family::mixture: {
    long double total = 0.0L;
    for (Index index = 0U; index < state.components.size(); ++index) {
      if (state.weights[index] == 0.0) {
        continue;
      }
      Distribution const component(state.components[index]);
      auto moment = rawMoment(component, order, policy);
      if (!moment) {
        return Result<Scalar>::failure(moment.status().code(),
                                       std::string(moment.status().message()));
      }
      total += static_cast<long double>(state.weights[index]) * moment.value();
    }
    return finish(total);
  }
  default:
    return Result<Scalar>::failure(ErrorCode::InvalidDomain,
                                   "raw moment is not defined for this family");
  }
}

Result<Scalar> DistributionEngine::quantile(const Distribution& law,
                                            Scalar probability,
                                            ProbabilityPolicy policy) {
  if (!law.state_) {
    return Result<Scalar>::failure(ErrorCode::InvalidDomain,
                                   "distribution is moved from");
  }
  if (!std::isfinite(probability)) {
    return Result<Scalar>::failure(ErrorCode::NonFiniteInput,
                                   "quantile probability must be finite");
  }
  if (probability < 0.0 || probability > 1.0) {
    return Result<Scalar>::failure(ErrorCode::InvalidDomain,
                                   "quantile probability must be in [0,1]");
  }
  if (!std::isfinite(policy.iteration.abs) ||
      !std::isfinite(policy.iteration.rel) || policy.iteration.abs <= 0.0 ||
      policy.iteration.rel <= 0.0) {
    return Result<Scalar>::failure(ErrorCode::InvalidTolerance,
                                   "quantile tolerances must be positive");
  }
  if (policy.iteration.max_iterations == 0U ||
      policy.iteration.max_evaluations == 0U) {
    return Result<Scalar>::failure(ErrorCode::InvalidDomain,
                                   "quantile work limits must be positive");
  }
  const Distribution::State& state = *law.state_;
  const auto& p = state.parameters;
  const Scalar infinity = std::numeric_limits<Scalar>::infinity();
  if (state.measure == Measure::atom_mass) {
    if (state.family == Family::empirical ||
        (state.family == Family::finite_convolution &&
         state.components.empty())) {
      if (probability == 0.0) {
        return Result<Scalar>::success(state.atoms.front());
      }
      if (probability == 1.0) {
        return Result<Scalar>::success(state.atoms.back());
      }
      const auto found =
          std::lower_bound(state.cumulativeMasses.begin(),
                           state.cumulativeMasses.end(), probability);
      if (found == state.cumulativeMasses.end()) {
        return Result<Scalar>::failure(ErrorCode::NoConvergence,
                                       "atom masses do not normalize to one");
      }
      return Result<Scalar>::success(state.atoms[static_cast<Index>(
          found - state.cumulativeMasses.begin())]);
    }
    const Limits limits{1024U, std::size_t{16U} * 1024U * 1024U,
                        policy.iteration.max_evaluations};
    auto support = finiteAtoms(law, limits);
    if (!support) {
      return Result<Scalar>::failure(support.status().code(),
                                     std::string(support.status().message()));
    }
    if (probability == 0.0 || probability == 1.0) {
      return Result<Scalar>::success(probability == 0.0
                                         ? support.value().front().first
                                         : support.value().back().first);
    }
    Scalar cumulative = 0.0;
    for (const auto& [atom, mass] : support.value()) {
      cumulative += mass;
      if (cumulative >= probability) {
        return Result<Scalar>::success(atom);
      }
    }
    return Result<Scalar>::failure(ErrorCode::NoConvergence,
                                   "atom masses do not normalize to one");
  }
  if (probability == 0.0) {
    switch (state.family) {
    case Family::bernoulli:
      return Result<Scalar>::success(p[0] == 1.0 ? 1.0 : 0.0);
    case Family::binomial:
    case Family::negative_binomial:
      return Result<Scalar>::success(0.0);
    case Family::poisson:
      return Result<Scalar>::success(0.0);
    case Family::gamma:
    case Family::chi_square:
    case Family::weibull:
    case Family::beta:
    case Family::fisher_f:
    case Family::lognormal:
      return Result<Scalar>::success(0.0);
    case Family::uniform:
    case Family::triangular:
      return Result<Scalar>::success(p[0]);
    case Family::truncated_normal:
      return Result<Scalar>::success(p[2]);
    case Family::truncated:
      return Result<Scalar>::success(p[0]);
    case Family::empirical:
      return Result<Scalar>::success(state.atoms.front());
    case Family::finite_convolution:
      if (state.components.size() == 2U &&
          state.components[0]->family == Family::uniform &&
          state.components[1]->family == Family::uniform) {
        return Result<Scalar>::success(state.components[0]->parameters[0] +
                                       state.components[1]->parameters[0]);
      }
      return Result<Scalar>::success(-infinity);
    default:
      return Result<Scalar>::success(-infinity);
    }
  }
  if (probability == 1.0) {
    switch (state.family) {
    case Family::bernoulli:
      return Result<Scalar>::success(p[0] == 0.0 ? 0.0 : 1.0);
    case Family::negative_binomial:
      return Result<Scalar>::success(p[1] == 1.0 ? 0.0 : infinity);
    case Family::poisson:
      return Result<Scalar>::success(p[0] == 0.0 ? 0.0 : infinity);
    case Family::gamma:
    case Family::weibull:
    case Family::fisher_f:
    case Family::lognormal:
      return Result<Scalar>::success(infinity);
    case Family::beta:
      return Result<Scalar>::success(1.0);
    case Family::binomial:
      return Result<Scalar>::success(
          static_cast<Scalar>(state.integerParameter));
    case Family::uniform:
    case Family::triangular:
      return Result<Scalar>::success(state.family == Family::uniform ? p[1]
                                                                     : p[2]);
    case Family::truncated_normal:
      return Result<Scalar>::success(p[3]);
    case Family::truncated:
      return Result<Scalar>::success(p[1]);
    case Family::empirical:
      return Result<Scalar>::success(state.atoms.back());
    case Family::finite_convolution:
      if (state.components.size() == 2U &&
          state.components[0]->family == Family::uniform &&
          state.components[1]->family == Family::uniform) {
        return Result<Scalar>::success(state.components[0]->parameters[1] +
                                       state.components[1]->parameters[1]);
      }
      return Result<Scalar>::success(infinity);
    default:
      return Result<Scalar>::success(infinity);
    }
  }

  const bool integerLaw = state.measure == Measure::integer_mass;
  if (integerLaw) {
    if (state.family == Family::bernoulli) {
      return Result<Scalar>::success(probability <= 1.0 - p[0] ? 0.0 : 1.0);
    }
    std::uint64_t lower = 0U;
    std::uint64_t upper = 1U;
    constexpr std::uint64_t exactIntegerMax = (std::uint64_t{1} << 53U) - 1U;
    if (state.family == Family::binomial) {
      upper = state.integerParameter;
    } else {
      long double guess = 1.0L;
      if (state.family == Family::poisson) {
        guess = static_cast<long double>(p[0]) +
                8.0L * std::sqrt(static_cast<long double>(p[0]) + 1.0L);
      } else {
        const long double r = p[0];
        const long double odds = (1.0L - p[1]) / p[1];
        guess = r * odds + 8.0L * std::sqrt(r * odds / p[1] + 1.0L);
      }
      guess =
          std::clamp(guess, 1.0L, static_cast<long double>(exactIntegerMax));
      upper = static_cast<std::uint64_t>(std::ceil(guess));
      std::uint64_t evaluations = 0U;
      while (true) {
        if (++evaluations > policy.iteration.max_evaluations) {
          return Result<Scalar>::failure(
              ErrorCode::ResourceLimit,
              "discrete quantile bracket cap exceeded");
        }
        auto atUpper = evaluate(law, static_cast<Scalar>(upper), policy);
        if (!atUpper) {
          return Result<Scalar>::failure(
              atUpper.status().code(), std::string(atUpper.status().message()));
        }
        if (atUpper.value().cdf >= probability) {
          break;
        }
        if (upper >= exactIntegerMax) {
          return Result<Scalar>::failure(
              ErrorCode::InvalidDomain,
              "quantile exceeds exact scalar integer range");
        }
        lower = upper + 1U;
        upper = std::min(exactIntegerMax, upper * 2U);
      }
    }
    if (state.family == Family::binomial) {
      auto endpoint = evaluate(law, static_cast<Scalar>(upper), policy);
      if (!endpoint || endpoint.value().cdf < probability) {
        return Result<Scalar>::failure(
            endpoint ? ErrorCode::NoConvergence : endpoint.status().code(),
            endpoint ? "binomial quantile bracket does not contain target"
                     : std::string(endpoint.status().message()));
      }
    }
    std::uint64_t evaluations = 0U;
    while (lower < upper) {
      if (evaluations >= policy.iteration.max_iterations ||
          evaluations >= policy.iteration.max_evaluations) {
        return Result<Scalar>::failure(
            ErrorCode::NoConvergence,
            "discrete quantile iteration cap exceeded");
      }
      ++evaluations;
      const std::uint64_t middle = lower + (upper - lower) / 2U;
      auto atMiddle = evaluate(law, static_cast<Scalar>(middle), policy);
      if (!atMiddle) {
        return Result<Scalar>::failure(
            atMiddle.status().code(), std::string(atMiddle.status().message()));
      }
      if (atMiddle.value().cdf >= probability) {
        upper = middle;
      } else {
        lower = middle + 1U;
      }
    }
    return Result<Scalar>::success(static_cast<Scalar>(lower));
  }

  Scalar lower = 0.0;
  Scalar upper = 0.0;
  if (policy.iteration.bracket) {
    lower = policy.iteration.bracket->lower;
    upper = policy.iteration.bracket->upper;
    if (!std::isfinite(lower) || !std::isfinite(upper) || lower >= upper) {
      return Result<Scalar>::failure(
          ErrorCode::InvalidDomain,
          "quantile bracket must be finite and increasing");
    }
  } else {
    Scalar center = 0.0;
    Scalar scale = 1.0;
    switch (state.family) {
    case Family::student_t:
      center = p[1];
      scale = p[2];
      break;
    case Family::normal:
    case Family::laplace:
    case Family::logistic:
    case Family::skew_normal:
      center = p[0];
      scale = p[1];
      break;
    case Family::lognormal:
      center = std::exp(p[0]);
      scale = center * std::max(0.25, p[1]);
      if (!std::isfinite(center) || !std::isfinite(scale)) {
        center = 0.0;
        scale = 1.0;
      }
      break;
    case Family::gamma:
      center = p[0] * p[1];
      scale = std::sqrt(p[0]) * p[1];
      break;
    case Family::chi_square:
      center = p[0];
      scale = std::sqrt(2.0 * p[0]);
      break;
    case Family::fisher_f:
    case Family::beta:
      center = 0.5;
      scale = 0.5;
      break;
    case Family::weibull:
      center = p[1];
      scale = p[1];
      break;
    case Family::uniform:
    case Family::triangular:
      lower = p[0];
      upper = state.family == Family::uniform ? p[1] : p[2];
      break;
    case Family::truncated_normal:
      lower = p[2];
      upper = p[3];
      if (!std::isfinite(lower) || !std::isfinite(upper)) {
        lower = 0.0;
        upper = 0.0;
        center = p[0];
        scale = p[1];
      }
      break;
    case Family::truncated:
      lower = p[0];
      upper = p[1];
      if (!std::isfinite(lower) || !std::isfinite(upper)) {
        lower = 0.0;
        upper = 0.0;
        center = 0.0;
        scale = 1.0;
      }
      break;
    default:
      break;
    }
    if (lower == upper) {
      scale = std::max(scale, std::numeric_limits<Scalar>::min());
      lower = center - scale;
      upper = center + scale;
      if (state.family == Family::gamma || state.family == Family::chi_square ||
          state.family == Family::weibull || state.family == Family::beta ||
          state.family == Family::fisher_f ||
          state.family == Family::truncated_normal) {
        lower = std::max(0.0, lower);
      }
    }
  }
  auto belowTarget = [&](Scalar x) -> Result<bool> {
    auto value = evaluate(law, x, policy);
    if (!value) {
      return Result<bool>::failure(value.status().code(),
                                   std::string(value.status().message()));
    }
    const bool compareLowerTail = probability <= 0.5;
    const Scalar targetLog =
        compareLowerTail ? std::log(probability) : std::log1p(-probability);
    const Scalar tail = compareLowerTail ? value.value().log_tails.log_cdf
                                         : value.value().log_tails.log_survival;
    return Result<bool>::success(compareLowerTail ? tail < targetLog
                                                  : tail > targetLog);
  };
  auto left = belowTarget(lower);
  auto right = belowTarget(upper);
  if (!left || !right) {
    const Status& status = !left ? left.status() : right.status();
    return Result<Scalar>::failure(status.code(),
                                   std::string(status.message()));
  }
  std::uint64_t evaluations = 2U;
  while (!left.value() || right.value()) {
    if (policy.iteration.bracket ||
        evaluations >= policy.iteration.max_evaluations) {
      return Result<Scalar>::failure(
          ErrorCode::InvalidDomain, "quantile bracket does not contain target");
    }
    const Scalar width = upper - lower;
    if (!std::isfinite(width) || width <= 0.0) {
      return Result<Scalar>::failure(ErrorCode::NonFiniteResult,
                                     "quantile bracket expansion overflowed");
    }
    if (!left.value()) {
      upper = lower;
      lower -= width;
      if (!std::isfinite(lower)) {
        return Result<Scalar>::failure(ErrorCode::NonFiniteResult,
                                       "quantile lower bracket overflowed");
      }
      left = belowTarget(lower);
      if (!left) {
        return Result<Scalar>::failure(left.status().code(),
                                       std::string(left.status().message()));
      }
    } else {
      lower = upper;
      upper += width;
      if (!std::isfinite(upper)) {
        return Result<Scalar>::failure(ErrorCode::NonFiniteResult,
                                       "quantile upper bracket overflowed");
      }
      right = belowTarget(upper);
      if (!right) {
        return Result<Scalar>::failure(right.status().code(),
                                       std::string(right.status().message()));
      }
    }
    ++evaluations;
  }
  for (Index iteration = 0U; iteration < policy.iteration.max_iterations;
       ++iteration) {
    if (evaluations >= policy.iteration.max_evaluations) {
      return Result<Scalar>::failure(ErrorCode::ResourceLimit,
                                     "quantile evaluation cap exceeded");
    }
    const Scalar midpoint = std::midpoint(lower, upper);
    if (midpoint == lower || midpoint == upper) {
      return Result<Scalar>::failure(
          ErrorCode::NoConvergence,
          "quantile bracket reached adjacent scalars");
    }
    auto middleBelow = belowTarget(midpoint);
    if (!middleBelow) {
      return Result<Scalar>::failure(
          middleBelow.status().code(),
          std::string(middleBelow.status().message()));
    }
    ++evaluations;
    if (middleBelow.value()) {
      lower = midpoint;
    } else {
      upper = midpoint;
    }
    const Scalar coordinateTolerance =
        policy.iteration.abs +
        policy.iteration.rel * std::max(std::abs(lower), std::abs(upper));
    if (upper - lower <= coordinateTolerance) {
      auto atUpper = evaluate(law, upper, policy);
      if (!atUpper) {
        return Result<Scalar>::failure(atUpper.status().code(),
                                       std::string(atUpper.status().message()));
      }
      const Scalar residual =
          probability <= 0.5
              ? std::abs(atUpper.value().cdf - probability)
              : std::abs(atUpper.value().survival - (1.0 - probability));
      if (residual <=
          policy.iteration.abs +
              policy.iteration.rel * std::min(probability, 1.0 - probability)) {
        return Result<Scalar>::success(upper);
      }
    }
  }
  return Result<Scalar>::failure(
      ErrorCode::NoConvergence,
      "quantile did not meet residual and bracket tolerances");
}

// Composite sampling recursively visits the explicitly depth-capped law tree.
// NOLINTNEXTLINE(misc-no-recursion)
Result<Scalar> DistributionEngine::sample(const Distribution& law, Rng& rng,
                                          ProbabilityPolicy policy) {
  if (!law.state_) {
    return Result<Scalar>::failure(ErrorCode::InvalidDomain,
                                   "distribution is moved from");
  }
  if (policy.max_draws == 0U) {
    return Result<Scalar>::failure(ErrorCode::InvalidDomain,
                                   "sampling draw limit must be positive");
  }
  if (law.state_->family == Family::finite_convolution &&
      !law.state_->components.empty()) {
    std::uint64_t drawCount = 0U;
    const auto countLeaves =
        // NOLINTNEXTLINE(misc-no-recursion): count leaves in the capped tree.
        [&](const auto& self,
            const std::shared_ptr<const Distribution::State>& state) -> bool {
      if (state->family != Family::finite_convolution ||
          state->components.empty()) {
        ++drawCount;
        return drawCount <= policy.max_draws;
      }
      return std::ranges::all_of(
          state->components,
          // NOLINTNEXTLINE(misc-no-recursion): bounded law-tree visitor.
          [&](const auto& child) { return self(self, child); });
    };
    if (!countLeaves(countLeaves, law.state_)) {
      return Result<Scalar>::failure(ErrorCode::ResourceLimit,
                                     "convolution sampling draw cap exceeded");
    }
    long double total = 0.0L;
    for (const auto& componentState : law.state_->components) {
      Distribution const component(componentState);
      auto draw = sample(component, rng, policy);
      if (!draw) {
        return Result<Scalar>::failure(draw.status().code(),
                                       std::string(draw.status().message()));
      }
      total += draw.value();
    }
    if (!std::isfinite(total) ||
        std::abs(total) > std::numeric_limits<Scalar>::max()) {
      return Result<Scalar>::failure(ErrorCode::NonFiniteResult,
                                     "convolution sample is not representable");
    }
    return Result<Scalar>::success(static_cast<Scalar>(total));
  }
  const std::uint64_t word = rng.next();
  const std::uint64_t j = word >> 12U;
  const Scalar probability = (static_cast<Scalar>(j) + 0.5) * 0x1.0p-52;
  return quantile(law, probability, policy);
}

// Composite variances recurse only through the explicitly depth-capped law
// tree. NOLINTNEXTLINE(misc-no-recursion)
Result<Scalar> DistributionEngine::variance(const Distribution& law,
                                            ProbabilityPolicy policy,
                                            Limits limits) {
  if (!law.state_) {
    return Result<Scalar>::failure(ErrorCode::InvalidDomain,
                                   "distribution is moved from");
  }
  if (limits.elements < 1U || limits.bytes < sizeof(Scalar) ||
      limits.evaluations == 0U) {
    return Result<Scalar>::failure(ErrorCode::ResourceLimit,
                                   "variance output exceeds configured limits");
  }
  const Distribution::State& state = *law.state_;
  const auto& p = state.parameters;
  const long double pi = std::numbers::pi_v<long double>;
  auto finish = [](long double value) -> Result<Scalar> {
    if (!std::isfinite(value) || value < 0.0L ||
        value > static_cast<long double>(std::numeric_limits<Scalar>::max())) {
      return Result<Scalar>::failure(ErrorCode::NonFiniteResult,
                                     "variance is outside scalar range");
    }
    return Result<Scalar>::success(static_cast<Scalar>(value));
  };
  switch (state.family) {
  case Family::normal:
    return finish(static_cast<long double>(p[1]) * p[1]);
  case Family::student_t:
    if (p[0] <= 2.0) {
      return Result<Scalar>::failure(ErrorCode::InvalidDomain,
                                     "Student-t variance does not exist");
    }
    return finish(static_cast<long double>(p[2]) * p[2] * p[0] /
                  (static_cast<long double>(p[0]) - 2.0L));
  case Family::laplace:
    return finish(2.0L * p[1] * p[1]);
  case Family::logistic:
    return finish(pi * pi * p[1] * p[1] / 3.0L);
  case Family::lognormal:
    return finish(
        std::exp(2.0L * p[0] + static_cast<long double>(p[1]) * p[1]) *
        std::expm1(static_cast<long double>(p[1]) * p[1]));
  case Family::gamma:
    return finish(static_cast<long double>(p[0]) * p[1] * p[1]);
  case Family::beta: {
    const long double total = static_cast<long double>(p[0]) + p[1];
    return finish(static_cast<long double>(p[0]) * p[1] /
                  (total * total * (total + 1.0L)));
  }
  case Family::weibull: {
    const long double first = std::tgamma(1.0L + 1.0L / p[0]);
    const long double second = std::tgamma(1.0L + 2.0L / p[0]);
    return finish(static_cast<long double>(p[1]) * p[1] *
                  std::max(0.0L, second - first * first));
  }
  case Family::skew_normal: {
    const long double delta = static_cast<long double>(p[2]) /
                              std::hypot(1.0L, static_cast<long double>(p[2]));
    return finish(static_cast<long double>(p[1]) * p[1] *
                  (1.0L - 2.0L * delta * delta / pi));
  }
  case Family::uniform: {
    const long double width = static_cast<long double>(p[1]) - p[0];
    return finish(width * width / 12.0L);
  }
  case Family::triangular: {
    const long double a = p[0];
    const long double b = p[1];
    const long double c = p[2];
    return finish((a * a + b * b + c * c - a * b - a * c - b * c) / 18.0L);
  }
  case Family::truncated_normal: {
    const long double alpha = (static_cast<long double>(p[2]) - p[0]) / p[1];
    const long double beta = (static_cast<long double>(p[3]) - p[0]) / p[1];
    auto lower = normalLogTails(static_cast<Scalar>(alpha), policy);
    auto upper = normalLogTails(static_cast<Scalar>(beta), policy);
    if (!lower || !upper) {
      const Status& status = !lower ? lower.status() : upper.status();
      return Result<Scalar>::failure(status.code(),
                                     std::string(status.message()));
    }
    Scalar logZ = NAN;
    if (alpha >= 0.0L) {
      logZ =
          logDifference(lower.value().log_survival, upper.value().log_survival);
    } else if (beta <= 0.0L) {
      logZ = logDifference(upper.value().log_cdf, lower.value().log_cdf);
    } else {
      const long double outside =
          std::exp(static_cast<long double>(lower.value().log_cdf)) +
          std::exp(static_cast<long double>(upper.value().log_survival));
      logZ = static_cast<Scalar>(std::log1p(-outside));
    }
    const long double z = std::exp(static_cast<long double>(logZ));
    const auto phi = [pi](long double x) {
      return std::isfinite(x) ? std::exp(-x * x / 2.0L) / std::sqrt(2.0L * pi)
                              : 0.0L;
    };
    const long double shift = (phi(alpha) - phi(beta)) / z;
    const long double correction =
        ((std::isfinite(alpha) ? alpha * phi(alpha) : 0.0L) -
         (std::isfinite(beta) ? beta * phi(beta) : 0.0L)) /
        z;
    return finish(static_cast<long double>(p[1]) * p[1] *
                  (1.0L + correction - shift * shift));
  }
  case Family::bernoulli:
    return finish(static_cast<long double>(p[0]) * (1.0L - p[0]));
  case Family::binomial:
    return finish(static_cast<long double>(state.integerParameter) * p[1] *
                  (1.0L - p[1]));
  case Family::poisson:
    return finish(p[0]);
  case Family::negative_binomial:
    return finish(static_cast<long double>(p[0]) * (1.0L - p[1]) /
                  (static_cast<long double>(p[1]) * p[1]));
  case Family::chi_square:
    return finish(2.0L * p[0]);
  case Family::fisher_f: {
    const long double d1 = p[0];
    const long double d2 = p[1];
    if (d2 <= 4.0L) {
      return Result<Scalar>::failure(ErrorCode::InvalidDomain,
                                     "F variance does not exist");
    }
    return finish(2.0L * d2 * d2 * (d1 + d2 - 2.0L) /
                  (d1 * (d2 - 2.0L) * (d2 - 2.0L) * (d2 - 4.0L)));
  }
  case Family::empirical: {
    auto first = rawMoment(law, 1U, policy);
    if (!first) {
      return Result<Scalar>::failure(first.status().code(),
                                     std::string(first.status().message()));
    }
    long double value = 0.0L;
    for (Index index = 0U; index < state.atoms.size(); ++index) {
      const long double difference =
          static_cast<long double>(state.atoms[index]) - first.value();
      value += static_cast<long double>(state.masses[index]) * difference *
               difference;
    }
    return finish(value);
  }
  case Family::finite_convolution: {
    if (state.components.empty()) {
      auto first = rawMoment(law, 1U, policy);
      if (!first) {
        return Result<Scalar>::failure(first.status().code(),
                                       std::string(first.status().message()));
      }
      long double value = 0.0L;
      for (Index index = 0U; index < state.atoms.size(); ++index) {
        const long double difference =
            static_cast<long double>(state.atoms[index]) - first.value();
        value += static_cast<long double>(state.masses[index]) * difference *
                 difference;
      }
      return finish(value);
    }
    long double total = 0.0L;
    for (const auto& componentState : state.components) {
      Distribution const component(componentState);
      auto value = variance(component, policy, limits);
      if (!value) {
        return Result<Scalar>::failure(value.status().code(),
                                       std::string(value.status().message()));
      }
      total += value.value();
    }
    return finish(total);
  }
  case Family::mixture: {
    auto overallMean = rawMoment(law, 1U, policy);
    if (!overallMean) {
      return Result<Scalar>::failure(
          overallMean.status().code(),
          std::string(overallMean.status().message()));
    }
    long double total = 0.0L;
    for (Index index = 0U; index < state.components.size(); ++index) {
      if (state.weights[index] == 0.0) {
        continue;
      }
      Distribution const component(state.components[index]);
      auto componentMean = rawMoment(component, 1U, policy);
      auto componentVariance = variance(component, policy, limits);
      if (!componentMean || !componentVariance) {
        const Status& status = !componentMean ? componentMean.status()
                                              : componentVariance.status();
        return Result<Scalar>::failure(status.code(),
                                       std::string(status.message()));
      }
      const long double delta =
          static_cast<long double>(componentMean.value()) - overallMean.value();
      total += static_cast<long double>(state.weights[index]) *
               (componentVariance.value() + delta * delta);
    }
    return finish(total);
  }
  default:
    return Result<Scalar>::failure(ErrorCode::InvalidDomain,
                                   "variance is not defined for this family");
  }
}

} // namespace detail

Result<TailValues> evaluate(const Distribution& law, Scalar x,
                            ProbabilityPolicy policy) {
  return detail::DistributionEngine::evaluate(law, x, policy);
}

Result<Scalar> quantile(const Distribution& law, Scalar probability,
                        ProbabilityPolicy policy) {
  return detail::DistributionEngine::quantile(law, probability, policy);
}

Result<Scalar> sample(const Distribution& law, Rng& rng,
                      ProbabilityPolicy policy) {
  return detail::DistributionEngine::sample(law, rng, policy);
}

Result<Scalar> rawMoment(const Distribution& law, Index order,
                         ProbabilityPolicy policy) {
  return detail::DistributionEngine::rawMoment(law, order, policy);
}

Result<Scalar> mean(const Distribution& law, ProbabilityPolicy policy,
                    Limits limits) {
  if (limits.elements < 1U || limits.bytes < sizeof(Scalar) ||
      limits.evaluations == 0U) {
    return Result<Scalar>::failure(ErrorCode::ResourceLimit,
                                   "mean output exceeds configured limits");
  }
  if (policy.iteration.max_evaluations > limits.evaluations) {
    policy.iteration.max_evaluations = static_cast<Index>(limits.evaluations);
  }
  return detail::DistributionEngine::rawMoment(law, 1U, policy);
}

Result<Scalar> variance(const Distribution& law, ProbabilityPolicy policy,
                        Limits limits) {
  if (policy.iteration.max_evaluations > limits.evaluations &&
      limits.evaluations > 0U) {
    policy.iteration.max_evaluations = static_cast<Index>(limits.evaluations);
  }
  return detail::DistributionEngine::variance(law, policy, limits);
}

Result<Scalar> mixtureLogPdf(const Distribution& law, Scalar x,
                             ProbabilityPolicy policy) {
  if (law.family() != Family::mixture) {
    return Result<Scalar>::failure(ErrorCode::InvalidDomain,
                                   "mixtureLogPdf requires a mixture law");
  }
  auto value = evaluate(law, x, policy);
  if (!value) {
    return Result<Scalar>::failure(value.status().code(),
                                   std::string(value.status().message()));
  }
  return Result<Scalar>::success(value.value().log_density_or_mass);
}

Result<Distribution> truncatedDistribution(const Distribution& base,
                                           IntervalSupport interval,
                                           ProbabilityPolicy policy) {
  return detail::DistributionEngine::truncated(
      base, interval, policy,
      Limits{1024U, std::size_t{16U} * 1024U * 1024U,
             policy.iteration.max_evaluations});
}

Result<Distribution> truncatedDistribution(const Distribution& base,
                                           IntervalSupport interval,
                                           ProbabilityPolicy policy,
                                           Limits limits) {
  return detail::DistributionEngine::truncated(base, interval, policy, limits);
}

Result<Scalar> empiricalCDF(const Distribution& empirical, Scalar x) {
  if (empirical.family() != Family::empirical) {
    return Result<Scalar>::failure(ErrorCode::InvalidDomain,
                                   "empiricalCDF requires an empirical law");
  }
  auto value = evaluate(empirical, x, defaultPolicy());
  if (!value) {
    return Result<Scalar>::failure(value.status().code(),
                                   std::string(value.status().message()));
  }
  return Result<Scalar>::success(value.value().cdf);
}

Result<Scalar> censoredLogProbability(const Distribution& law,
                                      IntervalSupport event,
                                      ProbabilityPolicy policy) {
  return detail::DistributionEngine::intervalLogProbability(law, event, policy);
}

Result<Scalar> quantizedLikelihood(const Distribution& law, IntervalSupport bin,
                                   ProbabilityPolicy policy) {
  return detail::DistributionEngine::intervalLogProbability(law, bin, policy);
}

} // namespace vss::math::probability
