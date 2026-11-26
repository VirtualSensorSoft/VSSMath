#include "probability/DistributionInternal.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <new>
#include <numbers>
#include <numeric>
#include <stdexcept>
#include <vector>
#include <vss/math/Error.hpp>
#include <vss/math/Integration.hpp>
#include <vss/math/MatrixView.hpp>
#include <vss/math/ScientificExtras.hpp>
#include <vss/math/VectorView.hpp>
#include <vss/math/linear/LU.hpp>

namespace vss::math {
namespace {

using Wide = long double;
constexpr Wide theta13 = 5.371920351148152L;
constexpr std::size_t probabilityWorkspaceElements = 1'000'000U;
constexpr std::size_t probabilityWorkspaceBytes = 64U * 1024U * 1024U;

Wide digamma(Wide x) {
  Wide result = 0.0L;
  while (x < 8.0L) {
    result -= 1.0L / x;
    x += 1.0L;
  }
  const Wide inverse = 1.0L / x;
  const Wide square = inverse * inverse;
  result +=
      std::log(x) - 0.5L * inverse -
      square * (1.0L / 12.0L -
                square * (1.0L / 120.0L -
                          square * (1.0L / 252.0L -
                                    square * (1.0L / 240.0L -
                                              square * (5.0L / 660.0L)))));
  return result;
}

Result<Matrix> matrixProduct(ConstMatrixView left, ConstMatrixView right) {
  const Extent n = left.rows();
  Matrix result(n, n, 0.0);
  for (Index i = 0U; i < n; ++i) {
    for (Index j = 0U; j < n; ++j) {
      Wide sum = 0.0L;
      Wide correction = 0.0L;
      for (Index k = 0U; k < n; ++k) {
        const Wide term = static_cast<Wide>(left.at(i, k)) * right.at(k, j);
        const Wide adjusted = term - correction;
        const Wide next = sum + adjusted;
        correction = (next - sum) - adjusted;
        sum = next;
      }
      if (!std::isfinite(sum) ||
          std::abs(sum) > std::numeric_limits<Scalar>::max()) {
        return Result<Matrix>::failure(ErrorCode::NonFiniteResult,
                                       "matrix exponential product overflowed");
      }
      result.at(i, j) = static_cast<Scalar>(sum);
    }
  }
  return Result<Matrix>::success(std::move(result));
}

void addScaled(Matrix& destination, ConstMatrixView source, Wide scale) {
  for (Index i = 0U; i < destination.rows(); ++i) {
    for (Index j = 0U; j < destination.columns(); ++j) {
      destination.at(i, j) += static_cast<Scalar>(scale * source.at(i, j));
    }
  }
}

} // namespace

Result<Matrix> matrixExponential(Matrix matrix, SolvePolicy policy,
                                 Limits limits) {
  if (matrix.rows() != matrix.columns()) {
    return Result<Matrix>::failure(
        ErrorCode::DimensionMismatch,
        "matrix exponential requires a square input");
  }
  const Index n = matrix.rows();
  if (n == 0U) {
    return Result<Matrix>::success(Matrix{});
  }
  for (Index i = 0U; i < n; ++i) {
    for (Index j = 0U; j < n; ++j) {
      if (!std::isfinite(matrix.at(i, j))) {
        return Result<Matrix>::failure(
            ErrorCode::NonFiniteInput,
            "matrix exponential input is non-finite");
      }
    }
  }
  if (n > std::numeric_limits<Index>::max() / n) {
    return Result<Matrix>::failure(ErrorCode::SizeOverflow,
                                   "matrix exponential extent overflows");
  }
  const Index square = n * n;
  constexpr Index matrixCount = 20U;
  if (square > std::numeric_limits<Index>::max() / matrixCount) {
    return Result<Matrix>::failure(ErrorCode::SizeOverflow,
                                   "matrix exponential workspace overflows");
  }
  const Index workspace = matrixCount * square;
  if (workspace > limits.elements ||
      workspace > limits.bytes / sizeof(Scalar)) {
    return Result<Matrix>::failure(
        ErrorCode::ResourceLimit,
        "matrix exponential workspace exceeds limits");
  }
  if (!std::isfinite(policy.rank_abs) || policy.rank_abs < 0.0 ||
      !std::isfinite(policy.rank_rel) || policy.rank_rel < 0.0 ||
      !std::isfinite(policy.reciprocal_condition_floor) ||
      policy.reciprocal_condition_floor < 0.0 ||
      policy.reciprocal_condition_floor > 1.0 ||
      !std::isfinite(policy.iteration.abs) || policy.iteration.abs < 0.0 ||
      !std::isfinite(policy.iteration.rel) || policy.iteration.rel < 0.0 ||
      policy.iteration.bracket.has_value()) {
    return Result<Matrix>::failure(
        ErrorCode::InvalidTolerance,
        "matrix exponential solve policy is invalid");
  }

  try {
    Wide oneNorm = 0.0L;
    for (Index j = 0U; j < n; ++j) {
      Wide column = 0.0L;
      for (Index i = 0U; i < n; ++i) {
        column += std::abs(static_cast<Wide>(matrix.at(i, j)));
      }
      oneNorm = std::max(oneNorm, column);
    }
    Index scaling = 0U;
    if (oneNorm > theta13) {
      const Wide raw = std::ceil(std::log2(oneNorm / theta13));
      if (!std::isfinite(raw) ||
          raw > static_cast<Wide>(std::numeric_limits<Index>::max())) {
        return Result<Matrix>::failure(ErrorCode::SizeOverflow,
                                       "matrix exponential scale overflows");
      }
      scaling = static_cast<Index>(raw);
    }
    if (scaling > policy.iteration.max_iterations) {
      return Result<Matrix>::failure(
          ErrorCode::ResourceLimit,
          "matrix exponential squaring cap exhausted");
    }
    const Scalar scale = std::ldexp(
        1.0,
        -static_cast<int>(std::min<Index>(
            scaling, static_cast<Index>(std::numeric_limits<int>::max()))));
    Matrix b(n, n, 0.0);
    for (Index i = 0U; i < n; ++i) {
      for (Index j = 0U; j < n; ++j) {
        b.at(i, j) = matrix.at(i, j) * scale;
      }
    }
    auto a2Result = matrixProduct(b.view(), b.view());
    if (!a2Result)
      return a2Result;
    Matrix a2 = std::move(a2Result).value();
    auto a4Result = matrixProduct(a2.view(), a2.view());
    if (!a4Result)
      return a4Result;
    Matrix a4 = std::move(a4Result).value();
    auto a6Result = matrixProduct(a4.view(), a2.view());
    if (!a6Result)
      return a6Result;
    Matrix a6 = std::move(a6Result).value();
    std::array<Wide, 14> coefficient{};
    coefficient[0] = 1.0L;
    for (Index j = 1U; j <= 13U; ++j) {
      coefficient[j] =
          coefficient[j - 1U] * (14U - j) / (static_cast<Wide>(j) * (27U - j));
    }

    Matrix uInner(n, n, 0.0);
    addScaled(uInner, a6.view(), coefficient[13]);
    addScaled(uInner, a4.view(), coefficient[11]);
    addScaled(uInner, a2.view(), coefficient[9]);
    auto uHigh = matrixProduct(a6.view(), uInner.view());
    if (!uHigh)
      return uHigh;
    Matrix uPolynomial = std::move(uHigh).value();
    addScaled(uPolynomial, a6.view(), coefficient[7]);
    addScaled(uPolynomial, a4.view(), coefficient[5]);
    addScaled(uPolynomial, a2.view(), coefficient[3]);
    for (Index i = 0U; i < n; ++i) {
      uPolynomial.at(i, i) += static_cast<Scalar>(coefficient[1]);
    }
    auto uResult = matrixProduct(b.view(), uPolynomial.view());
    if (!uResult)
      return uResult;
    Matrix u = std::move(uResult).value();

    Matrix vInner(n, n, 0.0);
    addScaled(vInner, a6.view(), coefficient[12]);
    addScaled(vInner, a4.view(), coefficient[10]);
    addScaled(vInner, a2.view(), coefficient[8]);
    auto vHigh = matrixProduct(a6.view(), vInner.view());
    if (!vHigh)
      return vHigh;
    Matrix v = std::move(vHigh).value();
    addScaled(v, a6.view(), coefficient[6]);
    addScaled(v, a4.view(), coefficient[4]);
    addScaled(v, a2.view(), coefficient[2]);
    for (Index i = 0U; i < n; ++i) {
      v.at(i, i) += static_cast<Scalar>(coefficient[0]);
    }

    Matrix denominator(n, n, 0.0);
    Matrix numerator(n, n, 0.0);
    for (Index i = 0U; i < n; ++i) {
      for (Index j = 0U; j < n; ++j) {
        denominator.at(i, j) = v.at(i, j) - u.at(i, j);
        numerator.at(i, j) = v.at(i, j) + u.at(i, j);
      }
    }
    linear::FactorizationOptions options{};
    options.rankTolerance = {policy.rank_abs, policy.rank_rel};
    options.conditionLimit = policy.reciprocal_condition_floor > 0.0
                                 ? 1.0 / policy.reciprocal_condition_floor
                                 : std::numeric_limits<Scalar>::max();
    auto factor = linear::factorizeLU(denominator.view(), options);
    if (!factor) {
      return Result<Matrix>::failure(factor.status().code(),
                                     std::string(factor.status().message()));
    }
    if (factor.value().isSingular()) {
      return Result<Matrix>::failure(
          ErrorCode::RankDeficient,
          "matrix exponential Padé denominator is singular");
    }
    auto solved = factor.value().solve(
        numerator.view(),
        {.illConditionedPolicy = linear::IllConditionedSolvePolicy::Reject});
    if (!solved) {
      return Result<Matrix>::failure(solved.status().code(),
                                     std::string(solved.status().message()));
    }
    Matrix result = solved.value().solution();
    for (Index squareIndex = 0U; squareIndex < scaling; ++squareIndex) {
      auto squared = matrixProduct(result.view(), result.view());
      if (!squared)
        return squared;
      result = std::move(squared).value();
    }
    return Result<Matrix>::success(std::move(result));
  } catch (const std::bad_alloc&) {
    return Result<Matrix>::failure(ErrorCode::ResourceLimit,
                                   "matrix exponential allocation failed");
  } catch (const std::length_error&) {
    return Result<Matrix>::failure(ErrorCode::ResourceLimit,
                                   "matrix exponential extent is unsupported");
  }
}

Result<Vector> adjustPValues(ConstVectorView pValues, MultiplicityRule rule,
                             Limits limits) {
  const Index count = pValues.size();
  if (count > std::numeric_limits<Index>::max() / 3U) {
    return Result<Vector>::failure(ErrorCode::SizeOverflow,
                                   "p-value workspace size overflows");
  }
  const Index required = 3U * count;
  if (required > limits.elements || required > limits.bytes / sizeof(Scalar)) {
    return Result<Vector>::failure(
        ErrorCode::ResourceLimit, "p-value adjustment exceeds resource limits");
  }
  if (rule != MultiplicityRule::bonferroni && rule != MultiplicityRule::holm &&
      rule != MultiplicityRule::benjamini_hochberg) {
    return Result<Vector>::failure(ErrorCode::InvalidDomain,
                                   "unknown multiplicity rule");
  }
  for (Index i = 0U; i < count; ++i) {
    const Scalar value = pValues.at(i);
    if (!std::isfinite(value)) {
      return Result<Vector>::failure(ErrorCode::NonFiniteInput,
                                     "p-values must be finite");
    }
    if (value < 0.0 || value > 1.0) {
      return Result<Vector>::failure(ErrorCode::InvalidDomain,
                                     "p-values must lie in [0,1]");
    }
  }
  try {
    Vector adjusted(count);
    if (count == 0U)
      return Result<Vector>::success(std::move(adjusted));
    if (rule == MultiplicityRule::bonferroni) {
      for (Index i = 0U; i < count; ++i) {
        adjusted.at(i) = static_cast<Scalar>(
            std::min<Wide>(1.0L, static_cast<Wide>(count) * pValues.at(i)));
      }
      return Result<Vector>::success(std::move(adjusted));
    }
    std::vector<Index> order(count);
    std::iota(order.begin(), order.end(), 0U);
    std::stable_sort(order.begin(), order.end(), [&](Index left, Index right) {
      return pValues.at(left) < pValues.at(right);
    });
    if (rule == MultiplicityRule::holm) {
      Wide running = 0.0L;
      for (Index rank = 0U; rank < count; ++rank) {
        const Wide value =
            static_cast<Wide>(count - rank) * pValues.at(order[rank]);
        running = std::max(running, value);
        adjusted.at(order[rank]) = static_cast<Scalar>(std::min(1.0L, running));
      }
    } else {
      Wide running = 1.0L;
      for (Index rank = count; rank > 0U; --rank) {
        const Wide value = static_cast<Wide>(count) *
                           pValues.at(order[rank - 1U]) /
                           static_cast<Wide>(rank);
        running = std::min(running, value);
        adjusted.at(order[rank - 1U]) =
            static_cast<Scalar>(std::min(1.0L, running));
      }
    }
    return Result<Vector>::success(std::move(adjusted));
  } catch (const std::bad_alloc&) {
    return Result<Vector>::failure(ErrorCode::ResourceLimit,
                                   "p-value adjustment allocation failed");
  } catch (const std::length_error&) {
    return Result<Vector>::failure(ErrorCode::ResourceLimit,
                                   "p-value workspace extent is unsupported");
  }
}

namespace {

Result<std::pair<Scalar, Scalar>>
fullSupportBounds(const probability::Distribution& law,
                  probability::IntervalSupport interval,
                  ProbabilityPolicy policy) {
  if (std::isnan(interval.lower) || std::isnan(interval.upper) ||
      !(interval.lower < interval.upper)) {
    return Result<std::pair<Scalar, Scalar>>::failure(
        ErrorCode::InvalidDomain, "probability interval is invalid");
  }
  auto lower = probability::quantile(law, 0.0, policy);
  auto upper = probability::quantile(law, 1.0, policy);
  if (!lower)
    return Result<std::pair<Scalar, Scalar>>::failure(
        lower.status().code(), std::string(lower.status().message()));
  if (!upper)
    return Result<std::pair<Scalar, Scalar>>::failure(
        upper.status().code(), std::string(upper.status().message()));
  if (law.measure() == probability::Measure::atom_mass ||
      law.measure() == probability::Measure::integer_mass ||
      law.measure() == probability::Measure::joint_atom_mass) {
    const auto atoms = law.finiteAtoms();
    if (!atoms.locations.empty()) {
      for (const Scalar atom : atoms.locations) {
        const bool above = atom > interval.lower ||
                           (interval.include_lower && atom == interval.lower);
        const bool below = atom < interval.upper ||
                           (interval.include_upper && atom == interval.upper);
        if (!(above && below)) {
          return Result<std::pair<Scalar, Scalar>>::failure(
              ErrorCode::InvalidDomain, "interval omits positive atom support");
        }
      }
    }
  } else {
    if (interval.lower > lower.value() && std::isfinite(interval.lower)) {
      auto atLower = probability::evaluate(law, interval.lower, policy);
      if (!atLower)
        return Result<std::pair<Scalar, Scalar>>::failure(
            atLower.status().code(), std::string(atLower.status().message()));
      if (atLower.value().cdf > 0.0) {
        return Result<std::pair<Scalar, Scalar>>::failure(
            ErrorCode::InvalidDomain, "interval omits lower-tail support mass");
      }
    }
    if (interval.upper < upper.value() && std::isfinite(interval.upper)) {
      auto atUpper = probability::evaluate(law, interval.upper, policy);
      if (!atUpper)
        return Result<std::pair<Scalar, Scalar>>::failure(
            atUpper.status().code(), std::string(atUpper.status().message()));
      if (atUpper.value().survival > 0.0) {
        return Result<std::pair<Scalar, Scalar>>::failure(
            ErrorCode::InvalidDomain, "interval omits upper-tail support mass");
      }
    }
  }
  return Result<std::pair<Scalar, Scalar>>::success(
      {interval.lower, interval.upper});
}

Wide logAddWide(Wide left, Wide right) {
  if (left == -std::numeric_limits<Wide>::infinity())
    return right;
  if (right == -std::numeric_limits<Wide>::infinity())
    return left;
  const Wide maximum = std::max(left, right);
  return maximum + std::log1p(std::exp(std::min(left, right) - maximum));
}

Wide binaryEntropyBound(Wide probability) {
  if (probability <= 0.0L)
    return 0.0L;
  return -probability * std::log(probability) -
         (1.0L - probability) * std::log1p(-probability);
}

Wide poissonOmissionBound(Wide rate, std::size_t lastRetained) {
  const Wide m = static_cast<Wide>(lastRetained + 1U);
  const Wide ratio = rate / (m + 1.0L);
  if (!(ratio > 0.0L && ratio < 1.0L)) {
    return std::numeric_limits<Wide>::infinity();
  }
  const Wide logFirst = -rate + m * std::log(rate) - std::lgamma(m + 1.0L);
  const Wide first = std::exp(logFirst);
  const Wide omittedMassBound = first / (1.0L - ratio);
  if (!(omittedMassBound < 0.5L) || !(first <= std::exp(-1.0L))) {
    return std::numeric_limits<Wide>::infinity();
  }
  const Wide weightEntropyBound =
      first * (-logFirst / (1.0L - ratio) -
               std::log(ratio) * ratio / ((1.0L - ratio) * (1.0L - ratio)));
  return binaryEntropyBound(omittedMassBound) +
         omittedMassBound * std::log(static_cast<Wide>(lastRetained + 1U)) +
         weightEntropyBound;
}

Wide gammaSpatialEntropyBound(Wide shape, Wide scale, Wide distance) {
  if (distance <= 0.0L)
    return std::numeric_limits<Wide>::infinity();
  const Wide logMaximum = (shape - 1.0L) * std::log(distance) -
                          distance / scale - std::lgamma(shape) -
                          shape * std::log(scale);
  const Wide maximum = std::exp(logMaximum);
  if (!(maximum <= std::exp(-1.0L))) {
    return std::numeric_limits<Wide>::infinity();
  }
  const Wide rate = shape <= 1.0L ? 1.0L / scale : 1.0L / (2.0L * scale);
  return maximum * (1.0L - logMaximum) / rate;
}

Wide normalSpatialEntropyBound(Wide standardDeviation, Wide distance) {
  if (distance <= 0.0L)
    return std::numeric_limits<Wide>::infinity();
  const Wide logMaximum =
      -0.5L * (distance / standardDeviation) * (distance / standardDeviation) -
      std::log(standardDeviation) - 0.9189385332046727417803297364056176L;
  const Wide maximum = std::exp(logMaximum);
  const Wide rate = distance / (standardDeviation * standardDeviation);
  if (!(maximum <= std::exp(-1.0L)) || !(rate > 0.0L)) {
    return std::numeric_limits<Wide>::infinity();
  }
  return 2.0L * maximum * (1.0L - logMaximum) / rate;
}

struct EntropyWorkLedger final {
  Index evaluations{};
  Index iterations{};
};

struct EntropyPanel final {
  Scalar lower{};
  Scalar upper{};
  Wide value{};
  Wide error{};
};

Result<IntegralResult> quadratureEntropy31(const UnaryFunction& function,
                                           Wide tolerance,
                                           ProbabilityPolicy policy,
                                           EntropyWorkLedger& work) {
  constexpr std::array<Scalar, 16> nodes{0.998002298693397060285172840152271209,
                                         0.987992518020485428489565718586612581,
                                         0.967739075679139134257347978784337225,
                                         0.937273392400705904307758947710209471,
                                         0.897264532344081900882509656454495883,
                                         0.848206583410427216200648320774216851,
                                         0.790418501442465932967649294817947347,
                                         0.724417731360170047416186326443938010,
                                         0.650996741297416970533735895313274693,
                                         0.570972172608538847537226737253910641,
                                         0.485081863640239680693655740232350613,
                                         0.394151347077563369897207370981045468,
                                         0.299180007153168812166780024266388963,
                                         0.201194093997434522300628303394596208,
                                         0.101142066918717499027074231447392339,
                                         0.0};
  constexpr std::array<Scalar, 16> kronrod{
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
  constexpr std::array<Scalar, 8> gauss{0.030753241996117268354628393577204418,
                                        0.070366047488108124709267416450667338,
                                        0.107159220467171935011869546685869303,
                                        0.139570677926154314447804231538028323,
                                        0.166269205816993933553200860481208811,
                                        0.186161000015562211026800561866422825,
                                        0.198431485327111576456118326443839325,
                                        0.202578241925561272880620199967519315};
  const auto evaluate = [&](Scalar lower,
                            Scalar upper) -> Result<EntropyPanel> {
    if (work.evaluations >= policy.iteration.max_evaluations ||
        policy.iteration.max_evaluations - work.evaluations < 31U) {
      return Result<EntropyPanel>::failure(
          ErrorCode::ResourceLimit, "entropy evaluation ledger exhausted");
    }
    const Wide center =
        static_cast<Wide>(lower) + (static_cast<Wide>(upper) - lower) / 2.0L;
    const Wide half = (static_cast<Wide>(upper) - lower) / 2.0L;
    std::array<Wide, 31> values{};
    const auto sample = [&](Scalar coordinate) -> Result<Scalar> {
      if (work.evaluations >= policy.iteration.max_evaluations) {
        return Result<Scalar>::failure(ErrorCode::ResourceLimit,
                                       "entropy evaluation ledger exhausted");
      }
      ++work.evaluations;
      return function(coordinate);
    };
    auto centerValue = sample(static_cast<Scalar>(center));
    if (!centerValue)
      return Result<EntropyPanel>::failure(
          centerValue.status().code(),
          std::string(centerValue.status().message()));
    values[30] = centerValue.value();
    Wide kronrodSum = kronrod[15] * values[30];
    Wide gaussSum = gauss[7] * values[30];
    Wide absoluteSum = std::abs(kronrod[15] * values[30]);
    for (Index node = 0U; node < 15U; ++node) {
      const Wide offset = half * nodes[node];
      auto left = sample(static_cast<Scalar>(center - offset));
      auto right = sample(static_cast<Scalar>(center + offset));
      if (!left || !right) {
        const Status& status = left ? right.status() : left.status();
        return Result<EntropyPanel>::failure(status.code(),
                                             std::string(status.message()));
      }
      values[2U * node] = left.value();
      values[2U * node + 1U] = right.value();
      const Wide pair = static_cast<Wide>(left.value()) + right.value();
      kronrodSum += kronrod[node] * pair;
      absoluteSum +=
          kronrod[node] * (std::abs(left.value()) + std::abs(right.value()));
      if (node % 2U == 1U)
        gaussSum += gauss[node / 2U] * pair;
    }
    const Wide mean = 0.5L * kronrodSum;
    Wide asc = kronrod[15] * std::abs(values[30] - mean);
    for (Index node = 0U; node < 15U; ++node) {
      asc += kronrod[node] * (std::abs(values[2U * node] - mean) +
                              std::abs(values[2U * node + 1U] - mean));
    }
    asc *= half;
    const Wide absolute = absoluteSum * half;
    Wide error = std::abs((kronrodSum - gaussSum) * half);
    if (asc != 0.0L && error != 0.0L) {
      error = asc * std::min(1.0L, std::pow(200.0L * error / asc, 1.5L));
    }
    error = std::max(error,
                     50.0L * std::numeric_limits<Scalar>::epsilon() * absolute);
    return Result<EntropyPanel>::success(
        {lower, upper, kronrodSum * half, error});
  };
  auto initial = evaluate(0.0, 1.0);
  if (!initial)
    return Result<IntegralResult>::failure(
        initial.status().code(), std::string(initial.status().message()));
  std::vector<EntropyPanel> panels{initial.value()};
  const auto totals = [&] {
    std::pair<Wide, Wide> result{};
    for (const auto& panel : panels) {
      result.first += panel.value;
      result.second += panel.error;
    }
    return result;
  };
  for (;;) {
    const auto [value, error] = totals();
    if (error <= tolerance) {
      return Result<IntegralResult>::success(
          {static_cast<Scalar>(value), static_cast<Scalar>(error), 0U,
           static_cast<Index>(panels.size()), Termination::converged});
    }
    if (work.iterations >= policy.iteration.max_iterations) {
      return Result<IntegralResult>::success(
          {static_cast<Scalar>(value), static_cast<Scalar>(error), 0U,
           static_cast<Index>(panels.size()), Termination::cap});
    }
    const auto selected = std::max_element(
        panels.begin(), panels.end(),
        [](const EntropyPanel& left, const EntropyPanel& right) {
          if (left.error != right.error)
            return left.error < right.error;
          return left.lower > right.lower;
        });
    const Scalar middle = std::midpoint(selected->lower, selected->upper);
    if (!(middle > selected->lower && middle < selected->upper)) {
      return Result<IntegralResult>::failure(ErrorCode::NoConvergence,
                                             "entropy panel cannot be split");
    }
    const Scalar lower = selected->lower;
    const Scalar upper = selected->upper;
    auto left = evaluate(lower, middle);
    if (!left)
      return Result<IntegralResult>::failure(
          left.status().code(), std::string(left.status().message()));
    auto right = evaluate(middle, upper);
    if (!right)
      return Result<IntegralResult>::failure(
          right.status().code(), std::string(right.status().message()));
    *selected = left.value();
    panels.push_back(right.value());
    ++work.iterations;
  }
}

struct EntropyPairPanel final {
  Scalar lower{};
  Scalar upper{};
  Wide value{};
  Wide error{};
  Wide auxiliary{};
  Wide auxiliaryError{};
};

struct EntropyPairEstimate final {
  Wide value{};
  Wide error{};
  Wide auxiliary{};
  Wide auxiliaryError{};
};

template <class Function>
Result<EntropyPairEstimate>
quadratureEntropyPair31(const Function& function, Wide tolerance,
                        ProbabilityPolicy policy, EntropyWorkLedger& work) {
  if (!(tolerance > 0.0L)) {
    return Result<EntropyPairEstimate>::failure(
        ErrorCode::InvalidTolerance,
        "paired entropy tolerance must be positive");
  }
  constexpr std::array<Scalar, 16> nodes{0.998002298693397060285172840152271209,
                                         0.987992518020485428489565718586612581,
                                         0.967739075679139134257347978784337225,
                                         0.937273392400705904307758947710209471,
                                         0.897264532344081900882509656454495883,
                                         0.848206583410427216200648320774216851,
                                         0.790418501442465932967649294817947347,
                                         0.724417731360170047416186326443938010,
                                         0.650996741297416970533735895313274693,
                                         0.570972172608538847537226737253910641,
                                         0.485081863640239680693655740232350613,
                                         0.394151347077563369897207370981045468,
                                         0.299180007153168812166780024266388963,
                                         0.201194093997434522300628303394596208,
                                         0.101142066918717499027074231447392339,
                                         0.0};
  constexpr std::array<Scalar, 16> kronrod{
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
  constexpr std::array<Scalar, 8> gauss{0.030753241996117268354628393577204418,
                                        0.070366047488108124709267416450667338,
                                        0.107159220467171935011869546685869303,
                                        0.139570677926154314447804231538028323,
                                        0.166269205816993933553200860481208811,
                                        0.186161000015562211026800561866422825,
                                        0.198431485327111576456118326443839325,
                                        0.202578241925561272880620199967519315};
  const auto evaluate = [&](Scalar lower,
                            Scalar upper) -> Result<EntropyPairPanel> {
    if (work.evaluations >= policy.iteration.max_evaluations ||
        policy.iteration.max_evaluations - work.evaluations < 31U) {
      return Result<EntropyPairPanel>::failure(
          ErrorCode::ResourceLimit, "entropy evaluation ledger exhausted");
    }
    const Wide center =
        static_cast<Wide>(lower) + (static_cast<Wide>(upper) - lower) / 2.0L;
    const Wide half = (static_cast<Wide>(upper) - lower) / 2.0L;
    std::array<std::array<Wide, 2>, 31> values{};
    const auto sample =
        [&](Scalar coordinate) -> Result<std::array<Scalar, 2>> {
      if (work.evaluations >= policy.iteration.max_evaluations) {
        return Result<std::array<Scalar, 2>>::failure(
            ErrorCode::ResourceLimit, "entropy evaluation ledger exhausted");
      }
      ++work.evaluations;
      return function(coordinate);
    };
    auto centerValue = sample(static_cast<Scalar>(center));
    if (!centerValue)
      return Result<EntropyPairPanel>::failure(
          centerValue.status().code(),
          std::string(centerValue.status().message()));
    values[30] = {centerValue.value()[0U], centerValue.value()[1U]};
    std::array<Wide, 2> kronrodSum{kronrod[15] * values[30][0],
                                   kronrod[15] * values[30][1]};
    std::array<Wide, 2> gaussSum{gauss[7] * values[30][0],
                                 gauss[7] * values[30][1]};
    std::array<Wide, 2> absoluteSum{std::abs(kronrodSum[0]),
                                    std::abs(kronrodSum[1])};
    for (Index node = 0U; node < 15U; ++node) {
      const Wide offset = half * nodes[node];
      auto left = sample(static_cast<Scalar>(center - offset));
      auto right = sample(static_cast<Scalar>(center + offset));
      if (!left || !right) {
        const Status& status = left ? right.status() : left.status();
        return Result<EntropyPairPanel>::failure(status.code(),
                                                 std::string(status.message()));
      }
      values[2U * node] = {left.value()[0U], left.value()[1U]};
      values[2U * node + 1U] = {right.value()[0U], right.value()[1U]};
      for (Index component = 0U; component < 2U; ++component) {
        const Wide pair =
            values[2U * node][component] + values[2U * node + 1U][component];
        kronrodSum[component] += kronrod[node] * pair;
        absoluteSum[component] +=
            kronrod[node] * (std::abs(values[2U * node][component]) +
                             std::abs(values[2U * node + 1U][component]));
        if (node % 2U == 1U)
          gaussSum[component] += gauss[node / 2U] * pair;
      }
    }
    std::array<Wide, 2> estimate{};
    std::array<Wide, 2> errors{};
    for (Index component = 0U; component < 2U; ++component) {
      const Wide mean = 0.5L * kronrodSum[component];
      Wide asc = kronrod[15] * std::abs(values[30][component] - mean);
      for (Index node = 0U; node < 15U; ++node) {
        asc += kronrod[node] *
               (std::abs(values[2U * node][component] - mean) +
                std::abs(values[2U * node + 1U][component] - mean));
      }
      asc *= half;
      const Wide absolute = absoluteSum[component] * half;
      Wide error =
          std::abs((kronrodSum[component] - gaussSum[component]) * half);
      if (asc != 0.0L && error != 0.0L) {
        error = asc * std::min(1.0L, std::pow(200.0L * error / asc, 1.5L));
      }
      errors[component] = std::max(
          error, 50.0L * std::numeric_limits<Scalar>::epsilon() * absolute);
      estimate[component] = kronrodSum[component] * half;
    }
    return Result<EntropyPairPanel>::success(
        {lower, upper, estimate[0], errors[0], estimate[1], errors[1]});
  };
  auto initial = evaluate(0.0, 1.0);
  if (!initial)
    return Result<EntropyPairEstimate>::failure(
        initial.status().code(), std::string(initial.status().message()));
  return Result<EntropyPairEstimate>::success(
      {initial.value().value, initial.value().error, initial.value().auxiliary,
       initial.value().auxiliaryError});
}

Result<Scalar> nestedConditionedEntropy(const probability::Distribution& law,
                                        probability::IntervalSupport truncation,
                                        ProbabilityPolicy policy,
                                        bool& handled) {
  using probability::Family;
  using probability::detail::DistributionSnapshot;
  handled = false;
  if (law.family() != Family::truncated) {
    return Result<Scalar>::failure(ErrorCode::InvalidDomain,
                                   "not a nested conditioned law");
  }
  auto root = probability::detail::DistributionEngine::snapshot(
      law, {probabilityWorkspaceElements, probabilityWorkspaceBytes,
            policy.iteration.max_evaluations});
  if (!root) {
    handled = true;
    return Result<Scalar>::failure(root.status().code(),
                                   std::string(root.status().message()));
  }
  if (root.value().parameters.size() < 3U ||
      root.value().components.size() != 1U) {
    return Result<Scalar>::failure(ErrorCode::InvalidDomain,
                                   "unsupported truncation expression");
  }
  const auto& outer = root.value().components.front();
  if (outer.family != Family::finite_convolution ||
      outer.components.size() != 2U) {
    return Result<Scalar>::failure(ErrorCode::InvalidDomain,
                                   "unsupported nested convolution expression");
  }
  const DistributionSnapshot* innerTruncated = nullptr;
  const DistributionSnapshot* outerLeaf = nullptr;
  for (const auto& child : outer.components) {
    if (child.family == Family::truncated)
      innerTruncated = &child;
    else if (child.family == Family::gamma || child.family == Family::poisson) {
      outerLeaf = &child;
    }
  }
  if (innerTruncated == nullptr || outerLeaf == nullptr ||
      innerTruncated->parameters.size() < 3U ||
      innerTruncated->components.size() != 1U) {
    return Result<Scalar>::failure(ErrorCode::InvalidDomain,
                                   "unsupported nested conditioned profile");
  }
  const auto& inner = innerTruncated->components.front();
  if (inner.family != Family::finite_convolution ||
      inner.components.size() != 2U || root.value().parameters[0U] != 0.0 ||
      root.value().parameters[1U] != 3.0 ||
      innerTruncated->parameters[0U] != -1.0 ||
      innerTruncated->parameters[1U] != 2.0 ||
      truncation.lower > root.value().parameters[0U] ||
      truncation.upper < root.value().parameters[1U]) {
    return Result<Scalar>::failure(
        ErrorCode::InvalidDomain,
        "nested entropy interval or profile is unsupported");
  }
  const DistributionSnapshot* normalLaw = nullptr;
  const DistributionSnapshot* uniformLaw = nullptr;
  for (const auto& child : inner.components) {
    if (child.family == Family::normal)
      normalLaw = &child;
    if (child.family == Family::uniform)
      uniformLaw = &child;
  }
  if (normalLaw == nullptr || uniformLaw == nullptr ||
      normalLaw->parameters.size() < 2U || uniformLaw->parameters.size() < 2U) {
    return Result<Scalar>::failure(ErrorCode::InvalidDomain,
                                   "nested base is not Normal plus Uniform");
  }
  const bool gammaOuter = outerLeaf->family == Family::gamma;
  if ((gammaOuter && (outerLeaf->parameters.size() < 2U ||
                      outerLeaf->parameters[0U] != 0.75 ||
                      outerLeaf->parameters[1U] != 1.0)) ||
      (!gammaOuter &&
       (outerLeaf->parameters.empty() || outerLeaf->parameters[0U] != 1.0))) {
    return Result<Scalar>::failure(
        ErrorCode::InvalidDomain,
        "nested outer law has no qualified entropy path");
  }
  handled = true;
  if (policy.iteration.bracket.has_value() ||
      !std::isfinite(policy.iteration.abs) || policy.iteration.abs < 0.0 ||
      !std::isfinite(policy.iteration.rel) || policy.iteration.rel < 0.0 ||
      policy.iteration.max_iterations == 0U ||
      policy.iteration.max_evaluations < 31U ||
      !std::isfinite(policy.abs_log_error) || policy.abs_log_error <= 0.0) {
    return Result<Scalar>::failure(ErrorCode::InvalidTolerance,
                                   "nested entropy policy is invalid");
  }
  const Wide tolerance =
      std::min({2.0e-9L, static_cast<Wide>(policy.iteration.abs),
                static_cast<Wide>(policy.abs_log_error)});
  if (!(tolerance > 0.0L)) {
    return Result<Scalar>::failure(
        ErrorCode::NoConvergence, "requested zero entropy error is unresolved");
  }
  const Wide innerNormalizer =
      std::exp(static_cast<Wide>(innerTruncated->parameters[2U]));
  const Wide outerNormalizer =
      std::exp(static_cast<Wide>(root.value().parameters[2U]));
  if (!(innerNormalizer > 0.0L && innerNormalizer <= 1.0L &&
        outerNormalizer > 0.0L && outerNormalizer <= 1.0L)) {
    return Result<Scalar>::failure(ErrorCode::NonFiniteResult,
                                   "nested truncation normalizer is invalid");
  }
  const Wide mean = normalLaw->parameters[0U];
  const Wide deviation = normalLaw->parameters[1U];
  const Wide uniformLower = uniformLaw->parameters[0U];
  const Wide uniformUpper = uniformLaw->parameters[1U];
  const Wide innerLower = innerTruncated->parameters[0U];
  const Wide innerUpper = innerTruncated->parameters[1U];
  const Wide outerShape = gammaOuter ? outerLeaf->parameters[0U] : 0.0L;
  const Wide outerScale = gammaOuter ? outerLeaf->parameters[1U] : 0.0L;
  const Wide poissonRate = gammaOuter ? 0.0L : outerLeaf->parameters[0U];
  const auto normalUniformDensity = [&](Wide x) {
    if (x < innerLower || x > innerUpper)
      return 0.0L;
    const Wide zLower = (x - uniformLower - mean) / deviation;
    const Wide zUpper = (x - uniformUpper - mean) / deviation;
    const Wide lowerCdf = 0.5L * std::erfc(-zLower / std::sqrt(2.0L));
    const Wide upperCdf = 0.5L * std::erfc(-zUpper / std::sqrt(2.0L));
    const Wide density = (lowerCdf - upperCdf) / (uniformUpper - uniformLower);
    return std::max(0.0L, density / innerNormalizer);
  };
  EntropyWorkLedger work;
  const auto densityAt = [&](Wide x) -> Result<Wide> {
    if (!gammaOuter) {
      const Wide low = std::max(0.0L, std::ceil(x - innerUpper));
      const Wide high = std::floor(x - innerLower);
      if (high < low)
        return Result<Wide>::success(0.0L);
      if (high - low > 64.0L) {
        return Result<Wide>::failure(ErrorCode::ResourceLimit,
                                     "nested Poisson support exceeds work cap");
      }
      Wide sum = 0.0L;
      Wide correction = 0.0L;
      for (std::uint64_t k = static_cast<std::uint64_t>(low);
           static_cast<Wide>(k) <= high; ++k) {
        if (work.evaluations >= policy.iteration.max_evaluations) {
          return Result<Wide>::failure(ErrorCode::ResourceLimit,
                                       "nested entropy work ledger exhausted");
        }
        ++work.evaluations;
        const Wide logWeight = -poissonRate +
                               static_cast<Wide>(k) * std::log(poissonRate) -
                               std::lgamma(static_cast<Wide>(k) + 1.0L);
        const Wide term = std::exp(logWeight) *
                          normalUniformDensity(x - static_cast<Wide>(k));
        const Wide adjusted = term - correction;
        const Wide next = sum + adjusted;
        correction = (next - sum) - adjusted;
        sum = next;
      }
      return Result<Wide>::success(sum / outerNormalizer);
    }
    const Wide lower = std::max(0.0L, x - innerUpper);
    const Wide upper = x - innerLower;
    if (!(upper > lower))
      return Result<Wide>::success(0.0L);
    const Wide width = upper - lower;
    const Index power =
        lower == 0.0L && outerShape < 2.0L
            ? static_cast<Index>(std::max(2.0L, std::ceil(2.0L / outerShape)))
            : 1U;
    const UnaryFunction kernel = [&](Scalar coordinate) -> Result<Scalar> {
      const Wide u = coordinate;
      const Wide offset = width * std::pow(u, static_cast<Wide>(power));
      const Wide jacobian = width * static_cast<Wide>(power) *
                            std::pow(u, static_cast<Wide>(power - 1U));
      const Wide v = lower + offset;
      const Wide base = normalUniformDensity(x - v);
      if (!(base > 0.0L) || !(v > 0.0L)) {
        return Result<Scalar>::success(0.0);
      }
      const Wide logGammaDensity = (outerShape - 1.0L) * std::log(v) -
                                   v / outerScale - std::lgamma(outerShape) -
                                   outerShape * std::log(outerScale);
      const Wide value =
          base * std::exp(logGammaDensity) * jacobian / outerNormalizer;
      if (!std::isfinite(value) || value > std::numeric_limits<Scalar>::max()) {
        return Result<Scalar>::failure(
            ErrorCode::NonFiniteResult,
            "nested Gamma density is not representable");
      }
      return Result<Scalar>::success(static_cast<Scalar>(value));
    };
    auto integral =
        quadratureEntropy31(kernel, tolerance / 32.0L, policy, work);
    if (!integral)
      return Result<Wide>::failure(integral.status().code(),
                                   std::string(integral.status().message()));
    if (integral.value().termination != Termination::converged) {
      return Result<Wide>::failure(
          ErrorCode::NoConvergence,
          "nested Gamma density did not converge within work policy");
    }
    return Result<Wide>::success(integral.value().value);
  };
  const auto runMesh = [&](bool refined) -> Result<Wide> {
    Wide sum = 0.0L;
    Wide correction = 0.0L;
    constexpr std::array<Wide, 4> breakpoints{0.0L, 1.0L, 2.0L, 3.0L};
    for (Index segment = 0U; segment + 1U < breakpoints.size(); ++segment) {
      const Wide segmentLeft = breakpoints[segment];
      const Wide segmentRight = breakpoints[segment + 1U];
      const Index pieces = refined ? 4U : 2U;
      const Wide width = segmentRight - segmentLeft;
      for (Index piece = 0U; piece < pieces; ++piece) {
        const Wide left = segmentLeft + width * piece / pieces;
        const Wide right = segmentLeft + width * (piece + 1U) / pieces;
        const Index power =
            gammaOuter && segmentLeft == 2.0L && piece == 0U ? 3U : 1U;
        const UnaryFunction entropyKernel =
            [&](Scalar coordinate) -> Result<Scalar> {
          const Wide u = coordinate;
          const Wide offset =
              (right - left) * std::pow(u, static_cast<Wide>(power));
          const Wide jacobian = (right - left) * power *
                                std::pow(u, static_cast<Wide>(power - 1U));
          auto density = densityAt(left + offset);
          if (!density)
            return Result<Scalar>::failure(
                density.status().code(),
                std::string(density.status().message()));
          if (density.value() == 0.0L)
            return Result<Scalar>::success(0.0);
          const Wide value =
              -density.value() * std::log(density.value()) * jacobian;
          if (!std::isfinite(value) ||
              std::abs(value) > std::numeric_limits<Scalar>::max()) {
            return Result<Scalar>::failure(
                ErrorCode::NoConvergence,
                "nested entropy integrand unresolved");
          }
          return Result<Scalar>::success(static_cast<Scalar>(value));
        };
        const Wide target =
            tolerance / (16.0L * (breakpoints.size() - 1U) * pieces);
        auto integral =
            quadratureEntropy31(entropyKernel, target, policy, work);
        if (!integral)
          return Result<Wide>::failure(
              integral.status().code(),
              std::string(integral.status().message()));
        if (integral.value().termination != Termination::converged) {
          return Result<Wide>::failure(ErrorCode::NoConvergence,
                                       "nested entropy panel did not converge");
        }
        const Wide adjusted = integral.value().value - correction;
        const Wide next = sum + adjusted;
        correction = (next - sum) - adjusted;
        sum = next;
      }
    }
    return Result<Wide>::success(sum);
  };
  auto coarse = runMesh(false);
  if (!coarse)
    return Result<Scalar>::failure(coarse.status().code(),
                                   std::string(coarse.status().message()));
  auto fine = runMesh(true);
  if (!fine)
    return Result<Scalar>::failure(fine.status().code(),
                                   std::string(fine.status().message()));
  const Wide innerBound = gammaOuter ? 3.0L * tolerance / 32.0L : 0.0L;
  if (std::abs(fine.value() - coarse.value()) + innerBound + tolerance / 8.0L >
      tolerance) {
    return Result<Scalar>::failure(
        ErrorCode::NoConvergence,
        "nested entropy refinement exceeds tolerance");
  }
  if (!std::isfinite(fine.value()) ||
      std::abs(fine.value()) > std::numeric_limits<Scalar>::max()) {
    return Result<Scalar>::failure(
        ErrorCode::NonFiniteResult,
        "nested entropy result is not representable");
  }
  return Result<Scalar>::success(static_cast<Scalar>(fine.value()));
}

struct BetaGammaDensity final {
  Wide value{};
  Wide error{};
};

/// @brief Integrate entropy for Beta(1,a)+Gamma(b,1) on collision-aware charts.
/// @details Distances from x=1 are passed directly to the inner kernel; the
///          nested quadrature shares one work ledger and reports no partial
///          value.
Result<Scalar> betaGammaEntropy(const probability::Distribution& law,
                                probability::IntervalSupport truncation,
                                ProbabilityPolicy policy, bool& handled) {
  using probability::Family;
  using probability::detail::DistributionSnapshot;
  handled = false;
  if (law.family() != Family::finite_convolution) {
    return Result<Scalar>::failure(ErrorCode::InvalidDomain,
                                   "not a Beta-Gamma convolution");
  }
  auto snapshot = probability::detail::DistributionEngine::snapshot(
      law, {probabilityWorkspaceElements, probabilityWorkspaceBytes,
            policy.iteration.max_evaluations});
  if (!snapshot) {
    handled = true;
    return Result<Scalar>::failure(snapshot.status().code(),
                                   std::string(snapshot.status().message()));
  }
  if (snapshot.value().components.size() != 2U) {
    return Result<Scalar>::failure(ErrorCode::InvalidDomain,
                                   "not a two-law convolution");
  }
  const DistributionSnapshot* beta = nullptr;
  const DistributionSnapshot* gamma = nullptr;
  for (const auto& child : snapshot.value().components) {
    if (child.family == Family::beta)
      beta = &child;
    if (child.family == Family::gamma)
      gamma = &child;
  }
  if (beta == nullptr || gamma == nullptr || beta->parameters.size() < 2U ||
      gamma->parameters.size() < 2U || beta->parameters[0U] != 1.0 ||
      gamma->parameters[1U] != 1.0) {
    return Result<Scalar>::failure(
        ErrorCode::InvalidDomain,
        "not a qualified Beta(1,a)+Gamma(b,1) profile");
  }
  handled = true;
  if (truncation.lower > 0.0 ||
      truncation.upper != std::numeric_limits<Scalar>::infinity() ||
      policy.iteration.bracket.has_value() ||
      !std::isfinite(policy.iteration.abs) || policy.iteration.abs <= 0.0 ||
      !std::isfinite(policy.iteration.rel) || policy.iteration.rel < 0.0 ||
      policy.iteration.max_iterations == 0U ||
      policy.iteration.max_evaluations < 31U ||
      !std::isfinite(policy.abs_log_error) || policy.abs_log_error <= 0.0) {
    return Result<Scalar>::failure(
        ErrorCode::InvalidTolerance,
        "Beta-Gamma entropy policy or support is invalid");
  }
  const Wide tolerance =
      std::min({2.0e-9L, static_cast<Wide>(policy.iteration.abs),
                static_cast<Wide>(policy.abs_log_error)});
  const Wide betaShape = beta->parameters[1U];
  const Wide gammaShape = gamma->parameters[0U];
  const Wide logGammaShape = std::lgamma(gammaShape);
  if (!(tolerance > 0.0L) || !std::isfinite(logGammaShape)) {
    return Result<Scalar>::failure(ErrorCode::InvalidTolerance,
                                   "Beta-Gamma tolerance or shape is invalid");
  }

  const auto regularizationPower = [](Wide exponent) -> Result<Index> {
    const Wide raw = std::max(2.0L, std::ceil(2.0L / exponent));
    if (!std::isfinite(raw) || raw > 1024.0L) {
      return Result<Index>::failure(
          ErrorCode::ResourceLimit,
          "singularity regularization exceeds work profile");
    }
    Index power = static_cast<Index>(raw);
    const Index last = std::min<Index>(32U, 2U * power);
    for (Index candidate = power; candidate <= last; ++candidate) {
      const Wide product = static_cast<Wide>(candidate) * exponent;
      if (product == std::floor(product))
        return Result<Index>::success(candidate);
    }
    const Wide fallback =
        std::max(static_cast<Wide>(power), std::ceil(8.0L / exponent));
    if (!std::isfinite(fallback) || fallback > 1024.0L) {
      return Result<Index>::failure(
          ErrorCode::ResourceLimit,
          "singularity regularization exceeds work profile");
    }
    return Result<Index>::success(static_cast<Index>(fallback));
  };
  auto betaPowerResult = regularizationPower(betaShape);
  auto gammaPowerResult = regularizationPower(gammaShape);
  auto originPowerResult = regularizationPower(1.0L + gammaShape);
  auto collisionPowerResult =
      regularizationPower(std::min(1.0L, betaShape + gammaShape));
  if (!betaPowerResult || !gammaPowerResult || !originPowerResult ||
      !collisionPowerResult) {
    const Status& status = !betaPowerResult     ? betaPowerResult.status()
                           : !gammaPowerResult  ? gammaPowerResult.status()
                           : !originPowerResult ? originPowerResult.status()
                                                : collisionPowerResult.status();
    return Result<Scalar>::failure(status.code(),
                                   std::string(status.message()));
  }
  const Index betaPower = betaPowerResult.value();
  const Index gammaPower = gammaPowerResult.value();
  const Index originPower = originPowerResult.value();
  const Index collisionPower = collisionPowerResult.value();
  EntropyWorkLedger work;
  const Wide relativeTarget = std::min(2.0e-12L, tolerance / 1000.0L);
  const auto densityAt = [&](bool right, Wide distance,
                             Wide upper) -> Result<BetaGammaDensity> {
    if (!(distance > 0.0L) || !(upper > 0.0L) || !std::isfinite(distance) ||
        !std::isfinite(upper)) {
      return Result<BetaGammaDensity>::failure(
          ErrorCode::NoConvergence,
          "Beta-Gamma chart lost a positive distance or integration bound");
    }
    const Wide k = right ? betaShape : gammaShape;
    const Wide j = right ? gammaShape : betaShape;
    const Wide c = std::min(distance, upper);
    if (!(c > 0.0L))
      return Result<BetaGammaDensity>::success({0.0L, 0.0L});
    const Wide collisionExponent = betaShape + gammaShape - 1.0L;
    const Wide logDensityScale =
        std::log(betaShape) - logGammaShape +
        collisionExponent * std::log(std::min(distance, 1.0L));
    const Wide densityScale =
        logDensityScale < std::log(std::numeric_limits<Wide>::max())
            ? std::exp(logDensityScale)
            : std::numeric_limits<Wide>::max();
    // The Gauss-Kronrod roundoff floor scales with the integral magnitude.
    // Scale its absolute target at the integrable collision so that a large
    // but finite density does not force futile subdivision below roundoff.
    const Wide innerTolerance = relativeTarget * std::max(1.0L, densityScale);
    Wide total = 0.0L;
    Wide correction = 0.0L;
    Wide error = 0.0L;
    const auto integrate = [&](Index power,
                               bool logarithmic) -> Result<Scalar> {
      const Wide ell =
          logarithmic ? ((upper - c) / c < 0.5L ? std::log1p((upper - c) / c)
                                                : std::log(upper) - std::log(c))
                      : 0.0L;
      const UnaryFunction kernel = [&](Scalar coordinate) -> Result<Scalar> {
        const Wide u = static_cast<Wide>(coordinate);
        if (!(u > 0.0L && u < 1.0L)) {
          return Result<Scalar>::failure(
              ErrorCode::NoConvergence,
              "open quadrature node was not preserved");
        }
        Wide z = 0.0L;
        Wide logJacobian = 0.0L;
        Wide logKernel = 0.0L;
        if (!logarithmic) {
          z = c * std::pow(u, static_cast<Wide>(power));
          logJacobian = std::log(static_cast<Wide>(power));
          logKernel = std::log(betaShape) - logGammaShape + k * std::log(c) +
                      (static_cast<Wide>(power) * k - 1.0L) * std::log(u) +
                      (j - 1.0L) * std::log(distance + z) -
                      (right ? distance + z : z) + logJacobian;
        } else {
          const Wide logZ = std::log(c) + u * ell;
          z = std::exp(logZ);
          logKernel = std::log(betaShape) - logGammaShape + k * logZ +
                      (j - 1.0L) * std::log(distance + z) -
                      (right ? distance + z : z) + std::log(ell);
        }
        const Wide value = std::exp(logKernel);
        if (!std::isfinite(value) ||
            value > static_cast<Wide>(std::numeric_limits<Scalar>::max())) {
          return Result<Scalar>::failure(
              ErrorCode::NonFiniteResult,
              "Beta-Gamma density kernel is not representable");
        }
        return Result<Scalar>::success(static_cast<Scalar>(value));
      };
      auto estimate = quadratureEntropy31(kernel, innerTolerance, policy, work);
      if (!estimate)
        return Result<Scalar>::failure(
            estimate.status().code(), std::string(estimate.status().message()));
      if (estimate.value().termination != Termination::converged) {
        return Result<Scalar>::failure(
            ErrorCode::ResourceLimit,
            "Beta-Gamma inner quadrature exhausted its work cap");
      }
      const Wide adjusted =
          static_cast<Wide>(estimate.value().value) - correction;
      const Wide next = total + adjusted;
      correction = (next - total) - adjusted;
      total = next;
      error += static_cast<Wide>(estimate.value().error_estimate);
      return Result<Scalar>::success(0.0);
    };
    const Index firstPower = right ? betaPower : gammaPower;
    auto first = integrate(firstPower, false);
    if (!first)
      return Result<BetaGammaDensity>::failure(
          first.status().code(), std::string(first.status().message()));
    if (c < upper) {
      auto remainder = integrate(1U, true);
      if (!remainder)
        return Result<BetaGammaDensity>::failure(
            remainder.status().code(),
            std::string(remainder.status().message()));
    }
    if (!std::isfinite(total) || total < 0.0L) {
      return Result<BetaGammaDensity>::failure(
          ErrorCode::NoConvergence, "Beta-Gamma density is unresolved");
    }
    return Result<BetaGammaDensity>::success({total, error});
  };

  Wide distance = std::max(4.0L, 2.0L * std::max(0.0L, gammaShape - 1.0L));
  const Wide tailBudget = tolerance / 8.0L;
  for (Index attempt = 0U;
       attempt < 64U &&
       gammaSpatialEntropyBound(gammaShape, 1.0L, distance) > tailBudget;
       ++attempt) {
    if (work.iterations >= policy.iteration.max_iterations ||
        work.evaluations >= policy.iteration.max_evaluations) {
      return Result<Scalar>::failure(
          ErrorCode::ResourceLimit,
          "Beta-Gamma tail search exhausted its work cap");
    }
    ++work.iterations;
    distance *= 2.0L;
  }
  const Wide tailError = gammaSpatialEntropyBound(gammaShape, 1.0L, distance);
  if (!(tailError <= tailBudget) || !std::isfinite(distance)) {
    return Result<Scalar>::failure(
        ErrorCode::NoConvergence,
        "Beta-Gamma tail allowance was not established");
  }
  if (distance > static_cast<Wide>(probabilityWorkspaceElements - 4U)) {
    return Result<Scalar>::failure(
        ErrorCode::ResourceLimit,
        "Beta-Gamma tail panel plan exceeds workspace");
  }
  std::vector<Wide> tailBreaks;
  try {
    tailBreaks.reserve(64U);
    tailBreaks.push_back(2.0L);
  } catch (const std::bad_alloc&) {
    return Result<Scalar>::failure(ErrorCode::ResourceLimit,
                                   "Beta-Gamma tail plan allocation failed");
  }
  for (Wide edge = 4.0L; edge < 1.0L + distance; edge *= 2.0L) {
    if (tailBreaks.size() >= probabilityWorkspaceElements / 2U ||
        tailBreaks.size() * sizeof(Wide) >= probabilityWorkspaceBytes) {
      return Result<Scalar>::failure(
          ErrorCode::ResourceLimit,
          "Beta-Gamma tail panel plan exceeds workspace");
    }
    try {
      tailBreaks.push_back(edge);
    } catch (const std::bad_alloc&) {
      return Result<Scalar>::failure(ErrorCode::ResourceLimit,
                                     "Beta-Gamma tail plan allocation failed");
    }
  }
  if (tailBreaks.back() < 1.0L + distance) {
    try {
      tailBreaks.push_back(1.0L + distance);
    } catch (const std::bad_alloc&) {
      return Result<Scalar>::failure(ErrorCode::ResourceLimit,
                                     "Beta-Gamma tail plan allocation failed");
    }
  }

  struct Chart final {
    bool right{};
    Wide left{};
    Wide width{};
    Index power{};
    bool tail{};
  };
  std::vector<Chart> charts;
  try {
    charts.reserve(tailBreaks.size() + 3U);
    charts.push_back({false, 0.0L, 0.5L, originPower, false});
    charts.push_back({false, 0.0L, 0.5L, collisionPower, false});
    charts.push_back({true, 0.0L, 1.0L, collisionPower, false});
    for (std::size_t i = 0U; i + 1U < tailBreaks.size(); ++i) {
      charts.push_back({true, tailBreaks[i] - 1.0L,
                        tailBreaks[i + 1U] - tailBreaks[i], 1U, true});
    }
  } catch (const std::bad_alloc&) {
    return Result<Scalar>::failure(ErrorCode::ResourceLimit,
                                   "Beta-Gamma chart allocation failed");
  }
  if (charts.size() > probabilityWorkspaceElements ||
      charts.size() > probabilityWorkspaceBytes / sizeof(Chart)) {
    return Result<Scalar>::failure(ErrorCode::ResourceLimit,
                                   "Beta-Gamma chart plan exceeds workspace");
  }
  const auto chartPoint = [&](const Chart& chart, Wide t) {
    struct Point final {
      Wide distance;
      Wide upper;
      Wide jacobian;
    };
    if (chart.tail)
      return Point{chart.left + chart.width * t, 1.0L, chart.width};
    const Wide offset =
        chart.width * std::pow(t, static_cast<Wide>(chart.power));
    const Wide jacobian = chart.width * chart.power *
                          std::pow(t, static_cast<Wide>(chart.power - 1U));
    if (&chart == &charts[0])
      return Point{1.0L - offset, offset, jacobian};
    if (&chart == &charts[1])
      return Point{offset, 1.0L - offset, jacobian};
    return Point{offset, 1.0L, jacobian};
  };
  const auto evaluateChartPanel =
      [&](const Chart& chart, Wide left,
          Wide width) -> Result<EntropyPairEstimate> {
    const auto integrand =
        [&](Scalar coordinate) -> Result<std::array<Scalar, 2>> {
      const Wide t = left + width * static_cast<Wide>(coordinate);
      const auto point = chartPoint(chart, t);
      if (!(point.jacobian > 0.0L)) {
        return Result<std::array<Scalar, 2>>::success({0.0, 0.0});
      }
      auto density = densityAt(chart.right, point.distance, point.upper);
      if (!density)
        return Result<std::array<Scalar, 2>>::failure(
            density.status().code(), std::string(density.status().message()));
      const Wide f = density.value().value;
      if (f == 0.0L)
        return Result<std::array<Scalar, 2>>::success({0.0, 0.0});
      const Wide entropy = -f * std::log(f) * point.jacobian * width;
      const Wide lower = std::max(0.0L, f - density.value().error);
      const Wide upper = f + density.value().error;
      const auto h = [](Wide value) {
        return value > 0.0L ? -value * std::log(value) : 0.0L;
      };
      Wide allowance =
          std::max(std::abs(h(lower) - h(f)), std::abs(h(upper) - h(f)));
      if (lower <= std::exp(-1.0L) && upper >= std::exp(-1.0L)) {
        allowance = std::max(allowance, std::abs(std::exp(-1.0L) - h(f)));
      }
      allowance *= point.jacobian * width;
      if (!std::isfinite(entropy) || !std::isfinite(allowance) ||
          std::abs(entropy) > std::numeric_limits<Scalar>::max() ||
          allowance > std::numeric_limits<Scalar>::max()) {
        return Result<std::array<Scalar, 2>>::failure(
            ErrorCode::NoConvergence,
            "Beta-Gamma entropy sample is unresolved");
      }
      return Result<std::array<Scalar, 2>>::success(
          {static_cast<Scalar>(entropy), static_cast<Scalar>(allowance)});
    };
    return quadratureEntropyPair31(integrand, tolerance / 64.0L, policy, work);
  };
  struct RefinedCell final {
    Index chartIndex{};
    Wide left{};
    Wide width{};
    EntropyPairEstimate parent{};
    EntropyPairEstimate childLeft{};
    EntropyPairEstimate childRight{};
    Wide discrepancy{};
  };
  std::vector<RefinedCell> cells;
  const auto splitCell =
      [&](Index chartIndex, Wide left, Wide width,
          EntropyPairEstimate parent) -> Result<RefinedCell> {
    if (work.iterations >= policy.iteration.max_iterations) {
      return Result<RefinedCell>::failure(
          ErrorCode::ResourceLimit, "Beta-Gamma refinement ledger exhausted");
    }
    ++work.iterations;
    const Wide half = width / 2.0L;
    if (!(half > 0.0L) || left + half == left || left + width == left + half) {
      return Result<RefinedCell>::failure(
          ErrorCode::NoConvergence, "Beta-Gamma chart can no longer be split");
    }
    auto leftChild = evaluateChartPanel(charts[chartIndex], left, half);
    auto rightChild = evaluateChartPanel(charts[chartIndex], left + half, half);
    if (!leftChild || !rightChild) {
      const Status& status =
          leftChild ? rightChild.status() : leftChild.status();
      return Result<RefinedCell>::failure(status.code(),
                                          std::string(status.message()));
    }
    const Wide childSum = leftChild.value().value + rightChild.value().value;
    return Result<RefinedCell>::success({chartIndex, left, width, parent,
                                         leftChild.value(), rightChild.value(),
                                         std::abs(parent.value - childSum)});
  };
  constexpr Index initialPieces = 16U;
  if (charts.size() > probabilityWorkspaceElements / initialPieces ||
      charts.size() >
          probabilityWorkspaceBytes / sizeof(RefinedCell) / initialPieces) {
    return Result<Scalar>::failure(
        ErrorCode::ResourceLimit,
        "Beta-Gamma initial refinement plan exceeds workspace");
  }
  try {
    cells.reserve(charts.size() * initialPieces);
  } catch (const std::bad_alloc&) {
    return Result<Scalar>::failure(ErrorCode::ResourceLimit,
                                   "Beta-Gamma refinement storage exhausted");
  }
  for (Index chartIndex = 0U; chartIndex < charts.size(); ++chartIndex) {
    for (Index piece = 0U; piece < initialPieces; ++piece) {
      const Wide left = static_cast<Wide>(piece) / initialPieces;
      const Wide width = 1.0L / initialPieces;
      auto parent = evaluateChartPanel(charts[chartIndex], left, width);
      if (!parent)
        return Result<Scalar>::failure(parent.status().code(),
                                       std::string(parent.status().message()));
      auto cell = splitCell(chartIndex, left, width, parent.value());
      if (!cell)
        return Result<Scalar>::failure(cell.status().code(),
                                       std::string(cell.status().message()));
      cells.push_back(cell.value());
    }
  }

  Wide total = 0.0L;
  Wide panelError = 0.0L;
  Wide discrepancy = 0.0L;
  Wide densityAllowance = 0.0L;
  Wide densityPanelError = 0.0L;
  Wide magnitude = 0.0L;
  Wide rounding = 0.0L;
  for (;;) {
    Wide positive = 0.0L;
    Wide positiveCorrection = 0.0L;
    Wide negative = 0.0L;
    Wide negativeCorrection = 0.0L;
    panelError = 0.0L;
    discrepancy = 0.0L;
    densityAllowance = 0.0L;
    densityPanelError = 0.0L;
    magnitude = 0.0L;
    for (const RefinedCell& cell : cells) {
      for (const EntropyPairEstimate* child :
           {&cell.childLeft, &cell.childRight}) {
        Wide& sum = child->value >= 0.0L ? positive : negative;
        Wide& correction =
            child->value >= 0.0L ? positiveCorrection : negativeCorrection;
        const Wide signedTerm =
            child->value >= 0.0L ? child->value : -child->value;
        const Wide adjusted = signedTerm - correction;
        const Wide next = sum + adjusted;
        correction = (next - sum) - adjusted;
        sum = next;
        panelError += child->error;
        densityAllowance += child->auxiliary;
        densityPanelError += child->auxiliaryError;
        magnitude += std::abs(child->value);
      }
      discrepancy += cell.discrepancy;
    }
    total = positive - negative;
    rounding = 64.0L * std::numeric_limits<Scalar>::epsilon() *
               (magnitude + positive + negative);
    const bool panelQualified = panelError <= tolerance / 4.0L;
    const bool refinementQualified = discrepancy <= tolerance / 4.0L;
    const bool densityQualified =
        densityAllowance + densityPanelError + rounding <= tolerance / 4.0L;
    if (tailError <= tolerance / 8.0L && panelQualified &&
        refinementQualified && densityQualified)
      break;
    if (cells.size() >= probabilityWorkspaceElements ||
        cells.size() >= probabilityWorkspaceBytes / sizeof(RefinedCell)) {
      return Result<Scalar>::failure(
          ErrorCode::ResourceLimit,
          "Beta-Gamma live refinement cells exceed workspace");
    }
    const auto selected = std::max_element(
        cells.begin(), cells.end(),
        [](const RefinedCell& first, const RefinedCell& second) {
          const Wide firstPriority = first.childLeft.error +
                                     first.childRight.error + first.discrepancy;
          const Wide secondPriority = second.childLeft.error +
                                      second.childRight.error +
                                      second.discrepancy;
          if (firstPriority != secondPriority)
            return firstPriority < secondPriority;
          if (first.chartIndex != second.chartIndex) {
            return first.chartIndex > second.chartIndex;
          }
          return first.left > second.left;
        });
    const Index chartIndex = selected->chartIndex;
    const Wide left = selected->left;
    const Wide half = selected->width / 2.0L;
    const EntropyPairEstimate parentLeft = selected->childLeft;
    const EntropyPairEstimate parentRight = selected->childRight;
    auto refinedLeft = splitCell(chartIndex, left, half, parentLeft);
    if (!refinedLeft)
      return Result<Scalar>::failure(
          refinedLeft.status().code(),
          std::string(refinedLeft.status().message()));
    auto refinedRight = splitCell(chartIndex, left + half, half, parentRight);
    if (!refinedRight)
      return Result<Scalar>::failure(
          refinedRight.status().code(),
          std::string(refinedRight.status().message()));
    *selected = refinedLeft.value();
    try {
      cells.push_back(refinedRight.value());
    } catch (const std::bad_alloc&) {
      return Result<Scalar>::failure(ErrorCode::ResourceLimit,
                                     "Beta-Gamma refinement storage exhausted");
    }
  }

  if (!std::isfinite(total) || tailError > tolerance / 8.0L ||
      panelError > tolerance / 4.0L || discrepancy > tolerance / 4.0L ||
      densityAllowance + densityPanelError + rounding > tolerance / 4.0L) {
    return Result<Scalar>::failure(
        ErrorCode::NoConvergence,
        "Beta-Gamma entropy error ratios tail=" +
            std::to_string(static_cast<Scalar>(tailError / tolerance)) +
            ", panel=" +
            std::to_string(static_cast<Scalar>(panelError / tolerance)) +
            ", refinement=" +
            std::to_string(static_cast<Scalar>(discrepancy / tolerance)) +
            ", density=" +
            std::to_string(static_cast<Scalar>(
                (densityAllowance + densityPanelError + rounding) /
                tolerance)));
  }
  if (std::abs(total) > std::numeric_limits<Scalar>::max()) {
    return Result<Scalar>::failure(
        ErrorCode::NonFiniteResult,
        "Beta-Gamma entropy result is not representable");
  }
  return Result<Scalar>::success(static_cast<Scalar>(total));
}

/// @brief Qualify entropy of a Poisson translation of a Normal or Gamma law.
/// @details A count-mixture truncation bound controls omitted Poisson labels;
///          spatial tails use monotone exponential envelopes. Every translated
///          Gamma pole is a cell boundary and is power-regularized before the
///          shared adaptive quadrature evaluates it.
Result<Scalar> poissonTranslatedEntropy(const probability::Distribution& law,
                                        probability::IntervalSupport truncation,
                                        ProbabilityPolicy policy,
                                        bool& handled) {
  using probability::Family;
  using probability::detail::DistributionSnapshot;
  handled = false;
  if (law.family() != Family::finite_convolution) {
    return Result<Scalar>::failure(ErrorCode::InvalidDomain,
                                   "not a translated finite convolution");
  }
  auto snapshot = probability::detail::DistributionEngine::snapshot(
      law, {probabilityWorkspaceElements, probabilityWorkspaceBytes,
            policy.iteration.max_evaluations});
  if (!snapshot) {
    handled = true;
    return Result<Scalar>::failure(snapshot.status().code(),
                                   std::string(snapshot.status().message()));
  }
  if (snapshot.value().components.size() != 2U) {
    return Result<Scalar>::failure(ErrorCode::InvalidDomain,
                                   "not a two-law translated convolution");
  }
  const DistributionSnapshot* count = nullptr;
  const DistributionSnapshot* base = nullptr;
  for (const auto& child : snapshot.value().components) {
    if (child.family == Family::poisson)
      count = &child;
    else if (child.family == Family::normal || child.family == Family::gamma) {
      base = &child;
    }
  }
  if (count == nullptr || base == nullptr || count->parameters.size() != 1U) {
    return Result<Scalar>::failure(ErrorCode::InvalidDomain,
                                   "not a supported Poisson translation");
  }
  handled = true;
  if (policy.iteration.bracket.has_value() ||
      !std::isfinite(policy.iteration.abs) || policy.iteration.abs < 0.0 ||
      !std::isfinite(policy.iteration.rel) || policy.iteration.rel < 0.0 ||
      policy.iteration.max_iterations == 0U ||
      policy.iteration.max_evaluations < 15U ||
      !std::isfinite(policy.abs_log_error) || policy.abs_log_error <= 0.0) {
    return Result<Scalar>::failure(ErrorCode::InvalidTolerance,
                                   "entropy integration policy is invalid");
  }
  const Wide tolerance =
      std::min({2.0e-9L, static_cast<Wide>(policy.iteration.abs),
                static_cast<Wide>(policy.abs_log_error)});
  if (!(tolerance > 0.0L)) {
    return Result<Scalar>::failure(
        ErrorCode::NoConvergence, "requested zero entropy error is unresolved");
  }
  const Wide rate = count->parameters[0U];
  if (!(rate > 0.0L)) {
    handled = false;
    return Result<Scalar>::failure(ErrorCode::InvalidDomain,
                                   "zero-rate translation uses the base law");
  }
  const bool gammaBase = base->family == Family::gamma;
  if (base->parameters.size() < 2U ||
      (truncation.lower >
       (gammaBase ? 0.0 : -std::numeric_limits<Scalar>::infinity())) ||
      truncation.upper != std::numeric_limits<Scalar>::infinity() ||
      std::isnan(truncation.lower)) {
    return Result<Scalar>::failure(
        ErrorCode::InvalidDomain,
        "entropy interval omits translated-law support");
  }

  const Wide countBudget = tolerance / 8.0L;
  std::size_t lastCount = 0U;
  while (lastCount < probabilityWorkspaceElements - 1U &&
         poissonOmissionBound(rate, lastCount) > countBudget) {
    ++lastCount;
  }
  const Wide countError = poissonOmissionBound(rate, lastCount);
  if (!(countError <= countBudget)) {
    return Result<Scalar>::failure(
        ErrorCode::NoConvergence,
        "Poisson entropy tail bound exceeds tolerance");
  }
  const Wide shape = gammaBase ? base->parameters[0U] : 0.0L;
  const Wide scale = gammaBase ? base->parameters[1U] : 0.0L;
  const Wide mean = gammaBase ? 0.0L : base->parameters[0U];
  const Wide standardDeviation = gammaBase ? 0.0L : base->parameters[1U];
  Wide distance =
      gammaBase ? std::max(4.0L * scale,
                           shape > 1.0L ? 2.0L * (shape - 1.0L) * scale : scale)
                : 4.0L * standardDeviation;
  const Wide spatialBudget = tolerance / 8.0L;
  const auto spatialBound = [&] {
    return gammaBase ? gammaSpatialEntropyBound(shape, scale, distance)
                     : normalSpatialEntropyBound(standardDeviation, distance);
  };
  for (Index attempt = 0U; attempt < 128U && spatialBound() > spatialBudget;
       ++attempt) {
    distance *= 1.2L;
    if (!std::isfinite(distance))
      break;
  }
  const Wide spatialError = spatialBound();
  if (!(spatialError <= spatialBudget) ||
      static_cast<Wide>(lastCount) + distance >
          static_cast<Wide>(probabilityWorkspaceElements)) {
    return Result<Scalar>::failure(
        ErrorCode::NoConvergence,
        "spatial entropy tail bound exceeds tolerance or workspace");
  }

  std::vector<Wide> logWeights;
  try {
    logWeights.reserve(lastCount + 1U);
    for (std::size_t k = 0U; k <= lastCount; ++k) {
      logWeights.push_back(-rate + static_cast<Wide>(k) * std::log(rate) -
                           std::lgamma(static_cast<Wide>(k) + 1.0L));
    }
  } catch (const std::bad_alloc&) {
    return Result<Scalar>::failure(ErrorCode::ResourceLimit,
                                   "Poisson entropy weights allocation failed");
  }
  if (logWeights.size() > probabilityWorkspaceBytes / sizeof(Wide)) {
    return Result<Scalar>::failure(ErrorCode::ResourceLimit,
                                   "Poisson entropy weights exceed byte limit");
  }
  Wide retainedMass = 0.0L;
  Wide retainedCorrection = 0.0L;
  for (const Wide logWeight : logWeights) {
    const Wide term = std::exp(logWeight);
    const Wide adjusted = term - retainedCorrection;
    const Wide next = retainedMass + adjusted;
    retainedCorrection = (next - retainedMass) - adjusted;
    retainedMass = next;
  }
  if (!(retainedMass > 0.0L && retainedMass <= 1.0L)) {
    return Result<Scalar>::failure(ErrorCode::NonFiniteResult,
                                   "retained Poisson mass is invalid");
  }
  const Wide logRetainedMass = std::log(retainedMass);
  const Wide domainLower = gammaBase ? 0.0L : mean - distance;
  const Wide upper = gammaBase ? static_cast<Wide>(lastCount) + distance
                               : mean + static_cast<Wide>(lastCount) + distance;
  const Wide firstCell = std::floor(domainLower);
  const Wide rawCellCount = std::ceil(upper - firstCell);
  if (!(rawCellCount > 0.0L) ||
      rawCellCount > static_cast<Wide>(probabilityWorkspaceElements)) {
    return Result<Scalar>::failure(ErrorCode::ResourceLimit,
                                   "entropy panel plan exceeds workspace");
  }
  const std::size_t fullCells = static_cast<std::size_t>(rawCellCount);
  if (fullCells == 0U || fullCells > probabilityWorkspaceElements ||
      fullCells > probabilityWorkspaceBytes / (8U * sizeof(Wide))) {
    return Result<Scalar>::failure(ErrorCode::ResourceLimit,
                                   "entropy panel plan exceeds workspace");
  }

  EntropyWorkLedger work;
  Wide estimatedError = 0.0L;
  const auto addCompensated = [](Wide value, Wide& sum, Wide& correction) {
    const Wide adjusted = value - correction;
    const Wide next = sum + adjusted;
    correction = (next - sum) - adjusted;
    sum = next;
  };
  const auto integrateSegment = [&](Wide cellLeft, Wide segmentLeft,
                                    Wide segmentRight, Index power,
                                    Wide target) -> Result<IntegralResult> {
    const Wide width = segmentRight - segmentLeft;
    if (!(width > 0.0L)) {
      return Result<IntegralResult>::success(
          {0.0, 0.0, 0U, 0U, Termination::converged});
    }
    const UnaryFunction integrand = [&](Scalar coordinate) -> Result<Scalar> {
      const Wide u = coordinate;
      const Wide offset = width * std::pow(u, static_cast<Wide>(power));
      const Wide jacobian = width * static_cast<Wide>(power) *
                            std::pow(u, static_cast<Wide>(power - 1U));
      const Wide x = segmentLeft + offset;
      Wide logDensity = -std::numeric_limits<Wide>::infinity();
      const std::size_t maximumCount =
          gammaBase
              ? std::min(lastCount, static_cast<std::size_t>(std::max<Wide>(
                                        0.0L, std::floor(cellLeft))))
              : lastCount;
      const Index terms = static_cast<Index>(maximumCount + 1U);
      if (work.evaluations >= policy.iteration.max_evaluations ||
          terms > policy.iteration.max_evaluations - work.evaluations) {
        return Result<Scalar>::failure(ErrorCode::ResourceLimit,
                                       "entropy work ledger exhausted");
      }
      for (std::size_t k = 0U; k <= maximumCount; ++k) {
        ++work.evaluations;
        Wide residual = 0.0L;
        Wide logBase = -std::numeric_limits<Wide>::infinity();
        if (gammaBase) {
          residual = (segmentLeft - static_cast<Wide>(k)) + offset;
          if (residual > 0.0L) {
            logBase = (shape - 1.0L) * std::log(residual) - residual / scale -
                      std::lgamma(shape) - shape * std::log(scale);
          }
        } else {
          residual = x - mean - static_cast<Wide>(k);
          const Wide z = residual / standardDeviation;
          logBase = -0.5L * z * z - std::log(standardDeviation) -
                    0.9189385332046727417803297364056176L;
        }
        if (logBase != -std::numeric_limits<Wide>::infinity()) {
          logDensity =
              logAddWide(logDensity, logWeights[k] - logRetainedMass + logBase);
        }
      }
      if (work.evaluations > policy.iteration.max_evaluations) {
        return Result<Scalar>::failure(ErrorCode::ResourceLimit,
                                       "entropy work ledger exhausted");
      }
      if (logDensity == -std::numeric_limits<Wide>::infinity()) {
        return Result<Scalar>::success(0.0);
      }
      const Wide density = std::exp(logDensity);
      const Wide value = -density * logDensity * jacobian;
      if (!std::isfinite(value) ||
          std::abs(value) > std::numeric_limits<Scalar>::max()) {
        return Result<Scalar>::failure(ErrorCode::NoConvergence,
                                       "entropy integrand is unresolved");
      }
      return Result<Scalar>::success(static_cast<Scalar>(value));
    };
    if (work.evaluations >= policy.iteration.max_evaluations ||
        work.iterations >= policy.iteration.max_iterations) {
      return Result<IntegralResult>::failure(ErrorCode::ResourceLimit,
                                             "entropy work ledger exhausted");
    }
    auto result = quadratureEntropy31(integrand, target, policy, work);
    if (!result)
      return result;
    return result;
  };

  const auto runMesh = [&](bool refined) -> Result<Wide> {
    Wide sum = 0.0L;
    Wide correction = 0.0L;
    for (std::size_t cell = 0U; cell < fullCells; ++cell) {
      const Wide cellLeft = firstCell + static_cast<Wide>(cell);
      const Wide cellRight = std::min(cellLeft + 1.0L, upper);
      if (!(cellRight > cellLeft))
        continue;
      const bool singular =
          gammaBase && cell <= lastCount && shape < 2.0L && shape != 1.0L;
      Index power = 1U;
      if (singular) {
        power = static_cast<Index>(std::max(2.0L, std::ceil(2.0L / shape)));
      }
      const Wide width = cellRight - cellLeft;
      const Index pieces = refined ? 4U : 2U;
      for (Index piece = 0U; piece < pieces; ++piece) {
        const Wide left = cellLeft + width * static_cast<Wide>(piece) / pieces;
        const Wide right =
            cellLeft + width * static_cast<Wide>(piece + 1U) / pieces;
        const Index segmentPower = singular && piece == 0U ? power : 1U;
        const Wide target =
            tolerance / (16.0L * static_cast<Wide>(fullCells) * pieces);
        auto integral =
            integrateSegment(cellLeft, left, right, segmentPower, target);
        if (!integral)
          return Result<Wide>::failure(
              integral.status().code(),
              std::string(integral.status().message()));
        if (integral.value().termination != Termination::converged) {
          return Result<Wide>::failure(
              ErrorCode::NoConvergence,
              "entropy panel did not converge: cell=" + std::to_string(cell) +
                  " error=" + std::to_string(integral.value().error_estimate) +
                  " target=" + std::to_string(static_cast<Scalar>(target)) +
                  " leaves=" + std::to_string(integral.value().panels) +
                  " iters=" + std::to_string(work.iterations) + "/" +
                  std::to_string(policy.iteration.max_iterations));
        }
        estimatedError += integral.value().error_estimate;
        addCompensated(integral.value().value, sum, correction);
      }
    }
    return Result<Wide>::success(sum);
  };
  const Wide errorBefore = estimatedError;
  auto coarse = runMesh(false);
  if (!coarse)
    return Result<Scalar>::failure(coarse.status().code(),
                                   std::string(coarse.status().message()));
  const Wide coarseError = estimatedError - errorBefore;
  auto fine = runMesh(true);
  if (!fine)
    return Result<Scalar>::failure(fine.status().code(),
                                   std::string(fine.status().message()));
  const Wide fineError = estimatedError - errorBefore - coarseError;
  const Wide discrepancy = std::abs(fine.value() - coarse.value());
  const Wide totalError =
      countError + spatialError + coarseError + fineError + discrepancy;
  if (totalError > tolerance || !std::isfinite(fine.value()) ||
      std::abs(fine.value()) > std::numeric_limits<Scalar>::max()) {
    return Result<Scalar>::failure(
        ErrorCode::NoConvergence,
        "entropy error and tail budget is unresolved");
  }
  return Result<Scalar>::success(static_cast<Scalar>(fine.value()));
}

Result<Scalar> transformedIntegral(const probability::Distribution& p,
                                   const probability::Distribution* q,
                                   probability::IntervalSupport support,
                                   ProbabilityPolicy policy) {
  if (policy.iteration.bracket.has_value() ||
      !std::isfinite(policy.iteration.abs) || policy.iteration.abs < 0.0 ||
      !std::isfinite(policy.iteration.rel) || policy.iteration.rel < 0.0 ||
      policy.iteration.max_iterations == 0U ||
      policy.iteration.max_evaluations == 0U ||
      !std::isfinite(policy.abs_log_error) || policy.abs_log_error <= 0.0) {
    return Result<Scalar>::failure(ErrorCode::InvalidTolerance,
                                   "entropy integration policy is invalid");
  }
  auto bounds = fullSupportBounds(p, support, policy);
  if (!bounds)
    return Result<Scalar>::failure(bounds.status().code(),
                                   std::string(bounds.status().message()));
  if (q != nullptr) {
    auto qBounds = fullSupportBounds(*q, support, policy);
    if (!qBounds && q->family() == p.family() &&
        qBounds.status().code() != ErrorCode::InvalidDomain) {
      return Result<Scalar>::failure(qBounds.status().code(),
                                     std::string(qBounds.status().message()));
    }
  }
  const Scalar lower = bounds.value().first;
  const Scalar upper = bounds.value().second;
  const bool lowerInfinite = !std::isfinite(lower);
  const bool upperInfinite = !std::isfinite(upper);
  const UnaryFunction integrand = [=](Scalar t) -> Result<Scalar> {
    Scalar x = 0.0;
    Wide jacobian = 1.0L;
    if (lowerInfinite && upperInfinite) {
      const Wide angle =
          std::numbers::pi_v<Wide> * (static_cast<Wide>(t) - 0.5L);
      x = static_cast<Scalar>(std::tan(angle));
      const Wide cosine = std::cos(angle);
      jacobian = std::numbers::pi_v<Wide> / (cosine * cosine);
    } else if (upperInfinite) {
      const Wide oneMinus = 1.0L - t;
      const Wide t2 = static_cast<Wide>(t) * t;
      const Wide t3 = t2 * t;
      const Wide t4 = t3 * t;
      x = static_cast<Scalar>(static_cast<Wide>(lower) + t4 / oneMinus);
      jacobian = t3 * (4.0L - 3.0L * t) / (oneMinus * oneMinus);
    } else if (lowerInfinite) {
      const Wide oneMinus = 1.0L - t;
      const Wide t2 = static_cast<Wide>(t) * t;
      const Wide t3 = oneMinus * oneMinus * oneMinus;
      x = static_cast<Scalar>(static_cast<Wide>(upper) -
                              oneMinus * oneMinus * oneMinus * oneMinus / t);
      jacobian = t3 * (1.0L + 3.0L * t) / (t2 * t);
    } else {
      const Wide halfAngle =
          std::numbers::pi_v<Wide> * static_cast<Wide>(t) / 2.0L;
      const Wide sine = std::sin(halfAngle);
      const Wide cosine = std::cos(halfAngle);
      const Wide sine2 = sine * sine;
      const Wide sine3 = sine2 * sine;
      const Wide sine4 = sine2 * sine2;
      const Wide width = static_cast<Wide>(upper) - lower;
      x = static_cast<Scalar>(static_cast<Wide>(lower) + width * sine4);
      jacobian = width * 2.0L * std::numbers::pi_v<Wide> * sine3 * cosine;
    }
    if (!std::isfinite(x) || !std::isfinite(jacobian)) {
      return Result<Scalar>::failure(ErrorCode::NoConvergence,
                                     "support transformation exceeded range");
    }
    auto pValue = probability::evaluate(p, x, policy);
    if (!pValue)
      return Result<Scalar>::failure(pValue.status().code(),
                                     std::string(pValue.status().message()));
    if (std::isinf(pValue.value().log_density_or_mass)) {
      // A selected density representative can be infinite at an integrable
      // isolated pole. Its value at that single point does not change a
      // Lebesgue log-integral; nearby panels resolve its contribution.
      return Result<Scalar>::success(0.0);
    }
    if (pValue.value().density_or_mass == 0.0) {
      return Result<Scalar>::success(0.0);
    }
    Wide value = 0.0L;
    if (q == nullptr) {
      value = -static_cast<Wide>(pValue.value().density_or_mass) *
              pValue.value().log_density_or_mass;
    } else {
      auto qValue = probability::evaluate(*q, x, policy);
      if (!qValue)
        return Result<Scalar>::failure(qValue.status().code(),
                                       std::string(qValue.status().message()));
      if (qValue.value().density_or_mass == 0.0) {
        return Result<Scalar>::failure(
            ErrorCode::NoConvergence,
            "KL cross-entropy needs support analysis");
      }
      if (std::isinf(qValue.value().log_density_or_mass)) {
        return Result<Scalar>::success(0.0);
      }
      value = static_cast<Wide>(pValue.value().density_or_mass) *
              (static_cast<Wide>(pValue.value().log_density_or_mass) -
               qValue.value().log_density_or_mass);
    }
    value *= jacobian;
    if (!std::isfinite(value) ||
        std::abs(value) > std::numeric_limits<Scalar>::max()) {
      return Result<Scalar>::failure(
          ErrorCode::NoConvergence, "log-integrand is unresolved at this node");
    }
    return Result<Scalar>::success(static_cast<Scalar>(value));
  };
  IterPolicy integrationPolicy = policy.iteration;
  integrationPolicy.abs = std::min(integrationPolicy.abs, policy.abs_log_error);
  const Limits resources{probabilityWorkspaceElements,
                         probabilityWorkspaceBytes,
                         policy.iteration.max_evaluations};
  Result<IntegralResult> integral = Result<IntegralResult>::failure(
      ErrorCode::NoConvergence, "integral not evaluated");
  integral =
      quadratureAdaptive(integrand, {0.0, 1.0}, integrationPolicy, resources);
  if (!integral)
    return Result<Scalar>::failure(integral.status().code(),
                                   std::string(integral.status().message()));
  const Wide allowed = static_cast<Wide>(policy.abs_log_error) +
                       static_cast<Wide>(policy.iteration.rel) *
                           std::abs(integral.value().value);
  if (integral.value().termination != Termination::converged ||
      integral.value().error_estimate > allowed) {
    return Result<Scalar>::failure(
        ErrorCode::NoConvergence,
        "log-integral error estimate exceeds tolerance");
  }
  return Result<Scalar>::success(integral.value().value);
}

} // namespace

Result<Scalar> numericalEntropy(const probability::Distribution& law,
                                probability::IntervalSupport truncation,
                                ProbabilityPolicy policy) {
  bool betaGammaHandled = false;
  auto betaGamma = betaGammaEntropy(law, truncation, policy, betaGammaHandled);
  if (betaGammaHandled)
    return betaGamma;
  bool nestedHandled = false;
  auto nested =
      nestedConditionedEntropy(law, truncation, policy, nestedHandled);
  if (nestedHandled)
    return nested;
  bool translatedHandled = false;
  auto translated =
      poissonTranslatedEntropy(law, truncation, policy, translatedHandled);
  if (translatedHandled)
    return translated;
  auto bounds = fullSupportBounds(law, truncation, policy);
  if (!bounds)
    return Result<Scalar>::failure(bounds.status().code(),
                                   std::string(bounds.status().message()));
  const auto atoms = law.finiteAtoms();
  if (!atoms.locations.empty()) {
    Wide entropy = 0.0L;
    for (const Scalar mass : atoms.masses) {
      if (mass > 0.0)
        entropy -= static_cast<Wide>(mass) * std::log(mass);
    }
    return Result<Scalar>::success(static_cast<Scalar>(entropy));
  }
  if (law.family() == probability::Family::bernoulli) {
    const Scalar success = law.parameters()[0U];
    Wide entropy = 0.0L;
    if (success > 0.0)
      entropy -= static_cast<Wide>(success) * std::log(success);
    if (success < 1.0) {
      const Scalar failure = 1.0 - success;
      entropy -= static_cast<Wide>(failure) * std::log(failure);
    }
    return Result<Scalar>::success(static_cast<Scalar>(entropy));
  }
  if (law.measure() == probability::Measure::integer_mass ||
      law.measure() == probability::Measure::atom_mass ||
      law.measure() == probability::Measure::joint_atom_mass) {
    return Result<Scalar>::failure(
        ErrorCode::NoConvergence,
        "unbounded discrete entropy requires a qualified tail sum");
  }
  if (law.family() == probability::Family::normal) {
    const auto parameters = law.parameters();
    const Wide sd = parameters[1U];
    return Result<Scalar>::success(static_cast<Scalar>(
        0.5L * std::log(2.0L * std::numbers::pi_v<Wide> *
                        std::numbers::e_v<Wide> * sd * sd)));
  }
  if (law.family() == probability::Family::uniform) {
    const auto parameters = law.parameters();
    return Result<Scalar>::success(std::log(parameters[1U] - parameters[0U]));
  }
  if (law.family() == probability::Family::gamma) {
    const auto parameters = law.parameters();
    const Wide shape = parameters[0U];
    const Wide scale = parameters[1U];
    const Wide entropy = shape + std::log(scale) + std::lgamma(shape) +
                         (1.0L - shape) * digamma(shape);
    return Result<Scalar>::success(static_cast<Scalar>(entropy));
  }
  if (law.family() == probability::Family::beta) {
    const auto parameters = law.parameters();
    const Wide a = parameters[0U];
    const Wide b = parameters[1U];
    const Wide entropy = std::lgamma(a) + std::lgamma(b) - std::lgamma(a + b) -
                         (a - 1.0L) * digamma(a) - (b - 1.0L) * digamma(b) +
                         (a + b - 2.0L) * digamma(a + b);
    return Result<Scalar>::success(static_cast<Scalar>(entropy));
  }
  return transformedIntegral(law, nullptr, truncation, policy);
}

Result<Scalar> numericalKLDivergence(const probability::Distribution& p,
                                     const probability::Distribution& q,
                                     probability::IntervalSupport support,
                                     ProbabilityPolicy policy) {
  if (p.measure() != q.measure()) {
    return Result<Scalar>::failure(ErrorCode::InvalidDomain,
                                   "KL laws must use the same measure");
  }
  auto bounds = fullSupportBounds(p, support, policy);
  if (!bounds)
    return Result<Scalar>::failure(bounds.status().code(),
                                   std::string(bounds.status().message()));
  if (p.family() == probability::Family::normal &&
      q.family() == probability::Family::normal) {
    const auto a = p.parameters();
    const auto b = q.parameters();
    if (a.size() == b.size() && a[0U] == b[0U] && a[1U] == b[1U]) {
      return Result<Scalar>::success(0.0);
    }
    const Wide varianceP = static_cast<Wide>(a[1U]) * a[1U];
    const Wide varianceQ = static_cast<Wide>(b[1U]) * b[1U];
    const Wide difference = static_cast<Wide>(a[0U]) - b[0U];
    const Wide value =
        0.5L * (std::log(varianceQ / varianceP) +
                (varianceP + difference * difference) / varianceQ - 1.0L);
    return Result<Scalar>::success(static_cast<Scalar>(value));
  }
  if (p.family() == probability::Family::bernoulli &&
      q.family() == probability::Family::bernoulli) {
    const Scalar successP = p.parameters()[0U];
    const Scalar successQ = q.parameters()[0U];
    if (successP == successQ)
      return Result<Scalar>::success(0.0);
    if ((successP > 0.0 && successQ == 0.0) ||
        (successP < 1.0 && successQ == 1.0)) {
      return Result<Scalar>::success(std::numeric_limits<Scalar>::infinity());
    }
    Wide divergence = 0.0L;
    if (successP > 0.0) {
      divergence += static_cast<Wide>(successP) * std::log(successP / successQ);
    }
    if (successP < 1.0) {
      divergence += static_cast<Wide>(1.0 - successP) *
                    std::log((1.0 - successP) / (1.0 - successQ));
    }
    return Result<Scalar>::success(static_cast<Scalar>(divergence));
  }
  if (p.family() == probability::Family::uniform &&
      q.family() == probability::Family::mixture) {
    auto qSnapshot = probability::detail::DistributionEngine::snapshot(
        q, {probabilityWorkspaceElements, probabilityWorkspaceBytes,
            policy.iteration.max_evaluations});
    if (!qSnapshot)
      return Result<Scalar>::failure(qSnapshot.status().code(),
                                     std::string(qSnapshot.status().message()));
    bool uniformComponents = !qSnapshot.value().components.empty();
    std::vector<std::pair<Wide, Wide>> intervals;
    for (Index i = 0U; i < qSnapshot.value().components.size(); ++i) {
      const auto& component = qSnapshot.value().components[i];
      if (i < qSnapshot.value().weights.size() &&
          qSnapshot.value().weights[i] == 0.0)
        continue;
      if (component.family != probability::Family::uniform ||
          component.parameters.size() < 2U) {
        uniformComponents = false;
        break;
      }
      intervals.emplace_back(component.parameters[0U],
                             component.parameters[1U]);
    }
    if (uniformComponents) {
      std::sort(intervals.begin(), intervals.end());
      const auto pParameters = p.parameters();
      Wide coveredUntil = pParameters[0U];
      for (const auto& [rawLower, rawUpper] : intervals) {
        const Wide lower = std::max<Wide>(rawLower, pParameters[0U]);
        const Wide upper = std::min<Wide>(rawUpper, pParameters[1U]);
        if (upper <= coveredUntil)
          continue;
        if (lower > coveredUntil) {
          return Result<Scalar>::success(
              std::numeric_limits<Scalar>::infinity());
        }
        coveredUntil = upper;
      }
      if (coveredUntil < pParameters[1U]) {
        return Result<Scalar>::success(std::numeric_limits<Scalar>::infinity());
      }
    }
  }
  const auto atoms = p.finiteAtoms();
  if (!atoms.locations.empty()) {
    Wide divergence = 0.0L;
    for (Index i = 0U; i < atoms.locations.size(); ++i) {
      const Scalar massP = atoms.masses[i];
      if (massP == 0.0)
        continue;
      auto valueQ = probability::evaluate(q, atoms.locations[i], policy);
      if (!valueQ)
        return Result<Scalar>::failure(valueQ.status().code(),
                                       std::string(valueQ.status().message()));
      const Scalar massQ = valueQ.value().density_or_mass;
      if (massQ == 0.0) {
        return Result<Scalar>::success(std::numeric_limits<Scalar>::infinity());
      }
      divergence += static_cast<Wide>(massP) *
                    (std::log(massP) - valueQ.value().log_density_or_mass);
    }
    return Result<Scalar>::success(static_cast<Scalar>(divergence));
  }
  if (p.measure() == probability::Measure::integer_mass ||
      p.measure() == probability::Measure::atom_mass ||
      p.measure() == probability::Measure::joint_atom_mass) {
    return Result<Scalar>::failure(
        ErrorCode::NoConvergence,
        "unbounded discrete KL requires a qualified tail sum");
  }
  auto qLower = probability::quantile(q, 0.0, policy);
  auto qUpper = probability::quantile(q, 1.0, policy);
  if (!qLower)
    return Result<Scalar>::failure(qLower.status().code(),
                                   std::string(qLower.status().message()));
  if (!qUpper)
    return Result<Scalar>::failure(qUpper.status().code(),
                                   std::string(qUpper.status().message()));
  if (bounds.value().first < qLower.value()) {
    auto massBelow = probability::evaluate(p, qLower.value(), policy);
    if (!massBelow)
      return Result<Scalar>::failure(massBelow.status().code(),
                                     std::string(massBelow.status().message()));
    if (massBelow.value().cdf > 0.0) {
      return Result<Scalar>::success(std::numeric_limits<Scalar>::infinity());
    }
  }
  if (bounds.value().second > qUpper.value()) {
    auto massAbove = probability::evaluate(p, qUpper.value(), policy);
    if (!massAbove)
      return Result<Scalar>::failure(massAbove.status().code(),
                                     std::string(massAbove.status().message()));
    if (massAbove.value().survival > 0.0) {
      return Result<Scalar>::success(std::numeric_limits<Scalar>::infinity());
    }
  }
  if (p.family() == probability::Family::student_t &&
      q.family() == probability::Family::normal && p.parameters()[0U] <= 2.0) {
    return Result<Scalar>::success(std::numeric_limits<Scalar>::infinity());
  }
  if (p.family() == probability::Family::student_t &&
      q.family() == probability::Family::normal && p.parameters()[0U] > 2.0) {
    const auto t = p.parameters();
    const auto normal = q.parameters();
    const Wide nu = t[0U];
    const Wide location = t[1U];
    const Wide scale = t[2U];
    const Wide variance = scale * scale * nu / (nu - 2.0L);
    const Wide offset = location - normal[0U];
    const Wide crossEntropy =
        0.5L * std::log(2.0L * std::numbers::pi_v<Wide> * normal[1U] *
                        normal[1U]) +
        (variance + offset * offset) / (2.0L * normal[1U] * normal[1U]);
    const Wide halfNu = nu / 2.0L;
    const Wide halfNuPlus = (nu + 1.0L) / 2.0L;
    const Wide logBeta =
        std::lgamma(halfNu) + std::lgamma(0.5L) - std::lgamma(halfNuPlus);
    const Wide entropy = std::log(scale) + 0.5L * std::log(nu) + logBeta +
                         halfNuPlus * (digamma(halfNuPlus) - digamma(halfNu));
    return Result<Scalar>::success(static_cast<Scalar>(crossEntropy - entropy));
  }
  return transformedIntegral(p, &q, support, policy);
}

} // namespace vss::math
