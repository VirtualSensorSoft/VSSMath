#include "vss/math/covariance/Propagation.hpp"

#include "vss/math/Array.hpp"
#include "vss/math/Core.hpp"
#include "vss/math/Error.hpp"
#include "vss/math/Limits.hpp"
#include "vss/math/Matrix.hpp"
#include "vss/math/MatrixView.hpp"
#include "vss/math/Result.hpp"
#include "vss/math/SampleView.hpp"
#include "vss/math/Tensor.hpp"
#include "vss/math/Tolerance.hpp"
#include "vss/math/VectorView.hpp"
#include "vss/math/covariance/Covariance.hpp"
#include "vss/math/covariance/PropagationResult.hpp"
#include "vss/math/differentiation/FiniteDifference.hpp"
#include "vss/math/linear/SolvePolicy.hpp"
#include "vss/math/linear/SymmetricEigen.hpp"
#include "vss/math/statistics/Moments.hpp"
#include "vss/math/statistics/Population.hpp"
#include "vss/math/statistics/Weighted.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace vss::math::covariance {
namespace {

template <class T> Result<T> fail(ErrorCode code, const char* message) {
  return Result<T>::failure(code, message);
}

bool checkedProduct(Extent lhs, Extent rhs, Extent& result) noexcept {
  if (rhs != 0U && lhs > std::numeric_limits<Extent>::max() / rhs) {
    return false;
  }
  result = lhs * rhs;
  return true;
}

bool withinLimits(Extent elements, Limits limits) noexcept {
  return elements <= limits.elements &&
         elements <= limits.bytes / sizeof(Scalar);
}

bool checkedAdd(Extent lhs, Extent rhs, Extent& result) noexcept {
  if (lhs > std::numeric_limits<Extent>::max() - rhs) {
    return false;
  }
  result = lhs + rhs;
  return true;
}

class WideSum final {
public:
  void add(long double value) noexcept {
    const long double next = sum_ + value;
    if (std::abs(sum_) >= std::abs(value)) {
      correction_ += (sum_ - next) + value;
    } else {
      correction_ += (value - next) + sum_;
    }
    sum_ = next;
  }

  [[nodiscard]] long double total() const noexcept {
    return sum_ + correction_;
  }

private:
  long double sum_{};
  long double correction_{};
};

bool finite(ConstVectorView value) noexcept {
  for (Index i = 0; i < value.size(); ++i) {
    if (!std::isfinite(value.at(i))) {
      return false;
    }
  }
  return true;
}

bool finite(ConstMatrixView value) noexcept {
  for (Index i = 0; i < value.rows(); ++i) {
    for (Index j = 0; j < value.columns(); ++j) {
      if (!std::isfinite(value.at(i, j))) {
        return false;
      }
    }
  }
  return true;
}

bool tensorShape(const Tensor& tensor, std::span<const Extent> expected) {
  return std::ranges::equal(tensor.shape().extents(), expected);
}

bool close(long double lhs, long double rhs, Tolerance tolerance) noexcept {
  return std::abs(lhs - rhs) <=
         tolerance.absolute +
             (tolerance.relative * std::max(std::abs(lhs), std::abs(rhs)));
}

bool checkedChoose(Extent n, Extent k, Extent& value) noexcept {
  k = std::min(k, n - std::min(n, k));
  value = 1U;
  for (Extent i = 1U; i <= k; ++i) {
    const Extent factor = n - k + i;
    if (value > std::numeric_limits<Extent>::max() / factor) {
      return false;
    }
    value = (value * factor) / i;
  }
  return true;
}

void sortMomentPrefix(std::array<Index, 4>& tuple, Index order) {
  switch (order) {
  case 1U:
    break;
  case 2U:
    std::ranges::sort(std::span<Index>{tuple}.first(2U));
    break;
  case 3U:
    std::ranges::sort(std::span<Index>{tuple}.first(3U));
    break;
  case 4U:
    std::ranges::sort(tuple);
    break;
  default:
    break;
  }
}

const statistics::MomentTerm* findTerm(const MomentEstimate& estimate,
                                       Index order) noexcept {
  for (const auto& term : estimate.terms.view()) {
    if (term.order == order) {
      return &term;
    }
  }
  return nullptr;
}

Result<Scalar> momentAt(const MomentEstimate& estimate, Index order,
                        std::span<const Index> tuple, Index dimension) {
  const auto* term = findTerm(estimate, order);
  if (term == nullptr) {
    return fail<Scalar>(ErrorCode::InvalidDomain,
                        "a required central-moment order is absent");
  }
  if (order == 0U || order > 4U) {
    return fail<Scalar>(ErrorCode::InvalidDomain,
                        "central-moment order exceeds the supported order");
  }
  if (tuple.size() != order) {
    return fail<Scalar>(ErrorCode::DimensionMismatch,
                        "central-moment tuple order is inconsistent");
  }
  std::array<Index, 4> sorted{};
  std::ranges::copy(tuple, sorted.begin());
  sortMomentPrefix(sorted, order);
  for (const Index coordinate : tuple) {
    if (coordinate >= dimension) {
      return fail<Scalar>(ErrorCode::IndexOutOfRange,
                          "central-moment tuple is outside its coordinates");
    }
  }
  const auto values = term->values.view();
  Index ordinal = 0U;
  switch (term->layout) {
  case statistics::MomentRepresentation::component:
    if (!std::ranges::all_of(
            tuple, [&](Index value) { return value == tuple.front(); })) {
      return fail<Scalar>(ErrorCode::InvalidDomain,
                          "component moments do not cover a mixed tuple");
    }
    ordinal = tuple.front();
    break;
  case statistics::MomentRepresentation::dense: {
    for (const Index coordinate : tuple) {
      if (!checkedProduct(ordinal, dimension, ordinal) ||
          ordinal > std::numeric_limits<Index>::max() - coordinate) {
        return fail<Scalar>(ErrorCode::SizeOverflow,
                            "dense moment index overflows");
      }
      ordinal += coordinate;
    }
    break;
  }
  case statistics::MomentRepresentation::packed: {
    Index previous = 0U;
    for (Index axis = 0U; axis < order; ++axis) {
      const Index current = sorted.at(axis);
      for (Index candidate = previous; candidate < current; ++candidate) {
        const Index remaining = order - axis - 1U;
        Index count = 1U;
        if (remaining != 0U &&
            !checkedChoose(dimension - candidate + remaining - 1U, remaining,
                           count)) {
          return fail<Scalar>(ErrorCode::SizeOverflow,
                              "packed moment index overflows");
        }
        if (ordinal > std::numeric_limits<Index>::max() - count) {
          return fail<Scalar>(ErrorCode::SizeOverflow,
                              "packed moment index overflows");
        }
        ordinal += count;
      }
      previous = current;
    }
    break;
  }
  case statistics::MomentRepresentation::selected: {
    const auto indices = term->flattened_tuples.view();
    Extent expectedIndices = 0U;
    if (!checkedProduct(values.size(), order, expectedIndices)) {
      return fail<Scalar>(ErrorCode::SizeOverflow,
                          "selected moment tuple count overflows");
    }
    if (indices.size() != expectedIndices) {
      return fail<Scalar>(ErrorCode::InvalidDomain,
                          "selected moment payload has an invalid shape");
    }
    bool found = false;
    Scalar selected = 0.0;
    for (Index i = 0U; i < values.size(); ++i) {
      std::array<Index, 4> candidate{};
      for (Index j = 0U; j < order; ++j) {
        candidate.at(j) = indices[(i * order) + j];
      }
      sortMomentPrefix(candidate, order);
      if (std::ranges::equal(std::span<const Index>{sorted}.first(order),
                             std::span<const Index>{candidate}.first(order))) {
        if (found && values.at(i) != selected) {
          return fail<Scalar>(ErrorCode::InvalidDomain,
                              "duplicate selected moments conflict");
        }
        found = true;
        selected = values.at(i);
      }
    }
    if (!found) {
      return fail<Scalar>(ErrorCode::InvalidDomain,
                          "a required selected central moment is absent");
    }
    if (!std::isfinite(selected)) {
      return fail<Scalar>(ErrorCode::NonFiniteInput,
                          "central moment contains a non-finite value");
    }
    return Result<Scalar>::success(selected);
  }
  case statistics::MomentRepresentation::contraction:
    return fail<Scalar>(ErrorCode::InvalidDomain,
                        "contracted moment layout is not a coordinate moment");
  }
  if (ordinal >= values.size()) {
    return fail<Scalar>(
        ErrorCode::InvalidDomain,
        "central-moment payload does not cover a required tuple");
  }
  if (!std::isfinite(values.at(ordinal))) {
    return fail<Scalar>(ErrorCode::NonFiniteInput,
                        "central moment contains a non-finite value");
  }
  return Result<Scalar>::success(values.at(ordinal));
}

Result<Index> inputDimension(const MomentEstimate& estimate) {
  const auto shape = estimate.observation_shape.view();
  if (shape.empty()) {
    return Result<Index>::success(1U);
  }
  Index product = 1U;
  for (const Index extent : shape) {
    if (extent == 0U || !checkedProduct(product, extent, product)) {
      return fail<Index>(extent == 0U ? ErrorCode::InvalidExtent
                                      : ErrorCode::SizeOverflow,
                         "moment observation shape is invalid");
    }
  }
  return Result<Index>::success(product);
}

EstimatorDiagnostics propagationDiagnostics(Approximation method,
                                            const char* algorithm,
                                            const char* premises) {
  return EstimatorDiagnostics{.estimand = Estimand::estimate,
                              .weight = WeightRule::dependent,
                              .center = CenterRule::estimated,
                              .approximation = method,
                              .global = {},
                              .coordinate_support = {},
                              .tuple_support = {},
                              .reciprocal_condition = std::nullopt,
                              .numerical_error = std::nullopt,
                              .sampling_error = std::nullopt,
                              .rank = std::nullopt,
                              .algorithm = algorithm,
                              .premises = premises,
                              .warnings = {}};
}

struct DiagnosticStorage final {
  Extent elements{};
  Extent bytes{};
  Extent coordinateElements{};
  Extent coordinateBytes{};
  Extent tupleElements{};
  Extent tupleBytes{};
  Extent warningElements{};
  Extent warningBytes{};
};

Result<DiagnosticStorage>
diagnosticStorage(const EstimatorDiagnostics& diagnostics) {
  DiagnosticStorage storage;
  storage.coordinateElements = diagnostics.coordinate_support.view().size();
  storage.tupleElements = diagnostics.tuple_support.view().size();
  storage.warningElements = diagnostics.warnings.view().size();
  if (!checkedProduct(storage.coordinateElements, sizeof(Support),
                      storage.coordinateBytes) ||
      !checkedProduct(storage.tupleElements, sizeof(TupleSupport),
                      storage.tupleBytes) ||
      !checkedProduct(storage.warningElements, sizeof(ErrorCode),
                      storage.warningBytes)) {
    return fail<DiagnosticStorage>(ErrorCode::SizeOverflow,
                                   "propagation diagnostic storage overflows");
  }
  for (const auto& tuple : diagnostics.tuple_support.view()) {
    const Extent count = tuple.coordinates.view().size();
    Extent bytes = 0U;
    if (!checkedAdd(storage.tupleElements, count, storage.tupleElements) ||
        !checkedProduct(count, sizeof(Index), bytes) ||
        !checkedAdd(storage.tupleBytes, bytes, storage.tupleBytes)) {
      return fail<DiagnosticStorage>(ErrorCode::SizeOverflow,
                                     "tuple-support storage overflows");
    }
  }
  storage.elements = storage.coordinateElements;
  if (!checkedAdd(storage.elements, storage.tupleElements, storage.elements) ||
      !checkedAdd(storage.elements, storage.warningElements,
                  storage.elements) ||
      !checkedAdd(storage.bytes, storage.coordinateBytes, storage.bytes) ||
      !checkedAdd(storage.bytes, storage.tupleBytes, storage.bytes) ||
      !checkedAdd(storage.bytes, storage.warningBytes, storage.bytes)) {
    return fail<DiagnosticStorage>(ErrorCode::SizeOverflow,
                                   "propagation diagnostic storage overflows");
  }
  return Result<DiagnosticStorage>::success(storage);
}

Result<EstimatorDiagnostics>
withInputDiagnostics(EstimatorDiagnostics output,
                     const EstimatorDiagnostics& input, Limits limits) {
  auto storage = diagnosticStorage(input);
  if (!storage.hasValue()) {
    return Result<EstimatorDiagnostics>::failure(
        storage.status().code(), std::string(storage.status().message()));
  }
  if (storage.value().elements > limits.elements ||
      storage.value().bytes > limits.bytes) {
    return fail<EstimatorDiagnostics>(
        ErrorCode::ResourceLimit,
        "propagation diagnostics exceed storage limits");
  }
  auto coordinates = Array<Support>::copy(
      input.coordinate_support.view(),
      Limits{.elements = storage.value().coordinateElements,
             .bytes = storage.value().coordinateBytes,
             .evaluations = 0U});
  auto tuples = Array<TupleSupport>::copy(
      input.tuple_support.view(),
      Limits{.elements = storage.value().tupleElements,
             .bytes = storage.value().tupleBytes,
             .evaluations = 0U});
  auto warnings = Array<ErrorCode>::copy(
      input.warnings.view(), Limits{.elements = storage.value().warningElements,
                                    .bytes = storage.value().warningBytes,
                                    .evaluations = 0U});
  if (!coordinates.hasValue() || !tuples.hasValue() || !warnings.hasValue()) {
    const Status* status = &warnings.status();
    if (!tuples.hasValue()) {
      status = &tuples.status();
    }
    if (!coordinates.hasValue()) {
      status = &coordinates.status();
    }
    return Result<EstimatorDiagnostics>::failure(
        status->code(), std::string(status->message()));
  }
  output.estimand = input.estimand;
  output.weight = input.weight;
  output.center = input.center;
  output.global = input.global;
  output.coordinate_support = std::move(coordinates).value();
  output.tuple_support = std::move(tuples).value();
  output.warnings = std::move(warnings).value();
  if (!input.algorithm.empty() || !input.premises.empty()) {
    output.premises +=
        " Input estimate: " + input.algorithm + ". " + input.premises;
  }
  return Result<EstimatorDiagnostics>::success(std::move(output));
}

bool withinCombinedLimits(Extent outputElements, Extent scratchElements,
                          Extent scratchElementBytes,
                          const DiagnosticStorage& diagnostics,
                          Limits limits) noexcept {
  Extent elements = 0U;
  Extent bytes = 0U;
  return checkedAdd(outputElements, scratchElements, elements) &&
         checkedAdd(elements, diagnostics.elements, elements) &&
         checkedProduct(outputElements, sizeof(Scalar), bytes) &&
         checkedAdd(bytes, scratchElementBytes, bytes) &&
         checkedAdd(bytes, diagnostics.bytes, bytes) &&
         elements <= limits.elements && bytes <= limits.bytes;
}

PropagationCovarianceCheck classify(ConstMatrixView value) {
  if (value.rows() == 0U) {
    return PropagationCovarianceCheck::positive_semidefinite;
  }
  auto diagnostics = analyzeCovariance(value);
  if (!diagnostics.hasValue()) {
    return PropagationCovarianceCheck::indefinite_approximation;
  }
  if (diagnostics.value().classification() ==
      CovarianceClassification::NotPositiveSemidefinite) {
    return PropagationCovarianceCheck::indefinite_approximation;
  }
  return PropagationCovarianceCheck::positive_semidefinite;
}

Result<Index> mapDimension(const QuadraticMap& map) {
  const Index d = map.jacobian.columns();
  const Index m = map.jacobian.rows();
  const std::array<Extent, 3> hessianShape{m, d, d};
  if (map.at_center.size() != m ||
      map.hessian.get().shape().extents().size() != 3U ||
      !tensorShape(map.hessian.get(), hessianShape) || d == 0U || m == 0U) {
    return fail<Index>(ErrorCode::DimensionMismatch,
                       "quadratic map dimensions or Hessian shape disagree");
  }
  if (!finite(map.at_center) || !finite(map.jacobian)) {
    return fail<Index>(ErrorCode::NonFiniteInput,
                       "quadratic map contains non-finite values");
  }
  for (Index a = 0U; a < m; ++a) {
    for (Index i = 0U; i < d; ++i) {
      for (Index j = 0U; j < d; ++j) {
        const std::array<Index, 3> ij{a, i, j};
        const Scalar value = map.hessian.get().at(ij);
        const std::array<Index, 3> ji{a, j, i};
        if (!std::isfinite(value)) {
          return fail<Index>(ErrorCode::NonFiniteInput,
                             "Hessian contains a non-finite value");
        }
        if (!close(value, map.hessian.get().at(ji),
                   Tolerance{.absolute = 0.0, .relative = 1.0e-12})) {
          // The caller-facing validation tolerance is applied by the API
          // operation after structural checks; exact symmetry is the safe
          // default for a raw tensor with no attached tolerance metadata.
          return fail<Index>(ErrorCode::NotSymmetric,
                             "Hessian input axes are not symmetric");
        }
      }
    }
  }
  return Result<Index>::success(d);
}

long double tensorValue(const Tensor& tensor,
                        std::initializer_list<Index> indices) {
  std::array<Index, 4> fixed{};
  std::ranges::copy(indices, fixed.begin());
  return tensor.at(std::span<const Index>(fixed.data(), indices.size()));
}

} // namespace

Result<PropagationResult> propagateQuadraticTaylor(const QuadraticMap& map,
                                                   const MomentEstimate& input,
                                                   Limits limits) {
  const auto dimension = mapDimension(map);
  if (!dimension.hasValue()) {
    return Result<PropagationResult>::failure(
        dimension.status().code(), std::string(dimension.status().message()));
  }
  const Index d = dimension.value();
  const Index m = map.jacobian.rows();
  auto inputD = inputDimension(input);
  if (!inputD.hasValue()) {
    return Result<PropagationResult>::failure(
        inputD.status().code(), std::string(inputD.status().message()));
  }
  if (inputD.value() != d || (input.population_mean.has_value() &&
                              input.population_mean->size() != d)) {
    return fail<PropagationResult>(
        ErrorCode::DimensionMismatch,
        "input moment coordinates disagree with the map");
  }
  if (!input.population_mean.has_value() ||
      input.diagnostics.center != CenterRule::estimated ||
      input.diagnostics.weight == WeightRule::componentwise) {
    return fail<PropagationResult>(
        ErrorCode::InvalidDomain,
        "moments must share an estimated common center");
  }
  for (Index i = 0U; i < d; ++i) {
    if (!std::isfinite(input.population_mean->at(i))) {
      return fail<PropagationResult>(ErrorCode::NonFiniteInput,
                                     "moment population mean is non-finite");
    }
  }
  Extent outputCount = 0U;
  Extent covarianceCount = 0U;
  if (!checkedProduct(m, m, covarianceCount) ||
      covarianceCount > std::numeric_limits<Extent>::max() - m) {
    return fail<PropagationResult>(ErrorCode::SizeOverflow,
                                   "propagation output size overflows");
  }
  outputCount = covarianceCount + m;
  auto supportStorage = diagnosticStorage(input.diagnostics);
  if (!supportStorage.hasValue()) {
    return Result<PropagationResult>::failure(
        supportStorage.status().code(),
        std::string(supportStorage.status().message()));
  }
  Extent meanScratchBytes = 0U;
  if (!checkedProduct(m, sizeof(long double), meanScratchBytes)) {
    return fail<PropagationResult>(ErrorCode::SizeOverflow,
                                   "propagation scratch size overflows");
  }
  if (!withinCombinedLimits(outputCount, m, meanScratchBytes,
                            supportStorage.value(), limits)) {
    return fail<PropagationResult>(
        ErrorCode::ResourceLimit,
        "propagation output and diagnostics exceed limits");
  }

  std::vector<long double> means(m, 0.0L);
  Matrix resultCovariance(m, m, 0.0);
  for (Index a = 0U; a < m; ++a) {
    WideSum meanCorrection;
    for (Index i = 0U; i < d; ++i) {
      for (Index j = 0U; j < d; ++j) {
        const long double coefficient =
            0.5L * tensorValue(map.hessian.get(), {a, i, j});
        if (coefficient == 0.0L) {
          continue;
        }
        const std::array<Index, 2> tuple{i, j};
        auto moment = momentAt(input, 2U, tuple, d);
        if (!moment.hasValue()) {
          return Result<PropagationResult>::failure(
              moment.status().code(), std::string(moment.status().message()));
        }
        meanCorrection.add(coefficient * moment.value());
      }
    }
    means[a] =
        static_cast<long double>(map.at_center.at(a)) + meanCorrection.total();
    if (!std::isfinite(means[a]) ||
        std::abs(means[a]) > std::numeric_limits<Scalar>::max()) {
      return fail<PropagationResult>(ErrorCode::NonFiniteResult,
                                     "quadratic propagated mean is not finite");
    }
  }

  for (Index a = 0U; a < m; ++a) {
    for (Index b = a; b < m; ++b) {
      WideSum sum;
      for (Index i = 0U; i < d; ++i) {
        for (Index j = 0U; j < d; ++j) {
          const long double linearCoefficient =
              static_cast<long double>(map.jacobian.at(a, i)) *
              map.jacobian.at(b, j);
          if (linearCoefficient != 0.0L) {
            const std::array<Index, 2> pair{i, j};
            auto second = momentAt(input, 2U, pair, d);
            if (!second.hasValue()) {
              return Result<PropagationResult>::failure(
                  second.status().code(),
                  std::string(second.status().message()));
            }
            sum.add(linearCoefficient * second.value());
          }
          for (Index k = 0U; k < d; ++k) {
            const long double mixed =
                0.5L * (static_cast<long double>(map.jacobian.at(a, i)) *
                            tensorValue(map.hessian.get(), {b, j, k}) +
                        static_cast<long double>(map.jacobian.at(b, i)) *
                            tensorValue(map.hessian.get(), {a, j, k}));
            if (mixed != 0.0L) {
              const std::array<Index, 3> triple{i, j, k};
              auto third = momentAt(input, 3U, triple, d);
              if (!third.hasValue()) {
                return Result<PropagationResult>::failure(
                    third.status().code(),
                    std::string(third.status().message()));
              }
              sum.add(mixed * third.value());
            }
          }
          for (Index k = 0U; k < d; ++k) {
            for (Index l = 0U; l < d; ++l) {
              const long double coefficient =
                  0.25L * tensorValue(map.hessian.get(), {a, i, j}) *
                  tensorValue(map.hessian.get(), {b, k, l});
              if (coefficient == 0.0L) {
                continue;
              }
              const std::array<Index, 4> fourthTuple{i, j, k, l};
              auto fourth = momentAt(input, 4U, fourthTuple, d);
              if (!fourth.hasValue()) {
                return Result<PropagationResult>::failure(
                    fourth.status().code(),
                    std::string(fourth.status().message()));
              }
              const std::array<Index, 2> ij{i, j};
              const std::array<Index, 2> kl{k, l};
              auto secondIJ = momentAt(input, 2U, ij, d);
              auto secondKL = momentAt(input, 2U, kl, d);
              if (!secondIJ.hasValue() || !secondKL.hasValue()) {
                const auto& error = !secondIJ.hasValue() ? secondIJ.status()
                                                         : secondKL.status();
                return Result<PropagationResult>::failure(
                    error.code(), std::string(error.message()));
              }
              sum.add(coefficient *
                      (static_cast<long double>(fourth.value()) -
                       static_cast<long double>(secondIJ.value()) *
                           secondKL.value()));
            }
          }
        }
      }
      if (!std::isfinite(sum.total()) ||
          std::abs(sum.total()) > std::numeric_limits<Scalar>::max()) {
        return fail<PropagationResult>(ErrorCode::NonFiniteResult,
                                       "quadratic covariance is not finite");
      }
      resultCovariance.at(a, b) = static_cast<Scalar>(sum.total());
      resultCovariance.at(b, a) = static_cast<Scalar>(sum.total());
    }
  }
  Vector mean(m);
  for (Index a = 0U; a < m; ++a) {
    mean.at(a) = static_cast<Scalar>(means[a]);
  }
  auto diagnostics = propagationDiagnostics(
      Approximation::quadratic, "quadratic_taylor",
      "Exact moments of the quadratic surrogate; input powers are "
      "common-support central population moments.");
  auto completeDiagnostics =
      withInputDiagnostics(std::move(diagnostics), input.diagnostics, limits);
  if (!completeDiagnostics.hasValue()) {
    return Result<PropagationResult>::failure(
        completeDiagnostics.status().code(),
        std::string(completeDiagnostics.status().message()));
  }
  const auto covarianceCheck = classify(resultCovariance.view());
  return Result<PropagationResult>::success(
      PropagationResult{.mean = std::move(mean),
                        .covariance = std::move(resultCovariance),
                        .method = Approximation::quadratic,
                        .diagnostics = std::move(completeDiagnostics).value(),
                        .covariance_check = covarianceCheck});
}

Result<PropagationResult>
propagateGaussianQuadraticTaylor(const QuadraticMap& map,
                                 ConstMatrixView covariance, Limits limits) {
  const auto dimension = mapDimension(map);
  if (!dimension.hasValue()) {
    return Result<PropagationResult>::failure(
        dimension.status().code(), std::string(dimension.status().message()));
  }
  const Index d = dimension.value();
  const Index m = map.jacobian.rows();
  if (covariance.rows() != d || covariance.columns() != d) {
    return fail<PropagationResult>(ErrorCode::DimensionMismatch,
                                   "Gaussian covariance dimensions disagree");
  }
  Extent md = 0U;
  Extent hessianElements = 0U;
  Extent covarianceElements = 0U;
  Extent outputElements = 0U;
  Extent eigensolverElements = 0U;
  Extent scalarElements = 0U;
  Extent outputBytes = 0U;
  Extent workspaceBytes = 0U;
  if (!checkedProduct(m, d, md) || !checkedProduct(md, d, hessianElements) ||
      !checkedProduct(d, d, covarianceElements) ||
      !checkedProduct(m, m, outputElements) ||
      !checkedProduct(covarianceElements, 3U, eigensolverElements) ||
      !checkedAdd(eigensolverElements, outputElements, scalarElements) ||
      !checkedAdd(scalarElements, m, scalarElements) ||
      !checkedAdd(scalarElements, hessianElements, scalarElements) ||
      !checkedProduct(hessianElements, sizeof(long double), workspaceBytes) ||
      !checkedProduct(eigensolverElements + outputElements + m, sizeof(Scalar),
                      outputBytes) ||
      !checkedAdd(outputBytes, workspaceBytes, outputBytes)) {
    return fail<PropagationResult>(ErrorCode::SizeOverflow,
                                   "Gaussian propagation storage overflows");
  }
  if (scalarElements > limits.elements || outputBytes > limits.bytes) {
    return fail<PropagationResult>(ErrorCode::ResourceLimit,
                                   "Gaussian propagation exceeds limits");
  }
  auto inputDiagnostics = analyzeCovariance(covariance);
  if (!inputDiagnostics.hasValue()) {
    return Result<PropagationResult>::failure(
        inputDiagnostics.status().code(),
        std::string(inputDiagnostics.status().message()));
  }
  if (inputDiagnostics.value().classification() ==
      CovarianceClassification::NotPositiveSemidefinite) {
    return fail<PropagationResult>(
        ErrorCode::NotPositiveSemidefinite,
        "Gaussian covariance is not positive semidefinite");
  }
  std::vector<long double> transformed(hessianElements, 0.0L);
  for (Index a = 0U; a < m; ++a) {
    for (Index i = 0U; i < d; ++i) {
      for (Index j = 0U; j < d; ++j) {
        long double value = 0.0L;
        for (Index k = 0U; k < d; ++k) {
          value +=
              tensorValue(map.hessian.get(), {a, i, k}) * covariance.at(k, j);
        }
        if (!std::isfinite(value)) {
          return fail<PropagationResult>(
              ErrorCode::NonFiniteResult,
              "Gaussian Hessian product is not finite");
        }
        transformed[((a * d + i) * d) + j] = value;
      }
    }
  }
  Matrix outputCovariance(m, m, 0.0);
  Vector mean(m);
  for (Index a = 0U; a < m; ++a) {
    WideSum trace;
    for (Index i = 0U; i < d; ++i) {
      for (Index j = 0U; j < d; ++j) {
        trace.add(0.5L * tensorValue(map.hessian.get(), {a, i, j}) *
                  covariance.at(j, i));
      }
    }
    const long double center = map.at_center.at(a) + trace.total();
    if (!std::isfinite(center) ||
        std::abs(center) > std::numeric_limits<Scalar>::max()) {
      return fail<PropagationResult>(ErrorCode::NonFiniteResult,
                                     "Gaussian propagated mean is not finite");
    }
    mean.at(a) = static_cast<Scalar>(center);
    for (Index b = a; b < m; ++b) {
      WideSum contraction;
      for (Index i = 0U; i < d; ++i) {
        for (Index j = 0U; j < d; ++j) {
          contraction.add(0.5L * transformed[((a * d + i) * d) + j] *
                          transformed[((b * d + j) * d) + i]);
        }
      }
      WideSum linear;
      for (Index i = 0U; i < d; ++i) {
        for (Index j = 0U; j < d; ++j) {
          linear.add(static_cast<long double>(map.jacobian.at(a, i)) *
                     covariance.at(i, j) * map.jacobian.at(b, j));
        }
      }
      const long double value = linear.total() + contraction.total();
      if (!std::isfinite(value) ||
          std::abs(value) > std::numeric_limits<Scalar>::max()) {
        return fail<PropagationResult>(
            ErrorCode::NonFiniteResult,
            "Gaussian propagated covariance is not finite");
      }
      outputCovariance.at(a, b) = static_cast<Scalar>(value);
      outputCovariance.at(b, a) = static_cast<Scalar>(value);
    }
  }
  auto diagnostics = propagationDiagnostics(
      Approximation::quadratic, "gaussian_quadratic_taylor",
      "Centered Gaussian input and exact covariance of the quadratic Taylor "
      "surrogate.");
  const auto check = classify(outputCovariance.view());
  return Result<PropagationResult>::success(
      PropagationResult{.mean = std::move(mean),
                        .covariance = std::move(outputCovariance),
                        .method = Approximation::quadratic,
                        .diagnostics = std::move(diagnostics),
                        .covariance_check = check});
}

Result<PropagationResult>
propagateFourthOrderCovariance(const CubicMap& map, const MomentEstimate& input,
                               Limits limits) {
  const auto dimension = mapDimension(map.quadratic);
  if (!dimension.hasValue()) {
    return Result<PropagationResult>::failure(
        dimension.status().code(), std::string(dimension.status().message()));
  }
  const Index d = dimension.value();
  const Index m = map.quadratic.jacobian.rows();
  const std::array<Extent, 4> thirdShape{m, d, d, d};
  if (!tensorShape(map.thirds.get(), thirdShape)) {
    return fail<PropagationResult>(
        ErrorCode::DimensionMismatch,
        "third-derivative tensor has the wrong shape");
  }
  if (!std::isfinite(map.amplitude) || map.amplitude < 0.0) {
    return fail<PropagationResult>(
        ErrorCode::InvalidDomain,
        "cubic-map amplitude must be finite and nonnegative");
  }
  for (Index a = 0U; a < m; ++a) {
    for (Index i = 0U; i < d; ++i) {
      for (Index j = 0U; j < d; ++j) {
        for (Index k = 0U; k < d; ++k) {
          const long double value = tensorValue(map.thirds.get(), {a, i, j, k});
          if (!std::isfinite(value)) {
            return fail<PropagationResult>(ErrorCode::NonFiniteInput,
                                           "third derivative is non-finite");
          }
          const long double ijk = value;
          const long double jik = tensorValue(map.thirds.get(), {a, j, i, k});
          const long double ikj = tensorValue(map.thirds.get(), {a, i, k, j});
          if (!close(ijk, jik,
                     Tolerance{.absolute = 0.0, .relative = 1.0e-12}) ||
              !close(ijk, ikj,
                     Tolerance{.absolute = 0.0, .relative = 1.0e-12})) {
            return fail<PropagationResult>(
                ErrorCode::NotSymmetric,
                "third-derivative input axes are not symmetric");
          }
        }
      }
    }
  }
  auto inputD = inputDimension(input);
  if (!inputD.hasValue()) {
    return Result<PropagationResult>::failure(
        inputD.status().code(), std::string(inputD.status().message()));
  }
  if (inputD.value() != d || (input.population_mean.has_value() &&
                              input.population_mean->size() != d)) {
    return fail<PropagationResult>(
        ErrorCode::DimensionMismatch,
        "input moment coordinates disagree with the map");
  }
  if (!input.population_mean.has_value() ||
      input.diagnostics.center != CenterRule::estimated ||
      input.diagnostics.weight == WeightRule::componentwise) {
    return fail<PropagationResult>(
        ErrorCode::InvalidDomain,
        "moments must share an estimated common center");
  }
  Extent covarianceCount = 0U;
  if (!checkedProduct(m, m, covarianceCount) ||
      covarianceCount > std::numeric_limits<Extent>::max() - m) {
    return fail<PropagationResult>(ErrorCode::SizeOverflow,
                                   "cubic propagation output size overflows");
  }
  auto supportStorage = diagnosticStorage(input.diagnostics);
  if (!supportStorage.hasValue()) {
    return Result<PropagationResult>::failure(
        supportStorage.status().code(),
        std::string(supportStorage.status().message()));
  }
  if (!withinCombinedLimits(covarianceCount + m, 0U, 0U, supportStorage.value(),
                            limits)) {
    return fail<PropagationResult>(
        ErrorCode::ResourceLimit,
        "cubic propagation and diagnostics exceed limits");
  }

  const long double epsilon = map.amplitude;
  const long double epsilon2 = epsilon * epsilon;
  const long double epsilon3 = epsilon2 * epsilon;
  const long double epsilon4 = epsilon2 * epsilon2;
  if (!std::isfinite(epsilon2) || !std::isfinite(epsilon3) ||
      !std::isfinite(epsilon4)) {
    return fail<PropagationResult>(
        ErrorCode::NonFiniteResult,
        "amplitude powers exceed the supported range");
  }
  if (map.amplitude == 0.0) {
    Vector nominal(m);
    for (Index a = 0U; a < m; ++a) {
      nominal.at(a) = map.quadratic.at_center.at(a);
    }
    auto diagnostics = propagationDiagnostics(
        Approximation::fourth_amplitude, "fourth_amplitude_cubic_taylor",
        "Zero amplitude returns the base value; no Taylor contraction is "
        "evaluated.");
    auto completeDiagnostics =
        withInputDiagnostics(std::move(diagnostics), input.diagnostics, limits);
    if (!completeDiagnostics.hasValue()) {
      return Result<PropagationResult>::failure(
          completeDiagnostics.status().code(),
          std::string(completeDiagnostics.status().message()));
    }
    return Result<PropagationResult>::success(PropagationResult{
        .mean = std::move(nominal),
        .covariance = Matrix(m, m, 0.0),
        .method = Approximation::fourth_amplitude,
        .diagnostics = std::move(completeDiagnostics).value(),
        .covariance_check = PropagationCovarianceCheck::positive_semidefinite});
  }
  Vector mean(m);
  Matrix covarianceOutput(m, m, 0.0);
  for (Index a = 0U; a < m; ++a) {
    WideSum quadraticMean;
    WideSum cubicMean;
    for (Index i = 0U; i < d; ++i) {
      for (Index j = 0U; j < d; ++j) {
        const long double h = tensorValue(map.quadratic.hessian, {a, i, j});
        if (h != 0.0L) {
          const std::array<Index, 2> tuple{i, j};
          auto second = momentAt(input, 2U, tuple, d);
          if (!second.hasValue()) {
            return Result<PropagationResult>::failure(
                second.status().code(), std::string(second.status().message()));
          }
          quadraticMean.add(0.5L * h * second.value());
        }
        for (Index k = 0U; k < d; ++k) {
          const long double third = tensorValue(map.thirds.get(), {a, i, j, k});
          if (third != 0.0L) {
            const std::array<Index, 3> tuple{i, j, k};
            auto moment = momentAt(input, 3U, tuple, d);
            if (!moment.hasValue()) {
              return Result<PropagationResult>::failure(
                  moment.status().code(),
                  std::string(moment.status().message()));
            }
            cubicMean.add(third * moment.value() / 6.0L);
          }
        }
      }
    }
    const long double value = map.quadratic.at_center.at(a) +
                              (epsilon2 * quadraticMean.total()) +
                              (epsilon3 * cubicMean.total());
    if (!std::isfinite(value) ||
        std::abs(value) > std::numeric_limits<Scalar>::max()) {
      return fail<PropagationResult>(ErrorCode::NonFiniteResult,
                                     "cubic propagated mean is not finite");
    }
    mean.at(a) = static_cast<Scalar>(value);
  }

  for (Index a = 0U; a < m; ++a) {
    for (Index b = a; b < m; ++b) {
      WideSum secondOrder;
      WideSum thirdOrder;
      WideSum fourthQuadratic;
      WideSum fourthCubicLinear;
      for (Index i = 0U; i < d; ++i) {
        for (Index j = 0U; j < d; ++j) {
          const std::array<Index, 2> ij{i, j};
          const long double linearCoefficient =
              static_cast<long double>(map.quadratic.jacobian.at(a, i)) *
              map.quadratic.jacobian.at(b, j);
          if (linearCoefficient != 0.0L) {
            auto second = momentAt(input, 2U, ij, d);
            if (!second.hasValue()) {
              return Result<PropagationResult>::failure(
                  second.status().code(),
                  std::string(second.status().message()));
            }
            secondOrder.add(linearCoefficient * second.value());
          }
          for (Index k = 0U; k < d; ++k) {
            const long double mixed =
                0.5L *
                (static_cast<long double>(map.quadratic.jacobian.at(a, i)) *
                     tensorValue(map.quadratic.hessian, {b, j, k}) +
                 static_cast<long double>(map.quadratic.jacobian.at(b, i)) *
                     tensorValue(map.quadratic.hessian, {a, j, k}));
            if (mixed != 0.0L) {
              const std::array<Index, 3> ijk{i, j, k};
              auto third = momentAt(input, 3U, ijk, d);
              if (!third.hasValue()) {
                return Result<PropagationResult>::failure(
                    third.status().code(),
                    std::string(third.status().message()));
              }
              thirdOrder.add(mixed * third.value());
            }
            for (Index l = 0U; l < d; ++l) {
              const long double hh =
                  0.25L * tensorValue(map.quadratic.hessian, {a, i, j}) *
                  tensorValue(map.quadratic.hessian, {b, k, l});
              if (hh != 0.0L) {
                const std::array<Index, 4> ijkl{i, j, k, l};
                auto fourth = momentAt(input, 4U, ijkl, d);
                auto secondIJ = momentAt(input, 2U, ij, d);
                const std::array<Index, 2> kl{k, l};
                auto secondKL = momentAt(input, 2U, kl, d);
                if (!fourth.hasValue() || !secondIJ.hasValue() ||
                    !secondKL.hasValue()) {
                  const Status* error = &secondKL.status();
                  if (!secondIJ.hasValue()) {
                    error = &secondIJ.status();
                  }
                  if (!fourth.hasValue()) {
                    error = &fourth.status();
                  }
                  return Result<PropagationResult>::failure(
                      error->code(), std::string(error->message()));
                }
                fourthQuadratic.add(
                    hh * (static_cast<long double>(fourth.value()) -
                          static_cast<long double>(secondIJ.value()) *
                              secondKL.value()));
              }
              const long double jk =
                  (static_cast<long double>(map.quadratic.jacobian.at(a, i)) *
                   tensorValue(map.thirds.get(), {b, j, k, l})) +
                  (static_cast<long double>(map.quadratic.jacobian.at(b, i)) *
                   tensorValue(map.thirds.get(), {a, j, k, l}));
              if (jk != 0.0L) {
                const std::array<Index, 4> ijkl{i, j, k, l};
                auto fourth = momentAt(input, 4U, ijkl, d);
                if (!fourth.hasValue()) {
                  return Result<PropagationResult>::failure(
                      fourth.status().code(),
                      std::string(fourth.status().message()));
                }
                fourthCubicLinear.add(jk * fourth.value() / 6.0L);
              }
            }
          }
        }
      }
      const long double value =
          (epsilon2 * secondOrder.total()) + (epsilon3 * thirdOrder.total()) +
          (epsilon4 * (fourthQuadratic.total() + fourthCubicLinear.total()));
      if (!std::isfinite(value) ||
          std::abs(value) > std::numeric_limits<Scalar>::max()) {
        return fail<PropagationResult>(
            ErrorCode::NonFiniteResult,
            "fourth-amplitude covariance is not finite");
      }
      covarianceOutput.at(a, b) = static_cast<Scalar>(value);
      covarianceOutput.at(b, a) = static_cast<Scalar>(value);
    }
  }
  auto diagnostics = propagationDiagnostics(
      Approximation::fourth_amplitude, "fourth_amplitude_cubic_taylor",
      "Truncated covariance of the cubic Taylor polynomial through amplitude "
      "order four; no full-map tail guarantee is implied.");
  auto completeDiagnostics =
      withInputDiagnostics(std::move(diagnostics), input.diagnostics, limits);
  if (!completeDiagnostics.hasValue()) {
    return Result<PropagationResult>::failure(
        completeDiagnostics.status().code(),
        std::string(completeDiagnostics.status().message()));
  }
  const auto covarianceCheck = classify(covarianceOutput.view());
  return Result<PropagationResult>::success(
      PropagationResult{.mean = std::move(mean),
                        .covariance = std::move(covarianceOutput),
                        .method = Approximation::fourth_amplitude,
                        .diagnostics = std::move(completeDiagnostics).value(),
                        .covariance_check = covarianceCheck});
}

Result<PropagationResult>
sigmaPointPropagate(ConstVectorView center, ConstMatrixView covariance,
                    const differentiation::VectorFunction& function,
                    const SigmaPointRule& rule, Limits limits) {
  const Index n = center.size();
  if (!function || !std::isfinite(rule.alpha) || rule.alpha <= 0.0 ||
      !std::isfinite(rule.beta) || !std::isfinite(rule.kappa) ||
      !std::isfinite(rule.solve.rank_abs) || rule.solve.rank_abs < 0.0 ||
      !std::isfinite(rule.solve.rank_rel) || rule.solve.rank_rel < 0.0 ||
      !std::isfinite(rule.solve.reciprocal_condition_floor) ||
      rule.solve.reciprocal_condition_floor < 0.0 ||
      rule.solve.reciprocal_condition_floor > 1.0 ||
      (rule.solve.rank != RankPolicy::reject &&
       rule.solve.rank != RankPolicy::supported_subspace &&
       rule.solve.rank != RankPolicy::minimum_norm)) {
    return fail<PropagationResult>(ErrorCode::InvalidDomain,
                                   "sigma-point callback or policy is invalid");
  }
  if (!finite(center)) {
    return fail<PropagationResult>(ErrorCode::NonFiniteInput,
                                   "sigma-point center is non-finite");
  }
  if (covariance.rows() != n || covariance.columns() != n) {
    return fail<PropagationResult>(
        ErrorCode::DimensionMismatch,
        "sigma-point covariance shape disagrees with center");
  }
  if (n > (std::numeric_limits<Extent>::max() - 1U) / 2U) {
    return fail<PropagationResult>(ErrorCode::SizeOverflow,
                                   "sigma-point count overflows");
  }
  const Extent pointCount = (2U * n) + 1U;
  if (limits.evaluations < pointCount) {
    return fail<PropagationResult>(
        ErrorCode::ResourceLimit,
        "sigma-point evaluation budget is insufficient");
  }
  Extent n2 = 0U;
  Extent pointElements = 0U;
  Extent eigensolverElements = 0U;
  Extent pointWorkspace = 0U;
  if (!checkedProduct(n, n, n2) ||
      !checkedProduct(pointCount, n, pointElements) ||
      !checkedProduct(n2, 3U, eigensolverElements) ||
      !checkedAdd(n2, eigensolverElements, pointWorkspace) ||
      !checkedAdd(pointWorkspace, pointElements, pointWorkspace) ||
      !checkedAdd(pointWorkspace, n, pointWorkspace)) {
    return fail<PropagationResult>(ErrorCode::SizeOverflow,
                                   "sigma-point storage overflows");
  }
  if (!withinLimits(pointWorkspace, limits)) {
    return fail<PropagationResult>(ErrorCode::ResourceLimit,
                                   "sigma-point inputs exceed storage limits");
  }

  Matrix root(n, n, 0.0);
  Index rank = 0U;
  if (n != 0U) {
    linear::SymmetricEigenOptions eigenOptions;
    eigenOptions.rankTolerance = {.absolute = rule.solve.rank_abs,
                                  .relative = rule.solve.rank_rel};
    if (rule.solve.iteration.max_iterations != 0U) {
      eigenOptions.maximumSweeps = rule.solve.iteration.max_iterations;
    }
    auto eigen = linear::decomposeSymmetricEigen(covariance, eigenOptions);
    if (!eigen.hasValue()) {
      return Result<PropagationResult>::failure(
          eigen.status().code(), std::string(eigen.status().message()));
    }
    if (eigen.value().definiteness() ==
        linear::PositiveDefiniteness::NotPositiveSemidefinite) {
      return fail<PropagationResult>(ErrorCode::NotPositiveSemidefinite,
                                     "sigma-point covariance is indefinite");
    }
    rank = eigen.value().rank();
    if (rule.solve.rank == RankPolicy::reject && rank != n) {
      return fail<PropagationResult>(
          ErrorCode::RankDeficient,
          "sigma-point covariance rank policy rejects singularity");
    }
    const Scalar largest = eigen.value().maximumEigenvalue();
    const Scalar smallest = eigen.value().minimumEigenvalue();
    if (rank != 0U && largest > 0.0 &&
        rule.solve.reciprocal_condition_floor > 0.0 && smallest > 0.0 &&
        smallest / largest < rule.solve.reciprocal_condition_floor) {
      return fail<PropagationResult>(
          ErrorCode::IllConditioned,
          "sigma-point covariance is below its reciprocal-condition floor");
    }
    const auto values = eigen.value().eigenvalues();
    const auto vectors = eigen.value().eigenvectors();
    for (Index i = 0U; i < n; ++i) {
      for (Index j = 0U; j < n; ++j) {
        long double value = 0.0L;
        for (Index k = 0U; k < n; ++k) {
          const Scalar lambda = values.at(k);
          if (lambda > eigen.value().rankThreshold()) {
            value += static_cast<long double>(vectors.at(i, k)) *
                     std::sqrt(lambda) * vectors.at(j, k);
          }
        }
        if (!std::isfinite(value) ||
            std::abs(value) > std::numeric_limits<Scalar>::max()) {
          return fail<PropagationResult>(
              ErrorCode::NonFiniteResult,
              "symmetric covariance root is not finite");
        }
        root.at(i, j) = static_cast<Scalar>(value);
      }
    }
  }

  const long double c = static_cast<long double>(rule.alpha) * rule.alpha *
                        (static_cast<long double>(n) + rule.kappa);
  if (!std::isfinite(c) || (n != 0U && c <= 0.0L) ||
      std::abs(c) > std::numeric_limits<Scalar>::max()) {
    return fail<PropagationResult>(
        ErrorCode::InvalidDomain,
        "scaled sigma-point spread must be finite and positive");
  }
  const long double meanCenterWeight = n == 0U ? 1.0L : 1.0L - (n / c);
  const long double offWeight = n == 0U ? 0.0L : 1.0L / (2.0L * c);
  const long double covarianceCenterWeight =
      n == 0U
          ? 1.0L
          : meanCenterWeight + 1.0L -
                (static_cast<long double>(rule.alpha) * rule.alpha) + rule.beta;
  if (!std::isfinite(meanCenterWeight) || !std::isfinite(offWeight) ||
      !std::isfinite(covarianceCenterWeight)) {
    return fail<PropagationResult>(ErrorCode::NonFiniteResult,
                                   "sigma-point weights are not finite");
  }

  auto evaluate = [&](ConstVectorView input,
                      std::optional<Index> expected) -> Result<Vector> {
    try {
      auto result = function(input);
      if (!result.hasValue()) {
        const auto code = result.status().code();
        return fail<Vector>(code == ErrorCode::DimensionMismatch
                                ? ErrorCode::DimensionMismatch
                                : ErrorCode::CallbackFailure,
                            "sigma-point map callback failed");
      }
      if (expected.has_value() && result.value().size() != *expected) {
        return fail<Vector>(ErrorCode::DimensionMismatch,
                            "sigma-point map output dimension changed");
      }
      if (!finite(result.value().view())) {
        return fail<Vector>(ErrorCode::NonFiniteResult,
                            "sigma-point map returned a non-finite value");
      }
      return result;
    } catch (...) {
      return fail<Vector>(ErrorCode::CallbackFailure,
                          "sigma-point map callback threw an exception");
    }
  };
  auto centerResult = evaluate(center, std::nullopt);
  if (!centerResult.hasValue()) {
    return Result<PropagationResult>::failure(
        centerResult.status().code(),
        std::string(centerResult.status().message()));
  }
  const Index m = centerResult.value().size();
  Extent outputCount = 0U;
  Extent sampleCount = 0U;
  Extent outputsWithMean = 0U;
  Extent allElements = 0U;
  if (!checkedProduct(m, m, outputCount) ||
      !checkedProduct(pointCount, m, sampleCount) ||
      !checkedAdd(outputCount, m, outputsWithMean) ||
      !checkedAdd(pointWorkspace, outputsWithMean, allElements) ||
      !checkedAdd(allElements, sampleCount, allElements)) {
    return fail<PropagationResult>(ErrorCode::SizeOverflow,
                                   "sigma-point output storage overflows");
  }
  if (allElements > limits.elements ||
      allElements > limits.bytes / sizeof(Scalar)) {
    return fail<PropagationResult>(
        ErrorCode::ResourceLimit,
        "sigma-point outputs exceed configured limits");
  }
  Matrix samples(pointCount, m, 0.0);
  for (Index j = 0U; j < m; ++j) {
    samples.at(0U, j) = centerResult.value().at(j);
  }
  const long double spread = n == 0U ? 0.0L : std::sqrt(c);
  for (Index i = 0U; i < n; ++i) {
    for (Index direction = 0U; direction < 2U; ++direction) {
      Vector point(n);
      for (Index j = 0U; j < n; ++j) {
        const long double offset = spread * root.at(j, i);
        const long double signedOffset = direction == 0U ? offset : -offset;
        const long double value =
            static_cast<long double>(center.at(j)) + signedOffset;
        if (!std::isfinite(value) ||
            std::abs(value) > std::numeric_limits<Scalar>::max()) {
          return fail<PropagationResult>(
              ErrorCode::NonFiniteResult,
              "sigma point is outside the scalar range");
        }
        point.at(j) = static_cast<Scalar>(value);
      }
      auto mapped = evaluate(point.view(), m);
      if (!mapped.hasValue()) {
        return Result<PropagationResult>::failure(
            mapped.status().code(), std::string(mapped.status().message()));
      }
      const Index row = 1U + (2U * i) + direction;
      for (Index j = 0U; j < m; ++j) {
        samples.at(row, j) = mapped.value().at(j);
      }
    }
  }
  Vector mean(m, 0.0);
  for (Index j = 0U; j < m; ++j) {
    WideSum sum;
    sum.add(meanCenterWeight * samples.at(0U, j));
    for (Index i = 1U; i < pointCount; ++i) {
      sum.add(offWeight * samples.at(i, j));
    }
    const long double value = sum.total();
    if (!std::isfinite(value) ||
        std::abs(value) > std::numeric_limits<Scalar>::max()) {
      return fail<PropagationResult>(ErrorCode::NonFiniteResult,
                                     "sigma-point mean is not finite");
    }
    mean.at(j) = static_cast<Scalar>(value);
  }
  Matrix outputCovariance(m, m, 0.0);
  for (Index i = 0U; i < m; ++i) {
    for (Index j = i; j < m; ++j) {
      WideSum sum;
      sum.add(covarianceCenterWeight * (samples.at(0U, i) - mean.at(i)) *
              (samples.at(0U, j) - mean.at(j)));
      for (Index k = 1U; k < pointCount; ++k) {
        sum.add(offWeight * (samples.at(k, i) - mean.at(i)) *
                (samples.at(k, j) - mean.at(j)));
      }
      const long double value = sum.total();
      if (!std::isfinite(value) ||
          std::abs(value) > std::numeric_limits<Scalar>::max()) {
        return fail<PropagationResult>(ErrorCode::NonFiniteResult,
                                       "sigma-point covariance is not finite");
      }
      outputCovariance.at(i, j) = static_cast<Scalar>(value);
      outputCovariance.at(j, i) = static_cast<Scalar>(value);
    }
  }
  auto diagnostics = propagationDiagnostics(
      Approximation::sigma_points, "scaled_symmetric_sigma_points",
      "Unscented sigma-point approximation; weights may be signed and "
      "covariance is not projected.");
  diagnostics.rank = rank;
  auto outputDiagnostics = analyzeCovariance(outputCovariance.view());
  auto covarianceCheck = classify(outputCovariance.view());
  if (outputDiagnostics.hasValue() &&
      outputDiagnostics.value().minimumEigenvalue() < 0.0 &&
      covarianceCheck == PropagationCovarianceCheck::positive_semidefinite) {
    covarianceCheck = PropagationCovarianceCheck::negative_within_roundoff;
  }
  return Result<PropagationResult>::success(
      PropagationResult{.mean = std::move(mean),
                        .covariance = std::move(outputCovariance),
                        .method = Approximation::sigma_points,
                        .diagnostics = std::move(diagnostics),
                        .covariance_check = covarianceCheck});
}

Result<SimulationResult>
monteCarloPropagate(const JointSampler& law,
                    const differentiation::VectorFunction& function,
                    const SimPolicy& policy, Rng& rng, Limits limits) {
  if (!law || !function || policy.draws == 0U || policy.batches == 0U ||
      policy.batches > policy.draws ||
      (policy.failure != InvalidDraw::reject &&
       policy.failure != InvalidDraw::report_conditional) ||
      !std::isfinite(policy.denominator_floor) ||
      policy.denominator_floor < 0.0) {
    return fail<SimulationResult>(
        ErrorCode::InvalidDomain,
        "Monte Carlo callbacks or policy are invalid");
  }
  const Index quantileCount =
      policy.quantiles.has_value() ? policy.quantiles->size() : 0U;
  if (policy.quantiles.has_value()) {
    for (Index i = 0U; i < quantileCount; ++i) {
      if (!std::isfinite(policy.quantiles->at(i)) ||
          policy.quantiles->at(i) < 0.0 || policy.quantiles->at(i) > 1.0) {
        return fail<SimulationResult>(
            ErrorCode::InvalidDomain,
            "quantile probabilities must lie in [0,1]");
      }
    }
  }
  if (policy.output_central_moments.has_value() &&
      (policy.output_central_moments->orders == 0U ||
       (policy.output_central_moments->orders & 0xF0U) != 0U)) {
    return fail<SimulationResult>(ErrorCode::InvalidDomain,
                                  "central-moment request is invalid");
  }
  if (policy.draws > std::numeric_limits<std::uint64_t>::max() / 2U) {
    return fail<SimulationResult>(ErrorCode::SizeOverflow,
                                  "Monte Carlo evaluation count overflows");
  }
  const std::uint64_t requiredEvaluations =
      static_cast<std::uint64_t>(policy.draws) * 2U;
  if (limits.evaluations < requiredEvaluations) {
    return fail<SimulationResult>(
        ErrorCode::ResourceLimit,
        "Monte Carlo callback budget is insufficient");
  }

  Index inputDimensionValue = std::numeric_limits<Index>::max();
  Index outputDimensionValue = std::numeric_limits<Index>::max();
  Index valid = 0U;
  Index rejected = 0U;
  Index actualCalls = 0U;
  Vector mean;
  std::vector<long double> centeredProducts;
  std::vector<long double> batchMeans;
  std::vector<Index> batchCounts;
  std::vector<Scalar> retainedValues;
  Limits momentLimits = limits;
  bool const retainSamples =
      quantileCount != 0U || policy.output_central_moments.has_value();
  const Index baseBatchSize = policy.draws / policy.batches;
  const Index extraBatchCount = policy.draws % policy.batches;
  Index activeBatch = 0U;
  Index nextBatchEnd = baseBatchSize;
  Index batchBoundaryRemainder = extraBatchCount;

  const auto conditionalEligible = [](ErrorCode code) {
    return code == ErrorCode::InvalidDomain ||
           code == ErrorCode::NonFiniteInput ||
           code == ErrorCode::NonFiniteResult;
  };
  const auto failDraw = [&](ErrorCode code,
                            const char* message) -> Result<SimulationResult> {
    return fail<SimulationResult>(code, message);
  };

  for (Index attempt = 0U; attempt < policy.draws; ++attempt) {
    Result<Vector> draw = Result<Vector>::failure(
        ErrorCode::CallbackFailure, "joint sampler did not return");
    try {
      ++actualCalls;
      draw = law(rng);
    } catch (...) {
      return failDraw(ErrorCode::CallbackFailure,
                      "joint sampler threw an exception");
    }
    if (!draw.hasValue()) {
      if (policy.failure == InvalidDraw::report_conditional &&
          conditionalEligible(draw.status().code())) {
        ++rejected;
        continue;
      }
      return failDraw(draw.status().code() == ErrorCode::DimensionMismatch
                          ? ErrorCode::DimensionMismatch
                          : ErrorCode::CallbackFailure,
                      "joint sampler rejected a draw");
    }
    if (inputDimensionValue == std::numeric_limits<Index>::max()) {
      inputDimensionValue = draw.value().size();
    } else if (draw.value().size() != inputDimensionValue) {
      return failDraw(ErrorCode::DimensionMismatch,
                      "joint sampler input dimension changed");
    }
    if (!finite(draw.value().view())) {
      if (policy.failure == InvalidDraw::report_conditional) {
        ++rejected;
        continue;
      }
      return failDraw(ErrorCode::NonFiniteInput,
                      "joint sampler produced a non-finite draw");
    }
    Result<Vector> mapped = Result<Vector>::failure(
        ErrorCode::CallbackFailure, "map callback did not return");
    try {
      ++actualCalls;
      mapped = function(draw.value().view());
    } catch (...) {
      return failDraw(ErrorCode::CallbackFailure,
                      "propagation map threw an exception");
    }
    if (!mapped.hasValue()) {
      if (policy.failure == InvalidDraw::report_conditional &&
          conditionalEligible(mapped.status().code())) {
        ++rejected;
        continue;
      }
      return failDraw(mapped.status().code() == ErrorCode::DimensionMismatch
                          ? ErrorCode::DimensionMismatch
                          : ErrorCode::CallbackFailure,
                      "propagation map rejected a draw");
    }
    const Index m = mapped.value().size();
    if (outputDimensionValue != std::numeric_limits<Index>::max() &&
        m != outputDimensionValue) {
      return failDraw(ErrorCode::DimensionMismatch,
                      "propagation map output dimension changed");
    }
    if (!finite(mapped.value().view())) {
      if (policy.failure == InvalidDraw::report_conditional) {
        ++rejected;
        continue;
      }
      return failDraw(ErrorCode::NonFiniteResult,
                      "propagation map returned a non-finite value");
    }
    if (outputDimensionValue == std::numeric_limits<Index>::max()) {
      outputDimensionValue = m;
      Extent m2 = 0U;
      Extent sampleElements = 0U;
      Extent batchElements = 0U;
      Extent quantileElements = 0U;
      if (!checkedProduct(m, m, m2) ||
          !checkedProduct(policy.draws, retainSamples ? m : 0U,
                          sampleElements) ||
          !checkedProduct(policy.batches, m, batchElements) ||
          !checkedProduct(m, quantileCount, quantileElements)) {
        return failDraw(ErrorCode::SizeOverflow,
                        "Monte Carlo summary storage overflows");
      }
      Extent count = 0U;
      const Extent sampleMatrixElements =
          policy.output_central_moments.has_value() ? sampleElements : 0U;
      const Extent quantileTemporary = quantileCount == 0U ? 0U : policy.draws;
      if (!checkedAdd(m2, m, count) ||
          !checkedAdd(count, batchElements, count) ||
          !checkedAdd(count, sampleElements, count) ||
          !checkedAdd(count, sampleMatrixElements, count) ||
          !checkedAdd(count, quantileElements, count) ||
          !checkedAdd(count, m2, count) || !checkedAdd(count, 3U * m, count) ||
          !checkedAdd(count, policy.batches, count) ||
          !checkedAdd(count, quantileTemporary, count)) {
        return failDraw(ErrorCode::SizeOverflow,
                        "Monte Carlo summary storage overflows");
      }
      Extent scalarByteElements = 0U;
      Extent wideElements = 0U;
      Extent scalarBytes = 0U;
      Extent wideBytes = 0U;
      Extent indexBytes = 0U;
      Extent requiredBytes = 0U;
      if (!checkedAdd(m2, 2U * m, scalarByteElements) ||
          !checkedAdd(scalarByteElements, sampleElements, scalarByteElements) ||
          !checkedAdd(scalarByteElements, sampleMatrixElements,
                      scalarByteElements) ||
          !checkedAdd(scalarByteElements, quantileElements,
                      scalarByteElements) ||
          !checkedAdd(scalarByteElements, quantileTemporary,
                      scalarByteElements) ||
          !checkedAdd(m2, batchElements, wideElements) ||
          !checkedAdd(wideElements, 2U * m, wideElements) ||
          !checkedProduct(scalarByteElements, sizeof(Scalar), scalarBytes) ||
          !checkedProduct(wideElements, sizeof(long double), wideBytes) ||
          !checkedProduct(policy.batches, sizeof(Index), indexBytes) ||
          !checkedAdd(scalarBytes, wideBytes, requiredBytes) ||
          !checkedAdd(requiredBytes, indexBytes, requiredBytes)) {
        return failDraw(ErrorCode::SizeOverflow,
                        "Monte Carlo summary byte count overflows");
      }
      if (count > limits.elements || requiredBytes > limits.bytes) {
        return failDraw(ErrorCode::ResourceLimit,
                        "Monte Carlo summaries exceed configured limits");
      }
      momentLimits = Limits{.elements = limits.elements - count,
                            .bytes = limits.bytes - requiredBytes,
                            .evaluations = limits.evaluations - actualCalls};
      if (policy.output_central_moments.has_value()) {
        if (momentLimits.elements < m ||
            momentLimits.bytes < m * sizeof(Scalar)) {
          return failDraw(ErrorCode::ResourceLimit,
                          "output-moment preflight exceeds remaining limits");
        }
        Matrix probe(1U, m, 0.0);
        const std::array<Index, 1> shape{m};
        const SampleView probeSamples{
            .rows = probe.view(), .observation_shape = shape, .valid = {}};
        const statistics::CenterSpec probeCenter{.rule = CenterRule::estimated,
                                                 .center = std::nullopt};
        const statistics::WeightSpec probeWeights{
            .rule = WeightRule::unweighted,
            .rowWeights = std::nullopt,
            .componentWeights = std::nullopt,
            .frequencyCounts = {}};
        const Limits probeLimits{.elements = momentLimits.elements - m,
                                 .bytes =
                                     momentLimits.bytes - (m * sizeof(Scalar)),
                                 .evaluations = momentLimits.evaluations};
        auto preflight = statistics::estimateCentralMoments(
            probeSamples, probeWeights, probeCenter,
            *policy.output_central_moments, probeLimits);
        if (!preflight.hasValue()) {
          return Result<SimulationResult>::failure(
              preflight.status().code(),
              std::string(preflight.status().message()));
        }
      }
      mean = Vector(m, 0.0);
      centeredProducts.assign(m2, 0.0L);
      batchMeans.assign(batchElements, 0.0L);
      batchCounts.assign(policy.batches, 0U);
      if (retainSamples) {
        retainedValues.reserve(sampleElements);
      }
    }

    const Index nextValid = valid + 1U;
    std::vector<long double> delta(m);
    std::vector<long double> deltaAfter(m);
    for (Index j = 0U; j < m; ++j) {
      delta[j] = static_cast<long double>(mapped.value().at(j)) - mean.at(j);
      mean.at(j) = static_cast<Scalar>(static_cast<long double>(mean.at(j)) +
                                       (delta[j] / nextValid));
      deltaAfter[j] =
          static_cast<long double>(mapped.value().at(j)) - mean.at(j);
    }
    for (Index i = 0U; i < m; ++i) {
      for (Index j = 0U; j < m; ++j) {
        const long double updated =
            centeredProducts[(i * m) + j] + (delta[i] * deltaAfter[j]);
        if (!std::isfinite(updated)) {
          return failDraw(ErrorCode::NonFiniteResult,
                          "online covariance accumulation is not finite");
        }
        centeredProducts[(i * m) + j] = updated;
      }
    }
    while (attempt >= nextBatchEnd && activeBatch + 1U < policy.batches) {
      ++activeBatch;
      nextBatchEnd += baseBatchSize;
      if (batchBoundaryRemainder >= policy.batches - extraBatchCount) {
        ++nextBatchEnd;
        batchBoundaryRemainder -= policy.batches - extraBatchCount;
      } else {
        batchBoundaryRemainder += extraBatchCount;
      }
    }
    const Index batch = activeBatch;
    const Index batchCount = ++batchCounts[batch];
    for (Index j = 0U; j < m; ++j) {
      const Index offset = (batch * m) + j;
      batchMeans[offset] += (static_cast<long double>(mapped.value().at(j)) -
                             batchMeans[offset]) /
                            batchCount;
      if (retainSamples) {
        retainedValues.push_back(mapped.value().at(j));
      }
    }
    valid = nextValid;
  }

  if (valid == 0U ||
      outputDimensionValue == std::numeric_limits<Index>::max()) {
    return fail<SimulationResult>(ErrorCode::InsufficientSamples,
                                  "no valid Monte Carlo draws were retained");
  }
  const Index m = outputDimensionValue;
  Matrix outputCovariance(m, m, 0.0);
  for (Index i = 0U; i < m; ++i) {
    for (Index j = i; j < m; ++j) {
      const long double value =
          0.5L *
          (centeredProducts[(i * m) + j] + centeredProducts[(j * m) + i]) /
          valid;
      if (!std::isfinite(value) ||
          std::abs(value) > std::numeric_limits<Scalar>::max()) {
        return fail<SimulationResult>(ErrorCode::NonFiniteResult,
                                      "Monte Carlo covariance is not finite");
      }
      outputCovariance.at(i, j) = static_cast<Scalar>(value);
      outputCovariance.at(j, i) = static_cast<Scalar>(value);
    }
  }
  Matrix quantiles(m, quantileCount, 0.0);
  if (policy.quantiles.has_value() && quantileCount != 0U) {
    std::vector<Scalar> coordinateValues(valid);
    for (Index coordinate = 0U; coordinate < m; ++coordinate) {
      for (Index row = 0U; row < valid; ++row) {
        coordinateValues[row] = retainedValues[(row * m) + coordinate];
      }
      std::ranges::sort(coordinateValues);
      for (Index q = 0U; q < quantileCount; ++q) {
        const long double position = static_cast<long double>(valid - 1U) *
                                     policy.quantiles.value().at(q);
        const auto lower = static_cast<Index>(std::floor(position));
        const Index upper = std::min(valid - 1U, lower + 1U);
        const long double fraction = position - lower;
        const long double value =
            coordinateValues[lower] +
            (fraction * (static_cast<long double>(coordinateValues[upper]) -
                         coordinateValues[lower]));
        if (!std::isfinite(value) ||
            std::abs(value) > std::numeric_limits<Scalar>::max()) {
          return fail<SimulationResult>(ErrorCode::NonFiniteResult,
                                        "empirical quantile is not finite");
        }
        quantiles.at(coordinate, q) = static_cast<Scalar>(value);
      }
    }
  }
  Vector monteCarloError;
  if (policy.batches >= 2U && policy.draws % policy.batches == 0U &&
      rejected == 0U) {
    const Index expectedBatch = policy.draws / policy.batches;
    const bool equalComplete = std::ranges::all_of(
        batchCounts, [&](Index count) { return count == expectedBatch; });
    if (equalComplete) {
      monteCarloError = Vector(m, 0.0);
      for (Index coordinate = 0U; coordinate < m; ++coordinate) {
        long double average = 0.0L;
        for (Index b = 0U; b < policy.batches; ++b) {
          average += batchMeans[(b * m) + coordinate] / policy.batches;
        }
        long double squared = 0.0L;
        for (Index b = 0U; b < policy.batches; ++b) {
          const long double difference =
              batchMeans[(b * m) + coordinate] - average;
          squared += difference * difference;
        }
        const long double standardError =
            std::sqrt(squared / (policy.batches - 1U) / policy.batches);
        if (!std::isfinite(standardError) ||
            standardError > std::numeric_limits<Scalar>::max()) {
          return fail<SimulationResult>(ErrorCode::NonFiniteResult,
                                        "batch standard error is not finite");
        }
        monteCarloError.at(coordinate) = static_cast<Scalar>(standardError);
      }
    }
  }

  std::optional<MomentEstimate> centralMoments;
  if (policy.output_central_moments.has_value()) {
    if (m == 0U) {
      return fail<SimulationResult>(
          ErrorCode::InvalidDomain,
          "central moments are undefined for an empty output");
    }
    Matrix sampleRows(valid, m, 0.0);
    for (Index row = 0U; row < valid; ++row) {
      for (Index column = 0U; column < m; ++column) {
        sampleRows.at(row, column) = retainedValues[(row * m) + column];
      }
    }
    const std::array<Index, 1> shape{m};
    const SampleView samples{
        .rows = sampleRows.view(), .observation_shape = shape, .valid = {}};
    const statistics::CenterSpec center{.rule = CenterRule::estimated,
                                        .center = std::nullopt};
    const statistics::WeightSpec weights{.rule = WeightRule::unweighted,
                                         .rowWeights = std::nullopt,
                                         .componentWeights = std::nullopt,
                                         .frequencyCounts = {}};
    auto moments = statistics::estimateCentralMoments(
        samples, weights, center, *policy.output_central_moments, momentLimits);
    if (!moments.hasValue()) {
      return Result<SimulationResult>::failure(
          moments.status().code(), std::string(moments.status().message()));
    }
    centralMoments = std::move(moments).value();
  }

  auto diagnostics = propagationDiagnostics(
      Approximation::monte_carlo, "joint_monte_carlo_pushforward",
      policy.failure == InvalidDraw::report_conditional
          ? "Summary is conditional on draws admitted by the selected "
            "invalid-draw policy."
          : "All requested attempts must be valid draws from the supplied "
            "joint sampler.");
  diagnostics.global = Support{.input = policy.draws,
                               .valid = valid,
                               .positive = valid,
                               .frequency_mass = std::nullopt,
                               .weight_mass = std::nullopt,
                               .normalized_weight_powers = {},
                               .effective_count = std::nullopt,
                               .degrees_of_freedom = std::nullopt,
                               .squared_weight_mass = std::nullopt};
  if (monteCarloError.empty()) {
    diagnostics.premises +=
        " Batch-mean standard error is unavailable because batches were not at "
        "least two equal-sized all-valid groups.";
  }
  diagnostics.algorithm += ":calls=" + std::to_string(actualCalls);
  const auto covarianceCheck = classify(outputCovariance.view());
  PropagationResult estimate{.mean = std::move(mean),
                             .covariance = std::move(outputCovariance),
                             .method = Approximation::monte_carlo,
                             .diagnostics = std::move(diagnostics),
                             .covariance_check = covarianceCheck};
  return Result<SimulationResult>::success(
      SimulationResult{.estimate = std::move(estimate),
                       .quantiles = std::move(quantiles),
                       .monte_carlo_error = std::move(monteCarloError),
                       .valid = valid,
                       .rejected = rejected,
                       .central_moments = std::move(centralMoments)});
}

} // namespace vss::math::covariance
