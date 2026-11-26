#include "Test.hpp"
#include "vss/math/Array.hpp"
#include "vss/math/Core.hpp"
#include "vss/math/Error.hpp"
#include "vss/math/Limits.hpp"
#include "vss/math/Matrix.hpp"
#include "vss/math/Result.hpp"
#include "vss/math/SampleView.hpp"
#include "vss/math/Tensor.hpp"
#include "vss/math/TensorShape.hpp"
#include "vss/math/Vector.hpp"
#include "vss/math/VectorView.hpp"
#include "vss/math/covariance/PropagationResult.hpp"
#include "vss/math/differentiation/FiniteDifference.hpp"
#include "vss/math/linear/SolvePolicy.hpp"
#include "vss/math/statistics/Moments.hpp"
#include "vss/math/statistics/Population.hpp"
#include "vss/math/statistics/Weighted.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <numbers>
#include <optional>
#include <span>
#include <utility>
#include <vss/math/covariance/Budget.hpp>
#include <vss/math/covariance/Propagation.hpp>

namespace {
using namespace vss::math;

Matrix matrix(std::initializer_list<std::initializer_list<Scalar>> rows) {
  const Index rowCount = rows.size();
  const Index columnCount = rows.size() == 0U ? 0U : rows.begin()->size();
  Matrix value(rowCount, columnCount);
  Index i = 0U;
  for (const auto& row : rows) {
    Index j = 0U;
    for (const Scalar entry : row) {
      value.at(i, j++) = entry;
    }
    ++i;
  }
  return value;
}

Tensor tensor(std::span<const Extent> shape, Scalar fill = 0.0) {
  auto ownedShape = TensorShape::create(shape);
  VSS_CHECK(ownedShape.hasValue());
  auto value = Tensor::create(ownedShape.value(), fill, TensorLimits{4096U});
  VSS_CHECK(value.hasValue());
  return std::move(value).value();
}

statistics::MomentEstimate moments(std::span<const Scalar> observations,
                                   Index orderBits) {
  Matrix rows(observations.size(), 1U);
  for (Index i = 0U; i < observations.size(); ++i) {
    rows.at(i, 0U) = observations[i];
  }
  const SampleView sample{
      .rows = rows.view(), .observation_shape = {}, .valid = {}};
  const statistics::CenterSpec center{.rule = CenterRule::estimated,
                                      .center = std::nullopt};
  const statistics::MomentRequest request{
      .orders = static_cast<std::uint8_t>(orderBits),
      .representation = statistics::MomentRepresentation::dense,
      .selected = {},
      .contraction = std::nullopt};
  const statistics::WeightSpec weights{.rule = WeightRule::unweighted,
                                       .rowWeights = std::nullopt,
                                       .componentWeights = std::nullopt,
                                       .frequencyCounts = {}};
  auto result =
      statistics::estimateCentralMoments(sample, weights, center, request,
                                         Limits{.elements = 4096U,
                                                .bytes = 4096U * sizeof(Scalar),
                                                .evaluations = 0U});
  VSS_CHECK(result.hasValue());
  return std::move(result).value();
}

void quadraticSkewedPopulation() {
  const std::array<Scalar, 3> values{-1.0, -1.0, 2.0};
  auto input = moments(values, 0b1110U);
  const std::array<Extent, 3> shape{1U, 1U, 1U};
  Tensor hessian = tensor(shape);
  const std::array<Index, 3> hIndex{0U, 0U, 0U};
  hessian.at(hIndex) = 2.0;
  const Vector center{1U, 0.0};
  const Matrix jacobian = matrix({{1.0}});
  const QuadraticMap map{.at_center = center.view(),
                         .jacobian = jacobian.view(),
                         .hessian = hessian};
  auto result = covariance::propagateQuadraticTaylor(
      map, input,
      Limits{.elements = 4096U,
             .bytes = 4096U * sizeof(Scalar),
             .evaluations = 0U});
  VSS_CHECK(result.hasValue());
  VSS_CHECK(std::abs(result.value().mean.at(0U) - 2.0) < 1.0e-12);
  VSS_CHECK(std::abs(result.value().covariance.at(0U, 0U) - 8.0) < 1.0e-12);
  VSS_CHECK(result.value().diagnostics.estimand == Estimand::observed);
  VSS_CHECK(result.value().diagnostics.global.input == 3U);
  VSS_CHECK(result.value().diagnostics.center == CenterRule::estimated);
}

void quadraticSharedSkewedInput() {
  const std::array<Scalar, 3> values{-1.0, -1.0, 2.0};
  auto input = moments(values, 0b1110U);
  const Vector center{2U, 0.0};
  const Matrix jacobian = matrix({{2.0}, {0.0}});
  Tensor hessian = tensor(std::array<Extent, 3>{2U, 1U, 1U});
  hessian.at(std::array<Index, 3>{1U, 0U, 0U}) = 2.0;
  const auto result = covariance::propagateQuadraticTaylor(
      {.at_center = center.view(),
       .jacobian = jacobian.view(),
       .hessian = hessian},
      input,
      Limits{.elements = 1024U,
             .bytes = 1024U * sizeof(Scalar),
             .evaluations = 0U});
  VSS_CHECK(result.hasValue());
  VSS_CHECK(result.value().mean.at(0U) == 0.0);
  VSS_CHECK(result.value().mean.at(1U) == 2.0);
  VSS_CHECK(result.value().covariance.at(0U, 0U) == 8.0);
  VSS_CHECK(result.value().covariance.at(0U, 1U) == 4.0);
  VSS_CHECK(result.value().covariance.at(1U, 0U) == 4.0);
  VSS_CHECK(result.value().covariance.at(1U, 1U) == 2.0);
}

void missingMomentAndMissingMeanFailSafely() {
  const std::array<Scalar, 3> values{-1.0, -1.0, 2.0};
  auto secondOnly = moments(values, 0b0010U);
  const Vector center{1U, 0.0};
  const Matrix jacobian = matrix({{0.0}});
  Tensor hessian = tensor(std::array<Extent, 3>{1U, 1U, 1U});
  hessian.at(std::array<Index, 3>{0U, 0U, 0U}) = 2.0;
  const QuadraticMap map{.at_center = center.view(),
                         .jacobian = jacobian.view(),
                         .hessian = hessian};
  const auto missingFourth = covariance::propagateQuadraticTaylor(
      map, secondOnly,
      Limits{.elements = 1024U,
             .bytes = 1024U * sizeof(Scalar),
             .evaluations = 0U});
  VSS_CHECK(!missingFourth.hasValue());
  VSS_CHECK(missingFourth.status().code() == ErrorCode::InvalidDomain);

  auto missingMean = moments(values, 0b1110U);
  missingMean.population_mean.reset();
  const auto noMean = covariance::propagateQuadraticTaylor(
      map, missingMean,
      Limits{.elements = 1024U,
             .bytes = 1024U * sizeof(Scalar),
             .evaluations = 0U});
  VSS_CHECK(!noMean.hasValue());
  VSS_CHECK(noMean.status().code() == ErrorCode::InvalidDomain);
}

void gaussianQuadratic() {
  const std::array<Extent, 3> shape{1U, 1U, 1U};
  Tensor hessian = tensor(shape);
  const std::array<Index, 3> hIndex{0U, 0U, 0U};
  hessian.at(hIndex) = 1.0;
  const Vector center{1U, 0.0};
  const Matrix jacobian = matrix({{0.0}});
  const Matrix covariance = matrix({{1.0}});
  const QuadraticMap map{.at_center = center.view(),
                         .jacobian = jacobian.view(),
                         .hessian = hessian};
  auto result = covariance::propagateGaussianQuadraticTaylor(
      map, covariance.view(),
      Limits{.elements = 4096U,
             .bytes = 4096U * sizeof(Scalar),
             .evaluations = 0U});
  VSS_CHECK(result.hasValue());
  VSS_CHECK(std::abs(result.value().mean.at(0U) - 0.5) < 1.0e-12);
  VSS_CHECK(std::abs(result.value().covariance.at(0U, 0U) - 0.5) < 1.0e-12);
}

void cubicFourthAmplitude() {
  const std::array<Scalar, 3> values{-1.0, -1.0, 2.0};
  auto input = moments(values, 0b1110U);
  const std::array<Extent, 3> hessianShape{1U, 1U, 1U};
  const std::array<Extent, 4> thirdShape{1U, 1U, 1U, 1U};
  Tensor hessian = tensor(hessianShape);
  Tensor third = tensor(thirdShape);
  const std::array<Index, 3> hIndex{0U, 0U, 0U};
  const std::array<Index, 4> kIndex{0U, 0U, 0U, 0U};
  hessian.at(hIndex) = 2.0;
  third.at(kIndex) = 6.0;
  const Vector center{1U, 0.0};
  const Matrix jacobian = matrix({{1.0}});
  const QuadraticMap quadratic{.at_center = center.view(),
                               .jacobian = jacobian.view(),
                               .hessian = hessian};
  const CubicMap map{.quadratic = quadratic, .thirds = third, .amplitude = 0.5};
  auto result = covariance::propagateFourthOrderCovariance(
      map, input,
      Limits{.elements = 4096U,
             .bytes = 4096U * sizeof(Scalar),
             .evaluations = 0U});
  VSS_CHECK(result.hasValue());
  VSS_CHECK(std::abs(result.value().mean.at(0U) - 0.75) < 1.0e-12);
  VSS_CHECK(std::abs(result.value().covariance.at(0U, 0U) - (15.0 / 8.0)) <
            1.0e-12);
}

void signedFourthAmplitudeIsPreserved() {
  const std::array<Scalar, 2> values{-1.0, 1.0};
  auto input = moments(values, 0b1110U);
  const std::array<Extent, 3> hessianShape{1U, 1U, 1U};
  const std::array<Extent, 4> thirdShape{1U, 1U, 1U, 1U};
  Tensor const hessian = tensor(hessianShape);
  Tensor third = tensor(thirdShape);
  const std::array<Index, 4> kIndex{0U, 0U, 0U, 0U};
  third.at(kIndex) = -6.0;
  const Vector center{1U, 0.0};
  const Matrix jacobian = matrix({{1.0}});
  const CubicMap map{.quadratic = {.at_center = center.view(),
                                   .jacobian = jacobian.view(),
                                   .hessian = hessian},
                     .thirds = third,
                     .amplitude = 1.0};
  auto result = covariance::propagateFourthOrderCovariance(
      map, input,
      Limits{.elements = 4096U,
             .bytes = 4096U * sizeof(Scalar),
             .evaluations = 0U});
  VSS_CHECK(result.hasValue());
  VSS_CHECK(result.value().covariance.at(0U, 0U) == -1.0);
  VSS_CHECK(result.value().covariance_check ==
            PropagationCovarianceCheck::indefinite_approximation);
}

void cubicSymmetricLawExposesOmittedTail() {
  const std::array<Scalar, 2> values{-1.0, 1.0};
  auto input = moments(values, 0b1110U);
  const Vector center{1U, 0.0};
  const Matrix jacobian = matrix({{0.0}});
  const Tensor hessian = tensor(std::array<Extent, 3>{1U, 1U, 1U});
  Tensor third = tensor(std::array<Extent, 4>{1U, 1U, 1U, 1U});
  third.at(std::array<Index, 4>{0U, 0U, 0U, 0U}) = 6.0;
  const CubicMap map{.quadratic = {.at_center = center.view(),
                                   .jacobian = jacobian.view(),
                                   .hessian = hessian},
                     .thirds = third,
                     .amplitude = 1.0};
  const auto result = covariance::propagateFourthOrderCovariance(
      map, input,
      Limits{
          .elements = 256U, .bytes = 256U * sizeof(Scalar), .evaluations = 0U});
  VSS_CHECK(result.hasValue());
  VSS_CHECK(result.value().method == Approximation::fourth_amplitude);
  VSS_CHECK(result.value().covariance.at(0U, 0U) == 0.0);
  VSS_CHECK(result.value().diagnostics.estimand == Estimand::observed);
  VSS_CHECK(result.value().diagnostics.global.input == 2U);
  VSS_CHECK(result.value().diagnostics.algorithm ==
            "fourth_amplitude_cubic_taylor");
}

void cubicDerivativeTensorLayout() {
  Tensor hessian = tensor(std::array<Extent, 3>{2U, 3U, 3U});
  Tensor third = tensor(std::array<Extent, 4>{2U, 3U, 3U, 3U});
  const auto setH = [&](Index output, Index i, Index j, Scalar value) {
    hessian.at(std::array<Index, 3>{output, i, j}) = value;
  };
  const auto setK = [&](Index output, Index i, Index j, Index k, Scalar value) {
    third.at(std::array<Index, 4>{output, i, j, k}) = value;
  };
  setH(0U, 0U, 0U, 4.0);
  setH(0U, 0U, 1U, 2.0);
  setH(0U, 1U, 0U, 2.0);
  setH(0U, 1U, 1U, 12.0);
  setH(1U, 0U, 1U, 3.0);
  setH(1U, 1U, 0U, 3.0);
  setH(1U, 0U, 2U, 2.0);
  setH(1U, 2U, 0U, 2.0);
  setH(1U, 1U, 2U, 1.0);
  setH(1U, 2U, 1U, 1.0);
  for (const auto& permutation :
       {std::array<Index, 3>{0U, 0U, 1U}, {0U, 1U, 0U}, {1U, 0U, 0U}}) {
    setK(0U, permutation[0], permutation[1], permutation[2], 2.0);
  }
  setK(0U, 1U, 1U, 1U, 6.0);
  for (const auto& permutation : {std::array<Index, 3>{0U, 1U, 2U},
                                  {0U, 2U, 1U},
                                  {1U, 0U, 2U},
                                  {1U, 2U, 0U},
                                  {2U, 0U, 1U},
                                  {2U, 1U, 0U}}) {
    setK(1U, permutation[0], permutation[1], permutation[2], 1.0);
  }
  VSS_CHECK(hessian.at(std::array<Index, 3>{0U, 0U, 0U}) == 4.0);
  VSS_CHECK(hessian.at(std::array<Index, 3>{0U, 0U, 1U}) == 2.0);
  VSS_CHECK(hessian.at(std::array<Index, 3>{0U, 1U, 0U}) == 2.0);
  VSS_CHECK(hessian.at(std::array<Index, 3>{0U, 1U, 1U}) == 12.0);
  VSS_CHECK(hessian.at(std::array<Index, 3>{1U, 1U, 2U}) == 1.0);
  VSS_CHECK(third.at(std::array<Index, 4>{0U, 1U, 1U, 1U}) == 6.0);
  VSS_CHECK(third.at(std::array<Index, 4>{1U, 0U, 1U, 2U}) == 1.0);
  VSS_CHECK(third.at(std::array<Index, 4>{1U, 2U, 1U, 0U}) == 1.0);

  const Vector center{2U, 10.0};
  const Matrix jacobian = matrix({{4.0, 13.0, 0.0}, {6.0, 3.0, 2.0}});
  const CubicMap map{.quadratic = {.at_center = center.view(),
                                   .jacobian = jacobian.view(),
                                   .hessian = hessian},
                     .thirds = third,
                     .amplitude = 1.0};
  const auto invalid = covariance::propagateFourthOrderCovariance(
      map, moments(std::array<Scalar, 2>{-1.0, 1.0}, 0b1110U),
      Limits{.elements = 4096U,
             .bytes = 4096U * sizeof(Scalar),
             .evaluations = 0U});
  VSS_CHECK(!invalid.hasValue());
  VSS_CHECK(invalid.status().code() == ErrorCode::DimensionMismatch);
}

void selectedMomentCoverageIsExact() {
  const std::array<Extent, 1> shapeData{2U};
  const std::array<Extent, 2> secondTuple{0U, 0U};
  const std::array<Extent, 4> fourthTuple{0U, 0U, 0U, 0U};
  const auto shape = Array<Index>::copy(
      shapeData,
      Limits{.elements = 8U, .bytes = 8U * sizeof(Index), .evaluations = 0U});
  const auto secondIndices = Array<Index>::copy(
      secondTuple,
      Limits{.elements = 8U, .bytes = 8U * sizeof(Index), .evaluations = 0U});
  const auto fourthIndices = Array<Index>::copy(
      fourthTuple,
      Limits{.elements = 8U, .bytes = 8U * sizeof(Index), .evaluations = 0U});
  VSS_CHECK(shape.hasValue() && secondIndices.hasValue() &&
            fourthIndices.hasValue());
  std::array<statistics::MomentTerm, 2> terms{
      {{.order = 2U,
        .layout = statistics::MomentRepresentation::selected,
        .flattened_tuples = secondIndices.value(),
        .values = Vector{1U, 2.0}},
       {.order = 4U,
        .layout = statistics::MomentRepresentation::selected,
        .flattened_tuples = fourthIndices.value(),
        .values = Vector{1U, 6.0}}}};
  auto ownedTerms = Array<statistics::MomentTerm>::copy(
      terms, Limits{.elements = 8U,
                    .bytes = 8U * sizeof(statistics::MomentTerm),
                    .evaluations = 0U});
  VSS_CHECK(ownedTerms.hasValue());
  const EstimatorDiagnostics diagnostics{
      .estimand = Estimand::observed,
      .weight = WeightRule::unweighted,
      .center = CenterRule::estimated,
      .approximation = Approximation::exact,
      .global = Support{.input = 3U,
                        .valid = 3U,
                        .positive = 3U,
                        .frequency_mass = std::nullopt,
                        .weight_mass = std::nullopt,
                        .normalized_weight_powers = {},
                        .effective_count = std::nullopt,
                        .degrees_of_freedom = std::nullopt,
                        .squared_weight_mass = std::nullopt},
      .coordinate_support = {},
      .tuple_support = {},
      .reciprocal_condition = std::nullopt,
      .numerical_error = std::nullopt,
      .sampling_error = std::nullopt,
      .rank = std::nullopt,
      .algorithm = "selected_fixture",
      .premises = "common support",
      .warnings = {}};
  MomentEstimate estimate{.observation_shape = shape.value(),
                          .terms = ownedTerms.value(),
                          .diagnostics = diagnostics,
                          .population_mean = Vector{2U, 0.0}};
  Tensor hessian = tensor(std::array<Extent, 3>{1U, 2U, 2U});
  hessian.at(std::array<Index, 3>{0U, 0U, 0U}) = 2.0;
  const Vector center{1U, 0.0};
  const Matrix jacobian(1U, 2U, 0.0);
  const auto selected = covariance::propagateQuadraticTaylor(
      {.at_center = center.view(),
       .jacobian = jacobian.view(),
       .hessian = hessian},
      estimate,
      Limits{
          .elements = 64U, .bytes = 64U * sizeof(Scalar), .evaluations = 0U});
  VSS_CHECK(selected.hasValue());
  VSS_CHECK(selected.value().mean.at(0U) == 2.0);
  VSS_CHECK(selected.value().covariance.at(0U, 0U) == 2.0);
  terms[1U].values.at(0U) = 5.0;
  auto missingFourthTerms = Array<statistics::MomentTerm>::copy(
      std::span<const statistics::MomentTerm>(terms.data(), 1U),
      Limits{.elements = 8U,
             .bytes = 8U * sizeof(statistics::MomentTerm),
             .evaluations = 0U});
  VSS_CHECK(missingFourthTerms.hasValue());
  estimate.terms = missingFourthTerms.value();
  const auto missing = covariance::propagateQuadraticTaylor(
      {.at_center = center.view(),
       .jacobian = jacobian.view(),
       .hessian = hessian},
      estimate,
      Limits{
          .elements = 64U, .bytes = 64U * sizeof(Scalar), .evaluations = 0U});
  VSS_CHECK(!missing.hasValue());
  VSS_CHECK(missing.status().code() == ErrorCode::InvalidDomain);
}

void sourceContributionBudget() {
  const std::array<Index, 1> firstCoordinate{0U};
  const std::array<Index, 1> secondCoordinate{1U};
  const Matrix firstCovariance = matrix({{1.0}});
  const Matrix secondCovariance = matrix({{1.0}});
  const Matrix crossCovariance = matrix({{0.5}});
  const std::array<SourceBlock, 2> sources{
      {{.id = 10U,
        .coordinates = firstCoordinate,
        .covariance = firstCovariance.view()},
       {.id = 20U,
        .coordinates = secondCoordinate,
        .covariance = secondCovariance.view()}}};
  const std::array<SourceCrossBlock, 1> cross{
      {{.left = 10U, .right = 20U, .covariance = crossCovariance.view()}}};
  const Matrix jacobian = matrix({{1.0, 1.0}});
  auto joint = covariance::uncertaintyContributionBudget(
      jacobian.view(), sources, cross, DependencePolicy::exact_joint,
      Limits{
          .elements = 128U, .bytes = 128U * sizeof(Scalar), .evaluations = 0U});
  VSS_CHECK(joint.hasValue());
  VSS_CHECK(std::abs(joint.value().covariance.at(0U, 0U) - 3.0) < 1.0e-12);
  VSS_CHECK(joint.value().terms.view().size() == 3U);
  VSS_CHECK(std::abs(joint.value().terms.view()[1U].contribution.at(0U, 0U) -
                     1.0) < 1.0e-12);

  const std::array<SourceBlock, 2> independentSources{
      {{.id = 10U,
        .coordinates = firstCoordinate,
        .covariance = firstCovariance.view()},
       {.id = 20U,
        .coordinates = secondCoordinate,
        .covariance = secondCovariance.view()}}};
  auto independent = covariance::uncertaintyContributionBudget(
      jacobian.view(), independentSources, {},
      DependencePolicy::declared_independent_blocks,
      Limits{
          .elements = 128U, .bytes = 128U * sizeof(Scalar), .evaluations = 0U});
  VSS_CHECK(independent.hasValue());
  VSS_CHECK(std::abs(independent.value().covariance.at(0U, 0U) - 2.0) <
            1.0e-12);
}

void reversedRectangularSourceCrossBlock() {
  const std::array<Index, 2> firstCoordinates{1U, 0U};
  const std::array<Index, 1> secondCoordinates{2U};
  const Matrix firstCovariance = matrix({{2.0, 0.5}, {0.5, 3.0}});
  const Matrix secondCovariance = matrix({{4.0}});
  const Matrix reversedCross = matrix({{0.2, -0.3}});
  const std::array<SourceBlock, 2> sources{
      {{.id = 10U,
        .coordinates = firstCoordinates,
        .covariance = firstCovariance.view()},
       {.id = 20U,
        .coordinates = secondCoordinates,
        .covariance = secondCovariance.view()}}};
  const std::array<SourceCrossBlock, 1> cross{
      {{.left = 20U, .right = 10U, .covariance = reversedCross.view()}}};
  const Matrix jacobian = matrix({{1.0, 2.0, 0.0}, {0.0, 1.0, 1.0}});
  auto result = covariance::uncertaintyContributionBudget(
      jacobian.view(), sources, cross, DependencePolicy::exact_joint,
      Limits{
          .elements = 512U, .bytes = 512U * sizeof(Scalar), .evaluations = 0U});
  VSS_CHECK(result.hasValue());
  const Matrix expected = matrix({{13.0, 4.6}, {4.6, 6.4}});
  for (Index i = 0U; i < 2U; ++i) {
    for (Index j = 0U; j < 2U; ++j) {
      VSS_CHECK(std::abs(result.value().covariance.at(i, j) -
                         expected.at(i, j)) < 1.0e-12);
    }
  }
  const auto terms = result.value().terms.view();
  VSS_CHECK(terms.size() == 3U);
  VSS_CHECK(terms[1U].left == 10U && terms[1U].right == 20U);
  const Matrix expectedCross = matrix({{0.0, 0.1}, {0.1, 0.4}});
  for (Index i = 0U; i < 2U; ++i) {
    for (Index j = 0U; j < 2U; ++j) {
      VSS_CHECK(std::abs(terms[1U].contribution.at(i, j) -
                         expectedCross.at(i, j)) < 1.0e-12);
    }
  }
}

void sourceBudgetRejectsDuplicateAndInvalidJoint() {
  const std::array<Index, 1> firstCoordinate{0U};
  const std::array<Index, 1> secondCoordinate{1U};
  const Matrix variance = matrix({{1.0}});
  const std::array<SourceBlock, 2> sources{{{.id = 1U,
                                             .coordinates = firstCoordinate,
                                             .covariance = variance.view()},
                                            {.id = 2U,
                                             .coordinates = secondCoordinate,
                                             .covariance = variance.view()}}};
  const Matrix zeroCross = matrix({{0.0}});
  const std::array<SourceCrossBlock, 2> duplicate{
      {{.left = 1U, .right = 2U, .covariance = zeroCross.view()},
       {.left = 2U, .right = 1U, .covariance = zeroCross.view()}}};
  const Matrix jacobian = matrix({{1.0, 1.0}});
  const Limits limits{
      .elements = 256U, .bytes = 256U * sizeof(Scalar), .evaluations = 0U};
  const auto duplicated = covariance::uncertaintyContributionBudget(
      jacobian.view(), sources, duplicate, DependencePolicy::exact_joint,
      limits);
  VSS_CHECK(!duplicated.hasValue());
  VSS_CHECK(duplicated.status().code() == ErrorCode::InvalidDomain);

  const Matrix excessiveCross = matrix({{2.0}});
  const std::array<SourceCrossBlock, 1> indefinite{
      {{.left = 1U, .right = 2U, .covariance = excessiveCross.view()}}};
  const auto nonPsd = covariance::uncertaintyContributionBudget(
      jacobian.view(), sources, indefinite, DependencePolicy::exact_joint,
      limits);
  VSS_CHECK(!nonPsd.hasValue());
  VSS_CHECK(nonPsd.status().code() == ErrorCode::NotPositiveSemidefinite);

  const std::array<SourceCrossBlock, 1> undeclaredDependency{
      {{.left = 1U, .right = 2U, .covariance = excessiveCross.view()}}};
  const auto dependent = covariance::uncertaintyContributionBudget(
      jacobian.view(), sources, undeclaredDependency,
      DependencePolicy::declared_independent_blocks, limits);
  VSS_CHECK(!dependent.hasValue());
  VSS_CHECK(dependent.status().code() == ErrorCode::InvalidDomain);
}

void dependentCorrectionBudgetIsOrderInvariant() {
  const std::array<Index, 2> yCoordinates{0U, 1U};
  const std::array<Index, 2> bCoordinates{2U, 3U};
  const Matrix covarianceY = matrix({{4.0, 1.0}, {1.0, 4.0}});
  const Matrix covarianceB = matrix({{2.0, 0.0}, {0.0, 2.0}});
  const Matrix covarianceYB = matrix({{1.0, 1.0}, {0.0, 1.0}});
  const Matrix covarianceBY = matrix({{1.0, 0.0}, {1.0, 1.0}});
  const std::array<SourceBlock, 2> normalSources{
      {{.id = 10U,
        .coordinates = yCoordinates,
        .covariance = covarianceY.view()},
       {.id = 20U,
        .coordinates = bCoordinates,
        .covariance = covarianceB.view()}}};
  const std::array<SourceCrossBlock, 1> normalCross{
      {{.left = 10U, .right = 20U, .covariance = covarianceYB.view()}}};
  const Matrix jacobian =
      matrix({{1.0, 0.0, -1.0, 0.0}, {0.0, 1.0, 0.0, -1.0}});
  const Limits limits{
      .elements = 2048U, .bytes = 2048U * sizeof(Scalar), .evaluations = 0U};
  const auto normal = covariance::uncertaintyContributionBudget(
      jacobian.view(), normalSources, normalCross,
      DependencePolicy::exact_joint, limits);
  VSS_CHECK(normal.hasValue());
  const Matrix expected = matrix({{4.0, 0.0}, {0.0, 4.0}});
  for (Index i = 0U; i < 2U; ++i) {
    for (Index j = 0U; j < 2U; ++j) {
      VSS_CHECK(normal.value().covariance.at(i, j) == expected.at(i, j));
    }
  }

  const std::array<SourceBlock, 2> reversedSources{
      {{.id = 20U,
        .coordinates = bCoordinates,
        .covariance = covarianceB.view()},
       {.id = 10U,
        .coordinates = yCoordinates,
        .covariance = covarianceY.view()}}};
  const std::array<SourceCrossBlock, 1> reversedCross{
      {{.left = 20U, .right = 10U, .covariance = covarianceBY.view()}}};
  const auto reversed = covariance::uncertaintyContributionBudget(
      jacobian.view(), reversedSources, reversedCross,
      DependencePolicy::exact_joint, limits);
  VSS_CHECK(reversed.hasValue());
  for (Index i = 0U; i < 2U; ++i) {
    for (Index j = 0U; j < 2U; ++j) {
      VSS_CHECK(reversed.value().covariance.at(i, j) == expected.at(i, j));
    }
  }
}

void selectedMoments64() {
  constexpr Index d = 64U;
  const std::array<Extent, 1> observationShape{d};
  const std::array<Extent, 2> secondTuple{0U, 0U};
  const std::array<Extent, 4> fourthTuple{0U, 0U, 0U, 0U};
  const auto secondIndices = Array<Index>::copy(
      secondTuple,
      Limits{.elements = 32U, .bytes = 32U * sizeof(Index), .evaluations = 0U});
  const auto fourthIndices = Array<Index>::copy(
      fourthTuple,
      Limits{.elements = 32U, .bytes = 32U * sizeof(Index), .evaluations = 0U});
  const auto shape = Array<Index>::copy(
      observationShape,
      Limits{.elements = d, .bytes = d * sizeof(Index), .evaluations = 0U});
  VSS_CHECK(secondIndices.hasValue() && fourthIndices.hasValue() &&
            shape.hasValue());
  std::array<statistics::MomentTerm, 2> terms{
      {{.order = 2U,
        .layout = statistics::MomentRepresentation::selected,
        .flattened_tuples = secondIndices.value(),
        .values = Vector{1U, 1.0}},
       {.order = 4U,
        .layout = statistics::MomentRepresentation::selected,
        .flattened_tuples = fourthIndices.value(),
        .values = Vector{1U, 3.0}}}};
  const auto ownedTerms = Array<statistics::MomentTerm>::copy(
      terms, Limits{.elements = 4U,
                    .bytes = 4U * sizeof(statistics::MomentTerm),
                    .evaluations = 0U});
  VSS_CHECK(ownedTerms.hasValue());
  const EstimatorDiagnostics diagnostics{
      .estimand = Estimand::observed,
      .weight = WeightRule::unweighted,
      .center = CenterRule::estimated,
      .approximation = Approximation::exact,
      .global = Support{.input = 3U,
                        .valid = 3U,
                        .positive = 3U,
                        .frequency_mass = std::nullopt,
                        .weight_mass = std::nullopt,
                        .normalized_weight_powers = {},
                        .effective_count = std::nullopt,
                        .degrees_of_freedom = std::nullopt,
                        .squared_weight_mass = std::nullopt},
      .coordinate_support = {},
      .tuple_support = {},
      .reciprocal_condition = std::nullopt,
      .numerical_error = std::nullopt,
      .sampling_error = std::nullopt,
      .rank = std::nullopt,
      .algorithm = "selected_fixture",
      .premises = "selected central Gaussian moments",
      .warnings = {}};
  MomentEstimate const estimate{.observation_shape = shape.value(),
                                .terms = ownedTerms.value(),
                                .diagnostics = diagnostics,
                                .population_mean = Vector{d, 0.0}};
  Tensor hessian = tensor(std::array<Extent, 3>{1U, d, d});
  const std::array<Index, 3> hIndex{0U, 0U, 0U};
  hessian.at(hIndex) = 1.0;
  const Vector center{1U, 0.0};
  const Matrix jacobian(1U, d, 0.0);
  auto result = covariance::propagateQuadraticTaylor(
      {.at_center = center.view(),
       .jacobian = jacobian.view(),
       .hessian = hessian},
      estimate,
      Limits{.elements = 131088U,
             .bytes = 131088U * sizeof(Scalar),
             .evaluations = 0U});
  VSS_CHECK(result.hasValue());
  VSS_CHECK(result.value().mean.at(0U) == 0.5);
  VSS_CHECK(result.value().covariance.at(0U, 0U) == 0.5);
}

void gaussianQuadraticHighDimension() {
  constexpr Index d = 64U;
  Matrix inputCovariance(d, d, 0.0);
  Tensor hessian = tensor(std::array<Extent, 3>{1U, d, d});
  for (Index i = 0U; i < d; ++i) {
    inputCovariance.at(i, i) = 1.0;
    const std::array<Index, 3> index{0U, i, i};
    hessian.at(index) = 1.0;
  }
  const Vector center{1U, 0.0};
  const Matrix jacobian(1U, d, 0.0);
  const auto result = covariance::propagateGaussianQuadraticTaylor(
      {.at_center = center.view(),
       .jacobian = jacobian.view(),
       .hessian = hessian},
      inputCovariance.view(),
      Limits{.elements = 131088U,
             .bytes = 131088U * sizeof(Scalar),
             .evaluations = 0U});
  VSS_CHECK(result.hasValue());
  VSS_CHECK(result.value().mean.at(0U) == 32.0);
  VSS_CHECK(result.value().covariance.at(0U, 0U) == 32.0);
}

void gaussianQuadraticSingularJoint() {
  const Vector center{2U, 0.0};
  const Matrix jacobian =
      matrix({{2.0 * std::numbers::sqrt2, 0.0}, {0.0, 0.0}});
  Tensor hessian = tensor(std::array<Extent, 3>{2U, 2U, 2U});
  hessian.at(std::array<Index, 3>{1U, 0U, 0U}) = 4.0;
  const Matrix covariance = matrix({{1.0, 1.0}, {1.0, 1.0}});
  const auto result = covariance::propagateGaussianQuadraticTaylor(
      {.at_center = center.view(),
       .jacobian = jacobian.view(),
       .hessian = hessian},
      covariance.view(),
      Limits{.elements = 1024U,
             .bytes = 1024U * sizeof(Scalar),
             .evaluations = 0U});
  VSS_CHECK(result.hasValue());
  VSS_CHECK(result.value().mean.at(0U) == 0.0);
  VSS_CHECK(result.value().mean.at(1U) == 2.0);
  VSS_CHECK(std::abs(result.value().covariance.at(0U, 0U) - 8.0) < 1.0e-12);
  VSS_CHECK(std::abs(result.value().covariance.at(0U, 1U)) < 1.0e-12);
  VSS_CHECK(std::abs(result.value().covariance.at(1U, 0U)) < 1.0e-12);
  VSS_CHECK(std::abs(result.value().covariance.at(1U, 1U) - 8.0) < 1.0e-12);
}

void gaussianScaledCancellation() {
  const Scalar largest = std::numeric_limits<Scalar>::max();
  const Vector center{1U, 0.0};
  const Matrix jacobian = matrix({{largest, -largest, 2.0}});
  const Tensor hessian = tensor(std::array<Extent, 3>{1U, 3U, 3U});
  const Matrix covariance =
      matrix({{1.0, 1.0, 0.0}, {1.0, 1.0, 0.0}, {0.0, 0.0, 1.0}});
  const auto result = covariance::propagateGaussianQuadraticTaylor(
      {.at_center = center.view(),
       .jacobian = jacobian.view(),
       .hessian = hessian},
      covariance.view(),
      Limits{
          .elements = 256U, .bytes = 256U * sizeof(Scalar), .evaluations = 0U});
  VSS_CHECK(result.hasValue());
  VSS_CHECK(result.value().covariance.at(0U, 0U) == 4.0);
}

void monteCarloPushforward() {
  std::array<Scalar, 2> prescribed{-2.0, 2.0};
  Index sampleIndex = 0U;
  Index samplerCalls = 0U;
  Index mapCalls = 0U;
  const JointSampler law = [&](Rng&) -> Result<Vector> {
    ++samplerCalls;
    return Result<Vector>::success(
        Vector{1U, prescribed.at(sampleIndex++ % prescribed.size())});
  };
  const differentiation::VectorFunction function = [&](ConstVectorView x) {
    ++mapCalls;
    return Result<Vector>::success(
        Vector{1U, (x.at(0U) * x.at(0U)) + x.at(0U)});
  };
  // Keep a deliberately non-sorted probability order to verify column order.
  Vector probability{3U, 0.0};
  probability.at(0U) = 1.0;
  probability.at(1U) = 0.5;
  probability.at(2U) = 0.0;
  const MomentRequest request{.orders = 0b1110U,
                              .representation =
                                  statistics::MomentRepresentation::dense,
                              .selected = {},
                              .contraction = std::nullopt};
  const SimPolicy policy{.draws = 2U,
                         .batches = 2U,
                         .quantiles = probability.view(),
                         .failure = InvalidDraw::reject,
                         .output_central_moments = request};
  Rng rng{71U};
  const auto result =
      covariance::monteCarloPropagate(law, function, policy, rng,
                                      Limits{.elements = 4096U,
                                             .bytes = 4096U * sizeof(Scalar),
                                             .evaluations = 4U});
  VSS_CHECK(result.hasValue());
  VSS_CHECK(samplerCalls == 2U && mapCalls == 2U);
  VSS_CHECK(result.value().valid == 2U && result.value().rejected == 0U);
  VSS_CHECK(result.value().estimate.mean.at(0U) == 4.0);
  VSS_CHECK(result.value().estimate.covariance.at(0U, 0U) == 4.0);
  VSS_CHECK(result.value().quantiles.at(0U, 0U) == 6.0);
  VSS_CHECK(result.value().quantiles.at(0U, 1U) == 4.0);
  VSS_CHECK(result.value().quantiles.at(0U, 2U) == 2.0);
  VSS_CHECK(result.value().monte_carlo_error.at(0U) == 2.0);
  const auto centralMoments = result.value().central_moments;
  VSS_CHECK(centralMoments.has_value());
  if (centralMoments.has_value()) {
    const auto central = centralMoments->terms.view();
    VSS_CHECK(central.size() == 3U);
    VSS_CHECK(central[0U].values.view().at(0U) == 4.0);
    VSS_CHECK(central[1U].values.view().at(0U) == 0.0);
    VSS_CHECK(central[2U].values.view().at(0U) == 16.0);
  }
}

void sigmaPointRules() {
  const Vector center{1U, 2.0};
  const Matrix variance = matrix({{4.0}});
  const differentiation::VectorFunction affine = [](ConstVectorView x) {
    return Result<Vector>::success(Vector{1U, 1.0 + (3.0 * x.at(0U))});
  };
  SolvePolicy solve{};
  solve.rank = RankPolicy::supported_subspace;
  const SigmaPointRule ordinary{
      .alpha = 1.0, .beta = 2.0, .kappa = 0.0, .solve = solve};
  const auto linear = covariance::sigmaPointPropagate(
      center.view(), variance.view(), affine, ordinary,
      Limits{
          .elements = 256U, .bytes = 256U * sizeof(Scalar), .evaluations = 3U});
  VSS_CHECK(linear.hasValue());
  VSS_CHECK(linear.value().mean.at(0U) == 7.0);
  VSS_CHECK(linear.value().covariance.at(0U, 0U) == 36.0);

  const Vector zeroCenter{1U, 0.0};
  const differentiation::VectorFunction square = [](ConstVectorView x) {
    return Result<Vector>::success(Vector{1U, x.at(0U) * x.at(0U)});
  };
  const SigmaPointRule signedRule{
      .alpha = 1.0, .beta = -1.0, .kappa = 0.0, .solve = solve};
  const auto signedResult = covariance::sigmaPointPropagate(
      zeroCenter.view(), matrix({{1.0}}).view(), square, signedRule,
      Limits{
          .elements = 256U, .bytes = 256U * sizeof(Scalar), .evaluations = 3U});
  VSS_CHECK(signedResult.hasValue());
  VSS_CHECK(signedResult.value().covariance.at(0U, 0U) == -1.0);
  VSS_CHECK(signedResult.value().covariance_check ==
            PropagationCovarianceCheck::indefinite_approximation);

  const Vector twoCenter{2U, 0.0};
  const Matrix singular = matrix({{1.0, 1.0}, {1.0, 1.0}});
  const differentiation::VectorFunction sum = [](ConstVectorView x) {
    return Result<Vector>::success(Vector{1U, x.at(0U) + x.at(1U)});
  };
  const auto correlated = covariance::sigmaPointPropagate(
      twoCenter.view(), singular.view(), sum, ordinary,
      Limits{
          .elements = 512U, .bytes = 512U * sizeof(Scalar), .evaluations = 5U});
  VSS_CHECK(correlated.hasValue());
  VSS_CHECK(std::abs(correlated.value().covariance.at(0U, 0U) - 4.0) < 1e-12);
  VSS_CHECK(correlated.value().diagnostics.rank == 1U);
}

void monteCarloConditionalAndHardShape() {
  Index attempt = 0U;
  const JointSampler alternating = [&](Rng&) -> Result<Vector> {
    ++attempt;
    if (attempt % 2U == 1U) {
      return Result<Vector>::failure(ErrorCode::InvalidDomain,
                                     "scripted rejection");
    }
    return Result<Vector>::success(Vector{1U, 2.0});
  };
  const differentiation::VectorFunction identity = [](ConstVectorView x) {
    return Result<Vector>::success(Vector{1U, x.at(0U)});
  };
  const Vector noQuantiles;
  const SimPolicy conditional{.draws = 8U,
                              .batches = 2U,
                              .quantiles = noQuantiles.view(),
                              .failure = InvalidDraw::report_conditional};
  Rng rng{17U};
  const auto retained =
      covariance::monteCarloPropagate(alternating, identity, conditional, rng,
                                      Limits{.elements = 256U,
                                             .bytes = 256U * sizeof(Scalar),
                                             .evaluations = 16U});
  VSS_CHECK(retained.hasValue());
  VSS_CHECK(retained.value().valid == 4U && retained.value().rejected == 4U);
  VSS_CHECK(retained.value().estimate.mean.at(0U) == 2.0);
  VSS_CHECK(retained.value().estimate.covariance.at(0U, 0U) == 0.0);
  VSS_CHECK(retained.value().monte_carlo_error.empty());

  const JointSampler varying = [&](Rng&) {
    return Result<Vector>::success(Vector{1U, 2.0});
  };
  const differentiation::VectorFunction drifting =
      [drawn = 0U](ConstVectorView x) mutable {
        ++drawn;
        return Result<Vector>::success(Vector{drawn == 1U ? 1U : 2U, x.at(0U)});
      };
  const SimPolicy reject{.draws = 2U,
                         .batches = 1U,
                         .quantiles = noQuantiles.view(),
                         .failure = InvalidDraw::report_conditional};
  Rng shapeRng{19U};
  const auto mismatch = covariance::monteCarloPropagate(
      varying, drifting, reject, shapeRng,
      Limits{
          .elements = 256U, .bytes = 256U * sizeof(Scalar), .evaluations = 4U});
  VSS_CHECK(!mismatch.hasValue());
  VSS_CHECK(mismatch.status().code() == ErrorCode::DimensionMismatch);
}

void seededBernoulliMonteCarlo() {
  const JointSampler bernoulli = [](Rng& rng) {
    return Result<Vector>::success(
        Vector{1U, (rng.next() >> 62U) == 0U ? 1.0 : 0.0});
  };
  const differentiation::VectorFunction identity = [](ConstVectorView x) {
    return Result<Vector>::success(Vector{1U, x.at(0U)});
  };
  const Vector noQuantiles;
  const SimPolicy policy{.draws = 1024U,
                         .batches = 32U,
                         .quantiles = noQuantiles.view(),
                         .failure = InvalidDraw::reject};
  for (const std::uint64_t seed : {3U, 29U, 71U}) {
    Rng rng{seed};
    const auto result =
        covariance::monteCarloPropagate(bernoulli, identity, policy, rng,
                                        Limits{.elements = 4096U,
                                               .bytes = 4096U * sizeof(Scalar),
                                               .evaluations = 2048U});
    VSS_CHECK(result.hasValue());
    const Scalar mean = result.value().estimate.mean.at(0U);
    const Scalar empiricalVariance =
        result.value().estimate.covariance.at(0U, 0U);
    VSS_CHECK(std::abs(mean - 0.25) <= 0.11);
    VSS_CHECK(std::abs(empiricalVariance - (mean * (1.0 - mean))) < 1.0e-12);
    VSS_CHECK(result.value().monte_carlo_error.size() == 1U);
    VSS_CHECK(std::isfinite(result.value().monte_carlo_error.at(0U)));
  }
}

void sigmaPointZeroDimensionalInput() {
  const Vector center;
  const Matrix covariance(0U, 0U, 0.0);
  Index calls = 0U;
  const differentiation::VectorFunction constant = [&](ConstVectorView input) {
    ++calls;
    VSS_CHECK(input.empty());
    return Result<Vector>::success(Vector{1U, 7.0});
  };
  const SolvePolicy solve{};
  const SigmaPointRule rule{
      .alpha = 1.0, .beta = 2.0, .kappa = 0.0, .solve = solve};
  const auto result = covariance::sigmaPointPropagate(
      center.view(), covariance.view(), constant, rule,
      Limits{.elements = 8U, .bytes = 8U * sizeof(Scalar), .evaluations = 1U});
  VSS_CHECK(result.hasValue());
  VSS_CHECK(calls == 1U);
  VSS_CHECK(result.value().mean.at(0U) == 7.0);
  VSS_CHECK(result.value().covariance.at(0U, 0U) == 0.0);
}

void monteCarloPreflightsCallbackBudget() {
  Index samplerCalls = 0U;
  const JointSampler law = [&](Rng&) {
    ++samplerCalls;
    return Result<Vector>::success(Vector{1U, 1.0});
  };
  const differentiation::VectorFunction identity = [](ConstVectorView x) {
    return Result<Vector>::success(Vector{1U, x.at(0U)});
  };
  const Vector quantiles;
  const SimPolicy policy{.draws = 4U,
                         .batches = 2U,
                         .quantiles = quantiles.view(),
                         .failure = InvalidDraw::reject};
  Rng rng{21U};
  const auto result = covariance::monteCarloPropagate(
      law, identity, policy, rng,
      Limits{
          .elements = 128U, .bytes = 128U * sizeof(Scalar), .evaluations = 7U});
  VSS_CHECK(!result.hasValue());
  VSS_CHECK(result.status().code() == ErrorCode::ResourceLimit);
  VSS_CHECK(samplerCalls == 0U);
}
} // namespace

int main() {
  quadraticSkewedPopulation();
  quadraticSharedSkewedInput();
  missingMomentAndMissingMeanFailSafely();
  gaussianQuadratic();
  cubicFourthAmplitude();
  signedFourthAmplitudeIsPreserved();
  cubicSymmetricLawExposesOmittedTail();
  cubicDerivativeTensorLayout();
  selectedMomentCoverageIsExact();
  sourceContributionBudget();
  reversedRectangularSourceCrossBlock();
  sourceBudgetRejectsDuplicateAndInvalidJoint();
  dependentCorrectionBudgetIsOrderInvariant();
  selectedMoments64();
  gaussianQuadraticHighDimension();
  gaussianQuadraticSingularJoint();
  gaussianScaledCancellation();
  monteCarloPushforward();
  sigmaPointRules();
  sigmaPointZeroDimensionalInput();
  monteCarloConditionalAndHardShape();
  seededBernoulliMonteCarlo();
  monteCarloPreflightsCallbackBudget();
  return 0;
}
