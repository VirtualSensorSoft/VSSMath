#include "Test.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <numbers>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>
#include <vss/math/Core.hpp>
#include <vss/math/Error.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/MatrixView.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/Vector.hpp>
#include <vss/math/VectorView.hpp>
#include <vss/math/covariance/Covariance.hpp>
#include <vss/math/linear/Cholesky.hpp>
#include <vss/math/linear/FactorizationOptions.hpp>
#include <vss/math/linear/LinearSolve.hpp>
#include <vss/math/linear/SymmetricEigen.hpp>

namespace {

template <typename Enum> Enum invalidEnumValue() {
  volatile const std::uint8_t rawValue = 255U;
  // Deliberately construct an out-of-range value for invalid-enumerator tests.
  // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
  return static_cast<Enum>(rawValue);
}

using vss::math::ConstMatrixView;
using vss::math::ConstVectorView;
using vss::math::ErrorCode;
using vss::math::Extent;
using vss::math::Index;
using vss::math::Matrix;
using vss::math::Scalar;
using vss::math::Vector;
using vss::math::covariance::CorrelationConversion;
using vss::math::covariance::CorrelationOptions;
using vss::math::covariance::CovarianceAnalysisOptions;
using vss::math::covariance::CovarianceClassification;
using vss::math::covariance::CovarianceDiagnostics;
using vss::math::covariance::WeakDirectionOutput;
using vss::math::linear::CholeskyFactorization;
using vss::math::linear::IllConditionedSolvePolicy;
using vss::math::linear::SolveOptions;

Matrix matrix(std::initializer_list<std::initializer_list<Scalar>> rows) {
  const Extent rowCount = rows.size();
  const Extent columnCount =
      rows.begin() == rows.end() ? 0U : rows.begin()->size();
  Matrix result(rowCount, columnCount);
  Extent row = 0U;
  for (const auto& values : rows) {
    VSS_CHECK(values.size() == columnCount);
    Extent column = 0U;
    for (const Scalar value : values) {
      result.at(row, column) = value;
      ++column;
    }
    ++row;
  }
  return result;
}

Vector vector(std::initializer_list<Scalar> values) {
  return Vector(std::vector<Scalar>(values));
}

void checkNear(Scalar actual, Scalar expected, Scalar absolute = 1.0e-12,
               Scalar relative = 1.0e-10) {
  const Scalar scale = std::max({1.0, std::abs(actual), std::abs(expected)});
  VSS_CHECK(std::abs(actual - expected) <= absolute + relative * scale);
}

void checkVectorNear(ConstVectorView actual, ConstVectorView expected,
                     Scalar absolute = 1.0e-12, Scalar relative = 1.0e-10) {
  VSS_CHECK(actual.size() == expected.size());
  for (Index index = 0U; index < actual.size(); ++index) {
    checkNear(actual.at(index), expected.at(index), absolute, relative);
  }
}

void checkVectorNear(const Vector& actual, const Vector& expected,
                     Scalar absolute = 1.0e-12, Scalar relative = 1.0e-10) {
  checkVectorNear(actual.view(), expected.view(), absolute, relative);
}

void checkMatrixNear(ConstMatrixView actual, ConstMatrixView expected,
                     Scalar absolute = 1.0e-12, Scalar relative = 1.0e-10) {
  VSS_CHECK(actual.rows() == expected.rows());
  VSS_CHECK(actual.columns() == expected.columns());
  for (Index row = 0U; row < actual.rows(); ++row) {
    for (Index column = 0U; column < actual.columns(); ++column) {
      checkNear(actual.at(row, column), expected.at(row, column), absolute,
                relative);
    }
  }
}

void checkMatrixNear(const Matrix& actual, const Matrix& expected,
                     Scalar absolute = 1.0e-12, Scalar relative = 1.0e-10) {
  checkMatrixNear(actual.view(), expected.view(), absolute, relative);
}

template <class ResultType>
void checkFailure(const ResultType& result, ErrorCode expected) {
  VSS_CHECK(!result.hasValue());
  VSS_CHECK(result.status().code() == expected);
}

void publicContractAndDefaults() {
  static_assert(!std::is_default_constructible_v<CovarianceDiagnostics>);
  static_assert(std::is_copy_constructible_v<CovarianceDiagnostics>);
  static_assert(std::is_nothrow_move_constructible_v<CovarianceDiagnostics>);
  static_assert(std::is_nothrow_move_assignable_v<CovarianceDiagnostics>);
  static_assert(!std::is_default_constructible_v<CorrelationConversion>);
  static_assert(std::is_copy_constructible_v<CorrelationConversion>);
  static_assert(std::is_nothrow_move_constructible_v<CorrelationConversion>);
  static_assert(std::is_nothrow_move_assignable_v<CorrelationConversion>);

  using AnalyzeSignature = vss::math::Result<CovarianceDiagnostics> (*)(
      ConstMatrixView, CovarianceAnalysisOptions);
  using MahalanobisSignature =
      vss::math::Result<vss::math::linear::SolveOutcome<Scalar>> (*)(
          const CholeskyFactorization&, ConstVectorView, SolveOptions);
  using PropagationSignature = vss::math::Result<Matrix> (*)(
      ConstMatrixView, ConstMatrixView,
      vss::math::covariance::CovarianceValidationOptions);
  using ToCorrelationSignature = vss::math::Result<CorrelationConversion> (*)(
      ConstMatrixView, vss::math::covariance::CovarianceValidationOptions);
  using FromCorrelationSignature = vss::math::Result<Matrix> (*)(
      ConstMatrixView, ConstVectorView, CorrelationOptions);
  static_assert(std::is_same_v<decltype(static_cast<AnalyzeSignature>(
                                   &vss::math::covariance::analyzeCovariance)),
                               AnalyzeSignature>);
  static_assert(std::is_same_v<decltype(static_cast<MahalanobisSignature>(
                                   &vss::math::covariance::mahalanobisSquared)),
                               MahalanobisSignature>);
  static_assert(
      std::is_same_v<decltype(static_cast<PropagationSignature>(
                         &vss::math::covariance::propagateCovariance)),
                     PropagationSignature>);
  static_assert(std::is_same_v<decltype(static_cast<ToCorrelationSignature>(
                                   &vss::math::covariance::toCorrelation)),
                               ToCorrelationSignature>);
  static_assert(std::is_same_v<decltype(static_cast<FromCorrelationSignature>(
                                   &vss::math::covariance::fromCorrelation)),
                               FromCorrelationSignature>);

  const CovarianceAnalysisOptions analysis{};
  VSS_CHECK(analysis.validation.eigen.maximumSweeps == 64U);
  VSS_CHECK(analysis.validation.eigen.rankTolerance.absolute == 0.0);
  VSS_CHECK(analysis.validation.eigen.rankTolerance.relative == 1.0e-12);
  VSS_CHECK(analysis.validation.eigen.symmetryTolerance.absolute == 0.0);
  VSS_CHECK(analysis.validation.eigen.symmetryTolerance.relative == 1.0e-12);
  VSS_CHECK(analysis.conditionLimit == 1.0e10);
  VSS_CHECK(analysis.weakDirections == WeakDirectionOutput::Omit);

  const CorrelationOptions correlation{};
  VSS_CHECK(correlation.unitDiagonalTolerance.absolute == 0.0);
  VSS_CHECK(correlation.unitDiagonalTolerance.relative == 1.0e-12);
  VSS_CHECK(static_cast<unsigned>(WeakDirectionOutput::Omit) == 0U);
  VSS_CHECK(static_cast<unsigned>(WeakDirectionOutput::Include) == 1U);
}

void covarianceToCorrelation() {
  const Matrix input = matrix({{4.0, 2.0}, {2.0, 9.0}});
  const auto result = vss::math::covariance::toCorrelation(input.view());
  VSS_CHECK(result.hasValue());
  checkVectorNear(result.value().standardDeviations(), vector({2.0, 3.0}));
  checkMatrixNear(result.value().correlation(),
                  matrix({{1.0, 1.0 / 3.0}, {1.0 / 3.0, 1.0}}));
  VSS_CHECK(result.value().correlation().at(0U, 0U) == 1.0);
  VSS_CHECK(result.value().correlation().at(1U, 1U) == 1.0);
  VSS_CHECK(result.value().correlation().at(0U, 1U) ==
            result.value().correlation().at(1U, 0U));
  checkMatrixNear(input, matrix({{4.0, 2.0}, {2.0, 9.0}}));
}

void correlationRoundTrip() {
  const Matrix correlation = matrix({{1.0, 1.0 / 3.0}, {1.0 / 3.0, 1.0}});
  const Vector deviations = vector({2.0, 3.0});
  const auto result = vss::math::covariance::fromCorrelation(correlation.view(),
                                                             deviations.view());
  VSS_CHECK(result.hasValue());
  checkMatrixNear(result.value(), matrix({{4.0, 2.0}, {2.0, 9.0}}));
  checkMatrixNear(correlation, matrix({{1.0, 1.0 / 3.0}, {1.0 / 3.0, 1.0}}));
  checkVectorNear(deviations, vector({2.0, 3.0}));
}

void whiteningAndMahalanobis() {
  const auto factorized = vss::math::linear::factorizeCholesky(
      matrix({{2.0, 0.0}, {0.0, 8.0}}).view());
  VSS_CHECK(factorized.hasValue());
  const Vector residual = vector({1.0, 2.0});
  const auto whitened = factorized.value().whiten(residual.view());
  VSS_CHECK(whitened.hasValue());
  checkVectorNear(
      whitened.value().solution(),
      vector({1.0 / std::numbers::sqrt2, 1.0 / std::numbers::sqrt2}));

  const auto result = vss::math::covariance::mahalanobisSquared(
      factorized.value(), residual.view());
  VSS_CHECK(result.hasValue());
  checkNear(result.value().solution(), 1.0);
  const auto& conditionReport = result.value().conditionReport();
  VSS_CHECK(conditionReport.has_value());
  if (!conditionReport.has_value()) {
    return;
  }
  VSS_CHECK(conditionReport->matrixOneNorm ==
            factorized.value().conditionReport().matrixOneNorm);
  VSS_CHECK(conditionReport->estimatedConditionNumber1 ==
            factorized.value().conditionReport().estimatedConditionNumber1);
  VSS_CHECK(conditionReport->conditionLimit ==
            factorized.value().conditionReport().conditionLimit);
  VSS_CHECK(conditionReport->classification ==
            factorized.value().conditionReport().classification);
}

void linearPropagation() {
  const Matrix covariance = matrix({{4.0, 2.0}, {2.0, 9.0}});
  const auto oneRow = vss::math::covariance::propagateCovariance(
      matrix({{1.0, 2.0}}).view(), covariance.view());
  VSS_CHECK(oneRow.hasValue());
  checkMatrixNear(oneRow.value(), matrix({{48.0}}));

  const auto twoRows = vss::math::covariance::propagateCovariance(
      matrix({{1.0, 2.0}, {3.0, -1.0}}).view(), covariance.view());
  VSS_CHECK(twoRows.hasValue());
  checkMatrixNear(twoRows.value(), matrix({{48.0, 4.0}, {4.0, 33.0}}));
  VSS_CHECK(twoRows.value().at(0U, 1U) == twoRows.value().at(1U, 0U));
}

void zeroVariancePsd() {
  const auto diagnostics = vss::math::covariance::analyzeCovariance(
      matrix({{1.0, 0.0}, {0.0, 0.0}}).view());
  VSS_CHECK(diagnostics.hasValue());
  VSS_CHECK(diagnostics.value().classification() ==
            CovarianceClassification::PositiveSemidefinite);
  VSS_CHECK(diagnostics.value().passedSymmetryValidation());
  VSS_CHECK(diagnostics.value().rank() == 1U);
  VSS_CHECK(diagnostics.value().minimumEigenvalue() == 0.0);
  VSS_CHECK(diagnostics.value().maximumEigenvalue() == 1.0);
  VSS_CHECK(!diagnostics.value().spectralConditionNumber().has_value());

  checkFailure(vss::math::covariance::toCorrelation(
                   matrix({{1.0, 0.0}, {0.0, 0.0}}).view()),
               ErrorCode::InvalidDomain);
}

void indefiniteDiagnostics() {
  const auto diagnostics = vss::math::covariance::analyzeCovariance(
      matrix({{1.0, 0.0}, {0.0, -1.0}}).view());
  VSS_CHECK(diagnostics.hasValue());
  VSS_CHECK(diagnostics.value().classification() ==
            CovarianceClassification::NotPositiveSemidefinite);
  VSS_CHECK(diagnostics.value().rank() == 2U);
  VSS_CHECK(diagnostics.value().minimumEigenvalue() == -1.0);
  VSS_CHECK(diagnostics.value().maximumEigenvalue() == 1.0);
  VSS_CHECK(!diagnostics.value().spectralConditionNumber().has_value());
}

void symmetryBoundary() {
  const Matrix accepted = matrix({{4.0, 2.0000000000001}, {2.0, 3.0}});
  const auto within = vss::math::covariance::analyzeCovariance(accepted.view());
  const auto exact = vss::math::covariance::analyzeCovariance(
      matrix({{4.0, 2.0}, {2.0, 3.0}}).view());
  VSS_CHECK(within.hasValue());
  VSS_CHECK(exact.hasValue());
  VSS_CHECK(within.value().minimumEigenvalue() ==
            exact.value().minimumEigenvalue());
  VSS_CHECK(within.value().maximumEigenvalue() ==
            exact.value().maximumEigenvalue());

  const auto outside = vss::math::covariance::analyzeCovariance(
      matrix({{1.0, 2.0}, {0.0, 1.0}}).view());
  checkFailure(outside, ErrorCode::NotSymmetric);
}

void weakDirection() {
  CovarianceAnalysisOptions options;
  options.weakDirections = WeakDirectionOutput::Include;
  const auto result = vss::math::covariance::analyzeCovariance(
      matrix({{1.0, 0.0}, {0.0, 1.0e-13}}).view(), options);
  VSS_CHECK(result.hasValue());
  VSS_CHECK(result.value().classification() ==
            CovarianceClassification::PositiveSemidefinite);
  VSS_CHECK(result.value().rank() == 1U);
  VSS_CHECK(result.value().minimumEigenvalue() == 1.0e-13);
  VSS_CHECK(result.value().maximumEigenvalue() == 1.0);
  checkNear(result.value().rankThreshold(), 1.0e-12);
  checkNear(result.value().weakDirectionEigenvectors().at(0U, 0U), 0.0);
  checkNear(result.value().weakDirectionEigenvectors().at(1U, 0U), 1.0);
  VSS_CHECK(result.value().weakDirectionEigenvectors().rows() == 2U);
  VSS_CHECK(result.value().weakDirectionEigenvectors().columns() == 1U);
}

void weakDirectionShapesAndConditioning() {
  const Matrix positive = matrix({{2.0, 0.0}, {0.0, 8.0}});
  const auto omitted =
      vss::math::covariance::analyzeCovariance(positive.view());
  VSS_CHECK(omitted.hasValue());
  VSS_CHECK(omitted.value().weakDirectionEigenvectors().rows() == 0U);
  VSS_CHECK(omitted.value().weakDirectionEigenvectors().columns() == 0U);

  CovarianceAnalysisOptions includeOptions;
  includeOptions.weakDirections = WeakDirectionOutput::Include;
  const auto included =
      vss::math::covariance::analyzeCovariance(positive.view(), includeOptions);
  VSS_CHECK(included.hasValue());
  VSS_CHECK(included.value().weakDirectionEigenvectors().rows() == 2U);
  VSS_CHECK(included.value().weakDirectionEigenvectors().columns() == 0U);
  VSS_CHECK(included.value().rank() == 2U);
  VSS_CHECK(included.value().minimumEigenvalue() == 2.0);
  VSS_CHECK(included.value().maximumEigenvalue() == 8.0);
  checkNear(included.value().rankThreshold(), 8.0e-12);
  const auto& spectralCondition = included.value().spectralConditionNumber();
  VSS_CHECK(spectralCondition.has_value());
  if (!spectralCondition.has_value()) {
    return;
  }
  checkNear(spectralCondition.value(), 4.0);

  const auto weak = vss::math::covariance::analyzeCovariance(
      matrix({{1.0, 0.0}, {0.0, 1.0e-13}}).view(), includeOptions);
  VSS_CHECK(weak.hasValue());
  CovarianceDiagnostics copy = weak.value();
  VSS_CHECK(copy.weakDirectionEigenvectors().data() !=
            weak.value().weakDirectionEigenvectors().data());
  VSS_CHECK(copy.weakDirectionEigenvectors().at(1U, 0U) ==
            weak.value().weakDirectionEigenvectors().at(1U, 0U));
  const CovarianceDiagnostics moved = std::move(copy);
  VSS_CHECK(moved.rank() == 1U);

  const auto conversion = vss::math::covariance::toCorrelation(positive.view());
  VSS_CHECK(conversion.hasValue());
  CorrelationConversion conversionCopy = conversion.value();
  VSS_CHECK(conversionCopy.correlation().data() !=
            conversion.value().correlation().data());
  VSS_CHECK(conversionCopy.standardDeviations().data() !=
            conversion.value().standardDeviations().data());
  const CorrelationConversion conversionMoved = std::move(conversionCopy);
  VSS_CHECK(conversionMoved.correlation().rows() == 2U);
  // The source is intentionally queried after move to verify its reset state.
  // NOLINTNEXTLINE(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
  VSS_CHECK(conversionCopy.correlation().rows() == 0U);
  // NOLINTNEXTLINE(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
  VSS_CHECK(conversionCopy.correlation().columns() == 0U);
  // NOLINTNEXTLINE(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
  VSS_CHECK(conversionCopy.standardDeviations().empty());
}

void spectralConditionOverflow() {
  CovarianceAnalysisOptions options;
  options.validation.eigen.rankTolerance.absolute = 0.0;
  options.validation.eigen.rankTolerance.relative = 0.0;
  const Scalar tiny = std::numeric_limits<Scalar>::denorm_min();
  const auto result = vss::math::covariance::analyzeCovariance(
      matrix({{tiny, 0.0}, {0.0, std::numeric_limits<Scalar>::max()}}).view(),
      options);
  VSS_CHECK(result.hasValue());
  VSS_CHECK(result.value().classification() ==
            CovarianceClassification::PositiveDefinite);
  const auto& spectralCondition = result.value().spectralConditionNumber();
  VSS_CHECK(spectralCondition.has_value());
  if (!spectralCondition.has_value()) {
    return;
  }
  VSS_CHECK(spectralCondition.value() > 0.0);
  VSS_CHECK(std::isinf(spectralCondition.value()));
  VSS_CHECK(std::isfinite(result.value().minimumEigenvalue()));
  VSS_CHECK(std::isfinite(result.value().maximumEigenvalue()));
  VSS_CHECK(std::isfinite(result.value().rankThreshold()));
}

void invalidCovarianceOptions() {
  CovarianceAnalysisOptions analysis;
  analysis.validation.eigen.maximumSweeps = 0U;
  checkFailure(
      vss::math::covariance::analyzeCovariance(Matrix{}.view(), analysis),
      ErrorCode::InvalidDomain);

  analysis = {};
  analysis.validation.eigen.rankTolerance.absolute =
      std::numeric_limits<Scalar>::quiet_NaN();
  checkFailure(vss::math::covariance::analyzeCovariance(
                   matrix({{std::numeric_limits<Scalar>::quiet_NaN()}}).view(),
                   analysis),
               ErrorCode::InvalidTolerance);

  analysis = {};
  analysis.validation.eigen.symmetryTolerance.relative =
      std::numeric_limits<Scalar>::infinity();
  checkFailure(
      vss::math::covariance::analyzeCovariance(Matrix{}.view(), analysis),
      ErrorCode::InvalidTolerance);

  analysis = {};
  analysis.conditionLimit = 1.0;
  checkFailure(
      vss::math::covariance::analyzeCovariance(Matrix{}.view(), analysis),
      ErrorCode::InvalidTolerance);

  analysis = {};
  analysis.weakDirections = invalidEnumValue<WeakDirectionOutput>();
  checkFailure(
      vss::math::covariance::analyzeCovariance(Matrix{}.view(), analysis),
      ErrorCode::InvalidDomain);

  CorrelationOptions correlation;
  correlation.unitDiagonalTolerance.absolute = -1.0;
  checkFailure(vss::math::covariance::fromCorrelation(
                   Matrix{}.view(), Vector{}.view(), correlation),
               ErrorCode::InvalidTolerance);

  SolveOptions solve;
  solve.illConditionedPolicy = invalidEnumValue<IllConditionedSolvePolicy>();
  const auto factorized =
      vss::math::linear::factorizeCholesky(matrix({{1.0}}).view());
  VSS_CHECK(factorized.hasValue());
  checkFailure(vss::math::covariance::mahalanobisSquared(
                   factorized.value(), Vector{}.view(), solve),
               ErrorCode::InvalidDomain);
}

void validationAndErrorPrecedence() {
  checkFailure(vss::math::covariance::analyzeCovariance(Matrix{}.view()),
               ErrorCode::InvalidExtent);
  checkFailure(vss::math::covariance::propagateCovariance(
                   matrix({{1.0}}).view(), Matrix{}.view()),
               ErrorCode::InvalidExtent);
  checkFailure(vss::math::covariance::fromCorrelation(
                   matrix({{1.0, 0.0}, {0.0, 1.0}}).view(), Vector{}.view()),
               ErrorCode::DimensionMismatch);

  const Scalar nan = std::numeric_limits<Scalar>::quiet_NaN();
  checkFailure(
      vss::math::covariance::propagateCovariance(
          matrix({{nan, 0.0}}).view(), matrix({{nan, 0.0}, {0.0, 1.0}}).view()),
      ErrorCode::NonFiniteInput);
  checkFailure(
      vss::math::covariance::fromCorrelation(
          matrix({{1.0, nan}, {nan, 1.0}}).view(), vector({nan, nan}).view()),
      ErrorCode::NonFiniteInput);

  checkFailure(vss::math::covariance::toCorrelation(
                   matrix({{0.0, 1.0}, {1.0, 0.0}}).view()),
               ErrorCode::InvalidDomain);
  checkFailure(
      vss::math::covariance::fromCorrelation(
          matrix({{0.0, 2.0}, {2.0, 0.0}}).view(), vector({1.0, 1.0}).view()),
      ErrorCode::InvalidDomain);
  checkFailure(vss::math::covariance::toCorrelation(
                   matrix({{1.0, 2.0}, {2.0, 1.0}}).view()),
               ErrorCode::NotPositiveSemidefinite);
  checkFailure(vss::math::covariance::propagateCovariance(
                   matrix({{1.0, 0.0}, {0.0, 1.0}}).view(),
                   matrix({{1.0, 2.0}, {2.0, 1.0}}).view()),
               ErrorCode::NotPositiveSemidefinite);
  checkFailure(
      vss::math::covariance::fromCorrelation(
          matrix({{1.0, 2.0}, {2.0, 1.0}}).view(), vector({1.0, 1.0}).view()),
      ErrorCode::NotPositiveSemidefinite);

  CorrelationOptions tolerance;
  tolerance.unitDiagonalTolerance = {0.0, 1.0e-12};
  const auto accepted = vss::math::covariance::fromCorrelation(
      matrix({{1.0 + 5.0e-13, 0.0}, {0.0, 1.0}}).view(),
      vector({1.0, 1.0}).view(), tolerance);
  VSS_CHECK(accepted.hasValue());
  checkFailure(vss::math::covariance::fromCorrelation(
                   matrix({{1.0 + 2.0e-12, 0.0}, {0.0, 1.0}}).view(),
                   vector({1.0, 1.0}).view(), tolerance),
               ErrorCode::InvalidDomain);

  const auto illConditioned = vss::math::linear::factorizeCholesky(
      matrix({{1.0, 0.0}, {0.0, 1.0e-11}}).view());
  VSS_CHECK(illConditioned.hasValue());
  checkFailure(vss::math::covariance::mahalanobisSquared(
                   illConditioned.value(), vector({nan, 1.0}).view()),
               ErrorCode::NonFiniteInput);
  checkFailure(vss::math::covariance::mahalanobisSquared(
                   illConditioned.value(), vector({1.0, 1.0}).view()),
               ErrorCode::IllConditioned);
}

void nonFiniteResultBoundaries() {
  const auto identity =
      vss::math::linear::factorizeCholesky(matrix({{1.0}}).view());
  VSS_CHECK(identity.hasValue());
  const Scalar max = std::numeric_limits<Scalar>::max();
  checkFailure(vss::math::covariance::mahalanobisSquared(
                   identity.value(), vector({max / 2.0}).view()),
               ErrorCode::NonFiniteResult);
  checkFailure(vss::math::covariance::propagateCovariance(
                   matrix({{max}}).view(), matrix({{1.0}}).view()),
               ErrorCode::NonFiniteResult);
  checkFailure(vss::math::covariance::fromCorrelation(matrix({{1.0}}).view(),
                                                      vector({max}).view()),
               ErrorCode::NonFiniteResult);

  const Scalar safeLarge = std::sqrt(max) / 2.0;
  const auto safeMahalanobis = vss::math::covariance::mahalanobisSquared(
      identity.value(), vector({safeLarge}).view());
  VSS_CHECK(safeMahalanobis.hasValue());
  checkNear(safeMahalanobis.value().solution(), safeLarge * safeLarge);
  const auto safePropagation = vss::math::covariance::propagateCovariance(
      matrix({{safeLarge}}).view(), matrix({{1.0}}).view());
  VSS_CHECK(safePropagation.hasValue());
  checkNear(safePropagation.value().at(0U, 0U), safeLarge * safeLarge);
  const auto safeConversion = vss::math::covariance::fromCorrelation(
      matrix({{1.0}}).view(), vector({safeLarge}).view());
  VSS_CHECK(safeConversion.hasValue());
  checkNear(safeConversion.value().at(0U, 0U), safeLarge * safeLarge);

  const Scalar tiny = std::numeric_limits<Scalar>::denorm_min();
  const auto tinyPropagation = vss::math::covariance::propagateCovariance(
      matrix({{1.0}}).view(), matrix({{tiny}}).view());
  VSS_CHECK(tinyPropagation.hasValue());
  VSS_CHECK(tinyPropagation.value().at(0U, 0U) == tiny);
  const auto tinyCorrelation =
      vss::math::covariance::toCorrelation(matrix({{tiny}}).view());
  VSS_CHECK(tinyCorrelation.hasValue());
  VSS_CHECK(tinyCorrelation.value().correlation().at(0U, 0U) == 1.0);
  VSS_CHECK(tinyCorrelation.value().standardDeviations().at(0U) ==
            std::sqrt(tiny));
}

void stridedViewsAndInputPreservation() {
  std::vector<Scalar> covarianceStorage{4.0, 99.0, 2.0, 99.0,
                                        2.0, 99.0, 9.0, 99.0};
  std::vector<Scalar> correlationStorage{1.0,       99.0, 1.0 / 3.0, 99.0,
                                         1.0 / 3.0, 99.0, 1.0,       99.0};
  std::vector<Scalar> jacobianStorage{1.0, 2.0, 99.0};
  std::vector<Scalar> residualStorage{1.0, 99.0, 2.0};
  std::vector<Scalar> deviationStorage{2.0, 99.0, 3.0};
  const auto covariance = vss::math::MatrixView::create(
      std::span<Scalar>(covarianceStorage), 2U, 2U, 4U, 2U);
  const auto correlation = vss::math::MatrixView::create(
      std::span<Scalar>(correlationStorage), 2U, 2U, 4U, 2U);
  const auto jacobian = vss::math::MatrixView::create(
      std::span<Scalar>(jacobianStorage), 1U, 2U, 3U, 1U);
  const auto residual =
      vss::math::VectorView::create(std::span<Scalar>(residualStorage), 2U, 2U);
  const auto deviations = vss::math::VectorView::create(
      std::span<Scalar>(deviationStorage), 2U, 2U);
  VSS_CHECK(covariance.hasValue());
  VSS_CHECK(correlation.hasValue());
  VSS_CHECK(jacobian.hasValue());
  VSS_CHECK(residual.hasValue());
  VSS_CHECK(deviations.hasValue());

  const auto diagnostics =
      vss::math::covariance::analyzeCovariance(covariance.value());
  VSS_CHECK(diagnostics.hasValue());
  const auto propagated = vss::math::covariance::propagateCovariance(
      jacobian.value(), covariance.value());
  VSS_CHECK(propagated.hasValue());
  checkMatrixNear(propagated.value(), matrix({{48.0}}));
  const auto converted =
      vss::math::covariance::toCorrelation(covariance.value());
  VSS_CHECK(converted.hasValue());
  const auto reconstructed = vss::math::covariance::fromCorrelation(
      correlation.value(), deviations.value());
  VSS_CHECK(reconstructed.hasValue());
  const auto factorized =
      vss::math::linear::factorizeCholesky(covariance.value());
  VSS_CHECK(factorized.hasValue());
  const auto mahalanobis = vss::math::covariance::mahalanobisSquared(
      factorized.value(), residual.value());
  VSS_CHECK(mahalanobis.hasValue());

  const std::vector<Scalar> covarianceExpected{4.0, 99.0, 2.0, 99.0,
                                               2.0, 99.0, 9.0, 99.0};
  const std::vector<Scalar> correlationExpected{
      1.0, 99.0, 1.0 / 3.0, 99.0, 1.0 / 3.0, 99.0, 1.0, 99.0};
  VSS_CHECK(covarianceStorage == covarianceExpected);
  VSS_CHECK(correlationStorage == correlationExpected);
  VSS_CHECK(jacobianStorage == std::vector<Scalar>({1.0, 2.0, 99.0}));
  VSS_CHECK(residualStorage == std::vector<Scalar>({1.0, 99.0, 2.0}));
  VSS_CHECK(deviationStorage == std::vector<Scalar>({2.0, 99.0, 3.0}));
}

} // namespace

int main() { // NOLINT(bugprone-exception-escape)
  publicContractAndDefaults();
  covarianceToCorrelation();
  correlationRoundTrip();
  whiteningAndMahalanobis();
  linearPropagation();
  zeroVariancePsd();
  indefiniteDiagnostics();
  symmetryBoundary();
  weakDirection();
  weakDirectionShapesAndConditioning();
  spectralConditionOverflow();
  invalidCovarianceOptions();
  validationAndErrorPrecedence();
  nonFiniteResultBoundaries();
  stridedViewsAndInputPreservation();
  return 0;
}

// VSS-MATH-TRACEABILITY: COV-01 COV-02 COV-03 COV-04 COV-05 COV-06 COV-07 COV-08
