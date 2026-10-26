#include <algorithm>
#include <cmath>
#include <detail/CompensatedSum.hpp>
#include <detail/ScaledNorm.hpp>
#include <detail/SolveOutcomeAccess.hpp>
#include <detail/SolveValidation.hpp>
#include <detail/SymmetricEigenValidation.hpp>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vss/math/Core.hpp>
#include <vss/math/Error.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/MatrixView.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/Tolerance.hpp>
#include <vss/math/Vector.hpp>
#include <vss/math/VectorView.hpp>
#include <vss/math/covariance/Covariance.hpp>
#include <vss/math/linear/Cholesky.hpp>
#include <vss/math/linear/LinearSolve.hpp>
#include <vss/math/linear/SymmetricEigen.hpp>

namespace vss::math::covariance {
namespace {

template <class T> Result<T> copyFailure(const Status& status) {
  return Result<T>::failure(status.code(), std::string(status.message()));
}

bool checkedProduct(Extent lhs, Extent rhs, Extent& product) noexcept {
  if (lhs != 0U && rhs > std::numeric_limits<Extent>::max() / lhs) {
    return false;
  }
  product = lhs * rhs;
  return true;
}

bool finite(ConstMatrixView matrix) {
  for (Index row = 0U; row < matrix.rows(); ++row) {
    for (Index column = 0U; column < matrix.columns(); ++column) {
      if (!std::isfinite(matrix.at(row, column))) {
        return false;
      }
    }
  }
  return true;
}

bool finite(ConstVectorView vector) {
  for (Index index = 0U; index < vector.size(); ++index) {
    if (!std::isfinite(vector.at(index))) {
      return false;
    }
  }
  return true;
}

bool nearlySymmetric(Scalar lhs, Scalar rhs, Tolerance tolerance) noexcept {
  if (lhs == rhs) {
    return true;
  }
  const Scalar scale = std::max(std::abs(lhs), std::abs(rhs));
  if (scale == 0.0) {
    return true;
  }
  const Scalar normalizedDifference = std::abs((lhs / scale) - (rhs / scale));
  return normalizedDifference <=
         (tolerance.absolute / scale) + tolerance.relative;
}

bool symmetric(ConstMatrixView matrix, Tolerance tolerance) {
  for (Index row = 0U; row < matrix.rows(); ++row) {
    for (Index column = row + 1U; column < matrix.columns(); ++column) {
      const Index first = row;
      const Index second = column;
      if (!nearlySymmetric(matrix.at(first, second), matrix.at(second, first),
                           tolerance)) {
        return false;
      }
    }
  }
  return true;
}

Scalar canonical(ConstMatrixView matrix, Index row, Index column) {
  const Index lower = std::min(row, column);
  const Index upper = std::max(row, column);
  return matrix.at(upper, lower);
}

Status validateEigenOptions(const CovarianceValidationOptions& options) {
  return linear::detail::validateSymmetricEigenOptions(options.eigen);
}

Status validateAnalysisOptions(const CovarianceAnalysisOptions& options) {
  Status eigenStatus = validateEigenOptions(options.validation);
  if (!eigenStatus.ok()) {
    return eigenStatus;
  }
  if (!std::isfinite(options.conditionLimit) || options.conditionLimit <= 1.0) {
    return Status::failure(ErrorCode::InvalidTolerance,
                           "covariance condition limit must be finite and > 1");
  }
  if (options.weakDirections != WeakDirectionOutput::Omit &&
      options.weakDirections != WeakDirectionOutput::Include) {
    return Status::failure(ErrorCode::InvalidDomain,
                           "covariance weak-direction output is invalid");
  }
  return Status::success();
}

Status validateCorrelationOptions(const CorrelationOptions& options) {
  Status eigenStatus = validateEigenOptions(options.validation);
  if (!eigenStatus.ok()) {
    return eigenStatus;
  }
  if (!std::isfinite(options.unitDiagonalTolerance.absolute) ||
      !std::isfinite(options.unitDiagonalTolerance.relative) ||
      options.unitDiagonalTolerance.absolute < 0.0 ||
      options.unitDiagonalTolerance.relative < 0.0) {
    return Status::failure(ErrorCode::InvalidTolerance,
                           "correlation diagonal tolerance is invalid");
  }
  return Status::success();
}

bool isPositiveSemidefinite(
    linear::PositiveDefiniteness definiteness) noexcept {
  return definiteness == linear::PositiveDefiniteness::PositiveDefinite ||
         definiteness == linear::PositiveDefiniteness::PositiveSemidefinite;
}

CovarianceClassification
classify(linear::PositiveDefiniteness definiteness) noexcept {
  switch (definiteness) {
  case linear::PositiveDefiniteness::PositiveDefinite:
    return CovarianceClassification::PositiveDefinite;
  case linear::PositiveDefiniteness::PositiveSemidefinite:
    return CovarianceClassification::PositiveSemidefinite;
  case linear::PositiveDefiniteness::NotPositiveSemidefinite:
    return CovarianceClassification::NotPositiveSemidefinite;
  }
  return CovarianceClassification::NotPositiveSemidefinite;
}

Status validateSquareNonEmpty(ConstMatrixView matrix, const char* operation) {
  if (matrix.rows() == 0U || matrix.columns() == 0U) {
    return Status::failure(ErrorCode::InvalidExtent,
                           std::string(operation) +
                               " requires a non-empty matrix");
  }
  if (matrix.rows() != matrix.columns()) {
    return Status::failure(ErrorCode::DimensionMismatch,
                           std::string(operation) +
                               " requires a square matrix");
  }
  return Status::success();
}

} // namespace

CovarianceDiagnostics::CovarianceDiagnostics(
    CovarianceClassification classification, bool passedSymmetryValidation,
    Extent rank, Scalar minimumEigenvalue, Scalar maximumEigenvalue,
    Scalar rankThreshold, std::optional<Scalar> spectralConditionNumber,
    Matrix weakDirectionEigenvectors) noexcept
    : classification_(classification),
      passedSymmetryValidation_(passedSymmetryValidation), rank_(rank),
      minimumEigenvalue_(minimumEigenvalue),
      maximumEigenvalue_(maximumEigenvalue), rankThreshold_(rankThreshold),
      spectralConditionNumber_(spectralConditionNumber),
      weakDirectionEigenvectors_(std::move(weakDirectionEigenvectors)) {}

void CovarianceDiagnostics::resetMovedFrom() noexcept {
  classification_ = CovarianceClassification::PositiveSemidefinite;
  passedSymmetryValidation_ = false;
  rank_ = 0U;
  minimumEigenvalue_ = 0.0;
  maximumEigenvalue_ = 0.0;
  rankThreshold_ = 0.0;
  spectralConditionNumber_.reset();
  weakDirectionEigenvectors_ = Matrix{};
}

CovarianceDiagnostics::CovarianceDiagnostics(
    CovarianceDiagnostics&& other) noexcept
    : classification_(other.classification_),
      passedSymmetryValidation_(other.passedSymmetryValidation_),
      rank_(other.rank_), minimumEigenvalue_(other.minimumEigenvalue_),
      maximumEigenvalue_(other.maximumEigenvalue_),
      rankThreshold_(other.rankThreshold_),
      spectralConditionNumber_(other.spectralConditionNumber_),
      weakDirectionEigenvectors_(std::move(other.weakDirectionEigenvectors_)) {
  other.resetMovedFrom();
}

CovarianceDiagnostics&
CovarianceDiagnostics::operator=(CovarianceDiagnostics&& other) noexcept {
  if (this != &other) {
    classification_ = other.classification_;
    passedSymmetryValidation_ = other.passedSymmetryValidation_;
    rank_ = other.rank_;
    minimumEigenvalue_ = other.minimumEigenvalue_;
    maximumEigenvalue_ = other.maximumEigenvalue_;
    rankThreshold_ = other.rankThreshold_;
    spectralConditionNumber_ = other.spectralConditionNumber_;
    weakDirectionEigenvectors_ = std::move(other.weakDirectionEigenvectors_);
    other.resetMovedFrom();
  }
  return *this;
}

CorrelationConversion::CorrelationConversion(Matrix correlation,
                                             Vector standardDeviations) noexcept
    : correlation_(std::move(correlation)),
      standardDeviations_(std::move(standardDeviations)) {}

void CorrelationConversion::resetMovedFrom() noexcept {
  correlation_ = Matrix{};
  standardDeviations_ = Vector{};
}

CorrelationConversion::CorrelationConversion(
    CorrelationConversion&& other) noexcept
    : correlation_(std::move(other.correlation_)),
      standardDeviations_(std::move(other.standardDeviations_)) {
  other.resetMovedFrom();
}

CorrelationConversion&
CorrelationConversion::operator=(CorrelationConversion&& other) noexcept {
  if (this != &other) {
    correlation_ = std::move(other.correlation_);
    standardDeviations_ = std::move(other.standardDeviations_);
    other.resetMovedFrom();
  }
  return *this;
}

Result<CovarianceDiagnostics>
analyzeCovariance(ConstMatrixView covariance,
                  CovarianceAnalysisOptions options) {
  const Status optionStatus = validateAnalysisOptions(options);
  if (!optionStatus.ok()) {
    return copyFailure<CovarianceDiagnostics>(optionStatus);
  }
  const Status shapeStatus = validateSquareNonEmpty(covariance, "covariance");
  if (!shapeStatus.ok()) {
    return copyFailure<CovarianceDiagnostics>(shapeStatus);
  }
  Extent elementCount = 0U;
  if (!checkedProduct(covariance.rows(), covariance.columns(), elementCount)) {
    return Result<CovarianceDiagnostics>::failure(
        ErrorCode::SizeOverflow, "covariance matrix size overflows");
  }
  (void)elementCount;
  if (!finite(covariance)) {
    return Result<CovarianceDiagnostics>::failure(
        ErrorCode::NonFiniteInput, "covariance contains a non-finite value");
  }
  if (!symmetric(covariance, options.validation.eigen.symmetryTolerance)) {
    return Result<CovarianceDiagnostics>::failure(
        ErrorCode::NotSymmetric,
        "covariance mirrored entries are inconsistent");
  }

  const auto decomposition =
      linear::decomposeSymmetricEigen(covariance, options.validation.eigen);
  if (!decomposition.hasValue()) {
    return copyFailure<CovarianceDiagnostics>(decomposition.status());
  }
  const auto& eigen = decomposition.value();
  const auto classification = classify(eigen.definiteness());
  std::optional<Scalar> spectralCondition;
  if (eigen.definiteness() == linear::PositiveDefiniteness::PositiveDefinite) {
    const Scalar limit = std::numeric_limits<Scalar>::max();
    const Scalar low = eigen.minimumEigenvalue();
    const Scalar high = eigen.maximumEigenvalue();
    if (low < high / limit) {
      spectralCondition = std::numeric_limits<Scalar>::infinity();
    } else {
      const Scalar condition = high / low;
      if (!std::isfinite(condition)) {
        return Result<CovarianceDiagnostics>::failure(
            ErrorCode::NonFiniteResult,
            "covariance spectral condition is not finite");
      }
      spectralCondition = condition;
    }
  }

  const Scalar weakThreshold =
      std::max(eigen.rankThreshold(),
               eigen.maximumEigenvalue() / options.conditionLimit);
  Extent weakCount = 0U;
  for (Index index = 0U; index < eigen.eigenvalues().size(); ++index) {
    if (eigen.eigenvalues().at(index) <= weakThreshold) {
      ++weakCount;
    }
  }
  Matrix weakDirections;
  if (options.weakDirections == WeakDirectionOutput::Include) {
    weakDirections = Matrix(covariance.rows(), weakCount, 0.0);
    Extent destination = 0U;
    for (Index index = 0U; index < eigen.eigenvalues().size(); ++index) {
      if (eigen.eigenvalues().at(index) <= weakThreshold) {
        for (Index row = 0U; row < covariance.rows(); ++row) {
          weakDirections.at(row, destination) =
              eigen.eigenvectors().at(row, index);
        }
        ++destination;
      }
    }
  }
  return Result<CovarianceDiagnostics>::success(CovarianceDiagnostics(
      classification, true, eigen.rank(), eigen.minimumEigenvalue(),
      eigen.maximumEigenvalue(), eigen.rankThreshold(), spectralCondition,
      std::move(weakDirections)));
}

Result<linear::SolveOutcome<Scalar>>
mahalanobisSquared(const linear::CholeskyFactorization& factorization,
                   ConstVectorView residual, linear::SolveOptions options) {
  const Status optionStatus = linear::detail::validateSolveOptions(options);
  if (!optionStatus.ok()) {
    return copyFailure<linear::SolveOutcome<Scalar>>(optionStatus);
  }
  if (factorization.size() == 0U) {
    return Result<linear::SolveOutcome<Scalar>>::failure(
        ErrorCode::InvalidExtent, "Cholesky factorization is moved-from");
  }
  if (residual.size() != factorization.size()) {
    return Result<linear::SolveOutcome<Scalar>>::failure(
        ErrorCode::DimensionMismatch,
        "Cholesky factorization and residual dimensions differ");
  }
  if (!finite(residual)) {
    return Result<linear::SolveOutcome<Scalar>>::failure(
        ErrorCode::NonFiniteInput,
        "Mahalanobis residual contains a non-finite value");
  }
  const auto whitened = factorization.whiten(residual, options);
  if (!whitened.hasValue()) {
    return copyFailure<linear::SolveOutcome<Scalar>>(whitened.status());
  }
  detail::ScaledNorm norm;
  for (Index index = 0U; index < whitened.value().solution().size(); ++index) {
    norm.add(whitened.value().solution().at(index));
  }
  const Scalar magnitude = norm.value();
  if (!std::isfinite(magnitude)) {
    return Result<linear::SolveOutcome<Scalar>>::failure(
        ErrorCode::NonFiniteResult, "Mahalanobis whitened norm is not finite");
  }
  Scalar squared = 0.0;
  if (magnitude != 0.0) {
    const Scalar maximum = std::numeric_limits<Scalar>::max();
    if (magnitude > maximum / magnitude) {
      return Result<linear::SolveOutcome<Scalar>>::failure(
          ErrorCode::NonFiniteResult,
          "Mahalanobis squared norm exceeds scalar range");
    }
    squared = magnitude * magnitude;
    if (!std::isfinite(squared)) {
      return Result<linear::SolveOutcome<Scalar>>::failure(
          ErrorCode::NonFiniteResult, "Mahalanobis squared norm is not finite");
    }
  }
  return Result<linear::SolveOutcome<Scalar>>::success(
      linear::detail::SolveOutcomeAccess<Scalar>::make(
          squared, whitened.value().conditionReport()));
}

Result<Matrix> propagateCovariance(ConstMatrixView jacobian,
                                   ConstMatrixView covariance,
                                   CovarianceValidationOptions options) {
  const Status optionStatus = validateEigenOptions(options);
  if (!optionStatus.ok()) {
    return copyFailure<Matrix>(optionStatus);
  }
  if (jacobian.rows() == 0U || jacobian.columns() == 0U ||
      covariance.rows() == 0U || covariance.columns() == 0U) {
    return Result<Matrix>::failure(
        ErrorCode::InvalidExtent,
        "covariance propagation requires non-empty inputs");
  }
  if (covariance.rows() != covariance.columns() ||
      jacobian.columns() != covariance.rows()) {
    return Result<Matrix>::failure(
        ErrorCode::DimensionMismatch,
        "covariance propagation dimensions are incompatible");
  }
  Extent temporaryCount = 0U;
  Extent covarianceCount = 0U;
  Extent resultCount = 0U;
  if (!checkedProduct(jacobian.rows(), jacobian.columns(), temporaryCount) ||
      !checkedProduct(covariance.rows(), covariance.columns(),
                      covarianceCount) ||
      !checkedProduct(jacobian.rows(), jacobian.rows(), resultCount)) {
    return Result<Matrix>::failure(ErrorCode::SizeOverflow,
                                   "covariance propagation size overflows");
  }
  (void)temporaryCount;
  (void)covarianceCount;
  (void)resultCount;
  if (!finite(jacobian)) {
    return Result<Matrix>::failure(ErrorCode::NonFiniteInput,
                                   "Jacobian contains a non-finite value");
  }
  if (!finite(covariance)) {
    return Result<Matrix>::failure(ErrorCode::NonFiniteInput,
                                   "covariance contains a non-finite value");
  }
  if (!symmetric(covariance, options.eigen.symmetryTolerance)) {
    return Result<Matrix>::failure(
        ErrorCode::NotSymmetric,
        "covariance mirrored entries are inconsistent");
  }
  const auto decomposition =
      linear::decomposeSymmetricEigen(covariance, options.eigen);
  if (!decomposition.hasValue()) {
    return copyFailure<Matrix>(decomposition.status());
  }
  if (!isPositiveSemidefinite(decomposition.value().definiteness())) {
    return Result<Matrix>::failure(ErrorCode::NotPositiveSemidefinite,
                                   "covariance is not positive semidefinite");
  }

  Matrix temporary(jacobian.rows(), covariance.rows(), 0.0);
  Matrix result(jacobian.rows(), jacobian.rows(), 0.0);
  for (Index row = 0U; row < jacobian.rows(); ++row) {
    for (Index column = 0U; column < covariance.rows(); ++column) {
      detail::CompensatedSum sum;
      for (Index index = 0U; index < jacobian.columns(); ++index) {
        const Scalar product =
            jacobian.at(row, index) * canonical(covariance, index, column);
        if (!std::isfinite(product)) {
          return Result<Matrix>::failure(
              ErrorCode::NonFiniteResult,
              "covariance propagation intermediate is not finite");
        }
        sum.add(product);
        if (!std::isfinite(sum.total())) {
          return Result<Matrix>::failure(
              ErrorCode::NonFiniteResult,
              "covariance propagation accumulator is not finite");
        }
      }
      temporary.at(row, column) = sum.total();
    }
  }
  for (Index row = 0U; row < result.rows(); ++row) {
    for (Index column = 0U; column <= row; ++column) {
      detail::CompensatedSum sum;
      for (Index index = 0U; index < temporary.columns(); ++index) {
        const Index first = column;
        const Index second = index;
        const Scalar product =
            temporary.at(row, index) * jacobian.at(first, second);
        if (!std::isfinite(product)) {
          return Result<Matrix>::failure(
              ErrorCode::NonFiniteResult,
              "covariance propagation output product is not finite");
        }
        sum.add(product);
        if (!std::isfinite(sum.total())) {
          return Result<Matrix>::failure(
              ErrorCode::NonFiniteResult,
              "covariance propagation output accumulator is not finite");
        }
      }
      const Scalar value = sum.total();
      if (!std::isfinite(value)) {
        return Result<Matrix>::failure(
            ErrorCode::NonFiniteResult,
            "covariance propagation output is not finite");
      }
      const Index first = row;
      const Index second = column;
      result.at(first, second) = value;
      result.at(second, first) = value;
    }
  }
  return Result<Matrix>::success(std::move(result));
}

Result<CorrelationConversion>
toCorrelation(ConstMatrixView covariance, CovarianceValidationOptions options) {
  const Status optionStatus = validateEigenOptions(options);
  if (!optionStatus.ok()) {
    return copyFailure<CorrelationConversion>(optionStatus);
  }
  const Status shapeStatus = validateSquareNonEmpty(covariance, "covariance");
  if (!shapeStatus.ok()) {
    return copyFailure<CorrelationConversion>(shapeStatus);
  }
  Extent resultCount = 0U;
  if (!checkedProduct(covariance.rows(), covariance.columns(), resultCount)) {
    return Result<CorrelationConversion>::failure(
        ErrorCode::SizeOverflow, "correlation result size overflows");
  }
  (void)resultCount;
  if (!finite(covariance)) {
    return Result<CorrelationConversion>::failure(
        ErrorCode::NonFiniteInput, "covariance contains a non-finite value");
  }
  if (!symmetric(covariance, options.eigen.symmetryTolerance)) {
    return Result<CorrelationConversion>::failure(
        ErrorCode::NotSymmetric,
        "covariance mirrored entries are inconsistent");
  }
  for (Index index = 0U; index < covariance.rows(); ++index) {
    if (covariance.at(index, index) <= 0.0) {
      return Result<CorrelationConversion>::failure(
          ErrorCode::InvalidDomain, "covariance variances must be positive");
    }
  }
  const auto decomposition =
      linear::decomposeSymmetricEigen(covariance, options.eigen);
  if (!decomposition.hasValue()) {
    return copyFailure<CorrelationConversion>(decomposition.status());
  }
  if (!isPositiveSemidefinite(decomposition.value().definiteness())) {
    return Result<CorrelationConversion>::failure(
        ErrorCode::NotPositiveSemidefinite,
        "covariance is not positive semidefinite");
  }
  Matrix correlation(covariance.rows(), covariance.columns(), 0.0);
  Vector deviations(covariance.rows(), 0.0);
  for (Index index = 0U; index < covariance.rows(); ++index) {
    deviations.at(index) = std::sqrt(covariance.at(index, index));
    if (!std::isfinite(deviations.at(index))) {
      return Result<CorrelationConversion>::failure(
          ErrorCode::NonFiniteResult,
          "correlation standard deviation is not finite");
    }
  }
  for (Index row = 0U; row < covariance.rows(); ++row) {
    for (Index column = 0U; column <= row; ++column) {
      const Scalar normalized = covariance.at(row, column) / deviations.at(row);
      const Scalar value = normalized / deviations.at(column);
      if (!std::isfinite(normalized) || !std::isfinite(value)) {
        return Result<CorrelationConversion>::failure(
            ErrorCode::NonFiniteResult,
            "correlation conversion produced a non-finite value");
      }
      const Index first = row;
      const Index second = column;
      correlation.at(first, second) = value;
      correlation.at(second, first) = value;
    }
  }
  for (Index index = 0U; index < covariance.rows(); ++index) {
    correlation.at(index, index) = 1.0;
  }
  return Result<CorrelationConversion>::success(
      CorrelationConversion(std::move(correlation), std::move(deviations)));
}

Result<Matrix> fromCorrelation(ConstMatrixView correlation,
                               ConstVectorView standardDeviations,
                               CorrelationOptions options) {
  const Status optionStatus = validateCorrelationOptions(options);
  if (!optionStatus.ok()) {
    return copyFailure<Matrix>(optionStatus);
  }
  const Status shapeStatus = validateSquareNonEmpty(correlation, "correlation");
  if (!shapeStatus.ok()) {
    return copyFailure<Matrix>(shapeStatus);
  }
  if (standardDeviations.size() != correlation.rows()) {
    return Result<Matrix>::failure(
        ErrorCode::DimensionMismatch,
        "correlation and standard-deviation dimensions differ");
  }
  Extent resultCount = 0U;
  if (!checkedProduct(correlation.rows(), correlation.columns(), resultCount)) {
    return Result<Matrix>::failure(ErrorCode::SizeOverflow,
                                   "covariance result size overflows");
  }
  (void)resultCount;
  if (!finite(correlation)) {
    return Result<Matrix>::failure(ErrorCode::NonFiniteInput,
                                   "correlation contains a non-finite value");
  }
  if (!finite(standardDeviations)) {
    return Result<Matrix>::failure(
        ErrorCode::NonFiniteInput,
        "standard deviations contain a non-finite value");
  }
  if (!symmetric(correlation, options.validation.eigen.symmetryTolerance)) {
    return Result<Matrix>::failure(
        ErrorCode::NotSymmetric,
        "correlation mirrored entries are inconsistent");
  }
  for (Index index = 0U; index < correlation.rows(); ++index) {
    if (!nearlySymmetric(correlation.at(index, index), 1.0,
                         options.unitDiagonalTolerance)) {
      return Result<Matrix>::failure(
          ErrorCode::InvalidDomain,
          "correlation diagonal entries must equal one");
    }
  }
  for (Index index = 0U; index < standardDeviations.size(); ++index) {
    if (standardDeviations.at(index) <= 0.0) {
      return Result<Matrix>::failure(ErrorCode::InvalidDomain,
                                     "standard deviations must be positive");
    }
  }
  const auto decomposition =
      linear::decomposeSymmetricEigen(correlation, options.validation.eigen);
  if (!decomposition.hasValue()) {
    return copyFailure<Matrix>(decomposition.status());
  }
  if (!isPositiveSemidefinite(decomposition.value().definiteness())) {
    return Result<Matrix>::failure(ErrorCode::NotPositiveSemidefinite,
                                   "correlation is not positive semidefinite");
  }
  Matrix covariance(correlation.rows(), correlation.columns(), 0.0);
  for (Index row = 0U; row < correlation.rows(); ++row) {
    for (Index column = 0U; column <= row; ++column) {
      const Scalar scaled =
          correlation.at(row, column) * standardDeviations.at(row);
      const Scalar value = scaled * standardDeviations.at(column);
      if (!std::isfinite(scaled) || !std::isfinite(value)) {
        return Result<Matrix>::failure(
            ErrorCode::NonFiniteResult,
            "covariance conversion produced a non-finite value");
      }
      const Index first = row;
      const Index second = column;
      covariance.at(first, second) = value;
      covariance.at(second, first) = value;
    }
  }
  return Result<Matrix>::success(std::move(covariance));
}

} // namespace vss::math::covariance
