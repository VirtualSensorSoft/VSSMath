#include <algorithm>
#include <cmath>
#include <detail/SymmetricEigenValidation.hpp>
#include <limits>
#include <numeric>
#include <string>
#include <utility>
#include <vector>
#include <vss/math/Core.hpp>
#include <vss/math/Error.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/MatrixView.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/Tolerance.hpp>
#include <vss/math/Vector.hpp>
#include <vss/math/VectorView.hpp>
#include <vss/math/linear/FactorizationOptions.hpp>
#include <vss/math/linear/SymmetricEigen.hpp>

namespace vss::math::linear {
namespace {

template <class T> Result<T> copyFailure(const Status& status) {
  return Result<T>::failure(status.code(), std::string(status.message()));
}

bool checkedSquare(Extent extent, Extent& count) noexcept {
  if (extent != 0U && extent > std::numeric_limits<Extent>::max() / extent) {
    return false;
  }
  count = extent * extent;
  return true;
}

bool finite(ConstMatrixView matrix) noexcept {
  for (Index row = 0U; row < matrix.rows(); ++row) {
    for (Index column = 0U; column < matrix.columns(); ++column) {
      if (!std::isfinite(matrix.at(row, column))) {
        return false;
      }
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

Scalar maximumAbsolute(ConstMatrixView matrix) noexcept {
  Scalar maximum = 0.0;
  for (Index row = 0U; row < matrix.rows(); ++row) {
    for (Index column = 0U; column < matrix.columns(); ++column) {
      maximum = std::max(maximum, std::abs(matrix.at(row, column)));
    }
  }
  return maximum;
}

Scalar maximumOffDiagonal(const Matrix& matrix) noexcept {
  Scalar maximum = 0.0;
  for (Index row = 0U; row < matrix.rows(); ++row) {
    for (Index column = row + 1U; column < matrix.columns(); ++column) {
      maximum = std::max(maximum, std::abs(matrix.at(row, column)));
    }
  }
  return maximum;
}

void copyCanonicalLowerTriangle(ConstMatrixView input, Matrix& output) {
  for (Index row = 0U; row < input.rows(); ++row) {
    for (Index column = 0U; column <= row; ++column) {
      const Scalar value = input.at(row, column);
      output.at(row, column) = value;
      const Index mirroredRow = column;
      const Index mirroredColumn = row;
      output.at(mirroredRow, mirroredColumn) = value;
    }
  }
}

bool finiteMatrix(const Matrix& matrix) noexcept {
  for (Index row = 0U; row < matrix.rows(); ++row) {
    for (Index column = 0U; column < matrix.columns(); ++column) {
      if (!std::isfinite(matrix.at(row, column))) {
        return false;
      }
    }
  }
  return true;
}

bool rotate(Matrix& matrix, Matrix& eigenvectors, Index p, Index q) {
  const Scalar app = matrix.at(p, p);
  const Scalar aqq = matrix.at(q, q);
  const Scalar apq = matrix.at(p, q);
  const Scalar delta = (aqq / 2.0) - (app / 2.0);
  const Scalar scale = std::max(std::abs(delta), std::abs(apq));
  if (scale == 0.0) {
    return true;
  }

  const Scalar normalizedDelta = delta / scale;
  const Scalar normalizedOffDiagonal = apq / scale;
  const Scalar denominator = std::abs(normalizedDelta) +
                             std::hypot(normalizedDelta, normalizedOffDiagonal);
  Scalar tangent = 0.0;
  if (denominator == 0.0) {
    tangent = std::copysign(1.0, apq);
  } else {
    tangent = std::abs(normalizedOffDiagonal) / denominator;
    tangent = std::copysign(tangent, delta);
    if (std::signbit(apq)) {
      tangent = -tangent;
    }
  }
  const Scalar cosine = 1.0 / std::sqrt(1.0 + (tangent * tangent));
  const Scalar sine = tangent * cosine;

  for (Index row = 0U; row < matrix.rows(); ++row) {
    if (row == p || row == q) {
      continue;
    }
    const Scalar rowP = matrix.at(row, p);
    const Scalar rowQ = matrix.at(row, q);
    const Scalar rotatedP = (cosine * rowP) - (sine * rowQ);
    const Scalar rotatedQ = (sine * rowP) + (cosine * rowQ);
    if (!std::isfinite(rotatedP) || !std::isfinite(rotatedQ)) {
      return false;
    }
    matrix.at(row, p) = rotatedP;
    matrix.at(p, row) = rotatedP;
    matrix.at(row, q) = rotatedQ;
    matrix.at(q, row) = rotatedQ;
  }

  const Scalar newApp = app - (tangent * apq);
  const Scalar newAqq = aqq + (tangent * apq);
  if (!std::isfinite(newApp) || !std::isfinite(newAqq)) {
    return false;
  }
  matrix.at(p, p) = newApp;
  matrix.at(q, q) = newAqq;
  matrix.at(p, q) = 0.0;
  matrix.at(q, p) = 0.0;

  for (Index row = 0U; row < eigenvectors.rows(); ++row) {
    const Scalar vectorP = eigenvectors.at(row, p);
    const Scalar vectorQ = eigenvectors.at(row, q);
    const Scalar rotatedP = (cosine * vectorP) - (sine * vectorQ);
    const Scalar rotatedQ = (sine * vectorP) + (cosine * vectorQ);
    if (!std::isfinite(rotatedP) || !std::isfinite(rotatedQ)) {
      return false;
    }
    eigenvectors.at(row, p) = rotatedP;
    eigenvectors.at(row, q) = rotatedQ;
  }
  return true;
}

void normalizeIsolatedSigns(Vector& eigenvalues, Matrix& eigenvectors,
                            Scalar clusterTolerance) {
  const Extent size = eigenvalues.size();
  for (Index column = 0U; column < size; ++column) {
    const bool sharesPrevious =
        column > 0U &&
        std::abs(eigenvalues.at(column) - eigenvalues.at(column - 1U)) <=
            clusterTolerance;
    const bool sharesNext =
        column + 1U < size &&
        std::abs(eigenvalues.at(column + 1U) - eigenvalues.at(column)) <=
            clusterTolerance;
    if (sharesPrevious || sharesNext) {
      continue;
    }
    Index pivot = 0U;
    Scalar pivotMagnitude = 0.0;
    for (Index row = 0U; row < size; ++row) {
      const Scalar magnitude = std::abs(eigenvectors.at(row, column));
      if (magnitude > pivotMagnitude) {
        pivotMagnitude = magnitude;
        pivot = row;
      }
    }
    if (eigenvectors.at(pivot, column) < 0.0) {
      for (Index row = 0U; row < size; ++row) {
        eigenvectors.at(row, column) = -eigenvectors.at(row, column);
      }
    }
  }
}

} // namespace

SymmetricEigenDecomposition::SymmetricEigenDecomposition(
    Extent size, Extent rank, Scalar rankThreshold, Vector eigenvalues,
    Matrix eigenvectors, PositiveDefiniteness definiteness,
    Extent completedSweeps, Scalar convergenceThreshold,
    Scalar finalMaximumOffDiagonal) noexcept
    : size_(size), rank_(rank), rankThreshold_(rankThreshold),
      eigenvalues_(std::move(eigenvalues)),
      eigenvectors_(std::move(eigenvectors)), definiteness_(definiteness),
      completedSweeps_(completedSweeps),
      convergenceThreshold_(convergenceThreshold),
      finalMaximumOffDiagonal_(finalMaximumOffDiagonal) {}

void SymmetricEigenDecomposition::resetMovedFrom() noexcept {
  size_ = 0U;
  rank_ = 0U;
  rankThreshold_ = 0.0;
  eigenvalues_ = Vector{};
  eigenvectors_ = Matrix{};
  definiteness_ = PositiveDefiniteness::PositiveSemidefinite;
  completedSweeps_ = 0U;
  convergenceThreshold_ = 0.0;
  finalMaximumOffDiagonal_ = 0.0;
}

SymmetricEigenDecomposition::SymmetricEigenDecomposition(
    SymmetricEigenDecomposition&& other) noexcept
    : size_(other.size_), rank_(other.rank_),
      rankThreshold_(other.rankThreshold_),
      eigenvalues_(std::move(other.eigenvalues_)),
      eigenvectors_(std::move(other.eigenvectors_)),
      definiteness_(other.definiteness_),
      completedSweeps_(other.completedSweeps_),
      convergenceThreshold_(other.convergenceThreshold_),
      finalMaximumOffDiagonal_(other.finalMaximumOffDiagonal_) {
  other.resetMovedFrom();
}

SymmetricEigenDecomposition& SymmetricEigenDecomposition::operator=(
    SymmetricEigenDecomposition&& other) noexcept {
  if (this != &other) {
    size_ = other.size_;
    rank_ = other.rank_;
    rankThreshold_ = other.rankThreshold_;
    eigenvalues_ = std::move(other.eigenvalues_);
    eigenvectors_ = std::move(other.eigenvectors_);
    definiteness_ = other.definiteness_;
    completedSweeps_ = other.completedSweeps_;
    convergenceThreshold_ = other.convergenceThreshold_;
    finalMaximumOffDiagonal_ = other.finalMaximumOffDiagonal_;
    other.resetMovedFrom();
  }
  return *this;
}

Extent SymmetricEigenDecomposition::size() const noexcept { return size_; }

Extent SymmetricEigenDecomposition::rank() const noexcept { return rank_; }

Scalar SymmetricEigenDecomposition::rankThreshold() const noexcept {
  return rankThreshold_;
}

ConstVectorView SymmetricEigenDecomposition::eigenvalues() const noexcept {
  return eigenvalues_.view();
}

ConstMatrixView SymmetricEigenDecomposition::eigenvectors() const noexcept {
  return eigenvectors_.view();
}

Scalar SymmetricEigenDecomposition::minimumEigenvalue() const noexcept {
  return size_ == 0U ? 0.0 : eigenvalues_.at(0U);
}

Scalar SymmetricEigenDecomposition::maximumEigenvalue() const noexcept {
  return size_ == 0U ? 0.0 : eigenvalues_.at(size_ - 1U);
}

PositiveDefiniteness
SymmetricEigenDecomposition::definiteness() const noexcept {
  return definiteness_;
}

Extent SymmetricEigenDecomposition::completedSweeps() const noexcept {
  return completedSweeps_;
}

Scalar SymmetricEigenDecomposition::convergenceThreshold() const noexcept {
  return convergenceThreshold_;
}

Scalar SymmetricEigenDecomposition::finalMaximumOffDiagonal() const noexcept {
  return finalMaximumOffDiagonal_;
}

Result<SymmetricEigenDecomposition>
decomposeSymmetricEigen(ConstMatrixView matrix, SymmetricEigenOptions options) {
  const Status optionStatus = detail::validateSymmetricEigenOptions(options);
  if (!optionStatus.ok()) {
    return Result<SymmetricEigenDecomposition>::failure(
        optionStatus.code(), std::string(optionStatus.message()));
  }
  if (matrix.rows() == 0U || matrix.columns() == 0U) {
    return Result<SymmetricEigenDecomposition>::failure(
        ErrorCode::InvalidExtent,
        "symmetric eigendecomposition requires a non-empty matrix");
  }
  if (matrix.rows() != matrix.columns()) {
    return Result<SymmetricEigenDecomposition>::failure(
        ErrorCode::DimensionMismatch,
        "symmetric eigendecomposition requires a square matrix");
  }
  Extent elementCount = 0U;
  if (!checkedSquare(matrix.rows(), elementCount)) {
    return Result<SymmetricEigenDecomposition>::failure(
        ErrorCode::SizeOverflow,
        "symmetric eigendecomposition matrix size overflows");
  }
  (void)elementCount;
  if (!finite(matrix)) {
    return Result<SymmetricEigenDecomposition>::failure(
        ErrorCode::NonFiniteInput,
        "symmetric eigendecomposition input contains a non-finite value");
  }
  for (Index row = 0U; row < matrix.rows(); ++row) {
    for (Index column = row + 1U; column < matrix.columns(); ++column) {
      const Index mirroredRow = column;
      const Index mirroredColumn = row;
      if (!nearlySymmetric(matrix.at(row, column),
                           matrix.at(mirroredRow, mirroredColumn),
                           options.symmetryTolerance)) {
        return Result<SymmetricEigenDecomposition>::failure(
            ErrorCode::NotSymmetric,
            "symmetric eigendecomposition mirrored entries are inconsistent");
      }
    }
  }

  const Extent size = matrix.rows();
  Matrix work(size, size, 0.0);
  Matrix eigenvectors(size, size, 0.0);
  copyCanonicalLowerTriangle(matrix, work);
  for (Index diagonal = 0U; diagonal < size; ++diagonal) {
    eigenvectors.at(diagonal, diagonal) = 1.0;
  }
  const Scalar inputScale = maximumAbsolute(work.view());
  const Scalar scale = std::max(inputScale, std::numeric_limits<Scalar>::min());
  const Scalar threshold = 8.0 * std::numeric_limits<Scalar>::epsilon() * scale;
  if (!std::isfinite(threshold)) {
    return Result<SymmetricEigenDecomposition>::failure(
        ErrorCode::NonFiniteResult,
        "symmetric eigendecomposition convergence threshold is not finite");
  }

  Scalar offDiagonal = maximumOffDiagonal(work);
  Extent completedSweeps = 0U;
  if (offDiagonal > threshold) {
    bool converged = false;
    for (Extent sweep = 0U; sweep < options.maximumSweeps; ++sweep) {
      for (Index p = 0U; p + 1U < size; ++p) {
        for (Index q = p + 1U; q < size; ++q) {
          if (std::abs(work.at(p, q)) > threshold &&
              !rotate(work, eigenvectors, p, q)) {
            return Result<SymmetricEigenDecomposition>::failure(
                ErrorCode::NonFiniteResult,
                "symmetric eigendecomposition rotation produced a non-finite "
                "value");
          }
        }
      }
      ++completedSweeps;
      offDiagonal = maximumOffDiagonal(work);
      if (!std::isfinite(offDiagonal)) {
        return Result<SymmetricEigenDecomposition>::failure(
            ErrorCode::NonFiniteResult,
            "symmetric eigendecomposition off-diagonal diagnostic is not "
            "finite");
      }
      if (offDiagonal <= threshold) {
        converged = true;
        break;
      }
    }
    if (!converged) {
      return Result<SymmetricEigenDecomposition>::failure(
          ErrorCode::NoConvergence,
          "symmetric eigendecomposition reached its sweep limit");
    }
  }

  Vector eigenvalues(size, 0.0);
  for (Index diagonal = 0U; diagonal < size; ++diagonal) {
    eigenvalues.at(diagonal) = work.at(diagonal, diagonal);
  }
  std::vector<Index> order(size);
  std::iota(order.begin(), order.end(), 0U);
  std::ranges::stable_sort(order, [&eigenvalues](Index lhs, Index rhs) {
    return eigenvalues.at(lhs) < eigenvalues.at(rhs);
  });
  Vector sortedValues(size, 0.0);
  Matrix sortedVectors(size, size, 0.0);
  for (Index sorted = 0U; sorted < size; ++sorted) {
    const Index source = order.at(sorted);
    sortedValues.at(sorted) = eigenvalues.at(source);
    for (Index row = 0U; row < size; ++row) {
      sortedVectors.at(row, sorted) = eigenvectors.at(row, source);
    }
  }
  const Scalar spectralScale = std::max(std::abs(sortedValues.at(0U)),
                                        std::abs(sortedValues.at(size - 1U)));
  const Scalar relativeThreshold =
      options.rankTolerance.relative * spectralScale;
  if (!std::isfinite(relativeThreshold)) {
    return Result<SymmetricEigenDecomposition>::failure(
        ErrorCode::NonFiniteResult,
        "symmetric eigendecomposition rank threshold is not finite");
  }
  const Scalar rankThreshold =
      std::max(options.rankTolerance.absolute, relativeThreshold);
  if (!std::isfinite(rankThreshold)) {
    return Result<SymmetricEigenDecomposition>::failure(
        ErrorCode::NonFiniteResult,
        "symmetric eigendecomposition rank threshold is not finite");
  }

  Extent rank = 0U;
  bool positiveDefinite = true;
  bool positiveSemidefinite = true;
  for (Index index = 0U; index < size; ++index) {
    const Scalar value = sortedValues.at(index);
    if (std::abs(value) > rankThreshold) {
      ++rank;
    }
    if (!(value > rankThreshold)) {
      positiveDefinite = false;
    }
    if (value < -rankThreshold) {
      positiveSemidefinite = false;
    }
  }
  PositiveDefiniteness definiteness =
      PositiveDefiniteness::NotPositiveSemidefinite;
  if (positiveDefinite) {
    definiteness = PositiveDefiniteness::PositiveDefinite;
  } else if (positiveSemidefinite) {
    definiteness = PositiveDefiniteness::PositiveSemidefinite;
  }
  normalizeIsolatedSigns(sortedValues, sortedVectors,
                         std::max(threshold, rankThreshold));
  if (!finiteMatrix(sortedVectors)) {
    return Result<SymmetricEigenDecomposition>::failure(
        ErrorCode::NonFiniteResult,
        "symmetric eigendecomposition eigenvectors are not finite");
  }
  return Result<SymmetricEigenDecomposition>::success(
      SymmetricEigenDecomposition(size, rank, rankThreshold,
                                  std::move(sortedValues),
                                  std::move(sortedVectors), definiteness,
                                  completedSweeps, threshold, offDiagonal));
}

} // namespace vss::math::linear
