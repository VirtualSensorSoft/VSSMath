#include "Test.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>
#include <vss/math/Core.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/Operations.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/Vector.hpp>
#include <vss/math/covariance/Covariance.hpp>
#include <vss/math/differentiation/FiniteDifference.hpp>
#include <vss/math/linear/Cholesky.hpp>
#include <vss/math/linear/LU.hpp>
#include <vss/math/linear/QR.hpp>
#include <vss/math/linear/SymmetricEigen.hpp>
#include <vss/math/linear/TriangularSolve.hpp>
#include <vss/math/statistics/Descriptive.hpp>

namespace {

using vss::math::Matrix;
using vss::math::Scalar;
using vss::math::Vector;

constexpr Scalar kAbsolute = 1.0e-10;
constexpr Scalar kRelative = 1.0e-9;
constexpr std::uint64_t kSeed = 0x5653534d415448ULL;

std::mt19937_64 makeGenerator() {
  // NOLINTNEXTLINE(cert-msc32-c,cert-msc51-cpp): normative fixed seed
  return std::mt19937_64{kSeed};
}

Scalar unit(std::uint64_t raw) {
  return static_cast<Scalar>(raw >> 11U) * 0x1.0p-53;
}

Scalar signedUnit(std::uint64_t raw) { return (2.0 * unit(raw)) - 1.0; }

bool close(Scalar lhs, Scalar rhs) {
  return std::isfinite(lhs) && std::isfinite(rhs) &&
         std::abs(lhs - rhs) <=
             kAbsolute + (kRelative * std::max(std::abs(lhs), std::abs(rhs)));
}

void checkMatrixClose(const Matrix& lhs, const Matrix& rhs) {
  VSS_CHECK(lhs.rows() == rhs.rows());
  VSS_CHECK(lhs.columns() == rhs.columns());
  for (std::size_t row = 0U; row < lhs.rows(); ++row) {
    for (std::size_t column = 0U; column < lhs.columns(); ++column) {
      VSS_CHECK(close(lhs.at(row, column), rhs.at(row, column)));
    }
  }
}

void checkVectorClose(const Vector& lhs, const Vector& rhs) {
  VSS_CHECK(lhs.size() == rhs.size());
  for (std::size_t index = 0U; index < lhs.size(); ++index) {
    VSS_CHECK(close(lhs.at(index), rhs.at(index)));
  }
}

Vector copyView(vss::math::ConstVectorView view) {
  Vector result(view.size());
  for (std::size_t index = 0U; index < view.size(); ++index) {
    result.at(index) = view.at(index);
  }
  return result;
}

Matrix transpose(const Matrix& input) {
  Matrix result(input.columns(), input.rows());
  for (std::size_t row = 0U; row < input.rows(); ++row) {
    for (std::size_t column = 0U; column < input.columns(); ++column) {
      result.at(column, row) = // NOLINT(readability-suspicious-call-argument):
                               // transpose intentionally reverses coordinates.
          input.at(row, column);
    }
  }
  return result;
}

Matrix product(const Matrix& lhs, const Matrix& rhs) {
  const auto result = vss::math::multiply(lhs.view(), rhs.view());
  VSS_CHECK(result.hasValue());
  return result.value();
}

Vector product(const Matrix& matrix, const Vector& vector) {
  const auto result = vss::math::multiply(matrix.view(), vector.view());
  VSS_CHECK(result.hasValue());
  return result.value();
}

Matrix lowerFactor(std::size_t n, std::mt19937_64& generator) {
  Matrix factor(n, n);
  for (std::size_t row = 0U; row < n; ++row) {
    factor.at(row, row) =
        1.0 + (static_cast<Scalar>(row + 1U) / (2.0 * static_cast<Scalar>(n)));
    for (std::size_t column = 0U; column < row; ++column) {
      factor.at(row, column) =
          signedUnit(generator()) / (32.0 * static_cast<Scalar>(n));
    }
  }
  return factor;
}

void triangularProperties() {
  auto generator = makeGenerator();
  for (std::size_t caseIndex = 0U; caseIndex < 32U; ++caseIndex) {
    const std::size_t n = 2U + (caseIndex % 7U);
    Matrix factor(n, n);
    for (std::size_t row = 0U; row < n; ++row) {
      factor.at(row, row) =
          2.0 + (static_cast<Scalar>(row + 1U) / static_cast<Scalar>(n));
      for (std::size_t column = 0U; column < row; ++column) {
        factor.at(row, column) =
            signedUnit(generator()) / (16.0 * static_cast<Scalar>(n));
      }
    }
    Vector expected(n);
    for (std::size_t index = 0U; index < n; ++index) {
      expected.at(index) = signedUnit(generator());
    }
    const Vector rightHandSide = product(factor, expected);
    const auto result = vss::math::linear::solveTriangular(
        factor.view(), rightHandSide.view(),
        vss::math::linear::TriangularSolveOptions{
            vss::math::linear::Triangle::Lower});
    VSS_CHECK(result.hasValue());
    checkVectorClose(result.value().solution(), expected);
  }
}

void luProperties() {
  auto generator = makeGenerator();
  for (std::size_t caseIndex = 0U; caseIndex < 32U; ++caseIndex) {
    const std::size_t n = 2U + (caseIndex % 7U);
    Matrix matrix(n, n);
    for (std::size_t row = 0U; row < n; ++row) {
      Scalar diagonal =
          2.0 + (static_cast<Scalar>(row + 1U) / static_cast<Scalar>(n));
      for (std::size_t column = 0U; column < n; ++column) {
        if (row != column) {
          matrix.at(row, column) =
              signedUnit(generator()) / (32.0 * static_cast<Scalar>(n));
          diagonal += std::abs(matrix.at(row, column));
        }
      }
      matrix.at(row, row) = diagonal;
    }
    Vector expected(n);
    for (std::size_t index = 0U; index < n; ++index) {
      expected.at(index) = signedUnit(generator());
    }
    const Vector rightHandSide = product(matrix, expected);
    const auto factorization = vss::math::linear::factorizeLU(matrix.view());
    VSS_CHECK(factorization.hasValue());
    const auto result = factorization.value().solve(rightHandSide.view());
    VSS_CHECK(result.hasValue());
    checkVectorClose(result.value().solution(), expected);
  }
}

void qrProperties() {
  auto generator = makeGenerator();
  for (std::size_t caseIndex = 0U; caseIndex < 32U; ++caseIndex) {
    const std::size_t columns = 2U + (caseIndex % 7U);
    const std::size_t rows = columns + 2U;
    Matrix matrix(rows, columns);
    for (std::size_t row = 0U; row < rows; ++row) {
      for (std::size_t column = 0U; column < columns; ++column) {
        if (row == column) {
          matrix.at(row, column) = 1.0 + (static_cast<Scalar>(column + 1U) /
                                          (2.0 * static_cast<Scalar>(columns)));
        } else {
          matrix.at(row, column) =
              signedUnit(generator()) / (32.0 * static_cast<Scalar>(columns));
        }
      }
    }
    Vector expected(columns);
    for (std::size_t index = 0U; index < columns; ++index) {
      expected.at(index) = signedUnit(generator());
    }
    const Vector rightHandSide = product(matrix, expected);
    const auto factorization = vss::math::linear::factorizeQR(matrix.view());
    VSS_CHECK(factorization.hasValue());
    VSS_CHECK(factorization.value().hasFullColumnRank());
    const auto result =
        factorization.value().leastSquares(rightHandSide.view());
    VSS_CHECK(result.hasValue());
    checkVectorClose(result.value().solution(), expected);
  }
}

void choleskyProperties() {
  auto generator = makeGenerator();
  for (std::size_t caseIndex = 0U; caseIndex < 32U; ++caseIndex) {
    const std::size_t n = 2U + (caseIndex % 7U);
    const Matrix factor = lowerFactor(n, generator);
    const Matrix covariance = product(factor, transpose(factor));
    const auto decomposition =
        vss::math::linear::factorizeCholesky(covariance.view());
    VSS_CHECK(decomposition.hasValue());
    checkMatrixClose(decomposition.value().lowerFactor(), factor);
  }
}

void eigenProperties() {
  auto generator = makeGenerator();
  for (std::size_t caseIndex = 0U; caseIndex < 32U; ++caseIndex) {
    const std::size_t n = 2U + (caseIndex % 7U);
    Matrix orthogonal(n, n, 0.0);
    for (std::size_t index = 0U; index < n; ++index) {
      orthogonal.at(index, index) = 1.0;
    }
    for (std::size_t column = 0U; column + 1U < n; ++column) {
      const Scalar theta = 0.125 * signedUnit(generator());
      const Scalar cosine = std::cos(theta);
      const Scalar sine = std::sin(theta);
      for (std::size_t row = 0U; row < n; ++row) {
        const Scalar oldColumn = orthogonal.at(row, column);
        const Scalar oldNext = orthogonal.at(row, column + 1U);
        orthogonal.at(row, column) = (cosine * oldColumn) - (sine * oldNext);
        orthogonal.at(row, column + 1U) =
            (sine * oldColumn) + (cosine * oldNext);
      }
    }
    Matrix diagonal(n, n, 0.0);
    for (std::size_t index = 0U; index < n; ++index) {
      diagonal.at(index, index) = 1.0 + (2.0 * static_cast<Scalar>(index + 1U) /
                                         static_cast<Scalar>(n));
    }
    const Matrix input =
        product(product(orthogonal, diagonal), transpose(orthogonal));
    const auto result =
        vss::math::linear::decomposeSymmetricEigen(input.view());
    VSS_CHECK(result.hasValue());
    const auto& values = result.value().eigenvalues();
    for (std::size_t index = 1U; index < values.size(); ++index) {
      VSS_CHECK(values.at(index - 1U) <= values.at(index));
    }
  }
}

void covarianceProperties() {
  auto generator = makeGenerator();
  for (std::size_t caseIndex = 0U; caseIndex < 32U; ++caseIndex) {
    const std::size_t n = 2U + (caseIndex % 7U);
    const Matrix factor = lowerFactor(n, generator);
    const Matrix covariance = product(factor, transpose(factor));
    const std::size_t outputRows = n + 1U;
    Matrix jacobian(outputRows, n);
    for (std::size_t row = 0U; row < outputRows; ++row) {
      for (std::size_t column = 0U; column < n; ++column) {
        jacobian.at(row, column) =
            signedUnit(generator()) / (8.0 * static_cast<Scalar>(n));
        if (row == column) {
          jacobian.at(row, column) += 1.0;
        }
      }
    }
    const auto result = vss::math::covariance::propagateCovariance(
        jacobian.view(), covariance.view());
    VSS_CHECK(result.hasValue());
    for (std::size_t row = 0U; row < outputRows; ++row) {
      for (std::size_t column = 0U; column < row; ++column) {
        VSS_CHECK(
            result.value().at(row, column) ==
            result.value()
                .at( // NOLINT(readability-suspicious-call-argument): symmetry
                     // assertion intentionally swaps coordinates.
                    column, row));
      }
    }
  }
}

void differentiationProperties() {
  auto generator = makeGenerator();
  for (std::size_t caseIndex = 0U; caseIndex < 32U; ++caseIndex) {
    const std::size_t n = 2U + (caseIndex % 7U);
    const Scalar constant = signedUnit(generator());
    Vector parameters(n);
    std::vector<vss::math::differentiation::ParameterStep> steps;
    steps.reserve(n);
    Vector linear(n);
    Vector quadratic(n);
    Vector cubic(n);
    for (std::size_t index = 0U; index < n; ++index) {
      linear.at(index) = signedUnit(generator());
      quadratic.at(index) = signedUnit(generator());
      cubic.at(index) = signedUnit(generator());
      parameters.at(index) = 0.5 * signedUnit(generator());
      steps.emplace_back(1.0);
    }
    const auto function =
        [constant, linear, quadratic, cubic](
            vss::math::ConstVectorView values) -> vss::math::Result<Scalar> {
      Scalar output = constant;
      for (std::size_t index = 0U; index < values.size(); ++index) {
        const Scalar value = values.at(index);
        output += (linear.at(index) * value) +
                  ((quadratic.at(index) * value) * value) +
                  (((cubic.at(index) * value) * value) * value);
      }
      return vss::math::Result<Scalar>::success(output);
    };
    const auto result = vss::math::differentiation::numericalGradient(
        function, parameters.view(), steps);
    VSS_CHECK(result.hasValue());
    for (std::size_t index = 0U; index < n; ++index) {
      const Scalar expected =
          linear.at(index) +
          ((2.0 * quadratic.at(index)) * parameters.at(index)) +
          (((3.0 * cubic.at(index)) * parameters.at(index)) *
           parameters.at(index));
      VSS_CHECK(std::abs(result.value().values().at(index) - expected) <=
                (1.0e-8 + (1.0e-8 * std::abs(expected))));
    }
  }
}

void statisticsProperties() {
  auto generator = makeGenerator();
  for (std::size_t caseIndex = 0U; caseIndex < 32U; ++caseIndex) {
    const std::size_t n = 8U + 2U + (caseIndex % 7U);
    Vector values(n);
    for (std::size_t index = 0U; index < n; ++index) {
      values.at(index) =
          (1024.0 * signedUnit(generator())) + static_cast<Scalar>(index % 3U);
    }
    const Scalar offset = static_cast<Scalar>(caseIndex + 1U) * 1048576.0;
    Vector translated(n);
    for (std::size_t index = 0U; index < n; ++index) {
      translated.at(index) = values.at(index) + offset;
    }
    const auto mean = vss::math::statistics::mean(values.view());
    const auto translatedMean = vss::math::statistics::mean(translated.view());
    const auto population =
        vss::math::statistics::populationVariance(values.view());
    const auto translatedPopulation =
        vss::math::statistics::populationVariance(translated.view());
    const auto sample = vss::math::statistics::sampleVariance(values.view());
    const auto translatedSample =
        vss::math::statistics::sampleVariance(translated.view());
    VSS_CHECK(mean.hasValue());
    VSS_CHECK(translatedMean.hasValue());
    VSS_CHECK(population.hasValue());
    VSS_CHECK(translatedPopulation.hasValue());
    VSS_CHECK(sample.hasValue());
    VSS_CHECK(translatedSample.hasValue());
    VSS_CHECK(std::abs((translatedMean.value() - offset) - mean.value()) <=
              (1.0e-8 + (1.0e-10 * std::abs(mean.value()))));
    VSS_CHECK(std::abs(translatedPopulation.value() - population.value()) <=
              (1.0e-8 + (1.0e-10 * std::abs(population.value()))));
    VSS_CHECK(std::abs(translatedSample.value() - sample.value()) <=
              (1.0e-8 + (1.0e-10 * std::abs(sample.value()))));
  }
}

void repeatability() {
  Matrix input(4U, 2U);
  input.at(0U, 0U) = 1.0;
  input.at(0U, 1U) = 0.0;
  input.at(1U, 0U) = 0.0;
  input.at(1U, 1U) = 1.0;
  input.at(2U, 0U) = 1.0;
  input.at(2U, 1U) = 1.0;
  input.at(3U, 0U) = 2.0;
  input.at(3U, 1U) = -1.0;
  const auto firstQr = vss::math::linear::factorizeQR(input.view());
  VSS_CHECK(firstQr.hasValue());
  for (std::size_t repetition = 0U; repetition < 100U; ++repetition) {
    const auto current = vss::math::linear::factorizeQR(input.view());
    VSS_CHECK(current.hasValue());
    const auto currentPermutation = current.value().permutation();
    const auto firstPermutation = firstQr.value().permutation();
    VSS_CHECK(currentPermutation.size() == firstPermutation.size());
    for (std::size_t index = 0U; index < currentPermutation.size(); ++index) {
      VSS_CHECK(currentPermutation[index] == firstPermutation[index]);
    }
    checkMatrixClose(current.value().upperFactor(),
                     firstQr.value().upperFactor());
    checkMatrixClose(current.value().economyQ(), firstQr.value().economyQ());
  }
  Matrix diagonal(3U, 3U, 0.0);
  diagonal.at(0U, 0U) = 1.0;
  diagonal.at(1U, 1U) = 2.0;
  diagonal.at(2U, 2U) = 3.0;
  const auto firstEigen =
      vss::math::linear::decomposeSymmetricEigen(diagonal.view());
  VSS_CHECK(firstEigen.hasValue());
  for (std::size_t repetition = 0U; repetition < 100U; ++repetition) {
    const auto current =
        vss::math::linear::decomposeSymmetricEigen(diagonal.view());
    VSS_CHECK(current.hasValue());
    checkVectorClose(copyView(current.value().eigenvalues()),
                     copyView(firstEigen.value().eigenvalues()));
  }
}

} // namespace

int main() { // NOLINT(bugprone-exception-escape): VSS_CHECK exits on failure.
  triangularProperties();
  luProperties();
  qrProperties();
  choleskyProperties();
  eigenProperties();
  covarianceProperties();
  differentiationProperties();
  statisticsProperties();
  repeatability();
}
