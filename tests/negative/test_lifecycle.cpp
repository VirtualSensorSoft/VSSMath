#include "Test.hpp"

#include <cmath>
#include <cstddef>
#include <thread>
#include <utility>
#include <vector>
#include <vss/math/Core.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/Vector.hpp>
#include <vss/math/covariance/Covariance.hpp>
#include <vss/math/differentiation/FiniteDifference.hpp>
#include <vss/math/linear/Cholesky.hpp>
#include <vss/math/linear/LU.hpp>
#include <vss/math/linear/QR.hpp>
#include <vss/math/linear/SymmetricEigen.hpp>

namespace {

using vss::math::Matrix;
using vss::math::Scalar;
using vss::math::Vector;

Matrix diagonal(std::size_t size) {
  Matrix result(size, size, 0.0);
  for (std::size_t index = 0U; index < size; ++index) {
    result.at(index, index) = static_cast<Scalar>(index + 2U);
  }
  return result;
}

void factorizationCopiesAndMoves() {
  Matrix input = diagonal(3U);
  const auto luResult = vss::math::linear::factorizeLU(input.view());
  const auto qrResult = vss::math::linear::factorizeQR(input.view());
  const auto choleskyResult =
      vss::math::linear::factorizeCholesky(input.view());
  const auto eigenResult =
      vss::math::linear::decomposeSymmetricEigen(input.view());
  VSS_CHECK(luResult.hasValue());
  VSS_CHECK(qrResult.hasValue());
  VSS_CHECK(choleskyResult.hasValue());
  VSS_CHECK(eigenResult.hasValue());

  auto luCopy = luResult.value();
  auto luMoved = std::move(luCopy);
  // NOLINTNEXTLINE(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
  VSS_CHECK(luCopy.size() == 0U);
  VSS_CHECK(luMoved.size() == 3U);
  auto qrCopy = qrResult.value();
  auto qrMoved = std::move(qrCopy);
  // NOLINTNEXTLINE(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
  VSS_CHECK(qrCopy.rows() == 0U);
  VSS_CHECK(qrMoved.rows() == 3U);
  auto choleskyCopy = choleskyResult.value();
  auto choleskyMoved = std::move(choleskyCopy);
  // NOLINTNEXTLINE(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
  VSS_CHECK(choleskyCopy.size() == 0U);
  VSS_CHECK(choleskyMoved.size() == 3U);
  auto eigenCopy = eigenResult.value();
  auto eigenMoved = std::move(eigenCopy);
  // NOLINTNEXTLINE(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
  VSS_CHECK(eigenCopy.size() == 0U);
  VSS_CHECK(eigenMoved.size() == 3U);
}

void derivativeAndCorrelationMoves() {
  Vector parameters(std::vector<Scalar>{1.0, 2.0});
  std::vector<vss::math::differentiation::ParameterStep> steps;
  steps.emplace_back(1.0);
  steps.emplace_back(1.0);
  const auto callback = [](vss::math::ConstVectorView values) {
    return vss::math::Result<Scalar>::success((values.at(0U) * values.at(0U)) +
                                              values.at(1U));
  };
  const auto gradient = vss::math::differentiation::numericalGradient(
      callback, parameters.view(), steps);
  VSS_CHECK(gradient.hasValue());
  auto gradientCopy = gradient.value();
  auto gradientMoved = std::move(gradientCopy);
  // NOLINTNEXTLINE(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
  VSS_CHECK(gradientCopy.values().empty());
  VSS_CHECK(gradientMoved.values().size() == 2U);

  Matrix covariance(2U, 2U);
  covariance.at(0U, 0U) = 4.0;
  covariance.at(0U, 1U) = 1.0;
  covariance.at(1U, 0U) = 1.0;
  covariance.at(1U, 1U) = 9.0;
  auto conversion = vss::math::covariance::toCorrelation(covariance.view());
  VSS_CHECK(conversion.hasValue());
  auto conversionMoved = std::move(conversion.value());
  VSS_CHECK(conversionMoved.correlation().rows() == 2U);
}

void constSolvesAreIndependent() {
  const Matrix input = diagonal(3U);
  const auto factorization = vss::math::linear::factorizeLU(input.view());
  VSS_CHECK(factorization.hasValue());
  const Vector rightHandSide(std::vector<Scalar>{2.0, 6.0, 12.0});
  Vector first;
  Vector second;
  std::thread firstThread([&]() {
    const auto result = factorization.value().solve(rightHandSide.view());
    VSS_CHECK(result.hasValue());
    first = result.value().solution();
  });
  std::thread secondThread([&]() {
    const auto result = factorization.value().solve(rightHandSide.view());
    VSS_CHECK(result.hasValue());
    second = result.value().solution();
  });
  firstThread.join();
  secondThread.join();
  VSS_CHECK(first.size() == 3U);
  VSS_CHECK(second.size() == 3U);
  VSS_CHECK(std::abs(first.at(0U) - 1.0) < 1.0e-12);
  VSS_CHECK(std::abs(second.at(2U) - 3.0) < 1.0e-12);
}

} // namespace

int main() { // NOLINT(bugprone-exception-escape): VSS_CHECK exits on failure.
  factorizationCopiesAndMoves();
  derivativeAndCorrelationMoves();
  constSolvesAreIndependent();
}
