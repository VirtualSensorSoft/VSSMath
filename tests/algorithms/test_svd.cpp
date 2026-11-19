#include "Test.hpp"

#include <algorithm>
#include <cmath>
#include <exception>
#include <iostream>
#include <limits>
#include <vss/math/Core.hpp>
#include <vss/math/Error.hpp>
#include <vss/math/Limits.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/linear/Decomposition.hpp>
#include <vss/math/linear/SolvePolicy.hpp>

namespace {

using namespace vss::math;

constexpr Limits budget{.elements = 4096, .bytes = 65536, .evaluations = 0};

SolvePolicy policy() {
  SolvePolicy settings{};
  settings.iteration.max_iterations = 128;
  settings.rank_rel = 1e-12;
  return settings;
}

void checkFactors(const Matrix& input, const SVD& factors, Scalar tolerance) {
  const auto count = std::min(input.rows(), input.columns());
  VSS_CHECK(factors.left.rows() == input.rows());
  VSS_CHECK(factors.left.columns() == count);
  VSS_CHECK(factors.singular.size() == count);
  VSS_CHECK(factors.right_transpose.rows() == count);
  VSS_CHECK(factors.right_transpose.columns() == input.columns());
  for (Index row = 0; row < input.rows(); ++row) {
    for (Index col = 0; col < input.columns(); ++col) {
      Scalar rebuilt = 0;
      for (Index axis = 0; axis < count; ++axis) {
        rebuilt += factors.left.at(row, axis) * factors.singular.at(axis) *
                   factors.right_transpose.at(axis, col);
      }
      VSS_CHECK(std::abs(input.at(row, col) - rebuilt) <= tolerance);
    }
  }
  for (Index i = 0; i < count; ++i) {
    for (Index j = 0; j < count; ++j) {
      Scalar left = 0;
      Scalar right = 0;
      for (Index row = 0; row < input.rows(); ++row) {
        left += factors.left.at(row, i) * factors.left.at(row, j);
      }
      for (Index col = 0; col < input.columns(); ++col) {
        right += factors.right_transpose.at(i, col) *
                 factors.right_transpose.at(j, col);
      }
      const Scalar expected = i == j ? 1.0 : 0.0;
      VSS_CHECK(std::abs(left - expected) < 1e-12);
      VSS_CHECK(std::abs(right - expected) < 1e-12);
    }
  }
}

void exactCutoff() {
  Matrix diagonal(3, 5);
  diagonal.at(0, 4) = -8;
  diagonal.at(1, 0) = 2;
  diagonal.at(2, 1) = 0.25;
  auto settings = policy();
  settings.rank_abs = 2;
  settings.rank_rel = 0.25;
  settings.iteration.max_iterations = 0;
  const auto result = linear::factorSVD(diagonal.view(), settings, budget);
  VSS_CHECK(result.hasValue());
  const auto& factors = result.value();
  VSS_CHECK(factors.rank == 1);
  VSS_CHECK(factors.cutoff == 2);
  VSS_CHECK(factors.singular.at(0) == 8);
  VSS_CHECK(factors.singular.at(1) == 2);
  VSS_CHECK(factors.singular.at(2) == 0.25);
  checkFactors(diagonal, factors, 0);
}

void denseSpectrum() {
  // The two columns have norms sqrt(18) and dot product 14, so their
  // squared singular values are 32 and 4.
  Matrix dense(3, 2);
  dense.at(0, 0) = 4;
  dense.at(1, 0) = 1;
  dense.at(2, 0) = 1;
  dense.at(0, 1) = 4;
  dense.at(1, 1) = -1;
  dense.at(2, 1) = -1;
  const auto result = linear::factorSVD(dense.view(), policy(), budget);
  VSS_CHECK(result.hasValue());
  VSS_CHECK(result.value().rank == 2);
  VSS_CHECK(std::abs(result.value().singular.at(0) - std::sqrt(32.0)) < 1e-12);
  VSS_CHECK(std::abs(result.value().singular.at(1) - 2.0) < 1e-12);
  checkFactors(dense, result.value(), 1e-12);

  auto settings = policy();
  settings.iteration.max_iterations = 0;
  VSS_CHECK(linear::factorSVD(dense.view(), settings, budget).status().code() ==
            ErrorCode::NoConvergence);
  settings = policy();
  settings.rank_rel = 1;
  VSS_CHECK(linear::factorSVD(dense.view(), settings, budget).status().code() ==
            ErrorCode::InvalidTolerance);
  constexpr Limits denied{.elements = 4096, .bytes = 1, .evaluations = 0};
  VSS_CHECK(linear::factorSVD(dense.view(), policy(), denied).status().code() ==
            ErrorCode::ResourceLimit);
  dense.at(2, 1) = std::numeric_limits<Scalar>::infinity();
  VSS_CHECK(linear::factorSVD(dense.view(), policy(), budget).status().code() ==
            ErrorCode::NonFiniteInput);
}

void multiStepReconstruction() {
  for (const Index rows : {3U, 5U, 7U}) {
    for (const Index cols : {3U, 5U, 7U}) {
      Matrix input(rows, cols);
      for (Index row = 0; row < rows; ++row) {
        for (Index col = 0; col < cols; ++col) {
          const auto sample = ((row + 3) * (col + 2) * (row + col + 1)) % 23;
          input.at(row, col) = static_cast<Scalar>(sample) - 11;
        }
      }
      const auto result = linear::factorSVD(input.view(), policy(), budget);
      VSS_CHECK(result.hasValue());
      checkFactors(input, result.value(), 1e-12);
      for (Index k = 1; k < result.value().singular.size(); ++k) {
        VSS_CHECK(result.value().singular.at(k - 1) >=
                  result.value().singular.at(k));
      }
    }
  }
}

void deficientAndOverflow() {
  Matrix input(3, 3);
  input.at(0, 0) = 2;
  input.at(0, 1) = 4;
  input.at(1, 0) = 1;
  input.at(1, 1) = 2;
  const auto result = linear::factorSVD(input.view(), policy(), budget);
  VSS_CHECK(result.hasValue());
  VSS_CHECK(result.value().rank == 1);
  VSS_CHECK(std::abs(result.value().singular.at(0) - 5) < 1e-12);
  checkFactors(input, result.value(), 1e-12);

  Matrix tooLarge(2, 1, std::numeric_limits<Scalar>::max());
  VSS_CHECK(
      linear::factorSVD(tooLarge.view(), policy(), budget).status().code() ==
      ErrorCode::NonFiniteResult);
}

void emptyAndScaled() {
  const Matrix empty(0, 4);
  const auto empty_result = linear::factorSVD(empty.view(), policy(), budget);
  VSS_CHECK(empty_result.hasValue());
  VSS_CHECK(empty_result.value().rank == 0);
  checkFactors(empty, empty_result.value(), 0);

  Matrix input(2, 2);
  input.at(0, 0) = std::scalbn(1.0, 900);
  input.at(1, 1) = std::scalbn(1.0, -900);
  auto settings = policy();
  settings.rank_rel = 0;
  const auto scaled = linear::factorSVD(input.view(), settings, budget);
  VSS_CHECK(scaled.hasValue());
  VSS_CHECK(scaled.value().rank == 2);
  VSS_CHECK(scaled.value().singular.at(1) == input.at(1, 1));
  checkFactors(input, scaled.value(), 0);
}

} // namespace

int main() try {
  exactCutoff();
  denseSpectrum();
  multiStepReconstruction();
  deficientAndOverflow();
  emptyAndScaled();
  return 0;
} catch (const std::exception& error) {
  std::cerr << error.what() << '\n';
  return 1;
}
