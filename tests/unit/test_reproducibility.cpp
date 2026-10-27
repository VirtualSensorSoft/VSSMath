#include "Test.hpp"

#include <bit>
#include <cstdint>
#include <limits>
#include <vector>
#include <vss/math/Core.hpp>
#include <vss/math/Error.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/Operations.hpp>
#include <vss/math/Vector.hpp>

namespace {

std::uint64_t bits(vss::math::Scalar value) {
  return std::bit_cast<std::uint64_t>(value);
}

void checkVectorBits(const vss::math::Vector& actual,
                     const vss::math::Vector& expected) {
  VSS_CHECK(actual.size() == expected.size());
  for (vss::math::Index index = 0U; index < actual.size(); ++index) {
    VSS_CHECK(bits(actual.at(index)) == bits(expected.at(index)));
  }
}

void checkMatrixBits(const vss::math::Matrix& actual,
                     const vss::math::Matrix& expected) {
  VSS_CHECK(actual.rows() == expected.rows());
  VSS_CHECK(actual.columns() == expected.columns());
  for (vss::math::Index row = 0U; row < actual.rows(); ++row) {
    for (vss::math::Index column = 0U; column < actual.columns(); ++column) {
      VSS_CHECK(bits(actual.at(row, column)) == bits(expected.at(row, column)));
    }
  }
}

template <class Operation, class Comparison>
void repeatSuccess(Operation operation, Comparison comparison) {
  const auto first = operation();
  VSS_CHECK(first.hasValue());
  for (unsigned repeat = 0U; repeat < 100U; ++repeat) {
    const auto current = operation();
    VSS_CHECK(current.hasValue());
    comparison(current.value(), first.value());
  }
}

template <class Operation>
void repeatFailure(Operation operation, vss::math::ErrorCode expected_code) {
  for (unsigned repeat = 0U; repeat < 100U; ++repeat) {
    const auto current = operation();
    VSS_CHECK(!current.hasValue());
    VSS_CHECK(current.status().code() == expected_code);
    VSS_CHECK(!current.status().message().empty());
  }
}

} // namespace

int main() {
  const vss::math::Vector lhs(std::vector<double>{1.0, -2.0, 3.5});
  const vss::math::Vector rhs(std::vector<double>{4.0, 5.0, -6.0});
  auto mutable_matrix = vss::math::Matrix(2U, 3U);
  mutable_matrix.at(0U, 0U) = 1.0;
  mutable_matrix.at(0U, 1U) = 2.0;
  mutable_matrix.at(0U, 2U) = 3.0;
  mutable_matrix.at(1U, 0U) = 4.0;
  mutable_matrix.at(1U, 1U) = 5.0;
  mutable_matrix.at(1U, 2U) = 6.0;
  const vss::math::Matrix right_matrix(3U, 2U, 2.0);
  const vss::math::Vector matrix_vector(std::vector<double>{2.0, -1.0, 3.0});
  repeatSuccess([&] { return vss::math::add(lhs.view(), rhs.view()); },
                [](const auto& actual, const auto& expected) {
                  checkVectorBits(actual, expected);
                });
  repeatSuccess([&] { return vss::math::subtract(lhs.view(), rhs.view()); },
                [](const auto& actual, const auto& expected) {
                  checkVectorBits(actual, expected);
                });
  repeatSuccess([&] { return vss::math::scale(lhs.view(), -2.0); },
                [](const auto& actual, const auto& expected) {
                  checkVectorBits(actual, expected);
                });
  repeatSuccess([&] { return vss::math::dot(lhs.view(), rhs.view()); },
                [](auto actual, auto expected) {
                  VSS_CHECK(bits(actual) == bits(expected));
                });
  repeatSuccess([&] { return vss::math::norm2(lhs.view()); },
                [](auto actual, auto expected) {
                  VSS_CHECK(bits(actual) == bits(expected));
                });
  repeatSuccess(
      [&] {
        return vss::math::multiply(mutable_matrix.view(), matrix_vector.view());
      },
      [](const auto& actual, const auto& expected) {
        checkVectorBits(actual, expected);
      });
  repeatSuccess(
      [&] {
        return vss::math::multiply(mutable_matrix.view(), right_matrix.view());
      },
      [](const auto& actual, const auto& expected) {
        checkMatrixBits(actual, expected);
      });
  repeatSuccess([&] { return vss::math::transpose(mutable_matrix.view()); },
                [](const auto& actual, const auto& expected) {
                  checkMatrixBits(actual, expected);
                });

  const vss::math::Vector mismatch(1U, 0.0);
  repeatFailure([&] { return vss::math::add(lhs.view(), mismatch.view()); },
                vss::math::ErrorCode::DimensionMismatch);
  repeatFailure(
      [&] {
        return vss::math::scale(
            vss::math::Vector(1U, std::numeric_limits<double>::max()).view(),
            2.0);
      },
      vss::math::ErrorCode::NonFiniteResult);
}
