#include "Test.hpp"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include <vss/math/Core.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/Operations.hpp>
#include <vss/math/Tolerance.hpp>
#include <vss/math/Vector.hpp>

namespace {

constexpr std::uint64_t seed = 0x5653534d415448ULL;
constexpr std::size_t iterations = 1000U;

[[noreturn]] void propertyFailure(std::size_t iteration, const char* property,
                                  const std::string& details) {
  std::cerr << "property failure\n"
            << "seed: " << seed << "\n"
            << "iteration: " << iteration << "\n"
            << "property: " << property << "\n"
            << details;
  std::exit(EXIT_FAILURE);
}

void require(bool condition, std::size_t iteration, const char* property,
             const std::string& details) {
  if (!condition) {
    propertyFailure(iteration, property, details);
  }
}

std::string describe(const vss::math::Vector& value) {
  std::ostringstream output;
  output << "vector[";
  for (vss::math::Index index = 0U; index < value.size(); ++index) {
    if (index != 0U) {
      output << ',';
    }
    output << value.at(index);
  }
  output << "]\n";
  return output.str();
}

std::string describe(const vss::math::Matrix& value) {
  std::ostringstream output;
  output << "matrix(" << value.rows() << 'x' << value.columns() << ")[";
  for (vss::math::Index row = 0U; row < value.rows(); ++row) {
    if (row != 0U) {
      output << ';';
    }
    for (vss::math::Index column = 0U; column < value.columns(); ++column) {
      if (column != 0U) {
        output << ',';
      }
      output << value.at(row, column);
    }
  }
  output << "]\n";
  return output.str();
}

bool nearlyEqual(const vss::math::Vector& lhs, const vss::math::Vector& rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  for (vss::math::Index index = 0U; index < lhs.size(); ++index) {
    if (!vss::math::nearlyEqual(lhs.at(index), rhs.at(index))) {
      return false;
    }
  }
  return true;
}

bool nearlyEqual(const vss::math::Matrix& lhs, const vss::math::Matrix& rhs) {
  if (lhs.rows() != rhs.rows() || lhs.columns() != rhs.columns()) {
    return false;
  }
  for (vss::math::Index row = 0U; row < lhs.rows(); ++row) {
    for (vss::math::Index column = 0U; column < lhs.columns(); ++column) {
      if (!vss::math::nearlyEqual(lhs.at(row, column), rhs.at(row, column))) {
        return false;
      }
    }
  }
  return true;
}

vss::math::Vector randomVector(std::mt19937_64& generator,
                               vss::math::Extent size,
                               std::uniform_real_distribution<double>& values) {
  vss::math::Vector result(size);
  for (vss::math::Index index = 0U; index < size; ++index) {
    result.at(index) = values(generator);
  }
  return result;
}

vss::math::Matrix randomMatrix(std::mt19937_64& generator,
                               vss::math::Extent rows,
                               vss::math::Extent columns,
                               std::uniform_real_distribution<double>& values) {
  vss::math::Matrix result(rows, columns);
  for (vss::math::Index row = 0U; row < rows; ++row) {
    for (vss::math::Index column = 0U; column < columns; ++column) {
      result.at(row, column) = values(generator);
    }
  }
  return result;
}

vss::math::Matrix identity(vss::math::Extent size) {
  vss::math::Matrix result(size, size, 0.0);
  for (vss::math::Index index = 0U; index < size; ++index) {
    result.at(index, index) = 1.0;
  }
  return result;
}

void checkReferenceFixtures() {
  const vss::math::Vector dot_lhs(std::vector<double>{1.0, 2.0});
  const vss::math::Vector dot_rhs(std::vector<double>{3.0, 4.0});
  VSS_CHECK(vss::math::dot(dot_lhs.view(), dot_rhs.view()).value() == 11.0);

  const vss::math::Vector dot2_lhs(std::vector<double>{-1.0, 2.0, 0.5});
  const vss::math::Vector dot2_rhs(std::vector<double>{4.0, -3.0, 2.0});
  VSS_CHECK(vss::math::dot(dot2_lhs.view(), dot2_rhs.view()).value() == -9.0);

  const vss::math::Vector norm(std::vector<double>{5.0, 12.0});
  VSS_CHECK(vss::math::norm2(norm.view()).value() == 13.0);

  const vss::math::Vector add_lhs(std::vector<double>{-2.0, 0.0, 3.0});
  const vss::math::Vector add_rhs(std::vector<double>{5.0, -1.0, 4.0});
  const auto added = vss::math::add(add_lhs.view(), add_rhs.view()).value();
  VSS_CHECK(added.at(0U) == 3.0);
  VSS_CHECK(added.at(1U) == -1.0);
  VSS_CHECK(added.at(2U) == 7.0);

  const vss::math::Vector scale_value(std::vector<double>{2.0, -3.0, 4.0});
  const auto scaled = vss::math::scale(scale_value.view(), -0.5).value();
  VSS_CHECK(scaled.at(0U) == -1.0);
  VSS_CHECK(scaled.at(1U) == 1.5);
  VSS_CHECK(scaled.at(2U) == -2.0);

  auto mv_matrix = vss::math::Matrix(2U, 2U, 0.0);
  mv_matrix.at(0U, 0U) = 2.0;
  mv_matrix.at(0U, 1U) = -1.0;
  mv_matrix.at(1U, 0U) = 0.0;
  mv_matrix.at(1U, 1U) = 3.0;
  const auto mv = vss::math::multiply(
                      mv_matrix.view(),
                      vss::math::Vector(std::vector<double>{4.0, 5.0}).view())
                      .value();
  VSS_CHECK(mv.at(0U) == 3.0);
  VSS_CHECK(mv.at(1U) == 15.0);

  auto row_matrix = vss::math::Matrix(1U, 3U, 0.0);
  row_matrix.at(0U, 0U) = 1.0;
  row_matrix.at(0U, 1U) = 2.0;
  row_matrix.at(0U, 2U) = 3.0;
  const auto row_product =
      vss::math::multiply(
          row_matrix.view(),
          vss::math::Vector(std::vector<double>{-1.0, 0.0, 2.0}).view())
          .value();
  VSS_CHECK(row_product.at(0U) == 5.0);

  auto lhs = vss::math::Matrix(2U, 2U, 0.0);
  lhs.at(0U, 0U) = 1.0;
  lhs.at(0U, 1U) = 2.0;
  lhs.at(1U, 0U) = 3.0;
  lhs.at(1U, 1U) = 4.0;
  auto rhs = vss::math::Matrix(2U, 2U, 0.0);
  rhs.at(0U, 0U) = 5.0;
  rhs.at(0U, 1U) = 6.0;
  rhs.at(1U, 0U) = 7.0;
  rhs.at(1U, 1U) = 8.0;
  const auto matrix_product =
      vss::math::multiply(lhs.view(), rhs.view()).value();
  VSS_CHECK(matrix_product.at(0U, 0U) == 19.0);
  VSS_CHECK(matrix_product.at(0U, 1U) == 22.0);
  VSS_CHECK(matrix_product.at(1U, 0U) == 43.0);
  VSS_CHECK(matrix_product.at(1U, 1U) == 50.0);

  auto transpose_value = vss::math::Matrix(2U, 3U, 0.0);
  transpose_value.at(0U, 0U) = 1.0;
  transpose_value.at(0U, 1U) = -2.0;
  transpose_value.at(0U, 2U) = 3.0;
  transpose_value.at(1U, 0U) = 4.0;
  transpose_value.at(1U, 1U) = 0.0;
  transpose_value.at(1U, 2U) = 5.0;
  const auto transposed = vss::math::transpose(transpose_value.view()).value();
  VSS_CHECK(transposed.at(0U, 0U) == 1.0);
  VSS_CHECK(transposed.at(0U, 1U) == 4.0);
  VSS_CHECK(transposed.at(1U, 0U) == -2.0);
  VSS_CHECK(transposed.at(1U, 1U) == 0.0);
  VSS_CHECK(transposed.at(2U, 0U) == 3.0);
  VSS_CHECK(transposed.at(2U, 1U) == 5.0);

  auto final_lhs = vss::math::Matrix(2U, 3U, 0.0);
  final_lhs.at(0U, 0U) = 1.0;
  final_lhs.at(0U, 1U) = 0.0;
  final_lhs.at(0U, 2U) = -1.0;
  final_lhs.at(1U, 0U) = 2.0;
  final_lhs.at(1U, 1U) = 3.0;
  final_lhs.at(1U, 2U) = 1.0;
  auto final_rhs = vss::math::Matrix(3U, 1U, 0.0);
  final_rhs.at(0U, 0U) = 2.0;
  final_rhs.at(1U, 0U) = 1.0;
  final_rhs.at(2U, 0U) = 4.0;
  const auto final_product =
      vss::math::multiply(final_lhs.view(), final_rhs.view()).value();
  VSS_CHECK(final_product.at(0U, 0U) == -2.0);
  VSS_CHECK(final_product.at(1U, 0U) == 11.0);
}

} // namespace

int main() {
  checkReferenceFixtures();

  // The fixed seed is recorded for reproducible properties.
  // NOLINTNEXTLINE(cert-msc32-c,cert-msc51-cpp)
  std::mt19937_64 generator(seed);
  std::uniform_int_distribution<unsigned> dimensions(0U, 8U);
  std::uniform_real_distribution<double> values(-100.0, 100.0);

  for (std::size_t iteration = 0U; iteration < iterations; ++iteration) {
    const auto size = static_cast<vss::math::Extent>(dimensions(generator));
    const auto vector = randomVector(generator, size, values);
    const auto other = randomVector(generator, size, values);
    const vss::math::Vector zero(size, 0.0);
    const auto context = describe(vector) + describe(other);

    const auto plus_zero = vss::math::add(vector.view(), zero.view());
    require(plus_zero.hasValue() && nearlyEqual(plus_zero.value(), vector),
            iteration, "v + 0 = v", context);
    const auto self_difference =
        vss::math::subtract(vector.view(), vector.view());
    require(self_difference.hasValue() &&
                nearlyEqual(self_difference.value(), zero),
            iteration, "v - v = 0", context);
    const auto dot_lhs_rhs = vss::math::dot(vector.view(), other.view());
    const auto dot_rhs_lhs = vss::math::dot(other.view(), vector.view());
    require(
        dot_lhs_rhs.hasValue() && dot_rhs_lhs.hasValue() &&
            vss::math::nearlyEqual(dot_lhs_rhs.value(), dot_rhs_lhs.value()),
        iteration, "dot(v,w) = dot(w,v)", context);
    const auto dot_zero = vss::math::dot(vector.view(), zero.view());
    require(dot_zero.hasValue() && dot_zero.value() == 0.0, iteration,
            "dot(v,0) = 0", context);
    const auto scale_one = vss::math::scale(vector.view(), 1.0);
    require(scale_one.hasValue() && nearlyEqual(scale_one.value(), vector),
            iteration, "scale(v,1) = v", context);
    const auto scale_zero = vss::math::scale(vector.view(), 0.0);
    require(scale_zero.hasValue() && nearlyEqual(scale_zero.value(), zero),
            iteration, "scale(v,0) = 0", context);

    const auto rows = static_cast<vss::math::Extent>(dimensions(generator));
    const auto columns = static_cast<vss::math::Extent>(dimensions(generator));
    const auto matrix = randomMatrix(generator, rows, columns, values);
    const auto matrix_context = describe(matrix) + describe(vector);
    const auto transpose_once = vss::math::transpose(matrix.view());
    const auto twice_transposed =
        vss::math::transpose(transpose_once.value().view());
    require(twice_transposed.hasValue() &&
                nearlyEqual(twice_transposed.value(), matrix),
            iteration, "transpose(transpose(A)) = A", matrix_context);
    require(transpose_once.value().rows() == matrix.columns() &&
                transpose_once.value().columns() == matrix.rows(),
            iteration, "transpose preserves swapped dimensions",
            matrix_context);

    const auto compatible_vector = randomVector(generator, columns, values);
    const auto another_vector = randomVector(generator, columns, values);
    const auto sum_vectors =
        vss::math::add(compatible_vector.view(), another_vector.view());
    const auto left_distributive =
        vss::math::multiply(matrix.view(), sum_vectors.value().view());
    const auto first_product =
        vss::math::multiply(matrix.view(), compatible_vector.view());
    const auto second_product =
        vss::math::multiply(matrix.view(), another_vector.view());
    const auto right_sum = vss::math::add(first_product.value().view(),
                                          second_product.value().view());
    require(left_distributive.hasValue() && right_sum.hasValue() &&
                nearlyEqual(left_distributive.value(), right_sum.value()),
            iteration, "A * (x + y) = A*x + A*y",
            describe(matrix) + describe(compatible_vector) +
                describe(another_vector));

    const auto other_matrix = randomMatrix(generator, rows, columns, values);
    auto matrix_sum = matrix;
    for (vss::math::Index row = 0U; row < rows; ++row) {
      for (vss::math::Index column = 0U; column < columns; ++column) {
        matrix_sum.at(row, column) += other_matrix.at(row, column);
      }
    }
    const auto lhs_sum_product =
        vss::math::multiply(matrix_sum.view(), compatible_vector.view());
    const auto first_matrix_product =
        vss::math::multiply(matrix.view(), compatible_vector.view());
    const auto second_matrix_product =
        vss::math::multiply(other_matrix.view(), compatible_vector.view());
    const auto rhs_sum_product =
        vss::math::add(first_matrix_product.value().view(),
                       second_matrix_product.value().view());
    require(lhs_sum_product.hasValue() && rhs_sum_product.hasValue() &&
                nearlyEqual(lhs_sum_product.value(), rhs_sum_product.value()),
            iteration, "(A+B)*x = A*x + B*x",
            describe(matrix) + describe(other_matrix) +
                describe(compatible_vector));

    const auto square_size =
        static_cast<vss::math::Extent>(dimensions(generator));
    const auto square =
        randomMatrix(generator, square_size, square_size, values);
    const auto id = identity(square_size);
    const auto identity_product = vss::math::multiply(id.view(), square.view());
    require(identity_product.hasValue() &&
                nearlyEqual(identity_product.value(), square),
            iteration, "I*A = A", describe(square));
    const auto identity_vector = randomVector(generator, square_size, values);
    const auto identity_vector_product =
        vss::math::multiply(id.view(), identity_vector.view());
    require(identity_vector_product.hasValue() &&
                nearlyEqual(identity_vector_product.value(), identity_vector),
            iteration, "I*x = x", describe(identity_vector));
    const vss::math::Matrix zero_matrix(rows, columns, 0.0);
    const auto zero_product =
        vss::math::multiply(zero_matrix.view(), compatible_vector.view());
    require(zero_product.hasValue() &&
                nearlyEqual(zero_product.value(), vss::math::Vector(rows, 0.0)),
            iteration, "0*x = 0",
            describe(zero_matrix) + describe(compatible_vector));

    const auto assoc_a_rows = 1U + dimensions(generator) % 4U;
    const auto assoc_shared = 1U + dimensions(generator) % 4U;
    const auto assoc_b_columns = 1U + dimensions(generator) % 4U;
    const auto assoc_c_columns = 1U + dimensions(generator) % 4U;
    const auto assoc_a =
        randomMatrix(generator, assoc_a_rows, assoc_shared, values);
    const auto assoc_b =
        randomMatrix(generator, assoc_shared, assoc_b_columns, values);
    const auto assoc_c =
        randomMatrix(generator, assoc_b_columns, assoc_c_columns, values);
    const auto ab = vss::math::multiply(assoc_a.view(), assoc_b.view());
    const auto bc = vss::math::multiply(assoc_b.view(), assoc_c.view());
    const auto left_assoc =
        vss::math::multiply(ab.value().view(), assoc_c.view());
    const auto right_assoc =
        vss::math::multiply(assoc_a.view(), bc.value().view());
    require(left_assoc.hasValue() && right_assoc.hasValue() &&
                nearlyEqual(left_assoc.value(), right_assoc.value()),
            iteration, "(A*B)*C = A*(B*C) within tolerance",
            describe(assoc_a) + describe(assoc_b) + describe(assoc_c));
  }

  return EXIT_SUCCESS;
}
