#include "Test.hpp"

#include <cmath>
#include <initializer_list>
#include <limits>
#include <vector>
#include <vss/math/Core.hpp>
#include <vss/math/Error.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/MatrixView.hpp>
#include <vss/math/Operations.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/Tolerance.hpp>
#include <vss/math/Vector.hpp>
#include <vss/math/VectorView.hpp>

namespace {

void checkVector(const vss::math::Vector& value,
                 std::initializer_list<vss::math::Scalar> expected) {
  VSS_CHECK(value.size() == expected.size());
  vss::math::Index index = 0U;
  for (const auto element : expected) {
    VSS_CHECK(value.at(index) == element);
    ++index;
  }
}

template <class T>
void checkFailure(const vss::math::Result<T>& result,
                  vss::math::ErrorCode code) {
  VSS_CHECK(!result.hasValue());
  VSS_CHECK(result.status().code() == code);
  VSS_CHECK(!result.status().message().empty());
}

} // namespace

int main() {
  vss::math::Vector a(std::vector<double>{1.0, -2.0, 3.0});
  vss::math::Vector b(std::vector<double>{4.0, 5.0, -6.0});
  const auto original_a = a;
  const auto original_b = b;

  const auto added = vss::math::add(a.view(), b.view());
  VSS_CHECK(added.hasValue());
  checkVector(added.value(), {5.0, 3.0, -3.0});
  const auto subtracted = vss::math::subtract(a.view(), b.view());
  VSS_CHECK(subtracted.hasValue());
  checkVector(subtracted.value(), {-3.0, -7.0, 9.0});
  const auto scaled = vss::math::scale(a.view(), -2.0);
  VSS_CHECK(scaled.hasValue());
  checkVector(scaled.value(), {-2.0, 4.0, -6.0});
  const auto product = vss::math::dot(a.view(), b.view());
  VSS_CHECK(product.hasValue());
  VSS_CHECK(product.value() == -24.0);
  const vss::math::Vector four(std::vector<double>{3.0, 4.0});
  const auto length = vss::math::norm2(four.view());
  VSS_CHECK(length.hasValue());
  VSS_CHECK(length.value() == 5.0);
  VSS_CHECK(a.at(0U) == original_a.at(0U));
  VSS_CHECK(a.at(2U) == original_a.at(2U));
  VSS_CHECK(b.at(0U) == original_b.at(0U));
  VSS_CHECK(b.at(2U) == original_b.at(2U));
  VSS_CHECK(added.value().data() != a.data());

  const vss::math::Matrix matrix(2U, 3U, 0.0);
  auto mutable_matrix = vss::math::Matrix(2U, 3U, 0.0);
  mutable_matrix.at(0U, 0U) = 1.0;
  mutable_matrix.at(0U, 1U) = 2.0;
  mutable_matrix.at(0U, 2U) = 3.0;
  mutable_matrix.at(1U, 0U) = 4.0;
  mutable_matrix.at(1U, 1U) = 5.0;
  mutable_matrix.at(1U, 2U) = 6.0;
  auto vector = vss::math::Vector(std::vector<double>{2.0, -1.0, 3.0});
  const auto matrix_vector =
      vss::math::multiply(mutable_matrix.view(), vector.view());
  VSS_CHECK(matrix_vector.hasValue());
  checkVector(matrix_vector.value(), {9.0, 21.0});

  auto right_matrix = vss::math::Matrix(3U, 2U, 0.0);
  right_matrix.at(0U, 0U) = 7.0;
  right_matrix.at(0U, 1U) = 8.0;
  right_matrix.at(1U, 0U) = 9.0;
  right_matrix.at(1U, 1U) = 10.0;
  right_matrix.at(2U, 0U) = 11.0;
  right_matrix.at(2U, 1U) = 12.0;
  const auto matrix_product =
      vss::math::multiply(mutable_matrix.view(), right_matrix.view());
  VSS_CHECK(matrix_product.hasValue());
  VSS_CHECK(matrix_product.value().at(0U, 0U) == 58.0);
  VSS_CHECK(matrix_product.value().at(0U, 1U) == 64.0);
  VSS_CHECK(matrix_product.value().at(1U, 0U) == 139.0);
  VSS_CHECK(matrix_product.value().at(1U, 1U) == 154.0);
  const auto transposed = vss::math::transpose(mutable_matrix.view());
  VSS_CHECK(transposed.hasValue());
  VSS_CHECK(transposed.value().rows() == 3U);
  VSS_CHECK(transposed.value().columns() == 2U);
  VSS_CHECK(transposed.value().at(0U, 0U) == 1.0);
  VSS_CHECK(transposed.value().at(0U, 1U) == 4.0);
  VSS_CHECK(transposed.value().at(2U, 0U) == 3.0);
  VSS_CHECK(transposed.value().at(2U, 1U) == 6.0);
  VSS_CHECK(matrix.rows() == 2U);
  VSS_CHECK(matrix.columns() == 3U);

  const vss::math::Vector empty_vector;
  VSS_CHECK(
      vss::math::add(empty_vector.view(), empty_vector.view()).value().empty());
  VSS_CHECK(vss::math::subtract(empty_vector.view(), empty_vector.view())
                .value()
                .empty());
  VSS_CHECK(vss::math::dot(empty_vector.view(), empty_vector.view()).value() ==
            0.0);
  VSS_CHECK(vss::math::norm2(empty_vector.view()).value() == 0.0);
  const vss::math::Matrix zero_by_three(0U, 3U);
  const vss::math::Matrix two_by_zero(2U, 0U);
  const vss::math::Matrix zero_by_two(0U, 2U);
  VSS_CHECK(
      vss::math::multiply(zero_by_three.view(), vss::math::Vector(3U).view())
          .value()
          .empty());
  const auto two_zero_vector =
      vss::math::multiply(two_by_zero.view(), empty_vector.view());
  VSS_CHECK(two_zero_vector.value().size() == 2U);
  VSS_CHECK(two_zero_vector.value().at(0U) == 0.0);
  const auto zero_matrix_product =
      vss::math::multiply(two_by_zero.view(), zero_by_three.view());
  VSS_CHECK(zero_matrix_product.value().rows() == 2U);
  VSS_CHECK(zero_matrix_product.value().columns() == 3U);
  VSS_CHECK(!zero_matrix_product.value().empty());
  VSS_CHECK(zero_matrix_product.value().at(0U, 0U) == 0.0);
  VSS_CHECK(zero_matrix_product.value().at(1U, 2U) == 0.0);
  VSS_CHECK(
      vss::math::multiply(zero_by_two.view(), vss::math::Matrix(2U, 3U).view())
          .value()
          .rows() == 0U);
  VSS_CHECK(vss::math::transpose(zero_by_three.view()).value().rows() == 3U);
  VSS_CHECK(vss::math::transpose(two_by_zero.view()).value().columns() == 2U);

  const auto infinity = std::numeric_limits<double>::infinity();
  const auto nan = std::numeric_limits<double>::quiet_NaN();
  const vss::math::Vector infinity_vector(2U, infinity);
  const vss::math::Vector nan_vector(2U, nan);
  const vss::math::Vector mismatch(1U, 0.0);
  const vss::math::OperationOptions invalid_options{{-1.0, 0.0}, true};
  checkFailure(
      vss::math::add(mismatch.view(), infinity_vector.view(), invalid_options),
      vss::math::ErrorCode::InvalidTolerance);
  checkFailure(vss::math::add(mismatch.view(), infinity_vector.view()),
               vss::math::ErrorCode::DimensionMismatch);
  checkFailure(vss::math::add(nan_vector.view(), infinity_vector.view()),
               vss::math::ErrorCode::NonFiniteInput);
  checkFailure(vss::math::scale(infinity_vector.view(), 2.0),
               vss::math::ErrorCode::NonFiniteInput);
  checkFailure(vss::math::scale(vss::math::Vector(1U, 1.0).view(), infinity),
               vss::math::ErrorCode::NonFiniteInput);
  checkFailure(vss::math::dot(nan_vector.view(), infinity_vector.view()),
               vss::math::ErrorCode::NonFiniteInput);
  checkFailure(vss::math::norm2(nan_vector.view()),
               vss::math::ErrorCode::NonFiniteInput);
  checkFailure(vss::math::multiply(mutable_matrix.view(), mismatch.view()),
               vss::math::ErrorCode::DimensionMismatch);

  const vss::math::Matrix infinity_matrix(1U, 1U, infinity);
  checkFailure(vss::math::multiply(infinity_matrix.view(),
                                   vss::math::Vector(1U, 1.0).view()),
               vss::math::ErrorCode::NonFiniteInput);
  checkFailure(
      vss::math::multiply(infinity_matrix.view(), infinity_matrix.view()),
      vss::math::ErrorCode::NonFiniteInput);
  checkFailure(vss::math::transpose(infinity_matrix.view()),
               vss::math::ErrorCode::NonFiniteInput);

  const vss::math::Extent maximum =
      std::numeric_limits<vss::math::Extent>::max();
  const vss::math::Matrix huge_rows(maximum, 0U);
  const vss::math::Matrix zero_columns(0U, 2U);
  checkFailure(vss::math::multiply(huge_rows.view(), zero_columns.view()),
               vss::math::ErrorCode::SizeOverflow);

  const auto overflow_scale = vss::math::scale(
      vss::math::Vector(1U, std::numeric_limits<double>::max()).view(), 2.0);
  checkFailure(overflow_scale, vss::math::ErrorCode::NonFiniteResult);
  const auto overflow_dot = vss::math::dot(
      vss::math::Vector(2U, std::numeric_limits<double>::max()).view(),
      vss::math::Vector(2U, 2.0).view());
  checkFailure(overflow_dot, vss::math::ErrorCode::NonFiniteResult);

  const vss::math::OperationOptions allow_non_finite{{}, false};
  VSS_CHECK(vss::math::scale(infinity_vector.view(), 2.0, allow_non_finite)
                .value()
                .at(0U) == infinity);
  VSS_CHECK(std::isnan(
      vss::math::add(nan_vector.view(), nan_vector.view(), allow_non_finite)
          .value()
          .at(0U)));
  const auto exact_options = vss::math::OperationOptions{{0.0, 0.0}, true};
  const auto loose_options =
      vss::math::OperationOptions{{1.0e-6, 1.0e-3}, true};
  VSS_CHECK(vss::math::add(a.view(), b.view(), exact_options).value().at(0U) ==
            vss::math::add(a.view(), b.view(), loose_options).value().at(0U));

  const vss::math::Vector large(std::vector<double>{3.0e200, 4.0e200});
  const vss::math::Vector small(std::vector<double>{3.0e-200, 4.0e-200});
  VSS_CHECK(std::abs(vss::math::norm2(large.view()).value() - 5.0e200) <
            1.0e188);
  VSS_CHECK(std::abs(vss::math::norm2(small.view()).value() - 5.0e-200) <
            1.0e-212);
}
