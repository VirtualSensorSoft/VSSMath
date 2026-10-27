#include "Test.hpp"

#include <limits>
#include <span>
#include <vector>
#include <vss/math/Core.hpp>
#include <vss/math/Error.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/MatrixView.hpp>
#include <vss/math/Operations.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/Tolerance.hpp>
#include <vss/math/Vector.hpp>

namespace {

template <class T>
void checkFailure(const vss::math::Result<T>& result,
                  vss::math::ErrorCode expected_code) {
  VSS_CHECK(!result.hasValue());
  VSS_CHECK(result.status().code() == expected_code);
  VSS_CHECK(!result.status().message().empty());
}

} // namespace

int main() {
  const auto infinity = std::numeric_limits<vss::math::Scalar>::infinity();
  const auto nan = std::numeric_limits<vss::math::Scalar>::quiet_NaN();
  const auto maximum = std::numeric_limits<vss::math::Extent>::max();

  const vss::math::Vector one_value(std::vector<double>{nan});
  const vss::math::Vector two_values(std::vector<double>{infinity, 1.0});
  const vss::math::OperationOptions invalid_options{{-1.0, 0.0}, true};

  // Tolerance validation precedes all later vector checks.
  checkFailure(
      vss::math::add(one_value.view(), two_values.view(), invalid_options),
      vss::math::ErrorCode::InvalidTolerance);
  checkFailure(vss::math::add(one_value.view(), two_values.view()),
               vss::math::ErrorCode::DimensionMismatch);

  const vss::math::Vector nan_value(std::vector<double>{nan});
  checkFailure(
      vss::math::add(nan_value.view(), nan_value.view(), invalid_options),
      vss::math::ErrorCode::InvalidTolerance);
  checkFailure(vss::math::add(nan_value.view(), nan_value.view()),
               vss::math::ErrorCode::NonFiniteInput);

  // Use valid zero-area views so the result-size overflow is checked without
  // allocating or reading a fabricated element.
  const std::span<vss::math::Scalar> empty_storage;
  const auto huge_rows =
      vss::math::MatrixView::create(empty_storage, maximum, 0U, 0U, 0U);
  const auto zero_by_two =
      vss::math::MatrixView::create(empty_storage, 0U, 2U, 0U, 0U);
  VSS_CHECK(huge_rows.hasValue());
  VSS_CHECK(zero_by_two.hasValue());
  checkFailure(vss::math::multiply(huge_rows.value(), zero_by_two.value()),
               vss::math::ErrorCode::SizeOverflow);

  const vss::math::Vector non_finite_vector(std::vector<double>{nan, infinity});
  checkFailure(vss::math::scale(non_finite_vector.view(), 1.0),
               vss::math::ErrorCode::NonFiniteInput);
  checkFailure(
      vss::math::dot(non_finite_vector.view(), non_finite_vector.view()),
      vss::math::ErrorCode::NonFiniteInput);
  checkFailure(vss::math::multiply(vss::math::Matrix(1U, 2U, 1.0).view(),
                                   vss::math::Vector(1U, 1.0).view()),
               vss::math::ErrorCode::DimensionMismatch);
  const vss::math::Vector max_vector(2U, std::numeric_limits<double>::max());
  const vss::math::Vector negative_max_vector(
      2U, -std::numeric_limits<double>::max());
  checkFailure(vss::math::add(max_vector.view(), max_vector.view()),
               vss::math::ErrorCode::NonFiniteResult);
  checkFailure(
      vss::math::subtract(negative_max_vector.view(), max_vector.view()),
      vss::math::ErrorCode::NonFiniteResult);
  checkFailure(vss::math::scale(max_vector.view(), 2.0),
               vss::math::ErrorCode::NonFiniteResult);
  checkFailure(
      vss::math::dot(max_vector.view(), vss::math::Vector(2U, 2.0).view()),
      vss::math::ErrorCode::NonFiniteResult);
}
