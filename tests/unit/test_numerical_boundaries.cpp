#include "Test.hpp"

#include <cmath>
#include <limits>
#include <numbers>
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
#include <vss/math/VectorView.hpp>

namespace {

template <class T>
void checkFailure(const vss::math::Result<T>& result,
                  vss::math::ErrorCode expected_code) {
  VSS_CHECK(!result.hasValue());
  VSS_CHECK(result.status().code() == expected_code);
  VSS_CHECK(!result.status().message().empty());
}

void checkFiniteNorm(const vss::math::Vector& value) {
  const auto result = vss::math::norm2(value.view());
  VSS_CHECK(result.hasValue());
  VSS_CHECK(vss::math::isFinite(result.value()));
  VSS_CHECK(result.value() >= 0.0);
}

} // namespace

int main() {
  const auto infinity = std::numeric_limits<vss::math::Scalar>::infinity();
  const auto nan = std::numeric_limits<vss::math::Scalar>::quiet_NaN();
  const auto maximum = std::numeric_limits<vss::math::Scalar>::max();
  const auto extent_max = std::numeric_limits<vss::math::Extent>::max();

  const vss::math::Vector three_four(std::vector<double>{3.0, 4.0});
  const auto norm = vss::math::norm2(three_four.view());
  VSS_CHECK(norm.hasValue());
  VSS_CHECK(norm.value() == 5.0);

  const vss::math::Vector scaled_max(2U, maximum / 4.0);
  const auto scaled_norm = vss::math::norm2(scaled_max.view());
  VSS_CHECK(scaled_norm.hasValue());
  const auto expected_scaled_norm = std::numbers::sqrt2 * (maximum / 4.0);
  VSS_CHECK(vss::math::nearlyEqual(scaled_norm.value(), expected_scaled_norm,
                                   {1.0e-12, 1.0e-12}));

  checkFiniteNorm(vss::math::Vector(2U, 0.0));
  checkFiniteNorm(vss::math::Vector(
      std::vector<double>{std::numeric_limits<double>::denorm_min(), 0.0}));

  const vss::math::Vector nan_values(2U, nan);
  const vss::math::Vector positive_infinity_values(2U, infinity);
  const vss::math::Vector negative_infinity_values(2U, -infinity);
  checkFailure(vss::math::scale(nan_values.view(), 1.0),
               vss::math::ErrorCode::NonFiniteInput);
  checkFailure(vss::math::scale(positive_infinity_values.view(), 1.0),
               vss::math::ErrorCode::NonFiniteInput);
  checkFailure(vss::math::scale(negative_infinity_values.view(), 1.0),
               vss::math::ErrorCode::NonFiniteInput);

  checkFailure(vss::math::add(vss::math::Vector(1U, maximum).view(),
                              vss::math::Vector(1U, maximum).view()),
               vss::math::ErrorCode::NonFiniteResult);
  checkFailure(vss::math::subtract(vss::math::Vector(1U, -maximum).view(),
                                   vss::math::Vector(1U, maximum).view()),
               vss::math::ErrorCode::NonFiniteResult);
  checkFailure(vss::math::scale(vss::math::Vector(1U, maximum).view(), 2.0),
               vss::math::ErrorCode::NonFiniteResult);
  checkFailure(vss::math::dot(vss::math::Vector(2U, maximum).view(),
                              vss::math::Vector(2U, 2.0).view()),
               vss::math::ErrorCode::NonFiniteResult);

  const vss::math::OperationOptions allow_non_finite{{}, false};
  const auto non_finite_sum =
      vss::math::add(nan_values.view(), nan_values.view(), allow_non_finite);
  VSS_CHECK(non_finite_sum.hasValue());
  VSS_CHECK(std::isnan(non_finite_sum.value().at(0U)));
  const auto non_finite_scale =
      vss::math::scale(positive_infinity_values.view(), 2.0, allow_non_finite);
  VSS_CHECK(non_finite_scale.hasValue());
  VSS_CHECK(std::isinf(non_finite_scale.value().at(0U)));
  const vss::math::Vector empty_vector;
  const auto empty_sum =
      vss::math::add(empty_vector.view(), empty_vector.view());
  VSS_CHECK(empty_sum.hasValue());
  VSS_CHECK(empty_sum.value().empty());
  const vss::math::Matrix zero_by_three(0U, 3U);
  const auto transposed = vss::math::transpose(zero_by_three.view());
  VSS_CHECK(transposed.hasValue());
  VSS_CHECK(transposed.value().rows() == 3U);
  VSS_CHECK(transposed.value().columns() == 0U);

  const std::span<vss::math::Scalar> empty_storage;
  checkFailure(vss::math::VectorView::create(empty_storage, 1U, 0U),
               vss::math::ErrorCode::InvalidExtent);
  const auto empty_stride_view =
      vss::math::VectorView::create(empty_storage, 0U, 0U);
  VSS_CHECK(empty_stride_view.hasValue());
  checkFailure(vss::math::VectorView::create(empty_storage, extent_max, 2U),
               vss::math::ErrorCode::SizeOverflow);

  checkFailure(vss::math::MatrixView::create(empty_storage, 1U, 1U, 0U, 1U),
               vss::math::ErrorCode::InvalidExtent);
  const auto empty_matrix_view =
      vss::math::MatrixView::create(empty_storage, 0U, 1U, 0U, 0U);
  VSS_CHECK(empty_matrix_view.hasValue());
  checkFailure(vss::math::MatrixView::create(empty_storage, extent_max, 1U,
                                             extent_max, 1U),
               vss::math::ErrorCode::SizeOverflow);
  checkFailure(vss::math::MatrixView::create(empty_storage, 2U, 2U, extent_max,
                                             extent_max),
               vss::math::ErrorCode::SizeOverflow);

  const auto invalid_negative = vss::math::validate({-1.0, 0.0});
  const auto invalid_nan = vss::math::validate({nan, 0.0});
  const auto invalid_infinity = vss::math::validate({0.0, infinity});
  VSS_CHECK(invalid_negative.code() == vss::math::ErrorCode::InvalidTolerance);
  VSS_CHECK(invalid_nan.code() == vss::math::ErrorCode::InvalidTolerance);
  VSS_CHECK(invalid_infinity.code() == vss::math::ErrorCode::InvalidTolerance);

  VSS_CHECK(vss::math::nearlyEqual(1.0, 1.0 + 5.0e-13, {1.0e-12, 0.0}));
  VSS_CHECK(vss::math::nearlyEqual(1.0e8, 1.0e8 + 5.0e-3, {0.0, 1.0e-10}));
  VSS_CHECK(vss::math::nearlyEqual(infinity, infinity));
  VSS_CHECK(!vss::math::nearlyEqual(infinity, -infinity));
  VSS_CHECK(!vss::math::nearlyEqual(nan, 0.0));
}
