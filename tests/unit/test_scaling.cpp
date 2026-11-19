#include "Test.hpp"

#include <array>
#include <limits>
#include <vss/math/Core.hpp>
#include <vss/math/Error.hpp>
#include <vss/math/Limits.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/Scaling.hpp>
#include <vss/math/Vector.hpp>

int main() {
  using namespace vss::math;
  const Limits ample{64, 4096, 64};

  const Vector values{std::array<Scalar, 2>{2, -3}};
  const Vector scale{std::array<Scalar, 2>{-2, 3}};
  const Vector offset{std::array<Scalar, 2>{1, 1}};
  const auto converted =
      convertLinearUnits(values.view(), scale.view(), offset.view(), ample);
  VSS_CHECK(converted.hasValue());
  VSS_CHECK(converted.value().at(0) == -3);
  VSS_CHECK(converted.value().at(1) == -8);

  const Vector zero_scale{std::array<Scalar, 2>{0, 0}};
  const Vector shifted{std::array<Scalar, 2>{7, 1}};
  const auto constant = convertLinearUnits(values.view(), zero_scale.view(),
                                           shifted.view(), ample);
  VSS_CHECK(constant.hasValue());
  VSS_CHECK(constant.value().at(0) == 7);
  VSS_CHECK(constant.value().at(1) == 1);
  VSS_CHECK(
      convertLinearUnits(values.view(), scale.view(), Vector(1).view(), ample)
          .status()
          .code() == ErrorCode::DimensionMismatch);
  const Vector infinite_scale{
      std::array<Scalar, 2>{1, std::numeric_limits<Scalar>::infinity()}};
  VSS_CHECK(convertLinearUnits(values.view(), infinite_scale.view(),
                               offset.view(), ample)
                .status()
                .code() == ErrorCode::NonFiniteInput);
  VSS_CHECK(convertLinearUnits(values.view(), scale.view(), offset.view(),
                               Limits{1, 4096, 64})
                .status()
                .code() == ErrorCode::ResourceLimit);

  Matrix covariance{2, 2};
  covariance.at(0, 0) = 4;
  covariance.at(0, 1) = 3;
  covariance.at(1, 0) = 3;
  covariance.at(1, 1) = 9;
  const auto transformed =
      scaleCovariance(covariance.view(), scale.view(), scale.view(), ample);
  VSS_CHECK(transformed.hasValue());
  VSS_CHECK(transformed.value().at(0, 0) == 16);
  VSS_CHECK(transformed.value().at(0, 1) == -18);
  VSS_CHECK(transformed.value().at(1, 0) == -18);
  VSS_CHECK(transformed.value().at(1, 1) == 81);

  const Vector right{std::array<Scalar, 2>{3, 4}};
  const auto cross_scaled =
      scaleCovariance(covariance.view(), scale.view(), right.view(), ample);
  VSS_CHECK(cross_scaled.hasValue());
  VSS_CHECK(cross_scaled.value().at(0, 0) == -24);
  VSS_CHECK(cross_scaled.value().at(0, 1) == -24);
  VSS_CHECK(cross_scaled.value().at(1, 0) == 27);
  VSS_CHECK(cross_scaled.value().at(1, 1) == 108);
  VSS_CHECK(
      scaleCovariance(covariance.view(), Vector(1).view(), scale.view(), ample)
          .status()
          .code() == ErrorCode::DimensionMismatch);
  VSS_CHECK(scaleCovariance(covariance.view(), scale.view(), scale.view(),
                            Limits{1, 4096, 64})
                .status()
                .code() == ErrorCode::ResourceLimit);

  Matrix rectangular{2, 3};
  rectangular.at(0, 0) = -1;
  rectangular.at(0, 1) = 2;
  rectangular.at(0, 2) = 3;
  rectangular.at(1, 0) = 2;
  rectangular.at(1, 1) = 1;
  rectangular.at(1, 2) = -3;
  const Vector row_scale{std::array<Scalar, 2>{-2, -2}};
  const Vector column_scale{std::array<Scalar, 3>{1, 0, 1}};
  const auto cross_covariance = scaleCovariance(
      rectangular.view(), row_scale.view(), column_scale.view(), ample);
  VSS_CHECK(cross_covariance.hasValue());
  const std::array<Scalar, 6> expected{2, 0, -6, -4, 0, 6};
  for (Index row = 0; row < 2; ++row) {
    for (Index column = 0; column < 3; ++column) {
      VSS_CHECK(cross_covariance.value().at(row, column) ==
                expected[row * 3 + column]);
    }
  }

  Matrix nonfinite{2, 2};
  nonfinite.at(0, 0) = std::numeric_limits<Scalar>::infinity();
  VSS_CHECK(scaleCovariance(nonfinite.view(), scale.view(), scale.view(), ample)
                .status()
                .code() == ErrorCode::NonFiniteInput);
  const Vector large{std::array<Scalar, 2>{1e308, 1e308}};
  VSS_CHECK(convertLinearUnits(large.view(), Vector(2, 2).view(), offset.view(),
                               ample)
                .status()
                .code() == ErrorCode::NonFiniteResult);
}
