#include "Test.hpp"

#include <limits>
#include <stdexcept>
#include <vss/math/Error.hpp>
#include <vss/math/Tolerance.hpp>

int main() {
  VSS_CHECK(vss::math::validate({}).ok());
  VSS_CHECK(vss::math::validate({0.0, 0.0}).ok());

  const auto invalid_absolute = vss::math::validate({-1.0e-12, 1.0e-10});
  VSS_CHECK(!invalid_absolute.ok());
  VSS_CHECK(invalid_absolute.code() == vss::math::ErrorCode::InvalidTolerance);

  const auto invalid_relative = vss::math::validate({1.0e-12, -1.0e-10});
  VSS_CHECK(!invalid_relative.ok());
  VSS_CHECK(invalid_relative.code() == vss::math::ErrorCode::InvalidTolerance);

  const auto invalid_nan =
      vss::math::validate({std::numeric_limits<double>::quiet_NaN(), 1.0e-10});
  VSS_CHECK(!invalid_nan.ok());
  VSS_CHECK(invalid_nan.code() == vss::math::ErrorCode::InvalidTolerance);

  const auto invalid_infinity =
      vss::math::validate({1.0e-12, std::numeric_limits<double>::infinity()});
  VSS_CHECK(!invalid_infinity.ok());
  VSS_CHECK(invalid_infinity.code() == vss::math::ErrorCode::InvalidTolerance);

  VSS_CHECK(vss::math::isFinite(0.0));
  VSS_CHECK(vss::math::isFinite(-42.5));
  VSS_CHECK(!vss::math::isFinite(std::numeric_limits<double>::quiet_NaN()));
  VSS_CHECK(!vss::math::isFinite(std::numeric_limits<double>::infinity()));
  VSS_CHECK(!vss::math::isFinite(-std::numeric_limits<double>::infinity()));

  VSS_CHECK(vss::math::nearlyEqual(1.0, 1.0));
  VSS_CHECK(vss::math::nearlyEqual(std::numeric_limits<double>::infinity(),
                                   std::numeric_limits<double>::infinity()));
  VSS_CHECK(!vss::math::nearlyEqual(std::numeric_limits<double>::infinity(),
                                    -std::numeric_limits<double>::infinity()));
  VSS_CHECK(
      !vss::math::nearlyEqual(std::numeric_limits<double>::quiet_NaN(), 1.0));
  VSS_CHECK(
      !vss::math::nearlyEqual(1.0, std::numeric_limits<double>::quiet_NaN()));

  VSS_CHECK(vss::math::nearlyEqual(0.0, 1.0e-12));
  VSS_CHECK(!vss::math::nearlyEqual(0.0, 1.01e-12));
  VSS_CHECK(vss::math::nearlyEqual(1.0e6, 1.0e6 + 1.0e-4));

  VSS_CHECK_THROWS_AS(vss::math::nearlyEqual(1.0, 1.0, {-1.0e-12, 1.0e-10}),
                      std::invalid_argument);
}
