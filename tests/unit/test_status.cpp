#include "Test.hpp"

#include <stdexcept>
#include <string>
#include <vss/math/Error.hpp>

int main() {
  const auto success = vss::math::Status::success();
  VSS_CHECK(success.ok());
  VSS_CHECK(success.code() == vss::math::ErrorCode::None);
  VSS_CHECK(success.message().empty());

  const auto failure = vss::math::Status::failure(
      vss::math::ErrorCode::DimensionMismatch, "shape mismatch");
  VSS_CHECK(!failure.ok());
  VSS_CHECK(failure.code() == vss::math::ErrorCode::DimensionMismatch);
  VSS_CHECK(failure.message() == "shape mismatch");

  VSS_CHECK_THROWS_AS(
      vss::math::Status::failure(vss::math::ErrorCode::None, "invalid"),
      std::invalid_argument);
  VSS_CHECK_THROWS_AS(
      vss::math::Status::failure(vss::math::ErrorCode::DimensionMismatch, ""),
      std::invalid_argument);
}
