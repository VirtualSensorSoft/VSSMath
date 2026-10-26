#include "Test.hpp"

#include <memory>
#include <stdexcept>
#include <utility>
#include <vss/math/Error.hpp>
#include <vss/math/Result.hpp>

int main() // NOLINT(bugprone-exception-escape): this test main intentionally
           // exercises exception-producing APIs through checkThrows.
{
  auto successful = vss::math::Result<int>::success(42);
  VSS_CHECK(successful.hasValue());
  VSS_CHECK(static_cast<bool>(successful));
  VSS_CHECK(successful.status().ok());
  VSS_CHECK(successful.value() == 42);

  auto failed = vss::math::Result<int>::failure(
      vss::math::ErrorCode::InvalidExtent, "invalid extent");
  VSS_CHECK(!failed.hasValue());
  VSS_CHECK(!static_cast<bool>(failed));
  VSS_CHECK(failed.status().code() == vss::math::ErrorCode::InvalidExtent);
  VSS_CHECK(failed.status().message() == "invalid extent");

  auto mutable_result = vss::math::Result<int>::success(42);
  mutable_result.value() = 7;
  VSS_CHECK(mutable_result.value() == 7);

  const auto& const_result = mutable_result;
  VSS_CHECK(const_result.value() == 7);

  auto pointer_result = vss::math::Result<std::unique_ptr<int>>::success(
      std::make_unique<int>(42));
  auto pointer = std::move(pointer_result).value();
  VSS_CHECK(pointer != nullptr);
  VSS_CHECK(*pointer == 42);

  auto failed_lvalue = vss::math::Result<int>::failure(
      vss::math::ErrorCode::InvalidExtent, "invalid extent");
  VSS_CHECK_THROWS_AS(failed_lvalue.value(), std::logic_error);

  const auto failed_const = vss::math::Result<int>::failure(
      vss::math::ErrorCode::InvalidExtent, "invalid extent");
  VSS_CHECK_THROWS_AS(failed_const.value(), std::logic_error);

  auto failed_rvalue = vss::math::Result<int>::failure(
      vss::math::ErrorCode::InvalidExtent, "invalid extent");
  VSS_CHECK_THROWS_AS(std::move(failed_rvalue).value(), std::logic_error);

  VSS_CHECK_THROWS_AS(
      vss::math::Result<int>::failure(vss::math::ErrorCode::None, "invalid"),
      std::invalid_argument);
  VSS_CHECK_THROWS_AS(
      vss::math::Result<int>::failure(vss::math::ErrorCode::InvalidExtent, ""),
      std::invalid_argument);
}
