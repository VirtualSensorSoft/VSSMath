#include "Test.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vss/math/Array.hpp>
#include <vss/math/Error.hpp>
#include <vss/math/Limits.hpp>

namespace {

void copiesValuesAndExposesOnlyAConstView() {
  constexpr std::array<std::uint8_t, 3> input{1U, 2U, 255U};
  const auto result = vss::math::Array<std::uint8_t>::copy(
      std::span<const std::uint8_t>{input},
      vss::math::Limits{.elements = 3U, .bytes = 3U, .evaluations = 0U});
  VSS_CHECK(result.hasValue());
  const auto copied = result.value().view();
  VSS_CHECK(copied.size() == input.size());
  VSS_CHECK(copied[0] == 1U);
  VSS_CHECK(copied[1] == 2U);
  VSS_CHECK(copied[2] == 255U);
}

void permitsEmptyValuesAtZeroLimit() {
  const auto result = vss::math::Array<std::size_t>::copy(
      {}, vss::math::Limits{.elements = 0U, .bytes = 0U, .evaluations = 0U});
  VSS_CHECK(result.hasValue());
  VSS_CHECK(result.value().view().empty());
}

void rejectsStorageAboveEitherLimit() {
  constexpr std::array<std::size_t, 2> input{3U, 7U};
  const auto elementLimit = vss::math::Array<std::size_t>::copy(
      input, vss::math::Limits{
                 .elements = 1U, .bytes = sizeof(input), .evaluations = 0U});
  const auto byteLimit = vss::math::Array<std::size_t>::copy(
      input, vss::math::Limits{.elements = input.size(),
                               .bytes = sizeof(std::size_t),
                               .evaluations = 0U});
  VSS_CHECK(!elementLimit.hasValue());
  VSS_CHECK(elementLimit.status().code() ==
            vss::math::ErrorCode::ResourceLimit);
  VSS_CHECK(!byteLimit.hasValue());
  VSS_CHECK(byteLimit.status().code() == vss::math::ErrorCode::ResourceLimit);
}

} // namespace

int main() {
  copiesValuesAndExposesOnlyAConstView();
  permitsEmptyValuesAtZeroLimit();
  rejectsStorageAboveEitherLimit();
}
