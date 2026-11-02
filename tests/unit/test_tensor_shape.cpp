#include "Test.hpp"

#include <array>
#include <cstddef>
#include <limits>
#include <span>
#include <vector>
#include <vss/math/Error.hpp>
#include <vss/math/TensorShape.hpp>

int main() {
  using vss::math::ErrorCode;
  using vss::math::TensorShape;

  const std::array<std::size_t, 3> extents{2U, 3U, 4U};
  auto shape = TensorShape::create(extents);
  VSS_CHECK(shape.hasValue());
  VSS_CHECK(shape.value().rank() == 3U);
  VSS_CHECK(shape.value().extent(0U) == 2U);
  VSS_CHECK(shape.value().elementCount() == 24U);

  auto scalar = TensorShape::create(std::span<const std::size_t>{});
  VSS_CHECK(scalar.hasValue());
  VSS_CHECK(scalar.value().rank() == 0U);
  VSS_CHECK(scalar.value().elementCount() == 1U);

  const std::array<std::size_t, 2> empty_extent{0U, 9U};
  auto empty = TensorShape::create(empty_extent);
  VSS_CHECK(empty.hasValue());
  VSS_CHECK(empty.value().elementCount() == 0U);

  auto overflow = TensorShape::create(
      std::array<std::size_t, 2>{std::numeric_limits<std::size_t>::max(), 2U});
  VSS_CHECK(!overflow.hasValue());
  VSS_CHECK(overflow.status().code() == ErrorCode::SizeOverflow);

  std::vector<std::size_t> copied{2U, 3U};
  auto owned = TensorShape::create(copied);
  copied[0] = 99U;
  VSS_CHECK(owned.hasValue());
  VSS_CHECK(owned.value().extent(0U) == 2U);
}
