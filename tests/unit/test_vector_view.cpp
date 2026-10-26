#include "Test.hpp"

#include <array>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vss/math/Core.hpp>
#include <vss/math/Error.hpp>
#include <vss/math/Vector.hpp>
#include <vss/math/VectorView.hpp>

static_assert(
    std::is_convertible_v<vss::math::VectorView, vss::math::ConstVectorView>);
static_assert(
    !std::is_convertible_v<vss::math::ConstVectorView, vss::math::VectorView>);

int main() // NOLINT(bugprone-exception-escape): this test main intentionally
           // exercises exception-producing APIs through checkThrows.
{
  std::array<double, 6> storage{0.0, 1.0, 2.0, 3.0, 4.0, 5.0};
  const std::span<double> span(storage);

  auto contiguous = vss::math::VectorView::create(span, 6U, 1U);
  VSS_CHECK(contiguous.hasValue());
  VSS_CHECK(contiguous.value().size() == 6U);
  VSS_CHECK(contiguous.value().stride() == 1U);
  VSS_CHECK(contiguous.value().at(0U) == 0.0);
  VSS_CHECK(contiguous.value().at(5U) == 5.0);

  auto strided = vss::math::VectorView::create(span, 3U, 2U);
  VSS_CHECK(strided.hasValue());
  VSS_CHECK(strided.value().at(0U) == 0.0);
  VSS_CHECK(strided.value().at(1U) == 2.0);
  VSS_CHECK(strided.value().at(2U) == 4.0);
  strided.value().at(1U) = 20.0;
  VSS_CHECK(storage[2] == 20.0);

  const auto const_view =
      static_cast<vss::math::ConstVectorView>(contiguous.value());
  VSS_CHECK(const_view.at(2U) == storage[2]);

  auto empty = vss::math::VectorView::create(std::span<double>{}, 0U, 1U);
  VSS_CHECK(empty.hasValue());
  VSS_CHECK(empty.value().empty());

  const auto invalid_empty =
      vss::math::VectorView::create(std::span<double>{}, 1U, 1U);
  VSS_CHECK(!invalid_empty.hasValue());
  VSS_CHECK(invalid_empty.status().code() ==
            vss::math::ErrorCode::InvalidExtent);

  const auto invalid_stride = vss::math::VectorView::create(span, 3U, 0U);
  VSS_CHECK(!invalid_stride.hasValue());
  VSS_CHECK(invalid_stride.status().code() ==
            vss::math::ErrorCode::InvalidExtent);

  std::array<double, 5> short_storage{};
  const auto insufficient =
      vss::math::VectorView::create(std::span<double>(short_storage), 3U, 3U);
  VSS_CHECK(!insufficient.hasValue());
  VSS_CHECK(insufficient.status().code() ==
            vss::math::ErrorCode::InvalidExtent);

  const auto overflow = vss::math::VectorView::create(
      std::span<double>{}, 3U, std::numeric_limits<vss::math::Extent>::max());
  VSS_CHECK(!overflow.hasValue());
  VSS_CHECK(overflow.status().code() == vss::math::ErrorCode::SizeOverflow);

  try {
    static_cast<void>(contiguous.value().at(6U));
    VSS_CHECK(false);
  } catch (const std::out_of_range& exception) {
    const std::string message(exception.what());
    VSS_CHECK(message.find("IndexOutOfRange") != std::string::npos);
  }

  auto middle = contiguous.value().subview(2U, 3U);
  VSS_CHECK(middle.hasValue());
  VSS_CHECK(middle.value().at(0U) == storage[2]);
  VSS_CHECK(middle.value().at(1U) == storage[3]);
  VSS_CHECK(middle.value().at(2U) == storage[4]);
  middle.value().at(0U) = 30.0;
  VSS_CHECK(storage[2] == 30.0);

  auto boundary = contiguous.value().subview(6U, 0U);
  VSS_CHECK(boundary.hasValue());
  VSS_CHECK(boundary.value().empty());
  const auto invalid_origin = contiguous.value().subview(7U, 0U);
  VSS_CHECK(!invalid_origin.hasValue());
  VSS_CHECK(invalid_origin.status().code() ==
            vss::math::ErrorCode::InvalidExtent);
  const auto invalid_range = contiguous.value().subview(5U, 2U);
  VSS_CHECK(!invalid_range.hasValue());
  VSS_CHECK(invalid_range.status().code() ==
            vss::math::ErrorCode::InvalidExtent);

  auto strided_subview = strided.value().subview(1U, 2U);
  VSS_CHECK(strided_subview.hasValue());
  VSS_CHECK(strided_subview.value().stride() == 2U);
  VSS_CHECK(strided_subview.value().at(0U) == storage[2]);
  VSS_CHECK(strided_subview.value().at(1U) == storage[4]);

  vss::math::Vector owner(3U, 1.0);
  auto owner_view = owner.view();
  VSS_CHECK(owner_view.size() == 3U);
  owner_view.at(1U) = 4.0;
  VSS_CHECK(owner.at(1U) == 4.0);
  const vss::math::Vector const_owner(3U, 2.0);
  const auto const_owner_view = const_owner.view();
  VSS_CHECK(const_owner_view.size() == 3U);
  VSS_CHECK(const_owner_view.at(2U) == 2.0);
}
