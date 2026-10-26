#include "Test.hpp"

#include <array>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vss/math/Core.hpp>
#include <vss/math/Error.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/MatrixView.hpp>

static_assert(
    std::is_convertible_v<vss::math::MatrixView, vss::math::ConstMatrixView>);
static_assert(
    !std::is_convertible_v<vss::math::ConstMatrixView, vss::math::MatrixView>);

int main() // NOLINT(bugprone-exception-escape): this test main intentionally
           // exercises exception-producing APIs through checkThrows.
{
  std::array<double, 12> storage{0.0, 1.0, 2.0, 3.0, 4.0,  5.0,
                                 6.0, 7.0, 8.0, 9.0, 10.0, 11.0};
  const std::span<double> span(storage);

  auto contiguous = vss::math::MatrixView::create(span, 3U, 4U, 4U, 1U);
  VSS_CHECK(contiguous.hasValue());
  VSS_CHECK(contiguous.value().rows() == 3U);
  VSS_CHECK(contiguous.value().columns() == 4U);
  VSS_CHECK(contiguous.value().at(0U, 0U) == 0.0);
  VSS_CHECK(contiguous.value().at(1U, 2U) == 6.0);
  VSS_CHECK(contiguous.value().at(2U, 3U) == 11.0);

  auto short_rows = vss::math::MatrixView::create(span, 2U, 3U, 4U, 1U);
  VSS_CHECK(short_rows.hasValue());
  VSS_CHECK(short_rows.value().at(1U, 2U) == 6.0);

  auto column_stride = vss::math::MatrixView::create(span, 3U, 2U, 4U, 2U);
  VSS_CHECK(column_stride.hasValue());
  VSS_CHECK(column_stride.value().at(0U, 1U) == 2.0);
  VSS_CHECK(column_stride.value().at(2U, 1U) == 10.0);
  column_stride.value().at(1U, 1U) = 50.0;
  VSS_CHECK(storage[6] == 50.0);

  const auto const_view =
      static_cast<vss::math::ConstMatrixView>(contiguous.value());
  VSS_CHECK(const_view.at(1U, 1U) == storage[5]);

  auto empty_rows =
      vss::math::MatrixView::create(std::span<double>{}, 0U, 4U, 0U, 0U);
  VSS_CHECK(empty_rows.hasValue());
  VSS_CHECK(empty_rows.value().empty());
  auto empty_columns =
      vss::math::MatrixView::create(std::span<double>{}, 3U, 0U, 0U, 0U);
  VSS_CHECK(empty_columns.hasValue());
  VSS_CHECK(empty_columns.value().empty());

  const auto invalid_row_stride =
      vss::math::MatrixView::create(span, 2U, 3U, 0U, 1U);
  VSS_CHECK(!invalid_row_stride.hasValue());
  VSS_CHECK(invalid_row_stride.status().code() ==
            vss::math::ErrorCode::InvalidExtent);
  const auto invalid_column_stride =
      vss::math::MatrixView::create(span, 2U, 3U, 4U, 0U);
  VSS_CHECK(!invalid_column_stride.hasValue());
  VSS_CHECK(invalid_column_stride.status().code() ==
            vss::math::ErrorCode::InvalidExtent);

  std::array<double, 11> insufficient_storage{};
  const auto insufficient = vss::math::MatrixView::create(
      std::span<double>(insufficient_storage), 3U, 4U, 4U, 1U);
  VSS_CHECK(!insufficient.hasValue());
  VSS_CHECK(insufficient.status().code() ==
            vss::math::ErrorCode::InvalidExtent);

  const auto overflow = vss::math::MatrixView::create(
      std::span<double>{}, 2U, 2U,
      std::numeric_limits<vss::math::Extent>::max(),
      std::numeric_limits<vss::math::Extent>::max());
  VSS_CHECK(!overflow.hasValue());
  VSS_CHECK(overflow.status().code() == vss::math::ErrorCode::SizeOverflow);

  try {
    static_cast<void>(contiguous.value().at(3U, 4U));
    VSS_CHECK(false);
  } catch (const std::out_of_range& exception) {
    const std::string message(exception.what());
    VSS_CHECK(message.find("IndexOutOfRange") != std::string::npos);
    VSS_CHECK(message.find("row") != std::string::npos);
    VSS_CHECK(message.find("column") == std::string::npos);
  }

  auto subview = contiguous.value().subview(1U, 1U, 2U, 3U);
  VSS_CHECK(subview.hasValue());
  VSS_CHECK(subview.value().at(0U, 0U) == 5.0);
  VSS_CHECK(subview.value().at(0U, 2U) == 7.0);
  VSS_CHECK(subview.value().at(1U, 0U) == 9.0);
  VSS_CHECK(subview.value().at(1U, 2U) == 11.0);
  subview.value().at(0U, 0U) = 55.0;
  VSS_CHECK(storage[5] == 55.0);

  auto empty_subview = contiguous.value().subview(3U, 4U, 0U, 0U);
  VSS_CHECK(empty_subview.hasValue());
  VSS_CHECK(empty_subview.value().empty());
  const auto invalid_origin = contiguous.value().subview(4U, 0U, 0U, 0U);
  VSS_CHECK(!invalid_origin.hasValue());
  const auto invalid_extent = contiguous.value().subview(2U, 3U, 2U, 1U);
  VSS_CHECK(!invalid_extent.hasValue());

  vss::math::Matrix owner(3U, 4U, 1.0);
  auto owner_view = owner.view();
  VSS_CHECK(owner_view.rows() == 3U);
  VSS_CHECK(owner_view.columns() == 4U);
  VSS_CHECK(owner_view.rowStride() == 4U);
  VSS_CHECK(owner_view.columnStride() == 1U);
  owner_view.at(1U, 2U) = 8.0;
  VSS_CHECK(owner.at(1U, 2U) == 8.0);
  const vss::math::Matrix const_owner(3U, 4U, 2.0);
  const auto const_owner_view = const_owner.view();
  VSS_CHECK(const_owner_view.rows() == 3U);
  VSS_CHECK(const_owner_view.at(2U, 3U) == 2.0);

  const vss::math::Matrix zero_rows_owner(0U, 4U);
  const auto zero_rows_view = zero_rows_owner.view();
  VSS_CHECK(zero_rows_view.empty());
  const vss::math::Matrix zero_columns_owner(3U, 0U);
  const auto zero_columns_view = zero_columns_owner.view();
  VSS_CHECK(zero_columns_view.empty());
}
