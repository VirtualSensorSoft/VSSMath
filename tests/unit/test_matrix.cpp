#include "Test.hpp"

#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>
#include <vss/math/Core.hpp>
#include <vss/math/Matrix.hpp>

int main() {
  const vss::math::Matrix empty;
  VSS_CHECK(empty.rows() == 0U);
  VSS_CHECK(empty.columns() == 0U);
  VSS_CHECK(empty.empty());
  VSS_CHECK(empty.empty());

  const vss::math::Matrix zero_shape(0U, 0U);
  VSS_CHECK(zero_shape.rows() == 0U);
  VSS_CHECK(zero_shape.columns() == 0U);
  VSS_CHECK(zero_shape.empty());
  VSS_CHECK(zero_shape.empty());

  const vss::math::Matrix zero_rows(0U, 3U);
  VSS_CHECK(zero_rows.rows() == 0U);
  VSS_CHECK(zero_rows.columns() == 3U);
  VSS_CHECK(zero_rows.empty());
  VSS_CHECK(zero_rows.empty());

  const vss::math::Matrix zero_columns(2U, 0U);
  VSS_CHECK(zero_columns.rows() == 2U);
  VSS_CHECK(zero_columns.columns() == 0U);
  VSS_CHECK(zero_columns.empty());
  VSS_CHECK(zero_columns.empty());

  vss::math::Matrix filled(2U, 3U, -1.5);
  VSS_CHECK(filled.rows() == 2U);
  VSS_CHECK(filled.columns() == 3U);
  VSS_CHECK(filled.size() == 6U);
  VSS_CHECK(!filled.empty());
  VSS_CHECK(filled.at(0U, 0U) == -1.5);
  VSS_CHECK(filled.at(1U, 2U) == -1.5);

  vss::math::Matrix& matrix = filled;
  matrix.at(0U, 0U) = 1.0;
  matrix.at(0U, 1U) = 2.0;
  matrix.at(0U, 2U) = 3.0;
  matrix.at(1U, 0U) = 4.0;
  matrix.at(1U, 1U) = 5.0;
  matrix.at(1U, 2U) = 6.0;
  // Verifies contiguous row-major storage.
  // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  VSS_CHECK(matrix.data()[0] == 1.0);
  // Verifies contiguous row-major storage.
  // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  VSS_CHECK(matrix.data()[1] == 2.0);
  // Verifies contiguous row-major storage.
  // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  VSS_CHECK(matrix.data()[2] == 3.0);
  // Verifies contiguous row-major storage.
  // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  VSS_CHECK(matrix.data()[3] == 4.0);
  // Verifies contiguous row-major storage.
  // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  VSS_CHECK(matrix.data()[4] == 5.0);
  // Verifies contiguous row-major storage.
  // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  VSS_CHECK(matrix.data()[5] == 6.0);
  VSS_CHECK(matrix.at(0U, 0U) == 1.0);
  VSS_CHECK(matrix.at(0U, 2U) == 3.0);
  VSS_CHECK(matrix.at(1U, 0U) == 4.0);
  VSS_CHECK(matrix.at(1U, 2U) == 6.0);
  // Verifies mutable contiguous storage.
  // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  matrix.data()[4] = 9.0;
  VSS_CHECK(matrix.at(1U, 1U) == 9.0);
  const auto& const_matrix = matrix;
  VSS_CHECK(const_matrix.at(1U, 1U) == 9.0);
  matrix.fill(7.0);
  // Verifies contiguous row-major storage.
  // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  VSS_CHECK(matrix.data()[0] == 7.0);
  // Verifies contiguous row-major storage.
  // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  VSS_CHECK(matrix.data()[5] == 7.0);

  vss::math::Matrix copied = matrix;
  VSS_CHECK(copied.rows() == matrix.rows());
  VSS_CHECK(copied.columns() == matrix.columns());
  VSS_CHECK(copied.data() != matrix.data());
  copied.at(0U, 0U) = 8.0;
  VSS_CHECK(matrix.at(0U, 0U) == 7.0);

  vss::math::Matrix move_source(2U, 3U, 4.0);
  const vss::math::Matrix moved(std::move(move_source));
  VSS_CHECK(moved.rows() == 2U);
  VSS_CHECK(moved.columns() == 3U);
  VSS_CHECK(moved.size() == 6U);
  // Verifies the documented moved-from state.
  // NOLINTNEXTLINE(clang-analyzer-cplusplus.Move)
  VSS_CHECK(move_source.rows() == 0U);
  // Verifies the documented moved-from state.
  // NOLINTNEXTLINE(clang-analyzer-cplusplus.Move)
  VSS_CHECK(move_source.columns() == 0U);
  // Verifies the documented moved-from state.
  // NOLINTNEXTLINE(clang-analyzer-cplusplus.Move)
  VSS_CHECK(move_source.empty());
  // Moved-from matrices are documented as reusable.
  // NOLINTNEXTLINE(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
  move_source.fill(3.0);
  VSS_CHECK(move_source.empty());
  move_source = vss::math::Matrix(1U, 2U, 6.0);
  VSS_CHECK(move_source.rows() == 1U);
  VSS_CHECK(move_source.columns() == 2U);

  vss::math::Matrix assignment_source(2U, 3U, 5.0);
  vss::math::Matrix assignment_target(1U, 1U, 0.0);
  assignment_target = std::move(assignment_source);
  VSS_CHECK(assignment_target.rows() == 2U);
  VSS_CHECK(assignment_target.columns() == 3U);
  VSS_CHECK(assignment_target.at(1U, 2U) == 5.0);
  // Verifies the documented moved-from state.
  // NOLINTNEXTLINE(clang-analyzer-cplusplus.Move)
  VSS_CHECK(assignment_source.rows() == 0U);
  // Verifies the documented moved-from state.
  // NOLINTNEXTLINE(clang-analyzer-cplusplus.Move)
  VSS_CHECK(assignment_source.columns() == 0U);
  // Verifies the documented moved-from state.
  // NOLINTNEXTLINE(clang-analyzer-cplusplus.Move)
  VSS_CHECK(assignment_source.empty());
  // Moved-from matrices are documented as reusable.
  // NOLINTNEXTLINE(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
  assignment_source.fill(3.0);
  VSS_CHECK(assignment_source.empty());

  try {
    static_cast<void>(matrix.at(2U, 0U));
    VSS_CHECK(false);
  } catch (const std::out_of_range& exception) {
    const std::string message(exception.what());
    VSS_CHECK(message.find("IndexOutOfRange") != std::string::npos);
    VSS_CHECK(message.find("row") != std::string::npos);
    VSS_CHECK(message.find('2') != std::string::npos);
  }

  try {
    static_cast<void>(matrix.at(0U, 3U));
    VSS_CHECK(false);
  } catch (const std::out_of_range& exception) {
    const std::string message(exception.what());
    VSS_CHECK(message.find("IndexOutOfRange") != std::string::npos);
    VSS_CHECK(message.find("column") != std::string::npos);
    VSS_CHECK(message.find('3') != std::string::npos);
  }

  try {
    static_cast<void>(matrix.at(2U, 3U));
    VSS_CHECK(false);
  } catch (const std::out_of_range& exception) {
    const std::string message(exception.what());
    VSS_CHECK(message.find("row") != std::string::npos);
    VSS_CHECK(message.find("column") == std::string::npos);
  }

  try {
    const vss::math::Matrix overflow(
        std::numeric_limits<vss::math::Extent>::max() / 2U + 1U, 2U);
    VSS_CHECK(false);
  } catch (const std::length_error& exception) {
    const std::string message(exception.what());
    VSS_CHECK(message.find("SizeOverflow") != std::string::npos);
  } catch (const std::bad_alloc&) {
    VSS_CHECK(false);
  }
}
