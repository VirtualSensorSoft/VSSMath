#include "Test.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <vss/math/Vector.hpp>

int main() {
  const vss::math::Vector empty;
  VSS_CHECK(empty.empty());
  VSS_CHECK(empty.empty());

  vss::math::Vector zeros(3U);
  VSS_CHECK(zeros.size() == 3U);
  VSS_CHECK(zeros.at(0U) == 0.0);
  VSS_CHECK(zeros.at(2U) == 0.0);

  vss::math::Vector filled(3U, -2.5);
  VSS_CHECK(filled.at(0U) == -2.5);
  VSS_CHECK(filled.at(2U) == -2.5);

  vss::math::Vector values(std::vector<double>{1.0, -2.0, 3.5});
  // Verifies contiguous storage.
  // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  VSS_CHECK(values.data() + 1 == &values.at(1U));
  // Verifies contiguous storage.
  // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  VSS_CHECK(values.data() + 2 == &values.at(2U));

  values.at(1U) = 4.0;
  const auto& const_values = values;
  VSS_CHECK(const_values.at(1U) == 4.0);

  values.fill(8.0);
  VSS_CHECK(values.at(0U) == 8.0);
  VSS_CHECK(values.at(1U) == 8.0);
  VSS_CHECK(values.at(2U) == 8.0);

  vss::math::Vector copy(std::vector<double>{1.0, -2.0, 3.5});
  vss::math::Vector copied = copy;
  VSS_CHECK(copied.size() == copy.size());
  VSS_CHECK(copied.data() != copy.data());
  copied.at(0U) = 9.0;
  VSS_CHECK(copy.at(0U) == 1.0);

  vss::math::Vector move_source(std::vector<double>{1.0, -2.0, 3.5});
  vss::math::Vector moved(std::move(move_source));
  VSS_CHECK(moved.size() == 3U);
  VSS_CHECK(moved.at(0U) == 1.0);
  VSS_CHECK(moved.at(2U) == 3.5);
  // Verifies the documented moved-from state.
  // NOLINTNEXTLINE(clang-analyzer-cplusplus.Move)
  VSS_CHECK(move_source.empty());
  VSS_CHECK(move_source.empty());
  // Moved-from vectors are documented as reusable.
  // NOLINTNEXTLINE(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
  move_source.fill(2.0);
  VSS_CHECK(move_source.empty());
  move_source = vss::math::Vector(2U, 6.0);
  VSS_CHECK(move_source.size() == 2U);

  vss::math::Vector assignment_source(std::vector<double>{1.0, -2.0, 3.5});
  vss::math::Vector assignment_target(1U, 0.0);
  assignment_target = std::move(assignment_source);
  VSS_CHECK(assignment_target.size() == 3U);
  VSS_CHECK(assignment_target.at(1U) == -2.0);
  // Verifies the documented moved-from state.
  // NOLINTNEXTLINE(clang-analyzer-cplusplus.Move)
  VSS_CHECK(assignment_source.empty());
  // Moved-from vectors are documented as reusable.
  // NOLINTNEXTLINE(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
  assignment_source.fill(3.0);
  VSS_CHECK(assignment_source.empty());

  VSS_CHECK(values.at(0U) == 8.0);
  VSS_CHECK(values.at(values.size() - 1U) == 8.0);

  try {
    static_cast<void>(values.at(values.size()));
    VSS_CHECK(false);
  } catch (const std::out_of_range& exception) {
    const std::string message(exception.what());
    VSS_CHECK(message.find("IndexOutOfRange") != std::string::npos);
    VSS_CHECK(message.find('3') != std::string::npos);
  }

  try {
    const auto& const_copy = copy;
    static_cast<void>(const_copy.at(const_copy.size()));
    VSS_CHECK(false);
  } catch (const std::out_of_range& exception) {
    const std::string message(exception.what());
    VSS_CHECK(message.find("IndexOutOfRange") != std::string::npos);
    VSS_CHECK(message.find('3') != std::string::npos);
  }

  const auto nan = std::numeric_limits<double>::quiet_NaN();
  const auto infinity = std::numeric_limits<double>::infinity();
  vss::math::Vector non_finite(2U, nan);
  non_finite.fill(infinity);
  VSS_CHECK(std::isinf(non_finite.at(0U)));
}
