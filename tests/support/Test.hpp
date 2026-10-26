#ifndef VSS_MATH_TEST_HPP
#define VSS_MATH_TEST_HPP

#include <cstdlib>
#include <iostream>
#include <utility>

namespace vss::math::test {

inline void check(bool condition, const char* expression, const char* file,
                  int line) {
  if (!condition) {
    std::cerr << file << ":" << line << ": check failed: " << expression
              << '\n';
    std::exit(EXIT_FAILURE);
  }
}

template <class Exception, class Callable>
inline void checkThrows(Callable&& callable, const char* expression,
                        const char* file, int line) {
  try {
    std::forward<Callable>(callable)();
  } catch (const Exception&) {
    return;
  } catch (...) {
    std::cerr << file << ":" << line
              << ": check failed: wrong exception type for " << expression
              << '\n';
    std::exit(EXIT_FAILURE);
  }
  std::cerr << file << ":" << line << ": check failed: no exception thrown for "
            << expression << '\n';
  std::exit(EXIT_FAILURE);
}

} // namespace vss::math::test

// clang-format off
// Keep the assertion macro spelling stable so the source-location capture and
// line-level analyzer suppression remain attached to the preprocessor line.
#define VSS_CHECK(expression) /* NOLINT(cppcoreguidelines-macro-usage): captures source location and preserves single evaluation for test diagnostics. */ \
  ::vss::math::test::check(static_cast<bool>(expression), #expression, \
                           __FILE__, __LINE__)

#define VSS_CHECK_THROWS_AS(statement, exception_type) /* NOLINT(cppcoreguidelines-macro-usage): captures source location and defers a statement for exception testing. */ \
  ::vss::math::test::checkThrows<exception_type>( \
      [&]() { static_cast<void>(statement); }, #statement, __FILE__, \
      __LINE__)
// clang-format on

#endif // VSS_MATH_TEST_HPP
