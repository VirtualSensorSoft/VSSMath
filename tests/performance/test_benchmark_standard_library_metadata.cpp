#include "benchmark_metadata.hpp"

#include <string_view>

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage): required stringification.
#define VSS_MATH_STRINGIFY_IMPL(value) #value
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage): expands version macro.
#define VSS_MATH_STRINGIFY(value) VSS_MATH_STRINGIFY_IMPL(value)

#if defined(__GLIBCXX__)
static_assert(vss::math::benchmark_metadata::stdlibId == "libstdc++");
static_assert(vss::math::benchmark_metadata::stdlibVersion ==
              VSS_MATH_STRINGIFY(__GLIBCXX__));
#elif defined(_LIBCPP_VERSION)
static_assert(vss::math::benchmark_metadata::stdlibId == "libc++");
static_assert(vss::math::benchmark_metadata::stdlibVersion ==
              VSS_MATH_STRINGIFY(_LIBCPP_VERSION));
#else
#error "Unsupported C++ standard library metadata"
#endif

#undef VSS_MATH_STRINGIFY
#undef VSS_MATH_STRINGIFY_IMPL

int main() {}
