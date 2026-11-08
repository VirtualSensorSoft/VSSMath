#include "Test.hpp"

#include <vss/math/version.hpp>

int main() {
  VSS_CHECK(vss::math::version::major == 0U);
  VSS_CHECK(vss::math::version::minor == 2U);
  VSS_CHECK(vss::math::version::patch == 0U);
  VSS_CHECK(vss::math::version::string == "0.2.0");
}
