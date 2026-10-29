#include "Test.hpp"

#include <vss/math/version.hpp>

int main() {
  VSS_CHECK(vss::math::version::major == 0U);
  VSS_CHECK(vss::math::version::minor == 1U);
  VSS_CHECK(vss::math::version::patch == 1U);
  VSS_CHECK(vss::math::version::string == "0.1.1");
}
