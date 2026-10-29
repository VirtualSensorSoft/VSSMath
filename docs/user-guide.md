# User guide

## Build and install

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF
cmake --build build --parallel 2
cmake --install build --prefix "$PWD/install"
```

Consumers add the prefix to `CMAKE_PREFIX_PATH` and request the exact
development package with
`find_package(VirtualSensorMath 0.1.1 EXACT CONFIG REQUIRED)`, then link the
exported `VirtualSensorSoft::math` target. Before 1.0.0, the package version
file uses CMake's same-minor compatibility rule: a request for the same major
and minor that is not newer than the installed version is compatible. This
policy does not claim long-term ABI stability.

The current development evidence is limited to a specific Ubuntu 24.04 x86-64
source revision and remains under review. Check [status and support](status.md)
and the [numerical contract](numerical-contract.md) before using this package in
production or on an unlisted platform.

## Example

```cpp
#include <vss/math/Matrix.hpp>
#include <vss/math/Operations.hpp>
#include <vss/math/Vector.hpp>

int main() {
  using namespace vss::math;
  Matrix matrix(2U, 2U);
  matrix.at(0U, 0U) = 1.0;
  matrix.at(0U, 1U) = 2.0;
  matrix.at(1U, 0U) = 3.0;
  matrix.at(1U, 1U) = 4.0;
  Vector vector(2U);
  vector.at(0U) = 5.0;
  vector.at(1U) = 6.0;

  const auto result = multiply(matrix.view(), vector.view());
  if (!result) {
    return 1;
  }
  return result.value().at(0U) == 17.0 && result.value().at(1U) == 39.0 ? 0 : 1;
}
```

`Result<T>` owns the operation result on success. Check it before accessing
`value()`; a failed result has no value and carries a `Status` with an error
code and diagnostic message. Views borrow their source storage and must not
outlive or overlap mutation of that storage.

See the public headers and [numerical contract](numerical-contract.md) for
algorithm-specific behavior and limitations.
