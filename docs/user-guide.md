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

## Tensor products

The checked tensor product API forms an outer product by concatenating shapes.
Contracted axes are supplied as paired zero-based lists; the result retains
uncontracted left axes first, followed by uncontracted right axes. An empty
pair of axis lists is equivalent to `outer`, and a rank-zero tensor behaves as
a scalar.

```cpp
#include <array>
#include <vss/math/Tensor.hpp>
#include <vss/math/TensorLimits.hpp>
#include <vss/math/TensorOperations.hpp>
#include <vss/math/TensorShape.hpp>

using namespace vss::math;
const std::array<Extent, 1> extents{2U};
const std::array<Scalar, 2> left_values{1.0, 2.0};
const std::array<Scalar, 2> right_values{3.0, 4.0};
const auto shape = TensorShape::create(extents);
if (!shape) return 1;
auto left = Tensor::create(shape.value(), 0.0, TensorLimits{16U});
auto right = Tensor::create(shape.value(), 0.0, TensorLimits{16U});
if (!left || !right) return 1;
for (Index i = 0U; i < extents[0]; ++i) {
  left.value().at(std::array<Index, 1>{i}) = left_values[i];
  right.value().at(std::array<Index, 1>{i}) = right_values[i];
}

const auto matrix = outer(left.value(), right.value(), TensorLimits{16U});
const std::array<Index, 1> axis{0U};
const auto dot = contract(left.value(), right.value(), axis, axis,
                          TensorLimits{1U});
if (!matrix || !dot) return 1;
// matrix has shape [2, 2] and values [3, 4, 6, 8]; dot is rank zero with 11.
```

Both operations return `Result<Tensor>`, honor the caller's element limit, and
reject incompatible or repeated contraction axes. By default, non-finite input
and output values are reported as checked failures.

## Moment wrappers

`FirstMoment`, `SecondMoment`, `ThirdMoment`, and `FourthMoment` own validated
copies of `Tensor` values with shapes `[d]`, `[d,d]`, `[d,d,d]`, and
`[d,d,d,d]`, respectively, where `d` is positive. The first moment represents
the mean; orders two through four represent central moments. Every element must
be finite, and orders two through four must be symmetric under all axis
permutations. This structural validation does not claim that the tensors are
positive semidefinite or that they define a realizable distribution. Each
wrapper exposes only a const reference to its owned tensor; mutating the input
tensor after creation does not change the wrapper.

```cpp
#include <array>
#include <vss/math/SecondMoment.hpp>
#include <vss/math/Tensor.hpp>
#include <vss/math/TensorLimits.hpp>
#include <vss/math/TensorShape.hpp>

using namespace vss::math;
const auto shape = TensorShape::create(std::array<Extent, 2>{2U, 2U});
if (!shape) return 1;
auto covariance = Tensor::create(shape.value(), 0.0, TensorLimits{4U});
if (!covariance) return 1;
covariance.value().at(std::array<Index, 2>{0U, 0U}) = 4.0;
covariance.value().at(std::array<Index, 2>{0U, 1U}) = 1.0;
covariance.value().at(std::array<Index, 2>{1U, 0U}) = 1.0;
covariance.value().at(std::array<Index, 2>{1U, 1U}) = 9.0;
const auto second = SecondMoment::create(covariance.value());
if (!second) return 1;
```

The wrappers represent present moment values, including an all-zero tensor.
Optional availability belongs to `MultivariateMoments`, which is introduced in
the following implementation step.
