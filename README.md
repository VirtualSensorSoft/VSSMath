# VirtualSensorMath

VirtualSensorMath is a standalone C++20 library of small dense numerical
primitives. It provides owning vectors and matrices, non-owning views,
arithmetic, descriptive statistics, linear factorizations and solves,
covariance operations, finite differences, and chi-square tail functions.

The implementation uses the C++ standard library and has no external
numerical-library runtime dependency. Ubuntu 24.04 x86-64 is the planned initial
qualification target; no platform qualification is claimed before hosted acceptance. See the
[architecture](docs/architecture.md), [user guide](docs/user-guide.md), and
[numerical contract](docs/numerical-contract.md) before relying on behavior.

## Build and test

Requirements: CMake 3.28 or later, a C++20 compiler, and (for the default API
documentation target) Doxygen with Graphviz `dot`.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTING=ON
cmake --build build --parallel 2
ctest --test-dir build --output-on-failure
```

The project version is `0.1.0`. The installed CMake package is
`VirtualSensorMath`, and the exported target is
`VirtualSensorSoft::math`.

## Rights

All rights are reserved. This repository does not grant a general license to
copy, modify, distribute, or use its contents. Public visibility alone is not
permission to reuse the code. See [LICENSE](LICENSE).
