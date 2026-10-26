# VirtualSensorMath

This repository is being established as a standalone C++20 numerical library. The current public bootstrap contains documentation and repository policy files only; it is not yet buildable and does not yet contain the numerical implementation or an installable package.

The planned library scope is small dense numerical primitives implemented with the C++ standard library and no external numerical-library runtime dependency. The [architecture](docs/architecture.md), [user guide](docs/user-guide.md), and [numerical contract](docs/numerical-contract.md) describe the intended public design; they are not claims that the implementation is already present or qualified.

The implementation and its build/test instructions will be introduced in a separately reviewed pull request. Ubuntu 24.04 x86-64 is the planned initial CI target; platform qualification will be claimed only after the hosted acceptance workflow has completed.

## Rights

All rights are reserved. This repository does not grant a general license to copy, modify, distribute, or use its contents. Public visibility alone is not permission to reuse the code. See [LICENSE](LICENSE).
