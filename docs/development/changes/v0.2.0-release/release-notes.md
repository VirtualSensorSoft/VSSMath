# Proposed VSSMath 0.2.0 development-source release notes

**Status:** proposed notes for owner review; not yet published.

VSSMath 0.2.0 is a pre-1.0 development source candidate for early integration
and evaluation. The project remains under active development. This is not a
stable, production, safety-certified, or universally validated release.

## Included

- Owning N-dimensional tensor storage with checked shape limits and arithmetic.
- Tensor products and contractions with dimension validation.
- Typed first-through-fourth moments and validated multivariate aggregation
  with derived statistical queries.
- Bounded stream output for numerical containers and moment records.
- CMake package version `0.2.0`, retaining package `VirtualSensorMath` and
  consumer target `VirtualSensorSoft::math`.

## Build and consume

Build from source with CMake 3.28 or later and a C++20 compiler. See the
[user guide](../../../user-guide.md) for the installed-package workflow and the
[status page](../../../status.md) for the tested scope and current support limits.

## Verification and limitations

The candidate's canonical hosted CI and controlled Debug/Release local timing
qualification are summarized in the [0.2.0 source assessment](evidence.md).
Raw local timing records are not distributed. The qualification is specific to
one controlled host profile and does not guarantee performance on other hosts.
No production, real-time, safety, or certification claim is made. No stable ABI
or long-term source-compatibility promise is made before 1.0.0.

## Rights

All rights are reserved. Public availability of the repository does not grant
permission to copy, modify, distribute, or use the source. See the repository
[LICENSE](../../../../LICENSE).
