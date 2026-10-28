# VirtualSensorMath 0.1.0 development source release

This is a source-only development release of VirtualSensorMath. It is intended
for early consumer integration and evaluation. The project remains under active
development; this release does not claim production acceptance, final
conformance, certification, or a qualified performance baseline. See [status
and support](../../../status.md) and the [numerical
contract](../../../numerical-contract.md) before relying on behavior.

## Included

VirtualSensorMath is a standalone C++20 library of small dense numerical
primitives. Version 0.1.0 includes owning vectors and matrices, non-owning
views, arithmetic and descriptive statistics, linear factorizations and solves,
covariance operations, finite differences, and chi-square tail functions. The
shipped library has no third-party numerical-library runtime dependency.

## Build and consume

Build from the source archive using CMake 3.28 or later and a C++20 compiler.
The installed CMake package is `VirtualSensorMath`; consumers link
`VirtualSensorSoft::math`. The package version file applies the documented
same-minor compatibility rule before 1.0.0. This does not promise ABI stability
or long-term source compatibility.

The current hosted evidence is limited to Ubuntu 24.04 x86-64 and the exact
compiler/configuration combinations recorded in the [development-release
evidence](evidence.md). Other operating systems, architectures, compilers,
standard libraries, and deployment environments are not established as
supported by this release.

## Validation and limitations

The source revision has hosted Debug, Release, sanitizer, analysis, security,
coverage, documentation, and policy evidence. Test counts are job-specific
selections; they are not a claim that every test was run in every configuration.
The conformance assessment remains under review, with two criteria pending and
one approved development-only timing exception. The timing diagnostics do not
establish a qualified performance baseline. See the linked evidence and
[development-only timing exception](../../deviations/development-timing.md) for
scope and review conditions.

## Rights

All rights are reserved. Public availability of this repository does not grant
permission to copy, modify, distribute, or use the source. See [LICENSE](../../../../LICENSE).
