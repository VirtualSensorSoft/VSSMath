# Status and support

The project is at version 0.1.0 and remains under active development. The
current evidence record assesses source commit
`20c8c9910a0f3ebf164bd513da6609cd59e8daeb` using canonical hosted CI run
[40579072074](https://github.com/VirtualSensorSoft/VSSMath/actions/runs/40579072074).
It covers Ubuntu 24.04.5 x86-64 with GCC 13.3.0 Debug/Release and Clang
18.1.3 Debug, plus the recorded sanitizer and policy jobs. The assessment is
Under-review, with two Pending rules and one approved development-only timing
deviation; it is not final conformance, production acceptance, or certification.

The supported-evidence profile is limited to the exact source revision and
configurations listed in the [development-release evidence](development/changes/1-vssmath-foundation/evidence.md).
Other operating systems, architectures, compiler/standard-library combinations,
and deployment environments require separate validation before being described
as supported. In particular, the Clang standard-library identity still needs to
be recorded.

The test suite covers documented numerical contracts and install consumption.
CI selections are configuration-specific; the reported selected-test counts do
not represent the full registered inventory. Tests do not prove universal
numerical correctness, safety for arbitrary domain-specific input, or performance
on unmeasured hardware. No qualified performance baseline is implied by the
current development-only timing exception.
