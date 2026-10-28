# Status and support

The project is at version 0.1.0 and remains under active development. The
current evidence record assesses source commit
`3d00dd8ddd7a443ce46fba9f7ccb125245443593` using hosted CI run
[40753766010](https://github.com/VirtualSensorSoft/VSSMath/actions/runs/40753766010).
The tested pull-request candidate and final source have identical Git trees,
and the numerical `src/` and public runtime `include/` trees remain byte-identical
to release source `63a26726f28bd903c7dcaa1d7560c570912010cd`. The assessment
covers Ubuntu 24.04.5 x86-64 with GCC 13.3.0 Debug/Release and ASan/UBSan, and
Clang 18.1.3 Debug; the captured standard-library identity is libstdc++ for
both compiler families. The record is Final for this bounded engineering
assessment: 82 rules Passed and one approved development-only timing deviation.
It is not ISO certification, production acceptance, or a qualified timing claim.

The supported-evidence profile is limited to the exact source revision and
configurations listed in the [development-release evidence](development/changes/1-vssmath-foundation/evidence.md).
Other operating systems, architectures, compiler/standard-library combinations,
and deployment environments require separate validation before being described
as supported. The captured profiles do not establish Clang with libc++.

The test suite covers documented numerical contracts and install consumption.
CI selections are configuration-specific; the reported selected-test counts do
not represent the full registered inventory. Tests do not prove universal
numerical correctness, safety for arbitrary domain-specific input, or performance
on unmeasured hardware. No qualified performance baseline is implied by the
current development-only timing exception.
