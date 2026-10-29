# Status and support

The 0.1.1 correction source is under review and has not been released or
installed. The current conformance record is therefore marked Under-review;
its prior 0.1.0 evidence remains historical and is not carried forward as an
assessment of 0.1.1. The immutable 0.1.0 release/install remains affected by the
LU validation defect corrected in the 0.1.1 source. No release, production,
ISO-certification, or qualified timing claim is made for 0.1.1.

The last completed supported-evidence profile is limited to the exact source
revision and configurations listed in the [0.1.0 development-release evidence](development/changes/1-vssmath-foundation/evidence.md).
Other operating systems, architectures, compiler/standard-library combinations,
and deployment environments require separate validation before being described
as supported. The captured profiles do not establish Clang with libc++.

The test suite covers documented numerical contracts and install consumption.
CI selections are configuration-specific; the reported selected-test counts do
not represent the full registered inventory. Tests do not prove universal
numerical correctness, safety for arbitrary domain-specific input, or performance
on unmeasured hardware. No qualified performance baseline is implied by the
current development-only timing exception.
