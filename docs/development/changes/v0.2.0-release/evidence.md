# VSSMath 0.2.0 source assessment and evidence

This packet assesses the proposed 0.2.0 development-source candidate at commit
`6def6b1fc8bfd39e11fdfb51dd23237d2ec22d37`. It is evidence for owner review,
not a publication attestation, stable-ABI promise, production acceptance, safety
certification, or claim of universal numerical correctness.

## Assessed Identity

The candidate adds owning N-dimensional tensors, checked tensor arithmetic and
contractions, typed first-through-fourth moments with validated multivariate
aggregation and derived queries, and bounded stream output for numerical and
moment records. Package version and installed-consumer checks identify 0.2.0.
The package remains C++20 and has no third-party numerical runtime dependency.

The immutable 0.1.0 and 0.1.1 tags are unchanged. Their historical release
statements remain specific to those versions; this packet does not rewrite their
records or claim that the new API existed in those artifacts.

## Canonical hosted CI

The exact candidate commit passed canonical hosted CI in [run
42910148842](https://github.com/VirtualSensorSoft/VSSMath/actions/runs/42910148842)
on the upstream `main` push. All 16 required jobs succeeded. The separate
`gcc-debug (fast fork/manual suite)` job was skipped by its event condition; the
full `gcc-debug` job succeeded. The workflow also completed Clang Debug,
GCC Release, GCC sanitizers, static analysis, coverage, thread-safety,
security, documentation and policy checks, plus the Debug/Release performance
contract and algorithm-performance-contract jobs.

The security job's success is limited to its configured scanners and does not
establish that the source has zero vulnerabilities. The scanner input limits
recorded in the [0.1.1 assessment](../../archive/changes/v0.1.1-release/evidence.md)
remain relevant.

## Controlled timing qualification

The controlled timing qualification ran against the exact source commit above
using host profile `performance-primary-local-linux-v1` and the approved GCC
Debug and Release profiles. Each configuration produced a candidate capture and
two verification captures. All three captures were stable and passed the
qualification's repeatability check; all 36 representative benchmark cases in
each capture passed their correctness checks. Both configuration result files
report `qualified_pass`.

The raw records, host fingerprint, and machine-specific metadata are retained
privately by the owner and are intentionally not committed or published. The
public record therefore does not disclose the CPU model, host fingerprint,
thermal/load trace, or per-case timing values. This is a qualification of one
controlled local host profile; it does not establish performance on other
hardware, real-time suitability, a cross-host regression threshold, or a
production-performance guarantee. No portable CI regression baseline is added
by this packet.

The qualification fulfills the controlled Debug/Release timing review requested
for a possible 0.2.0 development release. It does not change or retroactively
qualify the immutable 0.1.0 or 0.1.1 artifacts.

## Limitations

### Conformance status and release limits

`docs/development/conformance-record.tsv` remains the final rule-level record
for source commit `cf98e60530da9ce403741e38b894ae4989625aa7` (0.1.1). This packet
does not claim a new rule-by-rule conformance assessment for 0.2.0. The latest
CI and timing results do not establish ISO/IEC certification or industrial,
functional-safety, or production approval.

The proposed release remains a pre-1.0 development source release. The package
makes no stable ABI or long-term source-compatibility guarantee. Its supported
build and test scope is limited to the configurations evidenced by canonical CI
and the single local timing profile above; other operating systems, architectures,
standard libraries, and deployment environments are not established here.

This packet neither publishes nor installs the package. It grants no rights;
the repository's existing all-rights-reserved notice remains authoritative.
