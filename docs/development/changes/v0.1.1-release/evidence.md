# VSSMath 0.1.1 source assessment and evidence

This is the final engineering assessment for the merged 0.1.1 source candidate at commit
`cf98e60530da9ce403741e38b894ae4989625aa7`. It is not a release attestation, installed-artifact
record, production acceptance, timing qualification, or ISO certification. No 0.1.1 release or
installation has been published.

## Assessed Identity

- Canonical source commit: `cf98e60530da9ce403741e38b894ae4989625aa7` (merge of owner-reviewed PR
#22).
- Source tree: `f6a1a5e3b825ab8a9510619a832eef70eb113939`.
- Canonical hosted CI: [run
40851759772](https://github.com/VirtualSensorSoft/VSSMath/actions/runs/40851759772), push to `main`,
exact source SHA above; conclusion **success**.
- All 16 required jobs succeeded. The event-gated `gcc-debug (fast fork/manual suite)` job was
skipped; the full `gcc-debug` job succeeded. The skip is by workflow condition, not a failed or
incomplete required check.
- Assessment scope: source-only VSSMath 0.1.1 candidate on Ubuntu 24.04.5 x86-64. The prior 0.1.0
record remains historical and is not treated as an assessment of this source.

## Source boundary and evidence reuse

Compared with the prior assessed source `3d00dd8ddd7a443ce46fba9f7ccb125245443593`, the source-level
runtime change is confined to `src/linear/LU.cpp`; its validation/error-precedence regressions are
in `tests/algorithms/test_lu.cpp`. Package/version and installed-consumer verification changed in
CMake metadata, package probes and consumer tests. Small CI/documentation-policy changes are
included and were exercised by the canonical workflow. Public headers and function signatures did
not change. The LU behavior, error precedence, package compatibility and installation checks were
reassessed at `cf98e60530da9ce403741e38b894ae4989625aa7`; evidence for unchanged contracts and scope
is reused from the prior public assessment, whose boundaries are identified in the rule groups
below. The earlier statement that runtime source was byte-identical to 0.1.0 is not carried forward.

The 95-row numerical requirement map is retained. `NUM-GEN-009` still maps validation-order evidence
to `math.lu`, `math.algorithm_lifecycle`, and `math.algorithm_failures`; all requirement IDs are
preserved. The prior 0.1.0 evidence file is retained as historical evidence, not rewritten as if it
described 0.1.1.

## Hosted build and test evidence

Run 40851759772 reports the following selected CTest profiles; each count applies only to that job's
configured selection:

| Job/profile | Result |
|---|---:|
| GCC 13.3.0 Debug | 44/44 passed; 145.26 s |
| GCC 13.3.0 Release | 44/44 passed; 84.30 s |
| Clang 18.1.3 Debug | 44/44 passed; 111.96 s |
| GCC 13.3.0 ASan/UBSan | 43/43 passed; 113.78 s |
| Static analysis | 48/48 passed; 165.80 s |
| Coverage | 43/43 passed; 124.20 s; 2,664/3,179 lines, 83% |
| Thread safety | `math.algorithm_lifecycle`, 1/1 passed under ThreadSanitizer; 0.13 s |
| Shared extended policy | Tests #33, #35 and #38: 3/3 passed; 646.68 s total |
| Performance contract, Debug | 6/6 passed; 239.73 s |
| Performance contract, Release | 6/6 passed; 251.25 s |

The toolchain captures identify Ubuntu 24.04.5 LTS x86-64, CMake 3.31.6, Unix Makefiles, C++20
(`__cplusplus=202002L`), GCC 13.3.0 with libstdc++ release 13 (`__GLIBCXX__=20240904`, C++11 ABI),
and Clang 18.1.3 with libstdc++ release 14 (`__GLIBCXX__=20240908`, C++11 ABI). This does not
establish Clang with libc++ or any other platform.

The installed-package verifier exercised eight version requests: exact 0.1.1 and compatible 0.1,
0.1.0 succeeded; exact 0.1.0, 0.0.9, 0.1.2 (compatible and exact), and 0.2.0 were rejected. Existing
package-origin, missing/foreign prefix, and installed-consumer checks remain in the successful build
workflow.

## Diagnostics and security limits

The Debug and Release algorithm diagnostic artifacts both report `diagnostic_comparison_pass` /
`compatible_pass` for their reviewed diagnostic comparisons. These are diagnostic comparisons, not a
controlled timing baseline or a qualified performance claim. The approved, version-limited
`ENG-CPP-076` deviation remains recorded in
[development-timing-0.1.1.md](../../deviations/development-timing-0.1.1.md); it waives no
correctness, sanitizer, allocation, thread-safety, analysis, coverage, security-governance, or CI
requirements.

The `security` CI job succeeded with Trivy 0.69.3 against both the source tree and installed
package. Its logs report zero language-specific files/config files for
vulnerability/misconfiguration scanning and warn that those scanners had no supported inputs. The
secret scanner reports no issues for the scanned inputs. Therefore this run does **not** establish
zero vulnerabilities or misconfigurations; the current library has no shipped third-party numerical
runtime dependency or deployable OS image. Reassess scanning if a package manifest, external runtime
dependency, or deployable image is introduced.

## Rule-level evidence

The table gives the same 83 rule IDs as the public engineering catalogue. Group-level evidence is
summarized in the sections below; scope exclusions use [profile
applicability](../../standards/profiles.md). Security-governance process evidence is intentionally
described without private operational details.

| Rule | Assessment | Evidence group |
|---|---|---|
| `ENG-CPP-001` | Passed | LANGUAGE |
| `ENG-CPP-002` | Passed | LANGUAGE |
| `ENG-CPP-003` | Passed | OWNERSHIP |
| `ENG-CPP-004` | Passed | OWNERSHIP |
| `ENG-CPP-005` | Passed | OWNERSHIP |
| `ENG-CPP-006` | Passed | SCOPE |
| `ENG-CPP-007` | Passed | OWNERSHIP |
| `ENG-CPP-008` | Passed | OWNERSHIP |
| `ENG-CPP-009` | Passed | OWNERSHIP |
| `ENG-CPP-010` | Passed | OWNERSHIP |
| `ENG-CPP-011` | Passed | OWNERSHIP |
| `ENG-CPP-012` | Passed | SCOPE |
| `ENG-CPP-013` | Passed | API-ERRORS |
| `ENG-CPP-014` | Passed | API-ERRORS |
| `ENG-CPP-015` | Passed | API-ERRORS |
| `ENG-CPP-016` | Passed | API-ERRORS |
| `ENG-CPP-017` | Passed | API-ERRORS |
| `ENG-CPP-018` | Passed | API-ERRORS |
| `ENG-CPP-019` | Passed | API-ERRORS |
| `ENG-CPP-020` | Passed | API-ERRORS |
| `ENG-CPP-021` | Passed | API-ERRORS |
| `ENG-CPP-022` | Passed | API-ERRORS |
| `ENG-CPP-023` | Passed | SCOPE |
| `ENG-CPP-024` | Passed | SCOPE |
| `ENG-CPP-025` | Passed | API-ERRORS |
| `ENG-CPP-026` | Passed | API-ERRORS |
| `ENG-CPP-027` | Passed | DETERMINISM |
| `ENG-CPP-028` | Passed | DETERMINISM |
| `ENG-CPP-029` | Passed | DETERMINISM |
| `ENG-CPP-030` | Passed | DETERMINISM |
| `ENG-CPP-031` | Passed | DETERMINISM |
| `ENG-CPP-032` | Passed | PERFORMANCE |
| `ENG-CPP-033` | Passed | PERFORMANCE |
| `ENG-CPP-034` | Passed | PERFORMANCE |
| `ENG-CPP-035` | Passed | PERFORMANCE |
| `ENG-CPP-036` | Passed | BUILD |
| `ENG-CPP-037` | Passed | BUILD |
| `ENG-CPP-038` | Passed | BUILD |
| `ENG-CPP-039` | Passed | BUILD |
| `ENG-CPP-040` | Passed | BUILD |
| `ENG-CPP-041` | Passed | BUILD |
| `ENG-CPP-042` | Passed | TESTS |
| `ENG-CPP-043` | Passed | TESTS |
| `ENG-CPP-044` | Passed | TESTS |
| `ENG-CPP-045` | Passed | TESTS |
| `ENG-CPP-046` | Passed | TESTS |
| `ENG-CPP-047` | Passed | TESTS |
| `ENG-CPP-048` | Passed | TESTS |
| `ENG-CPP-049` | Passed | TESTS |
| `ENG-CPP-050` | Passed | TESTS |
| `ENG-CPP-051` | Passed | TESTS |
| `ENG-CPP-052` | Passed | DOCUMENTATION |
| `ENG-CPP-053` | Passed | DOCUMENTATION |
| `ENG-CPP-054` | Passed | DOCUMENTATION |
| `ENG-CPP-055` | Passed | SCOPE |
| `ENG-CPP-056` | Passed | SECURITY |
| `ENG-CPP-057` | Passed | SECURITY |
| `ENG-CPP-058` | Passed | SECURITY |
| `ENG-CPP-059` | Passed | SECURITY |
| `ENG-CPP-060` | Passed | SECURITY |
| `ENG-CPP-061` | Passed | SCOPE |
| `ENG-CPP-062` | Passed | SCOPE |
| `ENG-CPP-063` | Passed | SECURITY |
| `ENG-CPP-064` | Passed | SECURITY-GOVERNANCE |
| `ENG-CPP-065` | Passed | NUMERICAL |
| `ENG-CPP-066` | Passed | NUMERICAL |
| `ENG-CPP-067` | Passed | NUMERICAL |
| `ENG-CPP-068` | Passed | NUMERICAL |
| `ENG-CPP-069` | Passed | NUMERICAL |
| `ENG-CPP-070` | Passed | NUMERICAL |
| `ENG-CPP-071` | Passed | NUMERICAL |
| `ENG-CPP-072` | Passed | NUMERICAL |
| `ENG-CPP-073` | Passed | NUMERICAL |
| `ENG-CPP-074` | Passed | NUMERICAL |
| `ENG-CPP-075` | Passed | NUMERICAL |
| `ENG-CPP-076` | Approved deviation | TIMING |
| `ENG-CON-001` | Passed | CONCURRENCY |
| `ENG-CON-002` | Passed | CONCURRENCY |
| `ENG-CON-006` | Passed | CONCURRENCY |
| `ENG-CON-007` | Passed | CONCURRENCY |
| `ENG-CON-009` | Passed | CONCURRENCY |
| `ENG-DATA-002` | Passed | DATA |
| `ENG-DATA-003` | Passed | DATA |

### Evidence group interpretation

- **LANGUAGE, OWNERSHIP, API-ERRORS, DETERMINISM:** public rule definitions, headers, source, and
unchanged-contract evidence from the prior assessment are reused only where the contract/source
boundary is unchanged. LU error precedence and invalid-input behavior were reassessed with
`math.lu`, `math.algorithm_failures`, `math.algorithm_error_precedence`, and the installed consumer.
- **SCOPE:** applicability and exclusions are limited to the implemented static-library API and
the profile triggers documented in `standards/profiles.md`.
- **PERFORMANCE, TIMING:** hosted Debug/Release diagnostic jobs and the approved 0.1.1 exception;
no qualified baseline is claimed.
- **BUILD:** canonical GCC/Clang/sanitizer builds, installed-package verification and all eight
package-version probes at the assessed source.
- **TESTS, DOCUMENTATION:** canonical selected CTest profiles, full workflow checks, documentation
and documentation-policy jobs; counts are configuration-specific as shown above.
- **SECURITY:** pinned workflow/action policy, successful Trivy job, and the scanner-input
limitation stated above. The result is not a vulnerability-free claim.
- **SECURITY-GOVERNANCE:** owner-approved private process evidence is reused from the prior
assessment; confidential details are not reproduced.
- **NUMERICAL:** all 95 requirement IDs remain mapped; the affected LU validation-order mapping
now includes `math.lu`. GCC, Clang, sanitizer and extended-policy tests passed as listed.
- **CONCURRENCY:** all five profile rules apply; hosted ThreadSanitizer lifecycle evidence passed.
No internal worker threads or shared mutable caches are introduced.
- **DATA:** extent overflow and validation-before-result-publication behavior remain exercised by
the existing boundary/failure tests and mapped numerical requirements.

## Limitations

This assessment covers only the exact source commit and Ubuntu 24.04 x86-64 profiles above. It is
not an ISO/IEC 14882 certification, universal numerical correctness proof, safety/real-time
certification, production-readiness claim, qualified performance baseline, or evidence for other
operating systems, architectures, compilers, standard libraries, or deployment environments. The
immutable 0.1.0 artifact remains affected by the LU defect; this source candidate does not alter or
replace that artifact. Release publication and installation require the separate exact artifact
packet and owner approval.
