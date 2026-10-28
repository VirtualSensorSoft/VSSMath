# VSSMath development-release conformance assessment

- Final assessed source commit: `3d00dd8ddd7a443ce46fba9f7ccb125245443593` (canonical merge).
- Final hosted CI: [run 40753766010](https://github.com/VirtualSensorSoft/VSSMath/actions/runs/40753766010). Its pull-request candidate commit was `61d232cdcd1923aa0a324fb5ed9bcd6e1816e4b5`; its tree `eff0c3229ce654adb57dde7b2895acd011dde225` is identical to the canonical merge tree.
- The assessed source and public runtime headers (`src/` and `include/`) are byte-identical to the 0.1.0 release source commit `63a26726f28bd903c7dcaa1d7560c570912010cd`.
- Historical assessment: source `20c8c9910a0f3ebf164bd513da6609cd59e8daeb`, CI run [40579072074](https://github.com/VirtualSensorSoft/VSSMath/actions/runs/40579072074). This earlier record is superseded for the final source identity and does not contain the new standard-library capture.
- Assessment scope: development-only Ubuntu 24.04 x86-64 package at the exact final source revision; this is an engineering assessment, not certification or production acceptance.
- Final record status: 82 Passed, one Approved deviation (`ENG-CPP-076`), and no Pending rules. This does not claim ISO certification, universal numerical correctness, production readiness, or a qualified timing baseline.

## Assessed Identity

- Assessed source commit: `3d00dd8ddd7a443ce46fba9f7ccb125245443593`.
- Hosted CI run: [40753766010](https://github.com/VirtualSensorSoft/VSSMath/actions/runs/40753766010); the recorded artifact source SHA is the tested pull-request candidate `61d232cdcd1923aa0a324fb5ed9bcd6e1816e4b5` with the same tree as the merge.
- Source identity check: `src/` and `include/` are unchanged from release source `63a26726f28bd903c7dcaa1d7560c570912010cd`.
- Platform: Ubuntu 24.04.5 LTS, x86-64.
- Build matrix: GCC 13.3.0 Debug/Release and ASan/UBSan Debug; Clang 18.1.3 Debug; CMake 3.31.6; Unix Makefiles; C++20.

## Toolchain and configuration evidence

The GCC Debug, GCC Release, GCC ASan/UBSan and Clang Debug artifacts from run 40753766010 record the exact compiler and standard-library macros. GCC 13.3.0 uses libstdc++ release 13 (`__GLIBCXX__=20240904`, `_GLIBCXX_USE_CXX11_ABI=1`). Clang 18.1.3 uses libstdc++ release 14 (`__GLIBCXX__=20240908`, `_GLIBCXX_USE_CXX11_ABI=1`); this run does not establish a Clang/libc++ profile. All artifacts report `__cplusplus=202002L`. CMake 3.31.6, Unix Makefiles, Ubuntu 24.04.5 LTS and x86-64 are recorded. CMake disables compiler extensions; GCC sanitizer configuration enables ASan/UBSan. This evidence resolves `ENG-CPP-031` for these exact compiler/standard-library profiles only.

In run 40753766010, GCC Debug, GCC Release and Clang Debug each passed 44/44 selected CTests; GCC ASan/UBSan passed 43/43. These compiler lanes excluded `ci-once-per-full` tests. The full-selection checker reported 51 registered tests, 3 once-per-full tests and 44 fast tests; the `extended-policy` job passed tests #33, #35 and #38 (3/3). The selections are configuration-specific, not claims that every registered test ran in each job. All 16 executed CI checks passed; `gcc-debug (fast fork/manual suite)` was skipped by its event condition.

## Evidence groups

### Language

`CMakeLists.txt` sets `cxx_std_20` and `CXX_EXTENSIONS OFF`. The GCC Debug/Release and Clang Debug artifacts from run 40753766010 record Ubuntu 24.04.5 x86-64, CMake 3.31.6, compiler and standard-library versions, Unix Makefiles, build type, and sanitizer setting; their candidate source tree is identical to the final assessed commit.

### Ownership

Public Vector/Matrix and view/factorization headers specify ownership, moved-from state, aliasing, lifetimes and allocation behavior. `math.algorithm_lifecycle`, `math.algorithm_failures`, `math.installed_consumer`, and the GCC ASan/UBSan run provide evidence at the assessed source revision.

### Api-Errors

Public `ErrorCode`, `Status`, `Result<T>`, options and algorithm headers define normal failure categories, result ownership, validation order, and exception boundaries. `math.algorithm_failures`, `math.algorithm_error_precedence`, and the 95-row numerical requirement map link these contracts to tests.

### Determinism

`docs/numerical-contract.md`, `docs/numerical-requirements.tsv`, fixed-seed property tests, `math.reproducibility`, and performance diagnostic metadata record ordering, seeds, configuration and the limits of reproducibility claims.

### Performance

The public numerical-boundary and performance diagnostics use representative algorithm cases and explicit methodology. `math.numerical_boundaries`, `math.performance_gate_contract`, `math.algorithm_qualification_policy`, and `math.algorithm_comparison` passed. These diagnostics are not a qualified baseline; `ENG-CPP-076` has its separate approved development-only deviation.

### Build

`CMakeLists.txt`, exported `VirtualSensorSoft::math`, installed-package consumer, path checks, warning policy and full Ubuntu matrix verify out-of-source builds, installation, naming, warnings and artifact boundaries. Run 40753766010 passed all 16 executed CI checks.

### Tests

Run 40753766010: GCC Debug 44/44 selected CTests, GCC Release 44/44, Clang Debug 44/44, and GCC ASan/UBSan 43/43. The selector reports 51 registered tests, 3 once-per-full tests and 44 fast tests; the separate `extended-policy` job passed tests #33, #35 and #38 (3/3). The separate fork/manual suite was skipped for this event by design.

### Documentation

The public headers, user guide, numerical contract, development indexes, generated API documentation workflow, and documentation-policy job define and check the documented API and evidence boundaries. Run 40753766010 passed documentation and documentation-policy jobs.

### Security

`SECURITY.md`, `docs/security/threat-model.md`, `.github/workflows/ci.yml`, and the pinned action references define reporting boundaries, least-privilege workflow permissions and supply-chain controls. Run 40753766010 passed security, static-analysis and workflow-policy checks; Trivy 0.69.3 source and installed-artifact summaries each report zero findings.

### Scope

The current package is a static C++ library with no C ABI, plugin boundary, parser, persistence layer, release artifact or shipped third-party numerical runtime. `docs/development/standards/profiles.md` records each current exclusion and a reassessment trigger. This is a scoped assessment, not evidence for absent features; revisit before the trigger is introduced.

### Numerical

`docs/development/numerical-requirements.tsv` contains 95 public behavior rows and maps them to fixtures and CTest evidence. Algorithm property, lifecycle, failure, error-precedence, covariance, differentiation, statistics, probability, reproducibility and boundary tests passed in the hosted GCC/Clang/sanitizer runs. The assessed numerical source and public headers are unchanged from release source G; they were reviewed for factor reuse, owning results, aliasing/lifetimes, finite-value handling, rank/conditioning, validation order and checked extents.

### Concurrency

Public headers state caller synchronization and sequential ordering. `tests/negative/test_lifecycle.cpp` exercises independent concurrent access; the `thread-safety` job in run 40753766010 builds/runs the lifecycle test with ThreadSanitizer and passed. No production worker threads or shared mutable caches are present.

### Data

`VectorView::create`, `MatrixView::create`, owning constructors, algorithm input validation and the numerical-boundary/failure tests check representability and validate data before publishing successful results. The numerical requirement map names the boundary and failure fixtures.

### Security governance

This criterion concerns owner-controlled private governance, not a runtime property of the library. The owner has confirmed and approved a private incident-response and credential-rotation process. Its operational details, locations and records remain private and are intentionally not reproduced here. This assessment records process existence and ownership; it does not claim a live incident or credential-rotation exercise.

### Timing

The accepted specification authorizes only the bounded development-release exception recorded in `docs/development/deviations/development-timing.md`. Correctness, sanitizer, allocation, static-analysis, coverage, thread-safety and diagnostic checks remain required; no qualified timing claim is made.

## Rule-level assessment

<a id="rule-level-assessment"></a>

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

## Limitations

- Ubuntu 24.04 x86-64 is the only assessed platform. Other operating systems, architectures, compiler/standard-library combinations, and deployment environments are unassessed.
- Hosted algorithm timing outputs are diagnostics only. No controlled timing baseline, performance certification, hard-real-time guarantee, safety certification, or universal numerical-correctness claim is made.
- Rules for absent interfaces are scoped out only as described in the applicability records; reassess before adding those interfaces.
- Security incident and credential-rotation governance remains pending and must be handled privately before a final record or release readiness claim.
- Tests support the specific contracts and inputs exercised; they do not establish correctness for all possible floating-point inputs or domain-specific uses.
