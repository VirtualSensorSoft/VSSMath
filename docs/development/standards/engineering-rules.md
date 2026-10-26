# Engineering rules

This project-specific profile applies the ISO/IEC 14882 C++ language standard through the declared C++20 baseline. It is an engineering checklist, not a certification or claim of conformance to every industrial or safety standard. Each rule is assessed against the actual API, implementation and supported platform; applicability is explained rather than assumed.

The catalogue contains 64 general C++ engineering rules and 12 numerical/scientific rules. Rule identifiers are local to this public catalogue.

## ENG-CPP-001

Production C++ uses the declared C++20 baseline.

## ENG-CPP-002

Unintended compiler extensions are disabled.

## ENG-CPP-003

Undefined behavior is forbidden.

## ENG-CPP-004

Implementation-defined assumptions are documented and checked.

## ENG-CPP-005

Narrowing and signed/unsigned boundary conversions are explicit and validated.

## ENG-CPP-006

Platform-specific behavior is isolated behind a documented adapter.

## ENG-CPP-007

Every resource has explicit ownership and deterministic cleanup.

## ENG-CPP-008

RAII manages resources.

## ENG-CPP-009

Raw pointers/references do not hide ownership transfer.

## ENG-CPP-010

Borrowed views document lifetime, mutability, aliasing, and invalidation.

## ENG-CPP-011

Destructors do not emit exceptions.

## ENG-CPP-012

Cross-library creation and destruction use the owning component.

## ENG-CPP-013

Global mutable state requires an approved ownership and synchronization design.

## ENG-CPP-014

Public APIs expose ownership, optionality, units, dimensions, and failure behavior where applicable.

## ENG-CPP-015

Credibly confusable primitive values use strong or validated types.

## ENG-CPP-016

ABI and serialized widths are explicit when width is contractual.

## ENG-CPP-017

Public boolean parameters that obscure call-site meaning use named options or enumerations.

## ENG-CPP-018

Public APIs document preconditions, postconditions, invariants, complexity, and thread-safety where applicable.

## ENG-CPP-019

Private implementation details do not leak into public headers unnecessarily.

## ENG-CPP-020

Every fallible public operation defines failure categories and post-failure state.

## ENG-CPP-021

Expected recoverable/numerical outcomes use an explicit status/result contract.

## ENG-CPP-022

Any internal exception use has a documented exception-safety policy.

## ENG-CPP-023

C++ exceptions, standard-library objects, RTTI identities, and ownership do not cross the stable C ABI.

## ENG-CPP-024

ABI adapters catch exceptions and return stable error records.

## ENG-CPP-025

Failure never silently returns plausible scientific data.

## ENG-CPP-026

Errors and logs redact credentials, confidential inputs, and uncontrolled paths.

## ENG-CPP-027

Deterministic behavior does not depend accidentally on address, unordered iteration, locale, global state, or wall-clock timing.

## ENG-CPP-028

Randomness uses an explicit or recorded seed.

## ENG-CPP-029

Floating-point reproducibility claims state bitwise, bounded, or classification equivalence.

## ENG-CPP-030

Parallel ordering and reduction semantics are defined before deterministic parallel claims.

## ENG-CPP-031

Configuration and component versions required to reproduce output are recordable.

## ENG-CPP-032

Credible input-size and resource limits are documented.

## ENG-CPP-033

Excessive size, allocation failure, and resource exhaustion have defined behavior.

## ENG-CPP-034

Performance claims use representative cases and repeatable methodology.

## ENG-CPP-035

Real-time, lock-free, wait-free, constant-time, or bounded-memory claims require dedicated evidence.

## ENG-CPP-036

CMake uses out-of-source builds for C++ repositories.

## ENG-CPP-037

Reusable libraries export installable namespaced CMake targets.

## ENG-CPP-038

Public artifacts contain no accidental source/build paths.

## ENG-CPP-039

Supported GCC and Clang Debug plus at least one Release configuration pass.

## ENG-CPP-040

Organization-owned code has a zero accepted-warning baseline in required CI.

## ENG-CPP-041

Generated build, editor, credential, and local-install artifacts are excluded from Git.

## ENG-CPP-042

Behavioral changes follow RED/GREEN/refactor test-driven development.

## ENG-CPP-043

Applicable unit, component, integration, install-consumer, and release tests exist.

## ENG-CPP-044

Invalid input and failure states have negative tests.

## ENG-CPP-045

ASan and UBSan pass where supported.

## ENG-CPP-046

Concurrency code has race-detection evidence where the supported toolchain permits it.

## ENG-CPP-047

Static analysis uses the centrally selected configuration.

## ENG-CPP-048

Formatting is checked without silently changing reviewed code in CI.

## ENG-CPP-049

Each normative behavior maps to a test, inspection, or analysis result.

## ENG-CPP-050

Coverage evidence identifies untested critical paths without using one universal percentage as proof of quality.

## ENG-CPP-051

Supported examples compile and run automatically.

## ENG-CPP-052

Public APIs document purpose, ownership, inputs, outputs, errors, lifetime, and thread-safety where applicable.

## ENG-CPP-053

Numerical APIs additionally document units, shapes, ranges, tolerances, conditioning, and determinism.

## ENG-CPP-054

User guides, specifications, plans, generated API docs, and evidence remain distinct and linked.

## ENG-CPP-055

Releases document compatibility, migration, limitations, and supported environments.

## ENG-CPP-056

Credentials and confidential data are absent from source, logs, artifacts, and images.

## ENG-CPP-057

CI permissions and credentials use least privilege.

## ENG-CPP-058

CI actions and release inputs are immutable or precisely pinned.

## ENG-CPP-059

External source and tool inputs have an ownership/update policy.

## ENG-CPP-060

Applicable artifacts receive vulnerability scanning.

## ENG-CPP-061

Released artifacts are signed and carry verifiable provenance where release infrastructure supports it.

## ENG-CPP-062

An SBOM is produced when external shipped components or customer policy require it.

## ENG-CPP-063

Trust boundaries receive a threat model before release.

## ENG-CPP-064

Security incidents and credential rotation have an owned private process.

## ENG-CPP-065

Scalar, shape, storage, ownership, and aliasing contracts are exact.

## ENG-CPP-066

Dimensions and ranges are validated before allocation/arithmetic.

## ENG-CPP-067

Non-finite input and intermediate-result behavior is explicit.

## ENG-CPP-068

Tolerances are immutable and operation-local.

## ENG-CPP-069

Error precedence is defined for simultaneous invalid conditions.

## ENG-CPP-070

Singularity, rank, conditioning, and ill-conditioning semantics are defined where relevant.

## ENG-CPP-071

Independent analytic fixtures and fixed-seed properties verify results.

## ENG-CPP-072

Accuracy domains and tolerances have numerical justification.

## ENG-CPP-073

Covariance symmetry, definiteness, rank, and availability are distinct.

## ENG-CPP-074

Scientific covariance, numerical scaling, and display rounding remain distinct.

## ENG-CPP-075

Unestablished numerical results fail explicitly instead of being fabricated.

## ENG-CPP-076

Debug and Release catastrophic-regression baselines use representative cases.

# Applicable profile rules

## ENG-CON-001 — Concurrency

Public types document their thread-safety level and caller obligations.

## ENG-CON-002 — Concurrency

Any shared mutable state has explicit ownership and synchronization.

## ENG-CON-006 — Concurrency

Supported concurrent use receives race-detection evidence where the toolchain permits it.

## ENG-CON-007 — Concurrency

Ordering guarantees, including deterministic result ordering, are explicit.

## ENG-CON-009 — Concurrency

Sequential-only types explicitly state that concurrent use is unsupported.

## ENG-DATA-002 — Data and persistence

Public extent-based allocation and arithmetic validate representability and reject overflow before unsafe use.

## ENG-DATA-003 — Data and persistence

Operations validate their inputs before publishing an owning result.
