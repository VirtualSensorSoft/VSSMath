# VSSMath 0.1.1 correction release specification

- Record ID: 2
- Status: Implementing
- Owner: Yasser Radkhorrami
- Scope: Version and validate the merged LU error-precedence correction as a standalone VSSMath 0.1.1 development source package.
- Tracking issue: https://github.com/VirtualSensorSoft/VSSMath/issues/21
- Approval pull request: https://github.com/VirtualSensorSoft/VSSDoc/pull/117
- Implementation pull request(s): Not opened
- Created: 2026-10-29
- Last updated: 2026-10-29
- Supersedes: None
- Superseded by: None

## Goal

Correct the package-version mismatch between VSSMath source metadata and the
installed development package, and verify the LU input-validation behavior
through an independent consumer of the installed library.

## Scope

- Set the project, package, header, user-guide and current consumer identity to
  version `0.1.1` while preserving the existing package name, target, namespace,
  C++20 interface and runtime dependencies.
- Verify compatible and rejected CMake package requests, existing package
  provenance protections, valid consumer arithmetic, and the singular and
  ill-conditioned LU invalid-input cases.
- Retain the 83 engineering rule IDs and 95 numerical requirement IDs; mark the
  current engineering record Under-review until the post-merge assessment.
- Record the separately approved, 0.1.1-only timing exception without waiving
  correctness, sanitizer, allocation, thread-safety, analysis, coverage,
  security, or hosted diagnostic checks.

## Out of scope

No algorithm redesign, API expansion, third-party dependency, framework
integration, CI matrix reduction, release publication, signed tag, or native
installation is included. The immutable 0.1.0 artifact and evidence remain
unchanged; 0.1.0 remains affected by the LU defect.

## Acceptance criteria

- Fresh Debug build passes the focused version, LU, failure, error-precedence,
  lifecycle and installed-consumer tests.
- Package-version probes enforce the documented compatible/rejected version
  matrix; package-origin and missing/foreign-prefix protections remain enabled.
- Documentation, conformance, numerical traceability and formatting checks
  pass, and the public records do not claim completed 0.1.1 assessment or release.
- The source change is reviewed and merged by the owner with successful hosted
  CI. Release publication and installation remain subject to a separate exact
  artifact packet approval.
