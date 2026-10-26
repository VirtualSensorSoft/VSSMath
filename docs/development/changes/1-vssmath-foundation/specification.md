# VSSMath public foundation specification

- Record ID: 1
- Status: Provisional
- Owner: VirtualSensorSoft maintainers
- Scope: Establish the standalone public development, documentation, build, test, and hosted-CI foundation for VirtualSensorMath.
- Tracking issue: https://github.com/VirtualSensorSoft/VSSMath/issues/1
- Approval pull request: Not assigned
- Implementation pull request(s): Not assigned
- Created: 2026-10-26
- Last updated: 2026-10-26
- Supersedes: None
- Superseded by: None

## Goal

Make the repository independently understandable and provide a reviewable path to a buildable C++20 numerical library with an installable CMake package, documented numerical contracts, and GitHub-hosted CI.

## Scope

The foundation establishes public development records, documentation policy metadata, engineering and conformance guidance, policy checkers, and policy-only Ubuntu 24.04 x86-64 CI. The later numerical implementation is reviewed separately against the documented public contracts before it is accepted.

## Out of scope

This record does not claim that the initial bootstrap contains buildable library source, a released package, certified conformance, qualified performance, or support for platforms beyond the initial CI target.

## Acceptance criteria

- Repository documentation states accurately whether the current revision is buildable and qualified.
- The public documentation checker validates the standalone local policy profile and the complete Markdown inventory.
- Policy checkers have focused regression tests and the workflow grants read-only repository permissions and uses immutable action references.
- Public numerical source, package exports, full test matrix, and final conformance evidence are introduced and accepted through separate reviewed changes.
