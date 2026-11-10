# VSSMath 0.1.1 correction release implementation plan

- Record ID: 2
- Status: Completed
- Owner: VSSMath maintainers
- Scope: Implement and validate the approved VSSMath 0.1.1 source correction.
- Tracking issue: https://github.com/VirtualSensorSoft/VSSMath/issues/21
- Approval pull request: Not recorded in this repository
- Implementation pull request(s): [#20](https://github.com/VirtualSensorSoft/VSSMath/pull/20), [#22](https://github.com/VirtualSensorSoft/VSSMath/pull/22), and [#23](https://github.com/VirtualSensorSoft/VSSMath/pull/23)
- Specification: docs/development/archive/changes/v0.1.1-release/specification.md

## Ordered work

1. Add version and installed-consumer regressions first; demonstrate expected
   failure against 0.1.0 while preserving passing LU correction tests.
2. Update package/project/consumer version identity to 0.1.1 and test the exact
   compatible/rejected CMake package request matrix. Retain all origin and
   negative-package checks.
3. Extend the independent installed consumer to assert singular and
   ill-conditioned LU error precedence for non-finite vector/matrix RHS and
   reject a zero-column matrix RHS.
4. Add the 0.1.1-only timing deviation, mark current conformance Under-review,
   and index the release specification, plan, README and evidence.
5. Run the six planned focused CTests and repository documentation, conformance,
   numerical-traceability, formatting and diff checks. Do not run the expensive
   local full-simulation suite; canonical hosted CI remains unchanged.
6. Completed in source PRs [#20](https://github.com/VirtualSensorSoft/VSSMath/pull/20)
   and [#22](https://github.com/VirtualSensorSoft/VSSMath/pull/22); canonical hosted CI
   passed before owner merge.
7. Completed in assessment PR [#23](https://github.com/VirtualSensorSoft/VSSMath/pull/23).
   The separately approved 0.1.1 source prerelease was published at
   [tag 0.1.1](https://github.com/VirtualSensorSoft/VSSMath/releases/tag/0.1.1), and
   native installation acceptance was recorded on tracking issue
   [#21](https://github.com/VirtualSensorSoft/VSSMath/issues/21). Those later actions
   were outside this plan's original authorization.

## Validation

Configure a fresh Debug build with `BUILD_TESTING=ON` and documentation build
disabled, build with two workers, then run the focused CTest selection recorded
in the release evidence file. Run the existing repository policy checks and
inspect the final diff before pushing.
