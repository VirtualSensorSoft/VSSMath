# VSSMath 0.1.1 correction release implementation plan

- Record ID: 2
- Status: Implementing
- Owner: Yasser Radkhorrami
- Scope: Implement and validate the approved VSSMath 0.1.1 source correction.
- Tracking issue: https://github.com/VirtualSensorSoft/VSSMath/issues/21
- Approval pull request: https://github.com/VirtualSensorSoft/VSSDoc/pull/117
- Implementation pull request(s): Not opened
- Specification: docs/development/changes/v0.1.1-release/specification.md

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
6. Commit and push the coherent source PR, observe its hosted checks, repair
   in-scope failures, then provide the owner review comment, merge title and
   merge description. Owner merges the PR.
7. Continue with the separately authorized assessment/docs PR only after Task 1
   is merged and its canonical CI is successful. Do not publish, tag, or install
   until the exact artifact packet receives separate owner approval.

## Validation

Configure a fresh Debug build with `BUILD_TESTING=ON` and documentation build
disabled, build with two workers, then run the focused CTest selection recorded
in the release evidence file. Run the existing repository policy checks and
inspect the final diff before pushing.
