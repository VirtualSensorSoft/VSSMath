# VSSMath public foundation implementation plan

- Record ID: 1
- Status: Provisional
- Owner: VirtualSensorSoft maintainers
- Scope: Establish the standalone public development, documentation, and policy-CI foundation.
- Tracking issue: https://github.com/VirtualSensorSoft/VSSMath/issues/1
- Approval pull request: Not assigned
- Implementation pull request(s): Not assigned
- Specification: docs/development/changes/1-vssmath-foundation/specification.md

## Ordered work

1. Add the public documentation, engineering criteria, applicability guidance, initial conformance record, and empty analyzer suppression register.
2. Add the local documentation-policy metadata using the actual earlier public seed commit and the exact checker digest.
3. Add the minimal read-only workflow and policy regression tests. Do not claim numerical builds or tests until implementation files are present.
4. In a later reviewed change, import the approved numerical source, CMake package, tests, and complete hosted workflow; run the specified acceptance matrix and record actual results.

## Validation

Run the documentation-policy, conformance-record, analyzer-suppression, and workflow-policy checks. Confirm the workflow has only read access, immutable reviewed action pins, and no self-hosted runner dependency. Keep numerical conformance pending until source and its acceptance evidence exist.
