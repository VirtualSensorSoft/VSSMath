# VSSMath 0.2.0 development-source release review implementation plan

- Record ID: v0.2.0-release
- Status: Provisional
- Owner: VirtualSensorSoft maintainers
- Scope: Assemble a public, evidence-backed review packet for the candidate 0.2.0 development-source release.
- Tracking issue: https://github.com/VirtualSensorSoft/VSSMath/issues/19
- Specification: docs/development/changes/v0.2.0-release/specification.md
- Implementation pull request(s): https://github.com/VirtualSensorSoft/VSSMath/pull/42

## Ordered work

1. Confirm the merged upstream candidate SHA and canonical hosted CI run.
2. Verify private Debug and Release qualification markers, three stable captures per configuration, repeatability, and correctness for all benchmark cases; do not copy machine-specific records into the public repository.
3. Prepare the release review README, source assessment, proposed release notes, and accurate status/index/inventory links. Keep historical release records unchanged.
4. Run the documentation-policy checker and all five policy regression scripts; inspect the diff for private host data and formatting errors.
5. Verify the repository-local commit author and committer against the maintainer account, push the review branch to the maintainer's fork, and open a PR against organization `main` with a complete review packet.

## Validation and boundary

Record exact command results and the CI run URL in the PR. Stop at owner review/merge. Do not create a tag, publish a release, or install the package as part of this implementation unit.
