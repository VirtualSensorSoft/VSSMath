# VSSMath 0.2.0 development-source release review specification

- Record ID: v0.2.0-release
- Status: Provisional
- Owner: VirtualSensorSoft maintainers
- Scope: Record release-candidate identity, canonical CI, completed controlled timing qualification, proposed release notes, and explicit limits for a possible 0.2.0 development-source release.
- Tracking issue: https://github.com/VirtualSensorSoft/VSSMath/issues/19
- Approval pull request: Pending owner review
- Implementation pull request(s): Pending PR creation
- Candidate source: `6def6b1fc8bfd39e11fdfb51dd23237d2ec22d37`
- Created: 2026-11-11
- Supersedes: None
- Superseded by: None

## Goal

Give the owner an accurate, reviewable 0.2.0 source-release packet after the exact candidate passed canonical hosted CI and controlled Debug/Release timing qualification.

## Scope

Record the candidate revision and CI run, summarize the qualification without publishing host-identifying data, describe the development-source changes and supported evidence, and preserve the existing all-rights-reserved notice. Keep historical 0.1.0 and 0.1.1 records version-specific.

## Out of scope

This specification authorizes no tag, GitHub Release, source archive publication, signature, package installation, stable ABI promise, production or safety claim, full 0.2.0 rule-by-rule conformance assessment, or portable CI timing-regression baseline.

## Acceptance criteria

- CI and timing evidence are tied to the same exact source commit.
- Qualification summary reports Debug and Release outcomes and limits without disclosing raw host metadata.
- The current conformance-record source identity remains explicitly identified as 0.1.1.
- Proposed release notes preserve pre-1.0 compatibility and rights limitations.
- Documentation policy and its focused regression checks pass.
- Publication and installation remain separate owner-approved actions.
