# Development records and standards

This directory is the repository's local source of documentation policy, engineering criteria, implementation history and conformance evidence. It is maintained in this repository for the public codebase; the file-level checker validates its structure and links.

## Current guidance

- [Engineering rules](standards/engineering-rules.md)
- [AI engineering guidance](standards/ai-engineering-guidance.md)
- [Conformance process](standards/conformance.md)
- [Profile applicability](standards/profiles.md)
- [Deviation register](deviations/README.md)
- [Analyzer suppressions](deviations/analyzer-suppressions.md)
- [Numerical requirement map](numerical-requirements.tsv)
- [Development-release evidence](changes/1-vssmath-foundation/evidence.md)
- [0.1.1 correction release records](changes/v0.1.1-release/README.md)
- [0.1.1 correction specification](changes/v0.1.1-release/specification.md)
- [0.1.1 correction implementation plan](changes/v0.1.1-release/implementation-plan.md)
- [Archive](archive/README.md)

## Dependencies and toolchain updates

The shipped library has no third-party numerical runtime dependency; it uses
the C++ standard library. Ubuntu CI tools are installed from the Ubuntu 24.04
package repositories, and their relevant versions are recorded with acceptance
evidence. GitHub Actions are pinned to reviewed immutable revisions. Maintainers
review dependency, package-source, or action changes through pull requests and
rerun the affected policy and build checks before merging.

The repository began with a clean public history. Its migration inventory describes only files present in this repository; it does not represent prior private discussions, issues, or commits.
