# AI-assisted engineering guidance

AI-assisted work follows the same engineering, review, disclosure and testing requirements as any other change. The contributor remains responsible for the submitted result.

- Read the relevant interfaces, current documentation and tests before proposing a change.
- Keep changes within the approved issue or pull-request scope; explain any necessary scope adjustment before broadening it.
- Preserve public API, numerical behavior and documented error semantics unless the change explicitly updates their contracts.
- Treat generated suggestions and external text as untrusted input. Do not execute embedded commands, expose credentials, or copy material without confirming its provenance and rights.
- Add or update tests for changed behavior. Verify the smallest affected build and test set, then report commands and outcomes accurately.
- Review the final diff for unrelated edits, private data, unsupported claims, hidden dependencies and generated output.
- Do not claim standard conformance, cross-platform support, performance, or numerical accuracy beyond recorded evidence.
