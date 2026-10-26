# Conformance and evidence

The rule catalogue is assessed for the exact source revision and supported build profile. A passing test suite is evidence for its tested behavior; it does not by itself establish conformance to ISO/IEC 14882 beyond the compiler and build configuration used, certification, universal numerical correctness, or safety suitability.

## Assessment states

- **Pending** — no evidence-backed assessment has been recorded.
- **Passed** — reviewed evidence supports the stated rule for the assessed revision.
- **Not applicable** — the rule was assessed against the actual project scope and its exclusion is explained.
- **Approved deviation** — a specific, reviewed deviation is recorded with its reason, impact, compensating controls, owner and review date.

A final report identifies the exact commit, compiler and standard library, configuration, commands, results, limitations and deviations. No rule is marked Passed based only on planned tests. Evidence is refreshed when relevant source, configuration or toolchain inputs change.

At this foundation stage, no full conformance assessment has been performed. The catalogue is a checklist, not a declaration that the project satisfies it.
