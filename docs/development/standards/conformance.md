# Conformance and evidence

The rule catalogue is assessed for the exact source revision and supported build profile. A passing test suite is evidence for its tested behavior; it does not by itself establish conformance to ISO/IEC 14882 beyond the compiler and build configuration used, certification, universal numerical correctness, or safety suitability.

## Assessment states

- **Pending** — no evidence-backed assessment has been recorded.
- **Passed** — reviewed evidence supports the stated rule for the assessed revision.
- **Approved deviation** — a specific, reviewed deviation is recorded with its reason, impact, compensating controls, owner and review date.

A scope exclusion is recorded in [profile applicability](profiles.md), including the condition that would bring it into scope. The conformance table uses only states accepted by its checker; an explained exclusion may be assessed as Passed for the bounded current scope without claiming the absent capability was tested.

A final assessment identifies the exact source commit, compiler and standard-library identities available from retained evidence, configuration, commands, results, limitations and deviations. No rule is marked Passed based only on planned tests. Evidence is refreshed when relevant source, configuration or toolchain inputs change. The record is an engineering assessment, not certification or a claim beyond its stated environment.
