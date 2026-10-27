# Development-only timing exception

- Affected rule: `ENG-CPP-076`
- Scope: the first VSSMath development-only source prerelease, version `0.1.0`; this does not complete a controlled Debug/Release timing qualification.
- Reason: timing capture on an interactive development host is not sufficiently stable to establish an accepted performance baseline before the first consumer integration.
- User impact: the development prerelease makes no qualified timing, performance-regression-threshold, real-time, or production-performance claim.
- Compensating evidence retained: analytic and fixed-seed numerical checks, boundary and failure tests, GCC AddressSanitizer/UndefinedBehaviorSanitizer, ThreadSanitizer coverage, allocation-contract checks, static analysis, coverage review, and hosted Debug/Release performance diagnostics. Diagnostic timings are not a qualified baseline and do not waive allocation or correctness checks.
- Accountable owner: Yasser Radkhorrami.
- Review by: 2027-01-31, or before a `0.2.0` release or any production/stable-release claim, whichever comes first.
- Required review outcome: complete controlled timing qualification, approve a separately reviewed replacement exception, or withhold the subsequent release claim. Existing published artifacts remain immutable.

This exception does not authorize changes to numerical algorithms, tolerances, allocation behavior, or the original qualification records.
