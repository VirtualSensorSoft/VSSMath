# Development-only timing exception for 0.1.1

- Affected rule: `ENG-CPP-076`.
- Scope: VSSMath version `0.1.1` only. This permits review of the LU correctness
  correction without establishing controlled Debug/Release timing baselines.
- Reason: interactive development-host timing is not sufficiently stable to
  establish an accepted performance baseline before this correction release.
- User impact: version 0.1.1 makes no qualified timing, regression-threshold,
  real-time, or production-performance claim.
- Compensating evidence retained: correctness regressions, sanitizer coverage,
  allocation contracts, thread-safety checks, static analysis, coverage review,
  and hosted Debug/Release performance diagnostics. Diagnostics are not a
  qualified baseline and waive no correctness or allocation requirement.
- Accountable owner: Yasser Radkhorrami.
- Review by: 2027-01-31, before `0.2.0`, or before any production/stable-release
  claim, whichever occurs first. This exception does not reset the deadline.
- Required review outcome: complete controlled timing qualification, approve a
  separately reviewed replacement exception, or withhold the later release
  claim. Any subsequent release requires its own scope decision.

This is separate from the [0.1.0 exception](development-timing.md). It changes no
algorithm, tolerance, allocation behavior, or timing threshold and does not
qualify either release for production use.
