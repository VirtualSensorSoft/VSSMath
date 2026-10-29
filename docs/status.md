# Status and support

The merged 0.1.1 source candidate at commit `cf98e60530da9ce403741e38b894ae4989625aa7` passed all 16
required canonical hosted CI jobs. The source assessment and conformance record are prepared for
documentation review; no 0.1.1 release, tag, or installation has been published. The immutable 0.1.0
artifact remains affected by the LU validation defect corrected in the 0.1.1 source. No production,
ISO-certification, or qualified timing claim is made.

The current assessment is limited to Ubuntu 24.04.5 x86-64, GCC 13.3.0 with libstdc++ release 13,
and Clang 18.1.3 with libstdc++ release 14. The captured profiles do not establish Clang with
libc++, another operating system, another architecture, or another deployment environment. See the
[0.1.1 source assessment](development/changes/v0.1.1-release/evidence.md) for exact job selections,
results, scanner limitations, and rule-level evidence.

CI selection counts are configuration-specific and do not represent the full registered test
inventory. Tests do not prove universal numerical correctness, safety for arbitrary domain-specific
input, or performance on unmeasured hardware. The approved 0.1.1 development timing exception does
not establish a qualified performance baseline.
