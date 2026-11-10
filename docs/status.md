# Status and support

The merged 0.1.1 source candidate at commit `cf98e60530da9ce403741e38b894ae4989625aa7` passed all 16
required canonical hosted CI jobs. It was published as the 0.1.1 development source prerelease, and
native installation acceptance of the verified package was completed and recorded on
[issue #21](https://github.com/VirtualSensorSoft/VSSMath/issues/21). The immutable 0.1.0 artifact
remains affected by the LU validation defect corrected in the 0.1.1 source. Neither development
prerelease makes a production, ISO-certification, or qualified timing claim.

The 0.2.0 candidate at commit `6def6b1fc8bfd39e11fdfb51dd23237d2ec22d37` passed all 16
required canonical hosted CI jobs and the controlled Debug/Release timing qualification. The
qualification used one owner-controlled local host profile; raw host-specific records remain
private and are not part of the public repository. The source remains under owner release
review and has not been tagged, published, or installed. See the [0.2.0 source assessment]
(development/changes/v0.2.0-release/evidence.md) for its scope and limitations.

The 0.1.1 hosted source assessment is limited to Ubuntu 24.04.5 x86-64, GCC 13.3.0 with libstdc++
release 13, and Clang 18.1.3 with libstdc++ release 14. Those captured profiles do not establish
Clang with libc++, another operating system, another architecture, or another deployment environment.
See the [0.1.1 source assessment](development/archive/changes/v0.1.1-release/evidence.md) for its exact
job selections, results, scanner limitations, and rule-level evidence. The separate 0.2.0 candidate
scope and timing-qualification limitations are in its source assessment.

CI selection counts are configuration-specific and do not represent the full registered test
inventory. Tests do not prove universal numerical correctness, safety for arbitrary domain-specific
input, or performance on unmeasured hardware. The approved 0.1.1 development timing exception does
not establish a qualified performance baseline.
