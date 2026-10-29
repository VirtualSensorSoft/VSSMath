# VSSMath 0.1.1 development correction

Status: under review; no 0.1.1 release or installation has been published.

- [Specification](specification.md)
- [Implementation plan](implementation-plan.md)
- [Source evidence](evidence.md)

This patch aligns package metadata with version `0.1.1`, retains the LU solve
input-validation correction, and adds an independent installed-consumer check
for singular and ill-conditioned LU error precedence. It preserves the existing
package name, target, namespace, C++20 interface, and dependency profile.

The immutable 0.1.0 artifact is unchanged and remains affected by the LU defect.
The current assessment and evidence are not finalized until the implementation
is merged and its canonical hosted CI run is reviewed. See public tracking issue
[#21](https://github.com/VirtualSensorSoft/VSSMath/issues/21), existing LU issue
[#16](https://github.com/VirtualSensorSoft/VSSMath/issues/16), correction PR
[#20](https://github.com/VirtualSensorSoft/VSSMath/pull/20), and timing review
issue [#19](https://github.com/VirtualSensorSoft/VSSMath/issues/19).

The separately scoped [0.1.1 timing exception](../../deviations/development-timing-0.1.1.md)
does not waive correctness, sanitizers, allocation contracts, thread-safety,
static analysis, coverage, security, or hosted performance diagnostics. Release
publication and installation require a separately approved exact artifact
packet.
