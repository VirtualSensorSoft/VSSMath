# VSSMath 0.1.1 development correction

Status: merged source candidate; evidence PR under review. No 0.1.1 release or installation has been
published.

- [Specification](specification.md)
- [Implementation plan](implementation-plan.md)
- [Source assessment and evidence](evidence.md)
- [Development source notes](release-notes.md)

This patch aligns package metadata with version `0.1.1`, retains the LU solve input-validation
correction, and adds an independent installed-consumer check for singular and ill-conditioned LU
error precedence. The package name, target, namespace, C++20 interface, public API and runtime
dependency profile are unchanged.

The immutable 0.1.0 artifact is unchanged and remains affected by the LU defect. The exact 0.1.1
source candidate and hosted evidence are recorded in the linked assessment. The separate timing
exception applies only to `ENG-CPP-076`; no qualified timing or production claim is made. Release
publication and installation require a separately approved exact artifact packet.

Tracking: public issue [#21](https://github.com/VirtualSensorSoft/VSSMath/issues/21), LU issue
[#16](https://github.com/VirtualSensorSoft/VSSMath/issues/16), source correction PR
[#22](https://github.com/VirtualSensorSoft/VSSMath/pull/22), and timing review issue
[#19](https://github.com/VirtualSensorSoft/VSSMath/issues/19).
