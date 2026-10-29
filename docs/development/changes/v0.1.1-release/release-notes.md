# VSSMath 0.1.1 development source notes

Status: merged source candidate; not a published release or installed package.

## Correction

The LU solve path now preserves the documented validation precedence for invalid vector and matrix
right-hand sides, including non-finite inputs and a zero-column right-hand-side matrix. Regression
coverage exercises LU directly and through an independently built installed-package consumer.

## Package identity and compatibility

The project, installed package metadata, generated version header, guide and consumer fixtures
identify the candidate as `0.1.1`. The installed-package probe accepts exact `0.1.1`, compatible
`0.1`, and compatible `0.1.0` requests; it rejects exact `0.1.0`, `0.0.9`, `0.1.2` (compatible or
exact), and `0.2.0`. The package name, exported target, namespace, C++20 interface, public API and
runtime dependency profile are unchanged.

## Verification and limits

Canonical hosted CI at source commit
[`cf98e60530da9ce403741e38b894ae4989625aa7`](https://github.com/VirtualSensorSoft/VSSMath/commit/cf9
8e60530da9ce403741e38b894ae4989625aa7) passed its 16 required jobs. The exact tested configurations
and limitations are in the [0.1.1 source assessment](evidence.md). This candidate makes no qualified
timing, production, or cross-platform claim. The approved [development-only timing
exception](../../deviations/development-timing-0.1.1.md) remains limited to `0.1.1` and
`ENG-CPP-076`.

The `0.1.0` artifact and its history remain immutable and are still affected by the LU validation
defect. These notes describe merged source only; they do not announce publication, installation, or
replacement of `0.1.0`. VSSMath retains its existing rights-reservation notice; these changes add no
license grant.
