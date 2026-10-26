# Profile applicability review

This review evaluates profile rules against the repository’s implemented API and delivery boundary. It is not a final conformance assessment; evidence is recorded against an exact revision before any release claim.

## ABI and plugin profile

Not applicable to the current static-library package: the repository neither loads external modules nor defines a dynamically negotiated plugin ABI. CMake exports a normal library target. If a C ABI, dynamic module boundary, or plugin host is added, reassess this profile before that interface ships.

## Data and persistence profile

Two rules apply to current behavior: public extent-based operations must reject unrepresentable size arithmetic before unsafe access/allocation, and operations must validate data before publishing a result. The library has no parser, serializer, persistent format, stored artifact, or migration interface. Rules about persistent schema/versioning, corruption recovery, cross-version compatibility, provenance fields, parser fuzzing, and stored-artifact integrity are not applicable to this API. Reassess if persistence or parsing is introduced.

## Concurrency profile

Five profile rules are included in the public rule catalogue: documented thread-safety and caller synchronization; explicit treatment of shared mutable state; race-detection evidence for supported concurrent use; explicit ordering guarantees; and clear limits for sequential-only types. The library has no asynchronous operations, shared queues, locks, or startup/shutdown service lifecycle, so lock ordering, cancellation/timeout handling, backpressure, and service startup/shutdown rules are not applicable. Object-lifetime concurrency remains in scope because the public API permits independent objects and documented const access to be used concurrently.
