# Profile applicability review

This review evaluates profile rules against the implemented API and delivery boundary. Exclusions are scoped to the assessed revision and must be revisited when a listed trigger is introduced. The conformance table records the exclusion rationale in its evidence; it does not use a separate `Not applicable` state.

## ABI and plugin profile

The current package is a static C++ library with no C ABI, dynamically negotiated ABI, cross-library object destruction, or plugin host. CMake exports a normal namespaced library target. These boundaries are excluded from the current assessment. Reassess before introducing a C ABI, shared-library/plugin boundary, or cross-library create/destroy contract.

## Data and persistence profile

The two current data rules apply: public extent-based operations reject unrepresentable size arithmetic before unsafe access/allocation, and operations validate data before publishing a result. The library has no parser, serializer, persistent format, stored artifact, or migration interface. Persistent-schema, corruption-recovery, cross-version-data, provenance-field, parser-fuzzing, and stored-artifact controls are outside the current API. Reassess before adding persistence, parsing, or a stored interchange format.

## Concurrency profile

All five concurrency rules in the public catalogue apply. Public types document caller synchronization, production code has no shared mutable state, ordering is deterministic and sequential, and the hosted ThreadSanitizer job exercises concurrent lifecycle operations. The library does not create internal worker threads, asynchronous tasks, shared queues, or a service lifecycle. Reassess lock ordering, cancellation, backpressure, and startup/shutdown controls if internal parallelism, async APIs, shared mutable state, or a service lifecycle is introduced. Object-lifetime concurrency remains in scope for independent objects and documented const access.
