# Status and support

The project is at version 0.1.0 and remains under active development. Treat
the API and numerical behavior as an initial release; no long-term ABI or
cross-compiler bitwise identity is promised.

Ubuntu 24.04 x86-64 is the planned initial qualification target; qualification results are recorded only after the hosted acceptance workflow completes. Other operating systems,
architectures, compiler/standard-library combinations, and deployment
environments require separate validation before being described as supported.

The test suite covers documented numerical contracts and install consumption.
It does not prove universal numerical correctness, safety for arbitrary
domain-specific input, or performance on unmeasured hardware. No qualified
performance baseline is implied by a local benchmark run.
