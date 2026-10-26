# Contributing

Contributions are reviewed through GitHub pull requests. Before opening one:

1. Check existing issues and pull requests for related work.
2. Keep the change focused and explain the observable behavior it affects.
3. Add or update tests for numerical behavior, validation, and failure results.
4. Run the documented build and CTest suite for the affected configuration.
5. Include any API, compatibility, numerical-accuracy, or platform limitations.

Do not include credentials, local machine paths, private data, generated build
trees, or unreviewed performance claims. Numerical fixtures must be reproducible
without downloading data or running an external calculator. See the
[numerical contract](docs/numerical-contract.md) and
[development rules](docs/development/standards/engineering-rules.md).

The repository is currently maintained by VirtualSensorSoft. The maintainer
may request changes or close a proposal that does not meet the project
contracts. No contribution receives a reuse license through submission; see
[LICENSE](LICENSE).
