# Security and misuse boundaries

This repository supplies numerical library code. It does not authenticate users, access networks, persist data, control physical equipment, or implement application workflows.

## Inputs and failure behavior

Callers may supply malformed dimensions, non-finite values, extreme magnitudes, invalid tolerances or storage views whose lifetime has ended. Public operations validate the conditions documented by their contracts and report failures through status-bearing results. Callers remain responsible for keeping borrowed storage alive, synchronizing conflicting access, bounding their own allocation requests, and treating failed results as unusable.

## Build and contribution supply chain

Build tools, compiler actions, workflow actions and package installation can affect the integrity of produced artifacts. CI pins actions to reviewed revisions, uses least-privilege workflow permissions and does not publish releases from untrusted pull requests. Contributors should report security defects privately through the contact route in `SECURITY.md`.

## Limitations

The library is not a safety certification, a security boundary, or a substitute for domain validation. Numerical tests and sanitizer runs cover the tested inputs and configurations only. No claim is made that arbitrary caller input is safe or that all numerical failures can be detected.
