# How pan-agent is developed

- **Written by AI coding agents.** Code, tests and docs are written by Claude Code sessions under
  the maintainer's direction.
- **The maintainer reviews and decides.** Design decisions are recorded as ADRs; the ones that
  matter to users are in [decisions.md](decisions.md).
- **Contract tests** pin every Hermes assumption PAN relies on
  ([architecture.md](architecture.md#the-compat-module-and-contract-tests)). CI runs them against
  the tested Hermes commit.
- **Secret scanning** runs as a pre-commit hook on every commit.
- **Releases are exported** from a private workspace through an allowlist of paths and scanned for
  leaks before they are published.
