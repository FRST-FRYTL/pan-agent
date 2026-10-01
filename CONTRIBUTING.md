# Contributing to pan-agent

Thanks for your interest. pan-agent is a research preview maintained by one person in their spare
time. Issues and pull requests are welcome and handled **best-effort**. There are no response-time
or support promises. Small, focused changes are the easiest to review.

How the project itself is developed: [docs/development.md](docs/development.md).

## How changes flow

This public repo is published as **one squashed commit per release**. Development happens in a
private workspace, which also holds the benchmark harness and its non-public test data. Here is
what happens to a pull request:

1. You open a PR against `main` here. CI runs the tests against the tested Hermes version,
   a secret scan and a DCO check.
2. Review happens on the PR as usual.
3. When it is accepted, the maintainer applies your commits to the development workspace with
   your authorship and `Signed-off-by` intact. The PR is then closed, with a pointer to the release
   that contains it. It is not merged here.
4. Your change ships in the next release, and the CHANGELOG credits it.

So your PR shows as "closed", not "merged". That is expected.

## Developer Certificate of Origin (DCO)

Every commit must be signed off, which certifies the [DCO](https://developercertificate.org/):

```bash
git commit -s -m "memory: explain why X"
```

The `Signed-off-by:` line must match the commit author. CI rejects PRs with unsigned commits. To fix
the last commit: `git commit --amend -s`. To fix several: `git rebase --signoff main`.

## Development setup

The tests need Hermes Agent at the tested commit (file `HERMES_BASE`):

```bash
git clone https://github.com/NousResearch/hermes-agent ../hermes-agent
git -C ../hermes-agent checkout "$(cat HERMES_BASE)"
(cd ../hermes-agent && uv sync --frozen --extra dev)
uv pip install --python ../hermes-agent/.venv/bin/python -e ".[test]"
../hermes-agent/.venv/bin/python -m pytest -q          # unit, contract and e2e tests (no network, no GPU)
```

The unit tests alone also run without Hermes:
`uv venv && uv pip install -e ".[test]" && .venv/bin/python -m pytest tests/unit`.

The tests isolate `HERMES_HOME` in a temp directory, and the e2e tests use a scripted fake model.
Tests marked `live` need a real OpenAI-compatible endpoint (`PAN_LIVE_MODEL=http://host:port/v1`) and
are skipped otherwise. **Never point tests or `pan setup` at your real `~/.hermes`.**

## Ground rules

- **Contract tests** (`tests/contract/`) pin every assumption PAN makes about Hermes. If Hermes
  changes, the fix goes into `src/pan/hermes/compat.py`. Hermes itself is never patched.
- **Prompt caching must stay intact.** Memory is injected so that the prompt prefix stays
  append-only. A change that rewrites earlier context will be rejected.
- **Generic rules only.** Memory heuristics must not target specific benchmark questions. Please
  don't add benchmark items, answers or IDs to code, prompts or tests.
- **No private data in fixtures:** no real host names, paths, e-mail addresses or tokens. Use
  `example.com`, `/srv/projects/example`, RFC 5737 addresses and the like. CI runs gitleaks for secrets;
  everything else is checked in review.
- Keep the style of the surrounding code. Tests go with every behaviour change.

## Reporting bugs

Open an issue and include:
- `pan version`;
- your Hermes version;
- the OS;
- what you expected;
- the relevant lines of `pan memory inspect` or `pan status` output.

Redact anything private first. For security issues, see [SECURITY.md](SECURITY.md).
