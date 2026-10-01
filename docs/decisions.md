# Decisions

A digest of the architecture decision records (ADRs) that matter to users of pan-agent. The full
records are kept in the private development repository. Statuses are as of 0.1.0a1.
Decisions about the private development workflow are omitted.

See [architecture.md](architecture.md) for how these decisions show up in the code.

## ADR-002: Private first, open-source ready

**Status:** accepted. Its release consequence is replaced by ADR-013.

The development repository is kept publishable: no secrets or private data in commits, the MIT
licence and Nous Research notice kept, and commits signed off (DCO).

## ADR-004: Memory Plane as an external memory-provider plugin plus a background daemon

**Status:** accepted.

PAN is a memory provider named `pan`, loaded through Hermes' `hermes_agent.memory_providers`
entry point, with no edits to Hermes files. The in-process plugin only captures events and serves
reads; curation runs in the separate `pan-memoryd` process. Hermes' `MEMORY.md`/`USER.md` stay as
L1 hot memory, but PAN is the only automatic L1 writer (`memory.nudge_interval: 0`), writing
through Hermes' own `MemoryStore`. Consequences: only one external memory provider can be active,
and subagents get no PAN tools.

## ADR-005: Exact Hermes pin as a pip dependency

**Status:** superseded by ADR-011.

pan-agent originally depended on `hermes-agent==<exact version>`. That does not work for users:
Hermes is installed as a git checkout with its own venv, and the PyPI package lags behind it.

## ADR-006: Local model roles for the Memory Plane

**Status:** proposed. The curator part is superseded by ADR-008; the retrieval part is decided
by ADR-012.

Proposes local models per role: an embedding model and hybrid (FTS plus vector) retrieval, a
reranker, and later a small fine-tuned encoder as the gate, each with a fallback to the current
implementation. Hybrid retrieval shipped through ADR-012; a dedicated gate model has not.

## ADR-007: L1 reconciliation, and no save call for stated facts

**Status:** proposed.

(a) L1 is not strictly add-only. When the wiki supersedes a fact, PAN rewrites or removes an L1
entry that still states the old value, whether the agent or PAN wrote it. The rewrite is limited
to entries about the same subject; preferences are never touched. This is implemented. (b) The
original wording told the agent that stated facts need no tool call. That wording was revised: the
current prompt block says the agent's own memory tool works as usual and that `memory_search` and
`memory_read` are PAN's only tools. PAN captures and reconciles on top.

## ADR-008: The main model is also the Memory Plane's model

**Status:** accepted.

No separate model or server is added for memory work. Background memory calls go to the same
endpoint as the agent's main model and compete with it, so they run in the background and must
not block the agent. In 0.1.0a1 the only model call is the gate (`models.gate`, by default the
same endpoint); the curator is deterministic.

## ADR-009: Benchmark data policy and generic-only rules

**Status:** accepted.

Scenario data is split into dev (free to tune on), validation (accepts or rejects a change) and
test (external public data; final number only, never inspected, never tuned on). Rules and
heuristics must be generic: language-level, English and German parity, with a precision guard,
and never keyed on nouns from benchmark scenarios. This is the basis for the numbers in
[evaluation.md](evaluation.md).

## ADR-010: Quality-loop stopping rule, and a model-gate trial

**Status:** accepted.

Tuning stops after two consecutive iterations without a gain of at least 3 percentage points on
the validation set, with no regression on the held-out test set; smaller differences were within noise at
the set sizes used. After the rule-based classifier plateaued, a model-based gate was tried under
the same rule. It gave a clear gain and became the default, with the rules as automatic fallback. The test set was
run only at milestones, as a regression guard. Its contents were never shown to the agents doing the
tuning, and nothing was tuned on it.

## ADR-011: Add-on distribution with a runtime-checked Hermes pin, installed through a skill

**Status:** accepted (2026-09-29). Supersedes ADR-005.

pan-agent is installed into the user's existing Hermes venv and does not declare `hermes-agent`
as a dependency. The tested Hermes version (`pan.HERMES_PIN`, plus the exact upstream commit) is
checked at runtime: `pan setup` refuses an untested Hermes unless `--force` is given, and the
provider logs a warning at startup. `pan setup` is idempotent, has `--dry-run`, never enables
services, and backs up the Hermes settings it changes; `pan uninstall` restores them. A shipped
skill (`skills/install-pan/SKILL.md`) drives `pan setup` and asks the per-machine questions.

Not done yet:
- The provider does not yet disable itself when a contract assumption fails at import time. This
  hardening is proposed.
- The skill ships through a Claude Code plugin marketplace manifest in this repo. Whether the Hermes
  skills hub accepts third-party skills is still to be checked.

Its open questions on the tested Hermes range and the gate default were answered by ADR-014.

## ADR-013: Public release repository exported from an allowlist

**Status:** accepted.

The public repository receives squashed release commits exported from an allowlist of paths
(the add-on package, its tests and the install skill, plus curated docs). The development
history stays private. External pull requests are imported into the development repository with
their author and `Signed-off-by` kept. A revision on the same day set the scope: the project is
published as a research preview, installed from git tags only (no PyPI), with
best-effort issues and no support promises.

## ADR-014: Model-agnostic, with a tested reference stack and an optional vLLM cookbook

**Status:** accepted (2026-09-29).

- **Hermes:** each release is tested with one Hermes version. Other versions may work, but they are
  untested: PAN warns, and `pan setup` refuses them without `--force`.
- **Models:** any OpenAI-compatible model works.
  - Minimum capabilities: tool calling for the agent, and `json_schema` (or `json_object`) output
    for the memory gate.
  - With no suitable endpoint, the gate runs rules-only. It never falls back silently to a hosted
    provider.
- **Recommended stack:** Qwen3.8-27B-NVFP4 on vLLM with the cookbook settings. The published
  benchmark numbers hold only for that stack.
- **`pan doctor`** checks an endpoint's fitness.
- **The DGX Spark cookbook** is experimental and lives in its own repo,
  [spark-vllm-agent-cookbook](https://github.com/FRST-FRYTL/spark-vllm-agent-cookbook). It pins its base image by digest and the vLLM patch by
  commit, and it never redistributes images or weights.

## ADR-012: Hybrid retrieval with a local embedding and reranker sidecar

**Status:** accepted (2026-09-30).

- Retrieval combines FTS5 with dense vectors (Qwen3-Embedding-0.6B) through weighted
  reciprocal-rank fusion, and reranks the candidates with a cross-encoder (bge-reranker-v2-m3)
  within a 0.5 s budget. The curator uses the fusion without reranking.
- The models run in `pan-embedd`, a separate, stateless HTTP sidecar on the local host, so the agent
  venv stays free of torch. One sidecar serves every profile. The vector index is derived, like the
  FTS index, and records its embedding model.
- Hybrid retrieval is optional. When the sidecar is down, slow or serves another model, retrieval is
  FTS5 only, with a circuit breaker so a dead sidecar costs one timeout, not one per turn.
- Not done yet: the sidecar is not packaged for users, `pan doctor` does not check it, and the
  fallback shows only in the logs.

## ADR-015: Re-measuring against stock Hermes with a fixed harness

**Status:** accepted (2026-09-29).

The benchmark harness had started Hermes with `--source tool`, which hides earlier sessions from
`session_search`, so stock Hermes could not search its own history. The harness was fixed (version
3.0), and no PAN-vs-stock number is published unless it was measured with the fixed harness. The
multi-session numbers in [evaluation.md](evaluation.md) all come from harness 3.0.

## ADR-016: Assistant-reported claims as their own provenance class

**Status:** accepted (2026-09-30).

- What the assistant said it did, decided or set up, without tool output to confirm it, used to be
  dropped. It is now kept as an `assistant_reported` claim: set only by deterministic post-checks,
  never by the model, prefixed `Assistant reported:` and labelled `(reported)`.
- Reported claims are never authoritative: they never strike observed or inferred facts and never
  reach `MEMORY.md` / `USER.md`. A newer report replaces only an older report of the same attribute.
- The same decision tightened the curator: a bullet is struck only on the gate's explicit
  `supersedes` or a contradiction of the same attribute.
- Open: expiry or confirmation of old reports, a long-horizon test, whether the agent treats the
  label as unverified, and documented precedence rules before the storage format is called stable.
