# Changelog

All notable changes to pan-agent. Versions follow [PEP 440](https://peps.python.org/pep-0440/); git
tags are `pan-v<version>`. Each release is one squashed commit exported from the development
workspace.

## [0.1.0a1] - 2026-10-01

First public research preview. It is usable, not a supported product. APIs, configuration keys and
the storage format may change. Installed from the git tag `pan-v0.1.0a1`; there is no PyPI release.

**Tested with:** Hermes Agent 0.21.4, upstream commit `54c55307e3285a02f21185c1fd5b1f694489af80`.
Other Hermes versions may work with `pan setup --force`, but they are untested.
**Reference model stack:** Qwen3.8-27B-NVFP4 on vLLM 0.30.1rc1 + PR #50021 (MTP 3, prefix caching).
Any OpenAI-compatible model works.

### What's in it
- Memory provider plugin `pan` for Hermes, registered through the `hermes_agent.memory_providers`
  entry point. It includes hook-based capture into a durable SQLite event spool, a prefetch block,
  and the `memory_search` / `memory_read` tools.
- `pan-memoryd`, the background curator. For each finished turn, a model-based memory gate
  (`classifier.kind: llm`, any OpenAI-compatible endpoint) decides what is worth keeping, with
  deterministic post-checks for grounding, secrets and echo. The rule-based classifier takes over
  when the endpoint fails, and a circuit breaker protects against repeated failures. A deterministic
  curator then updates the wiki and Hermes' `MEMORY.md` / `USER.md`. The rules-only gate
  (`classifier.kind: rules`) applies the same secret filter as the LLM gate.
- The memory wiki: Markdown pages with YAML frontmatter and provenance, in a git repo under
  `$HERMES_HOME/pan/wiki`, with a derived SQLite FTS5 index and, for hybrid retrieval, a derived
  vector index (`vectors.db`).
- The `pan` CLI:
  - `setup`: dry run, a settings backup, and a refusal of untested Hermes versions unless `--force`;
  - `uninstall`: restores the backed-up settings;
  - `status`, `version`, `wiki`, `index`, `memory inspect|replay`, `memoryd run`;
  - `daemon install|uninstall`: a systemd user unit, never auto-enabled.
- `pan doctor`: checks an OpenAI-compatible endpoint. It covers reachability and model id, a streamed
  tool-call probe, a repeated tool-call corruption check and `json_schema` support.
- `skills/install-pan`: an agent skill for a guided install, shipped with a Claude Code plugin
  marketplace manifest.
- The tested vLLM server setup lives in its own repo,
  [spark-vllm-agent-cookbook](https://github.com/FRST-FRYTL/spark-vllm-agent-cookbook) (`v0.1.0`).
- Docs: the architecture (`docs/architecture.md`), the evaluation method and results
  (`docs/evaluation.md`), what PAN needs from a local model (`docs/local-models.md`), how the project
  is developed (`docs/development.md`), and a scripted two-session demo with a recording
  (`docs/demo/`).
- Unit, contract and end-to-end tests. The e2e tests drive the real Hermes with a scripted fake model.

### Memory quality work before release (September 2026 improvement round)
- **Hybrid retrieval** (`retrieval.mode: hybrid`, the default): FTS5 plus multilingual embeddings,
  fused with reciprocal-rank fusion and reranked by a cross-encoder, for prefetch and
  `memory_search`; the curator uses the same fusion, without reranking, to find the page to update.
  Embeddings and reranking come from the local `pan-embedd` sidecar, which is **not packaged in this
  release**. Without it, retrieval falls back to FTS5 only (with a circuit breaker and a logged
  warning); `retrieval.mode: fts` turns it off. `pan index rebuild` also re-embeds the wiki, and
  `pan status` reports the vector index.
- **Assistant-reported claims:** what the assistant said it did, decided or set up is kept as an
  `Assistant reported:` claim, labelled `(reported)`, instead of being dropped. Reported claims
  never override observed or inferred facts and never reach `MEMORY.md` / `USER.md`.
- **Curator supersession:** a bullet is struck only on the gate's explicit `supersedes` or a
  contradiction of the same attribute of the same subject; claims about different occasions no
  longer supersede each other. This fixes still-valid claims being struck when a second claim about
  the same subject landed on a page.
- **User facts and preferences:** gate post-checks no longer drop user facts because of the gate's
  own third-person rewrite, a hedge in another clause, or a fact stated inside a question; standing
  constraints (health triggers, budget limits, house rules) count as preferences; dated tool records
  (log lines, CSV rows) are kept long-term. The gate prompt is `llm-gate-v4`. L1 reconciliation
  follows the gate's explicit `supersedes` and replaces only the affected sentence of a
  multi-sentence entry.
- **Presentation of recalled memory:** the stored `(stated DATE)` stamp is shown as `(noted DATE)`;
  ids and provenance lines are left out; `memory_read` returns the current facts with superseded
  ones listed last; marker explanations appear only where a marker does; active preferences are
  ordered by relevance to the request. The system prompt block now says that the newest statement
  wins, that the user profile and listed pages are checked before saying there is no record, and
  that values must not be invented.
- **Token overhead:** a focused prefetch query (absolute paths shortened, long messages also scored
  per paragraph), a relative rerank floor, at most 4 prefetched pages, an overview that lists only
  the wiki's pages, and compact tool schemas. PAN's fixed overhead per request fell from about 680
  to about 440 tokens.
- **Hosted endpoints:** the gate drops `chat_template_kwargs` after an endpoint rejects it
  (`models.gate.template_kwargs: auto | always | never`).
- **Evaluation:** all multi-session stock-vs-PAN numbers were re-measured with a fixed harness that lets stock
  Hermes search its own sessions, and the held-out test sets were run once after the round (see
  `docs/evaluation.md`).

### Known limitations
See the README's [Known limitations](README.md#known-limitations).
