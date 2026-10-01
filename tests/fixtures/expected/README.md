# Expected outputs (golden files)

One JSON file per scenario in `../events/`, written by `tests/e2e/test_pipeline_golden.py`
(each scenario runs alone: seed wiki, fixture `hermes_home`, daemon clock fixed at 2026-09-24T12:00Z).
Each file holds the curation-log records (classification, claims, decision, outcome), the wiki
pages the daemon's commit created/updated (with their full text), whether they are indexed, and
the entries appended to `USER.md` and the final `MEMORY.md` entries. `l1_write` events are
applied to the L1 files through Hermes' MemoryStore before the daemon runs (the agent's tool
wrote them in the real system). Volatile values (timestamps, batch/commit ids, patches) are not
compared.

| Scenario | Expected classification | Expected decision |
|---|---|---|
| `preference.jsonl` | `user_preference` → `user` | L1 `USER.md` add |
| `env_fact.jsonl` | `environment` → `wiki` | UPDATE `learnings/vllm-tool-calling.md` (seed): the docker-inspect output is appended under `## Updates`; the explanation is already on the page |
| `decision.jsonl` | turn 1 `noise` (unconfirmed proposal), turn 2 `decision` → `wiki` | CREATE `decisions/ADR-002-…` (status `accepted`) |
| `duplicate.jsonl` | `environment` → `wiki` | IGNORE (already in seed page) |
| `noise.jsonl` | `noise` → `none` (twice: chit-chat turn, failed `ls` without a fix) | drop |
| `fact_update.jsonl` (M5) | 3 sessions: user fact (+ the agent's own `memory` write of it), user update (assistant *printed* the tool call), stale answer from L1 | CREATE `operations/vllm-server.md` (+ `index.md` link); the agent's L1 write links provenance (`l1:memory`); UPDATE strikes the port-8000 bullet and supersedes the `MEMORY.md` entry → port 8010; the stale answer is dropped |
| `toolcall_text.jsonl` (M5) | printed tool calls (qwen3_coder XML, `Memory (user):`, JSON + call syntax) next to a chit-chat turn, a decision and a fact | the chit-chat turn is `noise`; ADR-002 and `operations/grafana-container.md` from the user's words only |
| `gpu_fact.jsonl` (M5) | `terminal({})` error + retry, `nvidia-smi` JSON output | CREATE `operations/gpu.md` titled "GPU" (tags gpu/nvidia/hardware), no learning page |

Regenerate after an intended behaviour change, then review the diff:

```bash
PAN_UPDATE_GOLDEN=1 python -m pytest tests/e2e/test_pipeline_golden.py
```
