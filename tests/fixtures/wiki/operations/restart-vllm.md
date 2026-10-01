---
id: operations.restart-vllm
type: operations
status: active
created: 2026-09-22
updated: 2026-09-22
confidence: medium
sources:
  - session:sess-ops
related:
  - learnings/vllm-tool-calling.md
tags: [vllm, docker, runbook]
---

# Restart vLLM with tool calling enabled

## Procedure

1. Stop the container: `docker stop vllm-main`.
2. Start it again with `--enable-auto-tool-choice --tool-call-parser hermes` added to the serve
   command.
3. Check that `curl localhost:8000/v1/models` lists `primary`.

## Rollback

Start the previous container image without the extra flags.
