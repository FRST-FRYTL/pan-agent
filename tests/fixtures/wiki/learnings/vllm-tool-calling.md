---
id: learnings.vllm-tool-calling
type: learning
status: active
created: 2026-09-23
updated: 2026-09-23
confidence: high
sources:
  - session:sess-env
related: []
tags: [vllm, tool-calling, dgx-spark]
---

# vLLM serves without tool calling

vLLM on the DGX Spark serves `RedHatAI/Qwen3.6-35B-A3B-NVFP4` as `primary` on port 8000, but was
started without `--enable-auto-tool-choice --tool-call-parser`, so tool calls fail. Hermes needs
tool calling. (observed)
