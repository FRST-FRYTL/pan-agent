---
id: systems.dgx-spark-host
type: system
status: active
created: 2026-09-20
updated: 2026-09-23
confidence: high
sources:
  - observed:nvidia-smi
related:
  - systems/langfuse.md
  - learnings/vllm-tool-calling.md
tags: [dgx-spark, hardware, gpu]
---

# DGX Spark host

Single NVIDIA GB10 machine (aarch64) with about 121 GB of unified memory shared by CPU and GPU.

## Services

| Service | Port | Notes |
|---|---|---|
| vLLM | 8000 | model `primary` |
| Langfuse | 3000 | tracing, see [Langfuse](langfuse.md) |

## Memory budget

The inference server reserves 75 % of unified memory (`--gpu-memory-utilization 0.75`), which
leaves roughly 30 GB for the agent, embeddings and the rest of the host.
