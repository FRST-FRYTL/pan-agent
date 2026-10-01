---
id: systems.langfuse
type: system
status: active
created: 2026-09-20
updated: 2026-09-20
confidence: medium
sources:
  - observed:docker-ps
related: []
tags: [langfuse, observability, tracing, postgres]
---

# Langfuse

Langfuse v2 records agent traces. It runs in Docker with a Postgres 16 database and listens on
port 3000. Version 2 needs no ClickHouse, which keeps it light but it only gets maintenance fixes.

## Credentials

API keys are stored in the Hermes secret store, never in the wiki.
