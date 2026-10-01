---
id: incidents.2026-09-20-spool-disk-full
type: incident
status: active
created: 2026-09-20
updated: 2026-09-20
confidence: high
sources:
  - session:sess-incident
related: []
tags: [spool, disk, sqlite]
---

# Event spool disk full

## Symptoms

Hook writes failed with `database or disk is full`; the agent kept working but events were dropped.

## Cause

Old `done` events were never pruned, and the root partition filled up with logs.

## Resolution

Pruned `done` events older than 30 days and moved logs to a separate volume.
