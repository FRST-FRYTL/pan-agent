---
id: decisions.adr-001
type: decision
status: accepted
created: 2026-09-21
updated: 2026-09-22
confidence: high
sources:
  - session:sess-decision
related:
  - systems/dgx-spark-host.md
tags: [retrieval, sqlite, fts5, bm25]
---

# ADR-001: SQLite FTS5 for retrieval in the MVP

## Context

The Memory Plane needs keyword retrieval over the wiki without running another service. A vector
database would add a process and an embedding model before we know the query mix.

## Decision

Use SQLite FTS5 with BM25 ranking and the porter tokenizer as the only retrieval backend in the
MVP. The index is derived from the wiki and can be rebuilt at any time.

## Consequences

- No extra service; the index file lives next to the wiki.
- Semantic matches (synonyms) are missed until embeddings and a reranker are added (hybrid fusion).
