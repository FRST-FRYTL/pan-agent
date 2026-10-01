"""Retrieval quality on realistic question phrasing (M5): tiny wikis, subject-titled pages, fusion.

In a benchmark run: the GPU fact sat on a page titled after the command line and prefetch
scored 0.00 for "which GPU model does this machine have?" — BM25's idf is ~0 in a one-page wiki.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pan.index.fts import FtsIndex, page_claims
from pan.memory.retrieval import PREFETCH_MIN_SCORE, FtsRetriever, FusedRetriever, Retriever
from pan.wiki.frontmatter import Page
from pan.wiki.store import INDEX_TEMPLATE, WikiStore, new_page

GPU_BODY = """# GPU

User asked: "Run `nvidia-smi --query-gpu=name --format=csv,noheader` and tell me which GPU this machine has.".

## Facts

- GPU (`nvidia-smi --query-gpu=name --format=csv,noheader`) reports: NVIDIA GB10 (observed)

## Updates
"""
VLLM_BODY = """# vLLM server

User asked: "Note for later: our vLLM server runs on port 8000.".

## Facts

- ~~Our vLLM server runs on port 8000.~~ (observed; superseded 2026-09-24 by session:s2)

## Updates

### 2026-09-24 · session:s2

- We moved the vLLM server to port 8010. (observed)
"""
LANGFUSE_BODY = """# Langfuse instance

## Facts

- The Langfuse instance for this project runs on host spark01 at port 3300. (observed)
"""


def _page(pid: str, title: str, body: str, tags, subject: str) -> Page:
    p = new_page("environment", title, page_id=pid, path=f"operations/{pid.split('.')[-1]}.md", tags=tags,
                 status="active", date="2026-09-24", body=body)
    p.meta["subject"] = subject
    return p


def _wiki(tmp_path: Path, pages) -> FtsIndex:
    root = tmp_path / "wiki"
    store = WikiStore(root)
    root.mkdir()
    store.index_path.write_text(INDEX_TEMPLATE.format(today="2026-09-24"))
    for p in pages:
        store.write(p)
    idx = FtsIndex(tmp_path / "index.db")
    idx.rebuild(root)
    return idx


GPU = ("operations.gpu", "GPU", GPU_BODY, ["gpu", "nvidia", "hardware"], "GPU")
VLLM = ("operations.vllm-server", "vLLM server", VLLM_BODY, ["vllm"], "vLLM server")
LANGFUSE = ("operations.langfuse-instance", "Langfuse instance", LANGFUSE_BODY, ["langfuse"], "Langfuse instance")

QUESTIONS = [
    ("Without running any commands: which GPU model does this machine have?", "operations.gpu"),
    ("which GPU does this machine have?", "operations.gpu"),
    ("Which port does our vLLM server use? Answer from memory in one sentence.", "operations.vllm-server"),
    ("What host and port is our Langfuse instance on? Answer from memory.", "operations.langfuse-instance"),
]


@pytest.mark.parametrize("pages", [[GPU], [GPU, VLLM], [GPU, VLLM, LANGFUSE]], ids=["1-page", "2-pages", "3-pages"])
def test_questions_find_their_page_in_tiny_wikis(tmp_path, pages):
    idx = _wiki(tmp_path, [_page(*p) for p in pages])
    try:
        ids = {p[0] for p in pages}
        for question, expected in QUESTIONS:
            if expected not in ids:
                continue
            hits = idx.search(question)
            assert hits and hits[0].id == expected, (question, [(h.id, h.score) for h in hits])
            assert hits[0].score >= PREFETCH_MIN_SCORE, (question, hits[0].score)
    finally:
        idx.close()


@pytest.mark.parametrize("question", [
    "What is 2 + 2? Reply with just the number.",
    "Summarize what a hash map does.",
    "tell me a joke about cats",
    "By the way, my neighbour's cat is called Mr. Whiskers. What is 17 times 3?",
])
def test_unrelated_questions_stay_below_the_prefetch_threshold(tmp_path, question):
    idx = _wiki(tmp_path, [_page(*p) for p in (GPU, VLLM, LANGFUSE)])
    try:
        assert all(h.score < PREFETCH_MIN_SCORE for h in idx.search(question)), question
    finally:
        idx.close()


def test_claims_column_skips_superseded_text():
    page = _page(*VLLM)
    claims = page_claims(page)
    assert "8010" in claims and "8000" not in claims and claims.startswith("vLLM server")


class _Static(Retriever):
    def __init__(self, hits):
        self.hits = hits

    def retrieve(self, query, k=5, *, type=None):
        return self.hits[:k]


class _Broken(Retriever):
    def retrieve(self, query, k=5, *, type=None):
        raise RuntimeError("embedding service down")


def test_fused_retriever_orders_by_rrf_and_keeps_best_score(tmp_path):
    idx = _wiki(tmp_path, [_page(*p) for p in (GPU, VLLM, LANGFUSE)])
    try:
        fts = FtsRetriever(idx)
        fts_hits = fts.retrieve("vllm server port")
        assert fts_hits[0].id == "operations.vllm-server"
        # a "vector" retriever that prefers the Langfuse page
        from dataclasses import replace
        vec = _Static([replace(fts_hits[0], id="operations.langfuse-instance", score=0.9),
                       replace(fts_hits[0], score=0.3)])
        fused = FusedRetriever([(fts, 1.0), (vec, 1.0), (_Broken(), 1.0)]).retrieve("vllm server port", 3)
        assert fused[0].id == "operations.vllm-server"          # rank 1 + rank 2 beats rank 1 alone
        assert fused[0].score == max(fts_hits[0].score, 0.3)
        assert any(h.id == "operations.langfuse-instance" and h.score == 0.9 for h in fused)
        only_fts = FusedRetriever([(fts, 1.0)]).retrieve("vllm server port", 3)
        assert [h.id for h in only_fts] == [h.id for h in fts.retrieve("vllm server port", 3)]
    finally:
        idx.close()
