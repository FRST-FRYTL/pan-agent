"""DeterministicCurator M5: subject pages, supersession, inferred-vs-observed, provenance links."""

from __future__ import annotations

from pathlib import Path

import pytest

from pan.events.schema import CuratorAction, MemoryCandidate, MemoryType
from pan.index.fts import FtsIndex
from pan.memory.classifier import classification
from pan.memory.curator import DeterministicCurator, live_text, page_bullets
from pan.memory.retrieval import FtsRetriever
from pan.wiki.store import INDEX_TEMPLATE, WikiStore


@pytest.fixture
def env(tmp_path: Path):
    root = tmp_path / "wiki"
    store = WikiStore(root)
    store.init(git=False)
    store.index_path.write_text(INDEX_TEMPLATE.format(today="2026-09-23"))
    index = FtsIndex(tmp_path / "index.db")
    index.rebuild(root)
    curator = DeterministicCurator(store, FtsRetriever(index), today=lambda: "2026-09-24")

    def apply(cur):
        store.write(cur.page)
        index.update([store.get_path(cur.path)])
        return store.get_path(cur.path)
    yield curator, apply
    index.close()


def cand(claim: str, *, evidence: str = "observed", subject: str = "vLLM server", session: str = "s1",
         event: str = "01K0CUR0000000000000000001", supersedes=()) -> MemoryCandidate:
    return MemoryCandidate(event_ids=[event], classification=classification(MemoryType.ENVIRONMENT),
                           normalized_claims=[claim], retrieval_query=f"{subject} {claim}", id=f"{event}:env",
                           session_id=session, title=subject, tags=["vllm"], claim_evidence=[evidence],
                           subject=subject, supersedes=list(supersedes))


def test_fact_update_supersedes_the_old_bullet(env):
    curator, apply = env
    first = curator.curate(cand("Our vLLM server runs on port 8000."))
    assert first.decision.action is CuratorAction.CREATE and first.path == "operations/vllm-server.md"
    assert first.page.title == "vLLM server" and first.page.meta["subject"] == "vLLM server"
    apply(first)
    second = curator.curate(cand("We moved the vLLM server to port 8010.", session="s2",
                                 event="01K0CUR0000000000000000002"))
    d = second.decision
    assert d.action is CuratorAction.UPDATE and d.target_pages == ["operations.vllm-server"]
    assert "same subject" in d.rationale and "supersedes 1 older claim" in d.rationale
    page = apply(second)
    assert "- ~~Our vLLM server runs on port 8000.~~ (observed; superseded 2026-09-24 by session:s2)" in page.body
    assert [b for _, b, _ in page_bullets(page.body)] == ["We moved the vLLM server to port 8010."]
    assert "8000" not in live_text(page.body).split("## Updates")[1]
    # the old value is no longer "present": reverting to 8000 later is an update again, not an IGNORE
    back = curator.curate(cand("Our vLLM server runs on port 8000 again.", event="01K0CUR0000000000000000003"))
    assert back.decision.action is CuratorAction.UPDATE


def test_inferred_claim_never_overrides_an_observed_fact(env):
    curator, apply = env
    apply(curator.curate(cand("We moved the vLLM server to port 8010.")))
    stale = curator.curate(cand("Our vLLM server runs on port 8000.", evidence="inferred",
                                event="01K0CUR0000000000000000009"))
    assert stale.decision.action is CuratorAction.IGNORE
    assert "contradict observed facts" in stale.decision.rationale


def test_explicit_agent_replace_supersedes_even_when_inferred(env):
    curator, apply = env
    apply(curator.curate(cand("vLLM server runs on port 8000.", evidence="inferred")))
    cur = curator.curate(cand("vLLM server runs on port 8010.", evidence="inferred", event="01K0CUR0000000000000000004",
                              supersedes=["vLLM server runs on port 8000."]))
    page = apply(cur)
    assert "~~vLLM server runs on port 8000.~~ (inferred; superseded" in page.body


def test_contradicted_page_is_the_target_even_without_subject_match(env):
    curator, apply = env
    apply(curator.curate(cand("The Langfuse instance runs on host spark01 at port 3300.", subject="Langfuse instance")))
    moved = curator.curate(cand("Langfuse moved to port 3400.", subject="Langfuse", event="01K0CUR0000000000000000005"))
    assert moved.decision.action is CuratorAction.UPDATE
    assert moved.decision.target_pages == ["operations.langfuse-instance"]


def test_subject_candidates_do_not_update_unrelated_pages(env):
    curator, apply = env
    apply(curator.curate(cand("The DGX Spark host serves vLLM and Langfuse on ports 8000 and 3000.",
                              subject="DGX Spark host")))
    gpu = curator.curate(cand("GPU (`nvidia-smi`) reports: NVIDIA GB10", subject="GPU",
                              event="01K0CUR0000000000000000006"))
    assert gpu.decision.action is CuratorAction.CREATE and gpu.path == "operations/gpu.md"


def test_link_provenance_only_touches_sources(env):
    curator, apply = env
    page = apply(curator.curate(cand("Our vLLM server runs on port 8000.")))
    agent = cand("vLLM server runs on port 8000.", evidence="inferred", event="01K0CUR0000000000000000007")
    ignored = curator.curate(agent)
    assert ignored.decision.action is CuratorAction.IGNORE
    linked = curator.link_provenance(agent, ignored.decision.target_pages[0], extra=["l1:memory"])
    assert linked.page.body == page.body
    assert linked.page.meta["sources"][-2:] == ["event:01K0CUR0000000000000000007", "l1:memory"]
    apply(linked)
    assert curator.link_provenance(agent, page.id, extra=["l1:memory"]) is None  # nothing new to add
