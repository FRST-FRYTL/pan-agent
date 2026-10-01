"""DeterministicCurator (integration spec §4.8): CREATE / UPDATE / IGNORE against the seed wiki."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from conftest import scenario_episodes
from pan.events.schema import CuratorAction, Destination, MemoryType
from pan.index.fts import FtsIndex
from pan.memory.classifier import RuleClassifier, classification
from pan.memory.curator import PREFERENCES_PAGE, DeterministicCurator, append_update
from pan.memory.episodes import previous_turn
from pan.memory.retrieval import FtsRetriever
from pan.wiki.frontmatter import Page
from pan.wiki.store import WikiStore


@pytest.fixture
def curator(seed_wiki: Path, tmp_path: Path):
    index = FtsIndex(tmp_path / "index.db")
    index.rebuild(seed_wiki)
    store = WikiStore(seed_wiki)
    yield DeterministicCurator(store, FtsRetriever(index), today=lambda: "2026-09-24")
    index.close()


def wiki_candidate(name: str):
    eps = scenario_episodes(name)
    cands = [c for i, ep in enumerate(eps) for c in RuleClassifier().classify(ep, previous_turn(eps, i))]
    return next(c for c in cands if c.classification.destination is not Destination.NONE)


def test_duplicate_is_ignored(curator):
    cur = curator.curate(wiki_candidate("duplicate"))
    d = cur.decision
    assert d.action is CuratorAction.IGNORE
    assert d.target_pages == ["learnings.vllm-tool-calling"]
    assert cur.page is None and d.patch == ""
    assert d.provenance == ["session:sess-dup", "event:01K0DUP0000000000000000001"]


def test_env_fact_updates_the_seed_learning(curator):
    cur = curator.curate(wiki_candidate("env_fact"))
    d = cur.decision
    assert d.action is CuratorAction.UPDATE
    assert d.target_pages == ["learnings.vllm-tool-calling"] and cur.path == "learnings/vllm-tool-calling.md"
    # the inferred explanation is already on the page; only the observed docker output is new
    assert len(cur.new_claims) == 1 and cur.new_claims[0].startswith("`docker inspect vllm-main`")
    assert d.evidence == "observed"
    body = cur.page.body
    assert body.index("## Updates") < body.index("### 2026-09-24 · session:sess-env")
    assert "--max-model-len 32768 --reasoning-parser qwen3 (observed)" in body
    assert "_Provenance: pan-memoryd, 2026-09-24 · session:sess-env, event:01K0ENV0000000000000000001" in body
    assert cur.page.meta["updated"] == "2026-09-24" and cur.page.meta["created"] == "2026-09-23"
    assert "event:01K0ENV0000000000000000002" in cur.page.meta["sources"]
    assert d.patch.startswith("--- a/learnings/vllm-tool-calling.md\n+++ b/learnings/vllm-tool-calling.md")
    assert "+- `docker inspect vllm-main` reports" in d.patch
    assert 0 < d.confidence <= 0.8


def test_decision_creates_next_adr(curator):
    cur = curator.curate(wiki_candidate("decision"))
    d = cur.decision
    assert d.action is CuratorAction.CREATE
    assert cur.path == "decisions/ADR-002-run-curation-separate-background-service.md"
    page = cur.page
    assert page.id == "decisions.adr-002" and page.type == "decision" and page.status == "accepted"
    assert page.title == "ADR-002: Run curation in a separate background service (pan-memoryd)"
    for heading in ("## Context", "## Decision", "## Consequences"):
        assert heading in page.body
    assert "(observed)" in page.body and "confirmed by the user" in page.body
    assert page.meta["sources"][0] == "session:sess-dec"
    assert d.patch.startswith("--- /dev/null\n+++ b/decisions/ADR-002-")


def test_decision_never_updates_non_decision_pages(curator):
    # the spool incident page is the best FTS hit for this candidate, but it is not a decision
    cand = wiki_candidate("decision")
    hits = curator.retriever.retrieve(cand.retrieval_query)
    assert hits and hits[0].type != "decision"
    assert curator.curate(cand).decision.action is CuratorAction.CREATE


def test_created_page_is_valid_and_found_again(curator, tmp_path):
    cand = wiki_candidate("decision")
    cur = curator.curate(cand)
    curator.store.write(cur.page)
    assert curator.store.validate() == []
    curator.retriever.index.update([curator.store.get_path(cur.path)])
    again = curator.curate(cand)
    assert again.decision.action is CuratorAction.IGNORE
    assert again.decision.target_pages == ["decisions.adr-002"]


def test_new_environment_fact_creates_page(curator):
    cand = wiki_candidate("env_fact")
    cand = replace(cand, normalized_claims=["The Grafana container listens on port 3100 behind nginx."],
                   claim_evidence=["inferred"], title="The Grafana container listens on port 3100 behind nginx",
                   retrieval_query="Grafana container listens port 3100 nginx", tags=["nginx"], subject="")
    cur = curator.curate(cand)
    assert cur.decision.action is CuratorAction.CREATE
    assert cur.path == "operations/the-grafana-container-listens-on-port-3100-behind-nginx.md"
    assert cur.page.type == "environment" and cur.page.status == "draft" and cur.page.meta["confidence"] == "low"
    assert "## Facts" in cur.page.body and "(inferred)" in cur.page.body
    assert cur.decision.evidence == "inferred"


def test_create_avoids_path_collisions(curator):
    cand = replace(wiki_candidate("env_fact"), normalized_claims=["Totally new fact about quokkas."],
                   claim_evidence=["observed"], title="vLLM serves without tool calling",
                   retrieval_query="quokkas", subject="")
    curator.store.write(Page(meta={"id": "operations.vllm-serves-without-tool-calling", "type": "environment",
                                   "status": "active", "created": "2026-09-23", "updated": "2026-09-23",
                                   "confidence": "low"}, body="# x\n",
                             path="operations/vllm-serves-without-tool-calling.md"))
    cur = curator.curate(cand)
    assert cur.decision.action is CuratorAction.CREATE
    assert cur.path == "operations/vllm-serves-without-tool-calling-2.md"
    assert cur.page.id == "operations.vllm-serves-without-tool-calling-2"


def test_l1_plan_and_wiki_fallback_for_preferences(curator):
    cands = [c for ep in scenario_episodes("preference") for c in RuleClassifier().classify(ep)]
    pref = cands[0]
    plan = curator.plan_l1(pref)
    assert plan.l1_target == "user" and plan.decision.target_pages == ["l1:user"]
    assert plan.l1_entries == ['User preference (own words): "Write meeting notes as Markdown files in the repo, '
                               'don\'t e-mail them unless I ask."']
    cur = curator.curate(pref, to_wiki=True)
    assert cur.decision.action is CuratorAction.CREATE and cur.path == PREFERENCES_PAGE
    curator.store.write(cur.page)
    assert curator.curate(pref, to_wiki=True).decision.action is CuratorAction.IGNORE
    other = replace(pref, normalized_claims=["Use British English in documentation."])
    upd = curator.curate(other, to_wiki=True)
    assert upd.decision.action is CuratorAction.UPDATE and upd.path == PREFERENCES_PAGE


def test_inactive_pages_are_not_update_targets(curator):
    page = curator.store.get("learnings.vllm-tool-calling")
    page.meta["status"] = "deprecated"
    curator.store.write(page)
    curator.retriever.index.update([curator.store.get("learnings.vllm-tool-calling")])
    cur = curator.curate(wiki_candidate("env_fact"))
    assert cur.decision.target_pages != ["learnings.vllm-tool-calling"]


def test_append_update_positions():
    assert append_update("# T\n\nbody\n", "- x") == "# T\n\nbody\n\n## Updates\n\n- x\n"
    body = "# T\n\n## Updates\n\n- old\n\n## Later\n\ntext\n"
    assert append_update(body, "- new") == "# T\n\n## Updates\n\n- old\n\n- new\n\n## Later\n\ntext\n"


def test_classification_profile():
    c = classification(MemoryType.NOISE)
    assert not c.should_remember and c.destination is Destination.NONE


def test_same_claim_ignores_pan_wrapper():
    from pan.memory.curator import l1_claim, same_claim

    wrapped = 'User preference (own words): "Always write dates in ISO 8601 format (YYYY-MM-DD)."'
    assert l1_claim(wrapped) == "Always write dates in ISO 8601 format (YYYY-MM-DD)."
    assert l1_claim("Prefers tabs.") == "Prefers tabs."
    assert same_claim(wrapped, "Prefers all dates written in ISO 8601 format (YYYY-MM-DD).")
    assert not same_claim(wrapped, "Prefers dates written as DD.MM.YYYY.")
    assert not same_claim(wrapped, 'User preference (own words): "Always answer in German."')


def test_l1_covered_is_directional_and_polarity_aware():
    from pan.memory.curator import l1_covered

    new = 'User preference (own words): "Always write dates in ISO 8601 format (YYYY-MM-DD)."'
    # the agent's own, wordier phrasing from the M4 live run covers PAN's claim
    assert l1_covered(new, "User prefers dates written in ISO 8601 format (YYYY-MM-DD) in all responses.")
    # a more specific new claim is not covered by a short existing one
    assert not l1_covered('User preference (own words): "Always use port 8000 for vLLM and 8001 for embeddings."',
                          "Uses port 8000 for vLLM.")
    # opposite polarity is never a duplicate
    assert not l1_covered("Never use port 8000 for vLLM.", "Always use port 8000 for vLLM.")
    assert not l1_covered(new, "Prefers dates written as DD.MM.YYYY.")
