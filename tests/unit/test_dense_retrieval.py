"""M7–M8 (ADR-012): dense vectors, hybrid fusion, reranking and their fallbacks, against a fake sidecar."""

from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path

import pytest

from conftest import M3_SCENARIOS, fixed_clock, spool_scenarios
from fake_embedd import FakeEmbedd
from pan.config import PanConfig, RetrievalConfig, load_config
from pan.index.embed import EmbedClient, EmbedError
from pan.index.fts import FtsIndex
from pan.index.vectors import VectorIndex, page_units, sync_wiki
from pan.memory.reader import Reader
from pan.memory.retrieval import (DenseRetriever, FtsRetriever, FusedRetriever, RerankRetriever,
                                  build_retriever)
from pan.paths import PanPaths
from pan.wiki.store import INDEX_TEMPLATE, WikiStore, new_page

PORT_BODY = """# Langfuse instance

User asked: "Remember where Langfuse runs."

## Facts

- ~~Langfuse runs on port 3000.~~ (observed; superseded 2026-09-24 by session:s2)
- The Langfuse instance runs on port 3300. (observed)

## Updates

_Provenance: pan-memoryd, 2026-09-24 · session:s1_
"""
DOG_BODY = """# Pets

## Facts

- The user's dog is called Bruno. (observed)
- Bruno is a beagle. (observed)
"""
TRIP_BODY = """# Travel plans

We plan a long trip to Portugal next spring; the dates are not fixed yet and depend on work.
"""


def _cfg(url: str, **kw) -> RetrievalConfig:
    return dataclasses.replace(RetrievalConfig(), mode="hybrid", curator_mode="hybrid", embed_url=url,
                               dense_floor=0.0, dense_ceiling=1.0, **kw)


def _home(tmp_path: Path) -> PanPaths:
    paths = PanPaths.for_home(tmp_path / "home")
    paths.ensure()
    store = WikiStore(paths.wiki)
    store.index_path.write_text(INDEX_TEMPLATE.format(today="2026-09-24"))
    for pid, title, body, tags in (("systems.langfuse", "Langfuse instance", PORT_BODY, ["langfuse"]),
                                   ("projects.pets", "Pets", DOG_BODY, ["dog"]),
                                   ("projects.travel", "Travel plans", TRIP_BODY, ["travel"])):
        store.write(new_page("project", title, page_id=pid, path=f"projects/{pid.split('.')[-1]}.md", tags=tags,
                             status="active", date="2026-09-24", body=body))
    FtsIndex(paths.index_db).rebuild(paths.wiki)
    return paths


@pytest.fixture
def fake():
    with FakeEmbedd() as f:
        yield f


# -- units -------------------------------------------------------------------------------------------

def test_units_are_card_current_claims_and_prose():
    page = new_page("project", "Langfuse instance", page_id="systems.langfuse", path="p.md", tags=["langfuse"],
                    status="active", date="2026-09-24", body=PORT_BODY)
    units = page_units(page)
    assert units[0].kind == "card" and units[0].text.startswith("Langfuse instance")
    claims = [u.text for u in units if u.kind == "claim"]
    assert claims == ["The Langfuse instance runs on port 3300."]  # struck item and label dropped
    assert all(u.embed_text.startswith("Langfuse instance") for u in units)
    prose = [u.text for u in units if u.kind == "text"]
    assert prose and all("Provenance" not in t for t in prose)
    assert page_units(page) == units  # deterministic


# -- index ---------------------------------------------------------------------------------------------

def test_sync_embeds_once_reuses_and_drops(tmp_path, fake):
    paths = _home(tmp_path)
    client = EmbedClient(fake.url)
    idx = VectorIndex(paths.vectors_db)
    pages = WikiStore(paths.wiki).list_pages()
    first = idx.sync(pages, client)
    assert first["embedded_pages"] == 3 and first["embedded_units"] > 0
    assert idx.sync(pages, client)["embedded_units"] == 0  # unchanged wiki: nothing to embed
    status = VectorIndex(paths.vectors_db, readonly=True).status(paths.wiki)
    assert status["usable"] and status["pages"] == 3 and not status["stale"]
    assert status["model_id"] == "fake/concept-embedder"
    (paths.wiki / "projects/travel.md").unlink()
    again = idx.sync(WikiStore(paths.wiki).list_pages(), client)
    assert again["removed"] == 1 and again["embedded_units"] == 0


def test_model_change_reembeds_everything_and_reader_ignores_foreign_vectors(tmp_path, fake):
    paths = _home(tmp_path)
    sync_wiki(paths, _cfg(fake.url))
    fake.revision = "r2"  # the sidecar now serves another model revision
    client = EmbedClient(fake.url)
    dense = DenseRetriever(VectorIndex(paths.vectors_db, readonly=True), client, floor=0.0, ceiling=1.0)
    assert dense.retrieve("Hund", 3) == []  # no mixed vector spaces
    stats = VectorIndex(paths.vectors_db).sync(WikiStore(paths.wiki).list_pages(), EmbedClient(fake.url))
    assert stats["embedded_pages"] == 3 and stats["reused_units"] == 0
    assert dense.retrieve("Hund", 3)[0].id == "projects.pets"


def test_rebuild_is_deterministic(tmp_path, fake):
    paths = _home(tmp_path)
    cfg = _cfg(fake.url)
    sync_wiki(paths, cfg, rebuild=True)
    a = VectorIndex(paths.vectors_db, readonly=True).status()
    sync_wiki(paths, cfg, rebuild=True)
    b = VectorIndex(paths.vectors_db, readonly=True).status()
    assert {k: a[k] for k in ("pages", "units", "model_id")} == {k: b[k] for k in ("pages", "units", "model_id")}


# -- retrieval -----------------------------------------------------------------------------------------

def test_dense_finds_a_cross_language_match_fts_misses(tmp_path, fake):
    paths = _home(tmp_path)
    sync_wiki(paths, _cfg(fake.url))
    fts = FtsRetriever(FtsIndex(paths.index_db, readonly=True))
    assert fts.retrieve("Wie heißt mein Hund?", 3) == []
    dense = DenseRetriever(VectorIndex(paths.vectors_db, readonly=True), EmbedClient(fake.url), floor=0.0, ceiling=1.0)
    hits = dense.retrieve("Wie heißt mein Hund?", 3)
    assert hits[0].id == "projects.pets" and hits[0].via == "dense"
    assert "Bruno" in hits[0].snippet and hits[0].section == "Facts"
    fused = FusedRetriever([(fts, 1.0), (dense, 1.0)])
    assert fused.retrieve("Wie heißt mein Hund?", 3)[0].id == "projects.pets"


def test_fused_degrades_to_fts_when_the_sidecar_is_down(tmp_path, fake):
    paths = _home(tmp_path)
    sync_wiki(paths, _cfg(fake.url))
    cfg = _cfg("http://127.0.0.1:9")  # nothing listens there
    retriever = build_retriever(paths, cfg, purpose="reader")
    fts = FtsRetriever(FtsIndex(paths.index_db, readonly=True))
    q = "Which port does Langfuse use?"
    assert [h.id for h in retriever.retrieve(q, 3)] == [h.id for h in fts.retrieve(q, 3)]


def test_client_breaker_stops_calling_a_dead_sidecar():
    client = EmbedClient("http://127.0.0.1:9", cooldown_s=60)
    with pytest.raises(EmbedError):
        client.embed(["x"])
    t0 = time.monotonic()
    with pytest.raises(EmbedError, match="cooldown"):
        client.embed(["x"])
    assert time.monotonic() - t0 < 0.05
    assert client.health() is None


def test_rerank_reorders_and_falls_back_within_budget(tmp_path, fake):
    paths = _home(tmp_path)
    sync_wiki(paths, _cfg(fake.url))
    reader = Reader(paths)
    client = EmbedClient(fake.url)

    class Fixed(FtsRetriever):  # a base order with the right page last
        def retrieve(self, query, k=5, *, type=None):
            hits = FtsRetriever(FtsIndex(paths.index_db, readonly=True)).retrieve("langfuse pets travel plans", 5)
            return sorted(hits, key=lambda h: h.id != "projects.travel")

    rr = RerankRetriever(Fixed(None), client, reader.rerank_text, depth=5, timeout_s=2.0)
    hits = rr.retrieve("Wann ist unser Urlaub?", 3)
    assert hits[0].id == "projects.travel" and hits[0].via == "rerank"
    fake.rerank_delay = 0.5
    slow = RerankRetriever(Fixed(None), client, reader.rerank_text, depth=5, timeout_s=0.1)
    hits = slow.retrieve("Wann ist unser Urlaub?", 3)
    assert all(h.via == "fts" for h in hits)  # over budget: base order
    fake.rerank_delay = 0.0
    assert client.embed(["still up"])  # one timeout does not open the breaker


# -- reader / provider path ---------------------------------------------------------------------------

def test_reader_prefetch_uses_hybrid_and_rerank_thresholds(tmp_path, fake):
    paths = _home(tmp_path)
    cfg = _cfg(fake.url, rerank_min_score=0.5, search_min_score=0.3)
    sync_wiki(paths, cfg)
    reader = Reader(paths, config=cfg)
    out = reader.prefetch("Wie heißt mein Hund?")
    assert "[projects.pets]" in out and "Bruno" in out
    assert "[systems.langfuse]" not in out
    assert reader.prefetch("Erzähl mir etwas über Quantenphysik") == ""
    res = reader.search("Hund", limit=5)
    assert [r["id"] for r in res["results"]] == ["projects.pets"]


def test_reader_without_config_stays_fts_only(tmp_path, fake):
    paths = _home(tmp_path)
    sync_wiki(paths, _cfg(fake.url))
    assert Reader(paths).prefetch("Wie heißt mein Hund?") == ""
    assert fake.calls["rerank"] == 0


def test_shipped_default_is_hybrid(pin_fts_retrieval):
    assert pin_fts_retrieval == "hybrid"
    assert RetrievalConfig().mode == "fts"  # pinned for the test session


def test_config_keys_parse(tmp_path):
    paths = PanPaths.for_home(tmp_path)
    paths.ensure()
    paths.config.write_text("retrieval:\n  mode: fts\n  rerank: false\n  rerank_timeout_s: 0.25\n")
    cfg = load_config(paths).retrieval
    assert (cfg.mode, cfg.rerank, cfg.rerank_timeout_s) == ("fts", False, 0.25)


# -- daemon --------------------------------------------------------------------------------------------

def test_daemon_keeps_vectors_in_sync_and_curator_uses_them(pan_home, fake):
    from pan.daemon.memoryd import MemoryDaemon

    paths = PanPaths.for_home(pan_home)
    cfg = dataclasses.replace(PanConfig(), retrieval=_cfg(fake.url))
    daemon = MemoryDaemon(paths, config=cfg, clock=fixed_clock, worker_id="test")
    daemon.prepare()
    assert VectorIndex(paths.vectors_db, readonly=True).status(paths.wiki)["stale"] is False
    spool_scenarios(pan_home, M3_SCENARIOS)
    daemon.drain(flush=True)
    status = VectorIndex(paths.vectors_db, readonly=True).status(paths.wiki)
    assert status["usable"] and not status["stale"], status
    assert fake.calls["embeddings"] >= 2
    daemon.close()


def test_daemon_works_without_sidecar(pan_home):
    from pan.daemon.memoryd import MemoryDaemon

    paths = PanPaths.for_home(pan_home)
    cfg = dataclasses.replace(PanConfig(), retrieval=_cfg("http://127.0.0.1:9"))
    daemon = MemoryDaemon(paths, config=cfg, clock=fixed_clock, worker_id="test")
    daemon.prepare()
    spool_scenarios(pan_home, M3_SCENARIOS)
    reports = daemon.drain(flush=True)
    assert sum(r.done for r in reports) > 0
    assert not paths.vectors_db.exists() or not VectorIndex(paths.vectors_db, readonly=True).status()["usable"]
    daemon.close()


def test_cli_rebuild_writes_vectors(pan_home, fake, capsys, monkeypatch):
    from pan.cli import main

    paths = PanPaths.for_home(pan_home)
    paths.config.write_text(f"retrieval:\n  mode: hybrid\n  embed_url: {fake.url}\n")
    monkeypatch.setenv("HERMES_HOME", str(pan_home))
    assert main(["index", "rebuild"]) == 0
    assert "embedded" in capsys.readouterr().out
    assert main(["index", "status", "--json"]) == 0
    info = json.loads(capsys.readouterr().out)
    assert info["vectors"]["usable"] and not info["vectors"]["stale"]


def test_reranked_hit_kept_on_strong_prior(tmp_path):
    from pan.index.fts import Hit
    from pan.memory.retrieval import Retriever

    paths = _home(tmp_path)

    class Stub(Retriever):
        def retrieve(self, query, k=5, *, type=None):
            base = dict(path="projects/pets.md", title="Pets", type="project", status="active", bm25=0.0,
                        snippet="The user's dog is called Bruno.", via="rerank")
            return [Hit(id="projects.pets", score=0.0, prior=0.6, **base),
                    Hit(id="projects.travel", score=0.0, prior=0.1, **dict(base, path="projects/travel.md"))]

    cfg = _cfg("http://127.0.0.1:9")
    out = Reader(paths, retriever=Stub(), config=cfg).prefetch("Wie heißt mein Hund?")
    assert "[projects.pets]" in out and "[projects.travel]" not in out


def test_prefetch_without_sidecar_is_fts_and_fast(tmp_path, fake):
    """Sidecar stopped: the hybrid reader answers like FTS, without errors or a noticeable delay."""
    paths = _home(tmp_path)
    sync_wiki(paths, _cfg(fake.url))
    dead = _cfg("http://127.0.0.1:9")
    fts_reader = Reader(paths)
    reader = Reader(paths, config=dead)
    q = "Which port does the Langfuse instance use?"
    t0 = time.monotonic()
    out = [reader.prefetch(q) for _ in range(5)]
    res = reader.search(q)
    assert time.monotonic() - t0 < 0.5
    assert out[0] == fts_reader.prefetch(q) and out[0]
    assert [r["id"] for r in res["results"]] == [r["id"] for r in fts_reader.search(q)["results"]]


def test_rerank_queries_split_chatty_turns():
    from pan.memory.retrieval import rerank_queries

    assert rerank_queries("Wie heißt mein Hund?") == ["Wie heißt mein Hund?"]
    q = "I got an email from the school. When are the theory lessons? Can I come on Thursday? Thanks."
    out = rerank_queries(q)
    assert out[0] == q and "When are the theory lessons?" in out and len(out) == 3


def test_rerank_takes_the_best_segment_score(tmp_path, fake):
    paths = _home(tmp_path)
    sync_wiki(paths, _cfg(fake.url))
    reader = Reader(paths)
    base = FtsRetriever(FtsIndex(paths.index_db, readonly=True))

    class All(FtsRetriever):
        def retrieve(self, query, k=5, *, type=None):
            return base.retrieve("langfuse pets travel plans", 5)

    q = "Lots of unrelated context about my week and the weather and work. Wie heißt mein Hund?"
    whole = RerankRetriever(All(None), EmbedClient(fake.url), reader.rerank_text, segments=False, timeout_s=2).retrieve(q, 3)
    seg = RerankRetriever(All(None), EmbedClient(fake.url), reader.rerank_text, segments=True, timeout_s=2).retrieve(q, 3)
    pets = lambda hits: next(h.score for h in hits if h.id == "projects.pets")
    assert pets(seg) > pets(whole) and seg[0].id == "projects.pets"


def test_query_prefix_for_plain_openai_servers(fake):
    plain = EmbedClient(fake.url)
    prefixed = EmbedClient(fake.url, query_prefix="Hund ")
    assert plain.embed(["Katze"], "query") != prefixed.embed(["Katze"], "query")
    assert plain.embed(["Katze"], "document") == prefixed.embed(["Katze"], "document")
