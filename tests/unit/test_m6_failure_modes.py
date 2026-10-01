"""M6: failure modes found in a review of multi-session benchmark runs.

1. hallucinated save tool (system prompt wording), 3. assistant prose curated as fact, 4. prefetch
snippets that showed the user's question or a stale value, 5. L1 subject phrasing, 6. ADR slugs.
"""

from __future__ import annotations

import re

import pytest

from pan.events.schema import Actor, AgentEvent, Destination, EventType, MemoryType
pytest.importorskip("agent.memory_provider", reason="the provider module needs Hermes Agent")
from pan.hermes.provider import MEMORY_READ_SCHEMA, MEMORY_SEARCH_SCHEMA, SYSTEM_PROMPT_BLOCK
from pan.index.fts import FtsIndex
from pan.memory.classifier import INFERRED, OBSERVED, RuleClassifier, meta_talk
from pan.memory.episodes import Episode
from pan.memory.facts import contradiction
from pan.memory.retrieval import format_prefetch
from pan.wiki.frontmatter import Page, parse
from pan.wiki.markdown import compact_slug
from pan.wiki.store import WikiStore

clf = RuleClassifier()

# A benchmark session (PAN run): the final assistant text after the tool-name hunt.
HUNT_FINAL_ASSISTANT = (
    "Noted — vLLM server on port 8000.\n"
    'I did save the fact to my own persistent memory (both as a memory entry and a note about the '
    'tool-name confusion), so "vLLM server runs on port 8000" is retained across sessions.')


def _turn(user: str, assistant: str = "", n: int = 1) -> Episode:
    ev = AgentEvent(id=f"01K0CM6{n:019d}", ts="2026-09-23T10:00:00Z", event_type=EventType.TURN,
                    session_id="s", actor=Actor.MAIN_AGENT, content={"user": user, "assistant": assistant})
    return Episode("s", [ev], closed_by="turn")


# -- 1. system prompt ------------------------------------------------------------------------------

def test_prompt_block_names_the_tools_and_invites_no_save_tool():
    block = SYSTEM_PROMPT_BLOCK
    assert "memory_search" in block and "memory_read" in block and "directly" in block
    assert "automatically" in block and "no other tools" in block
    # the 2026-09-23 wording that preceded the `pan_memory_save` hunt
    for phrase in ("to record or correct a fact", "just state it", "tool for writing", "or use the memory tool"):
        assert phrase not in block
    assert "tool_call" not in block and "tool_search" not in block  # don't prime the bridge
    assert len(block) < 1400   # paid on every call
    assert "{" not in block  # static: no per-session formatting


def test_tool_descriptions_are_plain():
    assert "save" not in MEMORY_SEARCH_SCHEMA["description"].lower()
    assert "memory_search" in MEMORY_READ_SCHEMA["description"]


# -- 3. assistant prose ------------------------------------------------------------------------------

def test_scn002_assistant_meta_talk_is_not_curated():
    c = clf.classify(_turn("Note for later: our vLLM server runs on port 8000.", HUNT_FINAL_ASSISTANT))
    assert [(x.classification.type, x.classification.destination) for x in c] == [
        (MemoryType.ENVIRONMENT, Destination.WIKI)]
    assert c[0].normalized_claims == ["Our vLLM server runs on port 8000."]
    assert c[0].claim_evidence == [OBSERVED]


@pytest.mark.parametrize("assistant", [
    HUNT_FINAL_ASSISTANT.splitlines()[1],
    "I checked and the vLLM server is running on port 8000.",               # first person
    'The server config says "the vLLM server runs on port 8000 by default".',  # quoted restatement
    "The vLLM server runs on port 8000, " + "which " * 60 + "is fine.",       # prose, too long
])
def test_meta_or_prose_assistant_sentences_are_noise(assistant):
    c = clf.classify(_turn("What do you know about our setup?", assistant))
    assert [x.classification.type for x in c] == [MemoryType.NOISE]


def test_plain_assistant_environment_explanation_still_counts():
    c = clf.classify(_turn("why do tool calls fail?",
                           "The vLLM server was started without --enable-auto-tool-choice, so tool calling is disabled."))
    assert c[0].classification.type is MemoryType.ENVIRONMENT and c[0].claim_evidence == [INFERRED]


def test_meta_talk_detector():
    assert meta_talk("PAN knowledge wiki maintained by pan-memoryd daemon. Tool name for saving facts unknown.")
    assert not meta_talk("vLLM serves Qwen3.6 as 'primary' on port 8000 without tool calling enabled.")


# -- 4. snippets -------------------------------------------------------------------------------------

GPU_PAGE = """---
id: operations.gpu
type: environment
status: active
created: 2026-09-23
updated: 2026-09-23
confidence: medium
sources: [session:s1]
tags: [gpu, nvidia, hardware]
subject: GPU
---

# GPU

User asked: "Run `nvidia-smi --query-gpu=name --format=csv,noheader` and tell me which GPU this machine has.".

## Facts

- This machine has an NVIDIA GB10 GPU. (inferred)
- GPU (`nvidia-smi --query-gpu=name --format=csv,noheader`) reports: NVIDIA GB10 (observed)

## Updates

_Provenance: pan-memoryd, 2026-09-23 · session:s1_
"""

VLLM_PAGE = """---
id: operations.vllm-server
type: environment
status: active
created: 2026-09-23
updated: 2026-09-23
confidence: medium
sources: [session:s1, session:s2]
tags: [vllm]
subject: vLLM server
---

# vLLM server

## Facts

- ~~Our vLLM server runs on port 8000.~~ (observed; superseded 2026-09-23 by session:s2)

## Updates

### 2026-09-23 · session:s2

- We moved the vLLM server to port 8010. (observed)

_Provenance: pan-memoryd, 2026-09-23 · session:s2_
"""

SEED_PAGE = """---
id: learnings.langfuse
type: learning
status: active
created: 2026-09-23
updated: 2026-09-23
confidence: medium
sources: [session:s0]
tags: [langfuse]
---

# Langfuse tracing

Langfuse runs as a container next to the agent and receives traces over OTLP.

## Related

- [vLLM server](../operations/vllm-server.md)
"""


@pytest.fixture()
def index(tmp_path):
    root = tmp_path / "wiki"
    for rel, text in (("operations/gpu.md", GPU_PAGE), ("operations/vllm-server.md", VLLM_PAGE),
                      ("learnings/langfuse.md", SEED_PAGE)):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    idx = FtsIndex(tmp_path / "index.db")
    idx.rebuild(root)
    yield idx
    idx.close()


def test_gpu_snippet_is_the_fact_not_the_users_question(index):
    (hit,) = index.search("Without running any commands: which GPU model does this machine have?", limit=1)
    assert hit.id == "operations.gpu"
    assert "NVIDIA GB10" in hit.snippet and "User asked" not in hit.snippet
    assert "NVIDIA GB10" in format_prefetch([hit])


def test_updated_port_snippet_shows_only_the_current_value(index):
    hit = index.search("Which port does our vLLM server use? Answer from memory in one sentence.", limit=1)[0]
    assert hit.id == "operations.vllm-server"
    assert "8010" in hit.snippet and "8000" not in hit.snippet


def test_prose_pages_keep_the_window_snippet_and_skip_link_lists(index):
    hit = next(h for h in index.search("how are Langfuse traces received?") if h.id == "learnings.langfuse")
    assert "OTLP" in hit.snippet and "vllm-server.md" not in hit.snippet


# -- 5. L1 subject phrasing -------------------------------------------------------------------------

@pytest.mark.parametrize("old,same", [
    ("PAN vLLM server runs on port 8000.", True),
    ("Our vLLM inference server listens on port 8000.", True),
    ("vLLM metrics exporter runs on port 9100.", False),
    ("Langfuse server runs on port 3300.", False),
    ("The dev server runs on port 8000.", False),
])
def test_qualified_subjects(old, same):
    assert (contradiction("We moved the vLLM server to port 8010.", old) is not None) is same


# -- 6. ADR slugs -----------------------------------------------------------------------------------

def test_compact_slug():
    assert compact_slug("We use SQLite, not Postgres, for the bench results index") == \
        "sqlite-not-postgres-bench-results-index"
    assert compact_slug("Use FTS5 for retrieval") == "fts5-retrieval"
    assert len(compact_slug("x " * 10 + "a very long decision title about many different things at once")) <= 48
    assert compact_slug("the a of") == "page"


def test_new_decision_path_is_short(tmp_path):
    store = WikiStore(tmp_path / "wiki")
    (tmp_path / "wiki" / "decisions").mkdir(parents=True)
    page = store.new_decision("We use SQLite, not Postgres, for the bench results index", status="accepted",
                              body="", sources=["session:s"], tags=[], date="2026-09-23", confidence="medium")
    assert page.path == "decisions/ADR-001-sqlite-not-postgres-bench-results-index.md"
    assert page.meta["id"] == "decisions.adr-001"


def test_prompt_block_keeps_the_memory_tool_and_lets_the_newest_statement_win():
    """it1 (ADR-007 (b) rejected): the agent must keep saving with its memory tool. Newest wins —
    also within memory, so an older USER.md entry does not beat a later wiki update."""
    assert "memory tool works as usual" in SYSTEM_PROMPT_BLOCK and "Newest wins" in SYSTEM_PROMPT_BLOCK
    assert "what the user says now beats memory" in SYSTEM_PROMPT_BLOCK
    assert "older USER.md/MEMORY.md entry" in SYSTEM_PROMPT_BLOCK
    assert "no tool call" not in SYSTEM_PROMPT_BLOCK and "nothing needs to be saved" not in SYSTEM_PROMPT_BLOCK


def test_snippets_drop_evidence_labels(index):
    hit = index.search("Which port does our vLLM server use?", limit=1)[0]
    assert "(observed)" not in hit.snippet and "(inferred)" not in hit.snippet


def test_environment_qualifier_is_not_part_of_the_subject():
    from pan.memory.facts import subject_of
    assert subject_of("The vLLM server in this environment runs on port 8000.") == "vLLM server"
