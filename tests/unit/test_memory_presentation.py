"""How memory is presented to the agent: memory_read pages, recall snippets and section labels, the
active-preferences block and which USER.md entries count as preferences. The stored wiki text is
never changed — only what the agent sees."""

from __future__ import annotations

from pathlib import Path

import pytest

from pan.index.fts import FtsIndex, Hit, present_claim, present_heading
from pan.memory.l1 import is_preference_entry
from pan.memory.reader import SUPERSEDED_HEADING, Reader, render_page
from pan.memory.retrieval import _dense_snippet, format_prefetch
from pan.paths import PanPaths

CHOIR_PAGE = """---
id: projects.choir-rehearsals
type: project
status: active
created: 2026-08-02
updated: 2026-08-20
confidence: medium
sources: [session:s_early, event:01K0PRES000000000000000001, session:s_late]
tags: [choir, rehearsals]
subject: Choir rehearsals
---

# Choir rehearsals

User asked: "Please open notes/choir-plan.md and tell me when we rehearse.".

## Summary

- ~~Rehearsals are in room 12 of the parish hall. (stated 2026-08-02)~~ (observed; superseded 2026-08-20 by session:s_late)
- The choir rehearses on Thursdays at 19:00. (stated 2026-08-02) (observed)
- The choir has about 30 members. (stated 2026-08-02) (inferred)

## Status

## Updates

_Provenance: pan-memoryd, 2026-08-02 · session:s_early, event:01K0PRES000000000000000001_

### 2026-08-20 · session:s_late

- Rehearsals moved to room 21 of the parish hall. (stated 2026-08-20) (observed)
- Assistant reported: The room is booked until December. (stated 2026-08-20) (reported)

_Provenance: pan-memoryd, 2026-08-20 · session:s_late, event:01K0PRES000000000000000002_
"""

KILN_PAGE = """---
id: projects.kiln
type: project
status: active
created: 2026-08-05
updated: 2026-08-05
confidence: medium
sources: [session:s1]
tags: [kiln]
subject: Kiln
---

# Kiln

User asked: "Read firing-log.txt and summarize the settings.".

## Summary

- The kiln probably fires hotter than its display shows. (stated 2026-08-05) (inferred)
- The kiln fires glaze loads at cone 6. (stated 2026-08-05) (observed)

## Updates

_Provenance: pan-memoryd, 2026-08-05 · session:s1_
"""


@pytest.fixture
def home(tmp_path: Path) -> Path:
    paths = PanPaths.for_home(tmp_path / "home")
    for rel, text in (("projects/choir-rehearsals.md", CHOIR_PAGE), ("projects/kiln.md", KILN_PAGE)):
        (paths.wiki / rel).parent.mkdir(parents=True, exist_ok=True)
        (paths.wiki / rel).write_text(text)
    idx = FtsIndex(paths.index_db)
    idx.rebuild(paths.wiki)
    idx.close()
    return tmp_path / "home"


def _body(text: str) -> str:
    return text.split("\n---\n", 1)[1]


# -- memory_read ------------------------------------------------------------------------------------

def test_rendered_page_keeps_current_facts_and_lists_superseded_ones_last():
    out = render_page(_body(CHOIR_PAGE))
    for noise in ("~~", "session:", "event:", "_Provenance", "(observed)", "(stated", "(reported)", "## Status"):
        assert noise not in out
    assert "- The choir rehearses on Thursdays at 19:00. (noted 2026-08-02)" in out
    assert "- The choir has about 30 members. (noted 2026-08-02, inferred)" in out
    assert "- Assistant reported: The room is booked until December. (noted 2026-08-20)" in out
    assert "### 2026-08-20\n" in out and "## Updates (newer than the facts above" in out
    assert out.index("room 21") < out.index(SUPERSEDED_HEADING) < out.index("room 12")
    assert out.endswith("- Rehearsals are in room 12 of the parish hall. (noted 2026-08-02; superseded 2026-08-20)")


def test_rendered_page_without_superseded_claims_has_no_superseded_heading():
    out = render_page(_body(KILN_PAGE))
    assert SUPERSEDED_HEADING not in out and "## Updates" not in out  # only a provenance line under it
    assert out.startswith("# Kiln")


def test_memory_read_is_compact_and_sections_work_by_the_shown_heading(home: Path):
    r = Reader(PanPaths.for_home(home))
    try:
        res = r.read("projects.choir-rehearsals")
        assert set(res) - {"note"} == {"found", "id", "title", "type", "status", "updated", "content"}
        assert res["updated"] == "2026-08-20" and "room 21" in res["content"]
        sec = r.read("projects.choir-rehearsals", section="2026-08-20")
        assert sec["content"].startswith("### 2026-08-20") and "room 21" in sec["content"]
        assert "session:" not in sec["content"]
        assert "Summary" in r.read("projects.choir-rehearsals", section="Nope")["sections"]
        raw = r.read("projects.choir-rehearsals", raw=True)
        assert "~~Rehearsals are in room 12" in raw["content"] and raw["frontmatter"]["sources"]
    finally:
        r.close()


# -- recall snippets --------------------------------------------------------------------------------

def test_snippet_is_a_claim_not_the_context_line(home: Path):
    idx = FtsIndex(PanPaths.for_home(home).index_db, readonly=True)
    try:
        for query in ("choir", "what is in notes/choir-plan.md?"):
            hit = next(h for h in idx.search(query) if h.id == "projects.choir-rehearsals")
            assert "User asked" not in hit.snippet and "choir-plan" not in hit.snippet
            assert "(noted " in hit.snippet and "(stated " not in hit.snippet
    finally:
        idx.close()


def test_observed_claim_ranks_before_an_equally_matching_inferred_one(home: Path):
    idx = FtsIndex(PanPaths.for_home(home).index_db, readonly=True)
    try:
        (hit,) = [h for h in idx.search("how does the kiln fire?") if h.id == "projects.kiln"]
        assert hit.snippet.startswith("The kiln fires glaze loads at cone 6.")
    finally:
        idx.close()


def test_recall_section_label_has_no_source_ids():
    hit = Hit(id="projects.choir-rehearsals", path="p.md", title="Choir rehearsals", type="project",
              status="active", score=0.9, bm25=-1.0, section="2026-08-20 · session:s_late",
              snippet="Rehearsals moved to room 21. (noted 2026-08-20)")
    out = format_prefetch([hit])
    assert "session:" not in out and "§" not in out and "room 21" in out
    named = format_prefetch([Hit(**{**hit.to_dict(), "section": "Summary"})])
    assert "Choir rehearsals § Summary (project)" in named
    assert present_heading("2026-08-20 · session:x_1, event:01ABC") == "2026-08-20"


def test_dense_snippet_skips_the_context_line_and_renames_the_stamp():
    units = [(0.9, "text", "", "", 'User asked: "Please open notes/choir-plan.md".'),
             (0.8, "claim", "Summary", "summary", "The choir rehearses on Thursdays. (stated 2026-08-02)")]
    assert _dense_snippet(units) == ("Summary", "summary", "The choir rehearses on Thursdays. (noted 2026-08-02)")
    assert present_claim("She left yesterday (2026-08-01). (stated 2026-08-02)") == \
        "She left yesterday (2026-08-01). (noted 2026-08-02)"


# -- active preferences -----------------------------------------------------------------------------

def _provider():
    return pytest.importorskip("pan.hermes.provider", reason="the provider module needs Hermes Agent")


def test_preferences_relevant_first_without_fragments_or_near_duplicates():
    p = _provider()
    prefs = [
        "Always answer in Spanish.",
        "The user likes oolong tea.",
        "The user prefers oolong tea over black tea in the afternoon.",
        "I think I'll keep that one for now.",                  # quote without its context
        "I mentioned to you a board game and a recipe earlier.",  # talk about the conversation
        'the blue one." Board size: 40',                        # mangled quote
        "Enjoys long bike rides on gravel roads.",
        "The user does not care for very spicy food.",
    ]
    out = p.format_preferences(prefs, "Can you suggest a spicy dinner recipe for tonight?")
    lines = out.splitlines()
    assert lines[0] == p.ACTIVE_PREFERENCES_HEADER and "follow" in p.ACTIVE_PREFERENCES_HEADER
    assert lines[1] == "- The user does not care for very spicy food."      # relevant to the question
    assert lines[2] == "- Always answer in Spanish."                          # how to answer: always relevant
    assert "- The user likes oolong tea." not in lines                       # stated by the longer entry
    assert "- The user prefers oolong tea over black tea in the afternoon." in lines
    assert not any(x in out for x in ("keep that one", "mentioned to you", "Board size"))
    assert len(out) <= p.ACTIVE_PREFERENCES_BUDGET


def test_preferences_keep_a_negated_entry_next_to_a_similar_one():
    p = _provider()
    out = p.format_preferences(["No emojis.", "Use emojis in chat summaries."], "")
    assert "- No emojis." in out and "- Use emojis in chat summaries." in out


def test_relevant_preference_fits_the_budget_before_long_irrelevant_ones():
    p = _provider()
    filler = [f"Prefers topic {i} explained with a long list of historical background notes." for i in range(12)]
    out = p.format_preferences(filler + ["Prefers aisle seats on flights."], "Book me a flight with a good seat")
    assert "aisle seats" in out.splitlines()[1] and len(out) <= p.ACTIVE_PREFERENCES_BUDGET


@pytest.mark.parametrize("entry, pref", [
    ("Enjoys hobbies that need patience and focus.", True),
    ("Loves hiking in the mountains.", True),
    ("Can't stand crowded restaurants.", True),
    ("Is a big fan of jazz records.", True),
    ("Avoids caffeine after noon.", True),
    ("Favourite pastry is a cinnamon roll.", True),
    ("Doesn't care for super sweet drinks.", True),
    ("The friend loved the handmade gift.", False),  # a past event, not a standing taste
    ("The backup job runs at 02:00.", False),
])
def test_taste_entries_count_as_preferences(entry: str, pref: bool):
    assert is_preference_entry(entry) is pref
