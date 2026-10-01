"""Curator carry-over after supersession (m6/best). Own wording."""

from __future__ import annotations

from pan.memory.curator import supersede_bullets


def test_carry_over_never_synthesizes_values():
    """Regression: old "5,000" + new "7,500" was carried over as a fabricated "7,000 … (observed)"."""
    body = "## Facts\n\n- The cloud budget is 5,000 EUR per month, approved by Dana Kim. (observed)\n"
    new_body, gone, carried = supersede_bullets(body, ["The cloud budget is now 7,500 EUR per month."],
                                                "superseded")
    assert gone
    old = "The cloud budget is 5,000 EUR per month, approved by Dana Kim."
    for clause, _ in carried:
        assert clause in old                       # verbatim substring of the source
        assert "5,000" not in clause and "7,000" not in clause and "7,500" not in clause
    assert "7,000" not in new_body


def test_carry_over_keeps_an_unrelated_clause_verbatim():
    from pan.memory.curator import carry_clauses
    from pan.memory.facts import Contradiction
    old = "The report job runs on port 9310, and it writes to /srv/reports."
    hit = Contradiction(["port"], {"port": {"9310"}}, {"port": {"9320"}})
    assert carry_clauses(old, hit, ["The report job now runs on port 9320."]) == ["it writes to /srv/reports"]


from pan.memory.facts import contradiction  # noqa: E402


def test_narrower_subject_on_the_chosen_page_supersedes():
    old = "Our Kibana views are served on elk02 port 5601."
    new = "Kibana moved to port 5610 this morning after a clash with another service."
    assert contradiction(new, old) is None                      # globally: different subjects
    assert contradiction(new, old, on_page=True, subject="Kibana") is not None


def test_disjoint_subjects_on_the_same_page_are_not_superseded():
    body = "## Facts\n\n- Auth runs on port 7101. (observed)\n- Billing runs on port 7102. (observed)\n"
    new_body, gone, _ = supersede_bullets(body, ["Billing now listens on 7105."], "superseded", "Billing")
    assert gone == ["Billing runs on port 7102."] and "- Auth runs on port 7101. (observed)" in new_body


def test_snippet_holds_two_long_facts_and_drops_labels(tmp_path):
    from pan.index.fts import FtsIndex
    wiki = tmp_path / "wiki" / "operations"
    wiki.mkdir(parents=True)
    a = "`var/mirror.log` reports: ERROR mirror: push refused, volume cold-7 is at 3.8 TiB of 3.5 TiB, run halted"
    b = "`var/mirror.log` reports: WARN mirror: volume cold-6 at 91% of quota (2.7 TiB of 3.0 TiB), keep an eye on it"
    (wiki / "mirror.md").write_text(
        "---\nid: operations.mirror\ntype: environment\nstatus: active\ncreated: 2026-09-24\nupdated: 2026-09-24\n"
        f"confidence: medium\nsources: [session:s]\ntags: []\n---\n\n# Mirror\n\n## Facts\n\n- {a} (observed)\n"
        f"- {b} (observed; carried over)\n")
    idx = FtsIndex(tmp_path / "index.db")
    idx.rebuild(tmp_path / "wiki")
    (hit,) = idx.search("which volume stopped the mirror and how full was it", limit=1)
    idx.close()
    assert "cold-7" in hit.snippet and "cold-6" in hit.snippet and "carried over" not in hit.snippet


def test_file_content_claims_are_only_superseded_by_a_new_reading_of_the_same_file():
    body = "## Facts\n\n- `svc/relay.toml`: port = 7020 (observed)\n- The relay listens on port 7020. (observed)\n"
    new_body, gone, _ = supersede_bullets(body, ["We moved the relay to port 7045 last week."], "superseded", "Relay")
    assert gone == ["The relay listens on port 7020."] and "- `svc/relay.toml`: port = 7020 (observed)" in new_body
    _, gone2, _ = supersede_bullets(body, ["`svc/relay.toml`: port = 7045"], "superseded", "svc/relay.toml")
    assert "`svc/relay.toml`: port = 7020" in gone2
