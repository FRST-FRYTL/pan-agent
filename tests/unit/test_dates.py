"""M9 G3: session dates and relative-date resolution for claims. Own wording."""

from __future__ import annotations

import datetime as dt

import pytest

from pan.memory.dates import (annotate, bare, claim_date, event_date, resolve, session_date, stated_date,
                              stated_offset, strip_today_line)

MON = dt.date(2024, 3, 11)   # a Monday


@pytest.mark.parametrize("text,want", [
    ("The user ran a 10k race yesterday.", "yesterday (2024-03-10)"),
    ("The user repainted the fence today.", "today (2024-03-11)"),
    ("The user bought a kettle last Saturday.", "last Saturday (2024-03-09)"),
    ("The user moved flats last Monday.", "last Monday (2024-03-04)"),
    ("The user adopted a cat 3 days ago.", "3 days ago (2024-03-08)"),
    ("The user started piano lessons two weeks ago.", "two weeks ago (2024-02-26)"),
    ("Der Nutzer war gestern beim Zahnarzt.", "gestern (2024-03-10)"),
    ("Der Nutzer hat vor 2 Wochen den Job gewechselt.", "vor 2 Wochen (2024-02-26)"),
    ("The user visited the aquarium last month.", "last month (2024-02)"),
    ("The user joined a book club this month.", "this month (2024-03)"),
])
def test_relative_dates_resolve_against_the_session_date(text, want):
    assert want in resolve(text, MON)


def test_resolution_is_idempotent_and_leaves_plain_text_alone():
    once = resolve("The user ran yesterday.", MON)
    assert resolve(once, MON) == once
    assert resolve("The user owns three bikes.", MON) == "The user owns three bikes."


def test_the_event_timestamp_is_the_primary_source():
    assert claim_date("I repainted the fence yesterday.", "2026-09-25T10:00:00Z") == dt.date(2026, 9, 25)
    assert event_date("2026-09-25T23:59:59.000Z") == dt.date(2026, 9, 25)
    assert claim_date("no timestamp, no statement", "") is None


@pytest.mark.parametrize("text", [
    "Today is 11 March 2024 and I finally fixed the gate.",
    "today is March 11th, 2024. Anyway, the gate works.",
    "Today's date is 2024-03-11.",
    "It's Monday, 2024/03/11, and the gate works.",
    "Heute ist Montag, der 11.03.2024.",
    "Wir haben heute den 11. März 2024, das Tor geht wieder.",
    "Heute ist Mo. 11. März 2024",
    "[Today is Mon 2024-03-11] The gate works again.",   # one format among others, not a special case
])
def test_a_date_the_user_states_wins_over_the_timestamp(text):
    assert stated_date(text) == MON
    assert claim_date(text, "2026-09-25T10:00:00Z") == MON          # conflict: the user's date wins
    assert "2024" not in strip_today_line(text)


@pytest.mark.parametrize("text", ["The invoice from 2024-03-11 is paid.", "Die Rechnung vom 11.03.2024 ist bezahlt.",
                                  "We shipped version 2024.03 today."])
def test_other_dates_in_the_message_are_not_todays_date(text):
    assert stated_date(text) is None
    assert claim_date(text, "2026-09-25T10:00:00Z") == dt.date(2026, 9, 25)


def test_later_turns_keep_the_offset_of_a_stated_date():
    off = stated_offset("Today is 11 March 2024.", "2026-09-25T10:00:00Z")
    assert off == (MON - dt.date(2026, 9, 25)).days
    # a later turn of the same session, one day later by the clock, is one day after the stated date
    assert claim_date("How did it go?", "2026-09-26T09:00:00Z", offset_days=off) == dt.date(2024, 3, 12)
    assert session_date("no statement", "2026-09-25T10:00:00Z") == dt.date(2026, 9, 25)


def test_annotate_and_bare():
    a = annotate("The user ran a 10k race yesterday.", MON)
    assert a == "The user ran a 10k race yesterday (2024-03-10). (stated 2024-03-11)"
    assert bare(a) == "The user ran a 10k race yesterday (2024-03-10)."
    assert annotate(a, MON) == a   # re-annotating keeps one suffix
    assert annotate("Always answer briefly.", MON, stated=False) == "Always answer briefly."
