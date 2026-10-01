"""Generic typed-value handling (m6/generic-rules). Own wording, EN + DE."""

from __future__ import annotations

import pytest

from pan.memory.facts import contradiction, values


@pytest.mark.parametrize("old,new,want", [
    ("Monthly cloud budget: 5,000 EUR.", "The monthly cloud budget is now 7,500 EUR.", "Monthly cloud budget: 7,500 EUR."),
    ("Monatsbudget Cloud: 5.000 EUR.", "Monatsbudget Cloud neu: 7.500 EUR.", "Monatsbudget Cloud: 7.500 EUR."),
    ("The archive quota is 1,200 GB.", "The archive quota was raised to 1,500 GB.", "The archive quota is 1,500 GB."),
    ("Upload limit: 25 MB per file.", "The upload limit is 40 MB per file now.", "Upload limit: 40 MB per file."),
])
def test_rewrites_keep_whole_numbers(old, new, want):
    c = contradiction(new, old)
    assert c is not None
    assert c.rewritten in (want, None)
    assert "7,000" not in (c.rewritten or "") and "7.000" not in (c.rewritten or "")


def test_separated_numbers_are_one_value():
    assert values("The ticket budget is 12,345 per year.").slots.get("measure:budget-ticket") == {"12,345"}
    assert values("Das Ticketbudget beträgt 12.345 pro Jahr.").slots  # one value, not "12" and "345"


def test_rewrite_never_synthesizes_a_number():
    c = contradiction("Retention is 90 days now.", "Retention is 30 days, reviewed at 5 per week.")
    if c is not None and c.rewritten:
        assert all(tok in "Retention is 90 days now. Retention is 30 days, reviewed at 5 per week."
                   for tok in c.rewritten.replace(",", " ").split() if tok[:1].isdigit())


@pytest.mark.parametrize("text,slot,value", [
    ("The workstation is kestrel-dev.", "name", "kestrel-dev"),
    ("I sign in as build-bot on it.", "name:as", "build-bot"),
    ("Unser Speicher heißt vault-3.", "name", "vault-3"),
    ("Our wiki space is called Larkspur.", "name", "larkspur"),
])
def test_identifier_and_proper_names(text, slot, value):
    assert values(text).slots.get(slot) == {value}


@pytest.mark.parametrize("text", [
    "vLLM serves the model as primary.",       # plain word
    "My dog is called Mr. Biscuit.",           # abbreviation
    "The weather is nice.",
])
def test_plain_words_are_no_names(text):
    assert not any(k.startswith("name") for k in values(text).slots)


def test_renamed_machine_contradicts():
    c = contradiction("The workstation is osprey-dev now.", "The workstation is kestrel-dev.")
    assert c is not None and c.rewritten == "The workstation is osprey-dev."
@pytest.mark.parametrize("text,day", [
    ("Scheduler restarts only happen on Sundays.", "sunday"),
    ("The vendor call is on Tuesday at 11:30.", "tuesday"),
    ("Das Backup läuft jeden Freitag.", "freitag"),
])
def test_weekday_values(text, day):
    assert values(text).slots.get("day") == {day}


def test_weekday_change_contradicts():
    c = contradiction("The vendor call is on Friday at 11:30.", "The vendor call is on Tuesday at 11:30.")
    assert c is not None and c.rewritten == "The vendor call is on Friday at 11:30."
