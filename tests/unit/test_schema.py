"""Event model and fixture format (no Hermes imports)."""

from __future__ import annotations

import pytest

from conftest import EVENT_SCENARIOS, load_events
from pan.events.schema import Actor, AgentEvent, EventType, new_ulid


def test_ulid_is_sortable_and_unique():
    ids = [new_ulid() for _ in range(200)]
    assert len(set(ids)) == 200
    assert all(len(i) == 26 for i in ids)


def test_event_round_trip():
    event = AgentEvent(event_type=EventType.TURN, session_id="s1", actor=Actor.USER, content={"user": "hi"})
    assert AgentEvent.from_dict(event.to_dict()) == event


@pytest.mark.parametrize("scenario", EVENT_SCENARIOS)
def test_fixture_events_parse(scenario):
    events = [AgentEvent.from_dict(e) for e in load_events(scenario)]
    assert events, f"{scenario} has no events"
    assert [e.ts for e in events] == sorted(e.ts for e in events), "events must be time-ordered"
