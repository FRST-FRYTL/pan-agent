"""Episode grouping (integration spec §4.5): turn + its tool events; turn_end / close / idle."""

from __future__ import annotations

from conftest import FIXED_NOW, scenario_episodes
from pan.events.schema import Actor, AgentEvent, EventType
from pan.memory.episodes import group_events, parse_ts, previous_turn

T0 = parse_ts("2026-09-23T10:00:00Z")


def _ev(n: int, etype: EventType, session: str = "s1", seconds: float = 0, **content) -> AgentEvent:
    ts = f"2026-09-23T10:{int(seconds) // 60:02d}:{int(seconds) % 60:02d}.000Z"
    return AgentEvent(id=f"01K0TEST{n:018d}", ts=ts, event_type=etype, session_id=session,
                      actor=Actor.TOOL if etype is EventType.TOOL_CALL else Actor.MAIN_AGENT, content=content)


def test_tool_calls_join_the_following_turn():
    events = [_ev(1, EventType.TOOL_CALL, seconds=1, tool="terminal"),
              _ev(2, EventType.FILE_CHANGE, seconds=2, path="a.yaml"),
              _ev(3, EventType.TURN, seconds=3, user="u", assistant="a")]
    g = group_events(events, now=T0 + 10)
    assert len(g.episodes) == 1 and not g.pending
    ep = g.episodes[0]
    assert ep.event_ids == [e.id for e in events] and ep.closed_by == "turn"
    assert ep.user_text == "u" and ep.assistant_text == "a"
    assert len(ep.tool_calls) == 1 and len(ep.file_changes) == 1


def test_open_tool_group_stays_pending_until_idle():
    events = [_ev(1, EventType.TURN, seconds=0, user="u", assistant="a"),
              _ev(2, EventType.TOOL_CALL, seconds=5, tool="terminal")]
    g = group_events(events, now=T0 + 60)
    assert [ep.closed_by for ep in g.episodes] == ["turn"]
    assert [e.id for e in g.pending] == [events[1].id]
    g = group_events(events, now=T0 + 5 + 30 * 60)
    assert [ep.closed_by for ep in g.episodes] == ["turn", "idle"] and not g.pending
    g = group_events(events, now=T0 + 60, flush=True)
    assert [ep.closed_by for ep in g.episodes] == ["turn", "flush"]


def test_turn_end_seals_group_and_turn_arrives_later():
    """Plugin on_session_end (turn_end) may land before sync_turn's turn event."""
    events = [_ev(1, EventType.TOOL_CALL, seconds=1, tool="t1"),
              _ev(2, EventType.SESSION_END, seconds=2, kind="turn_end"),
              _ev(3, EventType.TOOL_CALL, seconds=3, tool="t2"),  # next turn already running
              _ev(4, EventType.TURN, seconds=4, user="first", assistant="a1")]
    g = group_events(events, now=T0 + 10)
    assert len(g.episodes) == 1
    assert g.episodes[0].event_ids == [events[0].id, events[3].id]
    assert [e.id for e in g.pending] == [events[2].id]
    assert [e.id for e in g.markers] == [events[1].id]


def test_close_and_subagent_stop_close_waiting_groups():
    events = [_ev(1, EventType.TOOL_CALL, seconds=1, tool="t"),
              _ev(2, EventType.SESSION_END, seconds=2, kind="close"),
              _ev(3, EventType.TOOL_CALL, session="child", seconds=3, tool="t"),
              _ev(4, EventType.SUBAGENT_STOP, session="child", seconds=4)]
    g = group_events(events, now=T0 + 10)
    assert sorted(ep.closed_by for ep in g.episodes) == ["close", "subagent_stop"]
    assert not g.pending and len(g.markers) == 2


def test_sessions_are_separate_and_markers_pass_through():
    events = [_ev(1, EventType.TOOL_CALL, session="a", seconds=1, tool="t"),
              _ev(2, EventType.TURN, session="b", seconds=2, user="b", assistant=""),
              _ev(3, EventType.L1_WRITE, session="a", seconds=3, action="add", target="user", content="x"),
              _ev(4, EventType.TURN, session="a", seconds=4, user="a", assistant="")]
    g = group_events(list(reversed(events)), now=T0 + 10)  # input order does not matter
    assert [(ep.session_id, len(ep.events)) for ep in g.episodes] == [("a", 2), ("b", 1)]
    assert [e.event_type for e in g.markers] == [EventType.L1_WRITE]


def test_previous_turn_is_same_session():
    events = [_ev(1, EventType.TURN, session="a", seconds=1, user="q", assistant="p"),
              _ev(2, EventType.TURN, session="b", seconds=2, user="x", assistant=""),
              _ev(3, EventType.TURN, session="a", seconds=3, user="yes", assistant="")]
    eps = group_events(events, now=T0 + 10).episodes
    assert previous_turn(eps, 2).id == events[0].id
    assert previous_turn(eps, 1) is None


def test_fixture_scenarios_group_as_expected():
    assert [len(ep.events) for ep in scenario_episodes("env_fact")] == [2]  # tool_call + turn
    assert [len(ep.events) for ep in scenario_episodes("decision")] == [1, 1]
    noise = scenario_episodes("noise")
    assert [ep.closed_by for ep in noise] == ["turn", "idle"]  # trailing failed ls closes by idle
    assert FIXED_NOW > parse_ts("2026-09-23T14:00:10Z")


def test_parse_ts_variants():
    assert parse_ts("2026-09-23T10:00:00Z") == parse_ts("2026-09-23T10:00:00+00:00") == T0
    assert parse_ts("garbage") == 0.0
