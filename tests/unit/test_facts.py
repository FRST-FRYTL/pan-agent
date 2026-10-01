"""Fact structure (M5): typed values, subjects and the contradiction rule (pan.memory.facts)."""

from __future__ import annotations

import pytest

from pan.memory.facts import command_topic, contradiction, same_subject, subject_of, subject_tags, values


@pytest.mark.parametrize("text,slots", [
    ("vLLM server runs on port 8000.", {"port": {"8000"}}),
    ("Langfuse runs on host spark01 at port 3300.", {"host": {"spark01"}, "port": {"3300"}}),
    ("Langfuse is at spark01:3300", {"host": {"spark01"}, "port": {"3300"}}),
    ("The API is http://localhost:8000/v1", {"url": {"http://localhost:8000/v1"}}),
    ("Config lives in /etc/vllm/serve.yaml", {"path": {"/etc/vllm/serve.yaml"}}),
    ("Hermes is pinned to version 0.21.4", {"version": {"0.21.4"}}),
    ("The GPU is an NVIDIA GB10.", {"id:gpu": {"gb10"}}),
    ("GPU (`nvidia-smi`) reports: NVIDIA **GB10**", {"id:gpu": {"gb10"}}),
    ("The box has 128 GB of RAM", {"size": {"128gb"}}),
    ("The server has 2 replicas.", {}),                   # bare numbers are never values
    ("The model is Qwen3.8-27B.", {"id:model": {"qwen3.8-27b"}}),
])
def test_values(text, slots):
    assert values(text).slots == slots


@pytest.mark.parametrize("new,old,rewritten", [
    # a benchmark run: the agent's L1 entry vs the user's update
    ("We moved the vLLM server to port 8010.", "vLLM server runs on port 8000.", "vLLM server runs on port 8010."),
    ("We moved the vLLM server to port 8010.", "User's vLLM server runs on port 8000.",
     "User's vLLM server runs on port 8010."),
    ("Update: vLLM moved from port 8000 to port 8010", "vLLM on port 8000", "vLLM on port 8010"),
    ("The GPU is an NVIDIA H100.", "GPU: NVIDIA GB10", "GPU: NVIDIA H100"),
    ("The box now has 256 GB of RAM", "The box has 128 GB of RAM", "The box has 256 GB of RAM"),
    ("Langfuse moved from spark01 to spark02:3300", "Langfuse instance runs on host spark01 at port 3300",
     "Langfuse instance runs on host spark02 at port 3300"),
])
def test_contradictions(new, old, rewritten):
    c = contradiction(new, old)
    assert c is not None and c.rewritten == rewritten


@pytest.mark.parametrize("new,old", [
    ("Langfuse runs on port 3300.", "vLLM server runs on port 8000."),              # other subject
    ("vLLM metrics exporter on port 9100", "vLLM server runs on port 8000."),       # other component
    ("vLLM config lives at /etc/vllm.yaml", "vLLM logs are in /var/log/vllm"),       # other attribute
    ("vLLM serves Qwen3.8-27B as model primary", "vLLM server runs on port 8000."),  # no common slot
    ("vLLM server runs on port 8000.", "vLLM server runs on port 8000."),           # same value
    ("The vLLM server no longer runs on port 8000.", "vLLM server runs on port 9000."),  # polarity
    ("Prefers answers under 60 words.", "Keeps the vLLM server under 80 GB."),      # other measure (M6)
])
def test_no_contradiction(new, old):
    assert contradiction(new, old) is None


@pytest.mark.parametrize("sentence,subject", [
    ("Note for later: our vLLM server runs on port 8000.", "vLLM server"),
    ("Update: we moved the vLLM server to port 8010.", "vLLM server"),
    ("FYI: the Langfuse instance for this project runs on host spark01 at port 3300.", "Langfuse instance"),
    ("The machine has an **NVIDIA GB10** GPU.", "GPU"),
    ("postgres is listening on 5432", "Postgres"),
    ("It runs on 8000", ""),
    ("Which port does it use?", ""),
])
def test_subject_of(sentence, subject):
    assert subject_of(sentence) == subject


def test_same_subject_and_tags():
    assert same_subject("Our vLLM server runs on port 8000.", "vLLM server moved to 8010")
    assert not same_subject("Our vLLM server runs on port 8000.", "Langfuse runs on 3300")
    assert subject_tags("Langfuse instance") == ["langfuse"]
    assert subject_tags("vLLM server") == ["vllm"]


@pytest.mark.parametrize("command,subject", [
    ("nvidia-smi --query-gpu=name --format=csv,noheader", "GPU"),
    ("sudo lscpu", "CPU"),
    ("free -h", "Memory (RAM)"),
    ("docker inspect vllm-main", "Docker container vllm-main"),
    ("systemctl --user status pan-memoryd", "systemd service pan-memoryd"),
    ("python -c 'print(1)'", ""),
])
def test_command_topics(command, subject):
    assert command_topic(command)[0] == subject
