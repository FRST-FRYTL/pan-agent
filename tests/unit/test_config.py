"""PAN config loading (integration spec §5)."""

from __future__ import annotations

from pathlib import Path

from pan.config import PanConfig, load_config
from pan.paths import PanPaths


def test_defaults_when_missing(tmp_path: Path):
    config = load_config(PanPaths.for_home(tmp_path))
    assert config == PanConfig()
    assert config.capture.skip_agent_contexts == ("cron", "flush")
    assert config.capture.tool_result_excerpt_bytes == 8192
    assert config.retrieval.prefetch_budget_chars == 1500
    assert config.wiki.path == "wiki" and config.wiki.auto_commit is True
    assert config.daemon.max_attempts == 3 and config.daemon.event_retention_days == 30


def test_partial_override_and_bad_values(tmp_path: Path):
    paths = PanPaths.for_home(tmp_path)
    paths.root.mkdir(parents=True)
    paths.config.write_text(
        "capture:\n  skip_agent_contexts: [cron]\n  tool_result_excerpt_bytes: 1024\n"
        "retrieval:\n  prefetch_min_score: nope\n"
        "wiki: not-a-mapping\n"
        "daemon:\n  poll_seconds: 5\n  unknown_key: 1\n"
        "extra_section: {}\n")
    config = load_config(paths)
    assert config.capture.skip_agent_contexts == ("cron",)
    assert config.capture.tool_result_excerpt_bytes == 1024
    assert config.retrieval.prefetch_min_score == 0.2
    assert config.retrieval.prefetch_budget_chars == 1500
    assert config.wiki == PanConfig().wiki
    assert config.daemon.poll_seconds == 5.0


def test_invalid_yaml_falls_back(tmp_path: Path):
    paths = PanPaths.for_home(tmp_path)
    paths.root.mkdir(parents=True)
    paths.config.write_text("capture: [unclosed\n")
    assert load_config(paths) == PanConfig()
