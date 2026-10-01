"""PAN settings from ``$HERMES_HOME/pan/config.yaml`` (integration spec §5). Missing file or keys = defaults.

M9: a ``pan:`` mapping in Hermes' own ``$HERMES_HOME/config.yaml`` supplies the same sections as a
lower-precedence layer (keys in ``pan/config.yaml`` win). Tools that only write Hermes' config
(e.g. ``pan.classifier.kind: llm`` written by a benchmark harness) can thus switch PAN settings."""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass, field
from typing import Any

from pan.paths import PanPaths

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CaptureConfig:
    skip_agent_contexts: tuple[str, ...] = ("cron", "flush")
    tool_result_excerpt_bytes: int = 8192


# M7 (ADR-012): hybrid retrieval by default; module-level so the test suite can pin "fts" and never
# reach a local retrieval sidecar (tests/conftest.py).
DEFAULT_RETRIEVAL_MODE = "hybrid"


@dataclass(frozen=True)
class RetrievalConfig:
    prefetch_budget_chars: int = 1500
    prefetch_min_score: float = 0.2
    # M7–M8 (ADR-012): hybrid = FTS5 + dense vectors (RRF) via the local ``pan-embedd`` sidecar; FTS
    # alone when the sidecar is down. ``fts`` = the pre-M7 behaviour.
    mode: str = field(default_factory=lambda: DEFAULT_RETRIEVAL_MODE)          # fts | hybrid (prefetch, memory_search)
    curator_mode: str = field(default_factory=lambda: DEFAULT_RETRIEVAL_MODE)  # fts | hybrid (curator lookup)
    embed_url: str = "http://127.0.0.1:8091"
    embed_model: str = ""                 # "" = the sidecar's default model
    embed_dimensions: int = 0             # 0 = the model's full dimension (Matryoshka truncation otherwise)
    embed_query_prefix: str = ""          # prepended to queries for a server that applies no query instruction
                                          # (a plain OpenAI-compatible /v1/embeddings; pan-embedd applies its own)
    timeout_s: float = 1.0                # query embedding
    dense_weight: float = 1.0             # RRF weight of the dense ranking (FTS = 1)
    rrf_k: int = 60
    # cosine → [0, 1] score, per embedding model (Qwen3-Embedding-0.6B, dev-v1 layer A): 0.2 (the
    # prefetch threshold, used when reranking is off or over budget) ≈ cos 0.5, above ~90 % of
    # non-relevant pages. The curator is stricter: its UPDATE threshold 0.45 ≈ cos 0.73.
    dense_floor: float = 0.4
    dense_ceiling: float = 0.9
    curator_dense_floor: float = 0.55
    curator_dense_ceiling: float = 0.95
    search_min_score: float = 0.002       # memory_search drops dense/reranked hits below this (``via``)
    rerank: bool = True                   # cross-encoder over the fused candidates (reader)
    curator_rerank: bool = False
    rerank_model: str = ""
    rerank_depth: int = 20
    rerank_timeout_s: float = 0.5         # latency budget; over it the fused order is used
    rerank_segments: bool = True          # also rerank against each question of a multi-sentence turn
    rerank_min_score: float = 0.002       # prefetch threshold for reranked hits (bge-reranker-v2-m3 relevance;
                                          # dev-v1: 90 % of non-relevant pages score ≤ 0.001; relevant pages of
                                          # chatty turns often score 0.002–0.01)
    rerank_keep_score: float = 0.4        # …or a pre-rerank (fused) score at least this high
    rerank_rel_floor: float = 0.01        # prefetch: a reranked hit also needs this share of the best hit's
                                          # relevance (unless kept on its fused score): next to a clear match,
                                          # pages at the noise floor are not recalled
    prefetch_k: int = 4                   # prefetch candidates (after reranking) that may enter the block


@dataclass(frozen=True)
class WikiConfig:
    path: str = "wiki"
    auto_commit: bool = True


@dataclass(frozen=True)
class DaemonConfig:
    poll_seconds: float = 2.0
    max_attempts: int = 3
    event_retention_days: int = 30
    batch_size: int = 200
    episode_idle_minutes: float = 30.0
    backfill_from_state_db: bool = True   # M5: complete crashed turns from Hermes' state.db (read-only)


# Default gate (M9; it beat the rules-only classifier on validation). Module-level so the test suite
# can pin "rules" (tests/conftest.py) and never reach the live model endpoint.
DEFAULT_CLASSIFIER_KIND = "llm"


@dataclass(frozen=True)
class ClassifierConfig:
    """M9 (classifier design §4.1): which gate decides per episode."""
    # llm (the LLM gate; rules-v2 is the automatic fallback when the endpoint is unreachable, times
    # out or answers invalid output) | rules (rules-v2 only) | shadow (both run; rules apply)
    kind: str = field(default_factory=lambda: DEFAULT_CLASSIFIER_KIND)
    timeout_s: float = 90.0        # budget per episode (all attempts together)
    attempts: int = 2              # tries for connection errors within the budget (timeouts, invalid output: none)
    breaker_failures: int = 3      # consecutive failed episodes that open the circuit breaker
    breaker_open_s: float = 600.0  # while open, episodes go straight to the rules fallback
    max_input_chars: int = 24000   # user-message budget (≈ 6k tokens); truncation is logged


@dataclass(frozen=True)
class GateModelConfig:
    base_url: str = "http://localhost:8000/v1"   # OpenAI-compatible endpoint (vLLM)
    model: str = "primary"
    api_key: str = "EMPTY"
    thinking: bool = False                       # chat_template_kwargs.enable_thinking
    temperature: float = 0.0
    max_tokens: int = 400
    structured: str = "json_schema"              # json_schema | json_object | none (strict parsing either way)
    template_kwargs: str = "auto"                # auto | always | never: send chat_template_kwargs (vLLM/SGLang
                                                 # extension); auto drops it for good after a 400/422 naming it


@dataclass(frozen=True)
class ModelsConfig:
    gate: GateModelConfig = field(default_factory=GateModelConfig)


@dataclass(frozen=True)
class PanConfig:
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    wiki: WikiConfig = field(default_factory=WikiConfig)
    daemon: DaemonConfig = field(default_factory=DaemonConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)


def _coerce(name: str, default: Any, value: Any) -> Any:
    """``value`` converted to the type of ``default``; the default (with a warning) when that fails."""
    try:
        if isinstance(default, bool):
            if isinstance(value, bool):
                return value
            if isinstance(value, str) and value.strip().lower() in {"true", "false", "yes", "no", "on", "off"}:
                return value.strip().lower() in {"true", "yes", "on"}
            raise ValueError(value)
        if isinstance(default, tuple):
            items = [value] if isinstance(value, str) else list(value)
            return tuple(str(v) for v in items)
        if isinstance(default, (int, float)) and isinstance(value, bool):
            raise ValueError(value)
        return type(default)(value)
    except (TypeError, ValueError):
        logger.warning("PAN config: invalid value for %s: %r (using %r)", name, value, default)
        return default


def _section(cls: type, name: str, raw: Any) -> Any:
    default = cls()
    if raw is None:
        return default
    if not isinstance(raw, dict):
        logger.warning("PAN config: section %r must be a mapping; using defaults", name)
        return default
    values = {}
    for f in dataclasses.fields(cls):
        if f.name in raw:
            current = getattr(default, f.name)
            if dataclasses.is_dataclass(current):
                values[f.name] = _section(type(current), f"{name}.{f.name}", raw[f.name])
            else:
                values[f.name] = _coerce(f"{name}.{f.name}", current, raw[f.name])
    return dataclasses.replace(default, **values)


def parse_config(raw: Any) -> PanConfig:
    """PanConfig from an already-parsed YAML mapping (unknown keys are ignored)."""
    raw = raw if isinstance(raw, dict) else {}
    return PanConfig(**{f.name: _section(f.default_factory, f.name, raw.get(f.name))  # type: ignore[misc]
                        for f in dataclasses.fields(PanConfig)})


DEFAULT_CONFIG_YAML = """\
# PAN settings (integration spec §5). Missing keys = defaults.
capture:
  skip_agent_contexts: [cron, flush]
  tool_result_excerpt_bytes: 8192
retrieval:
  prefetch_budget_chars: 1500
  prefetch_min_score: 0.2
  # M7–M8 hybrid retrieval (ADR-012; defaults shown, left commented for the same reason as below):
  # mode: hybrid              # fts | hybrid — needs the pan-embedd sidecar; FTS alone when it is down
  # curator_mode: hybrid
  # embed_url: http://127.0.0.1:8091
  # rerank: true
  # rerank_timeout_s: 0.5
  # rerank_min_score: 0.002
wiki:
  path: wiki
  auto_commit: true
daemon:
  poll_seconds: 2
  max_attempts: 3
  event_retention_days: 30
  batch_size: 200
  episode_idle_minutes: 30
# Memory gate (defaults shown; left commented so a `pan:` section in Hermes' config.yaml,
# e.g. written by a benchmark harness, is not overridden). kind: llm | rules | shadow
# (llm = one local-model call per episode; rules-v2 takes over automatically when it fails)
# classifier:
#   kind: llm
#   timeout_s: 90
#   attempts: 2
# models:
#   gate:
#     base_url: http://localhost:8000/v1
#     model: primary
#     thinking: false
#     temperature: 0
#     max_tokens: 400
#     structured: json_schema
#     template_kwargs: auto   # auto | always | never (chat_template_kwargs; hosted APIs may reject it)
"""


def write_default_config(paths: PanPaths) -> bool:
    """Create ``pan/config.yaml`` with the defaults if it does not exist; True if written."""
    if paths.config.exists():
        return False
    paths.config.parent.mkdir(parents=True, exist_ok=True)
    paths.config.write_text(DEFAULT_CONFIG_YAML, encoding="utf-8")
    return True


def _merge(low: Any, high: Any) -> Any:
    """Deep merge of two mappings; ``high`` wins."""
    if not isinstance(low, dict) or not isinstance(high, dict):
        return high if high is not None else low
    out = dict(low)
    for key, value in high.items():
        out[key] = _merge(low.get(key), value)
    return out


def _read_yaml(path: Any) -> Any:
    """Parsed YAML of ``path``; None when missing, unreadable or malformed (warning logged)."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError as exc:
        logger.warning("PAN config %s unreadable (%s); using defaults", path, exc)
        return None
    try:
        import yaml
        return yaml.safe_load(text)
    except Exception as exc:
        logger.warning("PAN config %s is not valid YAML (%s); using defaults", path, exc)
        return None


def load_config(paths: PanPaths) -> PanConfig:
    """Read ``paths.config`` over the ``pan:`` section of Hermes' ``config.yaml``; never raises
    (unreadable or malformed files fall back to defaults)."""
    raw = _read_yaml(paths.config)
    hermes = _read_yaml(paths.hermes_home / "config.yaml")
    base = hermes.get("pan") if isinstance(hermes, dict) and isinstance(hermes.get("pan"), dict) else None
    if base is None:
        return parse_config(raw)
    return parse_config(_merge(base, raw if isinstance(raw, dict) else None))
