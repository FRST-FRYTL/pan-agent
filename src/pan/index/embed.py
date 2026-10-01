"""Client for the local retrieval sidecar (``pan-embedd``, ADR-012): embeddings and reranking.

Standard library only (the agent process has no numpy/torch). Speaks the OpenAI embeddings shape
(plus ``input_type``) and the Cohere/Jina/vLLM rerank shape, so any compatible local server works.

Failure policy: the client never raises into retrieval callers' hot paths without a timeout, and
after a failure it stays *down* for ``cooldown_s`` (a circuit breaker), so a stopped sidecar costs one
timeout, not one per turn. Callers fall back to FTS.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

DEFAULT_URL = "http://127.0.0.1:8091"


class EmbedError(RuntimeError):
    """The sidecar is unreachable, timed out or answered with an error."""


class EmbedClient:
    MAX_TIMEOUTS = 3

    def __init__(self, url: str = DEFAULT_URL, *, timeout_s: float = 2.0, cooldown_s: float = 30.0,
                 embed_model: str = "", rerank_model: str = "", dimensions: int = 0,
                 query_prefix: str = "") -> None:
        self.url = url.rstrip("/")
        self.timeout_s = timeout_s
        self.cooldown_s = cooldown_s
        self.embed_model = embed_model
        self.rerank_model = rerank_model
        self.dimensions = dimensions
        self.query_prefix = query_prefix  # for servers that apply no query instruction themselves
        self._down_until = 0.0
        self._timeouts = 0
        self._health: Optional[Dict[str, Any]] = None
        self._lock = threading.Lock()

    # -- transport -----------------------------------------------------------------------------

    def _call(self, path: str, body: Optional[Dict[str, Any]], timeout: Optional[float]) -> Dict[str, Any]:
        if time.monotonic() < self._down_until:
            raise EmbedError(f"{self.url} marked down (cooldown)")
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.url + path, data=data, method="GET" if body is None else "POST",
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.timeout_s) as resp:
                result = json.loads(resp.read())
        except urllib.error.HTTPError as exc:  # the server answered: a request problem, not an outage
            try:
                detail = json.loads(exc.read()).get("error", "")
            except Exception:
                detail = ""
            raise EmbedError(f"{path}: HTTP {exc.code} {detail}".strip()) from exc
        except (OSError, ValueError) as exc:  # refused, timeout, bad JSON
            reason = getattr(exc, "reason", exc)
            timed_out = isinstance(reason, TimeoutError) or "timed out" in str(reason)
            self._timeouts = self._timeouts + 1 if timed_out else 0
            # a busy sidecar (one slow rerank over the latency budget) is not an outage: only a
            # refused connection, bad data or MAX_TIMEOUTS timeouts in a row open the breaker
            if not timed_out or self._timeouts >= self.MAX_TIMEOUTS:
                self._down_until = time.monotonic() + self.cooldown_s
                logger.warning("PAN retrieval sidecar %s unavailable (%s); dense retrieval off for %.0fs",
                               self.url, exc, self.cooldown_s)
            raise EmbedError(f"{path}: {exc}") from exc
        self._timeouts = 0
        if not isinstance(result, dict):
            raise EmbedError(f"{path}: unexpected response")
        return result

    # -- API -----------------------------------------------------------------------------------

    def health(self, *, refresh: bool = False, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """``/health`` of the sidecar (cached), or None when it is down."""
        with self._lock:
            if self._health is not None and not refresh:
                return self._health
            try:
                self._health = self._call("/health", None, timeout)
            except EmbedError:
                self._health = None
            return self._health

    def embed_model_id(self) -> Optional[Dict[str, Any]]:
        """{model_id, revision, dim} of the embedder this client uses (None when down)."""
        h = self.health()
        if not h:
            return None
        models = h.get("embed") or []
        pick = next((m for m in models if self.embed_model in (m.get("name"), m.get("model_id"))), None) \
            if self.embed_model else next((m for m in models if m.get("default")), models[0] if models else None)
        return pick

    def rerank_model_id(self) -> Optional[Dict[str, Any]]:
        h = self.health()
        if not h:
            return None
        models = h.get("rerank") or []
        if self.rerank_model:
            return next((m for m in models if self.rerank_model in (m.get("name"), m.get("model_id"))), None)
        return next((m for m in models if m.get("default")), models[0] if models else None)

    def embed(self, texts: Sequence[str], input_type: str = "document", *,
              timeout: Optional[float] = None) -> List[List[float]]:
        if not texts:
            return []
        if input_type == "query" and self.query_prefix:
            texts = [self.query_prefix + t for t in texts]
        body: Dict[str, Any] = {"input": list(texts), "input_type": input_type}
        if self.embed_model:
            body["model"] = self.embed_model
        if self.dimensions:
            body["dimensions"] = self.dimensions
        out = self._call("/v1/embeddings", body, timeout)
        rows = sorted(out.get("data") or [], key=lambda d: d.get("index", 0))
        if len(rows) != len(texts):
            raise EmbedError(f"/v1/embeddings returned {len(rows)} vectors for {len(texts)} texts")
        return [list(map(float, r["embedding"])) for r in rows]

    def rerank(self, query: str, documents: Sequence[str], *, timeout: Optional[float] = None) -> List[float]:
        """Relevance in [0, 1] per document, in input order."""
        if not documents:
            return []
        body: Dict[str, Any] = {"query": query, "documents": list(documents)}
        if self.rerank_model:
            body["model"] = self.rerank_model
        out = self._call("/v1/rerank", body, timeout)
        scores = [0.0] * len(documents)
        for r in out.get("results") or []:
            scores[int(r["index"])] = float(r.get("relevance_score", r.get("score", 0.0)))
        return scores
