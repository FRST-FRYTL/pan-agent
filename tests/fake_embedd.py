"""A deterministic stand-in for the ``pan-embedd`` sidecar (ADR-012) for tests.

Embeddings are normalised bags of *concepts*: words of ``CONCEPTS`` (with German and English
synonyms) share a dimension, every other word is hashed into the remaining dimensions. So "Hafen"
matches "port" (cross-language) while a query without shared concepts scores ~0. The reranker
returns the share of the query's concepts that the document contains.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, List

DIM = 512
CONCEPTS: Dict[str, int] = {}
for n, words in enumerate([
    ("port", "portnummer", "hafen", "anschluss"),
    ("gpu", "grafikkarte", "graphics"),
    ("dog", "hund", "puppy", "welpe"),
    ("birthday", "geburtstag"),
    ("sister", "schwester"),
    ("server", "dienst", "service"),
    ("bike", "fahrrad", "bicycle"),
    ("vacation", "urlaub", "holiday", "trip", "reise"),
]):
    for w in words:
        CONCEPTS[w] = n
STOP = frozenset("the a an of is on our my what which does do to in for and at was mein meine unser unsere "
                 "welche welcher was ist hat der die das auf den dem wie".split())


def concepts(text: str) -> List[int]:
    out = []
    for w in re.findall(r"\w+", text.lower()):
        if w in STOP or len(w) < 3:
            continue
        w = w[:-1] if w.endswith("s") and w[:-1] in CONCEPTS else w
        out.append(CONCEPTS[w] if w in CONCEPTS else 16 + int(hashlib.md5(w.encode()).hexdigest(), 16) % (DIM - 16))
    return out


def embed(text: str, dim: int = DIM) -> List[float]:
    v = [0.0] * dim
    for c in concepts(text):
        v[c % dim] += 1.0
    norm = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / norm for x in v]


class FakeEmbedd:
    """Run with ``with FakeEmbedd() as fake: fake.url``. Counts calls; ``delay`` slows reranking."""

    def __init__(self, model_id: str = "fake/concept-embedder", revision: str = "r1") -> None:
        self.model_id = model_id
        self.revision = revision
        self.calls = {"embeddings": 0, "rerank": 0, "embedded_texts": 0}
        self.rerank_delay = 0.0
        self.fail = False
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # silence
                pass

            def _send(self, code, body):
                data = json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                self._send(200, {"status": "ok", "embed": [{"name": "concept", "model_id": fake.model_id,
                                                            "revision": fake.revision, "dim": DIM, "default": True}],
                                 "rerank": [{"name": "concept-rr", "model_id": "fake/rr", "revision": "r1",
                                             "default": True}]})

            def do_POST(self):
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
                if fake.fail:
                    return self._send(500, {"error": "boom"})
                if self.path == "/v1/embeddings":
                    fake.calls["embeddings"] += 1
                    fake.calls["embedded_texts"] += len(req["input"])
                    dim = int(req.get("dimensions") or DIM)
                    return self._send(200, {"model": fake.model_id, "data": [
                        {"index": i, "embedding": embed(t, dim)} for i, t in enumerate(req["input"])]})
                if self.path == "/v1/rerank":
                    fake.calls["rerank"] += 1
                    if fake.rerank_delay:
                        import time
                        time.sleep(fake.rerank_delay)
                    q = set(concepts(req["query"]))
                    res = [{"index": i, "relevance_score": len(q & set(concepts(d))) / max(1, len(q))}
                           for i, d in enumerate(req["documents"])]
                    return self._send(200, {"results": res})
                self._send(404, {"error": "not found"})

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.httpd.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def __enter__(self) -> "FakeEmbedd":
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
