"""A tiny scripted OpenAI-compatible chat-completions server (stdlib only) for agent E2E tests.

``FakeOpenAI(script)`` serves ``/v1/chat/completions`` (streaming SSE and non-streaming JSON) and
``/v1/models`` on 127.0.0.1 in a background thread and records every request it receives.

The script is a list of steps; each **agent** request (one that offers ``tools``) consumes the next
step. A step is a dict with either

- ``{"content": "final text"}`` — a plain assistant message, or
- ``{"tool_calls": [{"name": "write_file", "arguments": {...}}, ...]}`` — tool calls (``content``
  optional).

Requests without ``tools`` (auxiliary tasks such as title generation) and agent requests after
the script ran out get ``default_reply``; they are recorded but do not consume steps — unless a
``plain_script`` is given (M9 gate tests): then each request without ``tools`` consumes its next
step, which may also carry ``"delay": seconds`` (sleep before answering) or ``"status": 500``
(an HTTP error with ``"error"`` as message). ``reject_fields`` makes it behave like a hosted API that
rejects unknown request fields: a body carrying one of them gets ``reject_status`` (400) with
``reject_message`` (default: OpenAI's "Unrecognized request argument supplied: <field>") and consumes
no step.

Usage::

    with FakeOpenAI([{"tool_calls": [...]}, {"content": "Done."}]) as server:
        base_url = server.base_url          # http://127.0.0.1:<port>/v1
        ...
        server.agent_requests               # request bodies that offered tools

``RecordingProxy(upstream)`` forwards to a real server instead and records the same way (live test).
"""

from __future__ import annotations

import itertools
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Sequence

MODEL = "fake-model"
CONTEXT_LENGTH = 131072


def _tool_calls(step: Dict[str, Any], counter: "itertools.count[int]") -> List[Dict[str, Any]]:
    calls = []
    for call in step.get("tool_calls") or []:
        args = call.get("arguments", {})
        calls.append({
            "id": call.get("id") or f"call_{next(counter)}",
            "type": "function",
            "function": {"name": call["name"],
                         "arguments": args if isinstance(args, str) else json.dumps(args)},
        })
    return calls


class FakeOpenAI:
    def __init__(self, script: List[Dict[str, Any]], *, model: str = MODEL,
                 default_reply: str = "OK.", plain_script: Optional[List[Dict[str, Any]]] = None,
                 reject_fields: Sequence[str] = (), reject_status: int = 400,
                 reject_message: Optional[str] = None) -> None:
        self.script = list(script)
        self.reject_fields = tuple(reject_fields)
        self.reject_status = reject_status
        self.reject_message = reject_message
        self.plain_script = list(plain_script) if plain_script is not None else None
        self._plain_step = 0
        self.model = model
        self.default_reply = default_reply
        self.requests: List[Dict[str, Any]] = []   # every chat-completions body, in order
        self.paths: List[str] = []                  # every request path (GET and POST)
        self._lock = threading.Lock()
        self._step = 0
        self._ids = itertools.count(1)
        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    # -- lifecycle ---------------------------------------------------------------------------------

    def start(self) -> "FakeOpenAI":
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, name="fake-openai", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None

    def __enter__(self) -> "FakeOpenAI":
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()

    @property
    def base_url(self) -> str:
        assert self._server is not None, "server not started"
        return f"http://127.0.0.1:{self._server.server_address[1]}/v1"

    # -- inspection --------------------------------------------------------------------------------

    @property
    def agent_requests(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [r for r in self.requests if r.get("tools")]

    @property
    def steps_used(self) -> int:
        with self._lock:
            return self._step

    # -- responses ---------------------------------------------------------------------------------

    def _next_step(self, body: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            self.requests.append(body)
            if body.get("tools") and self._step < len(self.script):
                step = self.script[self._step]
                self._step += 1
                return step
            if not body.get("tools") and self.plain_script is not None and self._plain_step < len(self.plain_script):
                step = self.plain_script[self._plain_step]
                self._plain_step += 1
                return step
        return {"content": self.default_reply}

    def _handler(self):
        fake = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args: Any) -> None:  # keep test output quiet
                pass

            def _json(self, status: int, payload: Dict[str, Any]) -> None:
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self) -> None:
                fake.paths.append(self.path)
                path = self.path.split("?", 1)[0].rstrip("/")
                if path.endswith("/models"):
                    self._json(200, {"object": "list", "data": [
                        {"id": fake.model, "object": "model", "owned_by": "fake",
                         "max_model_len": CONTEXT_LENGTH, "context_length": CONTEXT_LENGTH}]})
                elif path.endswith("/health") or path == "":
                    self._json(200, {"status": "ok"})
                else:
                    self._json(404, {"error": {"message": f"not found: {self.path}"}})

            def do_POST(self) -> None:
                fake.paths.append(self.path)
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length else b""
                try:
                    body = json.loads(raw or b"{}")
                except ValueError:
                    self._json(400, {"error": {"message": "invalid JSON"}})
                    return
                if not self.path.split("?", 1)[0].rstrip("/").endswith("/chat/completions"):
                    self._json(404, {"error": {"message": f"not found: {self.path}"}})
                    return
                rejected = next((f for f in fake.reject_fields if f in body), None)
                if rejected is not None:
                    with fake._lock:
                        fake.requests.append(body)
                    self._json(fake.reject_status, {"error": {
                        "message": fake.reject_message or f"Unrecognized request argument supplied: {rejected}"}})
                    return
                step = fake._next_step(body)
                if step.get("delay"):
                    time.sleep(float(step["delay"]))
                if step.get("status"):
                    try:
                        self._json(int(step["status"]), {"error": {"message": step.get("error") or "scripted error"}})
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                    return
                calls = _tool_calls(step, fake._ids)
                content = step.get("content")
                if content is None and not calls:
                    content = fake.default_reply
                finish = "tool_calls" if calls else "stop"
                cid = f"chatcmpl-{next(fake._ids)}"
                usage = {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110}
                if body.get("stream"):
                    self._stream(cid, content, calls, finish, usage, body)
                else:
                    message: Dict[str, Any] = {"role": "assistant", "content": content}
                    if calls:
                        message["tool_calls"] = calls
                    try:
                        self._send_completion(cid, message, finish, usage)
                    except (BrokenPipeError, ConnectionResetError):
                        pass  # the client gave up (timeout test)

            def _send_completion(self, cid: str, message: Dict[str, Any], finish: str,
                                 usage: Dict[str, Any]) -> None:
                self._json(200, {"id": cid, "object": "chat.completion", "created": int(time.time()),
                                 "model": fake.model, "usage": usage,
                                 "choices": [{"index": 0, "message": message, "finish_reason": finish}]})

            def _stream(self, cid: str, content: Optional[str], calls: List[Dict[str, Any]], finish: str,
                        usage: Dict[str, Any], body: Dict[str, Any]) -> None:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()
                created = int(time.time())

                def chunk(delta: Dict[str, Any], finish_reason: Optional[str] = None,
                          extra: Optional[Dict[str, Any]] = None) -> None:
                    payload: Dict[str, Any] = {"id": cid, "object": "chat.completion.chunk", "created": created,
                                               "model": fake.model,
                                               "choices": [{"index": 0, "delta": delta,
                                                            "finish_reason": finish_reason}]}
                    if extra:
                        payload.update(extra)
                    self.wfile.write(f"data: {json.dumps(payload)}\n\n".encode())
                    self.wfile.flush()

                chunk({"role": "assistant", "content": ""})
                if content:
                    mid = max(1, len(content) // 2)
                    for piece in (content[:mid], content[mid:]):
                        if piece:
                            chunk({"content": piece})
                for i, call in enumerate(calls):
                    chunk({"tool_calls": [{"index": i, "id": call["id"], "type": "function",
                                           "function": {"name": call["function"]["name"], "arguments": ""}}]})
                    chunk({"tool_calls": [{"index": i,
                                           "function": {"arguments": call["function"]["arguments"]}}]})
                chunk({}, finish)
                if (body.get("stream_options") or {}).get("include_usage"):
                    self.wfile.write(("data: " + json.dumps({"id": cid, "object": "chat.completion.chunk",
                                                             "created": created, "model": fake.model,
                                                             "choices": [], "usage": usage}) + "\n\n").encode())
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
                self.close_connection = True

        return Handler


class RecordingProxy:
    """Transparent HTTP proxy to a real OpenAI-compatible server that records chat request bodies.

    Used by the live test to inspect what Hermes sent (tools, prompt, prefetch context) while the
    real model answers. Streaming responses are passed through chunk by chunk.
    """

    def __init__(self, upstream: str, *, timeout: float = 600.0) -> None:
        self.upstream = upstream.rstrip("/")
        self.timeout = timeout
        self.requests: List[Dict[str, Any]] = []
        self.timings: List[float] = []            # seconds per chat request (until the last byte)
        self._lock = threading.Lock()
        self._server: Optional[ThreadingHTTPServer] = None

    def start(self) -> "RecordingProxy":
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self._server.daemon_threads = True
        threading.Thread(target=self._server.serve_forever, name="recording-proxy", daemon=True).start()
        return self

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None

    def __enter__(self) -> "RecordingProxy":
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()

    @property
    def base_url(self) -> str:
        assert self._server is not None, "proxy not started"
        return f"http://127.0.0.1:{self._server.server_address[1]}/v1"

    @property
    def agent_requests(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [r for r in self.requests if r.get("tools")]

    def _target(self, path: str) -> str:
        # upstream ends in /v1 and the client's paths start with /v1/... (or /api/v1/... for probes)
        root = self.upstream[:-3] if self.upstream.endswith("/v1") else self.upstream
        return root + path

    def _handler(self):
        import urllib.error
        import urllib.request

        proxy = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args: Any) -> None:
                pass

            def _forward(self, method: str, data: Optional[bytes]) -> None:
                headers = {k: v for k, v in self.headers.items()
                           if k.lower() in ("content-type", "authorization", "accept")}
                req = urllib.request.Request(proxy._target(self.path), data=data, method=method, headers=headers)
                started = time.monotonic()
                try:
                    resp = urllib.request.urlopen(req, timeout=proxy.timeout)
                except urllib.error.HTTPError as err:
                    resp = err
                except OSError as exc:
                    payload = json.dumps({"error": {"message": f"proxy: {exc}"}}).encode()
                    self.send_response(502)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                    return
                try:
                    with resp:
                        self.send_response(getattr(resp, "status", None) or resp.code)
                        for key, value in resp.headers.items():
                            if key.lower() not in ("transfer-encoding", "connection", "content-length"):
                                self.send_header(key, value)
                        self.send_header("Connection", "close")
                        self.end_headers()
                        while True:
                            chunk = resp.read1(65536) if hasattr(resp, "read1") else resp.read(65536)
                            if not chunk:
                                break
                            self.wfile.write(chunk)
                            self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass  # the client hung up early (e.g. a probe with a short timeout)
                finally:
                    self.close_connection = True
                    if method == "POST" and self.path.rstrip("/").endswith("/chat/completions"):
                        with proxy._lock:
                            proxy.timings.append(time.monotonic() - started)

            def do_GET(self) -> None:
                self._forward("GET", None)

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                data = self.rfile.read(length) if length else b""
                if self.path.rstrip("/").endswith("/chat/completions"):
                    try:
                        body = json.loads(data or b"{}")
                    except ValueError:
                        body = {"_raw": data.decode("utf-8", "replace")}
                    with proxy._lock:
                        proxy.requests.append(body)
                self._forward("POST", data)

        return Handler
