# Local models: what PAN needs and what to watch

PAN works with any OpenAI-compatible endpoint. This page covers three things:
- what the endpoint must support;
- how to check it;
- the harness-side problems that show up when Hermes and PAN's memory gate run against a local model.

Server-side details (building and serving vLLM on a DGX Spark, the upstream bugs, the flags) live in
the separate [spark-vllm-agent-cookbook](https://github.com/FRST-FRYTL/spark-vllm-agent-cookbook). Its
[field notes](https://github.com/FRST-FRYTL/spark-vllm-agent-cookbook/blob/v0.1.0/docs/field-notes.md)
are the reference for everything below the API.

## What the endpoint must support

| Capability | Needed by | If missing |
|---|---|---|
| Tool calling (structured `tool_calls`, streamed and non-streamed) | Hermes itself | the agent cannot use tools, including PAN's `memory_search` / `memory_read` |
| `response_format: json_schema` (strict) | PAN's memory gate | set `models.gate.structured: json_object` (or `none`); the output is still parsed strictly |
| Accepting `chat_template_kwargs` | PAN's memory gate (sent by default; switches off Qwen3-style thinking) | the gate drops the field after the first rejection (see below) |

Without a suitable endpoint, run the rules-only gate (`classifier: {kind: rules}`). It makes no
model calls.

## Check an endpoint: `pan doctor`

```bash
pan doctor                               # uses models.gate from $HERMES_HOME/pan/config.yaml
pan doctor --base-url http://localhost:8000/v1 --model <id> --no-thinking
```

`pan doctor` runs four checks:
- `models`: the model is listed.
- `tool_calls`: 6 streamed requests, each of which must be a structured call with valid JSON
  arguments.
- `tool_corruption`: one Hermes-shaped request (agent persona, tool-use rules, 7 tool schemas) sent
  5 times at temperature 0, so attempts 2 and later hit a server prefix cache.
- `json_schema`: one gate-style request; on HTTP 400 it retries with `json_object` and tells you
  which one works.

Run it after every server image or flag change, and before trusting any benchmark on a new setup.

## Tool calls that break only on repeated requests

**What you see.** The agent prints tool calls as plain text, emits garbled tool-call markup, or sends
empty arguments. In streaming mode the reply can be empty. It looks like an agent or PAN bug, and
the memory tools seem to "not work".

**Why.** Some server builds corrupt outputs on a prefix-cache hit. On vLLM, the combination of prefix
caching and MTP speculative decoding did this for hybrid Qwen3.x models (vLLM 0.21–0.25). An agent
harness hits the cache on almost every turn, because the system prompt and tool schemas repeat, so
the bug shows up exactly where agents run.

**What to do.** `pan doctor`'s `tool_corruption` check catches it: attempt 1 is clean and later
attempts fail. Use a fixed server build, or turn one of the two features off. The cookbook's field
notes have the upstream fixes and the tested configuration (0 corrupted calls in 914 responses with
both features on). After a server fix, re-run the baseline too: the bug depresses stock Hermes
and PAN alike.

## The gate's `chat_template_kwargs` and hosted APIs

**What you see.** The daemon log warns once: "gate endpoint rejected chat_template_kwargs; retrying
without it". In `pan memory inspect --json`, the gate info then lists `dropped_fields:
[chat_template_kwargs]`, and the gate keeps working with `status: ok`.

**Why.** PAN's gate sends `chat_template_kwargs: {enable_thinking: …}` (`models.gate.thinking`,
default off). It is a vLLM/SGLang extension, and some hosted APIs reject unknown request fields with
HTTP 400 or 422. With `models.gate.template_kwargs: auto` (the default), the gate retries once without
the field and keeps it off for the rest of the daemon's run. An error that names no field turns off
`response_format` first (structured output), then the field. A restart sends it again once.

**Settings.** `template_kwargs: never` never sends the field (for a hosted API you know rejects it);
`always` never drops it, so a rejecting endpoint fails the call and the rules-only classifier decides
the turn. Without the field, a model with a thinking phase thinks before every gate answer, which
is slower and can hit `max_tokens`.

**Detect.** `pan doctor --no-thinking` sends the field with every request, which matches the gate's
default request shape. Plain `pan doctor` does not.

Hermes' own requests are a separate setting: on a local Qwen3-style model, sending
`enable_thinking: false` makes each turn much faster (the demo's `NO_THINKING=1` does this).

## Server outages

A local server can crash or wedge under sustained load. The cookbook's field notes describe a restart
policy and a stall watchdog. On PAN's side nothing is lost:
- each gate call has one time budget per episode (`classifier.timeout_s`, default 90 s);
- after 3 consecutive failed episodes, a circuit breaker sends episodes to the rules-only classifier
  for 10 minutes without calling the model;
- events wait in the spool while the daemon or the server is down.

## Latency

Each finished turn costs one gate call in the background daemon: roughly 5–10 s under load with the
tested 27B model. The agent's own turn does not wait for it. Memory from a turn becomes available
once the daemon has processed it. Server-side latency effects, such as repeat time to first token
with speculative decoding and prefix caching, are covered in the cookbook's field notes.
