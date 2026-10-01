# Demo: memory across two Hermes sessions

`run-demo.sh` is a scripted terminal demo of pan-agent. It takes one to two minutes on a local
model, and every step prints the command it runs.

First the script writes a scratch profile: a minimal, headless Hermes `config.yaml` for your
endpoint and a `pan/config.yaml` that points PAN's memory gate at the same endpoint. Then:

0. **`pan doctor`** checks the endpoint: tool calls (for Hermes) and `json_schema` (for PAN's memory
   gate). It reads the API key from the gate config, so the key never appears on a command line.
1. **`pan setup`** installs PAN into the scratch profile (it keeps the gate config).
2. **Session 1.** One `hermes chat -q … -Q` turn in which the user states a fact and a preference:
   *"For the record: our staging database runs on port 5433, and please always answer me in short
   bullet points."*
3. **Curation.** `pan memoryd run --once` processes the turn. The script then shows:
   - the gate and curator decisions (`pan memory inspect`);
   - the wiki page PAN wrote (`pan wiki search`, `pan wiki read`);
   - its git commit in `$DEMO_HOME/pan/wiki` (`git log --stat -1`, `git show`);
   - Hermes' `USER.md` / `MEMORY.md`, where the preference goes.
4. **Session 2.** This is a new session: *"Which port does our staging database use?"* The script
   prints **PASS** if the answer contains `5433`. It also shows the prefetch block PAN injected into
   that turn and the events it captured (tool calls such as `memory_search` would appear there).
5. A summary and the cleanup command.

## Prerequisites

- Hermes Agent with pan-agent installed into its venv. `hermes` and `pan` must be on `PATH`, or
  set `HERMES_BIN` / `PAN_BIN`. See the [main README](../../README.md) for the tested Hermes
  version.
- `git`, and a POSIX shell with bash (Linux or macOS).
- An OpenAI-compatible endpoint that supports tool calling and `response_format: json_schema`. The
  same endpoint serves the agent and PAN's gate.

## Run

```bash
BASE_URL=http://localhost:8000/v1 NO_THINKING=1 bash run-demo.sh   # optional: MODEL=<id from /models>
```

| Variable | Default | Meaning |
|---|---|---|
| `BASE_URL` | `http://localhost:8000/v1` | OpenAI-compatible endpoint (Hermes and PAN's gate) |
| `MODEL` | the first id from `GET $BASE_URL/models` | model id served there (optional; the script prints the one it picked) |
| `API_KEY` | `EMPTY` | bearer token, if the endpoint needs one (masked in the output) |
| `NO_THINKING` | `0` | `1` sends `chat_template_kwargs.enable_thinking=false` with Hermes' and `pan doctor`'s requests (see below) |
| `HERMES_BIN`, `PAN_BIN` | on `PATH` | the executables to use |
| `DEMO_HOME` | a fresh `mktemp -d` | the scratch profile, used as `HERMES_HOME` |
| `DEMO_PAUSE` | `0` | seconds to pause between steps |
| `DOCTOR_ARGS` | empty | extra `pan doctor` arguments, e.g. `--skip tool_corruption` |
| `SETUP_ARGS` | empty | extra `pan setup` arguments, e.g. `--force` on an untested Hermes |

`DOCTOR_ARGS` and `SETUP_ARGS` are split on whitespace; shell quoting inside them is not supported.

The exit code is 0 on PASS and non-zero if the recall fails or a step errors.

**`NO_THINKING=1`** turns off the reasoning phase of Qwen3-style chat templates. It is recommended for
local vLLM or SGLang servers with such models, and `demo.cast` was recorded with it. Without it,
these models think before every answer, which makes the demo slower. Hosted APIs may reject the
unknown `chat_template_kwargs` field, so leave it at `0` there. PAN's memory gate sends
`enable_thinking` too (its `models.gate.thinking` setting), and drops it after the first rejection
(`models.gate.template_kwargs: auto`).

## Only a scratch profile

The demo never reads or writes your real `~/.hermes`. Everything goes into `$DEMO_HOME`: the Hermes
config, PAN's state, the wiki, the logs, and the captured answers (`session1.out`, `session2.out`).
Hermes runs in the empty `$DEMO_HOME/work`, so it doesn't load project files from the directory
you started it in.

The script refuses to run if `DEMO_HOME` is `~/.hermes` or your home directory, or if it is a
non-empty directory it did not create (for example, one with a foreign `config.yaml`). It
re-creates a directory from an earlier demo run, which it recognizes by a `.pan-demo` marker file.
When you are done, clean up with `rm -rf "$DEMO_HOME"`. The script prints this command at the end.

## Record it yourself

With [asciinema](https://asciinema.org) (`uv tool install asciinema`):

```bash
cd /tmp   # a neutral directory; the recording shows paths as $DEMO_HOME
DEMO_HOME=/tmp/pan-demo DEMO_PAUSE=1 asciinema rec --overwrite -c "bash /path/to/run-demo.sh" demo.cast
asciinema play demo.cast
```

`demo.cast` in this directory was recorded this way, on the reference stack from the main README,
with `NO_THINKING=1` and `DOCTOR_ARGS="--skip tool_corruption"`. The `tool_corruption` check repeats
a request five more times, so skipping it keeps the recording short. By default, the demo runs all
checks.

## Expected output (abridged)

```text
# 2. Session 1: tell the agent a fact and a preference
$ hermes chat -q "For the record: our staging database runs on port 5433, …" -Q --max-turns 8
  agent> Understood. Both notes recorded.

# 3. Curate: pan-memoryd processes the queued turn once and exits
decisions (2):
  …  environment→wiki             create  committed    operations.staging-database
  …  user_preference→user         create  l1_added     l1:user

$ pan wiki search "staging database port" --limit 3
0.90  operations.staging-database  (environment, active)  operations/staging-database.md
      The staging database runs on port 5433. (stated …)

# 4. Session 2: a new session asks for the fact
$ hermes chat -q "Which port does our staging database use?" -Q --max-turns 8
  agent> Port 5433.

PASS: the new session recalled port 5433
  recall| Possibly relevant PAN wiki pages (open with memory_read):
  recall| - [operations.staging-database] Staging database § Facts (environment, active): The staging database runs on port 5433. …
```

`(observed)` on the wiki page is PAN's evidence tag. It means the claim comes straight from a
source, here the user's own words in session 1 (checked against that turn's text), and was not
inferred by the assistant (`(inferred)`). It does not mean that PAN verified the fact.

The model's wording and the gate's exact decisions vary from run to run. For example, in some runs
the agent also saves the fact with Hermes' built-in memory tool, and then it appears in `MEMORY.md`
as well. The page id, the PASS line and the `recall` block are the parts to look for. If the gate
shows `status: fallback` in `pan memory inspect --json`, the endpoint was unreachable or too slow,
and the rules-only gate decided that turn instead.
