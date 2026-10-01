#!/usr/bin/env bash
# pan-agent demo: two Hermes sessions, one scratch profile, long-term memory in between.
#
#   Session 1 states a fact and a preference. pan-memoryd curates that turn into the wiki (a git
#   repo) and into Hermes' USER.md. Session 2 is a new session that asks for the fact.
#
# Everything happens in a scratch profile ($DEMO_HOME, used as HERMES_HOME). Your real ~/.hermes
# is never read or written.
#
# Usage:
#   BASE_URL=http://localhost:8000/v1 NO_THINKING=1 bash run-demo.sh
#
# Environment (all optional):
#   BASE_URL     OpenAI-compatible endpoint for Hermes and PAN's memory gate  [http://localhost:8000/v1]
#   MODEL        model id served there              [the first id from GET $BASE_URL/models]
#   API_KEY      bearer token for the endpoint (masked in the output)         [EMPTY]
#   NO_THINKING  1 = send chat_template_kwargs.enable_thinking=false with Hermes' and doctor's
#                requests (vLLM/SGLang with Qwen3-style templates); 0 = don't  [0]
#   HERMES_BIN   hermes executable                                            [hermes on PATH]
#   PAN_BIN      pan executable, installed in Hermes' venv                    [pan on PATH]
#   DEMO_HOME    scratch profile: new, empty, or a previous demo run          [fresh mktemp -d]
#   DEMO_PAUSE   seconds to pause between steps (e.g. 1 for a recording)      [0]
#   DOCTOR_ARGS  extra `pan doctor` arguments, e.g. "--skip tool_corruption"
#   SETUP_ARGS   extra `pan setup` arguments, e.g. "--force" on an untested Hermes version
#                (both: whitespace-separated words, no shell quoting)
set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8000/v1}"
MODEL="${MODEL:-}"
API_KEY="${API_KEY:-EMPTY}"
NO_THINKING="${NO_THINKING:-0}"
HERMES_BIN="${HERMES_BIN:-hermes}"
PAN_BIN="${PAN_BIN:-pan}"
DEMO_PAUSE="${DEMO_PAUSE:-0}"
DOCTOR_ARGS="${DOCTOR_ARGS:-}"
SETUP_ARGS="${SETUP_ARGS:-}"

FACT_PROMPT="For the record: our staging database runs on port 5433, and please always answer me in short bullet points."
RECALL_PROMPT="Which port does our staging database use?"
EXPECTED="5433"
MARKER=".pan-demo"   # marks a directory this script created

# -- output helpers -----------------------------------------------------------------------------

if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then
  BOLD=$'\033[1m'; DIM=$'\033[2m'; GREEN=$'\033[32m'; RED=$'\033[31m'; RESET=$'\033[0m'
else
  BOLD=""; DIM=""; GREEN=""; RED=""; RESET=""
fi

die()   { printf '%serror:%s %s\n' "$RED" "$RESET" "$*" >&2; exit 1; }
say()   { printf '\n%s# %s%s\n' "$BOLD" "$*" "$RESET"; }        # narration line
note()  { printf '%s  %s%s\n' "$DIM" "$*" "$RESET"; }
pause() { if [ "$DEMO_PAUSE" != "0" ]; then sleep "$DEMO_PAUSE"; fi; }

# Echo a command the way a user would type it, then run it. The scratch path is shown as
# $DEMO_HOME, the hermes/pan executables by their short names, and a real API key as ***.
shown_cmd() {
  local out="" arg
  for arg in "$@"; do
    case "$arg" in
      "$HERMES_BIN") arg="hermes" ;;
      "$PAN_BIN") arg="pan" ;;
    esac
    arg="${arg//"$DEMO_HOME"/\$DEMO_HOME}"
    if [ "$API_KEY" != "EMPTY" ] && [ -n "$API_KEY" ]; then arg="${arg//"$API_KEY"/***}"; fi
    case "$arg" in
      *[[:space:]\'\;\&\|\<\>\(\)\*\?]*) out+=" \"$arg\"" ;;
      *) out+=" $arg" ;;
    esac
  done
  # stderr, so the echo never ends up in a pipe or a captured answer
  printf '%s$%s%s\n' "$GREEN" "$RESET" "$out" >&2
}
run() { shown_cmd "$@"; "$@"; }

# Filter for command output: the scratch path as $DEMO_HOME, the hermes executable as `hermes`,
# a real API key as ***. Plain string replacement (no regex), so any path is safe.
mask() {
  local line
  while IFS= read -r line || [ -n "$line" ]; do
    line="${line//"$HERMES_BIN"/hermes}"
    line="${line//"$DEMO_HOME"/\$DEMO_HOME}"
    if [ "$API_KEY" != "EMPTY" ] && [ -n "$API_KEY" ]; then line="${line//"$API_KEY"/***}"; fi
    printf '%s\n' "$line"
  done
}

# First model id listed by GET $BASE_URL/models. The API key goes through the environment, so it
# never shows up in a process list.
detect_model() {
  DEMO_BASE_URL="$BASE_URL" DEMO_API_KEY="$API_KEY" "$PY" - <<'PY'
import json, os, sys, urllib.request
url = os.environ["DEMO_BASE_URL"].rstrip("/") + "/models"
req = urllib.request.Request(url, headers={"Authorization": "Bearer " + os.environ["DEMO_API_KEY"]})
try:
    with urllib.request.urlopen(req, timeout=15) as resp:
        ids = [m.get("id") for m in json.load(resp).get("data", []) if isinstance(m, dict) and m.get("id")]
except Exception as exc:
    sys.exit(f"GET {url} failed: {exc}")
if not ids:
    sys.exit(f"GET {url} lists no models")
print(ids[0])
PY
}

# Hermes' config.yaml (a custom OpenAI-compatible endpoint, no interactive setup needed) and PAN's
# pan/config.yaml (the memory gate on the same endpoint). Values are serialized by yaml.safe_dump
# (or as JSON strings, which YAML reads as quoted scalars), never pasted into YAML text.
write_configs() {
  DEMO_BASE_URL="$BASE_URL" DEMO_MODEL="$MODEL" DEMO_API_KEY="$API_KEY" DEMO_NO_THINKING="$NO_THINKING" \
    "$PY" - "$DEMO_HOME/config.yaml" "$DEMO_HOME/pan/config.yaml" <<'PY'
import json, os, sys
env = os.environ
base_url, model, key = env["DEMO_BASE_URL"], env["DEMO_MODEL"], env["DEMO_API_KEY"]
hermes = {"model": {"default": model, "provider": "custom", "base_url": base_url, "api_key": key,
                    "context_length": 131072}}   # context_length: skips context-length probing
if env["DEMO_NO_THINKING"] == "1":             # extra request fields for this endpoint
    hermes["custom_providers"] = [{"name": "demo-endpoint", "base_url": base_url, "api_key": key,
                                   "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}]
gate = {"classifier": {"kind": "llm"},
        "models": {"gate": {"base_url": base_url, "model": model, "api_key": key, "thinking": False}}}


def plain_dump(node, ind=""):
    lines = []
    for k, v in node.items():
        if isinstance(v, dict):
            lines += [f"{ind}{k}:"] + plain_dump(v, ind + "  ")
        elif isinstance(v, list):
            lines.append(f"{ind}{k}:")
            for item in v:
                sub = plain_dump(item, ind + "    ")
                lines += [f"{ind}  - {sub[0].lstrip()}"] + sub[1:]
        else:
            lines.append(f"{ind}{k}: {json.dumps(v)}")
    return lines


try:
    import yaml
    dump = lambda d: yaml.safe_dump(d, sort_keys=False, default_flow_style=False)
except ImportError:
    dump = lambda d: "\n".join(plain_dump(d)) + "\n"
for path, head, data in ((sys.argv[1], "pan-agent demo profile", hermes),
                         (sys.argv[2], "memory gate on the demo endpoint (pan setup keeps this file)", gate)):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"# run-demo.sh: {head}\n" + dump(data))
PY
}

# One headless Hermes turn: $1 = file prefix, rest = prompt. -Q prints only the final answer
# (session_id goes to stderr); </dev/null means it never waits for input.
hermes_turn() {
  local name="$1" prompt="$2"
  shown_cmd "$HERMES_BIN" chat -q "$prompt" -Q --max-turns 8
  "$HERMES_BIN" chat -q "$prompt" -Q --max-turns 8 </dev/null \
    >"$DEMO_HOME/$name.out" 2>"$DEMO_HOME/$name.err" ||
    die "$name failed: $(tail -n 5 "$DEMO_HOME/$name.err")"
  # Hide one harmless Hermes startup notice (optional tirith scanner not installed).
  grep -v 'tirith security scanner' "$DEMO_HOME/$name.out" | sed 's/^/  agent> /' || true
  sed -n 's/^session_id:[[:space:]]*//p' "$DEMO_HOME/$name.err" | tail -n 1 > "$DEMO_HOME/$name.id"
}

# The prefetch block PAN injected into a turn (recorded in the turn event as `recall`).
show_recall() {
  shown_cmd "$PAN_BIN" memory inspect --session "$1" --json
  "$PAN_BIN" memory inspect --session "$1" --json | "$PY" -c '
import json, sys
recall = [e["content"].get("recall") for e in json.load(sys.stdin)["events"] if e["event_type"] == "turn"]
text = next((r for r in recall if r), "")
print("\n".join("  recall| " + line for line in text.splitlines()) if text else "  (no prefetch recorded)")'
}

# -- inputs and safety checks -------------------------------------------------------------------

HERMES_BIN="$(command -v "$HERMES_BIN")" || die "hermes not found (set HERMES_BIN)"
PAN_BIN="$(command -v "$PAN_BIN")" || die "pan not found (set PAN_BIN)"
command -v git >/dev/null 2>&1 || die "git not found (PAN's wiki is a git repository)"
PY="$(dirname "$PAN_BIN")/python"            # Hermes' venv python (has PyYAML), else python3
[ -x "$PY" ] || PY="$(command -v python3)" || die "python3 not found"
# `pan setup` calls `hermes config set`; make sure it finds the same hermes.
PATH="$(dirname "$HERMES_BIN"):$PATH"

if [ -z "${DEMO_HOME:-}" ]; then
  DEMO_HOME="$(mktemp -d "${TMPDIR:-/tmp}/pan-demo.XXXXXX")"
  touch "$DEMO_HOME/$MARKER"
fi
case "$DEMO_HOME" in /*) ;; *) DEMO_HOME="$PWD/$DEMO_HOME" ;; esac
DEMO_HOME="${DEMO_HOME%/}"

real_path() { (cd "$1" 2>/dev/null && pwd -P) || printf '%s\n' "$1"; }
if [ "$(real_path "$DEMO_HOME")" = "$(real_path "$HOME/.hermes")" ] ||
   [ "$(real_path "$DEMO_HOME")" = "$(real_path "$HOME")" ] || [ "$DEMO_HOME" = "/" ]; then
  die "DEMO_HOME must be a scratch directory, not your Hermes profile or home ($DEMO_HOME)"
fi
if [ -e "$DEMO_HOME" ] && [ ! -d "$DEMO_HOME" ]; then
  die "DEMO_HOME exists and is not a directory: $DEMO_HOME"
fi
if [ -d "$DEMO_HOME" ] && [ -n "$(ls -A "$DEMO_HOME")" ]; then
  if [ -f "$DEMO_HOME/$MARKER" ] && [ -f "$DEMO_HOME/config.yaml" ]; then
    # A previous run of this demo: start over, so session 2 can only recall what session 1 said.
    note "removing the previous demo profile in $DEMO_HOME"
    rm -rf "$DEMO_HOME"
  elif [ -f "$DEMO_HOME/config.yaml" ]; then
    die "$DEMO_HOME has a config.yaml that this demo did not create; refusing to touch it"
  elif [ ! -f "$DEMO_HOME/$MARKER" ]; then
    die "$DEMO_HOME is not empty and not a demo profile; pick a new directory"
  fi
fi
mkdir -p "$DEMO_HOME/work"
touch "$DEMO_HOME/$MARKER"

# From here on, everything uses the scratch profile. Hermes runs in an empty work directory so it
# does not load project context files (AGENTS.md, ...) from wherever the demo was started.
export HERMES_HOME="$DEMO_HOME"
export NO_COLOR=1
cd "$DEMO_HOME/work"

if [ -z "$MODEL" ]; then
  MODEL="$(detect_model)" ||
    die "could not list the models at $BASE_URL/models; start your model server, or set BASE_URL / MODEL"
  MODEL_SRC="auto-detected: first id from GET $BASE_URL/models; set MODEL to override"
else
  MODEL_SRC="from MODEL"
fi

printf '%span-agent demo%s: endpoint %s, model %s (%s)\n' "$BOLD" "$RESET" "$BASE_URL" "$MODEL" "$MODEL_SRC"
printf 'scratch profile: DEMO_HOME=%s (used as HERMES_HOME)\n' "$DEMO_HOME"

# The scratch profile's config files come first, so `pan doctor` reads the endpoint and the API
# key from the PAN gate config (models.gate) instead of the command line.
mkdir -p "$DEMO_HOME/pan"
write_configs
note "wrote \$DEMO_HOME/config.yaml (Hermes: model.provider custom, base_url $BASE_URL)"
note "wrote \$DEMO_HOME/pan/config.yaml (PAN's memory gate: the LLM gate on the same endpoint)"
pause

# -- 0. endpoint check --------------------------------------------------------------------------

say "0. Check the endpoint: tool calls (for Hermes) and json_schema (for PAN's memory gate)"
# No --api-key: doctor takes it from models.gate in $DEMO_HOME/pan/config.yaml.
doctor=("$PAN_BIN" doctor --base-url "$BASE_URL" --model "$MODEL")
if [ "$NO_THINKING" = "1" ]; then doctor+=(--no-thinking); fi
if [ -n "$DOCTOR_ARGS" ]; then read -ra extra <<< "$DOCTOR_ARGS"; doctor+=("${extra[@]}"); fi
rc=0
run "${doctor[@]}" || rc=$?
case "$rc" in
  0) ;;
  2) die "endpoint $BASE_URL is unreachable; start your model server or set BASE_URL" ;;
  *) note "pan doctor reported a failed check (exit $rc); continuing, but results may suffer" ;;
esac
pause

# -- 1. pan setup -----------------------------------------------------------------------------------

say "1. Install PAN into the scratch profile: pan setup"
# The Hermes config and PAN's gate config were written above. pan setup sets memory.provider=pan
# (+3 related settings, backed up first) and creates $DEMO_HOME/pan/ with a git-backed wiki and a
# search index; it keeps the existing pan/config.yaml. It starts nothing.
setup=("$PAN_BIN" setup)
if [ -n "$SETUP_ARGS" ]; then read -ra extra <<< "$SETUP_ARGS"; setup+=("${extra[@]}"); fi
run "${setup[@]}" | mask
pause

# -- 2. session 1 -------------------------------------------------------------------------------

say "2. Session 1: tell the agent a fact and a preference"
hermes_turn session1 "$FACT_PROMPT"
SESSION1="$(cat "$DEMO_HOME/session1.id")"
note "session: ${SESSION1:-?}"
pause

# -- 3. curation --------------------------------------------------------------------------------

say "3. Curate: pan-memoryd processes the queued turn once and exits"
run "$PAN_BIN" memoryd run --once 2>&1 | mask
pause

say "   What the gate and the curator decided (every decision is logged)"
inspect=("$PAN_BIN" memory inspect)
if [ -n "$SESSION1" ]; then inspect+=(--session "$SESSION1"); fi
run "${inspect[@]}" | sed -n '/^decisions/,$p'
pause

say "   The wiki page PAN wrote: search it, then read it"
run "$PAN_BIN" wiki search "staging database port" --limit 3 | tee "$DEMO_HOME/search.out"
PAGE="$(awk 'NR==1 && $1 ~ /^[0-9.]+$/ {print $2}' "$DEMO_HOME/search.out")"
if [ -n "$PAGE" ]; then
  run "$PAN_BIN" wiki read "$PAGE"
else
  note "no wiki page matched (the decisions above say where the fact went)"
fi
pause

say "   The wiki is a git repository: every change is a pan-memoryd commit"
run git --no-pager -C "$DEMO_HOME/pan/wiki" log --stat -1
run git --no-pager -C "$DEMO_HOME/pan/wiki" show --format='commit %h%n%s%n' HEAD
pause

say "   Hermes' own short memory (USER.md / MEMORY.md, in the prompt of every new session)"
for f in USER.md MEMORY.md; do
  if [ -s "$DEMO_HOME/memories/$f" ]; then
    run cat "$DEMO_HOME/memories/$f"; echo
  else
    note "\$DEMO_HOME/memories/$f: empty"
  fi
done
pause

# -- 4. session 2 -------------------------------------------------------------------------------

say "4. Session 2: a new session asks for the fact"
hermes_turn session2 "$RECALL_PROMPT"
SESSION2="$(cat "$DEMO_HOME/session2.id")"
note "session: ${SESSION2:-?} (session 1 was ${SESSION1:-?})"

if grep -q "$EXPECTED" "$DEMO_HOME/session2.out"; then
  RESULT="PASS"; printf '\n%sPASS%s: the new session recalled port %s\n' "$GREEN" "$RESET" "$EXPECTED"
else
  RESULT="FAIL"; printf '\n%sFAIL%s: the answer does not contain %s\n' "$RED" "$RESET" "$EXPECTED"
fi
pause

if [ -n "$SESSION2" ]; then
  say "   What PAN prefetched into that turn (wiki hits + standing preferences)"
  show_recall "$SESSION2"
  say "   Events PAN captured for session 2 (tool calls such as memory_search show up here)"
  run "$PAN_BIN" memory inspect --session "$SESSION2" | sed '/^decisions/,$d'
  pause
fi

# -- 5. summary ---------------------------------------------------------------------------------

say "5. Summary"
echo "  - session 1 stated a fact and a preference; pan-memoryd curated the turn"
if [ -n "$PAGE" ]; then
  echo "  - wiki page $PAGE (Markdown in git, with its source session) holds the fact"
fi
if [ -s "$DEMO_HOME/memories/USER.md" ]; then
  echo "  - USER.md holds the preference; Hermes puts USER.md / MEMORY.md into every new session"
fi
echo "  - session 2, a new session, was asked for the fact: $RESULT"
echo "    (sources it had: MEMORY.md if the fact landed there, PAN's prefetch block, memory tools)"
echo "  - everything lives in the scratch profile: DEMO_HOME=$DEMO_HOME"
echo
echo "Clean up with:  rm -rf \"$DEMO_HOME\""
[ "$RESULT" = "PASS" ]
