---
name: install-pan
description: "Install, check or remove PAN (Persistent Agent Nexus), the structured long-term memory add-on for an existing Hermes Agent install. Use when the user wants PAN memory in their Hermes, asks whether PAN is working, or wants to uninstall it."
version: 0.1.0
author: PAN
license: MIT
platforms: [linux, macos]
metadata:
  hermes:
    tags: [hermes, memory, pan, install, setup, plugin]
    homepage: https://github.com/FRST-FRYTL/pan-agent
    related_skills: [hermes-agent]
---

# Install PAN into an existing Hermes

PAN is a memory-provider plugin (`memory.provider: pan`) plus a background curator
(`pan-memoryd`). It installs **into the user's existing Hermes venv**; nothing of Hermes is
copied or changed except four settings in the profile's `config.yaml`. Everything below is driven
by the `pan` CLI. This skill only decides *what* to run and asks the user where a choice is theirs.

## Hard rules
- **Never touch a real profile without asking.** Show which `HERMES_HOME` you will change and get
  a yes. For a trial, offer a scratch profile (`HERMES_HOME=$(mktemp -d)`) first.
- **Dry-run first.** Run `pan setup --dry-run` (and `pan daemon install --dry-run`) and show the
  output before the real run.
- **Record the rollback before changing anything** (step 5).
- **Never enable services on your own.** `pan daemon install` only writes the unit file; enabling
  it is the user's call.
- If anything in this skill disagrees with `pan --help` or the installed version's README, the
  CLI wins. Say so and follow the CLI.

## 1. Find Hermes and its venv
```bash
command -v hermes && hermes --version
readlink -f "$(command -v hermes)"       # usually <install>/venv/bin/hermes
```
- Derive the venv python: `PY=<install>/venv/bin/python`. Check it:
  `"$PY" -c "import hermes_cli; print(hermes_cli.__version__)"`.
- If `hermes` is a shell wrapper and not a venv entry point, read it to find the venv. The
  standard non-root install lives in `$HERMES_HOME/hermes-agent` (default `~/.hermes/hermes-agent`).
- Profile: `HERMES_HOME` (default `~/.hermes`). Ask which profile PAN should serve if the user has
  more than one.
- No Hermes found → stop and point to Hermes' installer; PAN does not install Hermes.

## 2. Check compatibility
- Each PAN release is tested against one Hermes version. Get it from the release notes, or after
  installing from `pan version`, which prints `pinned hermes-agent X, running Y`.
- If the running Hermes is not the tested one: tell the user plainly. `pan setup` refuses it. The
  choices are: install the matching PAN release, move Hermes to the tested version, or continue at
  their own risk with `pan setup --force` (PAN then logs a warning at every startup).

## 3. Install the package into Hermes' venv
```bash
# releases are git tags only (no PyPI); pick the tag from the release notes:
uv pip install --python "$PY" "git+https://github.com/FRST-FRYTL/pan-agent@pan-v0.1.0a1"
```
- `pan-agent` does not depend on `hermes-agent`; it must go into the venv Hermes runs from, never a
  separate one. Its only runtime dependency (PyYAML) is already in Hermes' venv.
- Hermes' venv is created by uv and may have no `pip`; use `uv pip`. If `uv` is missing, use
  `"$PY" -m pip` only if pip is present.
- Check: `"$(dirname "$PY")/pan" version`. From here on, call `pan` from that venv.

## 4. Choose the memory gate
Every agent turn goes through a gate that decides what is worth remembering. It needs an
OpenAI-compatible chat endpoint; without one, PAN falls back to its rule classifier (lower
quality, no model calls).
- Ask the user which endpoint to use (local vLLM, Ollama, LM Studio, a hosted provider). Test
  it: `curl -s <base_url>/models`.
- Point out that with a hosted endpoint, every conversation turn is sent to that provider.
- Write the choice to `$HERMES_HOME/pan/config.yaml` after `pan setup` has created the default
  file (step 6):
  ```yaml
  models:
    gate:
      base_url: http://localhost:8000/v1
      model: <model id from /models>
      api_key: EMPTY            # or the key; never echo keys back to the user
  ```
  For rules-only: `classifier: {kind: rules}`.

## 5. Rollback
`pan setup` records the previous values of the four settings it changes (and a copy of
`config.yaml`) in `$HERMES_HOME/pan/setup-backup.json` before the first change, and never
overwrites that record. Show it to the user after setup. `pan uninstall` restores it. An empty
`memory.provider` is normal: it means Hermes' built-in memory.

## 6. Run setup
```bash
HERMES_HOME="$HERMES_HOME" pan setup --dry-run   # show the output, get a yes
HERMES_HOME="$HERMES_HOME" pan setup
```
`pan setup` sets `memory.provider: pan` and three related settings, creates `$HERMES_HOME/pan/`
(config, wiki, index), and never starts anything. Then apply step 4's gate settings.

## 7. Curator service (optional, ask)
`pan-memoryd` turns captured events into wiki pages and hot-memory entries. Options:
- Foreground or ad hoc: `pan memoryd run` / `pan memoryd run --once`.
- systemd user unit: `pan daemon install --dry-run`, then `pan daemon install --hermes-home
  "$HERMES_HOME"`. It writes the unit only. Tell the user how to enable it
  (`systemctl --user enable --now <unit>`, with the unit name that `pan daemon install` printed).

## 8. Verify with a test turn
```bash
HERMES_HOME="$HERMES_HOME" hermes chat -q "For the record: my test marker is PAN-CHECK-7." -Q </dev/null
HERMES_HOME="$HERMES_HOME" pan memoryd run --once
HERMES_HOME="$HERMES_HOME" pan status
HERMES_HOME="$HERMES_HOME" pan wiki search "PAN-CHECK-7"
```
- Expected: `pan status` shows the spool and curation log growing, and the marker is found (or
  the curation log explains why the gate dropped it: `pan memory inspect --session <id>`).
- If the gate shows `status: fallback`, the endpoint from step 4 is unreachable or too slow.
- Run the test in the scratch profile if the user doesn't want a marker in their real memory.

## Uninstall
```bash
HERMES_HOME="$HERMES_HOME" pan daemon uninstall        # if a unit was installed (disable it first)
HERMES_HOME="$HERMES_HOME" pan uninstall --dry-run     # show the restore commands, get a yes
HERMES_HOME="$HERMES_HOME" pan uninstall               # restores the settings from setup-backup.json
uv pip uninstall --python "$PY" pan-agent
```
`$HERMES_HOME/pan/` (wiki, events, logs) stays. It is the user's data; delete it only if they
ask.
