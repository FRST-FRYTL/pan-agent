# Security policy

## Versions

pan-agent is a research preview. Fixes, if any, go into the latest release only.

## Reporting a vulnerability

Please **do not open a public issue** for security problems. Report them privately through GitHub's
[private vulnerability reporting](https://github.com/FRST-FRYTL/pan-agent/security/advisories/new)
(the Security tab → "Report a vulnerability").

Include:
- the affected version;
- a description of the problem and its impact;
- the steps to reproduce it.

You should get an acknowledgement within 7 days. This is a one-person project, so a fix may take
longer. You will be credited in the advisory unless you prefer otherwise.

## Scope and threat model

PAN stores what your agent sees and says. It handles sensitive data by design. These are in scope:

- secrets that get past the gate's secret filter into the memory wiki, `MEMORY.md`/`USER.md` or logs;
- prompt injection through stored memory that makes the agent act against the user;
- path traversal or file writes outside `$HERMES_HOME/pan/` and Hermes' memory files;
- anything that makes `pan setup` or `pan uninstall` damage a Hermes profile.

**Data at rest:** `$HERMES_HOME/pan/events.db` keeps captured turns and tool-output excerpts
unredacted for `daemon.event_retention_days` (default 30), and `dead/` keeps copies of failed
events. The curation log records the agent's own memory writes as written. PAN redacts secrets from
what it writes to the wiki and `MEMORY.md`/`USER.md`. Protect the profile directory accordingly.

**Your responsibility:** the memory gate sends each conversation turn to the endpoint you configure.
If that endpoint is a hosted provider, your conversations go to it. Use a local endpoint or the
rules-only gate if that is not acceptable.
