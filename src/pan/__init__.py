"""PAN (Persistent Agent Nexus): structured, auditable long-term memory for Hermes Agent."""

__version__ = "0.1.0a1"

# Hermes version this release was tested against (with the commit in HERMES_BASE; ADR-011). `pan setup`
# refuses other versions without --force, and the provider logs a warning at startup.
HERMES_PIN = "0.21.4"
