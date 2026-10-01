"""Hermes integration: entry point for the `hermes_agent.memory_providers` group (``pan = "pan.hermes"``)."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def register(ctx) -> None:
    """Called by Hermes' memory-provider loader: the provider, then the general hook subscriptions."""
    from pan.hermes.provider import PanMemoryProvider

    ctx.register_memory_provider(PanMemoryProvider())
    try:
        from pan.hermes.hooks import register_hooks
        register_hooks(ctx)
    except Exception as exc:  # hooks are best-effort; never cost Hermes the provider
        logger.warning("PAN: hook registration failed: %s", exc)
