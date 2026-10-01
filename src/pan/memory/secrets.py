"""Secret scanner / redactor (classifier design §6.1, failure mode F9).

:func:`redact` replaces secret *values* with ``[secret:<kind>]`` and reports which kinds it found.
It runs before a model sees an episode (Stage 0) and again on everything the gate outputs, so a
claim may keep the *fact* ("the staging API key is in ~/.config/app") but never the value.

Rules (small and explicit, gitleaks-like):

1. known token formats: OpenAI / Anthropic ``sk-…``, GitHub ``ghp_`` / ``github_pat_``, GitLab
   ``glpat-``, Slack ``xox?-``, AWS ``AKIA…``, Google ``AIza…``, Hugging Face ``hf_…``, Stripe
   ``sk_live_``, JWTs, ``Bearer <token>``, PEM private-key blocks;
2. credentials inside URLs (``scheme://user:password@host``);
3. ``password=`` / ``token: …`` pairs (EN + DE key words) — the value is redacted;
4. natural language: a key word (key, token, password, secret, Passwort, Kennwort, …) followed
   within a few words by "is" / "=" / "lautet" / "ist" and a token-like string (≥ 8 chars, letters
   and digits mixed, or ≥ 20 chars of high entropy).
"""

from __future__ import annotations

import math
import re
from typing import List, Tuple

_TOKEN_PATTERNS: Tuple[Tuple[str, re.Pattern], ...] = (
    ("private_key", re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----.*?(?:-----END [A-Z0-9 ]*PRIVATE KEY-----|\Z)",
                               re.S)),
    ("api_key", re.compile(r"\bsk-(?:ant-|proj-)?[A-Za-z0-9_-]{12,}")),
    ("api_key", re.compile(r"\b(?:sk|rk|pk)_(?:live|test)_[A-Za-z0-9]{12,}")),
    ("github_token", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})")),
    ("gitlab_token", re.compile(r"\bglpat-[A-Za-z0-9_-]{16,}")),
    ("slack_token", re.compile(r"\bxox[abposr]-[A-Za-z0-9-]{10,}")),
    ("aws_key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("google_key", re.compile(r"\bAIza[0-9A-Za-z_-]{30,}")),
    ("hf_token", re.compile(r"\bhf_[A-Za-z0-9]{30,}")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}")),
    ("bearer", re.compile(r"(?<=\bBearer )[A-Za-z0-9._~+/=-]{16,}")),
)
_URL_CRED = re.compile(r"(?P<pre>\b[a-z][a-z0-9+.-]*://[^\s:/@]+:)(?P<pw>[^\s@/]+)(?=@)", re.I)
_KEY_WORDS = (r"passwor[dt]|passwd|pwd|kennwort|secret|geheimnis|token|api[_ -]?key|access[_ -]?key|"
              r"client[_ -]?secret|private[_ -]?key|auth[_ -]?key|credentials?|zugangsdaten|schl(?:ü|ue)ssel")
_PAIR = re.compile(rf"(?P<pre>(?<!\[)\b(?:[\w-]*?(?:{_KEY_WORDS}))\s*[:=]\s*[\"']?)(?P<val>[^\s\"',;]{{4,}})", re.I)
_PROSE = re.compile(rf"(?P<pre>(?<!\[)\b(?:{_KEY_WORDS})\b[^.\n]{{0,60}}?\b(?:is|was|=|lautet|ist|heißt|heisst)\s+"
                    rf"[\"'`]?)(?P<val>[^\s\"'`,;]{{8,}})", re.I)
_PLACEHOLDER = re.compile(r"^\[secret:[a-z_]+\]$|^(?:x+|\*+|\.+|<[^>]*>|\$\{?[A-Z_]+\}?|changeme|redacted)$", re.I)


def _entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = {c: s.count(c) for c in set(s)}
    return -sum(n / len(s) * math.log2(n / len(s)) for n in counts.values())


def _token_like(value: str) -> bool:
    v = value.strip(".")
    if _PLACEHOLDER.match(v) or "/" in v or v.startswith(("~", "$", "[")):
        return False  # a path or a placeholder is where a secret lives, not the secret
    mixed = bool(re.search(r"[A-Za-z]", v)) and bool(re.search(r"\d", v))
    return (len(v) >= 8 and mixed) or (len(v) >= 20 and _entropy(v) >= 3.5)


def redact(text: str) -> Tuple[str, List[str]]:
    """(``text`` with secret values replaced by ``[secret:<kind>]``, kinds found in order)."""
    if not text:
        return text or "", []
    found: List[str] = []

    def sub(kind: str):
        def repl(m: re.Match) -> str:
            found.append(kind)
            return f"[secret:{kind}]"
        return repl

    out = str(text)
    for kind, pattern in _TOKEN_PATTERNS:
        out = pattern.sub(sub(kind), out)

    def url(m: re.Match) -> str:
        found.append("url_credentials")
        return m.group("pre") + "[secret:url_credentials]"
    out = _URL_CRED.sub(url, out)

    def pair(kind: str, need_token: bool):
        def repl(m: re.Match) -> str:
            val = m.group("val")
            if val.startswith("[secret:") or _PLACEHOLDER.match(val) or (need_token and not _token_like(val)):
                return m.group(0)
            if not need_token and (val.startswith(("/", "~", "$")) or len(val) < 6 and not re.search(r"\d", val)):
                return m.group(0)
            found.append(kind)
            return m.group("pre") + f"[secret:{kind}]"
        return repl
    out = _PAIR.sub(pair("credential", False), out)
    out = _PROSE.sub(pair("credential", True), out)
    return out, found


def contains_secret(text: str) -> bool:
    return bool(redact(text)[1])
