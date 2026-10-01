"""Memory classifier (integration spec §4.6, spec v0.1 §11). MVP: explicit rules.

``Classifier.classify(episode, previous)`` returns one or more :class:`MemoryCandidate` for an
episode; an episode with nothing worth remembering yields exactly one ``noise`` candidate (so every
episode leaves a labelled record in the curation log — the training set for a later encoder model).

Rules (checked in this order; at most one wiki candidate per episode, plus a user preference):

1. **user_preference → user** — a user sentence states a standing preference: "I prefer …",
   "from now on …", "going forward …", "by default …", or starts with always / never / don't /
   do not / stop / avoid. Questions are ignored.
2. **decision → wiki** — the previous turn's assistant text proposes something ("Proposal:",
   "I suggest", "we should", "let's", …) and this turn's user message confirms it ("yes", "ok",
   "go with …", "sounds good", …, no negation); or the user states a decision directly
   ("decision: …", "we decided …", "let's go with …").
3. **learning → wiki** — a tool call failed and a later call in the same episode succeeded with the
   same program/tool, or a file was changed after the failure (error followed by a fix).
4. **environment / configuration → wiki** — the user states an environment fact (M5: "Note for later:
   our vLLM server runs on port 8000", "Update: we moved the vLLM server to port 8010" — a declarative,
   non-imperative sentence with an infra noun and a value or state/move verb), an infrastructure
   command succeeded (docker, systemctl, vllm, nvidia-smi, curl localhost, …), a config/infra file
   was changed (→ configuration), or the assistant explains server/environment state ("… was
   started without …", "… listens on port …").
5. everything else → **noise → none** (dropped).

Candidates carry a **subject** (M5, :mod:`pan.memory.facts`): the noun phrase the facts are about
("vLLM server", "Langfuse instance"), or the topic of a known command (``nvidia-smi`` → "GPU").
It becomes the page title, so pages are named after what they describe, not after a command line
or the assistant's acknowledgement ("Got it — noted that …").

Recall is not observation (M4): an assistant explanation (``inferred`` claim) that mostly restates
what the agent just read from PAN memory in the same episode (``memory_search``/``memory_read``
results, the prefetch context — and, since M5, the L1 snapshot the daemon passes as ``known``) is dropped — otherwise every answer from the wiki would be written
back into the wiki as an "update". Failed calls of memory tools do not count as tool errors.
Likewise an assistant sentence that restates the *user's* words ("Saved: vLLM server on port 8000")
is dropped — the user's own sentence is the observed claim.

Meta-talk is not a fact (M6): an assistant sentence about memory, saving or the wiki, one that
quotes a phrase of three or more words, one longer than ``MAX_INFERRED_CHARS``, and — for
environment facts — one in the first person ("I did save the fact to my own persistent memory …")
never becomes an ``inferred`` claim. The same filter keeps an agent's L1 notes about memory/tools
out of the wiki mirror (``pan.daemon.memoryd``).

Tool calls printed as text (M5, :mod:`pan.memory.toolcall_text`) are removed from the assistant
text before any rule looks at it.

Claims are concise sentences taken from the episode. Each claim is labelled ``observed`` (tool
output, the user's own words, a user-confirmed proposal) or ``inferred`` (assistant explanations).
"""

from __future__ import annotations

import json
import re
from typing import List, Optional, Protocol, Sequence

from pan.events.schema import (AgentEvent, Destination, Lifetime, MemoryCandidate, MemoryClassification,
                               MemoryType)
from pan.memory.claims import normalize, shorten, token_set
from pan.memory.claims import sentences as _sentences
from pan.memory.episodes import RECALL_TOOLS, Episode
from pan.memory.facts import RETRACT_VERB, command_topic, display_subject, strip_lead, subject_of, subject_tags, values
from pan.memory.toolcall_text import strip_tool_call_text

OBSERVED = "observed"
INFERRED = "inferred"
REPORTED = "reported"   # T3: what the assistant said it did or found (not verified), see llm_gate
MAX_CLAIMS = 6
MAX_CLAIM_CHARS = 300
# An inferred claim with at least this share of its content tokens in the episode's recalled text
# (and every numeric token) is a restatement of wiki knowledge, not a new fact.
RECALL_ECHO_OVERLAP = 0.6
# Agent memory tools: their failures are not "tool errors followed by a fix" (learning rule).
MEMORY_TOOLS = RECALL_TOOLS | {"memory", "session_search"}

# -- rule vocabulary (keep small and explicit) ----------------------------------------------------

_PREF_LEAD = re.compile(r"^(?:(?:please|and|also|ok|okay|bitte)[,\s]+)*(?:from now on|going forward|in (?:the )?future|"
                        r"by default|for (?:all )?future (?:chats|sessions|conversations)|"
                        r"(?:a|one|my) (?:standing )?preference(?: for (?:all )?(?:future )?(?:chats|sessions))?|"
                        r"ab jetzt|von nun an|ab sofort|k(?:ü|ue|u)nftig|in zukunft|grunds(?:ä|ae|a)tzlich)\b[,:]?\s*",
                        re.I)
_PREF_ANY = re.compile(r"\bI(?: really)? (?:prefer|would prefer|'d prefer|dislike|hate it when|like it when)\b|"
                       r"\b(?:my|a|one|standing) preference\b|\bfrom now on\b|\bgoing forward\b|\bby default\b|"
                       r"\bfor (?:all )?future (?:chats|sessions|conversations)\b|"
                       # German, e.g. "Bitte antworte mir ab jetzt immer auf Deutsch"
                       r"\bab jetzt\b|\bvon nun an\b|\bab sofort\b|\bk(?:ü|ue|u)nftig\b|\bin zukunft\b|"
                       r"\bich (?:bevorzuge|m(?:ö|oe|o)chte (?:immer|nie|lieber)|h(?:ä|ae|a)tte (?:gern|lieber))\b", re.I)
_PREF_START = re.compile(r"^(?:please\s+|bitte\s+)?(?:always|never|don't|do not|stop|avoid|immer|nie|niemals)\b", re.I)
# "Don't open the file again", "don't run it": instructions for the task at hand, not standing
# preferences (stored as USER.md "preferences").
_TASK_SCOPED = re.compile(r"\b(?:again|it|this|that|these|those|them|yet|here|anymore|the (?:file|command|script|"
                          r"log|output|result)s?|for now|this time|today)\b", re.I)
# Personal profile facts, e.g. "my editor is Emacs, my shell is zsh, and I use a split
# keyboard" → USER.md. English + German ("mein Editor ist …").
_PROFILE = re.compile(r"\bmy (?:[\w-]+ ){0,2}?(?:editor|ide|shell|terminal|os|distro|laptop|keyboard|layout|"
                      r"browser|language|timezone|time zone|name|role|job|team|email|username|handle|pronouns|"
                      r"setup|machine|workflow|stack|font|theme)s? (?:is|are|=)\b|"
                      r"\bI (?:use|type (?:on|with)|work (?:with|on|in)|code in|live in|am based in|'m based in)\b|"
                      r"\bmein(?:e)? (?:[\w-]+ ){0,2}?(?:editor|shell|terminal|tastatur|layout|sprache|zeitzone|"
                      r"rechner|laptop|name|rolle|team)\b[^.?!]* (?:ist|sind)\b", re.I)

_PROPOSAL = re.compile(r"^(?:proposal|decision|recommendation|suggestion)\s*:\s*|"
                       r"^(?:I (?:propose|suggest|recommend)|we could|we should|let's|let us|"
                       r"my recommendation is|I'd go with)\b(?:\s+that)?\s*", re.I)
_PROPOSAL_ANY = re.compile(r"\b(?:I (?:propose|suggest|recommend)|we should|should we|shall we|"
                           r"I'd go with)\b", re.I)
_CONFIRM = re.compile(r"^\W*(?:yes|yep|yeah|ok(?:ay)?|sure|agreed|agree|sounds good|lgtm|approved?|"
                      r"confirmed?|do it|go ahead|go with|let's do|let's go|makes sense|perfect|great)\b", re.I)
_NEGATION = re.compile(r"\b(?:no|not|don't|nope|rather not|instead|wait|but)\b", re.I)
_USER_DECISION = re.compile(r"^(?:decision\s*:|we (?:decided|agreed)(?:\s+to|\s+that)?|"
                            r"(?:let's|we'll|we will) go with)\s*", re.I)

_INFRA_CMD = re.compile(r"^\s*(?:sudo\s+)?(?:docker(?:-compose)?|podman|systemctl|kubectl|vllm|nvidia-smi|ss|"
                        r"netstat|ufw|crontab)\b|\bcurl\b[^|;]*\b(?:localhost|127\.0\.0\.1)\b", re.I)
_CONFIG_PATH = re.compile(r"(?:\.ya?ml|\.toml|\.ini|\.conf|\.cfg|\.env|\.service|\.socket|\.timer)$|"
                          r"(?:^|/)(?:Dockerfile|docker-compose[^/]*|Caddyfile|nginx[^/]*|\.env[^/]*)$|^/etc/", re.I)
# Infrastructure nouns for facts *without* a typed value ("the vLLM server was started without …")
# and for assistant explanations. Statements with a typed value, explicit notes and retractions do
# not need them (the gate is structural); the opt-1 topic additions are removed.
_INFRA_NOUN = re.compile(r"\b(?:server|service|daemon|container|port|gpu|vllm|docker|systemd|host|database|"
                         r"postgres|redis|nginx|endpoint|cluster|volume|disk|partition|cuda|driver|kernel|"
                         r"config(?:uration)?|langfuse|model|instance|machine|cpu|ram|repo(?:sitory)?|url|api|"
                         r"vm|proxy|hostname|ip|registry|bucket|broker|queue|subnet|vpn|dns|index)\b|"
                         r"\b[\w-]*(?:server|dienst|datenbank|rechner)\b", re.I)
_STATE_VERB = re.compile(r"\b(?:is|are|was|were) (?:running|listening|started|stopped|configured|installed|"
                         r"enabled|disabled|mounted|bound|deployed|served|exposed|down|up|called|named|hosted|"
                         r"located|stored|kept)\b|"
                         r"\b(?:serves|listens on|runs on|runs as|started (?:with|without)|is set to|"
                         r"points to|uses port|on port \d+|lives (?:in|on|at))\b|"
                         r"\b(?:machine|host|server|system|box|node|computer|it)\s+(?:has|have|comes with|"
                         r"is equipped with)\s+(?:an?\s+|\d+\s*x?\s*)?(?!no\b)[^.]*\b(?:gpu|cpu|ram|cores?|disk|ssd)s?\b|"
                         # German: "heißt build-cache", "liegt im Rechenzentrum", "läuft auf …"
                         r"\b(?:hei(?:ß|ss)t|hei(?:ß|ss)en|liegt|liegen|l(?:ä|ae|a)uft|laufen|befindet sich|lautet|"
                         r"ist (?:unter|auf|in|im|bei))\b",
                         re.I)
# User sentences that ask for something (not statements of fact).
_REQUEST = re.compile(r"^(?:please\s+|pls\s+|can you\s+|could you\s+)?(?:run|show|tell|check|can|could|would|what|"
                      r"which|who|how|why|where|when|summarize|explain|write|create|fix|make|give|list|find|look|"
                      r"open|read|search|start|stop|restart|install|deploy|help|do|does|did|is|are|use|try|let|"
                      r"answer|reply|save|remember|"
                      # German requests / questions
                      r"schreib|schreibe|zeig|zeige|erkl(?:ä|ae|a)r|mach|gib|wie|was|welche[rsn]?|wo|wann|warum|"
                      r"kannst|k(?:ö|oe|o)nntest|bitte)\b", re.I)
_MOVE_VERB = re.compile(r"\b(?:moved|migrated|switched|changed|relocated|upgraded|downgraded|renamed|repointed)\b"
                        r".*\b(?:to|from)\b", re.I)
# Hedges, hypotheticals and conditionals: not statements of fact (EN + DE).
_HEDGE = re.compile(r"\b(?:maybe|might|probably|perhaps|not sure|I think|I guess|could be|hypothetical(?:ly)?|"
                    r"in theory|theoretically|suppose|supposing|imagine|if we (?:ever|were)|would (?:cost|be|need|take)|"
                    r"we'd|vielleicht|wahrscheinlich|eventuell|hypothetisch|theoretisch|angenommen|falls wir (?:mal|je)|"
                    r"würde[n]? (?:kosten|sein|brauchen|dauern)|wuerde[n]? (?:kosten|sein|brauchen|dauern))\b", re.I)
# Assistant sentences about itself, its memory or its tools are never facts about the system
# (seen in a bench run: 'I did save the fact to my own persistent memory …, so "vLLM
# server runs on port 8000" is retained' was curated as an inferred fact and later prefetched).
_META = re.compile(r"\b(?:memory|memories|remember(?:ed|s)?|save[ds]?|saving|noted|persist(?:ent|ed|s)?|"
                   r"retained|wiki|tool[- ]name)\b", re.I)
_FIRST_PERSON = re.compile(r"\b(?:I|I'm|I've|I'll|I'd|me|my|myself|mine)\b")
_QUOTE = re.compile(r'["“„][^"“”„]*\s[^"“”„]*\s[^"“”„]*["”]')  # a quoted phrase of 3+ words (a restatement)
# Inferred claims longer than this are prose, not a fact sentence.
MAX_INFERRED_CHARS = 200
_FIX_EXPLAIN = re.compile(r"\b(?:because|caused by|the fix|fixed|solution|root cause|needs?|required|"
                          r"missing|has to|must)\b", re.I)

KNOWN_TAGS = ("vllm", "docker", "systemd", "gpu", "cuda", "langfuse", "postgres", "sqlite", "git", "hermes",
              "pan-memoryd", "spool", "wiki", "fts5", "nginx", "redis", "tool-calling", "dgx-spark", "markdown")

# (should_remember, relevance, importance, confidence, lifetime, destination) per type.
_PROFILES = {
    MemoryType.USER_PREFERENCE: (True, 0.9, 0.7, 0.8, Lifetime.LONG_TERM, Destination.USER),
    MemoryType.DECISION: (True, 0.9, 0.8, 0.8, Lifetime.LONG_TERM, Destination.WIKI),
    MemoryType.LEARNING: (True, 0.8, 0.7, 0.7, Lifetime.LONG_TERM, Destination.WIKI),
    MemoryType.ENVIRONMENT: (True, 0.8, 0.6, 0.7, Lifetime.LONG_TERM, Destination.WIKI),
    MemoryType.CONFIGURATION: (True, 0.8, 0.6, 0.7, Lifetime.LONG_TERM, Destination.WIKI),
    MemoryType.NOISE: (False, 0.1, 0.0, 0.9, Lifetime.TRANSIENT, Destination.NONE),
}


class Classifier(Protocol):
    name: str

    def classify(self, episode: Episode, previous: Optional[Episode] = None, *,
                 known: str = "") -> List[MemoryCandidate]:
        """Candidates for ``episode``; ``previous`` = the session's preceding turn episode (or None);
        ``known`` = text the agent had in its prompt anyway (the L1 snapshot: MEMORY.md / USER.md)."""


def classification(mtype: MemoryType, *, confidence: Optional[float] = None) -> MemoryClassification:
    remember, relevance, importance, conf, lifetime, dest = _PROFILES[mtype]
    return MemoryClassification(should_remember=remember, relevance=relevance, importance=importance,
                                confidence=conf if confidence is None else confidence, lifetime=lifetime,
                                type=mtype, destination=dest)


# -- small text helpers ---------------------------------------------------------------------------

def sentences(text: str) -> List[str]:
    """Chat text: every line break is a sentence break."""
    return _sentences(text, join_lines=False)


def _clean(sentence: str) -> str:
    s = re.sub(r"\s+", " ", sentence).strip()
    if len(s) > MAX_CLAIM_CHARS:
        s = s[:MAX_CLAIM_CHARS].rsplit(" ", 1)[0] + " …"
    first = s.split(" ", 1)[0] if s else ""
    return s[:1].upper() + s[1:] if first.islower() else s  # keep "vLLM", "`cmd`", "ADR-3"


def _declarative(sentence: str) -> bool:
    return bool(sentence.strip()) and not sentence.rstrip().endswith("?")


def _command(event: AgentEvent) -> str:
    args = event.content.get("args")
    if isinstance(args, dict):
        cmd = args.get("command") or args.get("cmd")
        if cmd:
            return re.sub(r"\s+", " ", str(cmd)).strip()
    return ""


def _ok(event: AgentEvent) -> bool:
    return str(event.content.get("status") or "").lower() in ("ok", "success", "succeeded")


def _failed(event: AgentEvent) -> bool:
    status = str(event.content.get("status") or "").lower()
    return status not in ("", "ok", "success", "succeeded", "unknown")


def _program(event: AgentEvent) -> str:
    cmd = _command(event)
    words = [w for w in cmd.split() if w not in ("sudo", "env")]
    return words[0] if words else str(event.content.get("tool") or "")


def _tool_output(text: str) -> str:
    """Hermes' terminal tool returns JSON (``{"output": "...", "exit_code": 0}``): its ``output``."""
    raw = str(text or "").strip()
    if raw.startswith("{"):
        try:
            data = json.loads(raw)
        except ValueError:
            return raw
        if isinstance(data, dict):
            for key in ("output", "stdout", "result", "content"):
                if isinstance(data.get(key), str) and data[key].strip():
                    return data[key]
    return raw


def _first_line(text: str, max_len: int = 200) -> str:
    """First line with some substance (skips JSON brackets and blank lines)."""
    text = _tool_output(text)
    for line in str(text or "").splitlines():
        line = line.strip()
        if len(re.findall(r"[A-Za-z0-9]+", line)) >= 2:
            return line if len(line) <= max_len else line[:max_len].rsplit(" ", 1)[0] + " …"
    return ""


def _tags(texts: Sequence[str]) -> List[str]:
    blob = " " + normalize(" ".join(texts)).replace("tool calling", "tool-calling") + " "
    return [t for t in KNOWN_TAGS if re.search(rf"(?<![a-z0-9-]){re.escape(t)}(?![a-z0-9-])", blob)]


def _recalled(sentence: str, recall_tokens: set) -> bool:
    """``sentence`` mostly restates recalled wiki text (see RECALL_ECHO_OVERLAP)."""
    want = token_set(sentence)
    if not want or not recall_tokens:
        return False
    numeric = {t for t in want if any(c.isdigit() for c in t)}
    return numeric <= recall_tokens and len(want & recall_tokens) / len(want) >= RECALL_ECHO_OVERLAP


def meta_talk(sentence: str) -> bool:
    """The assistant talks about itself, memory/saving/tools, or quotes (restates) text."""
    return bool(_META.search(sentence) or _QUOTE.search(sentence))


def _inferred_ok(sentence: str, *, first_person_ok: bool = False) -> bool:
    """Stricter bar for claims taken from assistant text (evidence ``inferred``)."""
    if meta_talk(sentence) or len(sentence) > MAX_INFERRED_CHARS:
        return False
    return first_person_ok or not _FIRST_PERSON.search(sentence)


def _limit(claims: List[str], evidence: List[str]) -> tuple[List[str], List[str]]:
    seen, out_c, out_e = set(), [], []
    for c, e in zip(claims, evidence):
        key = normalize(c)
        if c and key not in seen:
            seen.add(key)
            out_c.append(c)
            out_e.append(e)
    return out_c[:MAX_CLAIMS], out_e[:MAX_CLAIMS]


# -- the rule classifier --------------------------------------------------------------------------

# Acknowledgements the assistant puts in front of a restatement ("Noted — …", "Saved: …").
_ACK = re.compile(r"^(?:\W*(?:noted|saved|got it|done|understood|okay|ok|sure|recorded|remembered|will do|"
                  r"(?:updated|saved|stored)(?: (?:it )?(?:to|in))?(?: (?:the|my))? memory|memory updated|updated|"
                  r"i(?:'ve| have) (?:saved|noted|updated|recorded|stored)(?: (?:it|that|this))?|i'll remember(?: that)?|"
                  r"i will remember(?: that)?|notiert|verstanden|alles klar|gespeichert|erledigt|gemerkt|"
                  r"ich habe (?:es |das )?(?:notiert|gespeichert|vermerkt))\b[\s,:;.!—–-]*)+", re.I)


def _unack(sentence: str) -> str:
    return _ACK.sub("", sentence).strip()


def _user_fact_sentences(user_text: str) -> List[str]:
    """Environment facts stated by the user (rule 4), lead-ins ("Note for later:") removed.

    M6: German sentences (lead-ins, nouns and verbs), a generic "<topic>:" lead before the fact
    ("We reshuffled things: billing now listens on 7005"), enumerations split into one claim per
    item ("auth runs on port 7001, billing on 7002" → two claims), retractions ("we cancelled daily
    standups") and project/process facts ("Our team's daily standup is at 10:15")."""
    out = []
    for s in sentences(user_text):
        body = strip_lead(s)
        head, sep, rest = body.partition(": ")
        lead_in = strip_lead(s) != s.strip()
        if sep and len(head.split()) <= 8 and not values(head).slots and len(rest.split()) >= 3:
            body = strip_lead(rest)
            lead_in = lead_in or len(head.split()) <= 3
        if not _declarative(s) or len(body.split()) < 4:
            continue
        request = ((_REQUEST.match(body) and not _FACT_VERB_2ND.match(body))  # "Search moved to …" is a fact
                   or _ASK.search(body))
        if (_PREF_ANY.search(body) or _PREF_START.match(body) or _USER_DECISION.match(body) or request
                or _PROFILE.search(body)):
            continue
        # A typed value (port, path, version, measure, time, date …) makes a first-party statement a
        # fact whatever its noun — also mentioned in passing inside a task ("Draft a commit message:
        # the poller now retries 4 times"). A bare "it's <value>" has no subject: not on its own.
        vals = values(body).slots
        valued = bool(vals) and not (_PRONOUN_COPULA.search(body) and not subject_of(body))
        # An explicit remember-intent lead-in ("Note for later:", "FYI", "Remember that …", "Zur Info:",
        # "Merk dir, dass …") keeps the note whatever it contains.
        intent = bool(_INTENT_LEAD.match(s.strip()) or (sep and _INTENT_LEAD.match(rest)))
        if _HEDGE.search(body) or _OTHER_PARTY.search(body) \
                or not (valued or intent or _INFRA_NOUN.search(body) or RETRACT_VERB.search(body)):
            continue
        if (intent or values(body).slots or _STATE_VERB.search(body) or _MOVE_VERB.search(body)
                or RETRACT_VERB.search(body)):
            out.extend(_clean(c) for c in split_enumeration(body))
    return out


# Attribution: whose fact is it? First-party subjects pass — "our/my/we/I + noun", "our team's …",
# "my laptop's …", a product's possessive ("Grafana's port"). Third-party attribution blocks
# capture (EN + DE):
#   - a person noun's possessive or a person noun as the owner: "my colleague's project", "my
#     friend Tom's NAS", "mein Kollege nutzt …", "bei meinem Freund …", "von meiner Nachbarin …";
#   - third-person possessive determiners: "his server", "their staging API", "sein Server";
#   - contrast with us: "(not ours)", "not our …", "nicht unser…", "nicht bei uns";
#   - reported speech about others: "my colleague said …", "Tom told me …", "laut meinem Kollegen".
_PERSON_NOUNS = (r"(?:friend|colleague|co-?worker|neighbou?r|boss|manager|wife|husband|partner|girlfriend|boyfriend|"
                 r"brother|sister|mother|father|mom|mum|dad|son|daughter|cousin|uncle|aunt|roommate|flatmate|"
                 r"client|customer|teammate|kid|child)s?")
_PERSON_NOUNS_DE = (r"(?:freund(?:in)?|kolleg(?:e|in|en)|nachbar(?:in|n)?|chef(?:in)?|frau|mann|bruder|schwester|"
                    r"mutter|vater|sohn|tochter|cousin[e]?|onkel|tante|mitbewohner(?:in)?|kund(?:e|in|en))")
_OTHER_PARTY = re.compile(
    rf"\b(?:my|our|a|the|his|her|their)\s+(?:[\w-]+\s+)?{_PERSON_NOUNS}(?:\s+[A-Z][a-z]+)?'s\b|"
    rf"\b(?:my|a)\s+{_PERSON_NOUNS}\s+(?:[A-Z][a-z]+\s+)?(?:uses|runs|has|hosts|keeps|said|says|told|mentioned|"
    rf"thinks|owns|set up|sets up)\b|"
    r"\b(?:his|their)\s+(?!own\b)[\w-]+|\bher\s+(?!own\b)[\w-]+|"
    r"\(?\bnot ours\b\)?|\bnot (?:our|my)\s+[\w-]+|"
    r"\b[A-Z][a-z]+\s+(?:said|says|told me|mentioned|thinks)\b|"
    rf"\b(?:mein|meine|meinem|meinen|meiner|bei meinem|bei meiner|von meinem|von meiner|laut meinem|laut meiner)\s+"
    rf"{_PERSON_NOUNS_DE}\b|"
    r"\b(?:sein|seine|seinem|seinen|seiner)\s+[\w-]+|"
    r"\bnicht (?:unser\w*|bei uns)\b", re.I)
# Lead-ins that ask to keep something (not conversational ones like "by the way" / "übrigens").
# "Remember" / "note" / "merk dir" need a colon, dash or that/dass ("Remember the checklist for
# later" states nothing).
_INTENT_LEAD = re.compile(r"^\W*(?:(?:note(?: for later| to self)?|please note|for the record|for later|remember|reminder|"
                          r"keep in mind|bear in mind|notiz|hinweis|merk dir|merke dir|für später|fuer spaeter)"
                          r"\s*(?:[:\-—–]|,? that\b|,? dass\b)|"
                          r"(?:fyi|just so you know|good to know|heads[- ]up|for your information|zur info|nur zur info|"
                          r"kurze info|zur kenntnis|gut zu wissen)\b)", re.I)
_PRONOUN_COPULA = re.compile(r"\b(?:it|that|this)(?:'s|\s+is|\s+was)\b|\b(?:es|das) (?:ist|war)\b", re.I)
# Asks anywhere in the sentence, not only at its start ("Have a look at x.toml and tell me …").
_ASK = re.compile(r"\b(?:tell me|show me|let me know|give me|can you|could you|would you|have a look|take a look|"
                  r"sag mir|zeig mir|gib mir|kannst du|könntest du|koenntest du|schau (?:dir )?(?:mal )?)\b", re.I)
_FACT_VERB_2ND = re.compile(r"^[\w-]+\s+(?:now\s+)?(?:moved|migrated|runs|is|are|was|were|listens|lives|has|uses|"
                            r"serves|sits|got)\b", re.I)
_INFRA_SLOTS = {"port", "host", "ip", "url", "path"}
# "Scratch that, it's May 5." right after "the launch review is on May 3" (M6 opt-1:
# the correction was noise, the wiki kept May 3 and the prefetch pulled the answer back to it).
_CORRECTION = re.compile(r"^\W*(?:scratch that|correction|sorry|oops|actually|wait|no,|nope|my bad|typo|i meant|"
                         r"ich meinte|korrektur|nein,|halt|moment)\b|\b(?:I misread|I mistyped|I meant|"
                         r"not \S+ but|ich habe mich (?:verlesen|vertippt))\b", re.I)


def _corrected_facts(user_text: str, previous: Optional[Episode]) -> List[str]:
    """The previous turn's user facts with the value this turn corrects (same slot, one value)."""
    if previous is None or not _CORRECTION.search(user_text or ""):
        return []
    new = values(user_text)
    out = []
    for fact in _user_fact_sentences(previous.user_text):
        old = values(fact)
        for slot in set(old.slots) & set(new.slots):
            if len(old.slots[slot]) == 1 and len(new.slots[slot]) == 1 and old.slots[slot] != new.slots[slot]:
                (ov,), (nv,) = old.slots[slot], new.slots[slot]
                out.append(fact.replace(old.spans[slot][ov], new.spans[slot][nv], 1))
                break
    return out


_FREED = re.compile(r"^(?:and\s+)?(?:port\s+)?\d{2,5}\s+(?:is|are)\s+(?:now\s+)?(?:free|unused|available|released)\b", re.I)
_ENUM_FIRST = re.compile(r"^(?P<ent>[A-Za-z][\w .-]{0,40}?)\s+(?P<mid>(?:now\s+)?(?:runs?|listens?|is|are|lives?|sits?|"
                         r"serves?|uses?)\b(?:\s+on)?(?:\s+port)?\s*)(?P<val>[\w.:/-]+)$", re.I)
_ENUM_NEXT = re.compile(r"^(?:and\s+)?(?P<ent>[A-Za-z][\w .-]{0,40}?)\s+(?:on|at|is|=|:)?\s*(?:port\s+)?"
                        r"(?P<val>[\w.:/-]*\d[\w.:/-]*)$", re.I)


def split_enumeration(body: str) -> List[str]:
    """``["auth runs on port 7001", "Billing runs on port 7002", …]`` for an enumeration of
    same-shaped facts; freed values ("7002 is free") are dropped; otherwise ``[body]``."""
    text = body.strip().rstrip(".")
    parts = [p.strip() for p in re.split(r",\s*(?:and\s+)?|;\s*|\s+and\s+(?=(?:the\s+|our\s+)?[A-Za-z][\w-]*\s+"
                                         r"(?:on|at|is|now)\b)", text)
             if p.strip()]
    kept = [p for p in parts if not _FREED.match(p)]
    if len(parts) < 2 or not kept:
        return [body]
    if len(kept) == 1:
        return [kept[0][:1].upper() + kept[0][1:] + "."]
    first = _ENUM_FIRST.match(kept[0])
    if first is None:
        return [body]
    out = [kept[0] + "."]
    for p in kept[1:]:
        if not re.search(r"\d", p) and not values(p).slots:
            return [body]
        if _ENUM_FIRST.match(p):
            out.append(p + ".")
            continue
        m = _ENUM_NEXT.match(p)
        if m is None:
            return [body]
        out.append(f"{m.group('ent').strip()} {first.group('mid').strip()} {m.group('val')}.")
    return [o[:1].upper() + o[1:] if o.split(" ", 1)[0].islower() else o for o in out]


class RuleClassifier:
    name = "rules-v2"

    def classify(self, episode: Episode, previous: Optional[Episode] = None, *,
                 known: str = "") -> List[MemoryCandidate]:
        assistant, printed_call = strip_tool_call_text(episode.assistant_text)
        ctx = _Context(episode, assistant, printed_call, known=known, previous=previous)
        out: List[MemoryCandidate] = []
        pref = self._preference(episode)
        if pref is not None:
            out.append(pref)
        # A fact the user states outranks a learning inferred from tool errors in the same turn.
        wiki = (self._decision(episode, previous) or (self._environment(ctx) if ctx.user_facts else None)
                or self._learning(ctx) or self._environment(ctx))
        if wiki is not None:
            out.append(wiki)
        if not out:
            out.append(self._noise(episode, printed_call))
        return out

    # -- candidates --------------------------------------------------------------------------------

    def _candidate(self, episode: Episode, mtype: MemoryType, claims: List[str], evidence: List[str], *,
                   title: str, rationale: str, context: str = "", event_ids: Optional[List[str]] = None,
                   confidence: Optional[float] = None, subject: str = "",
                   extra_tags: Sequence[str] = ()) -> MemoryCandidate:
        claims, evidence = _limit(claims, evidence)
        tags = _tags(claims + [subject])
        for t in [*extra_tags, *subject_tags(subject)]:
            if t not in tags:
                tags.append(t)
        return MemoryCandidate(
            event_ids=list(event_ids or episode.event_ids), classification=classification(mtype, confidence=confidence),
            normalized_claims=claims, retrieval_query=" ".join([title, *claims]),
            id=f"{episode.id}:{mtype.value}", session_id=episode.session_id, title=shorten(title),
            tags=tags[:10], claim_evidence=evidence, context=context, rationale=rationale, subject=subject)

    def _noise(self, episode: Episode, printed_call: bool = False) -> MemoryCandidate:
        why = "no rule matched" + ("; assistant printed a tool call as text (dropped)" if printed_call else "")
        return self._candidate(episode, MemoryType.NOISE, [], [], title="", rationale=why)

    def _preference(self, episode: Episode) -> Optional[MemoryCandidate]:
        claims = []
        for s in sentences(episode.user_text):
            if not _declarative(s) or len(s.split()) < 4:
                continue
            body = strip_lead(s)
            if _PREF_ANY.search(s) or (_PREF_START.match(body) and not _TASK_SCOPED.search(body)):
                claims.append(_clean(_PREF_LEAD.sub("", body)))
            elif _PROFILE.search(body) and not _HEDGE.search(body):
                claims.append(_clean(body))
        if not claims:
            return None
        return self._candidate(episode, MemoryType.USER_PREFERENCE, claims, [OBSERVED] * len(claims),
                               title=f"User preference: {claims[0]}",
                               rationale="user states a standing preference or a fact about themselves")

    def _decision(self, episode: Episode, previous: Optional[Episode]) -> Optional[MemoryCandidate]:
        user = episode.user_text.strip()
        if not user:
            return None
        first_clause = re.split(r"[.;!?]", user, maxsplit=1)[0]
        if previous is not None and _CONFIRM.match(user) and not _NEGATION.search(first_clause):
            prev_assistant = strip_tool_call_text(previous.assistant_text)[0]
            proposals = [s for s in sentences(prev_assistant)
                         if _declarative(s) and (_PROPOSAL.match(s) or _PROPOSAL_ANY.search(s))]
            if proposals:
                claims = [_clean(_PROPOSAL.sub("", s)) for s in proposals]
                context = (f'Proposed by the assistant in reply to "{shorten(previous.user_text, 160, title=False)}"; '
                           f'confirmed by the user: "{shorten(user, 160, title=False)}".')
                title = re.split(r"[;:]\s", claims[0], maxsplit=1)[0]
                return self._candidate(episode, MemoryType.DECISION, claims, [OBSERVED] * len(claims),
                                       title=title, context=context, event_ids=previous.event_ids + episode.event_ids,
                                       rationale="assistant proposal confirmed by the user in the next turn")
        direct = [s for s in sentences(user) if _declarative(s) and _USER_DECISION.match(s)]
        if direct:
            claims = [_clean(_USER_DECISION.sub("", s)) for s in direct]
            title = re.split(r"[;:]\s", claims[0], maxsplit=1)[0]
            return self._candidate(episode, MemoryType.DECISION, claims, [OBSERVED] * len(claims), title=title,
                                   context=f'Stated by the user: "{shorten(user, 160, title=False)}".',
                                   rationale="user states a decision")
        return None

    def _learning(self, ctx: "_Context") -> Optional[MemoryCandidate]:
        episode = ctx.episode
        calls = episode.tool_calls
        for i, err in enumerate(calls):
            if not _failed(err) or err.content.get("tool") in MEMORY_TOOLS:
                continue
            fix = next((c for c in calls[i + 1:] if _ok(c) and (_program(c) == _program(err)
                        or c.content.get("tool") == err.content.get("tool") and not _command(c))), None)
            # Edits the agent makes to PAN's own wiki are not fixes of the system (daemon adopts them).
            changes = [f for f in episode.file_changes
                       if f.id > err.id and "/pan/wiki/" not in str(f.content.get("path"))]
            if fix is None and not changes:
                continue
            if not err.content.get("args") and not changes:
                # A call without arguments retried correctly (layer-B run: terminal({}) → "expected
                # string, got NoneType") is a model hiccup, not a learning about the system.
                continue
            what = f"`{_command(err)}`" if _command(err) else f"Tool `{err.content.get('tool')}`"
            error = _first_line(err.content.get("error_message") or err.content.get("result_excerpt") or "") \
                or "an error"
            claims, evidence = [f"{what} failed: {error}"], [OBSERVED]
            if fix is not None:
                claims.append(f"Fixed by `{_command(fix)}`." if _command(fix)
                              else f"Fixed by calling `{fix.content.get('tool')}` again.")
            else:
                paths = ", ".join(f"`{c.content.get('path')}`" for c in changes[:3])
                claims.append(f"Fixed by changing {paths}.")
            evidence.append(OBSERVED)
            for s in ctx.assistant_sentences:
                if (_declarative(s) and _FIX_EXPLAIN.search(s) and not _HEDGE.search(s) and not ctx.echo(s)
                        and _inferred_ok(_unack(s), first_person_ok=True)):
                    claims.append(_clean(s))
                    evidence.append(INFERRED)
            if fix is not None and not _command(fix) and INFERRED not in evidence:
                # "Fixed by calling `patch` again" alone says nothing about the system — the agent
                # corrected its own call (e.g. a relative path). Needs an explanation to be a learning.
                continue
            _, topic_tags = command_topic(_command(err))
            title = f"{_program(err) or 'tool'} error: {error}"
            return self._candidate(episode, MemoryType.LEARNING, claims, evidence, title=title,
                                   context=_context(episode), rationale="tool error followed by a fix",
                                   extra_tags=topic_tags)
        return None

    def _environment(self, ctx: "_Context") -> Optional[MemoryCandidate]:
        episode = ctx.episode
        claims: List[str] = []
        evidence: List[str] = []
        subjects: List[str] = []
        topic_tags: List[str] = []
        mtype = MemoryType.ENVIRONMENT
        user_facts = ctx.user_facts
        for fact in user_facts:
            claims.append(fact)
            evidence.append(OBSERVED)
            subjects.append(subject_of(fact))
        tool_claims: List[tuple[str, str]] = []
        for call in episode.tool_calls:
            cmd = _command(call)
            if _ok(call) and cmd and _INFRA_CMD.search(cmd):
                line = _first_line(call.content.get("result_excerpt") or "")
                subject, tags = command_topic(cmd)
                label = f"{subject} (`{cmd}`)" if subject in HARDWARE_TOPICS else f"`{cmd}`"
                tool_claims.append((f"{label} reports: {line}" if line else f"`{cmd}` succeeded.", subject))
                topic_tags.extend(t for t in tags if t not in topic_tags)
        file_claims, file_subject = _file_claims(episode)
        tool_claims += [(c, file_subject) for c in file_claims]
        grounded = _grounded_sentences(ctx)
        read_path = next((_display_path(_call_path(c), episode.project) for c in episode.tool_calls
                          if _ok(c) and c.content.get("tool") in READ_TOOLS and _call_path(c)), "")
        # Grounded explanations read best: before the raw command/file claims.
        tool_claims[:0] = [(c, file_subject or read_path or next((x for _, x in tool_claims if x), ""))
                           for c in grounded]
        config_changes = [f for f in episode.file_changes if _CONFIG_PATH.search(str(f.content.get("path") or ""))]
        inferred: List[str] = []
        for s in ctx.assistant_sentences:
            if (_declarative(s) and _INFRA_NOUN.search(s) and _STATE_VERB.search(s) and not _HEDGE.search(s)
                    and not ctx.echo(s) and _inferred_ok(_unack(s)) and _clean(s) not in grounded
                    # the user stated the fact: an acknowledging restatement, or one with a number
                    # neither the user, the tools nor memory mentioned, adds nothing (or invents)
                    and (not user_facts or (_unack(s) == s.strip() and ctx.values_sourced(s)))):
                inferred.append(_clean(s))
        # Order: the user's words, then the explanation (reads best), then tool output.
        claims += inferred
        evidence += [INFERRED] * len(inferred)
        subjects += [subject_of(s) for s in inferred]
        for text, subject in tool_claims:
            claims.append(text)
            evidence.append(OBSERVED)
            subjects.append(subject)
        for change in config_changes:
            verb = {"write": "written", "replace": "edited", "update": "edited", "add": "created",
                    "delete": "deleted", "move": "moved"}.get(str(change.content.get("op")), "changed")
            claims.append(f"`{change.content.get('path')}` was {verb} by the agent.")
            evidence.append(OBSERVED)
            subjects.append("")
        if not claims:
            return None
        if config_changes and not any(_INFRA_CMD.search(_command(c)) for c in episode.tool_calls) and not user_facts:
            mtype = MemoryType.CONFIGURATION
        # Subject: the user's fact first, then the explanation's subject, then a known command topic.
        listed = list(dict.fromkeys(x for x in subjects[:len(user_facts)] if x))
        if len(listed) >= 2:  # an enumeration ("auth …, billing …, search …"): all names in the title
            subjects[0] = display_subject(", ".join(listed[:-1]) + " and " + listed[-1])
        subject = next((s for s in subjects[:len(user_facts)] if s), "") \
            or next((s for s in subjects[len(user_facts):len(user_facts) + len(inferred)] if s), "") \
            or next((s for _, s in tool_claims if s), "")
        confidence = 0.8 if OBSERVED in evidence else 0.6
        if mtype is MemoryType.CONFIGURATION:
            rationale = "config/infra file changed"
        elif user_facts:
            rationale = "user states an environment fact"
        else:
            rationale = "infra command succeeded or assistant explains environment state"
        title = subject or _title_from_claim(claims[0])
        # The user's own fact sentence is the claim; repeating it as context would keep the old
        # value on the page after an update (benchmark runs: the prefetch snippet showed it).
        context = "" if user_facts else _context(episode)
        return self._candidate(episode, mtype, claims, evidence, title=title, context=context,
                               confidence=confidence, rationale=rationale, subject=subject, extra_tags=topic_tags)


# -- facts observed in tool output (M6) ------------------------------------------------------------------------

READ_TOOLS = frozenset({"read_file", "view_file", "cat", "open_file"})
MAX_FILE_CLAIMS = 6
MAX_GROUNDED = 4
_KV_LINE = re.compile(r"^\s*[\"']?([A-Za-z_][\w.-]*)[\"']?\s*[=:]\s*(.+?)\s*,?$")
_NUMBERED = re.compile(r"^\s*\d+\|")
_DOTFILE = re.compile(r"(?:^|/)\.[\w-]+$")
_VALUE_TOKEN = re.compile(r"[A-Za-z0-9][\w.:/-]*\d[\w:/-]*|\d[\w.:/-]*")


def _call_path(call: AgentEvent) -> str:
    args = call.content.get("args")
    return str(args.get("path") or args.get("file_path") or "") if isinstance(args, dict) else ""


def _display_path(path: str, project: Optional[str]) -> str:
    if project and path.startswith(project.rstrip("/") + "/"):
        path = path[len(project.rstrip("/")) + 1:]
    if path.startswith("/"):
        m = re.search(r"/workspace/(.+)$", path)
        path = m.group(1) if m else "/".join(path.split("/")[-2:])
    return path


def _file_lines(call: AgentEvent) -> List[str]:
    text = _tool_output(call.content.get("result_excerpt") or "")
    return [_NUMBERED.sub("", ln).rstrip() for ln in text.splitlines()]


def _file_claims(episode: Episode) -> tuple[List[str], str]:
    """Settings read from config-like or tiny files: "`config/limits.toml`: rate_limit_per_min = 250",
    "`.python-version` contains `3.12.4`". Returns (claims, subject = the first such file path)."""
    claims: List[str] = []
    subject = ""
    for call in episode.tool_calls:
        if not _ok(call) or call.content.get("tool") not in READ_TOOLS:
            continue
        path = _call_path(call)
        if not path:
            continue
        lines = [ln for ln in _file_lines(call) if ln.strip()]
        shown = _display_path(path, episode.project)
        config_like = bool(_CONFIG_PATH.search(path) or _DOTFILE.search(path) or path.endswith(".json"))
        if len(lines) == 1 and len(lines[0]) <= 120 and (config_like or len(lines[0].split()) <= 3):
            claims.append(f"`{shown}` contains `{lines[0].strip()}`.")
        elif config_like:
            for ln in lines:
                if ln.lstrip().startswith(("#", "//", ";", "[", "{", "}")):
                    continue
                m = _KV_LINE.match(ln)
                if m and len(m.group(2)) <= 120:
                    claims.append(f"`{shown}`: {m.group(1)} = {m.group(2).strip().rstrip(',')}")
                if len(claims) >= MAX_FILE_CLAIMS:
                    break
        else:
            continue
        subject = subject or shown
    return claims[:MAX_FILE_CLAIMS], subject


def _grounded_sentences(ctx: "_Context") -> List[str]:
    """Assistant sentences whose every value token (ids, versions, numbers, times) occurs in the
    episode's successful non-memory tool output — facts the agent *read*, e.g. "It deployed build
    r4812 (commit 9c1e2fa) to prod." after reading a log. Meta-talk, hedges, questions and echoes
    of recalled memory are excluded."""
    outputs = "\n".join(_tool_output(c.content.get("result_excerpt") or "") for c in ctx.episode.tool_calls
                        if _ok(c) and c.content.get("tool") not in MEMORY_TOOLS).lower()
    if not outputs.strip():
        return []
    out: List[str] = []
    for s in ctx.assistant_sentences:
        body = _unack(s)
        if not _declarative(s) or _HEDGE.search(body) or not _inferred_ok(body, first_person_ok=True) or ctx.echo(s):
            continue
        toks = [t.strip(".:,;-/") for t in _VALUE_TOKEN.findall(re.sub(r"[`*]", "", body))]
        toks = [t for t in toks if len(t) >= 2 and not re.fullmatch(r"\d", t)]
        if toks and all(t.lower() in outputs for t in toks):
            out.append(_clean(body))
        if len(out) >= MAX_GROUNDED:
            break
    return out


# Command topics whose claim text names the topic ("GPU (`nvidia-smi …`) reports: NVIDIA GB10"), so a
# question in plain words ("which GPU does this machine have?") finds the page.
HARDWARE_TOPICS = frozenset({"GPU", "CPU", "Memory (RAM)", "Disks and filesystems", "Operating system",
                             "Listening ports"})


class _Context:
    """Per-episode text views shared by the rules: the assistant text without printed tool calls,
    and the "echo" check (restating recalled wiki text or the user's own words is not a new fact)."""

    def __init__(self, episode: Episode, assistant: str, printed_call: bool, *, known: str = "",
                 previous: Optional[Episode] = None) -> None:
        self.episode = episode
        self.assistant = assistant
        self.printed_call = printed_call
        self.assistant_sentences = sentences(assistant)
        # Recalled wiki text plus L1 (in the prompt): answering from memory is not observing.
        self._recall = token_set(episode.recalled_text + "\n" + known)
        self.known = known
        self._sources: Optional[str] = None
        # Only the user's *fact* sentences: they become observed claims themselves, so an assistant
        # restatement adds nothing ("Always use port 8000" is a preference, not such a claim).
        self.user_facts = _user_fact_sentences(episode.user_text) or _corrected_facts(episode.user_text, previous)
        self._user = token_set(" ".join(self.user_facts))

    def values_sourced(self, sentence: str) -> bool:
        """Every number in an assistant sentence occurs in the user's text, tool output or recalled
        memory (a restatement must not introduce a value)."""
        if self._sources is None:
            self._sources = "\n".join([self.episode.user_text, self.episode.recalled_text, self.known]
                                       + [_tool_output(c.content.get("result_excerpt") or "")
                                          for c in self.episode.tool_calls]).lower()
        nums = re.findall(r"(?<![\w.])\d[\d.,:]*\d|(?<![\w.])\d(?![\w.])", sentence)
        return all(n.lower() in self._sources for n in nums)

    def echo(self, sentence: str) -> bool:
        body = _unack(sentence)
        return _recalled(body, self._recall) or _recalled(body, self._user)


def _title_from_claim(claim: str) -> str:
    """Fallback title: the claim without a leading command line or acknowledgement."""
    text = _unack(claim)
    m = re.match(r"^`([^`]+)`\s*(?:reports:?|succeeded\.?|was \w+ by the agent\.?)?\s*(.*)$", text)
    if m:
        words = m.group(1).split()
        program = words[0] if words else "command"
        rest = m.group(2).strip()
        return f"{program}: {rest}" if rest else f"{program} output"
    return display_subject(text) if text else claim


def _context(episode: Episode) -> str:
    return f'User asked: "{shorten(episode.user_text, 160, title=False)}".' if episode.user_text.strip() else ""
