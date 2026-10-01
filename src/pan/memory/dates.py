"""Dates for claims (M9 G3): when was a fact stated, and which day do "yesterday" / "last Saturday" mean.

The gate keeps relative dates as the user wrote them; code resolves them against the **session date**
and appends it, so temporal questions ("how many days since …") can be answered from memory:

    "The user ran a 10k race yesterday."  →  "The user ran a 10k race yesterday (2024-03-10). (stated 2024-03-11)"

Which date (``claim_date``):

1. the **event timestamp** PAN recorded for the turn (primary);
2. a date the **user states** in the message ("today is 11 March 2024", "Today's date: 2024-03-11",
   "heute ist Montag, der 11.03.2024", "[Today is Mon 2024-03-11]"; ISO or written dates, EN + DE) wins
   over the timestamp for that turn;
3. later turns of the same session keep the offset between the stated date and the event time
   (the user said which day it is; the clock moved on from there).

``bare()`` removes the ``(stated …)`` suffix before claims are compared (presence, contradiction, L1
reconcile): two statements of the same fact on different days are the same fact.
"""

from __future__ import annotations

import datetime as _dt
import re
from typing import Optional

_MONTHS = {"january": 1, "jan": 1, "januar": 1, "jänner": 1, "february": 2, "feb": 2, "februar": 2, "march": 3,
           "mar": 3, "märz": 3, "maerz": 3, "april": 4, "apr": 4, "may": 5, "mai": 5, "june": 6, "jun": 6, "juni": 6,
           "july": 7, "jul": 7, "juli": 7, "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9,
           "october": 10, "oct": 10, "oktober": 10, "okt": 10, "november": 11, "nov": 11, "december": 12,
           "dec": 12, "dezember": 12, "dez": 12}
_MON = "|".join(sorted(_MONTHS, key=len, reverse=True))
_WDAY = (r"(?:mon|tue|tues|wed|thu|thur|thurs|fri|sat|sun|monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
         r"mo|di|mi|do|fr|sa|so|montag|dienstag|mittwoch|donnerstag|freitag|samstag|sonntag)\.?,?")
_DATE = (rf"(?P<iso>\d{{4}}-\d{{2}}-\d{{2}}|\d{{4}}/\d{{2}}/\d{{2}})|"
         rf"(?P<dmy>\d{{1,2}})\.\s?(?P<dmm>\d{{1,2}})\.\s?(?P<dmyy>\d{{4}})|"
         rf"(?P<d1>\d{{1,2}})(?:st|nd|rd|th|\.)?\s+(?:of\s+)?(?P<m1>{_MON})\.?,?\s+(?P<y1>\d{{4}})|"
         rf"(?P<m2>{_MON})\.?\s+(?P<d2>\d{{1,2}})(?:st|nd|rd|th)?,?\s+(?P<y2>\d{{4}})")
# A statement of today's date by the user (generic; the bracket form some tools prepend is one case).
_STATED_TODAY = re.compile(
    rf"(?P<all>\[?\s*(?:today is|today's date is|today's date:|the date today is|it is|it's|date:|"
    rf"heute ist(?: der)?|wir haben heute(?: den)?|heutiges datum:|datum:)\s+(?:{_WDAY}\s+)?(?:der\s+|the\s+)?"
    rf"(?:{_DATE})\s*\]?)", re.I)
_STATED = re.compile(r"\s*\(stated \d{4}-\d{2}-\d{2}\)\s*$")
_WEEKDAYS = {"monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6,
             "montag": 0, "dienstag": 1, "mittwoch": 2, "donnerstag": 3, "freitag": 4, "samstag": 5, "sonntag": 6}
_NUM = {"one": 1, "a": 1, "an": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
        "nine": 9, "ten": 10, "einem": 1, "einer": 1, "zwei": 2, "drei": 3, "vier": 4, "fünf": 5, "fuenf": 5,
        "sechs": 6, "sieben": 7, "acht": 8, "neun": 9, "zehn": 10}
_WD = "|".join(_WEEKDAYS)
_N = r"\d{1,3}|" + "|".join(_NUM)
_REL = re.compile(
    rf"\b(?P<today>today|this morning|this afternoon|this evening|tonight|heute(?: morgen| abend)?)\b|"
    rf"\b(?P<dby>the day before yesterday|vorgestern)\b|"
    rf"\b(?P<yday>yesterday|gestern)\b|"
    rf"\b(?P<tmrw>tomorrow)\b|"
    rf"\b(?:last|this past|letzten|letzte|vergangenen|am letzten|am vergangenen)\s+(?P<wd>{_WD})\b|"
    rf"\b(?P<n>{_N})\s+(?P<unit>days?|weeks?)\s+ago\b|"
    rf"\bvor\s+(?P<n2>{_N})\s+(?P<unit2>tag(?:en)?|woche(?:n)?)\b|"
    rf"\b(?P<lweek>last week|letzte woche|vergangene woche)\b|"
    rf"\b(?P<lmonth>last month|letzten monat|vergangenen monat)\b|"
    rf"\b(?P<tmonth>this month|diesen monat|in diesem monat)\b", re.I)


def stated_date(user_text: str) -> Optional[_dt.date]:
    """The date the user says today is, if the message states one."""
    m = _STATED_TODAY.search(user_text or "")
    if not m:
        return None
    g = m.groupdict()
    try:
        if g["iso"]:
            return _dt.date.fromisoformat(g["iso"].replace("/", "-"))
        if g["dmy"]:
            return _dt.date(int(g["dmyy"]), int(g["dmm"]), int(g["dmy"]))
        if g["d1"]:
            return _dt.date(int(g["y1"]), _MONTHS[g["m1"].lower().rstrip(".")], int(g["d1"]))
        if g["m2"]:
            return _dt.date(int(g["y2"]), _MONTHS[g["m2"].lower().rstrip(".")], int(g["d2"]))
    except (ValueError, KeyError):
        return None
    return None


def event_date(ts: str) -> Optional[_dt.date]:
    """Date of an event timestamp (ISO 8601)."""
    try:
        return _dt.date.fromisoformat(str(ts or "")[:10])
    except ValueError:
        return None


def claim_date(user_text: str, event_ts: str = "", *, offset_days: Optional[int] = None) -> Optional[_dt.date]:
    """Session date for a turn's claims (see module docstring): user-stated date > event date shifted by
    the session's stated offset > event date."""
    stated = stated_date(user_text)
    if stated is not None:
        return stated
    ev = event_date(event_ts)
    if ev is not None and offset_days:
        return ev + _dt.timedelta(days=offset_days)
    return ev


def stated_offset(user_text: str, event_ts: str) -> Optional[int]:
    """Days between the user-stated date and the event date (for later turns of the session)."""
    stated, ev = stated_date(user_text), event_date(event_ts)
    return (stated - ev).days if stated is not None and ev is not None else None


def session_date(user_text: str, turn_ts: str = "") -> Optional[_dt.date]:
    """Backwards-compatible alias of :func:`claim_date` without a session offset."""
    return claim_date(user_text, turn_ts)


def strip_today_line(text: str) -> str:
    """``text`` without the user's statement of today's date."""
    return _STATED_TODAY.sub("", text or "").strip()


def _num(s: str) -> int:
    return int(s) if s.isdigit() else _NUM.get(s.lower(), 0)


def _month(d: _dt.date) -> str:
    return f"{d.year:04d}-{d.month:02d}"


def resolve(text: str, day: _dt.date) -> str:
    """``text`` with relative date phrases followed by the date they mean (idempotent)."""
    def repl(m: re.Match) -> str:
        phrase = m.group(0)
        after = text[m.end():m.end() + 14]
        if re.match(r"\s*\((?:\d{4}-\d{2}(?:-\d{2})?|≈)", after):
            return phrase   # already resolved
        g = m.groupdict()
        if g["today"]:
            val = day.isoformat()
        elif g["dby"]:
            val = (day - _dt.timedelta(days=2)).isoformat()
        elif g["yday"]:
            val = (day - _dt.timedelta(days=1)).isoformat()
        elif g["tmrw"]:
            val = (day + _dt.timedelta(days=1)).isoformat()
        elif g["wd"]:
            back = (day.weekday() - _WEEKDAYS[g["wd"].lower()]) % 7 or 7
            val = (day - _dt.timedelta(days=back)).isoformat()
        elif g["n"] or g["n2"]:
            n = _num(g["n"] or g["n2"])
            unit = (g["unit"] or g["unit2"] or "").lower()
            if not n:
                return phrase
            val = (day - _dt.timedelta(days=n * (7 if unit.startswith(("week", "woche")) else 1))).isoformat()
        elif g["lweek"]:
            val = "≈ " + (day - _dt.timedelta(days=7)).isoformat()
        elif g["lmonth"]:
            first = day.replace(day=1) - _dt.timedelta(days=1)
            val = _month(first)
        elif g["tmonth"]:
            val = _month(day)
        else:
            return phrase
        return f"{phrase} ({val})"
    return _REL.sub(repl, text)


def annotate(claim: str, day: Optional[_dt.date], *, stated: bool = True) -> str:
    """Resolved relative dates plus the ``(stated YYYY-MM-DD)`` suffix."""
    if day is None or not claim:
        return claim
    out = resolve(bare(claim), day).rstrip()
    return f"{out} (stated {day.isoformat()})" if stated else out


def bare(claim: str) -> str:
    """``claim`` without its ``(stated …)`` suffix (for comparing facts)."""
    return _STATED.sub("", claim or "")
