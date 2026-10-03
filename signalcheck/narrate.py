"""Validated LLM narration of the findings, with template fallback (section 9.1).

The deterministic engine has already decided everything. This module only turns
a :class:`~signalcheck.models.Verdict` into one short paragraph of prose:

1. :func:`build_payload` reduces the verdict to a small, deterministic JSON object
   (label, direction, confidence, rule, reason, evidence summaries and numbers,
   change-my-mind conditions and numbers, caveats, window dates, source label).
   These are the only facts the narration may use.
2. If an LLM provider is configured (``LLM_PROVIDER``/``LLM_API_KEY``, optional
   ``LLM_MODEL``/``LLM_BASE_URL``), :data:`SYSTEM_PROMPT` plus that JSON is sent to
   it at temperature 0. Providers implement the small :class:`Narrator` protocol
   and are looked up in :data:`PROVIDERS`; the one concrete implementation,
   :class:`OpenAICompatibleNarrator`, speaks the OpenAI Chat Completions format
   over the shared :mod:`signalcheck.http` session (so it also works with GitHub
   Models, OpenRouter and other compatible gateways). Raw replies are cached.
3. :func:`validate_narration` checks the reply against the JSON: every number,
   number word, date, month and year must appear in it (after normalisation), the
   verdict label must appear verbatim and no other label may, and the word cap
   holds.
4. Anything going wrong (no provider or key, HTTP error, timeout, malformed reply,
   failed validation) yields :func:`template_narration` instead, which uses the
   same JSON and passes the same validator. Only the failure *category* is logged.

**Number-word policy.** Words zero-twenty are numbers and must match a value in
the JSON, with one exception: "one" is always allowed because it is mostly a
determiner or pronoun ("one source", "a one-off", "no one"), and a claim of "one"
of something cannot inflate a finding. Counts that are plainly visible in the JSON
also count as JSON numbers: how many checks there are, how many ran and were
skipped, how many hold each stance, and how many conditions and caveats there are,
so "three of the five checks support a trend" validates when it is true.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from datetime import date
from decimal import ROUND_HALF_EVEN, ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Any, Literal, Protocol

import numpy as np
import pandas as pd
import requests

from signalcheck import http
from signalcheck.cache import Cache, get_cache, make_key
from signalcheck.config import Config, get_config, get_secret
from signalcheck.engine.checks.common import NUMBER_WORDS
from signalcheck.models import Verdict

logger = logging.getLogger(__name__)

AI_LABEL = "AI-written summary of the findings above"
TEMPLATE_LABEL = "Summary (template)"

NarrationPath = Literal["llm", "template"]

TEMPERATURE = 0
SYSTEM_PROMPT = (
    "You summarise the output of a deterministic statistics engine that judged whether "
    "interest in a topic shows a real trend. The user message is a JSON object holding "
    "every fact you may use.\n"
    "Rules:\n"
    "- Use only facts in the JSON. Do not add numbers, dates, causes, forecasts or "
    "outside knowledge.\n"
    "- Copy every number and date exactly as it appears in the JSON. You may drop "
    "trailing decimals, but never compute new numbers (no differences, ratios, sums or "
    "averages).\n"
    "- State the verdict label exactly as given in verdict.label, in capitals (for "
    "example TREND or NO_CHANGE). Do not write any other verdict label in capitals.\n"
    "- Explain why the verdict was reached using the reason and the evidence, mention "
    "the most important caveat if there is one, and say what would change the verdict.\n"
    "- Write at most {max_words} words of plain prose in a single paragraph: no lists, "
    "headings, markdown or quotation of the JSON keys."
)
USER_PROMPT_PREFIX = "Findings JSON:\n"

LABEL_OPENINGS: dict[str, str] = {
    "TREND": "{source} shows a sustained {direction}ward trend",
    "FLUKE": "{source} shows a short-lived {blip} rather than a lasting shift",
    "SEASONAL": "{source} moved in line with its usual annual pattern",
    "NO_CHANGE": "{source} shows no meaningful change in the recent window",
    "INCONCLUSIVE": "The data from {source} does not support a firm call",
}
FLUKE_BLIPS: dict[str | None, str] = {"up": "spike", "down": "dip", None: "blip"}
LEAD_STANCES: dict[str, str] = {
    "TREND": "supports_trend",
    "FLUKE": "supports_fluke",
    "SEASONAL": "supports_seasonal",
    "NO_CHANGE": "supports_no_change",
}

# --- payload -----------------------------------------------------------------


def _json_safe(value: Any) -> Any:
    """``value`` as plain JSON types (numpy scalars unwrapped, NaN/inf -> ``None``)."""
    if value is None or isinstance(value, bool | str):
        return value
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, int | np.integer):
        return int(value)
    if isinstance(value, float | np.floating):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, pd.Timestamp | date):
        return value.isoformat()[:10]
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list | tuple | set | frozenset | np.ndarray):
        items = [_json_safe(v) for v in value]
        return sorted(items, key=str) if isinstance(value, set | frozenset) else items
    return str(value)


def _period_word(freq: str | None) -> str | None:
    from signalcheck.engine.preprocess import PERIOD_WORDS

    return PERIOD_WORDS.get(freq or "")


def build_payload(verdict: Verdict, series_meta: Mapping[str, Any]) -> dict[str, Any]:
    """The only facts a narration may use, as deterministic JSON-serialisable data.

    ``series_meta`` supplies ``source`` (id), ``source_label`` (display name;
    defaults to the id's label) and optionally ``freq`` (D/W/M, rendered as the
    period word). Everything else comes from ``verdict``.
    """
    from signalcheck.adapters.base import SOURCE_LABELS

    source = str(series_meta.get("source") or "")
    label = series_meta.get("source_label") or SOURCE_LABELS.get(source, source) or "This source"
    window = _json_safe(dict(verdict.window))
    period = _period_word(series_meta.get("freq"))
    if period is not None and window:
        window["period"] = period
    return {
        "source": str(label),
        "verdict": {
            "label": verdict.label,
            "direction": verdict.direction,
            "confidence": verdict.confidence,
            "rule_fired": verdict.rule_fired,
            "reason": verdict.reason,
        },
        "evidence": [
            {
                "check": ev.check,
                "stance": ev.stance,
                "summary": ev.summary,
                "numbers": _json_safe(ev.numbers),
            }
            for ev in verdict.evidence
        ],
        "change_my_mind": [
            {"text": str(cond), "numbers": _json_safe(getattr(cond, "numbers", {}))}
            for cond in verdict.change_my_mind
        ],
        "caveats": [str(c) for c in verdict.caveats],
        "window": window,
    }


def payload_json(payload: Mapping[str, Any]) -> str:
    """Canonical (sorted, compact) JSON text of ``payload``; the LLM's user message."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def system_prompt(cfg: Config | None = None) -> str:
    """:data:`SYSTEM_PROMPT` with the configured word cap filled in."""
    cfg = cfg if cfg is not None else get_config()
    return SYSTEM_PROMPT.format(max_words=int(cfg["narration"]["max_words"]))


# --- validator ---------------------------------------------------------------

MONTHS: dict[str, int] = {
    name: i
    for i, names in enumerate(
        (
            ("january", "jan"),
            ("february", "feb"),
            ("march", "mar"),
            ("april", "apr"),
            ("may",),
            ("june", "jun"),
            ("july", "jul"),
            ("august", "aug"),
            ("september", "sep", "sept"),
            ("october", "oct"),
            ("november", "nov"),
            ("december", "dec"),
        ),
        start=1,
    )
    for name in names
}
# "May" and "March" are ordinary words too; they only count as months inside a date.
AMBIGUOUS_MONTHS: frozenset[str] = frozenset({"may", "march", "mar"})

_MONTH = r"(?P<month>" + "|".join(sorted(MONTHS, key=len, reverse=True)) + r")\.?"
_ORD = r"(?:st|nd|rd|th)?"
_DASH = r"(?:\s*[-\u2013\u2014]\s*|\s+to\s+)"
_YEAR = r"(?P<year>(?:19|20)\d{2})(?!\d)"
ISO_DATE_RE = re.compile(r"(?<!\d)(\d{4})-(\d{2})-(\d{2})(?!\d)")
DAY_MONTH_RE = re.compile(
    r"(?<![\w.])(?P<day>\d{1,2})" + _ORD + r"(?:" + _DASH + r"(?P<day2>\d{1,2})" + _ORD + r")?"
    r"\s+(?:of\s+)?" + _MONTH + r"(?:,?\s+" + _YEAR + r")?\b",
    re.IGNORECASE,
)
MONTH_DAY_RE = re.compile(
    r"\b" + _MONTH + r"\s+(?P<day>\d{1,2})" + _ORD + r"(?!\d)"
    r"(?:" + _DASH + r"(?P<day2>\d{1,2})" + _ORD + r"(?!\d))?(?:,?\s+" + _YEAR + r")?\b",
    re.IGNORECASE,
)
MONTH_YEAR_RE = re.compile(r"\b" + _MONTH + r",?\s+" + _YEAR, re.IGNORECASE)
# Capitalised only: lower-case "may"/"march" are ordinary words.
BARE_MONTH_RE = re.compile(
    r"\b(" + "|".join(sorted((m.capitalize() for m in MONTHS), key=len, reverse=True)) + r")\b"
)
LABEL_RE = re.compile(r"\b(TREND|FLUKE|SEASONAL|NO[_ ]CHANGE|INCONCLUSIVE)\b")
NUMBER_TOKEN_RE = re.compile(
    r"(?:(?P<sign>[-\u2212])|(?<![\w.,]))"
    r"(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"(?![\d,]\d)(?P<pct>\s*%|\s+per\s?cent\b)?",
    re.IGNORECASE,
)
_SIGN_CONTEXT_RE = re.compile(r"[\w.,)]$")
_MULTIPLIER_RE = re.compile(
    r"(?<![\w])[x\u00d7](?=\d)|(?<=\d)[x\u00d7](?!\w)|\u00d7", re.IGNORECASE
)
_ORDINAL_RE = re.compile(r"(?<=\d)(?:st|nd|rd|th)\b", re.IGNORECASE)
NUMBER_WORD_TOKEN_RE = re.compile(r"\b(" + "|".join(NUMBER_WORDS) + r")\b", re.IGNORECASE)
EXEMPT_NUMBER_WORDS: frozenset[str] = frozenset({"one"})
PERCENT_KEY_SUFFIXES: tuple[str, ...] = ("_pct", "_percent")


@dataclass(frozen=True)
class Issue:
    """One validator finding: a ``code`` and the offending ``token`` from the text."""

    code: Literal[
        "empty", "too_long", "label_missing", "wrong_label", "number", "number_word", "date"
    ]
    token: str = ""

    def __str__(self) -> str:
        return f"{self.code}: {self.token}" if self.token else self.code


@dataclass(frozen=True)
class ValidationResult:
    """Outcome of :func:`validate_narration`; ``ok`` iff there are no issues."""

    issues: tuple[Issue, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.issues

    @property
    def reasons(self) -> list[str]:
        """Human-readable issues (for tests and debugging; never logged)."""
        return [str(issue) for issue in self.issues]

    @property
    def codes(self) -> tuple[str, ...]:
        """Distinct issue codes, sorted (safe to log: no narration content)."""
        return tuple(sorted({issue.code for issue in self.issues}))


@dataclass(frozen=True)
class NumberToken:
    """A number found in text: absolute value, explicit minus sign, percent suffix."""

    text: str
    value: Decimal
    negative: bool
    percent: bool


@dataclass(frozen=True)
class AllowedNumber:
    """A number the narration may use; ``percent`` if the JSON shows it as a percent."""

    value: Decimal
    percent: bool


@dataclass(frozen=True)
class DateMention:
    """A date expression in text; missing parts are ``None`` (e.g. "Sep 2026")."""

    text: str
    year: int | None
    month: int
    days: tuple[int, ...]


@dataclass
class Facts:
    """Everything in the payload a narration may refer to."""

    numbers: list[AllowedNumber] = field(default_factory=list)
    dates: set[date] = field(default_factory=set)

    @property
    def months(self) -> set[int]:
        return {d.month for d in self.dates}

    @property
    def years(self) -> set[int]:
        return {d.year for d in self.dates}


def _decimal(value: float | int | str) -> Decimal | None:
    try:
        number = Decimal(str(value))
    except InvalidOperation:
        return None
    return number if number.is_finite() else None


def _strip_dates(text: str) -> tuple[str, list[DateMention]]:
    """Remove date expressions from ``text``; return the rest and the dates found."""
    mentions: list[DateMention] = []

    def iso(match: re.Match[str]) -> str:
        y, m, d = (int(g) for g in match.groups())
        mentions.append(DateMention(match.group(0), y, m, (d,)))
        return " "

    def dated(match: re.Match[str]) -> str:
        groups = match.groupdict()
        days = tuple(int(groups[k]) for k in ("day", "day2") if groups.get(k))
        year = int(groups["year"]) if groups.get("year") else None
        mentions.append(DateMention(match.group(0), year, MONTHS[groups["month"].lower()], days))
        return " "

    text = ISO_DATE_RE.sub(iso, text)
    for pattern in (DAY_MONTH_RE, MONTH_DAY_RE, MONTH_YEAR_RE):
        text = pattern.sub(dated, text)
    return text, mentions


def _number_tokens(text: str) -> Iterator[NumberToken]:
    """Numbers in ``text`` (dates must already be stripped).

    Handles thousands separators ("1,234"), decimals, percent suffixes ("12.3%",
    "12.3 percent"), multipliers ("2x", "x2", also with the multiplication sign)
    and digit ordinals ("21st").
    A hyphen or Unicode minus counts as a minus sign only when it does not follow a word,
    digit or closing bracket, so ranges ("1-28", also with an en dash) and hyphenated words
    stay unsigned.
    """
    text = _ORDINAL_RE.sub("", _MULTIPLIER_RE.sub(" ", text))
    for match in NUMBER_TOKEN_RE.finditer(text):
        negative = match.group("sign") is not None
        if negative and _SIGN_CONTEXT_RE.search(text[: match.start()]):
            negative = False
        number = _decimal(match.group("num").replace(",", ""))
        if number is None:  # pragma: no cover - the regex only matches digits
            continue
        yield NumberToken(match.group(0).strip(), number, negative, match.group("pct") is not None)


def _walk(value: Any, key: str = "") -> Iterator[tuple[str, Any]]:
    if isinstance(value, Mapping):
        for k, v in value.items():
            yield from _walk(v, str(k))
    elif isinstance(value, list | tuple):
        for v in value:
            yield from _walk(v, key)
    else:
        yield key, value


def _structural_counts(payload: Mapping[str, Any]) -> list[int]:
    """Counts a reader can see in the JSON (see the number-word policy above)."""
    evidence = list(payload.get("evidence") or [])
    stances = [str(ev.get("stance")) for ev in evidence]
    counts = [
        len(evidence),
        sum(s != "skipped" for s in stances),
        len(payload.get("change_my_mind") or []),
        len(payload.get("caveats") or []),
    ]
    counts.extend(stances.count(s) for s in set(stances))
    return counts


def collect_facts(payload: Mapping[str, Any]) -> Facts:
    """Numbers and dates the narration may mention, read from ``payload``."""
    facts = Facts()
    for key, value in _walk(payload):
        if isinstance(value, bool) or value is None:
            continue
        if isinstance(value, int | float):
            number = _decimal(value)
            if number is not None:
                facts.numbers.append(AllowedNumber(number, key.endswith(PERCENT_KEY_SUFFIXES)))
        elif isinstance(value, str):
            rest, mentions = _strip_dates(value)
            for mention in mentions:
                for day in mention.days:
                    if mention.year is not None:
                        try:
                            facts.dates.add(date(mention.year, mention.month, day))
                        except ValueError:
                            continue
            facts.numbers.extend(
                AllowedNumber(-t.value if t.negative else t.value, t.percent)
                for t in _number_tokens(rest)
            )
    facts.numbers.extend(AllowedNumber(Decimal(n), False) for n in _structural_counts(payload))
    return facts


def _matches(narrated: Decimal, candidate: Decimal) -> bool:
    """``narrated`` equals ``candidate`` as shown or rounded to fewer decimals."""
    if narrated == candidate:
        return True
    exponent = narrated.as_tuple().exponent
    places = -exponent if isinstance(exponent, int) and exponent < 0 else 0
    quantum = Decimal(1).scaleb(-places)
    return any(
        candidate.quantize(quantum, rounding=mode) == narrated
        for mode in (ROUND_HALF_UP, ROUND_HALF_EVEN)
    )


def number_allowed(token: NumberToken, allowed: Iterable[AllowedNumber]) -> bool:
    """Whether ``token`` matches a JSON number (section 9.1 normalisation).

    A narrated number matches a JSON value as displayed or rounded to fewer
    decimals ("21" for 21.3, "2.10" for 2.1). Percent equivalence: "12.3%" also
    matches the fraction 0.123, and a plain "0.123" matches a value the JSON shows
    as a percent. Unsigned numbers match by magnitude ("fell 0.9%" for -0.9); an
    explicit minus sign must match a negative value.
    """
    for item in allowed:
        if token.negative and item.value >= 0:
            continue
        magnitude = abs(item.value)
        candidates = [magnitude]
        if token.percent:
            candidates.append(magnitude * 100)
        elif item.percent:
            candidates.append(magnitude / 100)
        if any(_matches(token.value, c) for c in candidates):
            return True
    return False


def _date_allowed(mention: DateMention, dates: set[date]) -> bool:
    for day in mention.days or (None,):
        if not any(
            d.month == mention.month
            and (mention.year is None or d.year == mention.year)
            and (day is None or d.day == day)
            for d in dates
        ):
            return False
    return True


def validate_narration(
    text: str, payload: Mapping[str, Any], cfg: Config | None = None
) -> ValidationResult:
    """Check ``text`` against ``payload`` (section 9.1); never raises.

    Rejects the text when it is empty or longer than ``narration.max_words``;
    lacks the verdict label in capitals or names another label in capitals; or
    contains a number, number word, date, month name or year that the JSON does
    not hold (see :func:`number_allowed` for the normalisation and the module
    docstring for the number-word policy).
    """
    cfg = cfg if cfg is not None else get_config()
    issues: list[Issue] = []
    words = len(text.split())
    if words == 0:
        return ValidationResult((Issue("empty"),))
    if words > int(cfg["narration"]["max_words"]):
        issues.append(Issue("too_long", str(words)))

    expected = str(payload["verdict"]["label"])
    found = {m.replace(" ", "_") for m in LABEL_RE.findall(text)}
    if expected not in found:
        issues.append(Issue("label_missing", expected))
    issues.extend(Issue("wrong_label", lbl) for lbl in sorted(found - {expected}))

    facts = collect_facts(payload)
    rest, mentions = _strip_dates(text)
    issues.extend(
        Issue("date", m.text.strip()) for m in mentions if not _date_allowed(m, facts.dates)
    )
    for month in BARE_MONTH_RE.findall(rest):
        if month.lower() not in AMBIGUOUS_MONTHS and MONTHS[month.lower()] not in facts.months:
            issues.append(Issue("date", month))
    for token in _number_tokens(rest):
        is_year = (
            not token.percent
            and token.value == token.value.to_integral_value()
            and 1900 <= token.value <= 2099
            and "." not in token.text
        )
        if is_year and int(token.value) in facts.years:
            continue
        if not number_allowed(token, facts.numbers):
            issues.append(Issue("date" if is_year else "number", token.text))
    for word in NUMBER_WORD_TOKEN_RE.findall(rest):
        if word.lower() in EXEMPT_NUMBER_WORDS:
            continue
        token = NumberToken(word, Decimal(NUMBER_WORDS.index(word.lower())), False, False)
        if not number_allowed(token, facts.numbers):
            issues.append(Issue("number_word", word))
    return ValidationResult(tuple(issues))


# --- template ----------------------------------------------------------------


def _sentence(text: str) -> str:
    text = text.strip().rstrip(".")
    return f"{text[:1].upper()}{text[1:]}." if text else ""


def _lower_first(text: str) -> str:
    """Lower-case the first letter unless the first word is an acronym ("MAD")."""
    first = text.split(" ", 1)[0]
    return text if len(first) > 1 and first.isupper() else text[:1].lower() + text[1:]


def template_narration(payload: Mapping[str, Any], cfg: Config | None = None) -> str:
    """Deterministic paragraph built only from ``payload``; passes the validator.

    Sentences in priority order: the verdict (label, confidence, rule), why the
    rule matched, the lead piece of agreeing evidence, the first change-my-mind
    condition and the windows compared. A sentence is added only while the text
    stays within ``narration.max_words``.
    """
    cfg = cfg if cfg is not None else get_config()
    max_words = int(cfg["narration"]["max_words"])
    v = payload["verdict"]
    label = str(v["label"])
    opening = LABEL_OPENINGS[label].format(
        source=payload["source"],
        direction=v.get("direction") or "",
        blip=FLUKE_BLIPS.get(v.get("direction"), "blip"),
    )
    pieces = [
        f"{opening}: the verdict is {label}, with {v['confidence']} confidence "
        f"(rule {v['rule_fired']}).",
        _sentence(f"Why: {v['reason']}") if v.get("reason") else "",
    ]
    lead = LEAD_STANCES.get(label)
    for ev in payload.get("evidence") or []:
        if ev.get("stance") == lead and ev.get("summary"):
            pieces.append(_sentence(str(ev["summary"])))
            break
    conditions = payload.get("change_my_mind") or []
    if conditions:
        pieces.append(_sentence(f"What would change this: {_lower_first(conditions[0]['text'])}"))
    window = payload.get("window") or {}
    if window.get("recent_start") and window.get("baseline_start"):
        pieces.append(
            f"The recent window ({window['recent_start']} to {window['recent_end']}) was "
            f"compared with a baseline from {window['baseline_start']} to "
            f"{window['baseline_end']}."
        )
    text = ""
    for piece in filter(None, pieces):
        candidate = f"{text} {piece}".strip()
        if not text or len(candidate.split()) <= max_words:
            text = candidate
    return text


# --- providers ---------------------------------------------------------------


class NarrationError(RuntimeError):
    """The LLM path failed; ``category`` is a short code that is safe to log."""

    def __init__(self, category: str) -> None:
        super().__init__(category)
        self.category = category


class Narrator(Protocol):
    """An LLM provider: turns a system prompt and user message into text.

    Implementations raise :class:`NarrationError` on any failure and never log
    prompt or response content.
    """

    @property
    def name(self) -> str:
        """Registry key of the provider (part of the cache key)."""
        ...

    @property
    def model(self) -> str:
        """Model id sent to the endpoint."""
        ...

    @property
    def base_url(self) -> str:
        """Endpoint root (part of the cache key)."""
        ...

    def complete(self, system: str, user: str) -> str:
        """The model's reply to ``system`` + ``user``."""
        ...


@dataclass(frozen=True)
class ProviderSettings:
    """Provider selection read from ``.env`` / ``st.secrets``."""

    provider: str | None
    api_key: str | None
    model: str | None
    base_url: str | None

    @classmethod
    def from_secrets(cls) -> ProviderSettings:
        """``LLM_PROVIDER``, ``LLM_API_KEY``, ``LLM_MODEL``, ``LLM_BASE_URL``."""
        return cls(
            provider=get_secret("LLM_PROVIDER"),
            api_key=get_secret("LLM_API_KEY"),
            model=get_secret("LLM_MODEL"),
            base_url=get_secret("LLM_BASE_URL"),
        )

    @property
    def enabled(self) -> bool:
        """A provider is selected (the key may still be missing)."""
        return bool((self.provider or "").strip())


def _http_error_category(exc: http.HttpError) -> str:
    if exc.status == 429:
        return "rate_limited"
    if exc.status in (401, 403):
        return "auth"
    if exc.status is None:
        return "timeout" if "timed out" in exc.reason else "connection"
    return "http_error"


@dataclass(frozen=True)
class OpenAICompatibleNarrator:
    """OpenAI Chat Completions over the shared HTTP session (no SDK).

    Works with any compatible endpoint via ``base_url`` (OpenAI, GitHub Models,
    OpenRouter, Azure-compatible gateways). Temperature 0, bounded completion
    tokens, narration-specific timeouts and retries from ``config.yaml``.
    """

    api_key: str
    model: str
    base_url: str
    cfg: Config
    session: requests.Session | None = None
    sleep: Callable[[float], None] | None = None
    name: str = "openai"

    @property
    def url(self) -> str:
        return f"{self.base_url.rstrip('/')}/chat/completions"

    def _http_cfg(self) -> Config:
        n = self.cfg["narration"]
        overrides = {k: n[k] for k in ("connect_timeout_s", "read_timeout_s", "max_retries")}
        return {**self.cfg, "http": {**self.cfg["http"], **overrides}}

    def complete(self, system: str, user: str) -> str:
        """POST the chat request and return the first choice's content."""
        n = self.cfg["narration"]
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": TEMPERATURE,
            str(n["max_tokens_param"]): int(n["max_tokens"]),
            "n": 1,
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        extra: dict[str, Any] = {} if self.sleep is None else {"sleep": self.sleep}
        try:
            response = http.request(
                "POST",
                self.url,
                headers=headers,
                json=body,
                session=self.session,
                cfg=self._http_cfg(),
                **extra,
            )
            data = response.json()
        except http.HttpError as exc:
            raise NarrationError(_http_error_category(exc)) from None
        except (ValueError, requests.RequestException):
            raise NarrationError("bad_response") from None
        try:
            choice = data["choices"][0]
            content = choice["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise NarrationError("bad_response") from None
        if choice.get("finish_reason") == "length":
            raise NarrationError("truncated")
        if not isinstance(content, str):
            raise NarrationError("bad_response")
        return content


def _openai_factory(settings: ProviderSettings, cfg: Config) -> Narrator:
    n = cfg["narration"]
    return OpenAICompatibleNarrator(
        api_key=settings.api_key or "",
        model=(settings.model or "").strip() or str(n["default_model"]),
        base_url=(settings.base_url or "").strip() or str(n["default_base_url"]),
        cfg=cfg,
    )


PROVIDERS: dict[str, Callable[[ProviderSettings, Config], Narrator]] = {
    "openai": _openai_factory,
    "openai_compatible": _openai_factory,
}


def make_narrator(settings: ProviderSettings, cfg: Config | None = None) -> Narrator:
    """The provider selected by ``settings``; :class:`NarrationError` if unusable.

    Categories: ``disabled`` (no ``LLM_PROVIDER``), ``unknown_provider``,
    ``no_key`` (no ``LLM_API_KEY``). No network call is made here.
    """
    cfg = cfg if cfg is not None else get_config()
    if not settings.enabled:
        raise NarrationError("disabled")
    factory = PROVIDERS.get((settings.provider or "").strip().lower())
    if factory is None:
        raise NarrationError("unknown_provider")
    if not (settings.api_key or "").strip():
        raise NarrationError("no_key")
    return factory(settings, cfg)


def narration_enabled(settings: ProviderSettings | None = None) -> bool:
    """Whether a provider is selected (cheap; used to skip the concurrent pool)."""
    settings = settings if settings is not None else ProviderSettings.from_secrets()
    return settings.enabled


def narration_cache_key(narrator: Narrator, system: str, user: str) -> str:
    """Cache key: provider, model and endpoint plus a hash of the prompt and payload."""
    digest = hashlib.sha256(f"{system}\n\n{user}".encode()).hexdigest()
    return make_key(
        "narration", f"{narrator.name}:{narrator.model}@{narrator.base_url}", {"prompt": digest}
    )


def _shared_cache() -> Cache | None:
    """The process-wide cache, or ``None`` if it can't be opened (narrate uncached)."""
    try:
        return get_cache()
    except Exception:
        logger.warning("narration cache unavailable")
        return None


def _complete_cached(narrator: Narrator, system: str, user: str, cache: Cache | None) -> str:
    key = narration_cache_key(narrator, system, user)
    if cache is not None:
        try:
            hit = cache.get(key)
        except Exception:
            hit = None
        if isinstance(hit, str):
            return hit
    text = narrator.complete(system, user)
    if cache is not None:
        try:
            cache.set(key, text)
        except Exception:
            logger.warning("narration cache write failed")
    return text


# --- entry point -------------------------------------------------------------


@dataclass(frozen=True)
class Narration:
    """The text shown in a card's narration slot and how it was produced."""

    text: str
    path: NarrationPath
    failure: str | None = None

    @property
    def label(self) -> str:
        """Caption for the UI: AI-written vs template."""
        return AI_LABEL if self.path == "llm" else TEMPLATE_LABEL


def template_only(
    verdict: Verdict,
    series_meta: Mapping[str, Any],
    cfg: Config | None = None,
    failure: str | None = "disabled",
) -> Narration:
    """The template narration for ``verdict`` (no provider involved)."""
    payload = build_payload(verdict, series_meta)
    return Narration(template_narration(payload, cfg), "template", failure)


def narrate(
    verdict: Verdict,
    series_meta: Mapping[str, Any],
    *,
    cfg: Config | None = None,
    settings: ProviderSettings | None = None,
    narrator: Narrator | None = None,
    cache: Cache | None = None,
) -> Narration:
    """Narrate ``verdict``: validated LLM text if possible, else the template.

    Never raises. On failure only the category (``disabled``, ``no_key``,
    ``unknown_provider``, ``rate_limited``, ``timeout``, ``auth``, ``http_error``,
    ``connection``, ``bad_response``, ``truncated``, ``validation``, ``error``) is
    logged: no prompt, reply or key.
    """
    cfg = cfg if cfg is not None else get_config()
    payload = build_payload(verdict, series_meta)
    template = template_narration(payload, cfg)

    def fallback(category: str, detail: str = "") -> Narration:
        if category != "disabled":
            logger.info("narration fell back to template: %s%s", category, detail)
        return Narration(template, "template", category)

    try:
        if narrator is None:
            settings = settings if settings is not None else ProviderSettings.from_secrets()
            narrator = make_narrator(settings, cfg)
            if cache is None:
                cache = _shared_cache()
        text = _complete_cached(
            narrator, system_prompt(cfg), USER_PROMPT_PREFIX + payload_json(payload), cache
        )
    except NarrationError as exc:
        return fallback(exc.category)
    except Exception:
        return fallback("error")
    text = " ".join(text.split())
    result = validate_narration(text, payload, cfg)
    if not result.ok:
        return fallback("validation", f" ({', '.join(result.codes)})")
    return Narration(text, "llm")


__all__ = [
    "AI_LABEL",
    "PROVIDERS",
    "SYSTEM_PROMPT",
    "TEMPLATE_LABEL",
    "Narration",
    "NarrationError",
    "Narrator",
    "OpenAICompatibleNarrator",
    "ProviderSettings",
    "ValidationResult",
    "build_payload",
    "make_narrator",
    "narrate",
    "template_narration",
    "validate_narration",
]
