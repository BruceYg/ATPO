"""Parsing of generated moderation responses into per-category decisions.

The JSON-style parser reproduces the extraction cascade of the historical reward
functions and inference scripts (``extract_moderation_result``):

1. quoted full keys      ``"C1(Sexual Content)": true``
2. quoted short IDs      ``"C1": true``                 (only if step 1 found nothing)
3. asterisk bullets      ``* C1(Sexual Content): true``  (only if steps 1-2 found nothing)
4. full keys inside the ``GUARDRAIL{...}`` block       (subsumed by step 1; kept for fidelity)

Matching is case-insensitive and uses the first occurrence of each key. A key
that is absent is *missing*; historical code treated missing keys as negative.

The ``simple`` parser reproduces ``extract_simple_moderation_result`` used for
SFT-format outputs such as ``GUARDRAIL: C2, C3``: every category ID mentioned as
a standalone token is positive.

Unlike the historical inference scripts, :func:`parse_response` reports whether
parsing succeeded. Callers decide how to treat invalid outputs; nothing here
turns an unparseable response into a safe prediction.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal, Optional

from .taxonomy import Taxonomy

ResponseFormat = Literal["think_answer", "triple_newline", "plain"]
OutputStyle = Literal["json", "simple"]
ParseScope = Literal["answer_only", "full_response", "answer_or_full"]
ParseStatus = Literal["ok", "partial", "invalid"]

RESPONSE_FORMATS: tuple[str, ...] = ("think_answer", "triple_newline", "plain")
OUTPUT_STYLES: tuple[str, ...] = ("json", "simple")
PARSE_SCOPES: tuple[str, ...] = ("answer_only", "full_response", "answer_or_full")

# Patterns of the reward functions used in the paper's runs.
_THINK_ANSWER_FORMAT = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.DOTALL)
_ANSWER_BLOCK = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
_THINK_BLOCK = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
_TRIPLE_NEWLINE_FORMAT = re.compile(r"^.*?\n\n\n.*?$", re.DOTALL | re.MULTILINE)
_TRIPLE_NEWLINE = "\n\n\n"


def format_ok(response: Optional[str], response_format: ResponseFormat) -> Optional[bool]:
    """Historical format check used by the format reward component.

    ``think_answer``: the stripped response must be exactly ``<think>..</think><answer>..</answer>``.
    ``triple_newline``: the response must contain a blank-line separator ``\\n\\n\\n``
    (Qwen3-VL recipes). ``plain`` has no format requirement and returns ``None``.
    """
    text = response or ""
    if response_format == "think_answer":
        return _THINK_ANSWER_FORMAT.fullmatch(text.strip()) is not None
    if response_format == "triple_newline":
        return _TRIPLE_NEWLINE_FORMAT.match(text) is not None
    if response_format == "plain":
        return None
    raise ValueError(f"unknown response_format {response_format!r}")


def extract_answer(response: Optional[str], response_format: ResponseFormat) -> Optional[str]:
    """Return the answer section, or ``None`` when the format defines one and it is absent."""
    if response_format == "think_answer":
        match = _ANSWER_BLOCK.search(response or "")
        return match.group(1).strip() if match else None
    if response_format == "triple_newline":
        if not response:
            return None
        parts = response.split(_TRIPLE_NEWLINE, 1)
        return parts[1].strip() if len(parts) == 2 else None
    if response_format == "plain":
        return response or ""
    raise ValueError(f"unknown response_format {response_format!r}")


def extract_explanation(response: Optional[str], response_format: ResponseFormat) -> Optional[str]:
    """Return generated reasoning text when the response format separates it from the answer."""
    if not response:
        return None
    if response_format == "think_answer":
        match = _THINK_BLOCK.search(response)
        return match.group(1).strip() if match else None
    if response_format == "triple_newline":
        parts = response.split(_TRIPLE_NEWLINE, 1)
        return parts[0].strip() if len(parts) == 2 else None
    return None


def parse_json_decisions(text: str, taxonomy: Taxonomy) -> Optional[dict[str, bool]]:
    """Historical JSON-style extraction cascade. Returns ``None`` if nothing was found."""
    text = text or ""
    result: dict[str, bool] = {}

    def _search(pattern: str, haystack: str) -> Optional[bool]:
        match = re.search(pattern, haystack, re.IGNORECASE)
        if match:
            return match.group(1).lower() == "true"
        return None

    full_keys = [(c.id, taxonomy.output_key(c)) for c in taxonomy.categories]
    for cid, key in full_keys:
        value = _search(rf'"{re.escape(key)}":\s*(true|false|True|False)', text)
        if value is not None:
            result[cid] = value
    if not result:
        for cid in taxonomy.ids:
            # Historical code interpolated the short key without escaping; IDs are
            # restricted to regex-safe characters except '.', which we escape.
            value = _search(rf'"{re.escape(cid)}":\s*(true|false|True|False)', text)
            if value is not None:
                result[cid] = value
    if not result:
        for cid, key in full_keys:
            value = _search(rf"\*\s*{re.escape(key)}:\s*(true|false|True|False)", text)
            if value is not None:
                result[cid] = value
    if not result:
        block = re.search(
            rf"{re.escape(taxonomy.block_name)}[:\s=]+\{{([^}}]+)\}}", text, re.IGNORECASE | re.DOTALL
        )
        if block:
            for cid, key in full_keys:
                value = _search(rf'"{re.escape(key)}":\s*(true|false|True|False)', block.group(1))
                if value is not None:
                    result[cid] = value
    return result or None


def parse_simple_decisions(text: str, taxonomy: Taxonomy) -> dict[str, bool]:
    """Historical simple-format extraction: an ID mentioned as a standalone token is positive."""
    text = text or ""
    return {
        cid: re.search(rf"\b{re.escape(cid)}\b", text, re.IGNORECASE) is not None
        for cid in taxonomy.ids
    }


@dataclass
class ParsedResponse:
    """Outcome of parsing one generated response.

    ``status``
        ``ok``: every category received an explicit decision;
        ``partial``: some categories were found, the rest are missing (``None``);
        ``invalid``: no decision could be extracted.
    ``labels``
        Positive category IDs in taxonomy order, or ``None`` when ``status`` is
        ``invalid``. For ``partial`` results, missing categories are not included.
    ``historical_labels``
        The labels the historical inference scripts would have stored: invalid
        JSON outputs became all-negative, and the simple parser never failed.
        Use only for reproducing historical metrics.
    """

    status: ParseStatus
    decisions: dict[str, Optional[bool]]
    labels: Optional[list[str]]
    historical_labels: list[str]
    source: Optional[str]
    answer_found: Optional[bool]
    format_ok: Optional[bool]
    error: Optional[str] = None
    explanation: Optional[str] = field(default=None, repr=False)

    @property
    def valid(self) -> bool:
        return self.status != "invalid"

    @property
    def missing(self) -> list[str]:
        return [cid for cid, value in self.decisions.items() if value is None]


def parse_response(
    response: Optional[str],
    taxonomy: Taxonomy,
    *,
    response_format: ResponseFormat = "think_answer",
    output_style: OutputStyle = "json",
    scope: ParseScope = "answer_only",
    require_block_prefix: bool = True,
) -> ParsedResponse:
    """Parse a generated response.

    ``scope`` selects the text that is parsed:

    ``answer_only``
        Only the answer section (historical training rewards with ``answer_only=true``).
        A missing answer section makes the response invalid.
    ``full_response``
        The entire response (historical inference and evaluation scripts).
    ``answer_or_full``
        The answer section when present, otherwise the entire response.

    For ``output_style="simple"`` and ``require_block_prefix=True`` a response
    without ``<BLOCK>:`` (for example ``GUARDRAIL:``) is reported as invalid,
    although its ``historical_labels`` still follow the historical parser.
    """
    if response_format not in RESPONSE_FORMATS:
        raise ValueError(f"unknown response_format {response_format!r}")
    if output_style not in OUTPUT_STYLES:
        raise ValueError(f"unknown output_style {output_style!r}")
    if scope not in PARSE_SCOPES:
        raise ValueError(f"unknown parse scope {scope!r}")

    text = response or ""
    fmt = format_ok(text, response_format)
    answer = extract_answer(text, response_format) if response_format != "plain" else None
    answer_found = None if response_format == "plain" else answer is not None
    explanation = extract_explanation(text, response_format)

    if scope == "answer_only" and response_format != "plain":
        target, source = answer, "answer"
    elif scope == "answer_or_full" and answer is not None:
        target, source = answer, "answer"
    else:
        target, source = text, "full_response"

    empty = {cid: None for cid in taxonomy.ids}
    if target is None:
        return ParsedResponse(
            status="invalid",
            decisions=empty,
            labels=None,
            historical_labels=[],
            source=None,
            answer_found=answer_found,
            format_ok=fmt,
            error="answer section not found",
            explanation=explanation,
        )

    if output_style == "simple":
        found = parse_simple_decisions(target, taxonomy)
        historical = [cid for cid in taxonomy.ids if found[cid]]
        has_prefix = re.search(rf"{re.escape(taxonomy.block_name)}\s*:", target, re.IGNORECASE)
        if require_block_prefix and not has_prefix:
            return ParsedResponse(
                status="invalid",
                decisions=empty,
                labels=None,
                historical_labels=historical,
                source=source,
                answer_found=answer_found,
                format_ok=fmt,
                error=f"'{taxonomy.block_name}:' prefix not found",
                explanation=explanation,
            )
        return ParsedResponse(
            status="ok",
            decisions=dict(found),
            labels=historical,
            historical_labels=historical,
            source=source,
            answer_found=answer_found,
            format_ok=fmt,
            explanation=explanation,
        )

    found_json = parse_json_decisions(target, taxonomy)
    if found_json is None:
        return ParsedResponse(
            status="invalid",
            decisions=empty,
            labels=None,
            historical_labels=[],
            source=source,
            answer_found=answer_found,
            format_ok=fmt,
            error="no category decisions found",
            explanation=explanation,
        )
    decisions: dict[str, Optional[bool]] = {cid: found_json.get(cid) for cid in taxonomy.ids}
    labels = [cid for cid in taxonomy.ids if decisions[cid]]
    status: ParseStatus = "ok" if all(v is not None for v in decisions.values()) else "partial"
    return ParsedResponse(
        status=status,
        decisions=decisions,
        labels=labels,
        historical_labels=list(labels),
        source=source,
        answer_found=answer_found,
        format_ok=fmt,
        explanation=explanation,
    )
