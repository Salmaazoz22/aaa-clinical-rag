# -*- coding: utf-8 -*-
"""Turning a model response into the answer object.

Neither supported model can be relied on to return a bare JSON object:
`openai/gpt-oss-120b` honours JSON mode but is a reasoning model, and
`deepseek/deepseek-r1:free` has no JSON mode at all and routinely wraps its
answer in a `<think>` block, a markdown fence, or both. The reference
implementation returns the model's text straight through to the caller, which
works when the contract is "some prose" and does not when the contract is "an
object whose citations will be checked against the retrieval".

So parsing is explicit and defensive, and it never repairs content: it locates
the JSON object and hands it over. Anything wrong *inside* the object is the
validator's business, and gets reported rather than fixed.
"""
from __future__ import annotations

import json
import re
from typing import Any

#: A complete reasoning block. DeepSeek-R1 emits these inline when the route does
#: not split reasoning onto its own field.
_THINK_BLOCK = re.compile(r"<think\b[^>]*>.*?</think\s*>", re.DOTALL | re.IGNORECASE)

#: An unterminated reasoning block: the response was cut off, or the closing tag
#: was dropped. Everything up to the last plausible answer start is discarded.
_THINK_OPEN = re.compile(r"<think\b[^>]*>", re.IGNORECASE)

_FENCE = re.compile(r"^\s*```(?:json|JSON)?\s*\n(.*?)\n?\s*```\s*$", re.DOTALL)


class AnswerParseError(ValueError):
    """Raised when no JSON object can be located in a model response."""


def strip_reasoning(text: str) -> str:
    """Remove inline reasoning blocks.

    A reasoning stream is not the answer and must never reach the validator: it
    quotes chunk_ids and guideline text while *considering* them, so parsing it
    as the answer would manufacture citations the model did not actually make.
    """
    cleaned = _THINK_BLOCK.sub("", text)
    match = _THINK_OPEN.search(cleaned)
    if match:
        # Unterminated block. The answer, if any, is after it.
        cleaned = cleaned[match.end():]
    return cleaned.strip()


def strip_code_fences(text: str) -> str:
    match = _FENCE.match(text.strip())
    return match.group(1).strip() if match else text.strip()


def find_json_object(text: str) -> str:
    """Return the first complete top-level JSON object in `text`.

    Brace counting, string- and escape-aware, so a `{` or `}` inside quoted
    guideline text cannot end the scan early. Naive approaches -- first `{` to
    last `}`, or a regex -- both break on the excerpts this layer asks for,
    which routinely contain braces from PDF table extraction.
    """
    start = text.find("{")
    if start == -1:
        raise AnswerParseError("no '{' in model response")

    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    raise AnswerParseError("unterminated JSON object in model response (response may be truncated)")


def parse_answer(text: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Parse a model response into `(answer, parse_meta)`.

    `parse_meta` records what had to be stripped, so an answer that needed
    unwrapping is distinguishable in the evaluation artifact from one that
    arrived clean.
    """
    if not isinstance(text, str) or not text.strip():
        raise AnswerParseError("model response is empty")

    raw = text
    meta: dict[str, Any] = {
        "had_reasoning_block": bool(_THINK_BLOCK.search(raw) or _THINK_OPEN.search(raw)),
        "had_code_fence": False,
        "had_text_outside_object": False,
    }

    cleaned = strip_reasoning(raw)
    unfenced = strip_code_fences(cleaned)
    meta["had_code_fence"] = unfenced != cleaned

    # The happy path: the whole response is the object.
    try:
        parsed = json.loads(unfenced)
    except json.JSONDecodeError:
        candidate = find_json_object(unfenced)
        meta["had_text_outside_object"] = candidate.strip() != unfenced.strip()
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as exc:
            raise AnswerParseError(f"model response is not valid JSON: {exc}") from exc

    if not isinstance(parsed, dict):
        raise AnswerParseError(f"model returned a JSON {type(parsed).__name__}, expected an object")

    return parsed, meta
