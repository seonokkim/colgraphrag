"""
Normalize WebQA gold field `A` (or an exported `answers[].answer` cell) to plain strings.

WebQA JSON often stores `A` as a list or as a string that looks like a Python list literal,
e.g. ['"No, ..."']. QA scoring tokenizes the whole cell, so list/bracket artifacts must be removed.
"""

from __future__ import annotations

import ast
from typing import Any


def _strip_outer_quotes(s: str) -> str:
    t = s.strip()
    if len(t) >= 2 and t[0] == t[-1] and t[0] in "\"'":
        return t[1:-1].strip()
    return t


def normalize_webqa_answer_strings(raw: Any) -> list[str]:
    """
    Return one or more reference strings suitable for list EM / list F1.

    - ``None`` / empty -> ``[""]``
    - ``list`` / ``tuple`` -> flatten recursively, drop empties, or ``[""]`` if none
    - ``str`` that parses as a list literal (``[...]``) -> ``ast.literal_eval`` then flatten
    - otherwise a single stripped string (outer quotes removed once)
    """
    if raw is None:
        return [""]
    if isinstance(raw, (list, tuple)):
        parts: list[str] = []
        for x in raw:
            parts.extend(normalize_webqa_answer_strings(x))
        parts = [p for p in parts if p]
        return parts if parts else [""]
    s = str(raw).strip()
    if not s:
        return [""]
    if s.startswith("[") and s.endswith("]"):
        try:
            parsed = ast.literal_eval(s)
        except (ValueError, SyntaxError, MemoryError, TypeError):
            parsed = None
        if isinstance(parsed, (list, tuple)):
            return normalize_webqa_answer_strings(list(parsed))
    one = _strip_outer_quotes(s)
    return [one] if one else [""]
