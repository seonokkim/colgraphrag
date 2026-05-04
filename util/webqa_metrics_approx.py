"""WebQA QA accuracy approximation (val/n100 local scoring).

Ported from the WebQA authors' approximation in
`reference/code/WebQA-main/WebQA-main/eval_webqa.md` (contributed by Qibin Chen,
see https://github.com/qibinc):

    def _webqa_acc_approx(prediction, ground_truth, domain=None): ...
    def webqa_metrics_approx(prediction, ground_truth, Qcate="text"): ...

WebQA golds are full natural-language sentences and MMQA-style ``list_em`` /
``list_f1`` penalizes that — this module implements the category-aware keyword
overlap the WebQA leaderboard uses to approximate ``QA-Acc`` on the val set.

Keyword sets are chosen to cover the WebQA color / shape / YesNo answer
vocabulary on val (see `eval_webqa.md` for the general recipe); for ``number``
we match any numeric token (digits, decimals, or spelled-out integers up to
twenty) via ``detect_numbers``.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Iterable

COLOR_SET: frozenset[str] = frozenset({
    "red", "orange", "yellow", "green", "blue", "purple", "pink", "brown",
    "white", "gray", "grey", "black", "beige", "tan", "turquoise", "maroon",
    "navy", "violet", "gold", "golden", "silver", "cyan", "magenta", "teal",
    "cream", "ivory", "indigo", "lavender", "bronze", "copper", "rose",
    "burgundy", "scarlet", "crimson", "khaki", "olive", "lime", "mint",
    "coral", "peach", "salmon", "charcoal", "amber",
})

SHAPE_SET: frozenset[str] = frozenset({
    "circle", "circular", "round", "oval", "ellipse", "elliptical",
    "square", "rectangular", "rectangle", "triangle", "triangular",
    "diamond", "rhombus", "hexagon", "hexagonal", "octagon", "octagonal",
    "pentagon", "pentagonal", "star", "heart", "crescent", "arch", "arched",
    "cone", "conical", "cylinder", "cylindrical", "cube", "cubic",
    "spherical", "sphere", "pyramid", "pyramidal", "cross", "arrow",
    "trapezoid", "trapezoidal", "parallelogram", "curved", "straight",
    "footprint", "footprints", "dome", "domed",
})

YESNO_SET: frozenset[str] = frozenset({"yes", "no"})

_NUMBER_WORDS: dict[str, int] = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20,
    "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
    "eighty": 80, "ninety": 90,
    "hundred": 100, "thousand": 1000, "million": 1_000_000,
}

_DOMAIN_FOR_QCATE: dict[str, frozenset[str] | None | str] = {
    "color": COLOR_SET,
    "shape": SHAPE_SET,
    "YesNo": YESNO_SET,
    "number": "NUMBER",
    "text": None,
    "Others": None,
    "choose": None,
}

_ARTICLE_RE = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_WS_RE = re.compile(r"\s+", re.UNICODE)
_PUNC_TABLE = str.maketrans("", "", string.punctuation)


def _normalize_text(text: str) -> str:
    """Lowercase, strip articles / punctuation, collapse whitespace."""
    s = (text or "").lower()
    s = s.translate(_PUNC_TABLE)
    s = _ARTICLE_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


def _detect_numbers(tokens: Iterable[str]) -> list[str]:
    out: list[str] = []
    for tok in tokens:
        if not tok:
            continue
        try:
            v = float(tok)
            if v.is_integer():
                out.append(str(int(v)))
            else:
                out.append(str(v))
            continue
        except ValueError:
            pass
        if tok in _NUMBER_WORDS:
            out.append(str(_NUMBER_WORDS[tok]))
    return out


def qcate_keyword_tokens(text: str, Qcate: str = "text") -> list[str]:
    """Return the Qcate-filtered keyword bag for ``text``.

    Mirrors the token pipeline inside :func:`webqa_acc_approx` so downstream
    scorers (e.g. MMQA-style list EM/F1) can reduce WebQA's sentence-level
    gold to the same keyword set the WebQA leaderboard would match on.

    * ``color`` / ``shape`` / ``YesNo`` -> tokens kept only if they appear in
      the domain keyword set.
    * ``number`` -> numeric tokens (digits or spelled-out 0..20) via
      :func:`_detect_numbers`.
    * ``text`` / ``Others`` / ``choose`` -> raw normalized tokens (no filter),
      matching ``webqa_acc_approx``'s free-text recall domain.
    """
    domain = _DOMAIN_FOR_QCATE.get(Qcate, None)
    tokens = _normalize_text(text).split()
    if domain == "NUMBER":
        return _detect_numbers(tokens)
    if isinstance(domain, frozenset):
        return [t for t in tokens if t in domain]
    return tokens


def webqa_acc_approx(
    prediction: str,
    ground_truth: str,
    Qcate: str = "text",
) -> dict[str, float]:
    """Return ``f1``, ``recall``, ``precision`` for one (prediction, gold) pair.

    Domain for ``Qcate`` picks either a keyword set (color/shape/YesNo), a
    number filter, or the raw bag-of-words (text/Others/choose).
    """
    domain = _DOMAIN_FOR_QCATE.get(Qcate, None)
    bow_pred = _normalize_text(prediction).split()
    bow_target = _normalize_text(ground_truth).split()
    if domain == "NUMBER":
        bow_pred = _detect_numbers(bow_pred)
        bow_target = _detect_numbers(bow_target)
    elif isinstance(domain, frozenset):
        bow_pred = [w for w in bow_pred if w in domain]
        bow_target = [w for w in bow_target if w in domain]
    if not bow_pred or not bow_target:
        return {"f1": 0.0, "recall": 0.0, "precision": 0.0}
    common = Counter(bow_target) & Counter(bow_pred)
    num_same = sum(common.values())
    if num_same == 0:
        return {"f1": 0.0, "recall": 0.0, "precision": 0.0}
    precision = num_same / len(bow_pred)
    recall = num_same / len(bow_target)
    f1 = 2 * precision * recall / (precision + recall)
    return {"f1": f1, "recall": recall, "precision": precision}


def webqa_metrics_approx(
    prediction: str,
    ground_truth: str,
    Qcate: str = "text",
) -> dict[str, float]:
    """Return ``acc_approx`` per WebQA val convention.

    For keyword / number categories (color, shape, number, YesNo), accuracy
    equals F1 over the filtered bag-of-words. For free-form categories
    (text, Others, choose), accuracy equals recall over raw tokens.
    """
    m = webqa_acc_approx(prediction, ground_truth, Qcate=Qcate)
    if Qcate in {"color", "shape", "number", "YesNo"}:
        acc = m["f1"]
    else:
        acc = m["recall"]
    return {
        "acc_approx": float(acc),
        "f1": float(m["f1"]),
        "recall": float(m["recall"]),
        "precision": float(m["precision"]),
    }
