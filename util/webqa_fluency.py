"""WebQA fluency (QA-FL) via BARTScore.

Implements a pinned-normalization of ``log p(gold | pred)`` from a BART-like
seq2seq so the value lives in ``[0, 1]`` and matches the WebQA paper's
``FL_i`` definition (Chang et al. CVPR 2022, Sec. 4.3).

Leaderboard field mapping (see
``reference/code/WebQA-main/WebQA-main/demo/Call_WebQA_eval_server_locally.py``):

* ``Fluency`` (``QA-FL``) <- :func:`fluency_score`
* ``Accuracy`` (``QA-Acc``) <- :mod:`util.webqa_metrics_approx`
* ``mul`` (``QA``) <- per-sample ``FL_i * Acc_i`` aggregated in
  :mod:`eval.evaluate_webqa_qa`.

Env knobs (read lazily on first call):

* ``WEBQA_FLUENCY_MODEL`` -- HF checkpoint (default ``facebook/bart-large-cnn``).
* ``WEBQA_FLUENCY_DEVICE`` -- ``cuda`` | ``cpu`` | ``auto`` (default auto).
* ``WEBQA_FLUENCY_MAX_TOKENS`` -- truncate pred+gold (default ``256``).
* ``WEBQA_FLUENCY_SCALE`` -- normalization temperature (default ``4.0``).
* ``WEBQA_FLUENCY_BACKEND`` -- ``bart`` | ``rouge`` (default ``bart``). Set to
  ``rouge`` on CPU-only boxes with no HF cache; the module also degrades to
  ROUGE-L F1 automatically if BART fails to load.

A last-resort fallback uses a tiny inline ROUGE-L F1 so the evaluator never
crashes when BART weights are unavailable.
"""
from __future__ import annotations

import logging
import math
import os
import threading
from typing import Iterable

LOG = logging.getLogger("webqa.fluency")

_LOCK = threading.Lock()
_BART_STATE: dict = {}


def _resolve_device() -> str:
    raw = os.getenv("WEBQA_FLUENCY_DEVICE", "auto").strip().lower()
    if raw in {"cpu", "cuda"}:
        return raw
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _load_bart() -> dict:
    """Lazy, process-wide singleton. Not called in the ROUGE-fallback branch."""
    global _BART_STATE
    with _LOCK:
        if _BART_STATE:
            return _BART_STATE
        import torch  # noqa: F401  (ensure torch is importable before HF)
        from transformers import BartForConditionalGeneration, BartTokenizer

        model_name = os.getenv(
            "WEBQA_FLUENCY_MODEL", "facebook/bart-large-cnn"
        ).strip()
        device = _resolve_device()
        LOG.info(
            "loading BART for QA-FL | model=%s device=%s", model_name, device
        )
        tok = BartTokenizer.from_pretrained(model_name)
        model = BartForConditionalGeneration.from_pretrained(model_name).to(
            device
        )
        model.eval()
        _BART_STATE = {
            "tok": tok,
            "model": model,
            "device": device,
            "name": model_name,
        }
        return _BART_STATE


def _bart_forward_score(pred: str, gold: str) -> float:
    """Return average ``log p(gold_token | pred)`` (higher is more fluent).

    Uses ``pred`` as the encoder source and ``gold`` as the decoder target so
    fluency is ``log p(gold | pred)`` -- the direction BARTScore's ``faithful``
    variant uses. This mirrors the WebQA paper's ``FL_i`` which rewards
    predictions that can regenerate the gold answer verbatim.
    """
    import torch

    state = _load_bart()
    tok, model, device = state["tok"], state["model"], state["device"]
    max_len = int(os.getenv("WEBQA_FLUENCY_MAX_TOKENS", "256"))
    src = tok(
        pred if pred else " ",
        return_tensors="pt",
        truncation=True,
        max_length=max_len,
    ).to(device)
    tgt = tok(
        gold if gold else " ",
        return_tensors="pt",
        truncation=True,
        max_length=max_len,
    ).to(device)
    with torch.no_grad():
        out = model(
            input_ids=src["input_ids"],
            attention_mask=src["attention_mask"],
            labels=tgt["input_ids"],
        )
    neg_logp = float(out.loss.detach().cpu())
    return -neg_logp


def _normalize_log_p(logp: float) -> float:
    """Map token-averaged log-likelihood into ``[0, 1]``.

    ``exp(logp)`` is already in ``(0, 1]`` but BART's raw scale is harsh, so
    the WebQA surrogate uses an affine+exp transform with a temperature. We
    expose that temperature via ``WEBQA_FLUENCY_SCALE`` so it can be pinned
    to match the hidden leaderboard calibration.
    """
    scale = float(os.getenv("WEBQA_FLUENCY_SCALE", "4.0"))
    val = math.exp(logp / max(scale, 1e-3))
    if val < 0.0:
        return 0.0
    if val > 1.0:
        return 1.0
    return val


def _rouge_l_fmeasure(pred: str, gold: str) -> float:
    """CPU fallback. Inline ROUGE-L F1 in ``[0, 1]``."""
    p = (pred or "").lower().split()
    g = (gold or "").lower().split()
    if not p or not g:
        return 0.0
    dp = [[0] * (len(g) + 1) for _ in range(len(p) + 1)]
    for i in range(1, len(p) + 1):
        for j in range(1, len(g) + 1):
            if p[i - 1] == g[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[-1][-1]
    if lcs == 0:
        return 0.0
    pr = lcs / len(p)
    rc = lcs / len(g)
    return 2 * pr * rc / (pr + rc)


def fluency_score(prediction: str, gold: str) -> float:
    """Return QA-FL in ``[0, 1]``. Falls back to ROUGE-L if BART is unavailable."""
    backend = os.getenv("WEBQA_FLUENCY_BACKEND", "bart").strip().lower()
    if backend == "rouge":
        return _rouge_l_fmeasure(prediction, gold)
    try:
        return _normalize_log_p(_bart_forward_score(prediction, gold))
    except Exception as exc:  # noqa: BLE001
        msg = f"{type(exc).__name__}: {exc}"
        LOG.warning("BART fluency failed, falling back to ROUGE-L | %s", msg)
        os.environ["WEBQA_FLUENCY_LAST_ERROR"] = msg
        return _rouge_l_fmeasure(prediction, gold)


def fluency_scores_batch(pairs: Iterable[tuple[str, str]]) -> list[float]:
    """Ordered list of FL scores. Kept for caller readability / future batching."""
    return [fluency_score(p, g) for p, g in pairs]


def active_backend() -> str:
    """Return the backend that ``fluency_score`` will try first on this process."""
    return os.getenv("WEBQA_FLUENCY_BACKEND", "bart").strip().lower()


def active_model_name() -> str:
    """Return the BART checkpoint used (default ``facebook/bart-large-cnn``)."""
    return os.getenv("WEBQA_FLUENCY_MODEL", "facebook/bart-large-cnn").strip()
