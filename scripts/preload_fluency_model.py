"""Preload the WebQA fluency (BART) model into the HF cache.

Run once in the target venv so the first ``eval/evaluate_webqa_qa.py`` call
does not pay the HuggingFace download cost mid-evaluation.

Usage (Command Prompt, inside C:\\workspace\\MultimodalGraphRAG\\.venv):
    python colgraphrag_webqa\\scripts\\preload_fluency_model.py

Override the checkpoint via ``WEBQA_FLUENCY_MODEL``.
"""
from __future__ import annotations

import os
import sys


def main() -> int:
    name = os.getenv("WEBQA_FLUENCY_MODEL", "facebook/bart-large-cnn").strip()
    print(f"[preload] target checkpoint: {name}")
    try:
        from transformers import BartForConditionalGeneration, BartTokenizer
    except Exception as exc:  # noqa: BLE001
        print(
            f"[preload] transformers import failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1
    try:
        print("[preload] downloading tokenizer...")
        BartTokenizer.from_pretrained(name)
        print("[preload] downloading model weights (may take a few minutes)...")
        BartForConditionalGeneration.from_pretrained(name)
    except Exception as exc:  # noqa: BLE001
        print(
            f"[preload] HF download failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        print(
            "[preload] evaluator will fall back to ROUGE-L; set "
            "WEBQA_FLUENCY_BACKEND=rouge to skip BART entirely.",
            file=sys.stderr,
        )
        return 2
    print("[preload] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
