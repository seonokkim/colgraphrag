"""
Integration tests for :mod:`mllm.ollama_gemma_4_e2b` and :mod:`mllm.ollama_gemma_4_e4b`.

Skipped when Ollama is not running or the model is missing.

Run from repo root (venv active):

    python -m unittest discover -s tests -v

Single file:

    python -m unittest tests.test_ollama -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mllm import ollama_gemma_4_e2b as gemma_e2b  # noqa: E402
from mllm import ollama_gemma_4_e4b as gemma_e4b  # noqa: E402

_SAMPLE_IMAGE = (
    _REPO_ROOT / "data" / "multimodalqa" / "final_dataset_images" / "0a3a47658239ae8209094020669304c0.jpg"
)

_DETERMINISTIC_OPTS = {"temperature": 0, "top_k": 1}


class TestOllamaModuleLocalGuards(unittest.TestCase):
    """No running Ollama daemon required."""

    def test_default_model_nonempty(self) -> None:
        self.assertTrue(gemma_e2b.default_model().strip())
        self.assertTrue(gemma_e4b.default_model().strip())

    def test_generate_from_image_missing_file_e2b(self) -> None:
        with self.assertRaises(FileNotFoundError):
            gemma_e2b.generate_from_image(
                _REPO_ROOT / "nonexistent" / "no_such_image.jpg",
                "Describe this image.",
            )

    def test_generate_from_image_missing_file_e4b(self) -> None:
        with self.assertRaises(FileNotFoundError):
            gemma_e4b.generate_from_image(
                _REPO_ROOT / "nonexistent" / "no_such_image.jpg",
                "Describe this image.",
            )


class TestOllamaGemma4E2B(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not gemma_e2b.configured():
            raise unittest.SkipTest(
                "Ollama unreachable or gemma4:e2b not in `ollama list`",
            )

    def test_text_to_text_capital_city(self) -> None:
        out = gemma_e2b.generate_text(
            "Answer with a single lowercase word only. Capital of France?",
            options=_DETERMINISTIC_OPTS,
        )
        self.assertIn("paris", out.lower())

    def test_chat_single_user_message(self) -> None:
        out = gemma_e2b.chat(
            [{"role": "user", "content": "Reply with one digit only, no words: 2+2=?"}],
            options=_DETERMINISTIC_OPTS,
        )
        compact = out.replace(" ", "").replace("\n", "")
        self.assertIn("4", compact)

    def test_image_to_text_umbrella_color(self) -> None:
        self.assertTrue(
            _SAMPLE_IMAGE.is_file(),
            f"fixture image missing: {_SAMPLE_IMAGE}",
        )
        out = gemma_e2b.generate_from_image(
            _SAMPLE_IMAGE,
            "What is the dominant color of the umbrella? Reply with one word only.",
            options=_DETERMINISTIC_OPTS,
        )
        self.assertIn("pink", out.lower())


class TestOllamaGemma4E4B(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not gemma_e4b.configured():
            raise unittest.SkipTest(
                "Ollama unreachable or configured E4B tag not in `ollama list`",
            )

    def test_text_to_text_capital_city(self) -> None:
        out = gemma_e4b.generate_text(
            "Answer with a single lowercase word only. Capital of France?",
            options=_DETERMINISTIC_OPTS,
        )
        self.assertIn("paris", out.lower())

    def test_chat_single_user_message(self) -> None:
        out = gemma_e4b.chat(
            [{"role": "user", "content": "Reply with one digit only, no words: 2+2=?"}],
            options=_DETERMINISTIC_OPTS,
        )
        compact = out.replace(" ", "").replace("\n", "")
        self.assertIn("4", compact)

    def test_image_to_text_umbrella_color(self) -> None:
        self.assertTrue(
            _SAMPLE_IMAGE.is_file(),
            f"fixture image missing: {_SAMPLE_IMAGE}",
        )
        out = gemma_e4b.generate_from_image(
            _SAMPLE_IMAGE,
            "What is the dominant color of the umbrella? Reply with one word only.",
            options=_DETERMINISTIC_OPTS,
        )
        self.assertIn("pink", out.lower())


if __name__ == "__main__":
    unittest.main()
