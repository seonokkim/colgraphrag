"""LLM services for live inference in the demo backend.

Self-contained — no imports from the repo-level mllm/ package.

Three LLM backends:
  - LLMService            : HuggingFace Gemma 4 E4B IT (local weights)
  - OllamaLLMService      : Ollama gemma4:e2b (``OLLAMA_GEMMA4_E2B_MODEL``)
  - OllamaE4BLLMService   : Ollama gemma4:e4b (``OLLAMA_GEMMA4_E4B_MODEL``)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx

logger = logging.getLogger(__name__)

_DEMO_BE_DIR = Path(__file__).resolve().parents[1]   # demo/be/
_REPO_ROOT   = Path(__file__).resolve().parents[3]   # repo root

# ── Prompt templates (shared) ────────────────────────────────────────────────

DIRECT_QA_PROMPT = r'''Goal
Answer the question. Provide a complete, self-contained answer sentence.

Important:
- For Yes/No questions: start with "Yes" or "No", then explain in a complete sentence.
- For "Which" or comparison questions, state the chosen option and the reason.
- For descriptive questions, state the answer with relevant details from your knowledge.
- Always respond in a complete sentence, not just a word or phrase.
- If you are uncertain, provide your best answer based on general knowledge.

######################
Question: {question}
######################
Output:'''

LLM_ANSWER_PROMPT = r'''Goal
Answer the question based on the knowledge graph. Provide a complete, self-contained answer sentence.

Steps:
1. Read the question carefully and identify what is being asked.
2. Look at the graph nodes and relationships for relevant information.
3. Write a complete sentence that directly answers the question.

Important:
- For Yes/No questions: start with "Yes" or "No", then explain in a complete sentence.
- For "Which" or comparison questions, state the chosen option and the reason.
- For descriptive questions, state the answer with relevant details.
- Always respond in a complete sentence, not just a word or phrase.
- Restate key terms from the question so the answer is self-contained.

Examples:
Question: Are the tops of the caps of two mushrooms the same color?
Answer: No, the tops of the caps of the two mushrooms are not the same color as each other.

Question: Are there any buildings shorter than the flag pole?
Answer: No, there are not any buildings shorter than the flag pole.

Question: Does the Cincinnati Music Hall have columns inside and outside?
Answer: Yes, the Cincinnati Music Hall has columns inside and outside.

Question: Which building has more windows, A or B?
Answer: Building A has more windows on the facing with its entrance.

Question: What is the capital of France?
Answer: The capital of France is Paris.

######################
-Real Data-
######################
Input:
Question: {question}
Knowledge Graph:
{GraphML}
######################
Output:'''


def graph_to_str(graph: nx.Graph) -> str:
    """Convert a networkx graph to LLM-readable text blocks."""
    output = []
    text_nodes: List[Dict] = []
    image_nodes: List[Dict] = []
    table_nodes: List[Dict] = []

    for node_id, node_data in graph.nodes(data=True):
        info = {
            "id": node_id,
            "name": node_data.get("entity_name", ""),
            "type": node_data.get("type", ""),
            "description": node_data.get("description", ""),
        }
        if node_id.endswith("IMAGE"):
            image_nodes.append(info)
        elif node_id.endswith("TABLE"):
            table_nodes.append(info)
        else:
            text_nodes.append(info)

    output.append("======= BEGIN: TEXT NODES BLOCK =======")
    for n in text_nodes:
        if n["name"] and n["type"]:
            output += [f"Name: {n['name']}", f"Type: {n['type']}", f"Description: {n['description']}", "---"]
    output.append("======= END: TEXT NODES BLOCK =======\n")

    output.append("======= BEGIN: IMAGE NODES BLOCK =======")
    for n in image_nodes:
        if n["name"]:
            output += [f"Name: {n['name']}", "Type: image", f"Description: {n['description']}", "---"]
    output.append("======= END: IMAGE NODES BLOCK =======\n")

    output.append("======= BEGIN: TABLE NODES BLOCK =======")
    for n in table_nodes:
        if n["name"]:
            output += [f"Name: {n['name']}", "Type: table", f"Description: {n['description']}", "---"]
    output.append("======= END: TABLE NODES BLOCK =======\n")

    output.append("======= BEGIN: RELATIONSHIPS BLOCK =======")
    for src, tgt, edata in graph.edges(data=True):
        sn = graph.nodes[src]
        tn = graph.nodes[tgt]
        if sn.get("entity_name") and tn.get("entity_name"):
            output.append(f"Node 1 Name: {sn['entity_name']}")
            if sn.get("type") and sn["type"] != "unspecified":
                output.append(f"Node 1 Type: {sn['type']}")
            output.append(f"Node 2 Name: {tn['entity_name']}")
            if tn.get("type") and tn["type"] != "unspecified":
                output.append(f"Node 2 Type: {tn['type']}")
            if edata.get("description") and edata["description"] != "unspecified":
                output.append(f"Relationship between Node 1 and Node 2: {edata['description']}")
            output.append("----------")
    output.append("======= END: RELATIONSHIPS BLOCK =======")

    return "\n".join(output)


def _build_prompt(question: str, graphs_dir: Path, qid: Optional[str], use_graph: bool) -> str:
    graph_path = graphs_dir / f"{qid}_graph.graphml" if qid else None
    if use_graph and graph_path is not None and graph_path.exists():
        G = nx.read_graphml(graph_path)
        return LLM_ANSWER_PROMPT.replace("{question}", question).replace("{GraphML}", graph_to_str(G))
    return DIRECT_QA_PROMPT.replace("{question}", question)


# ── HuggingFace Gemma 4 E4B ──────────────────────────────────────────────────

_HF_ENV_MODEL_DIR = "GEMMA4_E4B_IT_MODEL_PATH"
_HF_DEFAULT_MODEL_DIR = _REPO_ROOT / "models" / "mllm" / "gemma-4-E4B-it"

# module-level cache (one process loads the model once)
_hf_model: Any = None
_hf_processor: Any = None
_hf_model_root: Optional[str] = None


def _resolve_hf_model_dir() -> Path:
    env = os.environ.get(_HF_ENV_MODEL_DIR, "").strip()
    if env:
        p = Path(env).expanduser().resolve()
        if not p.is_dir():
            raise FileNotFoundError(f"{_HF_ENV_MODEL_DIR} is set but not a directory: {p}")
        return p
    if _HF_DEFAULT_MODEL_DIR.is_dir():
        return _HF_DEFAULT_MODEL_DIR
    raise FileNotFoundError(
        f"Gemma 4 E4B IT weights not found at {_HF_DEFAULT_MODEL_DIR}. "
        f"Set {_HF_ENV_MODEL_DIR} or run util/download_models.py --only gemma."
    )


def _hf_configured() -> bool:
    try:
        _resolve_hf_model_dir()
        return True
    except FileNotFoundError:
        return False


def _load_hf() -> Tuple[Any, Any]:
    global _hf_model, _hf_processor, _hf_model_root
    root = _resolve_hf_model_dir()
    root_str = str(root)
    if _hf_model_root == root_str and _hf_model is not None:
        return _hf_model, _hf_processor

    import torch
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    _allow_cpu = os.getenv("GEMMA4_ALLOW_CPU", "0").strip().lower() in ("1", "true", "yes")
    if not torch.cuda.is_available() and not _allow_cpu:
        raise RuntimeError("CUDA GPU required to load Gemma 4 E4B IT.")

    _dt = os.getenv("GEMMA4_E4B_IT_TORCH_DTYPE", "").strip().lower()
    dtype = {"fp32": torch.float32, "float32": torch.float32,
             "fp16": torch.float16, "float16": torch.float16,
             "bf16": torch.bfloat16, "bfloat16": torch.bfloat16}.get(_dt, torch.float32)

    _attn = os.getenv("GEMMA4_ATTN_IMPLEMENTATION", "eager").strip()
    logger.info("Loading HF Gemma 4 E4B IT from %s (dtype=%s)", root, dtype)

    processor = AutoProcessor.from_pretrained(str(root), trust_remote_code=True)
    kwargs: Dict[str, Any] = {"dtype": dtype, "trust_remote_code": True, "attn_implementation": _attn}
    if torch.cuda.is_available():
        kwargs["device_map"] = {"": "cuda:0"}
    model = AutoModelForMultimodalLM.from_pretrained(str(root), **kwargs)

    _hf_model, _hf_processor, _hf_model_root = model, processor, root_str
    return model, processor


def _hf_generate(prompt: str, max_new_tokens: int = 512) -> str:
    import torch

    model, processor = _load_hf()
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    )
    dev = next(model.parameters()).device
    inputs = inputs.to(dev)
    input_len = int(inputs["input_ids"].shape[-1])
    with torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    row = out_ids[0][input_len:]
    response = processor.decode(row, skip_special_tokens=False)
    if hasattr(processor, "parse_response"):
        parsed = processor.parse_response(response)
        if isinstance(parsed, dict) and "content" in parsed:
            return str(parsed["content"])
    return response


class LLMService:
    """HuggingFace Gemma 4 E4B IT — live inference service."""

    def __init__(self, graphs_dir: Path) -> None:
        self._graphs_dir = graphs_dir

    @property
    def available(self) -> bool:
        return _hf_configured()

    def generate_answer(
        self,
        question: str,
        qid: Optional[str] = None,
        use_graph: bool = True,
    ) -> Optional[str]:
        if not self.available:
            return None
        try:
            prompt = _build_prompt(question, self._graphs_dir, qid, use_graph)
            answer = _hf_generate(prompt)
            return answer.strip() or None
        except Exception as exc:
            logger.warning("HF LLM generation failed: %s", exc)
            return None


# ── Ollama (gemma4:e2b / gemma4:e4b) ──────────────────────────────────────────

_OLLAMA_ENV_MODEL = "OLLAMA_GEMMA4_E2B_MODEL"
_OLLAMA_DEFAULT_MODEL = "gemma4:e2b"

_OLLAMA_E4B_ENV_MODEL = "OLLAMA_GEMMA4_E4B_MODEL"
_OLLAMA_E4B_DEFAULT_MODEL = "gemma4:e4b"


class _OllamaBackedLLMService:
    """Shared Ollama chat client; subclasses pick env key + default tag."""

    def __init__(
        self,
        graphs_dir: Path,
        *,
        env_model_var: str,
        default_model_tag: str,
    ) -> None:
        self._graphs_dir = graphs_dir
        self._env_model_var = env_model_var
        self._default_model_tag = default_model_tag

    def _model_tag(self) -> str:
        return os.getenv(self._env_model_var, "").strip() or self._default_model_tag

    @property
    def available(self) -> bool:
        try:
            from ollama import Client

            cli = Client()
            lst = cli.list()
            models = lst.models if lst else []
            name = self._model_tag()
            return any(getattr(x, "model", None) == name for x in models)
        except Exception:
            return False

    def generate_answer(
        self,
        question: str,
        qid: Optional[str] = None,
        use_graph: bool = True,
    ) -> Optional[str]:
        try:
            prompt = _build_prompt(question, self._graphs_dir, qid, use_graph)
            from ollama import Client

            cli = Client()
            out = cli.chat(
                model=self._model_tag(),
                messages=[{"role": "user", "content": prompt}],
                stream=False,
            )
            answer = (out.message.content or "").strip()
            return answer or None
        except Exception as exc:
            logger.warning("Ollama LLM generation failed (%s): %s", self._model_tag(), exc)
            return None


class OllamaLLMService(_OllamaBackedLLMService):
    """Ollama gemma4:e2b — live inference service."""

    def __init__(self, graphs_dir: Path) -> None:
        super().__init__(
            graphs_dir,
            env_model_var=_OLLAMA_ENV_MODEL,
            default_model_tag=_OLLAMA_DEFAULT_MODEL,
        )


class OllamaE4BLLMService(_OllamaBackedLLMService):
    """Ollama gemma4:e4b — live inference service."""

    def __init__(self, graphs_dir: Path) -> None:
        super().__init__(
            graphs_dir,
            env_model_var=_OLLAMA_E4B_ENV_MODEL,
            default_model_tag=_OLLAMA_E4B_DEFAULT_MODEL,
        )
