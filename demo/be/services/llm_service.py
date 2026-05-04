"""Service for live LLM inference using Gemma 4 E4B IT."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional

import networkx as nx

logger = logging.getLogger(__name__)

_PIPELINE_ROOT = Path(__file__).resolve().parents[3]
if str(_PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_ROOT))

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
    text_nodes = []
    image_nodes = []
    table_nodes = []

    for node_id, node_data in graph.nodes(data=True):
        node_info = {
            'id': node_id,
            'name': node_data.get('entity_name', ''),
            'type': node_data.get('type', ''),
            'description': node_data.get('description', ''),
        }
        if node_id.endswith('IMAGE'):
            image_nodes.append(node_info)
        elif node_id.endswith('TABLE'):
            table_nodes.append(node_info)
        else:
            text_nodes.append(node_info)

    output.append("======= BEGIN: TEXT NODES BLOCK =======")
    for node in text_nodes:
        if node['name'] and node['type']:
            output.append(f"Name: {node['name']}")
            output.append(f"Type: {node['type']}")
            output.append(f"Description: {node['description']}")
            output.append("---")
    output.append("======= END: TEXT NODES BLOCK =======")
    output.append("")

    output.append("======= BEGIN: IMAGE NODES BLOCK =======")
    for node in image_nodes:
        if node['name']:
            output.append(f"Name: {node['name']}")
            output.append("Type: image")
            output.append(f"Description: {node['description']}")
            output.append("---")
    output.append("======= END: IMAGE NODES BLOCK =======")
    output.append("")

    output.append("======= BEGIN: TABLE NODES BLOCK =======")
    for node in table_nodes:
        if node['name']:
            output.append(f"Name: {node['name']}")
            output.append("Type: table")
            output.append(f"Description: {node['description']}")
            output.append("---")
    output.append("======= END: TABLE NODES BLOCK =======")
    output.append("")

    output.append("======= BEGIN: RELATIONSHIPS BLOCK =======")
    for edge in graph.edges(data=True):
        source_node = graph.nodes[edge[0]]
        target_node = graph.nodes[edge[1]]
        edge_data = edge[2]
        if source_node.get('entity_name') and target_node.get('entity_name'):
            output.append(f"Node 1 Name: {source_node['entity_name']}")
            if source_node.get('type') and source_node.get('type') != 'unspecified':
                output.append(f"Node 1 Type: {source_node['type']}")
            output.append(f"Node 2 Name: {target_node['entity_name']}")
            if target_node.get('type') and target_node.get('type') != 'unspecified':
                output.append(f"Node 2 Type: {target_node['type']}")
            if edge_data.get('description') and edge_data.get('description') != 'unspecified':
                output.append(
                    f"Relationship between Node 1 and Node 2: {edge_data['description']}"
                )
            output.append("----------")
    output.append("======= END: RELATIONSHIPS BLOCK =======")

    return '\n'.join(output)


class LLMService:
    """Manages Gemma 4 E4B IT model for live inference."""

    def __init__(self, graphs_dir: Path) -> None:
        self._graphs_dir = graphs_dir
        self._gemma_module: Any = None
        self._loaded = False

    def _load_gemma(self) -> Any:
        """Lazy-load the Gemma module."""
        if self._gemma_module is not None:
            return self._gemma_module
        try:
            from mllm import hf_gemma_4_e4b_it as gemma_module
            if gemma_module.configured():
                self._gemma_module = gemma_module
                logger.info(
                    "Gemma 4 E4B IT loaded from: %s",
                    gemma_module.resolve_model_dir(),
                )
                self._loaded = True
                return gemma_module
            else:
                logger.warning("Gemma 4 E4B IT not configured (model dir not found)")
        except ImportError as e:
            logger.warning("Could not import mllm.hf_gemma_4_e4b_it: %s", e)
        return None

    @property
    def available(self) -> bool:
        """Check if LLM is available without loading it."""
        if self._loaded:
            return True
        try:
            from mllm import hf_gemma_4_e4b_it as gemma_module
            return gemma_module.configured()
        except ImportError:
            return False

    def generate_answer(
        self,
        question: str,
        qid: Optional[str] = None,
        use_graph: bool = True,
    ) -> Optional[str]:
        """
        Generate an answer for the given question.
        
        Args:
            question: The user's question text.
            qid: Question ID to look up the graph for.
            use_graph: Whether to use the graph context. Set False when the
                       question doesn't meaningfully match the graph content.
        """
        gemma = self._load_gemma()
        if gemma is None:
            return None

        graph_path = self._graphs_dir / f"{qid}_graph.graphml" if qid else None
        if use_graph and graph_path is not None and graph_path.exists():
            G = nx.read_graphml(graph_path)
            prompt = LLM_ANSWER_PROMPT.replace("{question}", question).replace(
                "{GraphML}", graph_to_str(G)
            )
        else:
            prompt = DIRECT_QA_PROMPT.replace("{question}", question)

        answer = gemma.generate_text(prompt, max_new_tokens=512)
        return (answer or "").strip() or None
