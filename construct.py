"""
Phase 4 — assemble per-question knowledge graphs (Construct).

Pipeline placement:
  Phase 1 (data): slice from export_*_slice as ``*_questions.jsonl``, etc.
  Phase 2: ``phase2_pattern_cache`` (not read directly here)
  Phase 3: ``phase3_extraction_cache/{qid}_{text_doc_id}.json`` ``response`` text
  Phase 4 (this module): merge caches + slice table/image/text meta into ``nx.Graph``
  Phase 5: ``inference.py`` loads ``*_graph.graphml`` produced here

Behavior:
  - For each question, read Phase 3 JSONs for ``text_doc_ids`` in metadata,
    parse ``response`` records with ``##`` / ``<|>`` rules, merge entities/relations into the graph.
  - Slice tables/images (captions/external txt) attach as TABLE / IMAGE auxiliary nodes.
  - Outputs: ``{qid}_graph.graphml`` plus ``{qid}_graph.models.json`` for lineage GraphML cannot carry cleanly.

(English reference) Each slice question loads Phase 3 ``{qid}_{text_doc_id}.json``, parses entity/relationship
records, merges tables/images, writes GraphML + JSON sidecar.
"""

import io
import numbers
import os
import hashlib
import json
from pathlib import Path
from typing import List, Dict, Any, Mapping
import html
import re
import networkx as nx
import pandas as pd
import urllib.parse

from util.result_layout import resolve_pipeline_run_id

# Matches Phase 3 extraction.py output: record boundary ``##``, field delimiter ``<|>``.
record_delimiter = "##"
tuple_delimiter = "<|>"
join_descriptions_flag = True


def clean_str(input: Any) -> str:
    """Light normalization for node/edge attrs (strip HTML escapes, control chars)."""
    if not isinstance(input, str):
        return input

    result = html.unescape(input.strip())
    return re.sub(r"[\x00-\x1f\x7f-\x9f]", "", result)
def _unpack_descriptions(data: Mapping) -> list[str]:
    """Join multi-line accumulated descriptions on edges/nodes."""
    value = data.get("description", None)
    return [] if value is None else value.split("\n")


def _unpack_source_ids(data: Mapping) -> list[str]:
    """Parse provenance snippet id list from comma-separated ``source_id`` string."""
    value = data.get("source_id", None)
    return [] if value is None else value.split(", ")


def _canonical_doc_fragment(owner_qid: str, raw_key: Any) -> str:
    """Strip optional ``{qid}_`` prefix on cache keys for MMQA etc. to align with gold ``doc_id``."""
    if isinstance(raw_key, str):
        base = clean_str(raw_key)
    else:
        base = str(raw_key or "").strip()
    if owner_qid and base.startswith(owner_qid + "_"):
        return base[len(owner_qid) + 1 :].strip()
    return base.strip()
def _merge_doc_id_attr(existing: Any, new_fragment: str) -> str:
    """Merge ``doc_id`` strings as sorted union when multiple doc provenances attach."""
    parts: set[str] = set()
    if isinstance(existing, str) and existing.strip():
        for p in existing.split(", "):
            t = p.strip()
            if t:
                parts.add(t)
    if new_fragment.strip():
        parts.add(new_fragment.strip())
    return ", ".join(sorted(parts))
def load_jsonl_data(path):
    """Load entire slice JSONL (one JSON object per line)."""
    with open(path, "r", encoding='UTF-8') as file:
        return [json.loads(line) for line in file]
def extract_entity_by_wikiurl(url):
    """Human-readable entity label from image title URL tail (underscores → spaces)."""
    path = urllib.parse.urlparse(url).path
    entity = path.split('/')[-1]
    entity = urllib.parse.unquote(entity)
    entity = entity.replace('_', ' ')
    return entity
def table_to_markdown(table):
    """Markdown string for graph node ``description`` from slice table structure."""
    markdown = ""
    table = table['table']
    header = table['header']
    markdown += "| " + " | ".join(col['column_name'] for col in header) + " |\n"
    markdown += "|" + "---|" * len(header) + "\n"
    for row in table['table_rows']:
        markdown += "| " + " | ".join(cell['text'] for cell in row) + " |\n"
    return markdown.strip()


def construct_graph(text_answers, table, question, texts, images, owner_qid: str = ""):
    """Phase 3 cache rows + slice media hints → one undirected ``nx.Graph``.

    - ``text_answers``: split each ``response`` by ``##`` / ``<|>`` back into entity/relation tuples.
    - Merge descriptions, ``source_id``, ``doc_id`` for duplicate entities.
    - With ``table`` / ``images``, add TABLE·IMAGE scaffolding nodes and title anchors.
    - Internal node keys: ``NAME_UPPER + " Bt: " + TYPE`` to reduce collisions across sources.
    """
    graph = nx.Graph()

    entity_dict = {}
    for result in text_answers:
        source_doc_id = result.get('id', result.get('qid', 'unknown_source'))
        extracted_data = result['response']
        records = [r.strip() for r in extracted_data.split(record_delimiter)]

        for record in records:
            record = re.sub(r"^\(|\)$", "", record.strip())
            record_attributes = record.split(tuple_delimiter)

            # Treat as entity if first field is not "entity" but >=4 fields (tolerate LLM variants).
            first_field = record_attributes[0].strip('"').lower()
            is_entity_record = (
                first_field == "entity" or
                (first_field not in ("relationship", "relation") and len(record_attributes) >= 4)
            )
            if is_entity_record and len(record_attributes) >= 4:
                # Entity record → add graph node or merge attrs into existing node
                entity_name = clean_str(record_attributes[1]).strip('"')
                entity_type = clean_str(record_attributes[2]).strip('"')
                entity_description = clean_str(record_attributes[3]).strip('"')
                entity_type_upper = entity_type.upper()
                entity_name_upper = entity_name.upper()
                entity_uuid = entity_name_upper + " Bt: " + entity_type_upper
                if entity_uuid in graph.nodes():
                    node = graph.nodes[entity_uuid]
                    node["description"] = "\n".join(
                        list({
                            *_unpack_descriptions(node),
                            entity_description,
                        })
                    )
                    node["source_id"] = ", ".join(
                        list({
                            *_unpack_source_ids(node),
                            str(source_doc_id),
                        })
                    )
                    node["doc_id"] = _merge_doc_id_attr(
                        node.get("doc_id"), _canonical_doc_fragment(owner_qid, source_doc_id)
                    )
                else:
                    entity_dict[entity_name_upper] = entity_type_upper
                    graph.add_node(
                        entity_uuid,
                        entity_name=entity_name,
                        type=entity_type,
                        description=entity_description,
                        source_id=str(source_doc_id),
                        doc_id=_canonical_doc_fragment(owner_qid, source_doc_id),
                    )

            if (
                    record_attributes[0] == '"relationship"'
                    and len(record_attributes) >= 4
            ):
                source_entity_name = clean_str(record_attributes[1]).strip('"')
                source_entity_name_upper = source_entity_name.upper()
                target_entity_name = clean_str(record_attributes[2]).strip('"')
                target_entity_name_upper = target_entity_name.upper()
                edge_description = clean_str(record_attributes[3]).strip('"')
                edge_source_id = clean_str(str(source_doc_id))
                edge_doc_id = _canonical_doc_fragment(owner_qid, source_doc_id)
                weight = 1.0
                if source_entity_name_upper not in entity_dict:
                    entity_dict[source_entity_name_upper] = ""
                    source_entity_uuid = source_entity_name_upper + " Bt: "
                    if source_entity_uuid not in graph.nodes():
                        graph.add_node(
                            source_entity_uuid,
                            entity_name=source_entity_name,
                            type="",
                            description="",
                            source_id=edge_source_id,
                            doc_id=edge_doc_id,
                        )
                else:
                    source_entity_uuid = source_entity_name_upper + " Bt: " + entity_dict[source_entity_name_upper]
                if target_entity_name_upper not in entity_dict:
                    entity_dict[target_entity_name_upper] = ""
                    target_entity_uuid = target_entity_name_upper + " Bt: "
                    graph.add_node(
                        target_entity_uuid,
                        entity_name=target_entity_name,
                        type="",
                        description="",
                        source_id=edge_source_id,
                        doc_id=edge_doc_id,
                    )
                else:
                    target_entity_uuid = target_entity_name_upper + " Bt: " + entity_dict[target_entity_name_upper]

                if graph.has_edge(source_entity_uuid, target_entity_uuid):
                    edge_data = graph.get_edge_data(source_entity_uuid, target_entity_uuid)
                    if edge_data is not None:
                        weight += edge_data["weight"]
                        if join_descriptions_flag:
                            edge_description = "\n".join(
                                list({
                                    *_unpack_descriptions(edge_data),
                                    edge_description,
                                })
                            )
                        edge_source_id = ", ".join(
                            list({
                                *_unpack_source_ids(edge_data),
                                str(source_doc_id),
                            })
                        )
                        edge_doc_id = _merge_doc_id_attr(
                            edge_data.get("doc_id"), _canonical_doc_fragment(owner_qid, source_doc_id)
                        )
                graph.add_edge(
                    source_entity_uuid,
                    target_entity_uuid,
                    weight=weight,
                    description=edge_description,
                    source_id=edge_source_id,
                    doc_id=edge_doc_id,
                )
    if table is not None:
        markdown_table = table_to_markdown(table)
        table_title = table['title']
        table_title_upper = table_title.upper()
        if table_title_upper not in entity_dict:
            entity_dict[table_title_upper] = ""
            graph.add_node(
                table_title_upper + " Bt: ",
                entity_name=table_title,
                type="",
                description="",
                source_id=table['id'],
                doc_id=str(table['id']),
            )
            graph.add_node(
                table_title_upper + " " + table['table']['table_name'].upper() + " Bt: TABLE",
                entity_name=table['title'] + " " + table['table']['table_name'],
                type="TABLE",
                description=markdown_table,
                source_id=table['id'],
                doc_id=str(table['id']),
            )
            graph.add_edge(
                table_title_upper + " Bt: ",
                table_title_upper + " " + table['table']['table_name'].upper() + " Bt: TABLE",
                entity_name=table['title'] + " " + table['table']['table_name'],
                weight=1,
                description=table['title'] + " " + table['table']['table_name'] + " table",
                source_id=table['id'],
                doc_id=str(table['id']),
            )
        else:
            graph.add_node(
                table_title_upper + " " + table['table']['table_name'].upper() + " Bt: TABLE",
                entity_name=table['title'] + " " + table['table']['table_name'],
                type="TABLE",
                description=markdown_table,
                source_id=table['id'],
                doc_id=str(table['id']),
            )
            graph.add_edge(
                table_title_upper + " Bt: " + entity_dict[table_title_upper],
                table_title_upper + " " + table['table']['table_name'].upper() + " Bt: TABLE",
                entity_name=table['title'] + " " + table['table']['table_name'],
                weight=1,
                description=table['title'] + " " + table['table']['table_name'] + " table",
                source_id=table['id'],
                doc_id=str(table['id']),
            )
    # Images: add IMAGE nodes + anchor edges only when caption/description exists
    for image in images:
        img_id = image.get("id", image.get("image_id", ""))
        image_description = (
            (image.get("caption") or image.get("description") or "").strip()
        )
        if not image_description and os.getenv("WEBQA_USE_IMAGE_CAPTION", "1") != "0":
            image_description = (image.get("caption") or "").strip()
        if not image_description:
            desc_dir = os.getenv("WEBQA_IMAGE_DESC_DIR", "").strip()
            _mmqa_default = str(Path(__file__).resolve().parent / "data" / "multimodalqa" / "image_descriptions")
            legacy = os.getenv("MMQA_IMAGE_DESC_DIR", _mmqa_default).strip()
            for base in (desc_dir, legacy):
                if not base:
                    continue
                try:
                    p = os.path.join(base, str(img_id) + ".txt")
                    with open(p, "r", encoding="utf-8") as file:
                        image_description = file.read()
                    break
                except OSError:
                    continue
        if not image_description:
            continue
        image_entity_name = extract_entity_by_wikiurl(image['title'])
        image_entity_name_upper = image_entity_name.upper()
        if image_entity_name_upper not in entity_dict:
            entity_dict[image_entity_name_upper] = ""
            graph.add_node(
                image_entity_name_upper + " Bt: ",
                entity_name=image_entity_name,
                type="",
                description="",
                source_id=img_id,
                doc_id=str(img_id),
            )
            graph.add_node(
                image_entity_name_upper + " Bt: IMAGE",
                entity_name=image_entity_name if image['title'] == image_entity_name else f"{image_entity_name} {image['title']}",
                type="IMAGE",
                description=image_description,
                source_id=img_id,
                doc_id=str(img_id),
            )
            graph.add_edge(
                image_entity_name_upper + " Bt: ",
                image_entity_name_upper + " Bt: IMAGE",
                weight=1,
                description=image['title'] + " 's picture",
            )
        else:
            graph.add_node(
                image_entity_name_upper + " Bt: IMAGE",
                entity_name=image_entity_name if image['title'] == image_entity_name else f"{image_entity_name} {image['title']}",
                type="IMAGE",
                description=image_description,
                source_id=img_id,
                doc_id=str(img_id),
            )
            graph.add_edge(
                image_entity_name_upper + " Bt: " + entity_dict[image_entity_name_upper],
                image_entity_name_upper + " Bt: IMAGE",
                weight=1,
                description=image_entity_name + " 's picture",
            )
    return graph


def graph_to_graphml_str(graph):
    """Tests/dumps: GraphML bytes as UTF-8 string; CLI writes via ``nx.write_graphml`` directly."""
    with io.BytesIO() as byte_output:
        nx.write_graphml(graph, byte_output)
        byte_output.seek(0)
        graphml_str = byte_output.read().decode('utf-8')
    return graphml_str


# Example outputs: ``<CONSTRUCT_OUTPUT_GRAPH_DIR>/<qid>_graph.graphml`` + ``*_graph.models.json`` (Phase 3 llm rollup)


def main():
    """Resolve slice/cache paths from env, write GraphML + models sidecar for capped question count."""

    base_dir = Path(__file__).resolve().parent
    _dataset = os.getenv("MMGRAPHRAG_DATASET", "webqa").strip().lower()
    run_id = resolve_pipeline_run_id(base_dir, _dataset)
    # Slice root: prefer CONSTRUCT_SLICE_DIR; CONSTRUCT_WEBQA_SLICE_DIR is legacy compat
    slice_dir = Path(os.getenv(
        "CONSTRUCT_SLICE_DIR",
        os.getenv("CONSTRUCT_WEBQA_SLICE_DIR", str(base_dir / "result" / run_id / f"{_dataset}_slice")),
    ))
    question_file = os.getenv(
        "CONSTRUCT_QUESTION_FILE",
        str(slice_dir / f"{_dataset}_questions.jsonl"),
    )
    table_file = os.getenv(
        "CONSTRUCT_TABLE_FILE",
        str(slice_dir / f"{_dataset}_tables.jsonl"),
    )
    image_file = os.getenv(
        "CONSTRUCT_IMAGE_FILE",
        str(slice_dir / f"{_dataset}_images.jsonl"),
    )
    text_file = os.getenv(
        "CONSTRUCT_TEXT_FILE",
        str(slice_dir / f"{_dataset}_texts.jsonl"),
    )
    answer_text_cache = os.getenv(
        "CONSTRUCT_EXTRACTION_CACHE",
        str(base_dir / "result" / run_id / "phase3_extraction_cache"),
    )
    # Notebooks often use ``phase4_graphs_out``; CLI default ``phase4_graphs_real`` — override via env when aligning paths
    output_graph_dir = os.getenv(
        "CONSTRUCT_OUTPUT_GRAPH_DIR",
        str(base_dir / "result" / run_id / "phase4_graphs_real"),
    )
    max_questions = int(os.getenv("CONSTRUCT_MAX_QUESTIONS", "20"))
    os.makedirs(output_graph_dir, exist_ok=True)
    questiones = load_jsonl_data(question_file)
    if max_questions > 0:
        questiones = questiones[:max_questions]
    tables = load_jsonl_data(table_file)
    tables = {table.get('id', table.get('table_id', '')): table for table in tables}
    images = load_jsonl_data(image_file)
    # Image rows: accept either ``id`` or ``image_id``
    images = {img.get('id', img.get('image_id', '')): img for img in images}
    texts = load_jsonl_data(text_file)
    texts = {text.get('id', text.get('snippet_id', '')): text for text in texts}
    # One GraphML per question + upstream LLM lineage JSON (only rows remaining after CONSTRUCT_MAX_QUESTIONS cap)
    for question in questiones:
        qid = question.get("qid") or question.get("Guid") or ""
        md = question.get("metadata") or {}
        text_doc_ids = md.get("text_doc_ids") or []
        image_doc_ids = md.get("image_doc_ids") or []
        table_id = md.get("table_id", "webqa_dummy_table")
        text_results = []
        upstream_llm_tiers: List[Dict[str, Any]] = []
        for text_doc_id in text_doc_ids:
            cache_path = os.path.join(answer_text_cache, f"{qid}_{text_doc_id}.json")
            if os.path.exists(cache_path):
                with open(cache_path, "r", encoding="utf-8") as file:
                    cache_entry = json.load(file)
                text_results.append(cache_entry)
                # Phase 3 extraction.py ``llm`` meta (tier/model/url); older caches omit fields → legacy tier
                llm_entry = cache_entry.get("llm")
                if isinstance(llm_entry, dict):
                    upstream_llm_tiers.append(
                        {
                            "text_doc_id": text_doc_id,
                            "tier": llm_entry.get("tier", "unknown"),
                            "model": llm_entry.get("model", "(unknown)"),
                            "base_url": llm_entry.get("base_url", "(unknown)"),
                        }
                    )
                else:
                    upstream_llm_tiers.append(
                        {
                            "text_doc_id": text_doc_id,
                            "tier": "legacy",
                            "model": "(unknown)",
                            "base_url": "(unknown)",
                        }
                    )
        table = tables.get(table_id)
        img_list = [images[i] for i in image_doc_ids if i in images]
        q_graph = construct_graph(text_results, table, question, texts, img_list, owner_qid=qid)
        print(f"Graph has {q_graph.number_of_nodes()} nodes and {q_graph.number_of_edges()} edges")
        out_path = os.path.join(output_graph_dir, f"{qid}_graph.graphml")
        nx.write_graphml(q_graph, out_path)

        # Keep lineage out of GraphML schema; aggregate into sibling ``.models.json``
        model_counts: Dict[str, int] = {}
        tier_counts: Dict[str, int] = {}
        for entry in upstream_llm_tiers:
            model_counts[entry["model"]] = model_counts.get(entry["model"], 0) + 1
            tier_counts[entry["tier"]] = tier_counts.get(entry["tier"], 0) + 1
        sidecar_path = os.path.join(output_graph_dir, f"{qid}_graph.models.json")
        with open(sidecar_path, "w", encoding="utf-8") as sc:
            json.dump(
                {
                    "qid": qid,
                    "phase": "4_construct",
                    "upstream_extraction_caches": upstream_llm_tiers,
                    "summary": {"by_model": model_counts, "by_tier": tier_counts},
                },
                sc,
                ensure_ascii=False,
                indent=2,
            )


if __name__ == "__main__":
    from pathlib import Path

    from util.pipeline_session_log import run_with_session_stdio_tee

    run_with_session_stdio_tee(Path(__file__).resolve().parent, "construct", main)
