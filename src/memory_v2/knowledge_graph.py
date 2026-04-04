"""
memory-v2 -- Knowledge Graph (NetworkX)
Entity/relationship extraction, Leiden community detection,
HippoRAG-style Personalized PageRank retrieval.
"""

import json
import math
import os
import pickle
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np
import ollama

from . import db, embeddings

GRAPH_PATH = Path(os.environ.get(
    "MEMORY_V2_GRAPH",
    str(Path.home() / ".memory-v2" / "knowledge_graph.pickle")
))

EXTRACTION_MODEL = os.environ.get("MEMORY_V2_LLM_MODEL", "qwen2.5:3b")

# ---------- Schema ----------

NODE_TYPES = [
    "person", "project", "concept", "decision", "tool",
    "event", "emotion", "conversation", "chunk", "community",
]

EDGE_TYPES = [
    "discussed_in", "decided", "built", "uses", "part_of",
    "related_to", "preceded_by", "caused", "felt",
    "evolved_from", "member_of",
]

# ---------- Entity/Relationship Extraction ----------

EXTRACT_ENTITIES_PROMPT = """Extract entities and relationships from this text.

Entity types: person, project, concept, decision, tool, event, emotion
Relationship types: discussed_in, decided, built, uses, part_of, related_to, preceded_by, caused, felt, evolved_from

Rules:
- Normalize entity names to lowercase
- Each entity needs: name, type
- Each relationship needs: source, target, type, description
- Focus on meaningful relationships, skip trivial ones
- Keep entity names canonical (e.g., "myproject" not "the MyProject system")

Return ONLY a JSON object:
{{
  "entities": [{{"name": "...", "type": "...", "properties": {{}}}}, ...],
  "relationships": [{{"source": "...", "target": "...", "type": "...", "description": "..."}}, ...]
}}

Text:
{text}"""


def extract_entities_and_relations(text: str, source_file: str = "") -> dict:
    """Extract entities and relationships from text using local LLM."""
    prompt = EXTRACT_ENTITIES_PROMPT.format(text=text[:4000])

    try:
        resp = ollama.chat(
            model=EXTRACTION_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.1, "num_predict": 2000},
        )
        content = resp.message.content.strip()

        # Parse JSON
        if "```" in content:
            match = re.search(r'```(?:json)?\s*(.*?)```', content, re.DOTALL)
            content = match.group(1) if match else "{}"

        data = json.loads(content)
        return {
            "entities": data.get("entities", []),
            "relationships": data.get("relationships", []),
        }
    except (json.JSONDecodeError, Exception):
        return {"entities": [], "relationships": []}


# ---------- Graph Management ----------

def load_graph() -> nx.DiGraph:
    """Load the knowledge graph from pickle, or create empty."""
    if GRAPH_PATH.exists():
        with open(GRAPH_PATH, "rb") as f:
            return pickle.load(f)
    return nx.DiGraph()


def save_graph(G: nx.DiGraph):
    """Save the knowledge graph to pickle."""
    GRAPH_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(GRAPH_PATH, "wb") as f:
        pickle.dump(G, f)


def add_entities_to_graph(
    G: nx.DiGraph,
    entities: list[dict],
    relationships: list[dict],
    source_file: str = "",
    timestamp: str | None = None,
) -> dict:
    """Add extracted entities and relationships to the graph."""
    if timestamp is None:
        timestamp = datetime.now(timezone.utc).isoformat()

    stats = {"nodes_added": 0, "edges_added": 0, "nodes_merged": 0}

    # Add/merge entities
    for entity in entities:
        name = entity.get("name", "").lower().strip()
        etype = entity.get("type", "concept").lower()
        if not name or len(name) < 2:
            continue

        if etype not in NODE_TYPES:
            etype = "concept"

        if G.has_node(name):
            # Merge: update properties
            G.nodes[name].setdefault("sources", []).append(source_file)
            G.nodes[name]["last_seen"] = timestamp
            G.nodes[name]["mention_count"] = G.nodes[name].get("mention_count", 0) + 1
            stats["nodes_merged"] += 1
        else:
            G.add_node(
                name,
                type=etype,
                properties=entity.get("properties", {}),
                sources=[source_file],
                created_at=timestamp,
                last_seen=timestamp,
                mention_count=1,
            )
            stats["nodes_added"] += 1

    # Add relationships
    for rel in relationships:
        source = rel.get("source", "").lower().strip()
        target = rel.get("target", "").lower().strip()
        rtype = rel.get("type", "related_to").lower()

        if not source or not target:
            continue
        if rtype not in EDGE_TYPES:
            rtype = "related_to"

        # Ensure nodes exist
        for node in [source, target]:
            if not G.has_node(node):
                G.add_node(node, type="concept", sources=[source_file],
                           created_at=timestamp, last_seen=timestamp, mention_count=1)

        if G.has_edge(source, target):
            G.edges[source, target]["weight"] = G.edges[source, target].get("weight", 1) + 1
            G.edges[source, target]["last_seen"] = timestamp
        else:
            G.add_edge(
                source, target,
                type=rtype,
                description=rel.get("description", ""),
                source_file=source_file,
                created_at=timestamp,
                last_seen=timestamp,
                weight=1,
                valid_from=timestamp,
                valid_until=None,
                confidence=0.8,
            )
            stats["edges_added"] += 1

    return stats


# ---------- Build Graph from Vault ----------

def build_graph_from_vault(
    conn=None,
    vault_path: Path | None = None,
    progress_callback=None,
) -> dict:
    """Build the knowledge graph from all vault markdown files."""
    from .vault_indexer import VAULT_PATH, _get_vault_files

    vault_path = vault_path or VAULT_PATH
    files = _get_vault_files(vault_path)
    G = load_graph()

    stats = {"files_processed": 0, "total_nodes": 0, "total_edges": 0, "errors": []}

    for i, fpath in enumerate(files):
        rel_path = str(fpath.relative_to(vault_path)).replace("\\", "/")
        if progress_callback:
            progress_callback(i + 1, len(files), rel_path)

        try:
            text = fpath.read_text(encoding="utf-8", errors="replace")
            # Use first 4000 chars for extraction (LLM context limit)
            result = extract_entities_and_relations(text[:4000], rel_path)
            add_entities_to_graph(G, result["entities"], result["relationships"], rel_path)
            stats["files_processed"] += 1
        except Exception as e:
            stats["errors"].append(f"{rel_path}: {e}")

    save_graph(G)
    stats["total_nodes"] = G.number_of_nodes()
    stats["total_edges"] = G.number_of_edges()
    return stats


# ---------- Community Detection ----------

def detect_communities(G: nx.DiGraph | None = None, resolution: float = 1.0) -> dict:
    """
    Run Leiden community detection on the knowledge graph.
    Returns community assignments and summaries.
    """
    if G is None:
        G = load_graph()

    if G.number_of_nodes() == 0:
        return {"communities": [], "error": "Graph is empty"}

    try:
        import igraph as ig
        import leidenalg

        # Convert NetworkX to igraph (undirected for community detection)
        G_undirected = G.to_undirected()
        node_list = list(G_undirected.nodes())
        edge_list = [(node_list.index(u), node_list.index(v))
                     for u, v in G_undirected.edges()
                     if u in node_list and v in node_list]

        ig_graph = ig.Graph(n=len(node_list), edges=edge_list)

        # Run Leiden
        partition = leidenalg.find_partition(
            ig_graph,
            leidenalg.ModularityVertexPartition,
        )

        # Build community assignments
        communities = {}
        for node_idx, comm_id in enumerate(partition.membership):
            node_name = node_list[node_idx]
            if comm_id not in communities:
                communities[comm_id] = {
                    "id": comm_id,
                    "members": [],
                    "types": {},
                    "size": 0,
                }
            communities[comm_id]["members"].append(node_name)
            communities[comm_id]["size"] += 1

            node_type = G.nodes[node_name].get("type", "unknown")
            communities[comm_id]["types"][node_type] = \
                communities[comm_id]["types"].get(node_type, 0) + 1

        # Generate summaries for top communities
        sorted_comms = sorted(communities.values(), key=lambda c: c["size"], reverse=True)
        for comm in sorted_comms[:20]:
            members_str = ", ".join(comm["members"][:10])
            comm["summary"] = f"Cluster of {comm['size']} entities: {members_str}"

            # Tag nodes with community
            for member in comm["members"]:
                G.nodes[member]["community"] = comm["id"]

        save_graph(G)

        return {
            "num_communities": len(communities),
            "communities": sorted_comms[:20],
            "modularity": partition.modularity,
        }

    except ImportError as e:
        return {"error": f"Missing dependency: {e}. Install with: pip install memory-v2[graph]"}
    except Exception as e:
        return {"error": str(e)}


# ---------- HippoRAG-Style PPR Retrieval ----------

def ppr_search(
    query: str,
    G: nx.DiGraph | None = None,
    top_k: int = 10,
    alpha: float = 0.85,
) -> list[dict]:
    """
    Personalized PageRank retrieval:
    1. Extract entities from query
    2. Find matching nodes in graph
    3. Run PPR from matched nodes
    4. Return high-scoring nodes (multi-hop discovery)
    """
    if G is None:
        G = load_graph()

    if G.number_of_nodes() == 0:
        return []

    # Extract query entities (simple: split into words, match to graph nodes)
    query_words = set(w.lower() for w in re.split(r'\W+', query) if len(w) > 2)
    seed_nodes = []
    for node in G.nodes():
        node_words = set(w.lower() for w in re.split(r'\W+', node) if len(w) > 2)
        if query_words & node_words:
            seed_nodes.append(node)

    if not seed_nodes:
        # Fallback: try embedding similarity to find seed nodes
        try:
            query_emb = embeddings.embed_text(query)
            # Score each node by name similarity
            node_names = list(G.nodes())
            if len(node_names) > 100:
                # Sample for efficiency
                node_names = node_names[:100]
            node_embs = embeddings.embed_batch(node_names)
            from .scoring import cosine_similarity
            scores = [(name, cosine_similarity(query_emb, emb))
                      for name, emb in zip(node_names, node_embs)]
            scores.sort(key=lambda x: x[1], reverse=True)
            seed_nodes = [name for name, score in scores[:5] if score > 0.3]
        except Exception:
            return []

    if not seed_nodes:
        return []

    # Build personalization dict (uniform weight on seed nodes)
    personalization = {node: 0.0 for node in G.nodes()}
    weight = 1.0 / len(seed_nodes)
    for node in seed_nodes:
        if node in personalization:
            personalization[node] = weight

    # Run PageRank
    try:
        G_undirected = G.to_undirected()
        pr = nx.pagerank(G_undirected, alpha=alpha, personalization=personalization)
    except Exception:
        return []

    # Sort and return top-K (excluding seed nodes for discovery)
    results = []
    for node, score in sorted(pr.items(), key=lambda x: x[1], reverse=True):
        if len(results) >= top_k:
            break
        node_data = dict(G.nodes[node])
        results.append({
            "node": node,
            "ppr_score": round(score, 6),
            "type": node_data.get("type", "unknown"),
            "community": node_data.get("community"),
            "mention_count": node_data.get("mention_count", 0),
            "sources": node_data.get("sources", [])[:3],
            "is_seed": node in seed_nodes,
        })

    return results


# ---------- Get communities for MCP ----------

def get_communities(limit: int = 20) -> list[dict]:
    """Get community summaries for the list_topics MCP tool."""
    G = load_graph()
    if G.number_of_nodes() == 0:
        return []

    # Check if communities are already detected
    has_communities = any("community" in G.nodes[n] for n in G.nodes())
    if not has_communities:
        result = detect_communities(G)
        if "error" in result:
            return []
        return result.get("communities", [])[:limit]

    # Aggregate existing community assignments
    communities = {}
    for node in G.nodes():
        comm_id = G.nodes[node].get("community")
        if comm_id is not None:
            if comm_id not in communities:
                communities[comm_id] = {"id": comm_id, "members": [], "size": 0}
            communities[comm_id]["members"].append(node)
            communities[comm_id]["size"] += 1

    sorted_comms = sorted(communities.values(), key=lambda c: c["size"], reverse=True)
    for comm in sorted_comms[:limit]:
        members_str = ", ".join(comm["members"][:10])
        comm["summary"] = f"Cluster of {comm['size']} entities: {members_str}"

    return sorted_comms[:limit]


# ---------- Visualization ----------

def visualize_graph(
    G: nx.DiGraph | None = None,
    output_path: str = "graph.html",
    max_nodes: int = 500,
) -> str:
    """Generate an interactive PyVis visualization of the knowledge graph."""
    from pyvis.network import Network

    if G is None:
        G = load_graph()

    if G.number_of_nodes() == 0:
        return "Graph is empty"

    # Color mapping for node types
    colors = {
        "person": "#FF6B6B",
        "project": "#4ECDC4",
        "concept": "#45B7D1",
        "decision": "#FFA07A",
        "tool": "#98D8C8",
        "event": "#F7DC6F",
        "emotion": "#DDA0DD",
        "conversation": "#87CEEB",
        "chunk": "#D3D3D3",
        "community": "#FFD700",
    }

    # If graph is too large, take top nodes by mention count
    if G.number_of_nodes() > max_nodes:
        top_nodes = sorted(
            G.nodes(),
            key=lambda n: G.nodes[n].get("mention_count", 0),
            reverse=True,
        )[:max_nodes]
        G = G.subgraph(top_nodes).copy()

    net = Network(
        height="800px", width="100%",
        bgcolor="#1a1a2e", font_color="white",
        directed=True,
    )

    # Add nodes
    for node in G.nodes():
        ndata = G.nodes[node]
        ntype = ndata.get("type", "concept")
        color = colors.get(ntype, "#FFFFFF")
        size = min(10 + ndata.get("mention_count", 1) * 3, 50)
        title = f"{node}\nType: {ntype}\nMentions: {ndata.get('mention_count', 0)}"
        if ndata.get("community") is not None:
            title += f"\nCommunity: {ndata['community']}"
        net.add_node(node, label=node, color=color, size=size, title=title)

    # Add edges
    for u, v in G.edges():
        edata = G.edges[u, v]
        weight = edata.get("weight", 1)
        etype = edata.get("type", "related_to")
        net.add_edge(u, v, title=etype, value=weight, color="#555555")

    net.toggle_physics(True)
    net.save_graph(output_path)
    return output_path


# ---------- Graph Stats ----------

def graph_stats(G: nx.DiGraph | None = None) -> dict:
    """Get knowledge graph statistics."""
    if G is None:
        G = load_graph()

    if G.number_of_nodes() == 0:
        return {"nodes": 0, "edges": 0, "status": "empty"}

    type_counts = {}
    for node in G.nodes():
        ntype = G.nodes[node].get("type", "unknown")
        type_counts[ntype] = type_counts.get(ntype, 0) + 1

    edge_type_counts = {}
    for u, v in G.edges():
        etype = G.edges[u, v].get("type", "unknown")
        edge_type_counts[etype] = edge_type_counts.get(etype, 0) + 1

    # Top nodes by mention count
    top_nodes = sorted(
        G.nodes(),
        key=lambda n: G.nodes[n].get("mention_count", 0),
        reverse=True,
    )[:10]

    return {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "node_types": type_counts,
        "edge_types": edge_type_counts,
        "top_nodes": [
            {"name": n, "mentions": G.nodes[n].get("mention_count", 0),
             "type": G.nodes[n].get("type", "unknown")}
            for n in top_nodes
        ],
        "density": round(nx.density(G), 6),
        "connected_components": nx.number_weakly_connected_components(G),
    }
