"""
memory-v2 -- Knowledge Graph Builder (standalone)
Run this once to build the full graph, then incremental updates are fast.

Usage:
    memory-v2-build-kg          # incremental
    memory-v2-build-kg --full   # rebuild all
"""

import sys
import time
import hashlib
import json
import os
from pathlib import Path

from .knowledge_graph import (
    extract_entities_and_relations, load_graph, save_graph,
    add_entities_to_graph, detect_communities, visualize_graph, graph_stats,
)
from .vault_indexer import VAULT_PATH, _get_vault_files

KG_HASHES_FILE = Path(os.environ.get(
    "MEMORY_V2_KG_HASHES",
    str(Path.home() / ".memory-v2" / "kg_file_hashes.json")
))


def load_kg_hashes() -> dict:
    if KG_HASHES_FILE.exists():
        return json.loads(KG_HASHES_FILE.read_text())
    return {}


def save_kg_hashes(hashes: dict):
    KG_HASHES_FILE.parent.mkdir(parents=True, exist_ok=True)
    KG_HASHES_FILE.write_text(json.dumps(hashes, indent=2))


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:32]


def main():
    force = "--full" in sys.argv
    print("memory-v2 -- Knowledge Graph Builder")
    print(f"Vault: {VAULT_PATH}")
    print(f"Mode: {'FULL REBUILD' if force else 'INCREMENTAL'}")
    print()

    files = _get_vault_files(VAULT_PATH)
    hashes = {} if force else load_kg_hashes()
    G = load_graph()

    processed = 0
    skipped = 0
    errors = 0
    start = time.time()

    for i, fpath in enumerate(files):
        rel_path = str(fpath.relative_to(VAULT_PATH)).replace("\\", "/")
        current_hash = file_hash(fpath)

        # Skip unchanged files
        if not force and hashes.get(rel_path) == current_hash:
            skipped += 1
            continue

        try:
            text = fpath.read_text(encoding="utf-8", errors="replace")
            elapsed_file = time.time()
            result = extract_entities_and_relations(text[:4000], rel_path)
            elapsed_file = time.time() - elapsed_file

            stats = add_entities_to_graph(
                G, result["entities"], result["relationships"], rel_path
            )

            hashes[rel_path] = current_hash
            processed += 1

            print(
                f"  [{processed:3d}] {elapsed_file:5.1f}s "
                f"+{stats['nodes_added']}n +{stats['edges_added']}e "
                f"| {rel_path[:70]}"
            )

            # Save progress every 25 files
            if processed % 25 == 0:
                save_graph(G)
                save_kg_hashes(hashes)
                print(f"       [checkpoint: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges]")

        except Exception as e:
            errors += 1
            print(f"  [ERR] {rel_path}: {e}")

    # Final save
    save_graph(G)
    save_kg_hashes(hashes)

    elapsed_total = time.time() - start
    print()
    print(f"Complete in {elapsed_total:.0f}s ({elapsed_total/60:.1f} min)")
    print(f"  Processed: {processed}, Skipped: {skipped}, Errors: {errors}")
    print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Community detection
    if G.number_of_nodes() > 0:
        print()
        print("Running community detection...")
        comms = detect_communities(G)
        if "error" not in comms:
            print(f"  {comms['num_communities']} communities (modularity={comms['modularity']:.4f})")
            for c in comms.get("communities", [])[:5]:
                print(f"    #{c['id']}: {c['size']} members -- {c['summary'][:70]}")
        else:
            print(f"  {comms}")

        # Visualization
        print()
        print("Generating visualization...")
        path = visualize_graph(G)
        print(f"  Saved: {path}")

    print()
    gs = graph_stats(G)
    print("Node types:", gs.get("node_types", {}))
    print("Edge types:", gs.get("edge_types", {}))
    print("Top entities:")
    for n in gs.get("top_nodes", [])[:10]:
        print(f"  {n['name']:30s} ({n['type']:10s}) mentions={n['mentions']}")


if __name__ == "__main__":
    main()
