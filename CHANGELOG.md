# Changelog

All notable changes to this project will be documented in this file.

## [1.0.0] - 2026-04-04

### Added
- Complete rewrite from v1 (claude-memory). Zero shared code.
- **Storage**: SQLite + sqlite-vec (768-dim vectors) + FTS5, replacing ChromaDB
- **Search**: Reciprocal Rank Fusion (BM25 + vector similarity) with ACT-R activation reranking
- **Scoring**: ACT-R cognitive activation model (Anderson & Schooler 1991) with base-level activation, spreading activation, logistic noise, and retrieval probability
- **Decay**: FadeMem dual-layer system (STM/LTM) with power-law decay, importance scoring, automatic promotion/demotion, and archival with hysteresis
- **Knowledge Graph**: NetworkX directed graph with entity/relationship extraction via local LLM (qwen2.5:3b), Leiden community detection, and HippoRAG-style Personalized PageRank retrieval
- **Auto-Extraction**: Two-pass LLM pipeline for automatic fact extraction from conversations with novelty checking
- **Compaction**: CogCanvas-inspired 6-step compression pipeline with protected content extraction, verification, and receipts
- **Multi-Agent**: Authority chain with configurable hierarchy, Kafka-style consumer offsets for changelog sync, contradiction detection, and conflict logging
- **Security**: 13 credential scanning patterns with allowlist, 6 injection detection patterns, SHA-256 integrity manifests
- **MCP Server**: FastMCP server with 17 tools for Claude Code, Claude Desktop, or any MCP-compatible client
- **CLI**: Standalone command-line interface for non-MCP usage
- **Vault Indexer**: Incremental markdown indexing with SHA-256 delta detection, paragraph-aware chunking (500 tokens, 50 overlap), and frontmatter extraction
- All processing runs locally via Ollama (nomic-embed-text for embeddings, qwen2.5:3b for extraction). Zero API keys required.

### Predecessor
- [claude-memory v1](https://github.com/Haustorium12/claude-memory) -- ChromaDB-based CLI tool with 5 biological memory mechanisms (Ebbinghaus decay, evergreen exemptions, salience weighting, retrieval strengthening, consolidation). v2 is a complete successor, not a fork.
