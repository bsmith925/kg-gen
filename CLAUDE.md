# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

kg-gen is a Python library for extracting knowledge graphs from unstructured text using LLMs. It uses DSPy for structured output generation and supports any LLM provider via LiteLLM (OpenAI, Anthropic, Gemini, Ollama, etc.). The library can process both plain text and conversation messages, with features for chunking large texts, clustering similar entities/relations, and combining multiple graphs.

## Build and Test Commands

### Installation
```bash
# Install from source with dev dependencies
pip install -e '.[dev]'

# Install with MCP server support
pip install -e '.[mcp]'
```

### Testing
```bash
# Run all tests (includes both main tests and MCP tests)
pytest

# Run specific test file
python tests/test_basic.py

# Run MCP tests specifically
pytest mcp/tests/ -v

# Run MINE benchmark evaluation
cd experiments/MINE
python _1_evaluation.py --model openai/gpt-5-nano --evaluation-model local
python _2_compare_results.py
streamlit run _3_visualize.py
```

### Linting
```bash
# Run ruff linter and formatter (configured in pre-commit)
ruff check .
ruff format .

# Pre-commit hooks
pre-commit run --all-files
```

### Running the MCP Server
```bash
# Start MCP server (clears memory by default)
kggen mcp

# Keep existing memory
kggen mcp --keep-memory

# Use custom model and storage
kggen mcp --model gemini/gemini-2.0-flash --storage-path ./my_memory.json
```

### Running the Web App
```bash
# Install additional dependencies
pip install fastapi uvicorn[standard]

# Run local development server
uvicorn app.server:app --reload --port 8000
```

## Architecture

### Core Components

1. **KGGen class** (`src/kg_gen/kg_gen.py`): Main interface
   - `generate()`: Extract knowledge graph from text or messages
   - `aggregate()`: Combine multiple graphs
   - `cluster()`: Cluster similar entities and relations
   - `visualize()`: Create interactive HTML visualization
   - `retrieve()`: Find relevant entities/relations via semantic search

2. **Three-Step Extraction Pipeline** (`src/kg_gen/steps/`):
   - `_1_get_entities.py`: Extract entities using DSPy signatures (TextEntities/ConversationEntities)
   - `_2_get_relations.py`: Extract relations with fallback mechanism if strict typing fails
   - `_3_deduplicate.py`: Run deduplication via SemHash or LLM-based clustering

3. **Graph Model** (`src/kg_gen/models.py`): Pydantic model with:
   - `entities`, `edges`, `relations` (required)
   - `entity_clusters`, `edge_clusters` (optional, created during clustering)
   - `entity_metadata` (optional, stores additional context)
   - Serialization: `from_file()` and `to_file()` methods

4. **Utilities** (`src/kg_gen/utils/`):
   - `chunk_text.py`: Split large texts into chunks with sentence boundary awareness
   - `llm_deduplicate.py`: LLM-based entity/edge clustering with semantic embeddings
   - `deduplicate.py`: SemHash-based deduplication (faster, less accurate)
   - `visualize_kg.py`: Generate interactive HTML visualizations using template.html
   - `neo4j_integration.py`: Upload graphs to Neo4j databases (Aura or local)

5. **MCP Server** (`mcp/server.py`): Memory management for AI agents
   - Tools: `add_memories`, `retrieve_relevant_memories`, `visualize_memories`, `get_memory_stats`
   - Uses FastMCP framework
   - Persistent storage via JSON files

### Key Patterns

- **DSPy Integration**: All LLM calls go through DSPy signatures for structured outputs. The library configures DSPy's language model via `dspy.configure()` with LiteLLM models.

- **Conversation vs Text**: Two code paths based on `is_conversation` flag. Conversations extract entities/relations involving speakers (roles) and use different prompting.

- **Chunking Strategy**: Large texts are split into chunks, processed independently, then aggregated. Clustering is applied after aggregation to merge duplicates.

- **Fallback Extraction**: If strict Pydantic typing fails in relation extraction, falls back to string-based extraction then fixes relations to match entity list.

- **Model Configuration**: The library validates certain model-specific requirements (e.g., gpt-5 requires temperature=1.0 and max_tokens>=16000).

## Important Details

### Model Initialization
- LiteLLM routes model calls based on provider prefix (e.g., `openai/`, `gemini/`, `ollama_chat/`)
- Custom API base URLs supported via `api_base` parameter
- API keys can be passed directly or via environment variables (OPENAI_API_KEY, KG_API_KEY, etc.)

### Entity Metadata Feature
- `entity_metadata` maps entity names to sets of additional context strings
- Aggregated across chunks during generation
- Deduplicated during clustering to avoid redundant metadata

### Deduplication Methods
Two approaches available via `DeduplicateMethod` enum:
- `SEMHASH`: Fast, deterministic, uses semhash library (default for non-LLM clustering)
- `LLM`: Uses embeddings + LLM for higher quality clustering (used in `cluster()` method)

### Neo4j Integration
Helper functions for uploading graphs to Neo4j:
- `upload_to_neo4j()`: Main function to upload Graph objects
- `get_aura_connection_config()`: Get config for Neo4j Aura (cloud)
- `get_local_connection_config()`: Get config for local Neo4j instances

### Testing Conventions
- Tests use fixtures from `tests/fixtures.py` for KGGen instance
- `match_subset()` helper does fuzzy matching for entity/edge validation
- Environment variables loaded from `.env` files (see `.env.example`)

### MINE Benchmark
Located in `experiments/MINE/`:
- Evaluates KG generation quality on Q&A tasks
- Uses Hugging Face datasets for evaluation data
- Supports local and API-based evaluation models
- Includes visualization dashboard via Streamlit

## Common Gotchas

- **Graph validation**: When loading from file, `Graph.from_file()` auto-fixes missing entities/edges that appear in relations
- **Clustering context**: Providing a `context` string to `cluster()` significantly improves clustering quality
- **Chunk size**: Default chunk size is tuned for balance; very small chunks lose context, very large chunks may hit token limits
- **Message format**: Conversation input must be list of dicts with "role" and "content" keys
- **Temperature validation**: gpt-5 models will raise ValueError if temperature < 1.0 due to OpenAI API requirements
