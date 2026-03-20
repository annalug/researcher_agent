# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Configure API keys — copy and edit:
cp config/.env config/.env.backup  # (no .env.example exists; edit config/.env directly)
```

**Required env vars** in `config/.env`:

- **OpenRouter (recommended):** `OPENROUTER_API_KEY`, `QWEN_MODEL` (default: `qwen/qwen3-235b-a22b`)
- **DashScope (Alibaba):** `QWEN_API_KEY`, `QWEN_MODEL` (default: `qwen-plus`)
- **Optional:** `SEMANTIC_SCHOLAR_API_KEY` (improves rate limits)

## Running

```bash
python main.py           # Web UI at http://localhost:7860
python main.py --cli     # Interactive CLI
python main.py --test    # Verify API connection
```

## Architecture

This is a **LangGraph multi-agent system** for academic research assistance. All agents share a single `AgentState` TypedDict defined in `agents/graph.py`.

**Data flow:**
1. Every user message enters the **Supervisor** node, which decides (via LLM) which agent to route to
2. The chosen agent runs and appends an `AIMessage` to `state["messages"]`
3. All agents terminate at `END` — there is no feedback loop between agents

**Shared state fields:**
- `messages` — full conversation history (accumulated via `add_messages`)
- `next_agent` — routing decision set by Supervisor
- `draft_text` — user's paper draft (passed from UI or CLI)
- `research_context` — papers found by Researcher; persisted across turns (truncated to 3000 chars)
- `iteration` — safeguard counter (incremented by Supervisor)

**Agents** (`agents/graph.py`):
- `supervisor` — routes to one of: `researcher`, `critic`, `editor`, `direct`
- `researcher` — fetches memory context, calls ArXiv/Semantic Scholar tools, uses `extract_key_sections` on uploaded PDFs, synthesizes results in a second LLM call, then saves papers and conversation to memory
- `critic` — reviews `draft_text` against literature; can call ArXiv/Semantic Scholar tools; saves conversation to memory
- `editor` — formats/rewrites text; no tool calls, LLM-only; saves conversation to memory
- `direct` — general Q&A, no tools; saves conversation to memory

**Tool execution pattern** (Researcher and Critic): agents use `llm.bind_tools(tools)` for a first pass, manually execute any `tool_calls` returned, then make a second bare `llm.invoke()` call with the tool results injected as a `HumanMessage`.

**LLM client** (`config/llm_client.py`): `get_qwen_llm()` returns a `ChatOpenAI` instance pointed at either OpenRouter or DashScope, selected by which env var is present. OpenRouter takes priority.

**Tools** (`tools/search_tools.py`): All tools are LangChain `@tool` decorated functions. ArXiv uses the `arxiv` Python SDK; Semantic Scholar uses direct `httpx` calls to its Graph API. `extract_pdf_text` uses PyMuPDF (`fitz`); `fetch_paper_from_url` handles both HTML (BeautifulSoup) and PDF URLs.

**Memory** (`tools/memory_store.py`): `AcademicMemory` class manages two persistent Chroma vector stores under `data/memory/` — one for conversations and one for indexed papers. Uses `sentence-transformers/all-MiniLM-L6-v2` (local, offline) for embeddings. Key methods: `add_conversation()`, `add_paper()`, `search_conversations()`, `search_papers()`, `get_research_context()`. A global `memory` instance is created at graph init time and shared across all agents. The Researcher queries memory before searching to enrich prompts and avoid duplicate work.

**PDF utilities** (`tools/pdf_utils.py`): Stateless helpers for extracting content from raw PDF text. `extract_key_sections(pdf_text, max_chars)` returns Abstract + Introduction + Keywords (up to 5000 chars by default). `extract_paper_metadata()` extracts title, year, DOI, and arXiv ID from the header. `create_search_query_from_pdf()` generates a search query from key sections. Used by the Researcher when a PDF is uploaded via `draft_text`.

**UI** (`ui/app.py`): Gradio `Blocks` interface with 3 tabs:
- **💬 Chat** — main interaction with mode selector (Search Literature / Review Draft / Edit & Format / General Assistant). Mode adds a prefix to the user message before invoking the graph. Sidebar has a draft text box and PDF upload with auto-indexing.
- **📚 Paper Library** — lists all unique papers indexed in `memory.papers_db`.
- **🔍 Search History** — semantic search over past conversations and indexed papers via `memory.get_research_context()`.

`ui/app.py` has its own `AcademicMemory` instance (separate from the one in `graph.py`) used for PDF auto-indexing on upload (`process_pdf()` calls `extract_paper_metadata` and `memory.add_paper()`). PDF text is capped at 12 000 chars for `draft_text`. `conversation_state` is a module-level dict (global, single-user); `research_context` is trimmed to 3 000 chars after each turn. `GRAPH` is built once at import time; if it fails, `GRAPH_READY = False` and errors are shown in chat.

**Project structure:**
```
researcher_agent/
├── main.py                   # Entry point (web / --cli / --test)
├── config/
│   ├── .env                  # API keys (do not commit)
│   └── llm_client.py         # get_qwen_llm() → ChatOpenAI
├── agents/
│   └── graph.py              # AgentState, all agent nodes, build_academic_graph()
├── tools/
│   ├── search_tools.py       # @tool functions: ArXiv, Semantic Scholar, PDF fetch
│   ├── memory_store.py       # AcademicMemory (Chroma + HuggingFace embeddings)
│   ├── pdf_utils.py          # extract_key_sections, extract_paper_metadata, create_search_query_from_pdf
│   └── rate_limiter.py       # Rate limiting helpers
├── ui/
│   └── app.py                # Gradio interface (build_interface, CUSTOM_CSS)
└── data/
    └── memory/
        ├── conversations/    # Chroma DB for conversation history
        └── papers/           # Chroma DB for indexed papers
```
