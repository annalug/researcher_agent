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
- `researcher` — calls ArXiv/Semantic Scholar tools, then synthesizes results in a second LLM call
- `critic` — reviews `draft_text` against literature; can call ArXiv/Semantic Scholar tools
- `editor` — formats/rewrites text; no tool calls, LLM-only
- `direct` — general Q&A, no tools

**Tool execution pattern** (Researcher and Critic): agents use `llm.bind_tools(tools)` for a first pass, manually execute any `tool_calls` returned, then make a second bare `llm.invoke()` call with the tool results injected as a `HumanMessage`.

**LLM client** (`config/llm_client.py`): `get_qwen_llm()` returns a `ChatOpenAI` instance pointed at either OpenRouter or DashScope, selected by which env var is present. OpenRouter takes priority.

**Tools** (`tools/search_tools.py`): All tools are LangChain `@tool` decorated functions. ArXiv uses the `arxiv` Python SDK; Semantic Scholar uses direct `httpx` calls to its Graph API. `extract_pdf_text` uses PyMuPDF (`fitz`); `fetch_paper_from_url` handles both HTML (BeautifulSoup) and PDF URLs.

**UI** (`ui/app.py`): Gradio `Blocks` interface. Conversation `research_context` is stored in a module-level `conversation_state` dict (global, single-user). The `GRAPH` is built once at import time; if it fails, `GRAPH_READY = False` and errors are shown in the chat.
