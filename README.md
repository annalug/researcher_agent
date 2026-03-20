# 🎓 Academic Research Agent

**Multi-Agent** system for academic research assistance.
Built with **LangGraph + Qwen (open source)** + ArXiv + Semantic Scholar + **Persistent Memory**.

---

## ✨ Features

- **Multi-agent routing** — Supervisor automatically directs each request to the right specialist
- **Literature search** — ArXiv and Semantic Scholar with citation metrics and open-access indicators
- **Draft review** — Critic agent compares your paper against the state of the art
- **Scientific editing** — IEEE, Nature, ACM, Elsevier/Springer style formatting
- **PDF upload** — Extract text and auto-index papers in your library
- **Persistent memory** — All searches and conversations are stored in a local vector DB (Chroma); the Researcher uses past context to avoid duplicate work
- **Paper Library** — Browse all papers you have ever searched or uploaded
- **Search History** — Semantic search over past conversations and indexed papers

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                       LANGGRAPH GRAPH                        │
│                                                              │
│   [User Message]                                             │
│           │                                                  │
│           ▼                                                  │
│     ┌─────────────┐                                          │
│     │  SUPERVISOR  │  ← Decides which agent to trigger      │
│     └──────┬──────┘                                          │
│            │                                                 │
│     ┌──────┼──────────────────┐                             │
│     ▼      ▼                  ▼         ▼                   │
│  [🔍 RESEARCHER] [🔬 CRITIC] [✏️ EDITOR] [💬 DIRECT]        │
│     │         │          │                                   │
│     ▼         ▼          ▼                                   │
│   ArXiv  Semantic    Scientific          ┌─────────────────┐ │
│   Search  Scholar    Formatting          │  🧠 Memory (DB) │ │
│                                          │  conversations  │ │
│           [Final Response]               │  papers         │ │
│                  │                       └─────────────────┘ │
│                  └──────────────── saves to memory ──────────┤
└──────────────────────────────────────────────────────────────┘
```

### Agents

| Agent | Role | Tools |
|-------|------|-------|
| **Supervisor** | Routes to the correct agent | — |
| **Researcher** | Searches and synthesizes literature; queries memory for context | ArXiv, Semantic Scholar, memory |
| **Critic** | Reviews your draft vs. literature | ArXiv, Semantic Scholar |
| **Editor** | Formats and improves writing | IEEE/Nature/ACM style |
| **Direct** | General responses | — |

All agents save their conversations to the vector memory after each response.

---

## ⚙️ Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Key

Edit `config/.env` and add your key:

#### Option A: OpenRouter (recommended — access to Qwen3)
```env
OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxxxxx
QWEN_MODEL=qwen/qwen3-235b-a22b
```
- Create an account at: https://openrouter.ai
- Qwen3-235B has a free tier with daily limits

#### Option B: DashScope (official Alibaba API)
```env
QWEN_API_KEY=sk-xxxxxxxxxxxx
QWEN_MODEL=qwen-plus
```
- Register at: https://dashscope.aliyuncs.com

#### Optional
```env
SEMANTIC_SCHOLAR_API_KEY=...   # improves rate limits
```

### 3. Test the connection

```bash
python main.py --test
```

---

## 🚀 Usage

### Web Interface (recommended)

```bash
python main.py
# Access: http://localhost:7860
```

### Command Line

```bash
python main.py --cli
```

---

## 🖥️ Web Interface Tabs

### 💬 Chat
Main interaction panel. Select the agent mode before sending:

| Mode | What it does |
|------|-------------|
| 🔍 Search Literature | Searches ArXiv + Semantic Scholar and synthesizes findings |
| 🔬 Review my Draft | Critically reviews your text against the literature |
| ✏️ Edit and Format | Improves style, grammar, and scientific formatting |
| 💬 General Assistant | General academic Q&A |

Use the sidebar to paste a **draft text** or **upload a PDF** — the PDF is automatically extracted and indexed in your library.

### 📚 Paper Library
Lists all papers indexed so far (from searches and uploaded PDFs), showing title, authors, year, arXiv ID, and source.

### 🔍 Search History
Semantic search over your entire research history — past conversations and indexed papers — using natural language queries.

---

## 💡 Usage Examples

### Search literature
```
"Find papers on malware detection using variational autoencoders"
"What are the most cited surveys on federated learning in IoT?"
"Search recent work on synthetic data for imbalanced datasets (from 2022 onwards)"
```

### Review your draft
Paste your text in the "Draft" box and send:
```
"Review my abstract and tell me if the contribution is clear"
"Analyze my methodology section and compare with the state of the art"
"Identify bibliographic gaps in my Related Work"
```

### Edit and format
```
"Format this paragraph in IEEE style"
"Improve my abstract for submission to Nature Communications"
"Fix the scientific style of this excerpt and improve clarity"
```

### General questions
```
"How to structure an 8-page paper for SBSeg?"
"What is the difference between TSTR and TRTS in synthetic data evaluation?"
"How to calculate the p-value for classifier comparison?"
```

> General questions: Portuguese is fine. Paper searches: English is preferred.

---

## 📁 Project Structure

```
researcher_agent/
├── main.py                   # Entry point (web / --cli / --test)
├── requirements.txt          # Python dependencies
├── config/
│   ├── .env                  # API keys (DO NOT commit!)
│   └── llm_client.py         # Qwen client (OpenRouter/DashScope)
├── agents/
│   └── graph.py              # LangGraph graph + all agent nodes
├── tools/
│   ├── search_tools.py       # ArXiv, Semantic Scholar, PDF fetch tools
│   ├── memory_store.py       # AcademicMemory — Chroma vector DB
│   ├── pdf_utils.py          # Key section extraction from PDF text
│   └── rate_limiter.py       # API rate limiting helpers
├── ui/
│   └── app.py                # Gradio interface (3 tabs)
└── data/
    └── memory/               # Persisted vector databases (gitignored)
        ├── conversations/    # Past conversations
        └── papers/           # Indexed papers
```

---

## 🔧 Tool Reference

### `search_arxiv(query, max_results)`
- Searches ArXiv by relevance
- Returns: title, authors, date, ID, PDF link, abstract

### `search_semantic_scholar(query, limit, year_start)`
- Search with citation counts and Open Access indicator
- Filter by publication year

### `get_arxiv_paper_details(arxiv_id)`
- Full details of a paper by ID (e.g. `2303.08774`)

### `get_paper_citations(paper_id, limit)`
- Lists papers that cite a specific article

### `extract_pdf_text(pdf_path)`
- Extracts text from a local PDF (via PyMuPDF)
- Converts to clean Markdown

### `fetch_paper_from_url(url)`
- Fetches content from an online paper URL

---

## 🧠 Memory System

All interactions are stored in a local **Chroma** vector database under `data/memory/`:

| Store | What is saved |
|-------|--------------|
| `conversations/` | Every user message + agent response, tagged by agent type |
| `papers/` | Papers found via search or uploaded as PDF, chunked for retrieval |

Embeddings are generated locally using `sentence-transformers/all-MiniLM-L6-v2` (no external API needed).

When the Researcher receives a new query, it first retrieves relevant past conversations and indexed papers to enrich its prompt — avoiding repeated searches and building on previous work.

---

## 📝 Notes

- The agent uses **Qwen** (Alibaba's open-source model) via API
- **ArXiv** and **Semantic Scholar** searches are free
- Memory persists across sessions — the `data/` directory is gitignored
- For very long PDFs, text passed to agents is truncated at ~12 000 characters
