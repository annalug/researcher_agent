# 🎓 Academic Research Agent

**Multi-Agent** system for academic research assistance.
Built with **LangGraph + Qwen (open source)** + ArXiv + Semantic Scholar.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    LANGGRAPH GRAPH                       │
│                                                          │
│   [User Message]                                         │
│           │                                              │
│           ▼                                              │
│     ┌─────────────┐                                      │
│     │  SUPERVISOR  │  ← Decides which agent to trigger   │
│     └──────┬──────┘                                      │
│            │                                             │
│     ┌──────┼──────────────────┐                         │
│     ▼      ▼                  ▼         ▼               │
│  [🔍 RESEARCHER] [🔬 CRITIC] [✏️ EDITOR] [💬 DIRECT]    │
│     │         │          │                               │
│     ▼         ▼          ▼                               │
│   ArXiv  Semantic    Scientific                          │
│   Search  Scholar    Formatting                          │
│                                                          │
│           [Final Response]                               │
└─────────────────────────────────────────────────────────┘
```

### Agents

| Agent | Role | Tools |
|-------|------|-------|
| **Supervisor** | Routes to the correct agent | — |
| **Researcher** | Searches and synthesizes literature | ArXiv, Semantic Scholar |
| **Critic** | Reviews your draft vs. literature | ArXiv + critical analysis |
| **Editor** | Formats and improves writing | IEEE/Nature/ACM style |
| **Direct** | General responses | — |

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
Note:
General questions: Portuguese is fine.
Paper searches: English is preferred.
---

## 📁 Project Structure

```
researcher_agent/
├── main.py                 # Main entry point
├── requirements.txt        # Python dependencies
├── config/
│   ├── .env               # Your API keys (DO NOT commit!)
│   └── llm_client.py      # Qwen client (OpenRouter/DashScope)
├── agents/
│   └── graph.py           # LangGraph graph + all agents
├── tools/
│   └── search_tools.py    # Tools: ArXiv, Semantic Scholar, PDF
└── ui/
    └── app.py             # Gradio interface
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

## 🌐 Future Integrations

- [ ] **Firecrawl** — Online PDFs → high-quality Markdown
- [ ] **Zotero API** — Sync with your library
- [ ] **Overleaf** — Direct LaTeX editing
- [ ] **Local RAG** — Upload multiple PDFs with vector search
- [ ] **Open WebUI** — Alternative interface with local model support

---

## 📝 Notes

- The agent uses **Qwen** (Alibaba's open-source model) via API
- **ArXiv** and **Semantic Scholar** searches are free
- Research context is maintained during the session for chained queries
- For very long PDFs, text is truncated at ~12,000 characters
