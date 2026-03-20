"""
Gradio Web Interface for the Academic Research Agent
Access: http://localhost:7860
"""
import os
import sys
from datetime import datetime

import gradio as gr
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "config", ".env"))

from agents.graph import build_academic_graph
from langchain_core.messages import AIMessage, HumanMessage
from tools.memory_store import AcademicMemory

# Initialize memory
memory = AcademicMemory()

# ─────────────────────────────────────────────
# Global agent state (multi-turn conversation)
# ─────────────────────────────────────────────

conversation_state = {
    "messages": [],
    "draft_text": "",
    "research_context": "",
}

try:
    GRAPH = build_academic_graph()
    GRAPH_READY = True
except Exception as e:
    GRAPH = None
    GRAPH_READY = False
    GRAPH_ERROR = str(e)


def process_pdf(pdf_file) -> str:
    """Extracts text from an uploaded PDF and auto-indexes it."""
    if pdf_file is None:
        return ""
    try:
        import fitz

        # Extract text
        doc = fitz.open(pdf_file.name)
        full_text = "\n\n".join(page.get_text("text") for page in doc)

        # ✨ AUTO-INDEX: Save to memory when PDF is uploaded
        try:
            from tools.pdf_utils import extract_paper_metadata

            # Extract metadata
            metadata = extract_paper_metadata(full_text)

            # Use filename if no title found
            if not metadata.get("title"):
                metadata["title"] = pdf_file.name.replace(".pdf", "")

            # Add to memory
            memory.add_paper(
                paper_text=full_text[:15000],  # First 15k chars
                metadata={
                    "title": metadata.get("title", "Uploaded PDF"),
                    "authors": metadata.get("authors", "Unknown"),
                    "year": metadata.get("year", "N/A"),
                    "source": "uploaded_pdf",
                    "filename": pdf_file.name,
                    "timestamp": datetime.now().isoformat(),
                }
            )
            print(f"✅ Auto-indexed PDF: {metadata.get('title', pdf_file.name)}")
        except Exception as e:
            print(f"⚠️ Could not auto-index PDF: {e}")

        # Return text for draft_text (limited to 12k to avoid context overflow)
        return full_text[:12000]
    except Exception as e:
        return f"Error processing PDF: {str(e)}"


def chat(
    user_message: str,
    history: list,
    draft_text: str,
    pdf_file,
    mode: str
) -> tuple:
    """
    Main chat function. Processes the message and returns a response.
    """
    if not user_message.strip():
        return history, ""

    if not GRAPH_READY:
        history.append({"role": "user", "content": user_message})
        history.append({
            "role": "assistant",
            "content": f"❌ **Configuration error:**\n\n{GRAPH_ERROR}\n\nSet your API key in the `.env` file."
        })
        return history, ""

    # Extract PDF text if provided
    pdf_text = ""
    if pdf_file is not None:
        pdf_text = process_pdf(pdf_file)

    # Build the final draft
    final_draft = ""
    if draft_text and draft_text.strip():
        final_draft = draft_text
    if pdf_text:
        final_draft = (final_draft + "\n\n" + pdf_text).strip()

    # Adapt the message according to the selected mode
    mode_prefixes = {
        "🔍 Search Literature": "Search papers on: ",
        "🔬 Review my Draft": "Critically review my paper/draft: ",
        "✏️ Edit and Format": "Edit and format this text to scientific standard: ",
        "💬 General Assistant": "",
    }
    prefix = mode_prefixes.get(mode, "")

    # Add context from recent history
    recent_context = ""
    if conversation_state["research_context"] and len(history) > 0:
        recent_context = conversation_state["research_context"]

    try:
        result = GRAPH.invoke({
            "messages": [HumanMessage(content=prefix + user_message)],
            "next_agent": "",
            "draft_text": final_draft,
            "research_context": recent_context,
            "iteration": 0,
        })

        # Update research context for subsequent messages
        if result.get("research_context"):
            conversation_state["research_context"] = result["research_context"][-3000:]

        last_ai = next(
            (m for m in reversed(result["messages"]) if isinstance(m, AIMessage)), None
        )
        response = last_ai.content if last_ai else "No response from agent."

    except Exception as e:
        response = f"❌ **Error processing request:**\n\n```\n{str(e)}\n```\n\nCheck your API key and connection."

    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": response})
    return history, ""


def clear_context():
    """Clears the accumulated research context."""
    conversation_state["research_context"] = ""
    conversation_state["messages"] = []
    return [], ""


def search_past_research(query: str) -> str:
    """Searches in past conversations and indexed papers."""
    if not query.strip():
        return "Please enter a search query."

    try:
        context = memory.get_research_context(query, k=5)
        return context if context else "No relevant results found in memory."
    except Exception as e:
        return f"Error searching memory: {str(e)}"


def list_indexed_papers() -> str:
    """Lists all indexed papers in the vector database."""
    try:
        papers = memory.papers_db.get()

        if not papers or not papers.get('metadatas'):
            return "No papers indexed yet. Start searching for papers to build your library!"

        # Group by title to avoid duplicates (chunks)
        unique_papers = {}
        for metadata in papers['metadatas']:
            title = metadata.get('title', 'N/A')
            if title and title != 'N/A':
                if title not in unique_papers:
                    unique_papers[title] = metadata

        if not unique_papers:
            return "No papers indexed yet."

        output = "# 📚 Indexed Papers Library\n\n"
        for i, (title, meta) in enumerate(sorted(unique_papers.items()), 1):
            output += f"**{i}. {title}**\n"
            if meta.get('authors'):
                output += f"   - Authors: {meta['authors']}\n"
            if meta.get('year'):
                output += f"   - Year: {meta['year']}\n"
            if meta.get('arxiv_id'):
                output += f"   - ArXiv ID: {meta['arxiv_id']}\n"
            if meta.get('source'):
                output += f"   - Source: {meta['source']}\n"
            if meta.get('citations'):
                output += f"   - Citations: {meta['citations']}\n"
            output += "\n"

        return output
    except Exception as e:
        return f"Error listing papers: {str(e)}"


# ─────────────────────────────────────────────
# GRADIO INTERFACE STYLING
# ─────────────────────────────────────────────

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --primary: #1a1a2e;
    --secondary: #16213e;
    --accent: #0f3460;
    --gold: #e94560;
    --gold-soft: #c77dff;
    --text: #e0e0e0;
    --text-dim: #9a9ab8;
    --border: rgba(201, 125, 255, 0.2);
    --card: rgba(22, 33, 62, 0.8);
}

body, .gradio-container {
    background: var(--primary) !important;
    font-family: 'Sora', sans-serif !important;
    color: var(--text) !important;
}

.gradio-container {
    max-width: 1400px !important;
    margin: 0 auto !important;
}

/* Header */
.header-box {
    background: linear-gradient(135deg, #0f3460 0%, #1a1a2e 50%, #16213e 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 32px;
    text-align: center;
    margin-bottom: 20px;
    position: relative;
    overflow: hidden;
}

.header-box::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background: radial-gradient(ellipse at 50% 0%, rgba(201,125,255,0.15), transparent 70%);
    pointer-events: none;
}

/* Chatbot */
.chatbot {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
}

.chatbot .message {
    font-family: 'Sora', sans-serif !important;
}

.chatbot .user {
    background: linear-gradient(135deg, #0f3460, #16213e) !important;
    border: 1px solid rgba(201,125,255,0.3) !important;
}

.chatbot .bot {
    background: rgba(22, 33, 62, 0.6) !important;
    border: 1px solid rgba(233,69,96,0.2) !important;
}

/* Inputs */
textarea, input[type="text"] {
    background: var(--secondary) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    font-family: 'Sora', sans-serif !important;
    border-radius: 8px !important;
}

/* Buttons */
button.primary {
    background: linear-gradient(135deg, #e94560, #c77dff) !important;
    border: none !important;
    color: white !important;
    font-family: 'Sora', sans-serif !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    padding: 10px 24px !important;
    transition: all 0.2s !important;
}

button.primary:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(233,69,96,0.4) !important;
}

button.secondary {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-dim) !important;
    font-family: 'Sora', sans-serif !important;
    border-radius: 8px !important;
}

/* Tabs */
.tab-nav button {
    color: var(--text-dim) !important;
    border-bottom: 2px solid transparent !important;
}

.tab-nav button.selected {
    color: var(--gold-soft) !important;
    border-bottom: 2px solid var(--gold-soft) !important;
}

/* Radio buttons */
.radio-group label {
    color: var(--text) !important;
}

/* Labels */
label {
    color: var(--text-dim) !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
}

/* Accordion */
.accordion {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}

/* Code blocks in chat */
code {
    font-family: 'JetBrains Mono', monospace !important;
    background: rgba(15,52,96,0.5) !important;
    padding: 2px 6px !important;
    border-radius: 4px !important;
}

pre code {
    display: block !important;
    padding: 12px !important;
}

/* Markdown in outputs */
.markdown-text {
    color: var(--text) !important;
}

.markdown-text h1, .markdown-text h2, .markdown-text h3 {
    color: var(--gold-soft) !important;
}
"""

HEADER_HTML = """
<div class="header-box">
    <div style="font-size: 2.5rem; margin-bottom: 8px;">🎓</div>
    <h1 style="font-size: 2rem; font-weight: 700; background: linear-gradient(135deg, #c77dff, #e94560); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0 0 8px 0;">
        Academic Research Agent
    </h1>
    <p style="color: #9a9ab8; font-size: 0.95rem; margin: 0;">
        Multi-Agent System • LangGraph + Qwen • ArXiv & Semantic Scholar • Persistent Memory
    </p>
    <div style="display: flex; justify-content: center; gap: 16px; margin-top: 16px; flex-wrap: wrap;">
        <span style="background: rgba(201,125,255,0.15); border: 1px solid rgba(201,125,255,0.3); border-radius: 20px; padding: 4px 14px; font-size: 0.8rem; color: #c77dff;">🔍 Researcher</span>
        <span style="background: rgba(233,69,96,0.15); border: 1px solid rgba(233,69,96,0.3); border-radius: 20px; padding: 4px 14px; font-size: 0.8rem; color: #e94560;">🔬 Critic</span>
        <span style="background: rgba(15,52,96,0.4); border: 1px solid rgba(100,180,255,0.3); border-radius: 20px; padding: 4px 14px; font-size: 0.8rem; color: #64b4ff;">✏️ Editor</span>
        <span style="background: rgba(201,125,255,0.1); border: 1px solid rgba(201,125,255,0.25); border-radius: 20px; padding: 4px 14px; font-size: 0.8rem; color: #9a9ab8;">🧠 Memory</span>
    </div>
</div>
"""

EXAMPLE_QUERIES = [
    ["🔍 Search Literature", "Find the most cited papers on Android malware detection using machine learning in the last 3 years"],
    ["🔍 Search Literature", "What are the main works on synthetic data generation for imbalanced datasets?"],
    ["🔬 Review my Draft", "Analyze my abstract and tell me if the contribution is clear and if there are bibliographic gaps"],
    ["✏️ Edit and Format", "Format this paragraph in IEEE style and improve scientific clarity"],
    ["💬 General Assistant", "How to structure the Related Work section of a paper for a top-tier conference?"],
]


# ─────────────────────────────────────────────
# GRADIO INTERFACE BUILDER
# ─────────────────────────────────────────────

def build_interface():
    """Builds the complete Gradio interface with tabs."""
    with gr.Blocks(title="Academic Research Agent") as demo:
        gr.HTML(HEADER_HTML)

        with gr.Tabs():
            # ─── TAB 1: CHAT ───
            with gr.Tab("💬 Chat"):
                with gr.Row():
                    # Main chat column
                    with gr.Column(scale=2):
                        chatbot = gr.Chatbot(
                            label="",
                            height=500,
                            show_label=False,
                        )

                        with gr.Row():
                            msg_input = gr.Textbox(
                                placeholder="Type your request... (e.g. 'Search papers on transformers for anomaly detection')",
                                show_label=False,
                                scale=5,
                                lines=2,
                                max_lines=4,
                            )
                            send_btn = gr.Button("Send →", variant="primary", scale=1)

                        with gr.Row():
                            mode_radio = gr.Radio(
                                choices=[
                                    "🔍 Search Literature",
                                    "🔬 Review my Draft",
                                    "✏️ Edit and Format",
                                    "💬 General Assistant",
                                ],
                                value="🔍 Search Literature",
                                label="Agent Mode",
                                interactive=True,
                            )

                        with gr.Row():
                            clear_btn = gr.Button("🗑️ Clear Conversation", variant="secondary")

                        # Quick examples
                        gr.HTML("<p style='color:#9a9ab8; font-size:0.8rem; margin-top:8px; text-transform:uppercase; letter-spacing:0.05em;'>Quick examples:</p>")
                        with gr.Row():
                            for mode, ex in EXAMPLE_QUERIES[:3]:
                                gr.Button(ex[:50] + "...", size="sm").click(
                                    fn=lambda m=mode, e=ex: (e, m),
                                    outputs=[msg_input, mode_radio]
                                )

                    # Sidebar context column
                    with gr.Column(scale=1):
                        with gr.Accordion("📄 Your Draft / Paper", open=False):
                            draft_box = gr.Textbox(
                                placeholder="Paste your draft text here for the Critic and Editor agents to analyze...",
                                lines=10,
                                max_lines=20,
                                label="",
                            )

                        with gr.Accordion("📤 Upload PDF", open=False):
                            pdf_upload = gr.File(
                                label="",
                                file_types=[".pdf"],
                                type="filepath",
                            )
                            gr.HTML("<p style='color:#9a9ab8; font-size:0.75rem;'>PDF text will be extracted and auto-indexed in your library.</p>")

                        with gr.Accordion("ℹ️ How to use", open=True):
                            gr.HTML("""
                            <div style="font-size: 0.82rem; color: #9a9ab8; line-height: 1.7;">
                                <p><strong style="color:#c77dff;">🔍 Search Literature</strong><br>
                                Automatic search on ArXiv and Semantic Scholar</p>

                                <p><strong style="color:#e94560;">🔬 Review Draft</strong><br>
                                Paste your text in "Your Draft" and request a review</p>

                                <p><strong style="color:#64b4ff;">✏️ Edit and Format</strong><br>
                                Improves style, grammar, and scientific formatting</p>

                                <p><strong style="color:#9a9ab8;">💬 General Assistant</strong><br>
                                Questions about methodology, structure, etc.</p>

                                <hr style="border-color: rgba(201,125,255,0.2); margin: 12px 0;">
                                <p style="font-size:0.75rem;">✨ <strong>Memory enabled:</strong> All searches and conversations are indexed for future reference.</p>
                                
                                <p style="font-size:0.75rem;">Set your API key in the <code>.env</code> file<br>
                                Supports: OpenRouter (Qwen) or DashScope</p>
                            </div>
                            """)

                # Event handlers for chat tab
                def submit(msg, history, draft, pdf, mode):
                    return chat(msg, history, draft, pdf, mode)

                send_btn.click(
                    submit,
                    inputs=[msg_input, chatbot, draft_box, pdf_upload, mode_radio],
                    outputs=[chatbot, msg_input],
                )

                msg_input.submit(
                    submit,
                    inputs=[msg_input, chatbot, draft_box, pdf_upload, mode_radio],
                    outputs=[chatbot, msg_input],
                )

                clear_btn.click(
                    clear_context,
                    outputs=[chatbot, msg_input],
                )

            # ─── TAB 2: PAPER LIBRARY ───
            with gr.Tab("📚 Paper Library"):
                gr.Markdown("## Indexed Papers")
                gr.Markdown("All papers you've searched for are automatically indexed here for future reference.")

                with gr.Row():
                    list_papers_btn = gr.Button("📋 List All Papers", variant="primary")

                papers_output = gr.Markdown(
                    value="Click 'List All Papers' to see your indexed library.",
                    elem_classes=["markdown-text"]
                )

                list_papers_btn.click(
                    list_indexed_papers,
                    outputs=papers_output
                )

            # ─── TAB 3: SEARCH HISTORY ───
            with gr.Tab("🔍 Search History"):
                gr.Markdown("## Search in Past Conversations and Papers")
                gr.Markdown("Use semantic search to find relevant information from your research history.")

                with gr.Row():
                    search_input = gr.Textbox(
                        label="Search Query",
                        placeholder="Enter keywords or questions (e.g., 'malware detection GANs')",
                        lines=2,
                    )

                with gr.Row():
                    search_btn = gr.Button("🔎 Search", variant="primary")

                search_output = gr.Markdown(
                    value="Search results will appear here.",
                    elem_classes=["markdown-text"]
                )

                search_btn.click(
                    search_past_research,
                    inputs=search_input,
                    outputs=search_output
                )

                # Example searches
                gr.Markdown("### Quick Searches")
                example_searches = [
                    "synthetic data generation",
                    "malware detection machine learning",
                    "imbalanced datasets",
                ]

                with gr.Row():
                    for ex_search in example_searches:
                        gr.Button(ex_search).click(
                            lambda s=ex_search: s,
                            outputs=search_input
                        )

    return demo


if __name__ == "__main__":
    demo = build_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        inbrowser=True,
        css=CUSTOM_CSS,
    )