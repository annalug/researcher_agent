"""
Academic search tools:
- ArXiv API (free, no key required)
- Semantic Scholar API (free, with generous rate limits)
- PDF → Markdown converter (via PyMuPDF or Firecrawl)
"""
import os
import re
import time
import httpx
import arxiv
import fitz  # PyMuPDF
from typing import Optional
from langchain.tools import tool


# ─────────────────────────────────────────────
# ARXIV
# ─────────────────────────────────────────────

@tool
def search_arxiv(query: str, max_results: int = 5) -> str:
    """
    Searches papers on ArXiv. Returns title, authors, abstract and link for each paper.
    Use to find scientific articles on any topic.

    Args:
        query: Search terms in English (e.g. 'transformer attention mechanism NLP')
        max_results: Maximum number of results (default: 5)
    """
    try:
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )
        results = []
        for paper in client.results(search):
            results.append(
                f"**Title:** {paper.title}\n"
                f"**Authors:** {', '.join(str(a) for a in paper.authors[:4])}"
                f"{'et al.' if len(paper.authors) > 4 else ''}\n"
                f"**Published:** {paper.published.strftime('%Y-%m-%d')}\n"
                f"**ArXiv ID:** {paper.entry_id.split('/')[-1]}\n"
                f"**Link:** {paper.entry_id}\n"
                f"**PDF:** {paper.pdf_url}\n"
                f"**Abstract:** {paper.summary[:500]}...\n"
                f"{'─'*60}"
            )
        if not results:
            return "No papers found for this search."
        return f"## ArXiv results for: '{query}'\n\n" + "\n\n".join(results)
    except Exception as e:
        return f"Error searching ArXiv: {str(e)}"


@tool
def get_arxiv_paper_details(arxiv_id: str) -> str:
    """
    Gets full details of an ArXiv paper by its ID.

    Args:
        arxiv_id: Paper ID (e.g. '2303.08774' or 'https://arxiv.org/abs/2303.08774')
    """
    # Strip ID if a full URL is provided
    arxiv_id = arxiv_id.strip()
    if "arxiv.org" in arxiv_id:
        arxiv_id = arxiv_id.split("/")[-1]
    
    try:
        client = arxiv.Client()
        search = arxiv.Search(id_list=[arxiv_id])
        paper = next(client.results(search))
        return (
            f"# {paper.title}\n\n"
            f"**Authors:** {', '.join(str(a) for a in paper.authors)}\n"
            f"**Published:** {paper.published.strftime('%Y-%m-%d')}\n"
            f"**Categories:** {', '.join(paper.categories)}\n"
            f"**DOI:** {paper.doi or 'N/A'}\n"
            f"**Link:** {paper.entry_id}\n"
            f"**PDF:** {paper.pdf_url}\n\n"
            f"## Abstract\n{paper.summary}\n\n"
            f"## Comments\n{paper.comment or 'No additional comments.'}"
        )
    except StopIteration:
        return f"Paper not found: {arxiv_id}"
    except Exception as e:
        return f"Error: {str(e)}"


# ─────────────────────────────────────────────
# SEMANTIC SCHOLAR
# ─────────────────────────────────────────────


SEMANTIC_SCHOLAR_BASE = "https://api.semanticscholar.org/graph/v1"

def _ss_headers() -> dict:
    key = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
    return {"x-api-key": key} if key else {}


@tool
def search_semantic_scholar(query: str, limit: int = 5, year_start: Optional[int] = None) -> str:
    """
    Searches papers on Semantic Scholar with rich metadata (citations, references, open access).

    Args:
        query: Search terms
        limit: Number of results (default: 5)
        year_start: Filter papers from this year onwards (optional, e.g. 2020)
    """
    params = {
        "query": query,
        "limit": limit,
        "fields": "title,authors,year,abstract,citationCount,openAccessPdf,externalIds,url"
    }
    if year_start:
        params["year"] = f"{year_start}-"

    try:
        with httpx.Client(timeout=30) as client:
            resp = client.get(
                f"{SEMANTIC_SCHOLAR_BASE}/paper/search",
                params=params,
                headers=_ss_headers()
            )
            resp.raise_for_status()
            data = resp.json()

        papers = data.get("data", [])
        if not papers:
            return "No results found on Semantic Scholar."

        results = []
        for p in papers:
            authors = ", ".join(a["name"] for a in p.get("authors", [])[:4])
            if len(p.get("authors", [])) > 4:
                authors += " et al."
            pdf_url = ""
            if p.get("openAccessPdf"):
                pdf_url = f"\n**PDF (Open Access):** {p['openAccessPdf'].get('url','')}"

            results.append(
                f"**Title:** {p.get('title', 'N/A')}\n"
                f"**Authors:** {authors}\n"
                f"**Year:** {p.get('year', 'N/A')}\n"
                f"**Citations:** {p.get('citationCount', 0)}\n"
                f"**URL:** {p.get('url', 'N/A')}{pdf_url}\n"
                f"**Abstract:** {(p.get('abstract') or 'N/A')[:500]}...\n"
                f"{'─'*60}"
            )
        return f"## Semantic Scholar: '{query}'\n\n" + "\n\n".join(results)
    except httpx.HTTPStatusError as e:
        return f"Semantic Scholar HTTP error: {e.response.status_code}"
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def get_paper_citations(paper_id: str, limit: int = 10) -> str:
    """
    Returns papers that cite a given article (via Semantic Scholar).

    Args:
        paper_id: Semantic Scholar ID or DOI of the paper
        limit: Number of citations to return (default: 10)
    """
    try:
        with httpx.Client(timeout=30) as client:
            resp = client.get(
                f"{SEMANTIC_SCHOLAR_BASE}/paper/{paper_id}/citations",
                params={"limit": limit, "fields": "title,authors,year,citationCount"},
                headers=_ss_headers()
            )
            resp.raise_for_status()
            data = resp.json()

        citations = data.get("data", [])
        if not citations:
            return "No citations found or paper not located."

        lines = [f"## Papers citing '{paper_id}'\n"]
        for i, c in enumerate(citations, 1):
            p = c.get("citingPaper", {})
            authors = ", ".join(a["name"] for a in p.get("authors", [])[:3])
            lines.append(f"{i}. **{p.get('title','N/A')}** ({p.get('year','?')})\n   {authors} — {p.get('citationCount',0)} citações")
        
        return "\n".join(lines)
    except Exception as e:
        return f"Error fetching citations: {str(e)}"


# ─────────────────────────────────────────────
# PDF → MARKDOWN
# ─────────────────────────────────────────────

@tool
def extract_pdf_text(pdf_path: str) -> str:
    """
    Extracts and converts text from a local PDF file to readable Markdown.
    Use when the user uploads a paper PDF.

    Args:
        pdf_path: Local path to the PDF file
    """
    try:
        doc = fitz.open(pdf_path)
        pages_text = []
        for i, page in enumerate(doc):
            text = page.get_text("text")
            if text.strip():
                pages_text.append(f"### Page {i+1}\n{text}")

        full_text = "\n\n".join(pages_text)
        # Basic cleanup
        full_text = re.sub(r'\n{3,}', '\n\n', full_text)
        full_text = re.sub(r'- \n', '', full_text)  # Remove line-break hyphenation

        # Truncate if too long (to avoid exceeding LLM context)
        max_chars = 15000
        if len(full_text) > max_chars:
            full_text = full_text[:max_chars] + f"\n\n[... text truncated - {len(full_text)} total characters ...]"

        return f"## Content extracted from PDF: {os.path.basename(pdf_path)}\n\n{full_text}"
    except Exception as e:
        return f"Error processing PDF: {str(e)}"


@tool
def fetch_paper_from_url(url: str) -> str:
    """
    Fetches the content of a paper URL (HTML page or online PDF).
    Converts to clean text for analysis.

    Args:
        url: Paper URL (ArXiv page, Semantic Scholar, etc.)
    """
    try:
        import urllib.request
        from bs4 import BeautifulSoup
        
        headers = {"User-Agent": "Mozilla/5.0 (Academic Research Bot)"}
        req = urllib.request.Request(url, headers=headers)
        
        with urllib.request.urlopen(req, timeout=15) as response:
            content_type = response.headers.get("Content-Type", "")
            content = response.read()
        
        if "pdf" in content_type.lower():
            # Save temporarily and extract
            tmp_path = "/tmp/fetched_paper.pdf"
            with open(tmp_path, "wb") as f:
                f.write(content)
            return extract_pdf_text.invoke({"pdf_path": tmp_path})
        else:
            # HTML
            soup = BeautifulSoup(content, "html.parser")
            # Remove scripts, styles, navigation elements
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            text = soup.get_text(separator="\n", strip=True)
            text = re.sub(r'\n{3,}', '\n\n', text)
            return text[:8000]
    except Exception as e:
        return f"Error fetching URL: {str(e)}"
