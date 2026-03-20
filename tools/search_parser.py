"""
Parser utilities to extract structured metadata from search tool results
"""
import re
from typing import Optional


def parse_arxiv_results(result_text: str) -> list[dict]:
    """
    Parse ArXiv search results to extract paper metadata.

    Args:
        result_text: Raw text output from search_arxiv tool

    Returns:
        List of dictionaries with paper metadata
    """
    papers = []

    # Split by separator line
    paper_blocks = result_text.split("─" * 60)

    for block in paper_blocks:
        if not block.strip():
            continue

        # Extract fields using regex
        title_match = re.search(r'\*\*T[ií]tulo:\*\*\s*(.+?)(?=\n\*\*|\n\n|$)', block, re.DOTALL)
        authors_match = re.search(r'\*\*Autores:\*\*\s*(.+?)(?=\n\*\*|\n\n|$)', block)
        year_match = re.search(r'\*\*Publicado:\*\*\s*(\d{4})-\d{2}-\d{2}', block)
        arxiv_id_match = re.search(r'\*\*ArXiv ID:\*\*\s*(.+?)(?=\n\*\*|\n\n|$)', block)
        abstract_match = re.search(r'\*\*Resumo:\*\*\s*(.+?)(?=\n\*\*|\n─|$)', block, re.DOTALL)

        # English versions
        if not title_match:
            title_match = re.search(r'\*\*Title:\*\*\s*(.+?)(?=\n\*\*|\n\n|$)', block, re.DOTALL)
        if not authors_match:
            authors_match = re.search(r'\*\*Authors:\*\*\s*(.+?)(?=\n\*\*|\n\n|$)', block)
        if not year_match:
            year_match = re.search(r'\*\*Published:\*\*\s*(\d{4})-\d{2}-\d{2}', block)
        if not abstract_match:
            abstract_match = re.search(r'\*\*Abstract:\*\*\s*(.+?)(?=\n\*\*|\n─|$)', block, re.DOTALL)

        if title_match:
            paper = {
                "title": title_match.group(1).strip(),
                "authors": authors_match.group(1).strip() if authors_match else "Unknown",
                "year": year_match.group(1) if year_match else "N/A",
                "arxiv_id": arxiv_id_match.group(1).strip() if arxiv_id_match else "",
                "abstract": abstract_match.group(1).strip()[:500] if abstract_match else "",
                "source": "arxiv"
            }
            papers.append(paper)

    return papers


def parse_semantic_scholar_results(result_text: str) -> list[dict]:
    """
    Parse Semantic Scholar search results to extract paper metadata.

    Args:
        result_text: Raw text output from search_semantic_scholar tool

    Returns:
        List of dictionaries with paper metadata
    """
    papers = []

    # Split by separator line
    paper_blocks = result_text.split("─" * 60)

    for block in paper_blocks:
        if not block.strip():
            continue

        # Extract fields
        title_match = re.search(r'\*\*T[ií]tulo:\*\*\s*(.+?)(?=\n\*\*|\n\n|$)', block, re.DOTALL)
        authors_match = re.search(r'\*\*Autores:\*\*\s*(.+?)(?=\n\*\*|\n\n|$)', block)
        year_match = re.search(r'\*\*Ano:\*\*\s*(\d{4})', block)
        citations_match = re.search(r'\*\*Cita[çc][õo]es:\*\*\s*(\d+)', block)
        abstract_match = re.search(r'\*\*Resumo:\*\*\s*(.+?)(?=\n\*\*|\n─|$)', block, re.DOTALL)

        # English versions
        if not title_match:
            title_match = re.search(r'\*\*Title:\*\*\s*(.+?)(?=\n\*\*|\n\n|$)', block, re.DOTALL)
        if not authors_match:
            authors_match = re.search(r'\*\*Authors:\*\*\s*(.+?)(?=\n\*\*|\n\n|$)', block)
        if not year_match:
            year_match = re.search(r'\*\*Year:\*\*\s*(\d{4})', block)
        if not citations_match:
            citations_match = re.search(r'\*\*Citations:\*\*\s*(\d+)', block)
        if not abstract_match:
            abstract_match = re.search(r'\*\*Abstract:\*\*\s*(.+?)(?=\n\*\*|\n─|$)', block, re.DOTALL)

        if title_match:
            paper = {
                "title": title_match.group(1).strip(),
                "authors": authors_match.group(1).strip() if authors_match else "Unknown",
                "year": year_match.group(1) if year_match else "N/A",
                "citations": citations_match.group(1) if citations_match else "0",
                "abstract": abstract_match.group(1).strip()[:500] if abstract_match else "",
                "source": "semantic_scholar"
            }
            papers.append(paper)

    return papers


def extract_metadata_from_search_result(result_text: str, source: str) -> list[dict]:
    """
    Extract metadata from search results based on source.

    Args:
        result_text: Raw search result text
        source: 'arxiv' or 'semantic_scholar'

    Returns:
        List of paper metadata dictionaries
    """
    if "arxiv" in source.lower():
        return parse_arxiv_results(result_text)
    elif "semantic" in source.lower() or "scholar" in source.lower():
        return parse_semantic_scholar_results(result_text)
    else:
        return []


if __name__ == "__main__":
    # Test with sample ArXiv result
    sample_arxiv = """
    ## Resultados ArXiv para: 'malware detection'

    **Título:** MalGAN: Generating Adversarial Malware
    **Autores:** John Doe, Jane Smith et al.
    **Publicado:** 2023-05-15
    **ArXiv ID:** 2305.12345
    **Link:** https://arxiv.org/abs/2305.12345
    **PDF:** https://arxiv.org/pdf/2305.12345
    **Resumo:** This paper presents a novel approach...
    ────────────────────────────────────────────────────────────
    """

    papers = parse_arxiv_results(sample_arxiv)
    print("Parsed papers:", len(papers))
    if papers:
        print("First paper:", papers[0])