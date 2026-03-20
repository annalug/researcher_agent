"""
PDF Content Extraction Utilities
Focuses on extracting key sections for research context
"""
import re
from typing import Optional


def extract_key_sections(pdf_text: str, max_chars: int = 5000) -> str:
    """
    Extract the most relevant sections from a paper for similarity search.

    Prioritizes: Title, Abstract, Introduction, Keywords

    Args:
        pdf_text: Full extracted PDF text
        max_chars: Maximum characters to return

    Returns:
        Concatenated key sections
    """
    if not pdf_text or len(pdf_text) < 100:
        return pdf_text

    # Normalize text
    text_lower = pdf_text.lower()

    # Find section markers (case-insensitive)
    sections = {
        "abstract": None,
        "introduction": None,
        "keywords": None,
    }

    # Common patterns for sections
    patterns = {
        "abstract": r"\babstract\b",
        "introduction": r"\b(introduction|1\.|i\.)\b",
        "keywords": r"\b(keywords|index terms)\b",
    }

    # Find section start positions
    for section, pattern in patterns.items():
        match = re.search(pattern, text_lower)
        if match:
            sections[section] = match.start()

    # Extract content
    key_content = []

    # 1. Try to get Abstract (most important for similarity)
    if sections["abstract"] is not None:
        start = sections["abstract"]
        # Find end: next section or 1500 chars
        end = start + 1500

        # Look for next section
        next_sections = [pos for pos in sections.values() if pos and pos > start]
        if next_sections:
            end = min(end, min(next_sections))

        abstract = pdf_text[start:end].strip()
        key_content.append(f"## Abstract\n{abstract}\n")

    # 2. Try to get Introduction
    if sections["introduction"] is not None:
        start = sections["introduction"]
        # Take up to 2500 chars from intro
        end = start + 2500

        intro = pdf_text[start:end].strip()
        key_content.append(f"## Introduction\n{intro}\n")

    # 3. Keywords if found
    if sections["keywords"] is not None:
        start = sections["keywords"]
        end = start + 300
        keywords = pdf_text[start:end].strip()
        key_content.append(f"## Keywords\n{keywords}\n")

    # Combine sections
    combined = "\n".join(key_content)

    # If sections not found, take first N chars intelligently
    if not combined.strip():
        # Skip potential metadata/headers (first 500 chars often noise)
        start = 500 if len(pdf_text) > 500 else 0
        combined = pdf_text[start:start + max_chars]

    # Truncate if too long
    if len(combined) > max_chars:
        combined = combined[:max_chars] + "\n\n[... content truncated ...]"

    return combined


def extract_paper_metadata(pdf_text: str) -> dict:
    """
    Extract metadata from paper (title, authors, year, etc.)

    Args:
        pdf_text: Full extracted PDF text

    Returns:
        Dictionary with extracted metadata
    """
    metadata = {
        "title": None,
        "authors": None,
        "year": None,
        "doi": None,
        "arxiv_id": None,
    }

    # First 1000 chars usually contain metadata
    header = pdf_text[:1000]

    # Extract year (4 digits)
    year_match = re.search(r'\b(19|20)\d{2}\b', header)
    if year_match:
        metadata["year"] = year_match.group()

    # Extract DOI
    doi_match = re.search(r'10\.\d{4,}/[^\s]+', header)
    if doi_match:
        metadata["doi"] = doi_match.group()

    # Extract ArXiv ID
    arxiv_match = re.search(r'arXiv:\s*(\d{4}\.\d{4,5})', header, re.IGNORECASE)
    if arxiv_match:
        metadata["arxiv_id"] = arxiv_match.group(1)

    # Title: usually the longest line in first 500 chars
    lines = header[:500].split('\n')
    longest_line = max(lines, key=len, default="")
    if len(longest_line) > 20:  # Reasonable title length
        metadata["title"] = longest_line.strip()

    return metadata


def create_search_query_from_pdf(pdf_text: str) -> Optional[str]:
    """
    Automatically generate a search query from PDF content.

    Args:
        pdf_text: Full extracted PDF text

    Returns:
        Search query string or None
    """
    # Extract abstract
    key_sections = extract_key_sections(pdf_text, max_chars=2000)

    if not key_sections:
        return None

    # This would ideally use the LLM to generate query
    # For now, extract first sentence of abstract as query
    sentences = re.split(r'[.!?]+', key_sections)

    # Find sentences with technical terms (have capital letters mid-sentence)
    technical_sentences = [
        s for s in sentences
        if len(s) > 50 and re.search(r'[A-Z]{2,}', s)
    ]

    if technical_sentences:
        # Take first technical sentence as query basis
        query_base = technical_sentences[0].strip()
        # Extract key terms (words with capitals or 4+ chars)
        words = query_base.split()
        key_terms = [
            w for w in words
            if len(w) > 4 or w[0].isupper()
        ][:8]  # Max 8 terms

        return " ".join(key_terms)

    return None


if __name__ == "__main__":
    # Test with sample paper text
    sample = """
    MalGAN: Generating Adversarial Malware Examples for Black-box Attacks

    John Doe, Jane Smith
    University of Example
    2023

    ABSTRACT

    Machine learning has shown promise in malware detection, but adversarial
    examples pose significant challenges. We propose MalGAN, a generative
    adversarial network approach to creating adversarial malware samples
    that evade black-box detectors while maintaining functionality.

    Keywords: malware detection, adversarial examples, GAN, deep learning

    1. INTRODUCTION

    Malware detection systems increasingly rely on machine learning models
    to identify malicious software. However, these systems are vulnerable
    to adversarial attacks where malware authors craft samples specifically
    designed to evade detection. In this work, we present MalGAN...
    """

    print("=" * 60)
    print("TESTING PDF EXTRACTION")
    print("=" * 60)

    print("\n1. Key Sections Extraction:")
    print("-" * 60)
    key_sections = extract_key_sections(sample)
    print(key_sections)

    print("\n2. Metadata Extraction:")
    print("-" * 60)
    metadata = extract_paper_metadata(sample)
    for key, value in metadata.items():
        print(f"{key}: {value}")

    print("\n3. Auto-generated Search Query:")
    print("-" * 60)
    query = create_search_query_from_pdf(sample)
    print(query)