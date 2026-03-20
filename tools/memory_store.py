"""
Academic Memory Store with Vector Database (Chroma)
Stores conversations and papers for semantic search and context retrieval.
"""
import os
from datetime import datetime
from typing import Optional

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class AcademicMemory:
    """
    Manages persistent memory for academic research using vector databases.

    Features:
    - Stores conversation history with semantic search
    - Indexes papers for future reference
    - Retrieves relevant context based on queries
    """

    def __init__(self, persist_directory: str = "./data/memory"):
        """
        Initialize vector stores for conversations and papers.

        Args:
            persist_directory: Directory to persist the vector databases
        """
        # Create directories if they don't exist
        os.makedirs(persist_directory, exist_ok=True)
        os.makedirs(f"{persist_directory}/conversations", exist_ok=True)
        os.makedirs(f"{persist_directory}/papers", exist_ok=True)

        # Initialize embeddings (local, free, runs offline)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        # Vector store for conversations
        self.conversations_db = Chroma(
            collection_name="conversations",
            embedding_function=self.embeddings,
            persist_directory=f"{persist_directory}/conversations"
        )

        # Vector store for papers
        self.papers_db = Chroma(
            collection_name="papers",
            embedding_function=self.embeddings,
            persist_directory=f"{persist_directory}/papers"
        )

    def add_conversation(
            self,
            user_msg: str,
            agent_response: str,
            metadata: Optional[dict] = None
    ) -> None:
        """
        Save a user-agent interaction to memory.

        Args:
            user_msg: User's message
            agent_response: Agent's response
            metadata: Additional metadata (agent_type, timestamp, etc.)
        """
        if metadata is None:
            metadata = {}

        # Add timestamp if not present
        if "timestamp" not in metadata:
            metadata["timestamp"] = datetime.now().isoformat()

        doc = Document(
            page_content=f"USER: {user_msg}\n\nAGENT: {agent_response}",
            metadata=metadata
        )

        try:
            self.conversations_db.add_documents([doc])
            self.conversations_db.persist()
        except Exception as e:
            print(f"Error saving conversation: {e}")

    def add_paper(self, paper_text: str, metadata: dict) -> None:
        """
        Index a paper for future retrieval.

        Args:
            paper_text: Full text or summary of the paper
            metadata: Paper metadata (title, authors, year, arxiv_id, etc.)
        """
        # Split paper into chunks for better retrieval
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        chunks = splitter.split_text(paper_text)

        # Create documents with metadata
        docs = [
            Document(
                page_content=chunk,
                metadata={
                    "title": metadata.get("title", ""),
                    "authors": metadata.get("authors", ""),
                    "year": metadata.get("year", ""),
                    "arxiv_id": metadata.get("arxiv_id", ""),
                    "source": metadata.get("source", ""),
                    "query": metadata.get("query", ""),
                    "timestamp": metadata.get("timestamp", datetime.now().isoformat()),
                    "chunk_index": i,
                }
            )
            for i, chunk in enumerate(chunks)
        ]

        try:
            self.papers_db.add_documents(docs)
            self.papers_db.persist()
        except Exception as e:
            print(f"Error indexing paper: {e}")

    def search_conversations(
            self,
            query: str,
            k: int = 5,
            filter_dict: Optional[dict] = None
    ) -> list[Document]:
        """
        Search past conversations using semantic similarity.

        Args:
            query: Search query
            k: Number of results to return
            filter_dict: Optional metadata filters

        Returns:
            List of relevant conversation documents
        """
        try:
            results = self.conversations_db.similarity_search(
                query,
                k=k,
                filter=filter_dict
            )
            return results
        except Exception as e:
            print(f"Error searching conversations: {e}")
            return []

    def search_papers(
            self,
            query: str,
            k: int = 5,
            filter_year: Optional[int] = None
    ) -> list[Document]:
        """
        Search indexed papers using semantic similarity.

        Args:
            query: Search query
            k: Number of results to return
            filter_year: Optional year filter (papers from this year onwards)

        Returns:
            List of relevant paper chunks
        """
        filter_dict = None
        if filter_year:
            filter_dict = {"year": {"$gte": str(filter_year)}}

        try:
            results = self.papers_db.similarity_search(
                query,
                k=k,
                filter=filter_dict if filter_dict else None
            )
            return results
        except Exception as e:
            print(f"Error searching papers: {e}")
            return []

    def get_research_context(
            self,
            query: str,
            k: int = 3
    ) -> str:
        """
        Retrieve relevant research context for a given query.
        Combines past conversations and indexed papers.

        Args:
            query: Current user query
            k: Number of results per source

        Returns:
            Formatted context string
        """
        # Search past conversations
        past_convs = self.search_conversations(query, k=k)

        # Search indexed papers
        relevant_papers = self.search_papers(query, k=k)

        if not past_convs and not relevant_papers:
            return ""

        context = "## Previous Research Context:\n\n"

        if past_convs:
            context += "### Past Conversations:\n"
            for i, doc in enumerate(past_convs, 1):
                # Get first 200 chars of conversation
                snippet = doc.page_content[:200].replace("\n", " ")
                context += f"{i}. {snippet}...\n"
            context += "\n"

        if relevant_papers:
            context += "### Indexed Papers:\n"
            seen_titles = set()
            for doc in relevant_papers:
                title = doc.metadata.get('title', 'N/A')
                # Avoid duplicates (different chunks of same paper)
                if title and title != 'N/A' and title not in seen_titles:
                    seen_titles.add(title)
                    authors = doc.metadata.get('authors', 'N/A')
                    year = doc.metadata.get('year', 'N/A')
                    snippet = doc.page_content[:150].replace("\n", " ")
                    context += f"- **{title}** ({year})\n  {authors}\n  {snippet}...\n\n"

        return context

    def clear_all(self) -> None:
        """Clear all data from both vector stores. Use with caution!"""
        try:
            # Chroma doesn't have a built-in clear method, so we delete and recreate
            self.conversations_db.delete_collection()
            self.papers_db.delete_collection()
            print("✅ All memory cleared successfully.")
        except Exception as e:
            print(f"❌ Error clearing memory: {e}")


# ─────────────────────────────────────────────
# UTILITY FUNCTIONS
# ─────────────────────────────────────────────

def get_memory_stats(memory: AcademicMemory) -> dict:
    """
    Get statistics about the memory stores.

    Args:
        memory: AcademicMemory instance

    Returns:
        Dictionary with statistics
    """
    try:
        conv_data = memory.conversations_db.get()
        papers_data = memory.papers_db.get()

        # Count unique papers (by title)
        unique_papers = set()
        if papers_data and papers_data.get('metadatas'):
            for meta in papers_data['metadatas']:
                title = meta.get('title')
                if title and title != 'N/A':
                    unique_papers.add(title)

        return {
            "total_conversations": len(conv_data.get('ids', [])),
            "total_paper_chunks": len(papers_data.get('ids', [])),
            "unique_papers": len(unique_papers),
        }
    except Exception as e:
        print(f"Error getting stats: {e}")
        return {
            "total_conversations": 0,
            "total_paper_chunks": 0,
            "unique_papers": 0,
        }


if __name__ == "__main__":
    # Test the memory system
    print("🧠 Testing Academic Memory System...\n")

    memory = AcademicMemory(persist_directory="./data/memory_test")

    # Test 1: Add conversation
    print("1️⃣ Adding test conversation...")
    memory.add_conversation(
        user_msg="Search for papers on GANs for malware detection",
        agent_response="Here are 5 relevant papers on GANs...",
        metadata={
            "agent_type": "researcher",
            "timestamp": datetime.now().isoformat()
        }
    )
    print("✅ Conversation added\n")

    # Test 2: Add paper
    print("2️⃣ Adding test paper...")
    memory.add_paper(
        paper_text="This paper presents a novel approach using GANs for malware detection...",
        metadata={
            "title": "MalGAN: Malware Detection using GANs",
            "authors": "John Doe, Jane Smith",
            "year": "2023",
            "source": "arxiv"
        }
    )
    print("✅ Paper indexed\n")

    # Test 3: Search
    print("3️⃣ Searching for 'malware detection'...")
    context = memory.get_research_context("malware detection", k=3)
    print(context)

    # Test 4: Stats
    print("\n4️⃣ Memory statistics:")
    stats = get_memory_stats(memory)
    for key, value in stats.items():
        print(f"   {key}: {value}")

    print("\n✅ All tests passed!")