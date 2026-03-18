# tools/memory_store.py
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import os


class AcademicMemory:
    def __init__(self, persist_directory="./data/memory"):
        """Inicializa vector stores para conversas e papers"""
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
            # ↑ Modelo leve, gratuito, roda local (383MB)
        )

        # Vector store para conversas
        self.conversations_db = Chroma(
            collection_name="conversations",
            embedding_function=self.embeddings,
            persist_directory=f"{persist_directory}/conversations"
        )

        # Vector store para papers
        self.papers_db = Chroma(
            collection_name="papers",
            embedding_function=self.embeddings,
            persist_directory=f"{persist_directory}/papers"
        )

    def add_conversation(self, user_msg: str, agent_response: str, metadata: dict = None):
        """Salva interação usuário-agente"""
        doc = Document(
            page_content=f"USER: {user_msg}\nAGENT: {agent_response}",
            metadata={
                "timestamp": metadata.get("timestamp", ""),
                "agent_type": metadata.get("agent_type", ""),
                "topic": metadata.get("topic", ""),
                **metadata
            }
        )
        self.conversations_db.add_documents([doc])
        self.conversations_db.persist()

    def add_paper(self, paper_text: str, metadata: dict):
        """Indexa paper para busca futura"""
        # Divide paper em chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = splitter.split_text(paper_text)

        docs = [
            Document(
                page_content=chunk,
                metadata={
                    "title": metadata.get("title", ""),
                    "authors": metadata.get("authors", ""),
                    "year": metadata.get("year", ""),
                    "arxiv_id": metadata.get("arxiv_id", ""),
                    "chunk_index": i,
                    **metadata
                }
            )
            for i, chunk in enumerate(chunks)
        ]

        self.papers_db.add_documents(docs)
        self.papers_db.persist()

    def search_conversations(self, query: str, k: int = 5):
        """Busca conversas passadas similares"""
        results = self.conversations_db.similarity_search(query, k=k)
        return results

    def search_papers(self, query: str, k: int = 5, filter_year: int = None):
        """Busca papers indexados"""
        filter_dict = {}
        if filter_year:
            filter_dict["year"] = {"$gte": str(filter_year)}

        results = self.papers_db.similarity_search(
            query,
            k=k,
            filter=filter_dict if filter_dict else None
        )
        return results

    def get_research_context(self, query: str, k: int = 3):
        """Recupera contexto relevante para query atual"""
        # Busca conversas passadas
        past_convs = self.search_conversations(query, k=k)
        # Busca papers relevantes
        relevant_papers = self.search_papers(query, k=k)

        context = "## Contexto de Pesquisas Anteriores:\n\n"

        if past_convs:
            context += "### Conversas Passadas:\n"
            for doc in past_convs:
                context += f"- {doc.page_content[:200]}...\n"

        if relevant_papers:
            context += "\n### Papers Já Indexados:\n"
            for doc in relevant_papers:
                context += f"- [{doc.metadata.get('title', 'N/A')}] {doc.page_content[:150]}...\n"

        return context