"""
RAG Pipeline for Connected Car Security Research
Handles document ingestion, chunking, embedding, and retrieval.
"""

import os
import json
import pickle
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Document:
    """Represents a text chunk with metadata."""
    content: str
    metadata: Dict = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None

    def to_dict(self) -> Dict:
        return {
            "content": self.content,
            "metadata": self.metadata,
        }


class TextSplitter:
    """Splits text into overlapping chunks."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str, source: str = "document") -> List[Document]:
        """Split text into chunks with overlap."""
        words = text.split()
        chunks = []
        start = 0
        chunk_idx = 0

        while start < len(words):
            end = start + self.chunk_size
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)

            chunks.append(Document(
                content=chunk_text,
                metadata={
                    "source": source,
                    "chunk_index": chunk_idx,
                    "word_start": start,
                    "word_end": min(end, len(words)),
                }
            ))

            start = end - self.chunk_overlap
            chunk_idx += 1

        return chunks

    def split_by_sections(self, text: str, source: str = "document") -> List[Document]:
        """Split text by logical sections (numbered headings)."""
        import re
        # Split on section headings like "1  Introduction" or "4.1 CAN Bus"
        section_pattern = re.compile(r'\n(\d+[\.\d]*\s{1,2}[A-Z][^\n]+)', re.MULTILINE)
        parts = section_pattern.split(text)

        documents = []
        current_section = "Preamble"

        for i, part in enumerate(parts):
            part = part.strip()
            if not part:
                continue
            if section_pattern.match('\n' + part) or (i % 2 == 1 and re.match(r'\d+', part)):
                current_section = part
            else:
                if len(part) > 50:  # skip tiny fragments
                    sub_docs = self.split_text(part, source)
                    for doc in sub_docs:
                        doc.metadata["section"] = current_section
                    documents.extend(sub_docs)

        return documents if documents else self.split_text(text, source)


class EmbeddingModel:
    """Manages text embeddings using sentence-transformers or OpenAI."""

    def __init__(self, model_type: str = "sentence_transformers",
                 model_name: str = "all-MiniLM-L6-v2",
                 openai_api_key: Optional[str] = None):
        self.model_type = model_type
        self.model_name = model_name
        self.model = None
        self._initialize(openai_api_key)

    def _initialize(self, openai_api_key: Optional[str]):
        if self.model_type == "sentence_transformers":
            try:
                from sentence_transformers import SentenceTransformer
                print(f"[Embeddings] Loading {self.model_name}...")
                self.model = SentenceTransformer(self.model_name)
                print("[Embeddings] Model loaded successfully.")
            except ImportError:
                raise ImportError("Run: pip install sentence-transformers")

        elif self.model_type == "openai":
            try:
                import openai
                self.client = openai.OpenAI(api_key=openai_api_key or os.getenv("OPENAI_API_KEY"))
                self.model_name = "text-embedding-3-small"
            except ImportError:
                raise ImportError("Run: pip install openai")

    def embed(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        if self.model_type == "sentence_transformers":
            return self.model.encode(texts, show_progress_bar=False, normalize_embeddings=True)

        elif self.model_type == "openai":
            response = self.client.embeddings.create(input=texts, model=self.model_name)
            embeddings = [item.embedding for item in response.data]
            arr = np.array(embeddings, dtype=np.float32)
            # Normalize
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            return arr / np.maximum(norms, 1e-10)

    def embed_single(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


class VectorStore:
    """Simple in-memory vector store with cosine similarity search."""

    def __init__(self):
        self.documents: List[Document] = []
        self.embeddings: Optional[np.ndarray] = None

    def add_documents(self, documents: List[Document], embeddings: np.ndarray):
        """Add documents with their precomputed embeddings."""
        for doc, emb in zip(documents, embeddings):
            doc.embedding = emb
        self.documents.extend(documents)
        if self.embeddings is None:
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[Document, float]]:
        """Return top-k documents by cosine similarity."""
        if self.embeddings is None or len(self.documents) == 0:
            return []

        # Cosine similarity (embeddings already normalized)
        scores = self.embeddings @ query_embedding
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append((self.documents[idx], float(scores[idx])))
        return results

    def save(self, path: str):
        """Persist vector store to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "documents": [(d.content, d.metadata) for d in self.documents],
                "embeddings": self.embeddings,
            }, f)
        print(f"[VectorStore] Saved {len(self.documents)} documents to {path}")

    def load(self, path: str) -> bool:
        """Load vector store from disk."""
        path = Path(path)
        if not path.exists():
            return False
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.documents = [
            Document(content=c, metadata=m)
            for c, m in data["documents"]
        ]
        self.embeddings = data["embeddings"]
        for doc, emb in zip(self.documents, self.embeddings):
            doc.embedding = emb
        print(f"[VectorStore] Loaded {len(self.documents)} documents from {path}")
        return True


class RAGPipeline:
    """
    End-to-end Retrieval-Augmented Generation pipeline.
    Supports: sentence-transformers (local) or OpenAI embeddings + LLM.
    """

    def __init__(self, config: Dict):
        self.config = config
        self.splitter = TextSplitter(
            chunk_size=config.get("chunk_size", 400),
            chunk_overlap=config.get("chunk_overlap", 50),
        )
        self.embedder = EmbeddingModel(
            model_type=config.get("embedding_type", "sentence_transformers"),
            model_name=config.get("embedding_model", "all-MiniLM-L6-v2"),
            openai_api_key=config.get("openai_api_key"),
        )
        self.vector_store = VectorStore()
        self.store_path = config.get("store_path", "data/vector_store.pkl")

        # Try loading existing store
        self.vector_store.load(self.store_path)

    def ingest_text(self, text: str, source: str = "research_paper"):
        """Ingest raw text into the vector store."""
        print(f"[RAG] Chunking document: {source}")
        chunks = self.splitter.split_by_sections(text, source)
        print(f"[RAG] Created {len(chunks)} chunks. Embedding...")

        texts = [c.content for c in chunks]
        embeddings = self.embedder.embed(texts)

        self.vector_store.add_documents(chunks, embeddings)
        self.vector_store.save(self.store_path)
        print(f"[RAG] Ingested {len(chunks)} chunks from '{source}'")
        return len(chunks)

    def ingest_pdf(self, pdf_path: str):
        """Extract text from a PDF and ingest it."""
        try:
            import pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                text = "\n".join(page.extract_text() or "" for page in pdf.pages)
        except ImportError:
            try:
                from pypdf import PdfReader
                reader = PdfReader(pdf_path)
                text = "\n".join(page.extract_text() or "" for page in reader.pages)
            except ImportError:
                raise ImportError("Run: pip install pdfplumber or pypdf")

        source = Path(pdf_path).stem
        return self.ingest_text(text, source)

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[Document, float]]:
        """Retrieve relevant chunks for a query."""
        query_emb = self.embedder.embed_single(query)
        return self.vector_store.search(query_emb, top_k=top_k)

    def format_context(self, results: List[Tuple[Document, float]]) -> str:
        """Format retrieved chunks into a context string."""
        context_parts = []
        for i, (doc, score) in enumerate(results, 1):
            section = doc.metadata.get("section", "Unknown")
            context_parts.append(
                f"[Context {i} | Section: {section} | Relevance: {score:.3f}]\n{doc.content}"
            )
        return "\n\n---\n\n".join(context_parts)

    def get_stats(self) -> Dict:
        return {
            "total_chunks": len(self.vector_store.documents),
            "embedding_model": self.embedder.model_name,
            "embedding_type": self.embedder.model_type,
            "store_path": self.store_path,
        }
