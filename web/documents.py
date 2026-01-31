"""
Document Processing & RAG Module
================================

Uses open source libraries for document processing and retrieval:
- Unstructured.io: Document parsing (PDF, DOCX, PPTX, etc.)
- ChromaDB: Local vector database
- sentence-transformers: Local embeddings

Provides:
- Document upload and parsing
- Vector storage and retrieval
- Chat with documents/folders context
"""

import os
import json
import hashlib
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

# Storage paths
JOTTY_DIR = Path.home() / ".jotty"
DOCUMENTS_DIR = JOTTY_DIR / "documents"
VECTORS_DIR = JOTTY_DIR / "vectors"
DOCS_INDEX_FILE = JOTTY_DIR / "documents_index.json"


class DocumentProcessor:
    """
    Handles document parsing, embedding, and storage.

    Uses:
    - unstructured: For parsing various document formats
    - chromadb: For vector storage
    - sentence-transformers: For generating embeddings
    """

    def __init__(self):
        self._chroma_client = None
        self._collection = None
        self._embedding_model = None
        self._docs_index = self._load_docs_index()

        # Ensure directories exist
        DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
        VECTORS_DIR.mkdir(parents=True, exist_ok=True)

    def _load_docs_index(self) -> Dict[str, Any]:
        """Load documents index from disk."""
        if DOCS_INDEX_FILE.exists():
            try:
                return json.loads(DOCS_INDEX_FILE.read_text())
            except Exception as e:
                logger.error(f"Failed to load docs index: {e}")
        return {"documents": {}, "folders": {}}

    def _save_docs_index(self):
        """Save documents index to disk."""
        try:
            DOCS_INDEX_FILE.write_text(json.dumps(self._docs_index, indent=2, default=str))
        except Exception as e:
            logger.error(f"Failed to save docs index: {e}")

    @property
    def chroma_client(self):
        """Lazy-load ChromaDB client."""
        if self._chroma_client is None:
            try:
                import chromadb
                from chromadb.config import Settings

                self._chroma_client = chromadb.PersistentClient(
                    path=str(VECTORS_DIR),
                    settings=Settings(anonymized_telemetry=False)
                )
                logger.info(f"ChromaDB initialized at {VECTORS_DIR}")
            except ImportError:
                logger.warning("ChromaDB not installed. Run: pip install chromadb")
                raise
        return self._chroma_client

    @property
    def collection(self):
        """Get or create the documents collection."""
        if self._collection is None:
            self._collection = self.chroma_client.get_or_create_collection(
                name="jotty_documents",
                metadata={"hnsw:space": "cosine"}
            )
        return self._collection

    @property
    def embedding_model(self):
        """Lazy-load sentence-transformers model."""
        if self._embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer

                # Use a small, fast model for local embeddings
                model_name = "all-MiniLM-L6-v2"
                self._embedding_model = SentenceTransformer(model_name)
                logger.info(f"Loaded embedding model: {model_name}")
            except ImportError:
                logger.warning("sentence-transformers not installed. Run: pip install sentence-transformers")
                raise
        return self._embedding_model

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    def parse_document(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        """
        Parse a document using unstructured.io.

        Returns:
            Tuple of (extracted_text, metadata)
        """
        try:
            from unstructured.partition.auto import partition

            elements = partition(filename=file_path)

            # Extract text from elements
            texts = []
            element_types = {}

            for element in elements:
                text = str(element)
                if text.strip():
                    texts.append(text)
                    elem_type = type(element).__name__
                    element_types[elem_type] = element_types.get(elem_type, 0) + 1

            full_text = "\n\n".join(texts)

            metadata = {
                "element_count": len(elements),
                "element_types": element_types,
                "char_count": len(full_text),
                "word_count": len(full_text.split()),
            }

            return full_text, metadata

        except ImportError:
            logger.warning("unstructured not installed. Run: pip install 'unstructured[all-docs]'")
            # Fallback: try basic text extraction
            return self._fallback_parse(file_path)
        except Exception as e:
            logger.error(f"Failed to parse document: {e}")
            return self._fallback_parse(file_path)

    def _fallback_parse(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        """Fallback parsing for when unstructured is not available."""
        path = Path(file_path)
        suffix = path.suffix.lower()

        try:
            if suffix == '.txt':
                text = path.read_text(encoding='utf-8')
            elif suffix == '.md':
                text = path.read_text(encoding='utf-8')
            elif suffix == '.json':
                data = json.loads(path.read_text())
                text = json.dumps(data, indent=2)
            elif suffix == '.csv':
                import csv
                with open(path, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    rows = list(reader)
                text = "\n".join([", ".join(row) for row in rows])
            elif suffix == '.pdf':
                # Try pypdf as fallback
                try:
                    from pypdf import PdfReader
                    reader = PdfReader(file_path)
                    text = "\n\n".join([page.extract_text() or "" for page in reader.pages])
                except ImportError:
                    text = f"[PDF file - install pypdf or unstructured for parsing: {path.name}]"
            else:
                text = f"[Unsupported file type: {suffix}. Install unstructured for full support.]"

            return text, {"fallback": True, "char_count": len(text)}

        except Exception as e:
            return f"[Error reading file: {e}]", {"error": str(e)}

    def chunk_text(self, text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks for embedding."""
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size

            # Try to end at a sentence boundary
            if end < len(text):
                # Look for sentence end within last 100 chars
                for sep in ['. ', '.\n', '? ', '!\n', '\n\n']:
                    last_sep = text[end-100:end].rfind(sep)
                    if last_sep != -1:
                        end = end - 100 + last_sep + len(sep)
                        break

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            start = end - overlap

        return chunks

    async def upload_document(
        self,
        file_content: bytes,
        filename: str,
        folder_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Upload and process a document.

        Args:
            file_content: Raw file bytes
            filename: Original filename
            folder_id: Optional folder to associate with
            metadata: Optional additional metadata

        Returns:
            Document info dict
        """
        # Generate document ID
        doc_hash = hashlib.sha256(file_content).hexdigest()[:16]
        doc_id = f"doc_{doc_hash}"

        # Save file
        file_ext = Path(filename).suffix.lower()
        doc_path = DOCUMENTS_DIR / f"{doc_id}{file_ext}"
        doc_path.write_bytes(file_content)

        # Parse document
        text, parse_meta = self.parse_document(str(doc_path))

        # Chunk and embed
        chunks = self.chunk_text(text)

        try:
            embeddings = self.generate_embeddings(chunks)

            # Store in ChromaDB
            chunk_ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
            chunk_metas = [
                {
                    "doc_id": doc_id,
                    "chunk_index": i,
                    "folder_id": folder_id or "",
                    "filename": filename,
                }
                for i in range(len(chunks))
            ]

            self.collection.add(
                ids=chunk_ids,
                embeddings=embeddings,
                documents=chunks,
                metadatas=chunk_metas
            )

            embedded = True
        except Exception as e:
            logger.error(f"Failed to embed document: {e}")
            embedded = False

        # Create document record
        doc_info = {
            "id": doc_id,
            "filename": filename,
            "folder_id": folder_id,
            "file_path": str(doc_path),
            "file_size": len(file_content),
            "file_type": file_ext,
            "uploaded_at": datetime.now().isoformat(),
            "text_length": len(text),
            "chunk_count": len(chunks),
            "embedded": embedded,
            "metadata": {**(metadata or {}), **parse_meta}
        }

        # Save to index
        self._docs_index["documents"][doc_id] = doc_info
        if folder_id:
            if folder_id not in self._docs_index["folders"]:
                self._docs_index["folders"][folder_id] = []
            if doc_id not in self._docs_index["folders"][folder_id]:
                self._docs_index["folders"][folder_id].append(doc_id)

        self._save_docs_index()

        return doc_info

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get document info by ID."""
        return self._docs_index["documents"].get(doc_id)

    def get_document_text(self, doc_id: str) -> Optional[str]:
        """Get the full text of a document."""
        doc = self.get_document(doc_id)
        if not doc:
            return None

        file_path = doc.get("file_path")
        if file_path and Path(file_path).exists():
            text, _ = self.parse_document(file_path)
            return text
        return None

    def get_folder_documents(self, folder_id: str) -> List[Dict[str, Any]]:
        """Get all documents in a folder."""
        doc_ids = self._docs_index["folders"].get(folder_id, [])
        return [
            self._docs_index["documents"][doc_id]
            for doc_id in doc_ids
            if doc_id in self._docs_index["documents"]
        ]

    def delete_document(self, doc_id: str) -> bool:
        """Delete a document and its embeddings."""
        doc = self.get_document(doc_id)
        if not doc:
            return False

        # Delete file
        file_path = doc.get("file_path")
        if file_path:
            try:
                Path(file_path).unlink(missing_ok=True)
            except Exception as e:
                logger.error(f"Failed to delete file: {e}")

        # Delete from ChromaDB
        try:
            # Find and delete all chunks
            chunk_count = doc.get("chunk_count", 100)
            chunk_ids = [f"{doc_id}_chunk_{i}" for i in range(chunk_count)]
            self.collection.delete(ids=chunk_ids)
        except Exception as e:
            logger.error(f"Failed to delete embeddings: {e}")

        # Remove from index
        folder_id = doc.get("folder_id")
        if folder_id and folder_id in self._docs_index["folders"]:
            self._docs_index["folders"][folder_id] = [
                d for d in self._docs_index["folders"][folder_id] if d != doc_id
            ]

        del self._docs_index["documents"][doc_id]
        self._save_docs_index()

        return True

    def search_documents(
        self,
        query: str,
        folder_id: Optional[str] = None,
        doc_ids: Optional[List[str]] = None,
        n_results: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search documents using vector similarity.

        Args:
            query: Search query
            folder_id: Optional filter by folder
            doc_ids: Optional filter by specific documents
            n_results: Number of results to return

        Returns:
            List of relevant chunks with metadata
        """
        try:
            # Generate query embedding
            query_embedding = self.generate_embeddings([query])[0]

            # Build where filter
            where_filter = None
            if folder_id:
                where_filter = {"folder_id": folder_id}
            elif doc_ids:
                where_filter = {"doc_id": {"$in": doc_ids}}

            # Search
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where_filter,
                include=["documents", "metadatas", "distances"]
            )

            # Format results
            chunks = []
            for i, doc in enumerate(results["documents"][0]):
                chunks.append({
                    "text": doc,
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i],
                    "relevance": 1 - results["distances"][0][i],  # cosine similarity
                })

            return chunks

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def get_context_for_chat(
        self,
        query: str,
        context_type: str,
        context_id: str,
        max_tokens: int = 2000
    ) -> str:
        """
        Get relevant context for chat based on context type.

        Args:
            query: User's question
            context_type: 'folder', 'document', or 'chat'
            context_id: ID of the folder, document, or chat
            max_tokens: Approximate max tokens for context

        Returns:
            Formatted context string
        """
        context_parts = []

        if context_type == "folder":
            # Get relevant chunks from folder documents
            chunks = self.search_documents(query, folder_id=context_id, n_results=8)

            for chunk in chunks:
                if chunk["relevance"] > 0.3:  # Relevance threshold
                    filename = chunk["metadata"].get("filename", "Unknown")
                    context_parts.append(f"[From: {filename}]\n{chunk['text']}")

        elif context_type == "document":
            # Get relevant chunks from specific document
            chunks = self.search_documents(query, doc_ids=[context_id], n_results=5)

            for chunk in chunks:
                context_parts.append(chunk["text"])

        elif context_type == "chat":
            # For chat context, we'd load the chat history
            # This would be implemented separately
            pass

        # Combine and truncate
        combined = "\n\n---\n\n".join(context_parts)

        # Rough token estimation (4 chars per token)
        if len(combined) > max_tokens * 4:
            combined = combined[:max_tokens * 4] + "\n\n[Context truncated...]"

        return combined


# Global processor instance
_processor: Optional[DocumentProcessor] = None


def get_document_processor() -> DocumentProcessor:
    """Get the global document processor instance."""
    global _processor
    if _processor is None:
        _processor = DocumentProcessor()
    return _processor
