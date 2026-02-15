"""
Document routes - upload, list, search, RAG config.
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def register_document_routes(app, api):
    from fastapi import File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
    from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
    from pydantic import BaseModel

    @app.post("/api/documents/upload")
    async def upload_document(file: UploadFile = File(...), folder_id: Optional[str] = Form(None)):
        """
        Upload a document for RAG processing.

        Supported formats: PDF, DOCX, PPTX, TXT, MD, CSV, JSON, HTML
        """
        from .documents import get_document_processor

        try:
            processor = get_document_processor()
            content = await file.read()

            doc_info = await processor.upload_document(
                file_content=content, filename=file.filename, folder_id=folder_id
            )

            return {"success": True, "document": doc_info}
        except ImportError as e:
            raise HTTPException(
                status_code=501,
                detail=f"Document processing not available. Install dependencies: pip install chromadb sentence-transformers unstructured[all-docs]. Error: {str(e)}",
            )
        except Exception as e:
            logger.error(f"Upload failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/documents")
    async def list_documents(folder_id: Optional[str] = None):
        """List all documents, optionally filtered by folder."""
        from .documents import get_document_processor

        try:
            processor = get_document_processor()

            if folder_id:
                docs = processor.get_folder_documents(folder_id)
            else:
                docs = list(processor._docs_index.get("documents", {}).values())

            return {"documents": docs}
        except Exception as e:
            logger.error(f"Failed to list documents: {e}")
            return {"documents": []}

    @app.get("/api/documents/{doc_id}")
    async def get_document(doc_id: str, include_text: bool = False):
        """Get document info and optionally its text content."""
        from .documents import get_document_processor

        processor = get_document_processor()
        doc = processor.get_document(doc_id)

        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")

        result = {"document": doc}
        if include_text:
            result["text"] = processor.get_document_text(doc_id)

        return result

    @app.delete("/api/documents/{doc_id}")
    async def delete_document(doc_id: str):
        """Delete a document and its embeddings."""
        from .documents import get_document_processor

        processor = get_document_processor()
        success = processor.delete_document(doc_id)

        return {"success": success}

    @app.post("/api/documents/search")
    async def search_documents(request: dict):
        """
        Search documents using vector similarity.

        Args:
            query: Search text
            folder_id: Optional folder filter
            doc_ids: Optional list of document IDs to search
            n_results: Number of results (default 5)
        """
        from .documents import get_document_processor

        try:
            processor = get_document_processor()

            results = processor.search_documents(
                query=request.get("query", ""),
                folder_id=request.get("folder_id"),
                doc_ids=request.get("doc_ids"),
                n_results=request.get("n_results", 5),
            )

            return {"results": results}
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # ===== RAG CONFIGURATION ENDPOINTS =====

    @app.get("/api/rag/config")
    async def get_rag_config():
        """Get current RAG configuration."""
        from .documents import RAGConfig, get_document_processor

        processor = get_document_processor()
        return {
            "config": processor.config.to_dict(),
            "available_models": RAGConfig.EMBEDDING_MODELS,
        }

    class RAGConfigUpdateRequest(BaseModel):
        chunk_size: Optional[int] = None
        overlap: Optional[int] = None
        embedding_model: Optional[str] = None

    @app.post("/api/rag/config")
    async def update_rag_config(request: RAGConfigUpdateRequest):
        """Update RAG configuration."""
        from .documents import RAGConfig, get_document_processor

        processor = get_document_processor()

        # Validate embedding model
        if request.embedding_model and request.embedding_model not in RAGConfig.EMBEDDING_MODELS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid embedding model. Available: {list(RAGConfig.EMBEDDING_MODELS.keys())}",
            )

        # Validate chunk_size
        if request.chunk_size is not None and (
            request.chunk_size < 100 or request.chunk_size > 4000
        ):
            raise HTTPException(status_code=400, detail="chunk_size must be between 100 and 4000")

        # Validate overlap
        if request.overlap is not None and (request.overlap < 0 or request.overlap > 500):
            raise HTTPException(status_code=400, detail="overlap must be between 0 and 500")

        config = processor.update_config(
            chunk_size=request.chunk_size,
            overlap=request.overlap,
            embedding_model=request.embedding_model,
        )

        return {"success": True, "config": config.to_dict()}

    @app.post("/api/rag/reindex/{doc_id}")
    async def reindex_document(doc_id: str):
        """Re-index a document with current RAG settings."""
        from .documents import get_document_processor

        processor = get_document_processor()
        doc = processor.get_document(doc_id)

        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")

        try:
            # Get document text
            text = processor.get_document_text(doc_id)
            if not text:
                raise HTTPException(status_code=500, detail="Failed to read document")

            # Delete old embeddings
            chunk_count = doc.get("chunk_count", 100)
            chunk_ids = [f"{doc_id}_chunk_{i}" for i in range(chunk_count)]
            try:
                processor.collection.delete(ids=chunk_ids)
            except Exception:
                pass

            # Re-chunk and re-embed
            chunks = processor.chunk_text(text)
            embeddings = processor.generate_embeddings(chunks)

            new_chunk_ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
            chunk_metas = [
                {
                    "doc_id": doc_id,
                    "chunk_index": i,
                    "folder_id": doc.get("folder_id") or "",
                    "filename": doc.get("filename", ""),
                    "embedding_model": processor.config.embedding_model,
                    "chunk_size": processor.config.chunk_size,
                }
                for i in range(len(chunks))
            ]

            processor.collection.add(
                ids=new_chunk_ids, embeddings=embeddings, documents=chunks, metadatas=chunk_metas
            )

            # Update document info
            doc["chunk_count"] = len(chunks)
            doc["rag_config"] = processor.config.to_dict()
            doc["reindexed_at"] = datetime.now().isoformat()
            processor._save_docs_index()

            return {
                "success": True,
                "doc_id": doc_id,
                "chunk_count": len(chunks),
                "config": processor.config.to_dict(),
            }

        except Exception as e:
            logger.error(f"Reindex failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    class ChatWithContextRequest(BaseModel):
        message: str
        context_type: str  # 'folder', 'document', 'chat'
        context_id: str
        session_id: Optional[str] = None
