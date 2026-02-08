"""
Session routes - CRUD, folders, branching, temporary sessions.
"""

import asyncio
import json
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)



def register_sessions_routes(app, api):
    from fastapi import HTTPException, WebSocket, WebSocketDisconnect, UploadFile, File, Form
    from fastapi.responses import FileResponse, JSONResponse, StreamingResponse, HTMLResponse
    from pydantic import BaseModel

    @app.get("/api/sessions")
    async def list_sessions():
        """List all sessions."""
        sessions = api.get_sessions()
        return {"sessions": sessions}

    @app.get("/api/sessions/{session_id}")
    async def get_session(session_id: str):
        """Get session details and history."""
        session = api.get_session(session_id)
        if session:
            return session
        raise HTTPException(status_code=404, detail="Session not found")

    @app.delete("/api/sessions/{session_id}")
    async def delete_session(session_id: str):
        """Delete a session."""
        success = api.delete_session(session_id)
        return {"success": success}

    @app.post("/api/sessions/{session_id}/clear")
    async def clear_session(session_id: str):
        """Clear session history."""
        success = api.clear_session(session_id)
        return {"success": success}

    class SessionUpdateRequest(BaseModel):
        title: Optional[str] = None
        isPinned: Optional[bool] = None
        isArchived: Optional[bool] = None
        folderId: Optional[str] = None

    @app.patch("/api/sessions/{session_id}")
    async def update_session(session_id: str, request: SessionUpdateRequest):
        """Update session metadata (title, pin, archive, folder)."""
        updates = request.dict(exclude_none=True)
        success = api.update_session(session_id, updates)
        return {"success": success}

    # Folder management endpoints
    @app.get("/api/folders")
    async def list_folders():
        """List all folders."""
        folders = api.get_folders()
        return {"folders": folders}

    class FolderRequest(BaseModel):
        id: str
        name: str
        color: str = "#3b82f6"
        order: int = 0

    @app.post("/api/folders")
    async def create_folder(request: FolderRequest):
        """Create a new folder."""
        folder = request.dict()
        success = api.create_folder(folder)
        return {"success": success, "folder": folder}

    @app.delete("/api/folders/{folder_id}")
    async def delete_folder(folder_id: str):
        """Delete a folder."""
        success = api.delete_folder(folder_id)
        return {"success": success}

    class BulkFoldersRequest(BaseModel):
        folders: List[dict]

    @app.put("/api/folders/bulk")
    async def bulk_update_folders(request: BulkFoldersRequest):
        """Bulk update all folders (for reordering, renaming, etc.)."""
        success = api.save_folders(request.folders)
        return {"success": success}

    # ===== DOCUMENT UPLOAD & RAG ENDPOINTS =====

    @app.get("/api/sessions/{session_id}/branches")
    async def list_branches(session_id: str):
        """Get all branches for a session."""
        from ..cli.repl.session import InterfaceType

        registry = api._get_session_registry()
        session = registry.get_session(session_id, create=False, interface=InterfaceType.WEB)

        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        return {
            "branches": session.get_branches(),
            "active_branch": getattr(session, 'active_branch', 'main'),
            "tree": session.get_branch_tree()
        }

    class CreateBranchRequest(BaseModel):
        from_message_id: str
        branch_name: Optional[str] = None

    @app.post("/api/sessions/{session_id}/branch")
    async def create_branch(session_id: str, request: CreateBranchRequest):
        """Create a new branch from a message."""
        from ..cli.repl.session import InterfaceType

        registry = api._get_session_registry()
        session = registry.get_session(session_id, create=False, interface=InterfaceType.WEB)

        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        try:
            branch_id = session.create_branch(
                from_message_id=request.from_message_id,
                branch_name=request.branch_name
            )
            return {
                "success": True,
                "branch_id": branch_id,
                "branches": session.get_branches()
            }
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    class EditMessageRequest(BaseModel):
        new_content: str
        create_branch: bool = True

    @app.post("/api/sessions/{session_id}/messages/{message_id}/edit")
    async def edit_message(session_id: str, message_id: str, request: EditMessageRequest):
        """Edit a message, optionally creating a new branch."""
        from ..cli.repl.session import InterfaceType

        registry = api._get_session_registry()
        session = registry.get_session(session_id, create=False, interface=InterfaceType.WEB)

        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        try:
            branch_id = session.edit_message(
                message_id=message_id,
                new_content=request.new_content,
                create_branch=request.create_branch
            )
            return {
                "success": True,
                "branch_id": branch_id,
                "branches": session.get_branches()
            }
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    class SwitchBranchRequest(BaseModel):
        branch_id: str

    @app.post("/api/sessions/{session_id}/branch/switch")
    async def switch_branch(session_id: str, request: SwitchBranchRequest):
        """Switch to a different branch."""
        from ..cli.repl.session import InterfaceType

        registry = api._get_session_registry()
        session = registry.get_session(session_id, create=False, interface=InterfaceType.WEB)

        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        try:
            session.switch_branch(request.branch_id)
            return {
                "success": True,
                "active_branch": request.branch_id,
                "history": session.get_history()
            }
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.delete("/api/sessions/{session_id}/branches/{branch_id}")
    async def delete_branch(session_id: str, branch_id: str):
        """Delete a branch."""
        from ..cli.repl.session import InterfaceType

        registry = api._get_session_registry()
        session = registry.get_session(session_id, create=False, interface=InterfaceType.WEB)

        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        try:
            session.delete_branch(branch_id)
            return {
                "success": True,
                "branches": session.get_branches()
            }
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    # ===== WEB SEARCH ENDPOINTS (Uses existing web-search skill) =====

    class WebSearchRequest(BaseModel):
        query: str
        max_results: int = 10

    @app.get("/api/share/session/{session_id}")
    async def get_session_share_links(session_id: str):
        """Get all share links for a session."""
        from cli.repl.session import get_share_link_registry

        share_registry = get_share_link_registry()
        links = share_registry.get_session_links(session_id)

        return {"links": [link.to_dict() for link in links]}

    class CreateTempSessionRequest(BaseModel):
        expiry_days: Optional[int] = 30

    @app.post("/api/sessions/temporary")
    async def create_temporary_session(request: CreateTempSessionRequest = None):
        """Create a temporary (ephemeral) chat session."""
        from cli.repl.session import SessionManager, InterfaceType

        expiry_days = (request.expiry_days if request else None) or 30
        session = SessionManager(
            interface=InterfaceType.WEB,
            is_temporary=True
        )
        session.set_temporary(True, expiry_days)

        return {
            "session_id": session.session_id,
            "is_temporary": True,
            "expires_at": session.expires_at.isoformat() if session.expires_at else None
        }

    @app.post("/api/sessions/{session_id}/temporary")
    async def toggle_session_temporary(session_id: str, is_temporary: bool, expiry_days: Optional[int] = 30):
        """Toggle temporary mode for a session."""
        from cli.repl.session import get_session_registry

        registry = get_session_registry()
        session = registry.get_session(session_id, create=False)

        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        session.set_temporary(is_temporary, expiry_days)

        if not is_temporary:
            # Save the session now that it's permanent
            session.auto_save = True
            session.save()

        return {
            "session_id": session_id,
            "is_temporary": session.is_temporary,
            "expires_at": session.expires_at.isoformat() if session.expires_at else None
        }

    @app.post("/api/sessions/cleanup")
    async def cleanup_expired_sessions():
        """Clean up expired temporary sessions."""
        from cli.repl.session import SessionManager

        deleted = SessionManager.cleanup_expired_sessions()

        return {
            "success": True,
            "deleted_count": len(deleted),
            "deleted_sessions": deleted
        }

    @app.get("/api/sessions")
    async def list_all_sessions(include_temporary: bool = False, include_expired: bool = False):
        """List all available sessions with filters."""
        from cli.repl.session import SessionManager

        session_manager = SessionManager()
        sessions = session_manager.list_sessions(
            include_temporary=include_temporary,
            include_expired=include_expired
        )

        return {"sessions": sessions}

    # ===== UPDATED CAPABILITIES ENDPOINT =====
    # Override the earlier one to add feature flags

