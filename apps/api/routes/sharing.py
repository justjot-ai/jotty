"""
Sharing routes - share links, QR codes, public share pages.
"""

import asyncio
import json
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)



def register_sharing_routes(app, api):
    from fastapi import HTTPException, WebSocket, WebSocketDisconnect, UploadFile, File, Form
    from fastapi.responses import FileResponse, JSONResponse, StreamingResponse, HTMLResponse
    from pydantic import BaseModel

    class CreateShareLinkRequest(BaseModel):
        session_id: str
        title: Optional[str] = None
        expires_in_days: int = 30
        branch_id: Optional[str] = None

    @app.post("/api/share/create")
    async def create_share_link(request: CreateShareLinkRequest):
        """Create a shareable link for a conversation."""
        from Jotty.apps.cli.repl.session import get_share_link_registry, get_session_registry

        # Verify session exists
        registry = get_session_registry()
        session = registry.get_session(request.session_id, create=False)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        # Create share link
        share_registry = get_share_link_registry()
        link = share_registry.create_link(
            session_id=request.session_id,
            title=request.title,
            expires_in_days=request.expires_in_days,
            branch_id=request.branch_id
        )

        return {
            "success": True,
            "link": link.to_dict(),
            "url": f"/share/{link.token}"
        }

    @app.get("/api/share/{token}")
    async def get_share_link_info(token: str):
        """Get information about a share link."""
        from Jotty.apps.cli.repl.session import get_share_link_registry

        share_registry = get_share_link_registry()
        link = share_registry.get_link(token)

        if not link:
            raise HTTPException(status_code=404, detail="Share link not found or expired")

        return {"link": link.to_dict()}

    @app.get("/api/share/{token}/conversation")
    async def get_shared_conversation(token: str):
        """Get the shared conversation (public read-only view)."""
        from Jotty.apps.cli.repl.session import get_share_link_registry, get_session_registry

        share_registry = get_share_link_registry()
        link = share_registry.get_link(token)

        if not link:
            raise HTTPException(status_code=404, detail="Share link not found or expired")

        # Record access
        share_registry.record_access(token)

        # Get session
        registry = get_session_registry()
        session = registry.get_session(link.session_id, create=False)

        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        # Get messages for the branch
        messages = session.get_branch_messages(link.branch_id)

        return {
            "title": link.title or f"Shared Chat",
            "messages": [m.to_dict() for m in messages],
            "created_at": link.created_at.isoformat(),
            "access_count": link.access_count,
            "branch_id": link.branch_id
        }

    @app.post("/api/share/{token}/revoke")
    async def revoke_share_link(token: str):
        """Revoke a share link."""
        from Jotty.apps.cli.repl.session import get_share_link_registry

        share_registry = get_share_link_registry()
        success = share_registry.revoke_link(token)

        if not success:
            raise HTTPException(status_code=404, detail="Share link not found")

        return {"success": True}

    @app.post("/api/share/{token}/refresh")
    async def refresh_share_link(token: str, expires_in_days: int = 30):
        """Refresh a share link with new token and expiry."""
        from Jotty.apps.cli.repl.session import get_share_link_registry

        share_registry = get_share_link_registry()
        new_link = share_registry.refresh_link(token, expires_in_days)

        if not new_link:
            raise HTTPException(status_code=404, detail="Share link not found")

        return {
            "success": True,
            "link": new_link.to_dict(),
            "url": f"/share/{new_link.token}"
        }

    @app.get("/api/share/{token}/qrcode")
    async def get_share_qrcode(token: str, base_url: str = None):
        """Generate QR code for a share link."""
        from Jotty.apps.cli.repl.session import get_share_link_registry

        share_registry = get_share_link_registry()
        link = share_registry.get_link(token)

        if not link:
            raise HTTPException(status_code=404, detail="Share link not found or expired")

        # Generate QR code as data URL
        try:
            import qrcode
            import io
            import base64

            share_url = f"{base_url or ''}/share/{token}"
            qr = qrcode.QRCode(version=1, box_size=10, border=5)
            qr.add_data(share_url)
            qr.make(fit=True)

            img = qr.make_image(fill_color="black", back_color="white")
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            img_str = base64.b64encode(buffer.getvalue()).decode()

            return {
                "qrcode": f"data:image/png;base64,{img_str}",
                "url": share_url
            }
        except ImportError:
            # qrcode library not installed, return URL only
            return {
                "qrcode": None,
                "url": f"{base_url or ''}/share/{token}",
                "error": "QR code library not installed"
            }

    # Public share page (no auth required)
    @app.get("/share/{token}")
    async def share_page(token: str):
        """Serve the public share page."""
        from Jotty.apps.cli.repl.session import get_share_link_registry

        share_registry = get_share_link_registry()
        link = share_registry.get_link(token)

        if not link:
            raise HTTPException(status_code=404, detail="Share link not found or expired")

        # Return HTML page that will load the conversation
        return HTMLResponse(f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Shared Chat - Jotty</title>
    <link rel="stylesheet" href="/static/style.css">
    <style>
        .shared-container {{ max-width: 800px; margin: 0 auto; padding: 20px; }}
        .shared-header {{ text-align: center; margin-bottom: 30px; padding: 20px; background: var(--bg-secondary); border-radius: 10px; }}
        .shared-header h1 {{ margin: 0; font-size: 1.5rem; }}
        .shared-header p {{ color: var(--text-muted); margin: 10px 0 0; }}
        .shared-message {{ margin: 15px 0; padding: 15px; border-radius: 10px; }}
        .shared-message.user {{ background: var(--user-msg-bg); margin-left: 50px; }}
        .shared-message.assistant {{ background: var(--assistant-msg-bg); margin-right: 50px; }}
        .shared-footer {{ text-align: center; margin-top: 30px; color: var(--text-muted); }}
    </style>
</head>
<body>
    <div class="shared-container">
        <div class="shared-header">
            <h1 id="share-title">Shared Chat</h1>
            <p id="share-info">Loading...</p>
        </div>
        <div id="messages"></div>
        <div class="shared-footer">
            <p>Shared via <a href="/">Jotty</a></p>
        </div>
    </div>
    <script>
        async function loadSharedConversation() {{
            try {{
                const response = await fetch('/api/share/{token}/conversation');
                if (!response.ok) throw new Error('Failed to load conversation');
                const data = await response.json();

                document.getElementById('share-title').textContent = data.title;
                document.getElementById('share-info').textContent = `Viewed ${{data.access_count}} times`;

                const messagesDiv = document.getElementById('messages');
                data.messages.forEach(msg => {{
                    const div = document.createElement('div');
                    div.className = `shared-message ${{msg.role}}`;
                    div.innerHTML = `<strong>${{msg.role === 'user' ? 'You' : 'Assistant'}}</strong><div>${{msg.content}}</div>`;
                    messagesDiv.appendChild(div);
                }});
            }} catch (error) {{
                document.getElementById('messages').innerHTML = '<p style="color: red;">Failed to load conversation</p>';
            }}
        }}
        loadSharedConversation();
    </script>
</body>
</html>
        """)

    # ===== TEMPORARY CHAT ENDPOINTS =====

    class CreateTempSessionRequest(BaseModel):
        expiry_days: Optional[int] = 30

