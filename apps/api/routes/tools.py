"""
Tool routes - export, preview, proxy, MCP, code execution, web search.
"""

import asyncio
import json
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)



def register_tools_routes(app, api):
    from fastapi import HTTPException, WebSocket, WebSocketDisconnect, UploadFile, File, Form
    from fastapi.responses import FileResponse, JSONResponse, StreamingResponse, HTMLResponse
    from pydantic import BaseModel

    @app.post("/api/export")
    async def export_content(request: dict):
        """
        Export content to various formats.

        Args:
            content: Markdown content to export
            format: Target format (md, pdf, docx, epub, html, slides)
            filename: Optional filename (without extension)
            title: Optional document title (defaults to filename)

        Features:
            - Multiple PDF engines with fallback (xelatex -> pdflatex -> weasyprint)
            - Custom LaTeX templates for professional styling
            - Code syntax highlighting
            - Math equation support
            - Detailed error messages
        """
        from starlette.responses import FileResponse
        from .export_utils import export_content as do_export, ExportError

        content = request.get("content", "")
        export_format = request.get("format", "md").lower()
        filename = request.get("filename", "export")
        title = request.get("title", filename)

        if not content:
            raise HTTPException(status_code=400, detail="Content is required")

        try:
            # Use enhanced export utilities with fallbacks and templates
            output_file, media_type = do_export(
                content=content,
                format=export_format,
                filename=filename,
                title=title
            )

            if not output_file.exists():
                raise HTTPException(status_code=500, detail="Conversion failed - output file not created")

            # Add headers for inline viewing (especially for PDF)
            headers = {}
            if media_type == "application/pdf":
                # Content-Disposition: inline allows browser PDF viewer
                headers["Content-Disposition"] = f'inline; filename="{output_file.name}"'

            return FileResponse(
                path=str(output_file),
                filename=output_file.name,
                media_type=media_type,
                headers=headers,
                background=None  # Don't delete file immediately
            )

        except ExportError as e:
            logger.error(f"Export failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
        except Exception as e:
            logger.error(f"Export error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/preview")
    async def preview_content(request: dict):
        """
        Preview content in various formats (returns HTML/text for inline display).

        Args:
            content: Markdown content to preview
            format: Target format (html, docx-preview)
        """
        from starlette.responses import HTMLResponse, PlainTextResponse
        import tempfile
        import subprocess
        from pathlib import Path

        content = request.get("content", "")
        preview_format = request.get("format", "html").lower()

        if not content:
            raise HTTPException(status_code=400, detail="Content is required")

        temp_dir = Path(tempfile.mkdtemp())
        md_file = temp_dir / "preview.md"
        md_file.write_text(content, encoding="utf-8")

        try:
            if preview_format == "html":
                # Convert to standalone HTML
                output_file = temp_dir / "preview.html"
                subprocess.run([
                    "pandoc", str(md_file), "-o", str(output_file),
                    "--standalone",
                    "--metadata", "title=Preview",
                    "--css", "data:text/css,body{font-family:-apple-system,BlinkMacSystemFont,sans-serif;max-width:800px;margin:40px auto;padding:20px;line-height:1.6}pre{background:%23f5f5f5;padding:16px;border-radius:8px;overflow-x:auto}code{background:%23f0f0f0;padding:2px 6px;border-radius:4px}h1,h2,h3{margin-top:24px}"
                ], check=True)
                html_content = output_file.read_text(encoding="utf-8")
                return HTMLResponse(content=html_content)

            elif preview_format == "docx-preview":
                # Convert DOCX to HTML for preview
                # First create DOCX, then convert to HTML
                docx_file = temp_dir / "preview.docx"
                html_file = temp_dir / "preview.html"

                subprocess.run([
                    "pandoc", str(md_file), "-o", str(docx_file)
                ], check=True)

                subprocess.run([
                    "pandoc", str(docx_file), "-o", str(html_file),
                    "--standalone"
                ], check=True)

                html_content = html_file.read_text(encoding="utf-8")
                # Extract just the body content
                import re
                body_match = re.search(r'<body[^>]*>(.*?)</body>', html_content, re.DOTALL)
                if body_match:
                    return HTMLResponse(content=body_match.group(1))
                return HTMLResponse(content=html_content)

            else:
                raise HTTPException(status_code=400, detail=f"Unsupported preview format: {preview_format}")

        except subprocess.CalledProcessError as e:
            logger.error(f"Preview conversion failed: {e}")
            raise HTTPException(status_code=500, detail=f"Preview failed: {str(e)}")
        except Exception as e:
            logger.error(f"Preview error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # ===== VOICE CHAT ENDPOINTS =====

    @app.get("/api/proxy")
    async def proxy_url(url: str):
        """
        Proxy a URL and strip headers that prevent iframe embedding.

        This allows loading external websites in the inline browser
        even if they set X-Frame-Options or CSP headers.
        """
        import httpx
        from starlette.responses import Response

        if not url:
            raise HTTPException(status_code=400, detail="URL is required")

        # Validate URL
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url

        try:
            async with httpx.AsyncClient(
                follow_redirects=True,
                timeout=30.0,
                headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.5',
                }
            ) as client:
                response = await client.get(url)

                # Get content type
                content_type = response.headers.get('content-type', 'text/html')

                # Build response headers - copy safe headers, skip restrictive ones
                safe_headers = {}
                skip_headers = {
                    'x-frame-options',
                    'content-security-policy',
                    'content-security-policy-report-only',
                    'x-content-type-options',
                    'strict-transport-security',
                    'transfer-encoding',
                    'content-encoding',
                    'content-length',  # Will be recalculated
                }

                for key, value in response.headers.items():
                    if key.lower() not in skip_headers:
                        safe_headers[key] = value

                # For HTML content, inject base tag to fix relative URLs
                content = response.content
                if 'text/html' in content_type:
                    try:
                        html = content.decode('utf-8', errors='replace')
                        # Parse the base URL
                        from urllib.parse import urlparse
                        parsed = urlparse(url)
                        base_url = f"{parsed.scheme}://{parsed.netloc}"

                        # Inject base tag if not present
                        if '<base' not in html.lower():
                            # Insert base tag after <head>
                            if '<head>' in html:
                                html = html.replace('<head>', f'<head><base href="{base_url}/">', 1)
                            elif '<head ' in html:
                                html = html.replace('<head ', f'<base href="{base_url}/"><head ', 1)
                            elif '<HEAD>' in html:
                                html = html.replace('<HEAD>', f'<HEAD><base href="{base_url}/">', 1)

                        content = html.encode('utf-8')
                    except Exception as e:
                        logger.debug(f"Failed to inject base tag: {e}")

                return Response(
                    content=content,
                    status_code=response.status_code,
                    headers=safe_headers,
                    media_type=content_type.split(';')[0]  # Just the mime type, not charset
                )

        except httpx.TimeoutException:
            raise HTTPException(status_code=504, detail="Request timed out")
        except httpx.RequestError as e:
            logger.error(f"Proxy request error: {e}")
            raise HTTPException(status_code=502, detail=f"Failed to fetch URL: {str(e)}")
        except Exception as e:
            logger.error(f"Proxy error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # ===== LIBRECHAT-STYLE FEATURES =====

    # ===== MCP TOOLS ENDPOINTS =====

    @app.get("/api/mcp/tools")
    async def list_mcp_tools():
        """List available MCP tools with enable/disable status."""
        try:
            from ..core.integration.mcp_client import MCPClient

            # Try to get tools from MCP server
            try:
                client = MCPClient()
                await client.connect()
                tools = await client.list_tools()
                await client.disconnect()

                return {
                    "tools": [
                        {
                            "name": t.get("name", ""),
                            "description": t.get("description", ""),
                            "inputSchema": t.get("inputSchema", {}),
                            "enabled": True
                        }
                        for t in tools
                    ],
                    "count": len(tools),
                    "connected": True
                }
            except Exception as e:
                logger.warning(f"MCP connection failed: {e}")
                # Return fallback tools
                return {
                    "tools": [
                        {"name": "create_idea", "description": "Create a new idea/note", "enabled": True},
                        {"name": "list_ideas", "description": "List all ideas", "enabled": True},
                        {"name": "search_ideas", "description": "Search ideas by query", "enabled": True},
                    ],
                    "count": 3,
                    "connected": False,
                    "error": str(e)
                }
        except ImportError:
            return {"tools": [], "count": 0, "connected": False, "error": "MCP client not available"}

    class MCPExecuteRequest(BaseModel):
        tool_name: str
        arguments: dict = {}

    @app.post("/api/mcp/execute")
    async def execute_mcp_tool(request: MCPExecuteRequest):
        """Execute an MCP tool and return result."""
        import time
        start_time = time.time()

        try:
            from ..core.integration.mcp_client import call_justjot_mcp_tool

            result = await call_justjot_mcp_tool(
                tool_name=request.tool_name,
                arguments=request.arguments
            )

            duration_ms = int((time.time() - start_time) * 1000)

            return {
                "success": True,
                "tool_name": request.tool_name,
                "result": result,
                "duration_ms": duration_ms
            }
        except Exception as e:
            logger.error(f"MCP tool execution failed: {e}")
            return {
                "success": False,
                "tool_name": request.tool_name,
                "error": str(e),
                "duration_ms": int((time.time() - start_time) * 1000)
            }

    # ===== ARTIFACTS ENDPOINTS =====

    @app.post("/api/artifacts/extract")
    async def extract_artifacts(request: dict):
        """Extract artifacts (code blocks, diagrams, etc.) from text."""
        from .artifacts import extract_artifacts as do_extract

        text = request.get("text", "")
        if not text:
            return {"artifacts": [], "count": 0}

        artifacts = do_extract(text)
        return {"artifacts": artifacts, "count": len(artifacts)}

    @app.post("/api/artifacts/render")
    async def render_artifact(request: dict):
        """
        Render an artifact to displayable format.

        For HTML: returns sanitized HTML
        For Mermaid: returns SVG
        For code: returns syntax-highlighted HTML
        """
        artifact_type = request.get("type", "code")
        content = request.get("content", "")

        if not content:
            raise HTTPException(status_code=400, detail="Content is required")

        if artifact_type == "mermaid":
            # For mermaid, return content as-is (frontend will use mermaid.js)
            return {
                "type": "mermaid",
                "content": content,
                "render_mode": "client"  # Render on client side
            }

        if artifact_type == "html":
            # Return HTML for sandboxed iframe
            return {
                "type": "html",
                "content": content,
                "render_mode": "iframe"
            }

        if artifact_type == "svg":
            return {
                "type": "svg",
                "content": content,
                "render_mode": "inline"
            }

        # Default: code block
        return {
            "type": "code",
            "content": content,
            "language": request.get("language", ""),
            "render_mode": "highlight"
        }

    # ===== CODE INTERPRETER ENDPOINTS =====

    class CodeExecuteRequest(BaseModel):
        code: str
        language: str = "python"
        timeout: int = 30

    @app.post("/api/code/execute")
    async def execute_code(request: CodeExecuteRequest):
        """Execute code in sandboxed environment."""
        from .code_interpreter import execute_code as do_execute

        if not request.code.strip():
            raise HTTPException(status_code=400, detail="Code is required")

        result = await do_execute(request.code, request.language)
        return result

    @app.get("/api/code/execute/stream")
    @app.get("/api/code/languages")
    async def list_code_languages():
        """List supported programming languages for code execution."""
        return {
            "languages": [
                {"id": "python", "name": "Python", "extension": ".py", "available": True},
                {"id": "javascript", "name": "JavaScript", "extension": ".js", "available": True},
                {"id": "typescript", "name": "TypeScript", "extension": ".ts", "available": False},
                {"id": "bash", "name": "Bash", "extension": ".sh", "available": False},
            ]
        }

    # ===== CONVERSATION BRANCHING ENDPOINTS =====

    class WebSearchRequest(BaseModel):
        query: str
        max_results: int = 10

    @app.post("/api/search")
    async def web_search(request: WebSearchRequest):
        """
        Search the web using the existing web-search skill (DuckDuckGo).

        Args:
            query: Search query
            max_results: Maximum number of results (default 10)

        Returns:
            Search results with title, url, snippet
        """
        try:
            # Use existing skill directly
            import sys
            from pathlib import Path
            skills_path = Path(__file__).parent.parent / "skills" / "web-search"
            if str(skills_path) not in sys.path:
                sys.path.insert(0, str(skills_path))
            from tools import search_web_tool

            result = await asyncio.to_thread(
                search_web_tool,
                {"query": request.query, "max_results": request.max_results}
            )

            return result  # Already has success, results, count, query
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/search")
    async def web_search_get(query: str, max_results: int = 10):
        """GET endpoint for web search using existing skill."""
        try:
            import sys
            from pathlib import Path
            skills_path = Path(__file__).parent.parent / "skills" / "web-search"
            if str(skills_path) not in sys.path:
                sys.path.insert(0, str(skills_path))
            from tools import search_web_tool

            result = await asyncio.to_thread(
                search_web_tool,
                {"query": query, "max_results": max_results}
            )

            return result
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # ===== SHAREABLE LINKS ENDPOINTS =====

    class CreateShareLinkRequest(BaseModel):
        session_id: str
        title: Optional[str] = None
        expires_in_days: Optional[int] = None
        branch_id: str = "main"

