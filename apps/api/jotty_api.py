"""
JottyAPI - Core API business logic.

Manages request processing, session integration, command execution,
and folder management. Used by FastAPI routes in web/routes/.

All execution flows through ModeRouter for consistent behavior.
"""

import asyncio
import json
import logging
import uuid
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime

logger = logging.getLogger(__name__)

# Absolute imports - single source of truth
from Jotty.core.interface.api.mode_router import ModeRouter, get_mode_router, RouteResult
# Import SDK types from public package
from Jotty.sdk import (
    ExecutionContext, ExecutionMode, ChannelType, ResponseFormat,
)


class JottyAPI:
    """
    Jotty API handler.

    Manages request processing and session integration.
    All execution flows through ModeRouter for consistent behavior
    across CLI, Web, Gateway, and SDK.
    """

    def __init__(self):
        self._router: Optional[ModeRouter] = None
        self._registry = None
        self._cli = None  # Shared CLI instance for commands

    @property
    def router(self) -> ModeRouter:
        """Get ModeRouter singleton."""
        if self._router is None:
            self._router = get_mode_router()
        return self._router

    def _get_cli(self):
        """Get shared JottyCLI instance for command execution."""
        if self._cli is None:
            from Jotty.cli.app import JottyCLI
            self._cli = JottyCLI(no_color=True)  # No color for web output
        return self._cli

    def _get_session_registry(self):
        """Get session registry."""
        if self._registry is None:
            from Jotty.cli.repl.session import get_session_registry
            self._registry = get_session_registry()
        return self._registry

    async def _execute_with_images(self, task: str, images: List[str], status_cb=None):
        """
        Execute task with image attachments using multimodal LLM.

        Args:
            executor: The executor instance
            task: The task/prompt text
            images: List of base64 image data URLs
            status_cb: Status callback function

        Returns:
            ExecutorResult with LLM response
        """
        import os

        if status_cb:
            status_cb("analyzing", f"Processing {len(images)} image(s)...")

        # Create result dataclass
        from dataclasses import dataclass
        @dataclass
        class ImageResult:
            success: bool = True
            content: str = ""
            output_format: str = "markdown"
            output_path: str = None
            steps_taken: int = 1
            error: str = None

        try:
            # Build multimodal message content
            message_content = []

            # Add images first
            for i, img_data in enumerate(images):
                # Extract base64 data from data URL (data:image/png;base64,...)
                if img_data.startswith('data:'):
                    # Split off the header and get base64 data
                    header, b64_data = img_data.split(',', 1)
                    media_type = header.split(':')[1].split(';')[0]
                else:
                    b64_data = img_data
                    media_type = "image/jpeg"

                message_content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": b64_data
                    }
                })

            # Add text content
            message_content.append({
                "type": "text",
                "text": task
            })

            if status_cb:
                status_cb("processing", "Analyzing with vision model...")

            logger.info(f"_execute_with_images: processing {len(images)} images")
            logger.info(f"  First image data starts with: {images[0][:60] if images else 'none'}...")

            # Try Anthropic SDK directly (best for vision); respects ANTHROPIC_BASE_URL (CCR)
            from Jotty.core.infrastructure.foundation.anthropic_client_kwargs import get_anthropic_client_kwargs
            client_kwargs = get_anthropic_client_kwargs()
            api_key = client_kwargs.get("api_key")
            logger.info(f"  ANTHROPIC_API_KEY present: {bool(api_key)}")
            if api_key:
                try:
                    import anthropic
                    client = anthropic.Anthropic(**client_kwargs)

                    response = client.messages.create(
                        model="claude-sonnet-4-20250514",
                        max_tokens=4096,
                        messages=[{"role": "user", "content": message_content}]
                    )

                    # Extract text from response
                    content = ""
                    for block in response.content:
                        if hasattr(block, 'text'):
                            content += block.text

                    if status_cb:
                        status_cb("complete", "Image analysis complete")

                    return ImageResult(success=True, content=content)

                except Exception as e:
                    logger.warning(f"Anthropic vision failed: {e}")

            # Try OpenAI SDK (GPT-4V format)
            openai_key = os.environ.get("OPENAI_API_KEY")
            if openai_key:
                try:
                    import openai
                    client = openai.OpenAI(api_key=openai_key)

                    # Convert to OpenAI format
                    openai_content = []
                    for item in message_content:
                        if item["type"] == "image":
                            # OpenAI uses URL format for base64
                            b64 = item["source"]["data"]
                            media = item["source"]["media_type"]
                            openai_content.append({
                                "type": "image_url",
                                "image_url": {"url": f"data:{media};base64,{b64}"}
                            })
                        else:
                            openai_content.append(item)

                    response = client.chat.completions.create(
                        model="gpt-4o",
                        max_tokens=4096,
                        messages=[{"role": "user", "content": openai_content}]
                    )

                    content = response.choices[0].message.content

                    if status_cb:
                        status_cb("complete", "Image analysis complete")

                    return ImageResult(success=True, content=content)

                except Exception as e:
                    logger.warning(f"OpenAI vision failed: {e}")

            # Try OpenRouter (supports Claude vision via API)
            openrouter_key = os.environ.get("OPENROUTER_API_KEY")
            logger.info(f"Checking OpenRouter: key present = {bool(openrouter_key)}")
            if openrouter_key:
                try:
                    import openai
                    client = openai.OpenAI(
                        api_key=openrouter_key,
                        base_url="https://openrouter.ai/api/v1"
                    )

                    # Convert to OpenAI format (OpenRouter uses same format)
                    openrouter_content = []
                    for item in message_content:
                        if item["type"] == "image":
                            b64 = item["source"]["data"]
                            media = item["source"]["media_type"]
                            openrouter_content.append({
                                "type": "image_url",
                                "image_url": {"url": f"data:{media};base64,{b64}"}
                            })
                        else:
                            openrouter_content.append(item)

                    logger.info("Trying OpenRouter for vision...")
                    response = client.chat.completions.create(
                        model="anthropic/claude-sonnet-4",  # Vision-capable model
                        max_tokens=4096,
                        messages=[{"role": "user", "content": openrouter_content}]
                    )

                    content = response.choices[0].message.content

                    if status_cb:
                        status_cb("complete", "Image analysis complete")

                    return ImageResult(success=True, content=content)

                except Exception as e:
                    logger.warning(f"OpenRouter vision failed: {e}")

            # Fallback: describe that images were provided but couldn't be processed
            logger.warning("No vision-capable API available, falling back to text-only")
            enhanced_task = f"[Note: User attached {len(images)} image(s) but no vision API is configured. Please respond to: {task}]"
            fallback_ctx = ExecutionContext(mode=ExecutionMode.CHAT, channel=ChannelType.WEB)
            result = await self.router.chat(enhanced_task, fallback_ctx)
            return ImageResult(success=result.success, content=result.content or "", error=result.error)

        except Exception as e:
            logger.error(f"Image processing error: {e}", exc_info=True)
            # Fallback to text-only execution via ModeRouter
            fallback_ctx = ExecutionContext(mode=ExecutionMode.CHAT, channel=ChannelType.WEB)
            result = await self.router.chat(task, fallback_ctx)
            return ImageResult(success=result.success, content=result.content or "", error=result.error)

    async def process_message(
        self,
        message: str,
        session_id: str,
        user_id: str = "web_user",
        stream_callback=None,
        status_callback=None,
        attachments: List[Dict] = None
    ) -> Dict[str, Any]:
        """
        Process a chat message with optional image attachments.

        Routes through ModeRouter for consistent execution.

        Args:
            message: User message
            session_id: Session ID
            user_id: User identifier
            stream_callback: Optional callback for streaming
            status_callback: Optional callback for status updates
            attachments: List of image attachments [{type, name, data, size}]

        Returns:
            Response dict with content, output_path, etc.
        """
        from Jotty.cli.repl.session import InterfaceType

        # Get session
        registry = self._get_session_registry()
        session = registry.get_session(
            session_id,
            create=True,
            interface=InterfaceType.WEB
        )

        # Process attachments: images (for vision) and documents (for text extraction)
        image_descriptions = []
        image_data_list = []
        document_contexts = []

        if attachments:
            for att in attachments:
                if att.get("type") == "image" and att.get("data"):
                    image_data_list.append(att["data"])
                    image_descriptions.append(f"[Attached image: {att.get('name', 'image')}]")
                elif att.get("type") == "document" and att.get("docId"):
                    try:
                        from Jotty.web.documents import get_document_processor
                        processor = get_document_processor()
                        doc_text = processor.get_document_text(att["docId"])
                        if doc_text:
                            max_len = 8000
                            if len(doc_text) > max_len:
                                doc_text = doc_text[:max_len] + f"\n\n[... truncated, {len(doc_text) - max_len} more chars ...]"
                            document_contexts.append(f"=== Content from {att.get('name', 'document')} ===\n{doc_text}")
                    except Exception as e:
                        logger.error(f"Failed to extract document text: {e}")

        # Build message content with attachments context
        full_message = message
        context_parts = []

        if document_contexts:
            context_parts.append("ATTACHED DOCUMENTS:\n\n" + "\n\n".join(document_contexts))

        if image_descriptions:
            context_parts.append(' '.join(image_descriptions))

        if context_parts:
            full_message = "\n\n".join(context_parts) + f"\n\nUSER REQUEST: {message}" if message else "\n\n".join(context_parts)

        # Add user message to session
        message_id = str(uuid.uuid4())[:12]
        session.add_message(
            role="user",
            content=full_message,
            interface=InterfaceType.WEB,
            user_id=user_id,
            metadata={"message_id": message_id, "has_images": len(image_data_list) > 0}
        )

        # Wrap status callback to log + forward
        def status_cb(stage, detail=""):
            logger.debug(f"Status: {stage} - {detail}")
            if status_callback:
                try:
                    status_callback(stage, detail)
                except Exception as e:
                    logger.debug(f"Status callback error: {e}")

        # Build task with conversation context
        history = session.get_history()
        if len(history) > 1:
            context_messages = history[-10:-1]
            if context_messages:
                context_str = "\n".join([
                    f"{'User' if m.get('role') == 'user' else 'Assistant'}: {m.get('content', '')[:500]}"
                    for m in context_messages
                ])
                task_with_context = f"Previous conversation:\n{context_str}\n\nCurrent request: {full_message}"
            else:
                task_with_context = full_message
        else:
            task_with_context = full_message

        try:
            # Handle images separately (multimodal - needs direct API call)
            if image_data_list:
                result = await self._execute_with_images(task_with_context, image_data_list, status_cb)
                response_id = str(uuid.uuid4())[:12]

                if result.success:
                    session.add_message(
                        role="assistant",
                        content=result.content,
                        interface=InterfaceType.WEB,
                        metadata={"message_id": response_id}
                    )
                    return {
                        "success": True,
                        "message_id": response_id,
                        "content": result.content,
                        "output_format": getattr(result, 'output_format', 'markdown'),
                        "output_path": getattr(result, 'output_path', None),
                        "steps": getattr(result, 'steps_taken', 1),
                    }
                else:
                    return {"success": False, "error": getattr(result, 'error', 'Unknown error')}

            # Route through ModeRouter (canonical path)
            context = ExecutionContext(
                mode=ExecutionMode.CHAT,
                channel=ChannelType.WEB,
                session_id=session_id,
                user_id=user_id,
                status_callback=status_cb,
                stream_callback=stream_callback,
            )

            route_result = await self.router.chat(task_with_context, context)

            response_id = str(uuid.uuid4())[:12]

            if route_result.success:
                session.add_message(
                    role="assistant",
                    content=route_result.content,
                    interface=InterfaceType.WEB,
                    metadata={
                        "message_id": response_id,
                        "output_format": route_result.metadata.get("output_format", "markdown"),
                        "output_path": route_result.metadata.get("output_path"),
                        "steps": route_result.steps_executed,
                    }
                )

                return {
                    "success": True,
                    "message_id": response_id,
                    "content": route_result.content,
                    "output_format": route_result.metadata.get("output_format", "markdown"),
                    "output_path": route_result.metadata.get("output_path"),
                    "steps": route_result.steps_executed,
                }
            else:
                return {
                    "success": False,
                    "error": route_result.error or "Unknown error",
                    "steps": route_result.steps_executed,
                }

        except Exception as e:
            logger.error(f"Processing error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    def get_commands(self) -> List[Dict[str, Any]]:
        """Get available CLI commands."""
        from Jotty.cli.commands import CommandRegistry
        from Jotty.cli.commands import register_all_commands

        registry = CommandRegistry()
        register_all_commands(registry)

        commands = []
        for name, cmd in registry._commands.items():
            commands.append({
                "name": name,
                "description": getattr(cmd, "description", ""),
                "usage": getattr(cmd, "usage", f"/{name}"),
                "aliases": getattr(cmd, "aliases", []),
            })
        return commands

    async def execute_command(
        self,
        command: str,
        args: str = "",
        session_id: str = None
    ) -> Dict[str, Any]:
        """
        Execute a CLI command using the shared JottyCLI instance.

        This ensures all interfaces (CLI, Telegram, Web) have identical
        command behavior - they all use the same JottyCLI core.
        """
        import io
        import sys
        from contextlib import redirect_stdout, redirect_stderr

        try:
            # Get shared CLI instance
            cli = self._get_cli()

            # Capture stdout/stderr for web display
            output_buffer = io.StringIO()

            # Execute command through CLI's command handler
            cmd_input = f"/{command} {args}".strip()

            # Temporarily replace renderer output capture
            original_print = cli.renderer.print
            captured_output = []

            def capture_print(text, *args, **kwargs):
                # Strip rich markup for web
                import re
                clean_text = re.sub(r'\[/?[^\]]+\]', '', str(text))
                captured_output.append(clean_text)

            # Monkey-patch renderer methods to capture output
            cli.renderer.print = capture_print
            cli.renderer.info = lambda t: captured_output.append(f"ℹ️ {t}")
            cli.renderer.success = lambda t: captured_output.append(f"✅ {t}")
            cli.renderer.warning = lambda t: captured_output.append(f"⚠️ {t}")
            cli.renderer.error = lambda t: captured_output.append(f"❌ {t}")

            # Capture panel output
            original_panel = getattr(cli.renderer, 'panel', None)
            cli.renderer.panel = lambda content, **kwargs: captured_output.append(f"📋 {kwargs.get('title', 'Panel')}:\n{content}")

            # Capture tree output
            original_tree = getattr(cli.renderer, 'tree', None)
            def capture_tree(data, **kwargs):
                title = kwargs.get('title', 'Data')
                if isinstance(data, dict):
                    lines = [f"🌳 {title}:"]
                    for k, v in data.items():
                        lines.append(f"  • {k}: {v}")
                    captured_output.append("\n".join(lines))
                else:
                    captured_output.append(f"🌳 {title}: {data}")
            cli.renderer.tree = capture_tree

            # Capture table output - patch the tables component
            original_print_table = cli.renderer.tables.print_table
            def capture_table(table):
                # Use Rich Console to render to string, then strip ANSI
                try:
                    from rich.console import Console
                    from io import StringIO
                    string_io = StringIO()
                    console = Console(file=string_io, force_terminal=False, no_color=True)
                    console.print(table)
                    table_text = string_io.getvalue()
                    captured_output.append(table_text)
                except Exception:
                    # Fallback: just convert to string
                    captured_output.append(str(table))
            cli.renderer.tables.print_table = capture_table

            try:
                # Execute via CLI's command handler
                result = await cli._handle_command(cmd_input)

                return {
                    "success": True,
                    "output": "\n".join(captured_output) if captured_output else "Command executed",
                    "data": None,
                }
            finally:
                # Restore original methods
                cli.renderer.print = original_print
                cli.renderer.tables.print_table = original_print_table
                if original_panel:
                    cli.renderer.panel = original_panel
                if original_tree:
                    cli.renderer.tree = original_tree

        except Exception as e:
            logger.error(f"Command execution error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    def get_sessions(self) -> List[Dict[str, Any]]:
        """Get all sessions with metadata."""
        from Jotty.cli.repl.session import SessionManager
        import json

        manager = SessionManager()
        sessions = manager.list_sessions()

        # Enhance sessions with metadata fields
        for session in sessions:
            session_file = manager.session_dir / f"{session.get('session_id')}.json"
            if session_file.exists():
                try:
                    with open(session_file, 'r') as f:
                        data = json.load(f)
                    # Add metadata fields if they exist
                    for key in ['title', 'isPinned', 'isArchived', 'folderId']:
                        if key in data:
                            session[key] = data[key]
                except Exception:
                    pass
        return sessions

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session details."""
        from Jotty.cli.repl.session import InterfaceType

        registry = self._get_session_registry()
        session = registry.get_session(
            session_id,
            create=False,
            interface=InterfaceType.WEB
        )

        if session:
            return {
                **session.to_dict(),
                "history": session.get_history(),
            }
        return None

    def clear_session(self, session_id: str) -> bool:
        """Clear session history."""
        from Jotty.cli.repl.session import InterfaceType

        registry = self._get_session_registry()
        session = registry.get_session(
            session_id,
            create=False,
            interface=InterfaceType.WEB
        )

        if session:
            session.clear_history()
            session.save()
            return True
        return False

    def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        from Jotty.cli.repl.session import SessionManager

        registry = self._get_session_registry()
        registry.remove_session(session_id)

        manager = SessionManager()
        manager.delete_session(session_id)
        return True

    def update_session(self, session_id: str, updates: Dict[str, Any]) -> bool:
        """Update session metadata (title, isPinned, isArchived, folderId)."""
        from Jotty.cli.repl.session import SessionManager, InterfaceType

        manager = SessionManager()
        session_file = manager.session_dir / f"{session_id}.json"

        if not session_file.exists():
            return False

        try:
            import json
            with open(session_file, 'r') as f:
                data = json.load(f)

            # Update allowed fields
            for key in ['title', 'isPinned', 'isArchived', 'folderId']:
                if key in updates:
                    data[key] = updates[key]

            with open(session_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            return True
        except Exception as e:
            logger.error(f"Failed to update session {session_id}: {e}")
            return False

    def get_folders(self) -> List[Dict[str, Any]]:
        """Get all folders."""
        folder_file = Path.home() / ".jotty" / "folders.json"
        if folder_file.exists():
            try:
                import json
                with open(folder_file, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        return []

    def create_folder(self, folder: Dict[str, Any]) -> bool:
        """Create a new folder."""
        import json
        folder_file = Path.home() / ".jotty" / "folders.json"
        folder_file.parent.mkdir(parents=True, exist_ok=True)

        folders = self.get_folders()
        folders.append(folder)

        with open(folder_file, 'w') as f:
            json.dump(folders, f, indent=2)
        return True

    def delete_folder(self, folder_id: str) -> bool:
        """Delete a folder."""
        import json
        folder_file = Path.home() / ".jotty" / "folders.json"

        folders = self.get_folders()
        folders = [f for f in folders if f.get('id') != folder_id]

        with open(folder_file, 'w') as f:
            json.dump(folders, f, indent=2)
        return True

    def save_folders(self, folders: List[Dict[str, Any]]) -> bool:
        """Save all folders (bulk update for reordering, renaming, color changes)."""
        import json
        folder_file = Path.home() / ".jotty" / "folders.json"
        folder_file.parent.mkdir(parents=True, exist_ok=True)

        with open(folder_file, 'w') as f:
            json.dump(folders, f, indent=2)
        return True

