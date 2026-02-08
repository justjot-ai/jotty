"""
JottyAPI - Core API business logic.

Manages request processing, session integration, command execution,
and folder management. Used by FastAPI routes in web/routes/.
"""

import asyncio
import json
import logging
import uuid
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime

logger = logging.getLogger(__name__)


class JottyAPI:
    """
    Jotty API handler.

    Manages request processing and session integration.
    Uses shared JottyCLI instance for command execution to ensure
    all interfaces (CLI, Telegram, Web) have identical behavior.
    """

    def __init__(self):
        self._executor = None
        self._registry = None
        self._lm_configured = False
        self._cli = None  # Shared CLI instance for commands

    def _get_cli(self):
        """Get shared JottyCLI instance for command execution."""
        if self._cli is None:
            from ..cli.app import JottyCLI
            self._cli = JottyCLI(no_color=True)  # No color for web output
        return self._cli

    def _ensure_lm_configured(self):
        """Ensure DSPy LM is configured (same as CLI)."""
        if self._lm_configured:
            return True

        import dspy

        # Check if already configured
        if hasattr(dspy.settings, 'lm') and dspy.settings.lm is not None:
            self._lm_configured = True
            return True

        try:
            # Use the same unified_lm_provider as CLI
            from ..core.foundation.unified_lm_provider import configure_dspy_lm

            # Auto-detect: tries claude-cli first (free), then API providers
            lm = configure_dspy_lm()
            if lm:
                self._lm_configured = True
                model_name = getattr(lm, 'model', None) or getattr(lm, 'model_name', 'unknown')
                logger.info(f"LLM configured: {model_name}")
                return True
        except Exception as e:
            logger.error(f"Failed to configure LLM: {e}")

        return False

    def _get_executor(self, status_callback=None, stream_callback=None):
        """Get UnifiedExecutor with callbacks (auto-detects provider)."""
        from ..core.orchestration.v2.unified_executor import UnifiedExecutor
        return UnifiedExecutor(
            status_callback=status_callback,
            stream_callback=stream_callback
        )

    def _get_session_registry(self):
        """Get session registry."""
        if self._registry is None:
            from ..cli.repl.session import get_session_registry
            self._registry = get_session_registry()
        return self._registry

    async def _execute_with_images(self, executor, task: str, images: List[str], status_cb=None):
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

            # Try Anthropic SDK directly (best for vision)
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            logger.info(f"  ANTHROPIC_API_KEY present: {bool(api_key)}")
            if api_key:
                try:
                    import anthropic
                    client = anthropic.Anthropic(api_key=api_key)

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
            return await executor.execute(enhanced_task)

        except Exception as e:
            logger.error(f"Image processing error: {e}", exc_info=True)
            # Fallback to text-only execution
            return await executor.execute(task)

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
        from ..cli.repl.session import InterfaceType

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
                    # Image: use for multimodal vision
                    image_data_list.append(att["data"])
                    image_descriptions.append(f"[Attached image: {att.get('name', 'image')}]")
                elif att.get("type") == "document" and att.get("docId"):
                    # Document: extract text for context
                    try:
                        from .documents import get_document_processor
                        processor = get_document_processor()
                        doc_text = processor.get_document_text(att["docId"])
                        if doc_text:
                            # Truncate if too long
                            max_len = 8000
                            if len(doc_text) > max_len:
                                doc_text = doc_text[:max_len] + f"\n\n[... truncated, {len(doc_text) - max_len} more chars ...]"
                            document_contexts.append(f"=== Content from {att.get('name', 'document')} ===\n{doc_text}")
                    except Exception as e:
                        logger.error(f"Failed to extract document text: {e}")

        # Build message content with attachments context
        # NOTE: Web search is handled by UnifiedExecutor via LLM tool calling (not regex)
        # The LLM uses web_search tool when it needs current info
        full_message = message
        context_parts = []

        if document_contexts:
            context_parts.append("ATTACHED DOCUMENTS:\n\n" + "\n\n".join(document_contexts))

        if image_descriptions:
            context_parts.append(' '.join(image_descriptions))

        if context_parts:
            full_message = "\n\n".join(context_parts) + f"\n\nUSER REQUEST: {message}" if message else "\n\n".join(context_parts)

        # Add user message
        message_id = str(uuid.uuid4())[:12]
        session.add_message(
            role="user",
            content=full_message,
            interface=InterfaceType.WEB,
            user_id=user_id,
            metadata={"message_id": message_id, "has_images": len(image_data_list) > 0}
        )

        # Create status callback that calls both logger and external callback
        def status_cb(stage, detail):
            logger.debug(f"Status: {stage} - {detail}")
            if status_callback:
                try:
                    status_callback(stage, detail)
                except Exception as e:
                    logger.debug(f"Status callback error: {e}")

        executor = self._get_executor(
            status_callback=status_cb,
            stream_callback=stream_callback
        )

        # Build task with conversation context
        history = session.get_history()
        if len(history) > 1:
            # Include recent conversation for context (last 5 exchanges max)
            context_messages = history[-10:-1]  # Exclude current message
            if context_messages:
                context_str = "\n".join([
                    f"{'User' if m.get('role') == 'user' else 'Assistant'}: {m.get('content', '')[:500]}"
                    for m in context_messages
                ])
                task_with_context = f"""Previous conversation:
{context_str}

Current request: {full_message}"""
            else:
                task_with_context = full_message
        else:
            task_with_context = full_message

        try:
            # Execute with context and images
            # If we have images, try to use multimodal LLM
            if image_data_list:
                result = await self._execute_with_images(executor, task_with_context, image_data_list, status_cb)
            else:
                result = await executor.execute(task_with_context)

            response_id = str(uuid.uuid4())[:12]

            if result.success:
                # Add assistant response
                session.add_message(
                    role="assistant",
                    content=result.content,
                    interface=InterfaceType.WEB,
                    metadata={
                        "message_id": response_id,
                        "output_format": result.output_format,
                        "output_path": result.output_path,
                        "steps": result.steps_taken,
                    }
                )

                return {
                    "success": True,
                    "message_id": response_id,
                    "content": result.content,
                    "output_format": result.output_format,
                    "output_path": result.output_path,
                    "steps": result.steps_taken,
                }
            else:
                return {
                    "success": False,
                    "error": result.error or "Unknown error",
                    "steps": result.steps_taken,
                }

        except Exception as e:
            logger.error(f"Processing error: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
            }

    def get_commands(self) -> List[Dict[str, Any]]:
        """Get available CLI commands."""
        from ..cli.commands import CommandRegistry
        from ..cli.commands import register_all_commands

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
        from ..cli.repl.session import SessionManager
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
        from ..cli.repl.session import InterfaceType

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
        from ..cli.repl.session import InterfaceType

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
        from ..cli.repl.session import SessionManager

        registry = self._get_session_registry()
        registry.remove_session(session_id)

        manager = SessionManager()
        manager.delete_session(session_id)
        return True

    def update_session(self, session_id: str, updates: Dict[str, Any]) -> bool:
        """Update session metadata (title, isPinned, isArchived, folderId)."""
        from ..cli.repl.session import SessionManager, InterfaceType

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

