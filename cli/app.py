"""
Jotty CLI Main Application
==========================

Main JottyCLI class that orchestrates the interactive CLI.
"""

import asyncio
import logging
from pathlib import Path
from typing import Optional, Any

from .config.loader import ConfigLoader
from .config.schema import CLIConfig
from .ui.renderer import RichRenderer
from .repl.engine import REPLEngine, SimpleREPL
from .repl.session import SessionManager
from .repl.history import HistoryManager
from .commands.base import CommandRegistry, ParsedArgs
from .commands import register_all_commands
from .commands.help_cmd import QuitCommand, ClearCommand, HistoryCommand
from .plugins.loader import PluginLoader

logger = logging.getLogger(__name__)

# Check for prompt_toolkit
try:
    import prompt_toolkit
    PROMPT_TOOLKIT_AVAILABLE = True
except ImportError:
    PROMPT_TOOLKIT_AVAILABLE = False


class JottyCLI:
    """
    Main Jotty CLI Application.

    Provides an interactive REPL with:
    - Slash commands for agent/skill/swarm operations
    - Natural language task execution
    - Rich terminal UI
    - Session management
    - Plugin system
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        no_color: bool = False,
        debug: bool = False,
    ):
        """
        Initialize JottyCLI.

        Args:
            config_path: Path to config file
            no_color: Disable colored output
            debug: Enable debug mode
        """
        # Load configuration
        self.config_loader = ConfigLoader(config_path)
        self.config = self.config_loader.load()
        self.config.no_color = no_color
        self.config.debug = debug

        if debug:
            logging.basicConfig(level=logging.DEBUG)

        # UI Renderer
        self.renderer = RichRenderer(
            theme=self.config.ui.theme,
            no_color=no_color,
            max_width=self.config.ui.max_width,
        )

        # Command registry
        self.command_registry = CommandRegistry()
        self._register_commands()

        # Session manager
        self.session = SessionManager(
            session_dir=self.config.session.session_dir,
            context_window=self.config.session.context_window,
            auto_save=self.config.session.auto_save,
        )

        # REPL engine
        if PROMPT_TOOLKIT_AVAILABLE:
            self.repl = REPLEngine(
                command_registry=self.command_registry,
                history_file=self.config.session.history_file,
                prompt_text=self.renderer.prompt(),
            )
        else:
            self.repl = SimpleREPL(
                command_registry=self.command_registry,
                prompt_text="jotty> ",
            )

        # Plugin loader
        self.plugin_loader = PluginLoader()

        # Lazy-initialized components
        self._swarm_manager = None
        self._skills_registry = None

        logger.info("JottyCLI initialized")

    def _register_commands(self):
        """Register all built-in commands."""
        register_all_commands(self.command_registry)

        # Register utility commands
        self.command_registry.register(QuitCommand())
        self.command_registry.register(ClearCommand())
        self.command_registry.register(HistoryCommand())

    async def get_swarm_manager(self):
        """
        Get or create SwarmManager (lazy initialization).

        Returns:
            SwarmManager instance
        """
        if self._swarm_manager is None:
            try:
                # Suppress HuggingFace/BERT warnings before loading SwarmManager
                import os
                import warnings
                import logging as _logging
                os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS', '1')
                os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
                os.environ.setdefault('TQDM_DISABLE', '1')
                warnings.filterwarnings('ignore', message='.*unauthenticated.*')
                warnings.filterwarnings('ignore', message='.*huggingface.*')
                for _name in ['safetensors', 'sentence_transformers', 'transformers', 'huggingface_hub']:
                    _logging.getLogger(_name).setLevel(_logging.ERROR)

                # Suppress stdout/stderr during model loading
                import sys
                import io
                _old_stdout, _old_stderr = sys.stdout, sys.stderr

                # Try relative import first (for installed package)
                try:
                    from ..core.orchestration.v2 import SwarmManager
                    from ..core.foundation.data_structures import JottyConfig
                    from ..core.foundation.unified_lm_provider import configure_dspy_lm
                except ImportError:
                    # Fallback for direct execution
                    from Jotty.core.orchestration.v2 import SwarmManager
                    from Jotty.core.foundation.data_structures import JottyConfig
                    from Jotty.core.foundation.unified_lm_provider import configure_dspy_lm

                # Configure DSPy LM before creating SwarmManager
                # This ensures all DSPy modules have access to an LM
                # Use auto-detection which tries claude-cli first (free), then API providers
                lm_configured = False
                try:
                    # Auto-detect: tries claude-cli, cursor-cli, then API providers
                    lm = configure_dspy_lm()  # Uses built-in fallback logic
                    if lm:
                        # Extract meaningful provider/model info
                        model_name = getattr(lm, 'model', None) or getattr(lm, 'model_name', None) or 'unknown'
                        # model_name might be like "anthropic/claude-sonnet-4-20250514"
                        if '/' in str(model_name):
                            provider_name, model_short = str(model_name).split('/', 1)
                        else:
                            provider_name = type(lm).__name__
                            model_short = str(model_name)
                        self.renderer.info(f"LLM: {provider_name} ({model_short[:30]})")
                        lm_configured = True
                except Exception as e:
                    logger.debug(f"Auto-configure failed: {e}")
                    # Try explicit fallback order from config
                    for provider in self.config.provider.fallback_order:
                        try:
                            lm = configure_dspy_lm(provider=provider)
                            if lm:
                                self.renderer.info(f"LLM configured: {provider}")
                                lm_configured = True
                                break
                        except Exception as e2:
                            logger.debug(f"Provider {provider} not available: {e2}")
                            continue

                if not lm_configured:
                    self.renderer.warning(
                        "No LLM provider available. Options:\n"
                        "  1. Install Claude CLI: pip install claude-cli (uses your Anthropic login)\n"
                        "  2. Set API keys in environment:\n"
                        "     - ANTHROPIC_API_KEY for Claude API\n"
                        "     - OPENAI_API_KEY for OpenAI\n"
                        "     - GROQ_API_KEY for Groq (free tier available)"
                    )

                # Create JottyConfig from CLI config
                jotty_config = JottyConfig(
                    enable_adaptive_alpha=self.config.swarm.enable_learning,
                )

                self._swarm_manager = SwarmManager(
                    config=jotty_config,
                    enable_zero_config=self.config.swarm.enable_zero_config,
                )

                logger.info("SwarmManager initialized")

            except Exception as e:
                logger.error(f"Failed to initialize SwarmManager: {e}")
                raise

        return self._swarm_manager

    def get_skills_registry(self):
        """
        Get or create SkillsRegistry (lazy initialization).

        Returns:
            SkillsRegistry instance
        """
        if self._skills_registry is None:
            try:
                # Try relative import first (for installed package)
                try:
                    from ..core.registry.skills_registry import get_skills_registry
                except ImportError:
                    # Fallback for direct execution
                    from Jotty.core.registry.skills_registry import get_skills_registry

                self._skills_registry = get_skills_registry()
                logger.info("SkillsRegistry initialized")
            except Exception as e:
                logger.error(f"Failed to initialize SkillsRegistry: {e}")
                raise

        return self._skills_registry

    async def run_interactive(self):
        """Run interactive REPL mode."""
        # Auto-resume last session if enabled
        if self.config.session.auto_resume:
            await self._try_auto_resume()

        # Load plugins if enabled
        if self.config.features.plugin_system:
            try:
                self.plugin_loader.load_skill_plugins(self)
            except Exception as e:
                logger.warning(f"Plugin loading failed: {e}")

        # Update completer with skill names
        if hasattr(self.repl, "completer"):
            try:
                registry = self.get_skills_registry()
                if not registry.initialized:
                    registry.init()
                skill_names = [s.name for s in registry.loaded_skills.values()]
                self.repl.completer.set_skill_names(skill_names)
            except Exception:
                pass

        # Run REPL
        await self.repl.run(
            handler=self._handle_input,
            welcome_callback=self._show_welcome,
        )

        # Save session on exit
        if self.config.session.auto_save:
            self.session.save()

    async def run_once(self, command: str) -> int:
        """
        Run a single command.

        Args:
            command: Command to execute

        Returns:
            Exit code (0 for success)
        """
        result = await self._handle_input(command)

        if self.config.session.auto_save:
            self.session.save()

        return 0 if result else 1

    async def _handle_input(self, user_input: str) -> bool:
        """
        Handle user input.

        Args:
            user_input: User input string

        Returns:
            True to continue, False to exit
        """
        user_input = user_input.strip()

        if not user_input:
            return True

        # Check for slash command
        if user_input.startswith("/"):
            return await self._handle_command(user_input)

        # Natural language input - route to /run
        return await self._handle_natural_language(user_input)

    async def _handle_command(self, input_text: str) -> bool:
        """
        Handle slash command.

        Args:
            input_text: Command input (starts with /)

        Returns:
            True to continue, False to exit
        """
        # Parse command
        parts = input_text[1:].split(maxsplit=1)
        cmd_name = parts[0] if parts else ""
        cmd_args = parts[1] if len(parts) > 1 else ""

        # Built-in /export command for last output
        if cmd_name == "export":
            await self._handle_export_command(cmd_args)
            return True

        # Find command
        command = self.command_registry.get(cmd_name)

        if not command:
            self.renderer.error(f"Unknown command: /{cmd_name}")
            self.renderer.info("Type /help for available commands")
            return True

        # Parse arguments
        args = command.parse_args(cmd_args)

        # Execute command
        try:
            result = await command.execute(args, self)

            if result.should_exit:
                return False

            if result.error and not result.success:
                # Error already displayed by command
                pass

            return True

        except Exception as e:
            self.renderer.error(f"Command error: {e}")
            if self.config.debug:
                import traceback
                traceback.print_exc()
            return True

    def _is_simple_query(self, text: str) -> bool:
        """Check if query is simple greeting (not a task)."""
        # Only greetings and very basic queries - NOT task requests
        simple_greetings = ['hello', 'hi', 'hey', 'help', 'thanks', 'bye']
        text_lower = text.lower().strip()

        # Only exact greetings or very short non-task queries
        if text_lower in simple_greetings:
            return True

        # Very short AND no action words = simple
        if len(text.split()) <= 3:
            action_words = ['search', 'find', 'create', 'generate', 'send', 'compare', 'analyze', 'research', 'pdf', 'telegram']
            if not any(w in text_lower for w in action_words):
                return True

        return False

    async def _handle_natural_language(self, text: str) -> bool:
        """
        Handle natural language input (route to SwarmManager).

        Args:
            text: Natural language input

        Returns:
            True to continue
        """
        import time
        import re

        # Add to conversation
        self.session.add_message("user", text)
        start_time = time.time()

        try:
            swarm = await self.get_swarm_manager()

            # Agent Activity Panel - World-class CLI experience
            # Shows real-time agent activity with beautiful icons
            agent_icons = {
                'analyze': '🧠', 'analyzing': '🧠',
                'search': '🔍', 'searching': '🔍',
                'generate': '📝', 'generating': '📝',
                'write': '✍️', 'writing': '✍️',
                'read': '📖', 'reading': '📖',
                'save': '💾', 'saving': '💾',
                'send': '📤', 'sending': '📤',
                'decision': '🎯', 'deciding': '🎯',
                'output': '📦', 'created': '✅',
            }

            def get_agent_icon(stage: str) -> str:
                """Get icon for agent activity."""
                stage_lower = stage.lower()
                for key, icon in agent_icons.items():
                    if key in stage_lower:
                        return icon
                return '→'

            def status_callback(stage: str, detail: str = ""):
                import sys
                stage_lower = stage.lower()
                icon = get_agent_icon(stage)

                # Search results - show with icons
                if 'search' in stage_lower and 'result' in stage_lower:
                    count = 0
                    if detail:
                        import re
                        match = re.search(r'(\d+)\s*results?', detail)
                        if match:
                            count = int(match.group(1))
                    self.renderer.search_query(detail or stage, count if count else None)

                # Step progress
                elif stage_lower.startswith('step '):
                    import re
                    match = re.match(r'step\s+(\d+)/(\d+)', stage_lower)
                    if match:
                        step_num = int(match.group(1))
                        total = int(match.group(2))
                        status = 'done' if '✓' in detail or 'succeeded' in detail.lower() else 'running'
                        if '✗' in detail or 'failed' in detail.lower():
                            status = 'failed'
                        self.renderer.step_progress(step_num, total, detail, status)
                    else:
                        self.renderer.print(f"  [{self.renderer.theme.muted}]{icon}[/{self.renderer.theme.muted}] {stage}: {detail}" if detail else f"  {icon} {stage}")

                # File operations - beautiful cards
                elif any(x in stage_lower for x in ['writing', 'creating file', 'saving']):
                    self.renderer.print(f"  [bold green]{icon}[/bold green] [dim]{stage}[/dim]")
                    if detail:
                        self.renderer.print(f"     [cyan]{detail}[/cyan]")

                elif any(x in stage_lower for x in ['reading', 'loading']):
                    self.renderer.print(f"  [bold blue]{icon}[/bold blue] [dim]{stage}[/dim]")
                    if detail:
                        self.renderer.print(f"     [cyan]{detail}[/cyan]")

                # Agent activity cards - the beautiful part
                elif any(x in stage_lower for x in ['analyz', 'decision', 'generat', 'search']):
                    self.renderer.print(f"  [bold yellow]{icon}[/bold yellow] [white]{stage}[/white]")
                    if detail:
                        # Truncate long details
                        display_detail = detail[:80] + "..." if len(detail) > 80 else detail
                        self.renderer.print(f"     [dim]{display_detail}[/dim]")

                # Tool success with output path
                elif '✓' in stage and detail:
                    self.renderer.tool_output(stage.replace('✓', '').strip(), detail if '/' in detail else None)

                # Default format with icon
                else:
                    if detail:
                        self.renderer.print(f"  [cyan]{icon}[/cyan] {stage}: {detail}")
                    else:
                        self.renderer.print(f"  [cyan]{icon}[/cyan] {stage}")

                sys.stdout.flush()  # Force flush for real-time output

            # LEAN MODE: All queries go through UnifiedExecutor
            # LLM intelligently decides: needs_external_data, output_format, etc.
            # Single agent, no ensemble, no multi-agent decomposition
            # This is what actually works well!
            result = await self._execute_lean_mode(text, status_callback)

            elapsed = time.time() - start_time

            # Add response to history
            output_text = str(result.output) if result.output else "Task completed"
            self.session.add_message("assistant", output_text)

            # Display result with clear success/failure
            self.renderer.newline()

            if result.success:
                # Show steps taken if available (UnifiedExecutor)
                steps = getattr(result, 'steps_taken', None)
                if steps:
                    self.renderer.success(f"Completed in {elapsed:.1f}s ({len(steps)} steps: {' → '.join(steps)})")
                else:
                    self.renderer.success(f"Completed in {elapsed:.1f}s")

                # Extract output - handle LeanResult, EpisodeResult, ExecutionResult, dict, or string
                output = result.output if hasattr(result, 'output') else result

                file_paths = []
                summary = {}

                # Check for direct output_path attribute (LeanResult)
                if hasattr(result, 'output_path') and result.output_path:
                    fmt = getattr(result, 'output_format', 'file')
                    file_paths.append((fmt.upper(), result.output_path))

                # For ExecutionResult (from AutoAgent), extract outputs dict
                elif hasattr(output, 'outputs') and hasattr(output, 'final_output'):
                    # ExecutionResult from AutoAgent
                    outputs_dict = output.outputs or {}

                    # Extract file paths from all step outputs (deduplicated)
                    seen_paths = set()
                    for step_key, step_result in outputs_dict.items():
                        if isinstance(step_result, dict):
                            for key in ['pdf_path', 'md_path', 'output_path', 'file_path', 'image_path']:
                                if key in step_result and step_result[key]:
                                    path = step_result[key]
                                    if path not in seen_paths:
                                        file_paths.append((key.replace('_', ' ').title(), path))
                                        seen_paths.add(path)
                            # Also extract summary info
                            for key in ['success', 'ticker', 'company_name', 'word_count', 'telegram_sent']:
                                if key in step_result and step_result[key]:
                                    summary[key] = step_result[key]

                elif isinstance(output, dict):
                    for key in ['pdf_path', 'md_path', 'output_path', 'file_path', 'image_path']:
                        if key in output and output[key]:
                            file_paths.append((key.replace('_', ' ').title(), output[key]))
                    for key in ['success', 'ticker', 'company_name', 'word_count', 'telegram_sent']:
                        if key in output and output[key]:
                            summary[key] = output[key]

                elif isinstance(output, str) and not file_paths:
                    # Look for paths in string output
                    path_matches = re.findall(r'(/[\w/\-_.]+\.(pdf|md|txt|html|json|csv|png|jpg|docx))', output)
                    for match in path_matches:
                        file_paths.append(('Output', match[0]))

                # Display file paths prominently with inline preview
                if file_paths:
                    self.renderer.newline()
                    self.renderer.print("[bold green]📁 Generated Files:[/bold green]")
                    for label, path in file_paths:
                        self.renderer.print(f"   {label}: [cyan]{path}[/cyan]")

                    # Store last output path for /preview last
                    self._last_output_path = file_paths[0][1]

                    # Auto-preview first file (inline, truncated)
                    await self._auto_preview_file(file_paths[0][1])

                # Show summary
                if summary:
                    self.renderer.newline()
                    self.renderer.panel(
                        "\n".join([f"• {k}: {v}" for k, v in summary.items()]),
                        title="Summary",
                        style="green"
                    )
                elif not file_paths:
                    # No summary and no files
                    full_content = str(output)

                    # Check if content was already streamed (don't re-render)
                    was_streamed = getattr(result, 'was_streamed', False)

                    if not was_streamed:
                        # Not streamed, render markdown content beautifully
                        self.renderer.newline()
                        self.renderer.markdown(full_content)

                    # Store content in history for export (always)
                    if not hasattr(self, '_output_history'):
                        self._output_history = []
                    self._output_history.append(full_content)
                    # Keep last 20 outputs
                    if len(self._output_history) > 20:
                        self._output_history = self._output_history[-20:]

                    # Show interactive export options
                    self.renderer.newline()
                    await self._show_export_options(full_content)
            else:
                self.renderer.error(f"Failed after {elapsed:.1f}s")

                # Show error details
                error_msg = getattr(result, 'error', None)
                if not error_msg and hasattr(result, 'alerts') and result.alerts:
                    error_msg = "; ".join(result.alerts[:3])

                if error_msg:
                    self.renderer.panel(error_msg, title="Error Details", style="red")

        except Exception as e:
            self.renderer.error(f"Error: {e}")
            self.session.add_message("assistant", f"Error: {e}")
            if self.config.debug:
                import traceback
                traceback.print_exc()

        return True

    async def _execute_lean_mode(self, task: str, status_callback=None) -> Any:
        """
        Execute task in lean mode - Clean LLM-first architecture.

        Philosophy:
        - LLM is the BRAIN, not a skill
        - Skills are I/O TOOLS that work ON LLM content
        - No hardcoded rules, intelligent decisions

        Flow:
        1. ANALYZE: LLM decides what's needed (external data? output format?)
        2. INPUT: If needed, use input skills (web-search, file-read)
        3. GENERATE: LLM creates content (always happens) - WITH STREAMING
        4. OUTPUT: If needed, use output skills (docx, pdf, telegram)

        Args:
            task: Task description
            status_callback: Optional progress callback

        Returns:
            ExecutionResult with content and output path
        """
        # Import UnifiedExecutor
        try:
            from ..core.orchestration.v2.unified_executor import UnifiedExecutor, ExecutionResult
        except ImportError:
            from Jotty.core.orchestration.v2.unified_executor import UnifiedExecutor, ExecutionResult

        # Clean task: Remove any accumulated context pollution
        clean_task = self._clean_task_for_lean_execution(task)

        # Streaming callback - shows content as it's generated
        streaming_content = []
        stream_started = False

        def stream_callback(chunk: str):
            nonlocal stream_started
            import sys

            if not stream_started:
                # Start streaming output section
                stream_started = True
                self.renderer.newline()
                self.renderer.print("[dim]" + "─" * 60 + "[/dim]")

            # Print chunk without newline, flush immediately
            sys.stdout.write(chunk)
            sys.stdout.flush()
            streaming_content.append(chunk)

        # Execute with UnifiedExecutor + streaming (auto-detects provider)
        executor = UnifiedExecutor(
            status_callback=status_callback,
            stream_callback=stream_callback
        )
        result = await executor.execute(clean_task)

        # End streaming section if started
        if stream_started:
            import sys
            sys.stdout.write('\n')
            sys.stdout.flush()
            self.renderer.print("[dim]" + "─" * 60 + "[/dim]")

        # Convert to EpisodeResult-like object for compatibility
        class LeanResult:
            def __init__(self, exec_result, streamed=False):
                self.success = exec_result.success
                self.output = exec_result.content
                self.error = exec_result.error
                self.alerts = [exec_result.error] if exec_result.error else []
                self.output_path = exec_result.output_path
                self.output_format = exec_result.output_format
                self.steps_taken = exec_result.steps_taken
                self.was_streamed = streamed  # Flag to skip re-rendering

        return LeanResult(result, streamed=stream_started)

    def _clean_task_for_lean_execution(self, task: str) -> str:
        """
        Clean task string for lean execution.

        Removes any polluting context like:
        - Q-learning lessons
        - Transfer learning suggestions
        - Multi-perspective analysis context
        - Previous enrichment artifacts

        This is CRITICAL to prevent query pollution.

        Args:
            task: Original task string

        Returns:
            Clean task string
        """
        # Markers that indicate enrichment context (to be stripped)
        context_markers = [
            '\n[Multi-Perspective Analysis',
            '\nLearned Insights:',
            '\n# Transferable Learnings',
            '\n# Q-Learning Lessons',
            '\n## Task Type Pattern',
            '\n## Role Advice',
            '\n## Meta-Learning Advice',
            '\n\n---\n',  # Common separator before context
            '\nBased on previous learnings:',
            '\nRecommended approach:',
        ]

        cleaned = task
        for marker in context_markers:
            if marker in cleaned:
                cleaned = cleaned.split(marker)[0]

        return cleaned.strip()

    async def _handle_export_command(self, args: str):
        """
        Handle /export command to export outputs.

        Usage:
            /export             - Export last output (interactive)
            /export c/d/p/m     - Export last with format(s)
            /export list        - Show recent outputs to choose
            /export 2 cdp       - Export 2nd-to-last output as copy+docx+pdf
            /export all m       - Export all outputs to markdown
        """
        # Initialize output history if needed
        if not hasattr(self, '_output_history'):
            self._output_history = []

        args_parts = args.strip().lower().split()

        # /export list - show recent outputs
        if args_parts and args_parts[0] == 'list':
            await self._show_output_list()
            return

        # /export all [formats] - export entire conversation
        if args_parts and args_parts[0] == 'all':
            formats = args_parts[1] if len(args_parts) > 1 else ''
            await self._export_all_outputs(formats)
            return

        # Parse message index and format flags
        msg_index = 1  # Default: last message
        format_flags = ''

        for part in args_parts:
            if part.isdigit():
                msg_index = int(part)
            elif part.replace('-', '').isdigit():
                msg_index = abs(int(part))
            else:
                format_flags += part

        # Get the selected output
        if not self._output_history:
            self.renderer.warning("No outputs to export. Run a query first.")
            return

        if msg_index > len(self._output_history):
            self.renderer.warning(f"Only {len(self._output_history)} outputs available. Use /export list")
            return

        # Get content (1 = last, 2 = second-to-last, etc.)
        content = self._output_history[-msg_index]

        if not format_flags:
            # Show interactive options
            self.renderer.info(f"Exporting output #{msg_index} (of {len(self._output_history)})")
            await self._show_export_options(content)
            return

        # Process export flags
        if 'c' in format_flags:
            await self._copy_to_clipboard(content)
        if 'd' in format_flags:
            await self._export_to_docx(content)
        if 'p' in format_flags:
            await self._export_to_pdf(content)
        if 'm' in format_flags:
            await self._export_to_markdown(content)

        if not any(c in format_flags for c in 'cdpm'):
            self.renderer.info("Usage: /export [N] [c]opy [d]ocx [p]df [m]arkdown")
            self.renderer.info("  /export         - last output, interactive")
            self.renderer.info("  /export cdp     - last output as copy+docx+pdf")
            self.renderer.info("  /export 2 m     - 2nd-to-last as markdown")
            self.renderer.info("  /export list    - show all outputs")
            self.renderer.info("  /export all m   - entire conversation")

    async def _show_output_list(self):
        """Show list of recent outputs for selection."""
        if not hasattr(self, '_output_history') or not self._output_history:
            self.renderer.warning("No outputs yet.")
            return

        self.renderer.print("\n[bold]Recent Outputs:[/bold]")
        for i, content in enumerate(reversed(self._output_history), 1):
            preview = content[:80].replace('\n', ' ')
            if len(content) > 80:
                preview += "..."
            self.renderer.print(f"  [cyan]{i}[/cyan]: {preview}")

        self.renderer.print("\n[dim]Use: /export N cdpm (e.g., /export 2 dp)[/dim]")

    async def _export_all_outputs(self, formats: str):
        """Export all outputs as single document."""
        if not hasattr(self, '_output_history') or not self._output_history:
            self.renderer.warning("No outputs to export.")
            return

        # Combine all outputs
        combined = "\n\n---\n\n".join([
            f"## Output {i+1}\n\n{content}"
            for i, content in enumerate(self._output_history)
        ])

        formats = formats or 'm'  # Default to markdown

        if 'c' in formats:
            await self._copy_to_clipboard(combined)
        if 'd' in formats:
            await self._export_to_docx(combined)
        if 'p' in formats:
            await self._export_to_pdf(combined)
        if 'm' in formats:
            await self._export_to_markdown(combined)

    async def _show_export_options(self, content: str):
        """
        Show interactive export options after output.

        Options:
        - [c] Copy to clipboard
        - [d] Download as DOCX
        - [p] Download as PDF
        - [m] Save as Markdown
        - [Enter] Continue
        """
        self.renderer.print("[dim]" + "─" * 60 + "[/dim]")
        self.renderer.print(
            "[dim]Export:[/dim] "
            "[bold cyan]\\[c][/bold cyan]opy  "
            "[bold cyan]\\[d][/bold cyan]ocx  "
            "[bold cyan]\\[p][/bold cyan]df  "
            "[bold cyan]\\[m][/bold cyan]arkdown  "
            "[dim]\\[Enter] done[/dim]  "
            "[dim](combine: cdpm)[/dim]"
        )

        try:
            # Get user choice - supports multiple actions like "cdp" = copy + docx + pdf
            choice = input("  → ").strip().lower()

            if not choice:
                return  # Done

            # Process each character as an action
            actions_taken = []

            if 'c' in choice:
                await self._copy_to_clipboard(content)
                actions_taken.append('copy')

            if 'd' in choice:
                await self._export_to_docx(content)
                actions_taken.append('docx')

            if 'p' in choice:
                await self._export_to_pdf(content)
                actions_taken.append('pdf')

            if 'm' in choice:
                await self._export_to_markdown(content)
                actions_taken.append('markdown')

        except (EOFError, KeyboardInterrupt):
            pass  # User cancelled

    async def _copy_to_clipboard(self, content: str):
        """Copy content to system clipboard."""
        try:
            import subprocess

            # Try different clipboard methods
            # 1. xclip (Linux)
            try:
                process = subprocess.Popen(
                    ['xclip', '-selection', 'clipboard'],
                    stdin=subprocess.PIPE
                )
                process.communicate(content.encode('utf-8'))
                self.renderer.success("Copied to clipboard!")
                return
            except FileNotFoundError:
                pass

            # 2. xsel (Linux)
            try:
                process = subprocess.Popen(
                    ['xsel', '--clipboard', '--input'],
                    stdin=subprocess.PIPE
                )
                process.communicate(content.encode('utf-8'))
                self.renderer.success("Copied to clipboard!")
                return
            except FileNotFoundError:
                pass

            # 3. pbcopy (macOS)
            try:
                process = subprocess.Popen(['pbcopy'], stdin=subprocess.PIPE)
                process.communicate(content.encode('utf-8'))
                self.renderer.success("Copied to clipboard!")
                return
            except FileNotFoundError:
                pass

            # 4. pyperclip fallback
            try:
                import pyperclip
                pyperclip.copy(content)
                self.renderer.success("Copied to clipboard!")
                return
            except ImportError:
                pass

            self.renderer.warning("Clipboard not available. Install xclip or pyperclip.")

        except Exception as e:
            self.renderer.error(f"Copy failed: {e}")

    async def _export_to_docx(self, content: str):
        """Export content to DOCX file with LaTeX math support."""
        from pathlib import Path
        from datetime import datetime
        import subprocess

        output_dir = Path.home() / "jotty" / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"export_{timestamp}.docx"

        # Prepare content with proper LaTeX formatting
        prepared_content = self._prepare_content_with_latex(content)

        # Check if content has LaTeX
        has_latex = '$' in content or '\\[' in content or '\\(' in content

        try:
            # Method 1: Use pandoc for LaTeX → DOCX (converts to OMML equations)
            if has_latex:
                try:
                    md_path = output_dir / f"export_{timestamp}.md"
                    md_path.write_text(prepared_content)
                    subprocess.run(
                        ['pandoc', str(md_path), '-o', str(output_path)],
                        check=True, capture_output=True
                    )
                    self.renderer.success(f"Saved: {output_path}")
                    return
                except (subprocess.CalledProcessError, FileNotFoundError):
                    pass  # Fall through to other methods

            registry = self.get_skills_registry()
            skill = registry.get_skill('docx-tools')

            if skill:
                # Try professional checklist first if content looks like a checklist
                if '- [ ]' in content or '- [x]' in content:
                    tool = skill.tools.get('create_professional_checklist_tool')
                    if tool:
                        result = tool({
                            'content': content,
                            'output_path': str(output_path),
                            'title': 'Export'
                        })
                        if result.get('success'):
                            self.renderer.success(f"Saved: {result.get('file_path')}")
                            return

                # Fall back to regular docx
                tool = skill.tools.get('create_docx_tool')
                if tool:
                    result = tool({
                        'content': content,
                        'output_path': str(output_path)
                    })
                    if result.get('success'):
                        self.renderer.success(f"Saved: {result.get('file_path')}")
                        return

            # Ultimate fallback: save as markdown
            await self._export_to_markdown(content)

        except Exception as e:
            self.renderer.error(f"DOCX export failed: {e}")

    async def _export_to_pdf(self, content: str):
        """Export content to PDF file with LaTeX math support."""
        from pathlib import Path
        from datetime import datetime
        import subprocess

        output_dir = Path.home() / "jotty" / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        md_path = output_dir / f"export_{timestamp}.md"
        pdf_path = output_dir / f"export_{timestamp}.pdf"

        # Prepare content with proper LaTeX formatting
        prepared_content = self._prepare_content_with_latex(content)

        try:
            # Save markdown first
            md_path.write_text(prepared_content)

            # Check if content has LaTeX
            has_latex = '$' in content or '\\[' in content or '\\(' in content

            # Method 1: Pandoc with xelatex (best for LaTeX)
            try:
                cmd = ['pandoc', str(md_path), '-o', str(pdf_path),
                       '--pdf-engine=xelatex', '-V', 'geometry:margin=1in']
                if has_latex:
                    cmd.extend(['--mathjax'])  # Enable math processing
                subprocess.run(cmd, check=True, capture_output=True)
                self.renderer.success(f"Saved: {pdf_path}")
                return
            except (subprocess.CalledProcessError, FileNotFoundError):
                pass

            # Method 2: Pandoc with pdflatex
            try:
                subprocess.run(
                    ['pandoc', str(md_path), '-o', str(pdf_path)],
                    check=True, capture_output=True
                )
                self.renderer.success(f"Saved: {pdf_path}")
                return
            except (subprocess.CalledProcessError, FileNotFoundError):
                pass

            # Method 3: WeasyPrint with MathJax (for LaTeX rendering)
            try:
                import weasyprint

                # Include MathJax for LaTeX rendering
                mathjax_script = ""
                if has_latex:
                    mathjax_script = """
                    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
                    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
                    """

                html_content = f"""
                <!DOCTYPE html>
                <html><head>
                <meta charset="UTF-8">
                {mathjax_script}
                <style>
                body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 40px; line-height: 1.6; color: #333; }}
                h1 {{ color: #294172; border-bottom: 2px solid #294172; padding-bottom: 10px; }}
                h2 {{ color: #294172; }}
                h3 {{ color: #4a6fa5; }}
                ul, ol {{ margin-left: 20px; }}
                li {{ margin: 5px 0; }}
                code {{ background: #f4f4f4; padding: 2px 6px; border-radius: 3px; font-family: 'Consolas', monospace; }}
                pre {{ background: #f4f4f4; padding: 15px; border-radius: 5px; overflow-x: auto; }}
                .math-block {{ text-align: center; margin: 20px 0; font-size: 1.1em; }}
                .math-inline {{ }}
                blockquote {{ border-left: 4px solid #294172; margin: 10px 0; padding-left: 15px; color: #555; }}
                table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 10px; text-align: left; }}
                th {{ background: #294172; color: white; }}
                tr:nth-child(even) {{ background: #f9f9f9; }}
                </style>
                </head><body>
                {self._markdown_to_html(content)}
                </body></html>
                """
                weasyprint.HTML(string=html_content).write_pdf(str(pdf_path))
                self.renderer.success(f"Saved: {pdf_path}")
                return
            except ImportError:
                pass
            except Exception as e:
                logger.debug(f"WeasyPrint failed: {e}")

            # Method 4: Document-converter skill
            registry = self.get_skills_registry()
            skill = registry.get_skill('document-converter')
            if skill:
                tool = skill.tools.get('convert_to_pdf_tool')
                if tool:
                    result = tool({
                        'input_path': str(md_path),
                        'output_path': str(pdf_path)
                    })
                    if result.get('success'):
                        self.renderer.success(f"Saved: {pdf_path}")
                        return

            # Fallback
            self.renderer.success(f"Saved markdown: {md_path}")
            self.renderer.warning("PDF: Install pandoc (`apt install pandoc texlive-xetex`) or weasyprint (`pip install weasyprint`)")

        except Exception as e:
            self.renderer.error(f"PDF export failed: {e}")

    def _markdown_to_html(self, md_content: str) -> str:
        """Convert markdown to HTML with LaTeX support."""
        import re
        html = md_content

        # Protect LaTeX blocks from other processing
        latex_blocks = []

        # Block LaTeX: $$...$$
        def save_block_latex(match):
            latex_blocks.append(('block', match.group(1)))
            return f'__LATEX_BLOCK_{len(latex_blocks)-1}__'
        html = re.sub(r'\$\$(.+?)\$\$', save_block_latex, html, flags=re.DOTALL)

        # Inline LaTeX: $...$
        def save_inline_latex(match):
            latex_blocks.append(('inline', match.group(1)))
            return f'__LATEX_INLINE_{len(latex_blocks)-1}__'
        html = re.sub(r'\$(.+?)\$', save_inline_latex, html)

        # Headers
        html = re.sub(r'^### (.+)$', r'<h3>\1</h3>', html, flags=re.MULTILINE)
        html = re.sub(r'^## (.+)$', r'<h2>\1</h2>', html, flags=re.MULTILINE)
        html = re.sub(r'^# (.+)$', r'<h1>\1</h1>', html, flags=re.MULTILINE)

        # Bold and italic
        html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html)
        html = re.sub(r'\*(.+?)\*', r'<em>\1</em>', html)

        # Code
        html = re.sub(r'`(.+?)`', r'<code>\1</code>', html)

        # Lists
        html = re.sub(r'^- \[ \] (.+)$', r'<li>☐ \1</li>', html, flags=re.MULTILINE)
        html = re.sub(r'^- \[x\] (.+)$', r'<li>☑ \1</li>', html, flags=re.MULTILINE)
        html = re.sub(r'^- (.+)$', r'<li>\1</li>', html, flags=re.MULTILINE)
        html = re.sub(r'^• (.+)$', r'<li>\1</li>', html, flags=re.MULTILINE)

        # Paragraphs
        html = re.sub(r'\n\n', r'</p><p>', html)
        html = f'<p>{html}</p>'

        # Restore LaTeX with MathJax/KaTeX compatible format
        for i, (latex_type, latex_content) in enumerate(latex_blocks):
            if latex_type == 'block':
                html = html.replace(f'__LATEX_BLOCK_{i}__',
                    f'<div class="math-block">\\[{latex_content}\\]</div>')
            else:
                html = html.replace(f'__LATEX_INLINE_{i}__',
                    f'<span class="math-inline">\\({latex_content}\\)</span>')

        return html

    def _prepare_content_with_latex(self, content: str) -> str:
        """Prepare content for export, ensuring LaTeX is properly formatted."""
        import re

        # Ensure LaTeX delimiters are consistent
        # Convert \[ \] to $$ $$ for pandoc compatibility
        content = re.sub(r'\\\[(.+?)\\\]', r'$$\1$$', content, flags=re.DOTALL)
        content = re.sub(r'\\\((.+?)\\\)', r'$\1$', content)

        return content

    async def _export_to_markdown(self, content: str):
        """Export content to Markdown file."""
        from pathlib import Path
        from datetime import datetime

        output_dir = Path.home() / "jotty" / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"export_{timestamp}.md"

        try:
            output_path.write_text(content)
            self.renderer.success(f"Saved: {output_path}")
        except Exception as e:
            self.renderer.error(f"Markdown export failed: {e}")

    async def _auto_preview_file(self, file_path: str, max_lines: int = 15):
        """Show inline preview of generated file."""
        from pathlib import Path

        path = Path(file_path)
        if not path.exists():
            return

        suffix = path.suffix.lower()

        self.renderer.newline()
        self.renderer.print("[dim]─── Preview ───[/dim]")

        try:
            # Quick preview based on file type
            if suffix in ['.md', '.txt']:
                content = path.read_text()
                lines = content.split('\n')[:max_lines]
                for line in lines:
                    print(line)
                if len(content.split('\n')) > max_lines:
                    self.renderer.print(f"[dim]... ({len(content.split(chr(10))) - max_lines} more lines)[/dim]")

            elif suffix == '.pdf':
                import subprocess
                import shutil
                pdftotext = shutil.which('pdftotext')
                if pdftotext:
                    result = subprocess.run(
                        [pdftotext, '-layout', '-nopgbrk', str(path), '-'],
                        capture_output=True, text=True, timeout=10
                    )
                    if result.returncode == 0:
                        lines = result.stdout.split('\n')[:max_lines]
                        for line in lines:
                            print(line)
                        if len(result.stdout.split('\n')) > max_lines:
                            self.renderer.print("[dim]...[/dim]")
                else:
                    self.renderer.print("[dim]Install pdftotext for preview: apt install poppler-utils[/dim]")

            elif suffix in ['.docx', '.doc']:
                import subprocess
                import shutil
                catdoc = shutil.which('catdoc')
                if catdoc:
                    result = subprocess.run(
                        [catdoc, str(path)],
                        capture_output=True, text=True, timeout=10
                    )
                    if result.returncode == 0:
                        lines = result.stdout.split('\n')[:max_lines]
                        for line in lines:
                            print(line)
                        if len(result.stdout.split('\n')) > max_lines:
                            self.renderer.print("[dim]...[/dim]")
                else:
                    self.renderer.print("[dim]Install catdoc for preview: apt install catdoc[/dim]")

            elif suffix in ['.png', '.jpg', '.jpeg', '.gif']:
                import subprocess
                import shutil
                chafa = shutil.which('chafa')
                if chafa:
                    result = subprocess.run(
                        [chafa, '--size=60x20', str(path)],
                        capture_output=True, text=True, timeout=10
                    )
                    if result.returncode == 0:
                        print(result.stdout)
                else:
                    self.renderer.print("[dim]Install chafa for image preview: apt install chafa[/dim]")

            else:
                # Generic file - show first few lines
                try:
                    with open(path, 'r', errors='replace') as f:
                        for i, line in enumerate(f):
                            if i >= max_lines:
                                self.renderer.print("[dim]...[/dim]")
                                break
                            print(line.rstrip())
                except:
                    self.renderer.print("[dim]Binary file[/dim]")

        except Exception as e:
            logger.debug(f"Auto-preview failed: {e}")

        self.renderer.print("[dim]─── /preview for more ───[/dim]")

    async def _try_auto_resume(self):
        """Try to auto-resume the last session."""
        try:
            sessions = self.session.list_sessions()
            if not sessions:
                return

            # Find the most recent session that isn't current
            for session_info in sessions:
                if session_info['session_id'] != self.session.session_id:
                    # Load this session
                    self.session.load(session_info['session_id'])
                    msg_count = len(self.session.conversation_history)
                    self.renderer.info(
                        f"Auto-resumed session: {session_info['session_id'][:8]}... "
                        f"({msg_count} messages)"
                    )

                    # Restore output history
                    self._output_history = []
                    for msg in self.session.conversation_history:
                        if msg.role == "assistant" and len(msg.content) > 100:
                            self._output_history.append(msg.content)

                    break
        except Exception as e:
            logger.debug(f"Auto-resume failed: {e}")

    def _show_welcome(self):
        """Show welcome banner."""
        from .. import __version__ as jotty_version
        from . import __version__ as cli_version

        self.renderer.welcome(version=f"{jotty_version} (CLI {cli_version})")

        # Show quick stats
        try:
            registry = self.get_skills_registry()
            if not registry.initialized:
                registry.init()
            skill_count = len(registry.loaded_skills)
            self.renderer.info(f"Skills loaded: {skill_count}")
        except Exception:
            pass

    async def execute_task(self, goal: str) -> Any:
        """
        Execute a task directly.

        Args:
            goal: Task goal

        Returns:
            Execution result
        """
        swarm = await self.get_swarm_manager()
        return await swarm.run(goal)


def main():
    """Main entry point."""
    import sys
    from . import __main__
    sys.exit(__main__.main())


if __name__ == "__main__":
    main()
