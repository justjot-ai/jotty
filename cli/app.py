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
        """Check if query is simple (should use fast mode)."""
        simple_patterns = [
            'hello', 'hi', 'hey', 'help', 'what', 'who', 'how',
            'explain', 'define', 'list', 'show', 'tell me'
        ]
        text_lower = text.lower().strip()

        # Short queries are simple
        if len(text.split()) <= 5:
            return True

        # Queries starting with simple patterns
        for pattern in simple_patterns:
            if text_lower.startswith(pattern):
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

            # Detect if fast mode should be used
            fast_mode = self._is_simple_query(text)

            # Status callback for streaming updates
            def status_callback(stage: str, detail: str = ""):
                # Print status update on new line
                if detail:
                    self.renderer.print(f"  [cyan]→[/cyan] {stage}: {detail}")
                else:
                    self.renderer.print(f"  [cyan]→[/cyan] {stage}")

            # Show what mode we're using
            if fast_mode:
                self.renderer.info("Fast mode: direct execution")
            else:
                self.renderer.info("Full mode: research → plan → execute")

            if fast_mode:
                # Fast mode: skip autonomous setup, direct to agent
                status_callback("Executing agent")
                result = await swarm.run(text, skip_autonomous_setup=True)
            else:
                # Full mode with status updates
                status_callback("Parsing intent")
                result = await swarm.run(text, status_callback=status_callback)

            elapsed = time.time() - start_time

            # Add response to history
            output_text = str(result.output) if result.output else "Task completed"
            self.session.add_message("assistant", output_text)

            # Display result with clear success/failure
            self.renderer.newline()

            if result.success:
                self.renderer.success(f"Completed in {elapsed:.1f}s")

                # Extract output - handle EpisodeResult, ExecutionResult, dict, or string
                output = result.output if hasattr(result, 'output') else result

                # For ExecutionResult (from AutoAgent), extract outputs dict
                file_paths = []
                summary = {}

                # Check if output is an ExecutionResult object
                if hasattr(output, 'outputs') and hasattr(output, 'final_output'):
                    # ExecutionResult from AutoAgent
                    outputs_dict = output.outputs or {}
                    final_output = output.final_output

                    # Extract file paths from all step outputs
                    for step_key, step_result in outputs_dict.items():
                        if isinstance(step_result, dict):
                            for key in ['pdf_path', 'md_path', 'output_path', 'file_path', 'image_path']:
                                if key in step_result and step_result[key]:
                                    file_paths.append((key.replace('_', ' ').title(), step_result[key]))
                            # Also extract summary info
                            for key in ['success', 'ticker', 'company_name', 'word_count', 'telegram_sent']:
                                if key in step_result and step_result[key]:
                                    summary[key] = step_result[key]

                    # Also check final_output for paths
                    if isinstance(final_output, dict):
                        for key in ['pdf_path', 'md_path', 'output_path', 'file_path']:
                            if key in final_output and final_output[key]:
                                file_paths.append((key.replace('_', ' ').title(), final_output[key]))

                elif isinstance(output, dict):
                    for key in ['pdf_path', 'md_path', 'output_path', 'file_path', 'image_path']:
                        if key in output and output[key]:
                            file_paths.append((key.replace('_', ' ').title(), output[key]))
                    for key in ['success', 'ticker', 'company_name', 'word_count', 'telegram_sent']:
                        if key in output and output[key]:
                            summary[key] = output[key]

                elif isinstance(output, str):
                    # Look for paths in string output
                    path_matches = re.findall(r'(/[\w/\-_.]+\.(pdf|md|txt|html|json|csv|png|jpg))', output)
                    for match in path_matches:
                        file_paths.append(('Output', match[0]))

                # Display file paths prominently
                if file_paths:
                    self.renderer.newline()
                    self.renderer.print("[bold green]📁 Generated Files:[/bold green]")
                    for label, path in file_paths:
                        self.renderer.print(f"   {label}: [cyan]{path}[/cyan]")

                # Show summary
                if summary:
                    self.renderer.newline()
                    self.renderer.panel(
                        "\n".join([f"• {k}: {v}" for k, v in summary.items()]),
                        title="Summary",
                        style="green"
                    )
                elif not file_paths:
                    # No summary and no files, show brief output
                    self.renderer.newline()
                    if hasattr(output, 'final_output') and output.final_output:
                        out_str = str(output.final_output)[:300]
                    else:
                        out_str = str(output)[:300]
                    if len(out_str) >= 300:
                        out_str += "..."
                    self.renderer.panel(out_str, title="Output", style="green")
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
