"""
SDK Test Command
================

Test the SDK client with real-time event display in the CLI.

Usage:
    /sdk chat "Hello there"
    /sdk workflow "Calculate 7 * 9"
    /sdk stream "Tell me a story"
    /sdk events on/off
"""

import asyncio
import logging
from typing import Optional, Dict, Any
from datetime import datetime

from .base import BaseCommand, CommandResult, ParsedArgs

logger = logging.getLogger(__name__)


class SDKCommand(BaseCommand):
    """
    Test the SDK client with real-time event visualization.

    Shows all SDK events as they happen:
    - START, THINKING, PLANNING
    - SKILL_START, SKILL_COMPLETE
    - AGENT_START, AGENT_COMPLETE
    - STREAM, COMPLETE, ERROR
    """

    name = "sdk"
    aliases = ["test-sdk", "sdktest"]
    description = "Test SDK client with event display"
    usage = "/sdk <mode> <message>\n  Modes: chat, workflow, stream, events"
    category = "development"

    def __init__(self):
        super().__init__()
        self._show_events = True
        self._event_icons = {
            "start": "🚀",
            "thinking": "🧠",
            "planning": "📋",
            "skill_start": "🔧",
            "skill_complete": "✅",
            "agent_start": "🤖",
            "agent_complete": "✅",
            "stream": "💬",
            "complete": "🎉",
            "error": "❌",
        }

    async def execute(
        self,
        args: ParsedArgs,
        cli: "JottyCLI"
    ) -> CommandResult:
        """Execute SDK test command."""
        if not args.positional:
            return self._show_help(cli)

        mode = args.positional[0].lower()
        message = " ".join(args.positional[1:]) if len(args.positional) > 1 else ""

        if mode in ("help", "?"):
            return self._show_help(cli)
        elif mode == "events":
            return self._toggle_events(cli, message)
        elif mode == "chat":
            return await self._test_chat(cli, message)
        elif mode == "workflow":
            return await self._test_workflow(cli, message)
        elif mode == "stream":
            return await self._test_stream(cli, message)
        elif mode == "skill":
            return await self._test_skill(cli, message)
        elif mode == "health":
            return await self._test_health(cli)
        elif mode == "skills":
            return await self._list_skills(cli)
        else:
            cli.renderer.warning(f"Unknown mode: {mode}")
            return self._show_help(cli)

    def _show_help(self, cli) -> CommandResult:
        """Show SDK command help."""
        cli.renderer.panel("""
[bold cyan]SDK Test Command[/bold cyan]

Test the SDK client with real-time event visualization.

[bold]Usage:[/bold]
  /sdk chat "Hello there"           - Test chat mode
  /sdk workflow "Calculate 7 * 9"   - Test workflow mode
  /sdk stream "Tell me a story"     - Test streaming mode
  /sdk skill web-search "AI news"   - Test direct skill
  /sdk health                       - Check SDK health
  /sdk skills                       - List available skills
  /sdk events on/off                - Toggle event display

[bold]Events shown:[/bold]
  🚀 START      - Request started
  🧠 THINKING   - LLM thinking
  📋 PLANNING   - Workflow planning
  🔧 SKILL_START - Skill execution started
  ✅ SKILL_COMPLETE - Skill execution done
  🤖 AGENT_START - Agent started
  💬 STREAM     - Streaming content
  🎉 COMPLETE   - Request complete
  ❌ ERROR      - Error occurred
""", title="SDK Help", style="cyan")
        return CommandResult.ok()

    def _toggle_events(self, cli, setting: str) -> CommandResult:
        """Toggle event display."""
        if setting.lower() in ("on", "true", "1"):
            self._show_events = True
            cli.renderer.success("Event display: ON")
        elif setting.lower() in ("off", "false", "0"):
            self._show_events = False
            cli.renderer.success("Event display: OFF")
        else:
            cli.renderer.info(f"Event display: {'ON' if self._show_events else 'OFF'}")
        return CommandResult.ok()

    def _create_event_callback(self, cli) -> callable:
        """Create event callback for displaying events."""
        def event_callback(event):
            if not self._show_events:
                return

            event_type = event.type.value if hasattr(event.type, 'value') else str(event.type)
            icon = self._event_icons.get(event_type, "→")

            # Format timestamp
            ts = datetime.now().strftime("%H:%M:%S")

            # Format event data
            data_str = ""
            if event.data:
                if isinstance(event.data, dict):
                    # Show key details only
                    for key in ['message', 'goal', 'skill', 'agent', 'error']:
                        if key in event.data:
                            val = str(event.data[key])[:60]
                            data_str = f": {val}"
                            break
                else:
                    data_str = f": {str(event.data)[:60]}"

            # Color based on event type
            color = "cyan"
            if "complete" in event_type:
                color = "green"
            elif "error" in event_type:
                color = "red"
            elif "start" in event_type:
                color = "yellow"

            cli.renderer.print(
                f"  [{cli.renderer.theme.muted}]{ts}[/{cli.renderer.theme.muted}] "
                f"[bold {color}]{icon}[/bold {color}] "
                f"[{color}]{event_type.upper()}{data_str}[/{color}]"
            )

        return event_callback

    async def _test_chat(self, cli, message: str) -> CommandResult:
        """World-class interactive chat with arrow keys, history, and rich UI."""
        try:
            from prompt_toolkit import PromptSession
            from prompt_toolkit.history import InMemoryHistory
            from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
            from prompt_toolkit.key_binding import KeyBindings
            from prompt_toolkit.styles import Style
            from prompt_toolkit.formatted_text import HTML
        except ImportError:
            cli.renderer.warning("prompt_toolkit not installed. Using basic input.")
            return await self._test_chat_basic(cli, message)

        try:
            from ...sdk.client import Jotty
            from ...core.foundation.types.sdk_types import SDKEventType

            # Create client
            client = Jotty().use_local()

            # Engaging status messages (only show these, hide technical ones)
            _status_messages = {
                'Searching': '🔍 Searching the web...',
                'Analyzing': '🧠 Analyzing...',
                'Generating': '✍️ Crafting response...',
                'Processing': '💭 Processing...',
                'Retrying': '🔄 Refining response...',
                'Thinking': '💭 Thinking...',
                'Reading': '📖 Reading file...',
            }
            # Technical statuses to hide from user
            _hidden_statuses = {'Decision', 'Generated', 'Preparing', 'step'}
            _last_status = [None]

            # Event callback for status updates
            def chat_event_callback(event):
                event_type = event.type.value if hasattr(event.type, 'value') else str(event.type)

                # Handle thinking events - show engaging messages only
                if event_type == 'thinking' and event.data and isinstance(event.data, dict):
                    status = event.data.get('status', '')

                    # Skip technical/internal status messages
                    if status in _hidden_statuses or '=' in status:
                        return

                    # Show engaging message for known statuses
                    if status in _status_messages:
                        msg = _status_messages[status]
                        if _last_status[0] != status:
                            _last_status[0] = status
                            print(f"\033[90m  {msg}\033[0m")
                    elif 'search' in status.lower():
                        if _last_status[0] != 'search':
                            _last_status[0] = 'search'
                            print(f"\033[90m  🔍 Searching the web...\033[0m")
                    return

                # Only show other events if _show_events is enabled
                if not self._show_events:
                    return

                if event_type == 'error':
                    icon = self._event_icons.get(event_type, "→")
                    error_msg = ""
                    if event.data and isinstance(event.data, dict):
                        error_msg = event.data.get('error', '')
                        # Don't show raw tracebacks
                        if 'Traceback' in error_msg or 'File "/' in error_msg:
                            error_msg = "An error occurred. Please try again."
                    print(f"  {icon} {error_msg}")

            for event_type in SDKEventType:
                client.on(event_type, chat_event_callback)

            # Conversation history
            conv_history = []

            # Custom key bindings
            kb = KeyBindings()

            @kb.add('c-l')
            def clear_screen(event):
                """Clear screen with Ctrl+L."""
                event.app.renderer.clear()

            @kb.add('c-c')
            def interrupt(event):
                """Handle Ctrl+C gracefully."""
                event.app.exit(result=None)

            # Prompt style
            style = Style.from_dict({
                'prompt': '#00ff00 bold',
                'input': '#ffffff',
            })

            # Create prompt session with history
            input_history = InMemoryHistory()
            session = PromptSession(
                history=input_history,
                auto_suggest=AutoSuggestFromHistory(),
                key_bindings=kb,
                style=style,
                enable_history_search=True,
            )

            # Header
            print()
            cli.renderer.print("[bold cyan]╔════════════════════════════════════════════════════════════════╗[/bold cyan]")
            cli.renderer.print("[bold cyan]║                      Jotty Chat                                ║[/bold cyan]")
            cli.renderer.print("[bold cyan]╠════════════════════════════════════════════════════════════════╣[/bold cyan]")
            cli.renderer.print("[bold cyan]║[/bold cyan] [dim]↑/↓[/dim] History  [dim]Ctrl+L[/dim] Clear  [dim]Ctrl+C[/dim] Cancel  [dim]/exit[/dim] Quit       [bold cyan]║[/bold cyan]")
            cli.renderer.print("[bold cyan]║[/bold cyan] [dim]/clear[/dim] Reset conversation  [dim]/events on|off[/dim] Toggle events  [bold cyan]║[/bold cyan]")
            cli.renderer.print("[bold cyan]╚════════════════════════════════════════════════════════════════╝[/bold cyan]")
            print()

            # Process initial message if provided
            if message:
                print(f"\033[92myou>\033[0m {message}")
                await self._process_chat_message(cli, client, message, conv_history)

            # Main chat loop
            while True:
                try:
                    # Get input with rich prompt
                    user_input = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: session.prompt(HTML('<prompt>you></prompt> '))
                    )

                    if user_input is None:
                        break

                    user_input = user_input.strip()

                    # Handle commands
                    if user_input.lower() in ('/exit', '/quit', 'exit', 'quit', 'bye'):
                        print("\n\033[90mGoodbye!\033[0m")
                        break

                    if not user_input:
                        continue

                    if user_input.lower() == '/clear':
                        conv_history.clear()
                        # Clear screen
                        import os
                        os.system('clear' if os.name != 'nt' else 'cls')
                        # Reprint header
                        cli.renderer.print("[bold cyan]╔════════════════════════════════════════════════════════════════╗[/bold cyan]")
                        cli.renderer.print("[bold cyan]║                      Jotty Chat                                ║[/bold cyan]")
                        cli.renderer.print("[bold cyan]╚════════════════════════════════════════════════════════════════╝[/bold cyan]")
                        print("\033[90mConversation cleared.\033[0m\n")
                        continue

                    if user_input.lower().startswith('/events'):
                        parts = user_input.split()
                        if len(parts) > 1:
                            self._show_events = parts[1].lower() in ('on', 'true', '1')
                        print(f"\033[90mEvents: {'ON' if self._show_events else 'OFF'}\033[0m")
                        continue

                    if user_input.startswith('/'):
                        print(f"\033[90mUnknown command: {user_input}\033[0m")
                        continue

                    # Process the message
                    await self._process_chat_message(cli, client, user_input, conv_history)

                except KeyboardInterrupt:
                    print("\n\033[90mType /exit to quit or continue chatting.\033[0m")
                    continue
                except EOFError:
                    break

            await client.close()
            return CommandResult.ok()

        except Exception as e:
            cli.renderer.error(f"SDK chat error: {e}")
            if cli.config.debug:
                import traceback
                traceback.print_exc()
            return CommandResult.fail(str(e))

    async def _test_chat_basic(self, cli, message: str) -> CommandResult:
        """Basic chat fallback when prompt_toolkit is not available."""
        try:
            from ...sdk.client import Jotty
            from ...core.foundation.types.sdk_types import SDKEventType

            client = Jotty().use_local()
            conv_history = []

            _status_msgs = {
                'Searching': '🔍 Searching the web...',
                'Analyzing': '🧠 Analyzing...',
                'Generating': '✍️ Crafting response...',
                'Processing': '💭 Processing...',
                'Retrying': '🔄 Refining response...',
                'Thinking': '💭 Thinking...',
            }
            _hidden = {'Decision', 'Generated', 'Preparing', 'step'}
            _last_status = [None]

            def chat_event_callback(event):
                event_type = event.type.value if hasattr(event.type, 'value') else str(event.type)

                if event_type == 'thinking' and event.data and isinstance(event.data, dict):
                    status = event.data.get('status', '')
                    # Skip technical statuses
                    if status in _hidden or '=' in status:
                        return
                    if status in _status_msgs:
                        if _last_status[0] != status:
                            _last_status[0] = status
                            print(f"\033[90m  {_status_msgs[status]}\033[0m")
                    elif 'search' in status.lower():
                        if _last_status[0] != 'search':
                            _last_status[0] = 'search'
                            print(f"\033[90m  🔍 Searching the web...\033[0m")
                    return

                if not self._show_events:
                    return
                if event_type == 'error':
                    print(f"  ❌ Something went wrong. Please try again.")

            for event_type in SDKEventType:
                client.on(event_type, chat_event_callback)

            cli.renderer.print("\n[bold cyan]═══ Jotty Chat (Basic Mode) ═══[/bold cyan]")
            cli.renderer.print("[dim]Type /exit to quit[/dim]\n")

            if message:
                await self._process_chat_message(cli, client, message, conv_history)

            while True:
                try:
                    user_input = input("\033[92myou>\033[0m ")
                    if user_input.lower() in ('/exit', 'exit', 'quit'):
                        break
                    if user_input.strip():
                        await self._process_chat_message(cli, client, user_input.strip(), conv_history)
                except (KeyboardInterrupt, EOFError):
                    break

            await client.close()
            return CommandResult.ok()
        except Exception as e:
            return CommandResult.fail(str(e))

    async def _process_chat_message(self, cli, client, message: str, history: list):
        """Process a single chat message and update history."""
        # Add user message to history
        history.append({"role": "user", "content": message})

        start = datetime.now()
        response = await client.chat(message, history=history[:-1])  # Send history without current message
        elapsed = (datetime.now() - start).total_seconds()

        if response.success:
            content = str(response.content or "No response")
            # Add assistant response to history
            history.append({"role": "assistant", "content": content})

            # Display response with nice formatting
            print()
            print("\033[94mjotty>\033[0m ", end="")
            cli.renderer.markdown(content)
            print(f"\033[90m({elapsed:.1f}s)\033[0m\n")
        else:
            # Show friendly message, not raw error
            content = response.content or "I couldn't process that request. Please try again."
            print()
            print("\033[94mjotty>\033[0m ", end="")
            print(f"\033[93m{content}\033[0m")
            print()

    async def _test_workflow(self, cli, goal: str) -> CommandResult:
        """Test workflow mode."""
        if not goal:
            cli.renderer.warning("Please provide a goal: /sdk workflow <goal>")
            return CommandResult.fail("No goal provided")

        cli.renderer.print("\n[bold magenta]═══ SDK Workflow Test ═══[/bold magenta]")
        cli.renderer.print(f"[dim]Goal: {goal}[/dim]\n")

        try:
            from ...sdk.client import Jotty
            from ...core.foundation.types.sdk_types import SDKEventType

            client = Jotty().use_local()

            # Register event listeners
            event_callback = self._create_event_callback(cli)
            for event_type in SDKEventType:
                client.on(event_type, event_callback)

            # Execute workflow
            start = datetime.now()
            response = await client.workflow(goal)
            elapsed = (datetime.now() - start).total_seconds()

            # Display result
            cli.renderer.newline()
            cli.renderer.print("[dim]" + "─" * 60 + "[/dim]")

            if response.success:
                cli.renderer.success(f"Workflow completed in {elapsed:.2f}s")
                cli.renderer.print(f"[cyan]Steps: {response.steps_executed}[/cyan]")
                if response.skills_used:
                    cli.renderer.print(f"[cyan]Skills: {', '.join(response.skills_used)}[/cyan]")
                cli.renderer.newline()
                cli.renderer.markdown(str(response.content or "No output"))
            else:
                cli.renderer.error(f"Workflow failed: {response.error}")

            cli.renderer.print("[dim]" + "─" * 60 + "[/dim]")

            await client.close()
            return CommandResult.ok(data=response)

        except Exception as e:
            cli.renderer.error(f"SDK workflow error: {e}")
            if cli.config.debug:
                import traceback
                traceback.print_exc()
            return CommandResult.fail(str(e))

    async def _test_stream(self, cli, message: str) -> CommandResult:
        """Test streaming mode."""
        if not message:
            cli.renderer.warning("Please provide a message: /sdk stream <message>")
            return CommandResult.fail("No message provided")

        cli.renderer.print("\n[bold yellow]═══ SDK Stream Test ═══[/bold yellow]")
        cli.renderer.print(f"[dim]Message: {message}[/dim]\n")

        try:
            from ...sdk.client import Jotty
            from ...core.foundation.types.sdk_types import SDKEventType

            client = Jotty().use_local()

            cli.renderer.print("[dim]Streaming...[/dim]\n")

            import sys
            async for event in client.stream(message):
                event_type = event.type.value if hasattr(event.type, 'value') else str(event.type)
                icon = self._event_icons.get(event_type, "→")

                if event_type == "stream" and event.data:
                    # Stream content directly
                    sys.stdout.write(str(event.data))
                    sys.stdout.flush()
                elif event_type == "complete":
                    cli.renderer.newline()
                    # Show result content if available
                    if event.data and isinstance(event.data, dict):
                        content = event.data.get('content')
                        if content:
                            cli.renderer.print(f"\n[bold green]Result:[/bold green]\n{content}")
                    cli.renderer.success("Stream complete")
                elif event_type == "error":
                    # Show friendly error, hide tracebacks
                    error_msg = "Something went wrong"
                    if event.data and isinstance(event.data, dict):
                        raw_error = event.data.get('error', '')
                        if raw_error and 'Traceback' not in raw_error and 'File "/' not in raw_error:
                            error_msg = raw_error[:100]  # Limit length
                    cli.renderer.print(f"  [❌] {error_msg}")
                elif event_type == "thinking":
                    # Show engaging status messages, hide technical ones
                    if event.data and isinstance(event.data, dict):
                        status = event.data.get('status', '')
                        # Skip technical statuses
                        if status in ('Decision', 'Generated', 'Preparing') or '=' in status:
                            continue
                        if 'search' in status.lower() or status == 'Searching':
                            cli.renderer.print(f"  [🔍] Searching the web...")
                        elif status == 'Analyzing':
                            cli.renderer.print(f"  [🧠] Analyzing...")
                        elif status == 'Generating':
                            cli.renderer.print(f"  [✍️] Crafting response...")
                        elif status == 'Processing':
                            cli.renderer.print(f"  [💭] Processing...")
                        elif status == 'Retrying':
                            cli.renderer.print(f"  [🔄] Refining response...")
                        elif status == 'Thinking':
                            cli.renderer.print(f"  [💭] Thinking...")
                elif event_type in ("skill_start", "skill_complete"):
                    # Show skill events if enabled
                    if self._show_events and event.data and isinstance(event.data, dict):
                        skill = event.data.get('skill', 'unknown')
                        cli.renderer.print(f"  [{icon}] Using: {skill}")
                elif self._show_events:
                    cli.renderer.print(f"  [{icon}] {event_type.upper()}")

            await client.close()
            return CommandResult.ok()

        except Exception as e:
            cli.renderer.error(f"SDK stream error: {e}")
            if cli.config.debug:
                import traceback
                traceback.print_exc()
            return CommandResult.fail(str(e))

    async def _test_skill(self, cli, args: str) -> CommandResult:
        """Test direct skill execution."""
        parts = args.split(maxsplit=1)
        if len(parts) < 1:
            cli.renderer.warning("Usage: /sdk skill <skill-name> [params]")
            return CommandResult.fail("No skill name provided")

        skill_name = parts[0]
        params_str = parts[1] if len(parts) > 1 else ""

        cli.renderer.print(f"\n[bold green]═══ SDK Skill Test: {skill_name} ═══[/bold green]")

        try:
            from ...sdk.client import Jotty
            from ...core.foundation.types.sdk_types import SDKEventType

            client = Jotty().use_local()

            # Register event listeners
            event_callback = self._create_event_callback(cli)
            for event_type in SDKEventType:
                client.on(event_type, event_callback)

            # Parse params
            import json
            try:
                params = json.loads(params_str) if params_str.startswith("{") else {"input": params_str}
            except:
                params = {"input": params_str}

            # Execute skill
            start = datetime.now()
            skill = client.skill(skill_name)
            response = await skill.run(**params)
            elapsed = (datetime.now() - start).total_seconds()

            # Display result
            cli.renderer.newline()
            if response.success:
                cli.renderer.success(f"Skill completed in {elapsed:.2f}s")
                cli.renderer.print(f"\n[dim]Result:[/dim]")
                cli.renderer.markdown(str(response.content))
            else:
                cli.renderer.error(f"Skill failed: {response.error}")

            await client.close()
            return CommandResult.ok(data=response)

        except Exception as e:
            cli.renderer.error(f"SDK skill error: {e}")
            if cli.config.debug:
                import traceback
                traceback.print_exc()
            return CommandResult.fail(str(e))

    async def _test_health(self, cli) -> CommandResult:
        """Test SDK health."""
        cli.renderer.print("\n[bold]SDK Health Check[/bold]")

        try:
            from ...sdk.client import Jotty

            client = Jotty().use_local()
            health = await client.health()

            cli.renderer.success("SDK is healthy")
            cli.renderer.print(f"  Status: {health.get('status', 'unknown')}")
            cli.renderer.print(f"  Mode: {health.get('mode', 'unknown')}")

            await client.close()
            return CommandResult.ok(data=health)

        except Exception as e:
            cli.renderer.error(f"Health check failed: {e}")
            return CommandResult.fail(str(e))

    async def _list_skills(self, cli) -> CommandResult:
        """List available skills."""
        cli.renderer.print("\n[bold]Available Skills[/bold]")

        try:
            from ...sdk.client import Jotty

            client = Jotty().use_local()
            skills = await client.list_skills()

            cli.renderer.print(f"\n[cyan]Found {len(skills)} skills:[/cyan]\n")

            # Display in columns
            for i, skill in enumerate(sorted(skills)):
                cli.renderer.print(f"  • {skill}")

            await client.close()
            return CommandResult.ok(data=skills)

        except Exception as e:
            cli.renderer.error(f"Failed to list skills: {e}")
            return CommandResult.fail(str(e))

    def get_completions(self, partial: str) -> list:
        """Get completions for SDK command."""
        modes = ["chat", "workflow", "stream", "skill", "health", "skills", "events"]
        return [m for m in modes if m.startswith(partial)]
