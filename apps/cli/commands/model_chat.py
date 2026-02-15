"""
Model Chat Command
==================

Chat with your ML models using natural language.
Query performance, compare models, get improvement suggestions.

Usage:
    /model-chat What's my best model for RELIANCE?
    /model-chat Compare HDFCBANK vs ICICIBANK vs SBIN
    /model-chat --interactive
"""

from typing import TYPE_CHECKING, Any, List

from .base import BaseCommand, CommandResult, ParsedArgs

if TYPE_CHECKING:
    from ..app import JottyCLI


class ModelChatCommand(BaseCommand):
    """Chat with your ML models using natural language."""

    name = "model-chat"
    aliases = ["mc", "mlchat", "talk-to-model"]
    description = "Chat with ML models - query performance, compare, get suggestions"
    usage = "/model-chat <query>"
    category = "ml"

    EXAMPLE_QUERIES = [
        "What's the best model for RELIANCE?",
        "Compare HDFCBANK vs ICICIBANK vs SBIN",
        "Show top features for banking stocks",
        "Why is my TCS model underperforming?",
        "Suggest improvements for INFY",
        "List all runs from today",
    ]

    async def execute(self, args: ParsedArgs, cli: "JottyCLI") -> CommandResult:
        """Execute model chat command."""
        from Jotty.core.modes.agent.model_chat_agent import ModelChatAgent

        # Show help if no query and not interactive
        if not args.positional and "interactive" not in args.flags and "i" not in args.flags:
            return self._show_help(cli)

        # Initialize agent
        llm_model = args.flags.get("model", args.flags.get("m", "sonnet"))
        agent = ModelChatAgent(llm_model=llm_model)

        # Interactive mode
        if "interactive" in args.flags or "i" in args.flags:
            return await self._interactive_mode(agent, cli)

        # Single query mode
        query = " ".join(args.positional)
        cli.renderer.info(f"Query: {query}")
        cli.renderer.info("")

        result = await agent.chat(query)

        cli.renderer.info(result["response"])

        return CommandResult.ok(data=result)

    async def _interactive_mode(self, agent: Any, cli: Any) -> CommandResult:
        """Interactive chat session."""
        cli.renderer.header("Model Chat")
        cli.renderer.info("Chat with your ML models (type 'exit' to quit)")
        cli.renderer.info("Example: 'What's my best model for RELIANCE?'")
        cli.renderer.info("")

        while True:
            try:
                # Get user input
                query = input("You: ").strip()

                if not query:
                    continue

                if query.lower() in ["exit", "quit", "q", "/exit", "/quit"]:
                    cli.renderer.info("Goodbye!")
                    break

                if query.lower() in ["help", "/help", "?"]:
                    self._show_inline_help(cli)
                    continue

                if query.lower() in ["clear", "/clear"]:
                    agent.clear_history()
                    cli.renderer.info("Conversation history cleared.")
                    continue

                # Process query
                result = await agent.chat(query)

                cli.renderer.info("")
                cli.renderer.info(f"Assistant: {result['response']}")
                cli.renderer.info("")

            except KeyboardInterrupt:
                cli.renderer.info("\nGoodbye!")
                break
            except EOFError:
                break
            except Exception as e:
                cli.renderer.error(f"Error: {e}")

        return CommandResult.ok()

    def _show_inline_help(self, cli: Any) -> None:
        """Show help within interactive mode."""
        cli.renderer.info("")
        cli.renderer.info("Commands:")
        cli.renderer.info("  exit, quit, q  - Exit chat")
        cli.renderer.info("  clear          - Clear conversation history")
        cli.renderer.info("  help, ?        - Show this help")
        cli.renderer.info("")
        cli.renderer.info("Example queries:")
        for q in self.EXAMPLE_QUERIES[:4]:
            cli.renderer.info(f"  - {q}")
        cli.renderer.info("")

    def _show_help(self, cli: Any) -> CommandResult:
        """Show command help."""
        cli.renderer.header("Model Chat - Talk to Your ML Models")
        cli.renderer.info("")
        cli.renderer.info("Usage:")
        cli.renderer.info("  /model-chat <query>           Single query")
        cli.renderer.info("  /model-chat --interactive     Interactive chat session")
        cli.renderer.info("  /model-chat -i                Short form for interactive")
        cli.renderer.info("")
        cli.renderer.info("Options:")
        cli.renderer.info("  --model, -m <model>  LLM model (sonnet, opus, haiku)")
        cli.renderer.info("")
        cli.renderer.info("Example queries:")
        for q in self.EXAMPLE_QUERIES:
            cli.renderer.info(f"  - {q}")
        cli.renderer.info("")
        cli.renderer.info("The agent can:")
        cli.renderer.info("  - Query best models by any metric (AUC, Sharpe, ROMAD)")
        cli.renderer.info("  - Compare models across different stocks")
        cli.renderer.info("  - Analyze feature importance patterns")
        cli.renderer.info("  - Suggest improvements using LLM analysis")
        cli.renderer.info("  - List and search experiment runs")
        cli.renderer.info("")
        cli.renderer.info("Prerequisites:")
        cli.renderer.info("  Run /stock-ml with --mlflow flag to log experiments first:")
        cli.renderer.info("  /stock-ml RELIANCE --mlflow --backtest")

        return CommandResult.ok()

    def get_completions(self, partial: str) -> List[str]:
        """Get autocomplete suggestions."""
        completions = []

        # Flag completions
        if partial.startswith("--"):
            flags = ["--interactive", "--model"]
            completions.extend([f for f in flags if f.startswith(partial)])
        elif partial.startswith("-"):
            flags = ["-i", "-m"]
            completions.extend([f for f in flags if f.startswith(partial)])

        # Stock symbol completions
        common_stocks = [
            "RELIANCE",
            "TCS",
            "HDFCBANK",
            "INFY",
            "ICICIBANK",
            "SBIN",
            "KOTAKBANK",
            "AXISBANK",
            "TATAMOTORS",
            "WIPRO",
        ]
        if partial.isupper() or (len(partial) > 0 and partial[0].isupper()):
            completions.extend([s for s in common_stocks if s.startswith(partial.upper())])

        return completions
