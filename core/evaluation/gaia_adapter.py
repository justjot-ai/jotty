"""
Jotty GAIA Adapter — sync→async bridge for GAIA benchmark evaluation.

Wraps the async Jotty.run() behind the sync agent.run(question) interface
that GAIABenchmark.evaluate_task() expects.

Uses a single long-lived event loop in a dedicated thread so that multiple
sequential tasks do not hit "Event loop is closed" (asyncio.run() per task
would create/close a loop each time and leave Jotty internals with stale refs).
"""

import asyncio
import logging
import os
import threading
from typing import Callable, Optional

# Ensure ANTHROPIC_API_KEY is in the environment for DSPy/litellm.
# The native Anthropic SDK picks it up from .env files, but litellm doesn't.
if not os.environ.get("ANTHROPIC_API_KEY"):
    for env_file in (".env.anthropic", ".env"):
        try:
            from pathlib import Path
            env_path = Path(env_file)
            if not env_path.exists():
                # Try project root
                env_path = Path(__file__).resolve().parent.parent.parent.parent / env_file
            if env_path.exists():
                for line in env_path.read_text().splitlines():
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, _, val = line.partition('=')
                        if key.strip() == "ANTHROPIC_API_KEY" and val.strip():
                            os.environ["ANTHROPIC_API_KEY"] = val.strip()
                            break
                if os.environ.get("ANTHROPIC_API_KEY"):
                    break
        except Exception:
            pass

logger = logging.getLogger(__name__)


def _extract_answer_from_output(output) -> str:
    """
    Extract the answer string from ExecutionResult.output for GAIA scoring.

    Tier 4/5 can set result.output to AgenticExecutionResult (has .final_output, .outputs)
    or other nested structures. GAIA needs a single string to compare to expected.
    """
    if output is None:
        return ""
    if isinstance(output, str):
        # Handle string representations of dicts (e.g. from skill results)
        stripped = output.strip()
        if stripped.startswith('{') and stripped.endswith('}'):
            try:
                import ast
                parsed = ast.literal_eval(stripped)
                if isinstance(parsed, dict):
                    return _extract_answer_from_output(parsed)
            except (ValueError, SyntaxError):
                pass
        return output
    # AgenticExecutionResult: use final_output, else last step content from outputs
    if hasattr(output, "final_output") and output.final_output is not None:
        return _extract_answer_from_output(output.final_output)
    if hasattr(output, "outputs") and isinstance(output.outputs, dict):
        for key in reversed(list(output.outputs.keys())):
            val = output.outputs[key]
            if isinstance(val, dict):
                for field in ("content", "response", "text", "output", "result"):
                    if field in val and val[field]:
                        return str(val[field]).strip()
            elif isinstance(val, str) and len(val) > 10:
                return val.strip()
    # Nested EpisodeResult or similar
    if hasattr(output, "output") and output.output is not None:
        return _extract_answer_from_output(output.output)
    # Plain dict
    if isinstance(output, dict):
        for field in ("content", "response", "text", "output", "result"):
            if field in output and output[field]:
                return str(output[field]).strip()
        # Skill-style dict: {'success': True, 'results': [str or dict, ...]}
        if "results" in output and isinstance(output["results"], list):
            # Collect all text fragments from results and return the longest
            candidates = []
            for item in output["results"]:
                if isinstance(item, str) and len(item.strip()) > 2:
                    candidates.append(item.strip())
                elif isinstance(item, dict):
                    for f in ("content", "response", "text", "output", "result", "summary", "answer", "message"):
                        if f in item and item[f]:
                            candidates.append(str(item[f]).strip())
                            break
            if candidates:
                # Return longest candidate (most likely the actual answer)
                return max(candidates, key=len)
        # Last resort for dicts: 'answer' or 'message' fields
        for field in ("answer", "message", "summary"):
            if field in output and output[field]:
                return str(output[field]).strip()
    # Avoid returning object repr (e.g. "AgenticExecutionResult(success=...")
    if hasattr(output, "summary"):
        summary = output.summary() if callable(output.summary) else output.summary
        return (summary or "").strip()
    return str(output).strip()


# GAIA-optimized system prompt: forces concise answers for exact-match scoring.
GAIA_SYSTEM_PROMPT = (
    "You are answering a benchmark question. "
    "Answer concisely. Give ONLY the final answer with no explanation. "
    "Do not include units unless the question explicitly asks for them. "
    "If the answer is a number, output only that number. "
    "Use available tools (web search, voice_to_text, read_file, etc.) when the task requires it; do not refuse for lack of direct access. "
    "If the question refers to a video, file, or external resource, use the provided tools to access it and base your answer on that. "
    "You MUST use the appropriate tool to process any attached file(s) before answering: "
    "use voice_to_text_tool for audio files (.mp3, .wav, .m4a), read_file for text/data files."
)

# Audio file extensions for tool routing
AUDIO_EXTENSIONS = {'.mp3', '.wav', '.m4a', '.mp4', '.mpeg', '.mpga', '.webm'}

# Patterns that suggest the model refused instead of answering (for logging / retry).
REFUSAL_PATTERNS = (
    "cannot ",
    "can't ",
    "unable to",
    "do not have access",
    "don't have access",
    "no direct access",
    "cannot confidently",
    "without verifying",
    "i apologize, but",
    "i cannot actually",
    "no audio",
    "no file",
    "has not been successfully uploaded",
)


def _looks_like_refusal(text: str) -> bool:
    """True if the model output looks like a refusal rather than an answer."""
    if not text or len(text.strip()) < 20:
        return False
    lower = text.strip().lower()
    return any(p in lower for p in REFUSAL_PATTERNS)


def _required_skills_for_gaia(question: str, attachment_paths: list) -> list:
    """Return explicit skill names for GAIA task based on content analysis."""
    skills = ["web-search"]  # Always useful for GAIA
    q_lower = question.lower()

    for path in (attachment_paths or []):
        ext = ('.' + path.rsplit('.', 1)[-1].lower()) if '.' in path else ''
        if ext in AUDIO_EXTENSIONS:
            skills.extend(["voice", "openai-whisper-api"])
        elif ext in {'.xlsx', '.xls'}:
            skills.append("xlsx-tools")
        elif ext == '.pdf':
            skills.append("pdf-tools")
        else:
            skills.append("file-operations")

    if any(kw in q_lower for kw in ['calculat', 'how many', 'sum ', 'average', 'comput']):
        skills.append("calculator")

    return skills


class JottyGAIAAdapter:
    """
    Sync adapter wrapping Jotty for GAIA evaluation.

    GAIABenchmark.evaluate_task() calls adapter.run(question) synchronously.
    This adapter translates that into an async Jotty.run() call.

    After each call, `last_result` holds the full ExecutionResult so the
    runner can extract cost/tokens/latency.

    Usage:
        adapter = JottyGAIAAdapter(tier="DIRECT")
        result = benchmark.evaluate_task(task, adapter)
        cost = adapter.last_result.cost_usd
    """

    def __init__(
        self,
        tier: Optional[str] = None,
        model: Optional[str] = None,
        dry_run: bool = False,
        use_llm_doc_sources: bool = False,
        progress_callback: Optional[Callable[[str, str], None]] = None,
    ):
        """
        Args:
            tier: Execution tier name (DIRECT, AGENTIC, etc.). None = auto-detect.
            model: Model override (e.g. 'claude-sonnet-4-20250514').
            dry_run: If True, skip Jotty entirely and return placeholder.
            use_llm_doc_sources: If True, append open-source LLM doc references (Microsoft, Hugging Face, etc.) to context.
            progress_callback: Optional (stage, detail) callback for real-time progress; can also be set on .progress_callback before each run.
        """
        self.tier = tier
        self.model = model
        self.dry_run = dry_run
        self.use_llm_doc_sources = use_llm_doc_sources
        self.progress_callback = progress_callback  # (stage, detail) for live progress
        self.last_result = None  # ExecutionResult from last run
        self.last_raw_answer = None  # Raw text before DSPy normalization (for failure logging)
        self._jotty = None  # Lazy-initialized
        self._loop = None
        self._loop_thread = None
        self._loop_ready = threading.Event()

    def _get_jotty(self):
        """Lazy-initialize the Jotty instance (call from main thread only for dry_run; else from loop thread)."""
        if self._jotty is None:
            import os
            from pathlib import Path

            # Load Jotty/.env so ANTHROPIC_API_KEY etc. are available
            # (may not be in the environment when run from CI or Claude Code)
            env_path = Path(__file__).resolve().parent.parent.parent / '.env'
            if env_path.exists() and not os.environ.get('ANTHROPIC_API_KEY'):
                for line in env_path.read_text().splitlines():
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, _, value = line.partition('=')
                        os.environ.setdefault(key.strip(), value.strip())

            # Configure DSPy to use Anthropic API directly (not Claude CLI)
            # to avoid "cannot be launched inside another Claude Code session" errors
            try:
                import dspy
                api_lm = dspy.LM("anthropic/claude-sonnet-4-20250514")
                dspy.configure(lm=api_lm)
            except Exception as e:
                logger.debug(f"DSPy API LM configuration: {e}")

            from Jotty.jotty import Jotty
            self._jotty = Jotty()
        return self._jotty

    def shutdown(self) -> None:
        """Stop the dedicated event-loop thread. Safe to call multiple times."""
        if self._loop is None:
            return
        try:
            self._loop.call_soon_threadsafe(self._loop.stop)
            if self._loop_thread and self._loop_thread.is_alive():
                self._loop_thread.join(timeout=5.0)
        except Exception as e:
            logger.debug("GAIA adapter shutdown: %s", e)
        finally:
            self._loop = None
            self._loop_thread = None

    def _build_prompt(self, question: str, attachment_paths: Optional[list] = None) -> str:
        """Wrap question with GAIA-optimized prompt; include attachment paths and optional LLM doc sources."""
        parts = [GAIA_SYSTEM_PROMPT]
        if self.use_llm_doc_sources:
            try:
                from Jotty.core.evaluation.llm_doc_sources import list_sources, to_context_snippet
                sources = list_sources()
                snippet = to_context_snippet(sources, max_items=6)
                if snippet:
                    parts.append("")
                    parts.append(snippet)
            except Exception:
                pass
        parts.append("")
        # Hint so planner injects web-search / voice / read_file when using AGENTIC (keyword-based injection)
        parts.append(
            "Use web_search, voice_to_text_tool, read_file, or calculator when needed to answer. "
            "For factual questions, ALWAYS search the web first — do not guess."
        )
        parts.append("")
        parts.append(f"Question: {question}")
        if attachment_paths:
            parts.append("")
            for path in attachment_paths:
                ext = ('.' + path.rsplit('.', 1)[-1].lower()) if '.' in path else ''
                if ext in AUDIO_EXTENSIONS:
                    parts.append(
                        f"Attached audio file: {path}\n"
                        f"IMPORTANT: Use voice_to_text_tool with audio_path='{path}' to transcribe this audio file. "
                        f"Do NOT use read_file for audio. Read the full transcript before answering."
                    )
                elif ext in {'.xlsx', '.xls', '.csv'}:
                    parts.append(f"Attached spreadsheet: {path}\nUse read_file to read this file.")
                else:
                    parts.append(f"Attached file: {path}\nUse read_file to read this file.")
        return "\n".join(parts)

    def run(self, question: str, attachment_paths: Optional[list] = None, **kwargs) -> str:
        """
        Synchronous entry point for GAIABenchmark.evaluate_task().

        Args:
            question: The GAIA question text.
            attachment_paths: Optional list of absolute paths to attached files (Excel, PDF, etc.).
            expected_answer: Optional expected answer string; when provided, DSPy is used to
                extract/normalize the final answer from raw output (signature-driven, not regex).

        Returns the agent's answer as a string.
        """
        expected_answer = kwargs.pop("expected_answer", None)
        force_tier = kwargs.pop("force_tier", None)

        if self.dry_run:
            self.last_result = None
            return "[DRY RUN]"

        prompt = self._build_prompt(question, attachment_paths=attachment_paths)

        # Build kwargs for Jotty.run()
        run_kwargs = {}
        tier_to_use = force_tier or self.tier
        if tier_to_use:
            from Jotty.core.execution.types import ExecutionTier
            tier_map = {t.name: t for t in ExecutionTier}
            tier_enum = tier_map.get(str(tier_to_use).upper())
            if tier_enum:
                run_kwargs['tier'] = tier_enum

        if self.model:
            from Jotty.core.execution.types import ExecutionConfig
            run_kwargs['config'] = ExecutionConfig(model=self.model)

        if self.progress_callback:
            run_kwargs['status_callback'] = self.progress_callback

        # Bypass ComplexityGate so GAIA tasks always get full planning + tools
        run_kwargs['skip_complexity_gate'] = True

        # Inject explicit skills so the planner has voice/whisper/search available
        run_kwargs['hint_skills'] = _required_skills_for_gaia(question, attachment_paths)

        # Skip swarm keyword selection — GAIA system prompt contains words like
        # "report", "data", "content" that falsely match ReviewSwarm, DataAnalysisSwarm, etc.
        # This forces tier4/tier5 to go directly to the Orchestrator.
        run_kwargs['skip_swarm_selection'] = True

        # Forward any remaining kwargs (e.g. skip_validation=False from retry logic)
        run_kwargs.update(kwargs)

        result = self._run_async(prompt, **run_kwargs)
        self.last_result = result

        # Extract answer string (result.output can be AgenticExecutionResult, dict, or str)
        raw = result.output if result else None
        raw_text = _extract_answer_from_output(raw)
        self.last_raw_answer = raw_text  # For failure logging in runner

        # DSPy answer normalization: disabled — extraction + benchmark matching
        # is more reliable than an extra LLM call that can fail/timeout/empty.
        # The _extract_answer_from_output() + GAIABenchmark.validate_answer()
        # combo already handles verbose outputs, numeric comparison, lists, etc.
        return raw_text

    def _ensure_loop_thread(self):
        """Start the dedicated event-loop thread if not already running."""
        if self._loop is not None and self._loop_thread is not None and self._loop_thread.is_alive():
            return

        def _run_loop():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop_ready.set()
            self._loop.run_forever()

        self._loop_ready.clear()
        self._loop_thread = threading.Thread(target=_run_loop, daemon=True)
        self._loop_thread.start()
        self._loop_ready.wait(timeout=10.0)
        if self._loop is None:
            raise RuntimeError("GAIA adapter: event loop thread failed to start")

    def _run_async(self, prompt: str, **kwargs):
        """Run the async Jotty.run() from sync context using a single long-lived loop."""
        async def _exec():
            # Lazy-init Jotty in the loop thread so it uses this loop (avoids "Event loop is closed").
            jotty = self._get_jotty()
            return await jotty.run(prompt, **kwargs)

        # If we're already inside an event loop (e.g. Jupyter), run in a thread to avoid nested loop.
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None

        if running is not None:
            # Nested: run in a thread with a fresh loop (one-off).
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(self._run_async_standalone, prompt, **kwargs)
                return future.result()

        # Main thread, no existing loop: use a single long-lived loop so multiple
        # tasks don't see "Event loop is closed" from Jotty internals.
        self._ensure_loop_thread()
        future = asyncio.run_coroutine_threadsafe(_exec(), self._loop)
        return future.result(timeout=300)  # 5 min per task

    def _run_async_standalone(self, prompt: str, **kwargs):
        """Run in a fresh loop (used when already inside another loop)."""
        jotty = self._get_jotty()

        async def _exec():
            return await jotty.run(prompt, **kwargs)

        return asyncio.run(_exec())
