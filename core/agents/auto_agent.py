"""
AutoAgent - Autonomous task execution with fully agentic planning.

Uses TaskPlanner for all planning decisions (no hardcoded logic).
Takes any open-ended task, discovers relevant skills, plans execution,
and runs the workflow automatically.

Refactored to inherit from AutonomousAgent for unified infrastructure.
"""
import logging
import inspect
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime

from .agentic_planner import TaskPlanner
from .base import AutonomousAgent, AutonomousAgentConfig
from Jotty.core.foundation.exceptions import (
    AgentExecutionError,
    ToolExecutionError,
    LLMError,
    DSPyError,
)
from Jotty.core.utils.async_utils import safe_status, StatusReporter, AgentEventBroadcaster, AgentEvent
from ._execution_types import TaskType, ExecutionResult, _clean_for_display

# Mode-specific system prompts for tools with different backends
_MODE_PROMPTS = {
    'browser-automation:playwright': (
        "You have Playwright browser automation. Use async page actions: "
        "goto(), click(), fill(), screenshot(). Pages render JS fully. "
        "Always wait_for_selector() before interacting with dynamic content."
    ),
    'browser-automation:selenium': (
        "You have Selenium browser automation with CDP support. "
        "Use WebDriverWait for dynamic elements. For Electron apps, "
        "connect via cdp_url parameter. Screenshots are sync."
    ),
    'terminal-session': (
        "You have persistent terminal sessions via pexpect. "
        "Working directory and env vars persist across commands. "
        "Use expect patterns for interactive prompts (sudo, ssh)."
    ),
}

logger = logging.getLogger(__name__)


class AutoAgent(AutonomousAgent):
    """
    Autonomous agent that discovers and executes skills for any task.

    Inherits from AutonomousAgent for:
    - Skill discovery and selection
    - Execution planning via TaskPlanner
    - Multi-step execution with dependency resolution
    - Adaptive replanning on failures
    - Memory and context integration
    - Retry logic and error handling

    Adds AutoAgent-specific features:
    - Ensemble prompting for multi-perspective analysis
    - Output skill integration (messaging)
    - Backwards-compatible execute() returning ExecutionResult

    Usage:
        agent = AutoAgent()
        result = await agent.execute("RNN vs CNN")
    """

    def __init__(
        self,
        default_output_skill: Optional[str] = None,
        enable_output: bool = False,
        max_steps: int = 10,
        timeout: int = 300,
        planner: Optional[TaskPlanner] = None,
        skill_filter: Optional[str] = None,
        system_prompt: Optional[str] = None,
        name: Optional[str] = None,
    ):
        """
        Initialize AutoAgent.

        Args:
            default_output_skill: Optional skill for final output (e.g., messaging skill)
            enable_output: Whether to send output via messaging (requires default_output_skill)
            skill_filter: Optional category filter for skill discovery
            max_steps: Maximum execution steps
            timeout: Default timeout for operations
            planner: Optional TaskPlanner instance (creates new if None)
            system_prompt: Optional system prompt to customize agent behavior
                (essential for multi-agent scenarios where agents need different roles)
            name: Optional agent name (default: "AutoAgent")
        """
        # Create config for base class
        config = AutonomousAgentConfig(
            name=name or "AutoAgent",
            max_steps=max_steps,
            timeout=float(timeout),
            enable_replanning=True,  # Retry with reflective replanning on failure
            max_replans=3,
            skill_filter=skill_filter,
            default_output_skill=default_output_skill,
            enable_output=enable_output and default_output_skill is not None,
            system_prompt=system_prompt or "",
        )
        super().__init__(config)

        # AutoAgent-specific state
        self.default_output_skill = default_output_skill
        self.enable_output = enable_output and default_output_skill is not None

        # Override planner if provided
        if planner is not None:
            self._planner = planner

        if skill_filter:
            logger.info(f"AutoAgent initialized with skill filter: {skill_filter}")

    # =========================================================================
    # ENSEMBLE PROMPTING (AutoAgent-specific)
    # =========================================================================

    async def _execute_ensemble(
        self,
        task: str,
        strategy: str = 'multi_perspective',
        status_fn: Optional[Callable] = None,
        max_perspectives: int = 4,
    ) -> Dict[str, Any]:
        """
        Execute prompt ensembling for multi-perspective analysis.

        DRY: Uses the ensemble_prompt_tool skill (which already handles
        parallel execution and adaptive sizing). Only falls back to
        inline DSPy if the skill is unavailable.
        """
        _status = StatusReporter(status_fn)

        try:
            # Use the ensemble skill (preferred — parallel + adaptive)
            if self.skills_registry:
                skill = self.skills_registry.get_skill('claude-cli-llm')
                if skill:
                    ensemble_tool = skill.tools.get('ensemble_prompt_tool')
                    if ensemble_tool:
                        _status("Ensemble", f"{strategy} ({max_perspectives} perspectives)")
                        result = ensemble_tool({
                            'prompt': task,
                            'strategy': strategy,
                            'synthesis_style': 'structured',
                            'max_perspectives': max_perspectives,
                        })
                        return result

            # Fallback: Use DSPy directly (parallel via ThreadPoolExecutor)
            import dspy
            if not hasattr(dspy.settings, 'lm') or dspy.settings.lm is None:
                return {'success': False, 'error': 'No LLM configured'}

            lm = dspy.settings.lm

            all_perspectives = [
                ("analytical", "Analyze this from a data-driven, logical perspective:"),
                ("creative", "Consider unconventional angles and innovative solutions:"),
                ("critical", "Play devil's advocate - identify risks and problems:"),
                ("practical", "Focus on feasibility and actionable steps:"),
            ]
            perspectives = all_perspectives[:max_perspectives]

            # Optima-inspired: parallel perspective generation
            from concurrent.futures import ThreadPoolExecutor, as_completed
            responses = {}

            def _gen(name, prefix):
                prompt = f"{prefix}\n\n{task}"
                response = lm(prompt=prompt)
                return name, response[0] if isinstance(response, list) else str(response)

            with ThreadPoolExecutor(max_workers=min(len(perspectives), 4)) as executor:
                futures = {executor.submit(_gen, n, p): n for n, p in perspectives}
                for future in as_completed(futures):
                    try:
                        name, text = future.result()
                        responses[name] = text
                        _status(f"  {name}", "done")
                    except Exception as e:
                        logger.warning(f"Perspective '{futures[future]}' failed: {e}")

            if not responses:
                return {'success': False, 'error': 'All perspectives failed'}

            _status("Synthesizing", f"{len(responses)} perspectives")
            synthesis_prompt = f"""Synthesize these {len(responses)} expert perspectives:

Question: {task}

{chr(10).join(f'**{k.upper()}:** {v[:500]}' for k, v in responses.items())}

Provide:
1. **Consensus**: Where perspectives agree
2. **Tensions**: Where they diverge
3. **Recommendation**: Balanced conclusion"""

            synthesis = lm(prompt=synthesis_prompt)
            final_response = synthesis[0] if isinstance(synthesis, list) else str(synthesis)

            return {
                'success': True,
                'response': final_response,
                'perspectives_used': list(responses.keys()),
                'individual_responses': responses,
                'strategy': strategy,
                'confidence': len(responses) / len(perspectives)
            }

        except Exception as e:
            logger.error(f"Ensemble execution failed: {e}", exc_info=True)
            return {'success': False, 'error': str(e)}

    def _should_auto_ensemble(self, task: str):
        """
        Determine if ensemble should be auto-enabled and with how many perspectives.

        DRY: Delegates to the single source of truth in swarm_ensemble.py.

        Returns:
            (bool, int) tuple: (should_ensemble, max_perspectives)
        """
        from Jotty.core.orchestration.swarm_ensemble import should_auto_ensemble
        return should_auto_ensemble(task)

    # =========================================================================
    # DSPY REACT STREAMING CALLBACK (Pattern #3)
    # =========================================================================

    @staticmethod
    def _create_dspy_stream_callback(broadcaster: AgentEventBroadcaster, agent_id: str):
        """
        Create a DSPy callback that extracts ReAct intermediate steps
        (thoughts, actions, observations) and broadcasts them as AgentEvents.

        Returns a callback class instance if DSPy callbacks are available,
        otherwise None.
        """
        try:
            from dspy.utils.callback import BaseCallback

            class _ReActStreamCallback(BaseCallback):
                """Hooks into DSPy ReAct to broadcast intermediate reasoning."""

                def on_module_start(self, call_id, instance, inputs):
                    broadcaster.emit(AgentEvent(
                        type="step_start",
                        data={"module": type(instance).__name__, "inputs_keys": list(inputs.keys())},
                        agent_id=agent_id,
                    ))

                def on_module_end(self, call_id, outputs, exception):
                    # Extract ReAct internals from DSPy _store
                    store = getattr(outputs, '_store', {}) if outputs else {}
                    data = {}
                    for key in ('next_thought', 'next_tool_name', 'next_tool_args', 'observation'):
                        if key in store:
                            val = store[key]
                            data[key] = str(val)[:500] if val else None
                    broadcaster.emit(AgentEvent(
                        type="step_end",
                        data=data,
                        agent_id=agent_id,
                    ))

                def on_tool_start(self, call_id, tool_name, tool_input):
                    broadcaster.emit(AgentEvent(
                        type="tool_start",
                        data={"tool": tool_name, "input_preview": str(tool_input)[:200]},
                        agent_id=agent_id,
                    ))

                def on_tool_end(self, call_id, tool_output):
                    broadcaster.emit(AgentEvent(
                        type="tool_end",
                        data={"output_length": len(str(tool_output))},
                        agent_id=agent_id,
                    ))

            return _ReActStreamCallback()
        except (ImportError, AttributeError):
            return None

    # =========================================================================
    # MODE-SPECIFIC SYSTEM PROMPTS (Pattern #8)
    # =========================================================================

    def _get_mode_prompts(self, skill_names: set) -> str:
        """
        Build mode-specific prompt additions based on active skills.

        Different tool backends (Playwright vs Selenium, pexpect vs subprocess)
        have different capabilities and gotchas. Injecting backend-specific
        guidance reduces hallucinated tool calls.
        """
        import os
        parts = []
        for skill_key, prompt in _MODE_PROMPTS.items():
            skill_name = skill_key.split(':')[0]
            if skill_name in skill_names:
                # Check backend variant if applicable
                if ':' in skill_key:
                    variant = skill_key.split(':')[1]
                    backend = os.environ.get('BROWSER_BACKEND', 'playwright')
                    if variant != backend:
                        continue
                parts.append(prompt)
        return "\n\n".join(parts)

    # =========================================================================
    # TASK TYPE INFERENCE (Override for TaskType enum)
    # =========================================================================

    def _infer_task_type(self, task: str) -> str:
        """
        Infer task type using agentic planner.

        Returns string task type for compatibility with base class.
        """
        if self._planner is not None:
            try:
                task_type, reasoning, confidence = self._planner.infer_task_type(task)
                logger.debug(f"Task type inference: {task_type.value} (confidence: {confidence:.2f})")
                return task_type.value
            except Exception as e:
                logger.warning(f"Task type inference failed: {e}")

        # Fallback to base class implementation
        return super()._infer_task_type(task)

    def _infer_task_type_enum(self, task: str) -> TaskType:
        """
        Infer task type and return as TaskType enum.

        Used for ExecutionResult compatibility.
        """
        type_str = self._infer_task_type(task)
        try:
            return TaskType(type_str)
        except ValueError:
            return TaskType.UNKNOWN

    # =========================================================================
    # MAIN EXECUTE (Backwards-compatible signature)
    # =========================================================================

    async def execute(self, task: str, **kwargs) -> ExecutionResult:
        """
        Execute a task automatically.

        Routes through BaseAgent infrastructure for hooks and metrics
        while keeping AutonomousAgent's internal replanning as the retry
        mechanism (BaseAgent retry is NOT used here — autonomous agents
        handle failure via replanning, not blind retry).

        Args:
            task: Task description (can be minimal like "RNN vs CNN")
            status_callback: Optional callback(stage, detail) for progress updates
            ensemble: Enable prompt ensembling for multi-perspective analysis
            ensemble_strategy: Strategy for ensembling

        Returns:
            ExecutionResult with outputs and status
        """
        import time as _time
        start_time = datetime.now()
        wall_start = _time.time()

        # Extract status callback for streaming progress
        status_callback = kwargs.pop('status_callback', None)
        streaming_callback = kwargs.pop('streaming_callback', None)
        ensemble = kwargs.pop('ensemble', None)  # None = auto-detect
        ensemble_strategy = kwargs.pop('ensemble_strategy', 'multi_perspective')
        # Learning context: kept separate from task string to prevent pollution
        learning_context = kwargs.pop('learning_context', None)

        _status = StatusReporter(status_callback, logger, emoji=" ")

        # Register streaming callback as temporary event listener
        _broadcaster = AgentEventBroadcaster.get_instance()
        _streaming_listener = None
        _dspy_callback = None
        if streaming_callback is not None:
            _streaming_listener = streaming_callback
            for etype in ("tool_start", "tool_end", "step_start", "step_end", "status", "error", "streaming"):
                _broadcaster.subscribe(etype, _streaming_listener)

        # Ensure initialized
        self._ensure_initialized()

        # Set up DSPy ReAct streaming callback (Pattern #3)
        if streaming_callback is not None:
            _dspy_callback = self._create_dspy_stream_callback(_broadcaster, self.config.name)
            if _dspy_callback is not None:
                try:
                    import dspy
                    if hasattr(dspy.settings, 'callbacks') and isinstance(dspy.settings.callbacks, list):
                        dspy.settings.callbacks.append(_dspy_callback)
                except Exception:
                    _dspy_callback = None

        # Run pre-execution hooks (from BaseAgent)
        try:
            await self._run_pre_hooks(task=task, **kwargs)
        except Exception as e:
            logger.warning(f"Pre-hook failed: {e}")

        # Optima-inspired deduplication (Chen et al., 2024):
        # If caller already enriched the task with ensemble context (e.g. Orchestrator),
        # skip re-ensembling to avoid 2x LLM calls for the same thing.
        # Also skip for direct_llm sub-agents — they're specialized, no need for ensemble.
        _is_direct_llm = kwargs.get('direct_llm', False)
        already_ensembled = (
            kwargs.get('ensemble_context') is not None
            or _is_direct_llm
        )

        # Auto-detect ensemble for certain task types
        max_perspectives = 4  # default
        if ensemble is None and not already_ensembled:
            ensemble, max_perspectives = self._should_auto_ensemble(task)
            if ensemble:
                _status("Auto-ensemble", f"enabled ({max_perspectives} perspectives)")
        elif already_ensembled:
            ensemble = False
            logger.debug("Ensemble skipped: caller already provided ensemble context")

        # Optional: Ensemble pre-analysis for enriched context
        enriched_task = task
        if ensemble:
            _status("Ensembling", f"strategy={ensemble_strategy}, perspectives={max_perspectives}")
            ensemble_result = await self._execute_ensemble(task, ensemble_strategy, _status, max_perspectives=max_perspectives)
            if ensemble_result.get('success'):
                _status("Ensemble complete", f"{len(ensemble_result.get('perspectives_used', []))} perspectives")

                # Enrich the task with ensemble synthesis
                synthesis = ensemble_result.get('response', '')
                if synthesis:
                    enriched_task = f"""{task}

[Multi-Perspective Analysis - Use these insights to guide your work]:
{synthesis[:3000]}"""
                    _status("Task enriched", "with multi-perspective synthesis")

                kwargs['ensemble_context'] = ensemble_result

        _status("AutoAgent", "starting task execution")

        # Get task type enum for result
        # OPTIMIZATION: skip LLM-based inference for direct_llm mode (sub-agents)
        direct_llm = kwargs.get('direct_llm', False)
        if direct_llm:
            task_type_enum = TaskType.UNKNOWN
        else:
            task_type_enum = self._infer_task_type_enum(task)

        # Execute via AutonomousAgent._execute_impl (handles its own replanning)
        # Pass learning_context separately so it can enrich LLM prompts without polluting task string
        try:
            result = await super()._execute_impl(
                task=enriched_task,
                status_callback=status_callback,
                learning_context=learning_context,
                **kwargs
            )
        except (AgentExecutionError, ToolExecutionError) as e:
            logger.error(f"AutoAgent execution error: {e}")
            result = {
                'success': False,
                'errors': [f"{type(e).__name__}: {e}"],
                'skills_used': [],
                'steps_executed': 0,
                'outputs': {},
                'final_output': None,
            }
        except (LLMError, DSPyError) as e:
            logger.error(f"AutoAgent LLM/DSPy error: {e}")
            result = {
                'success': False,
                'errors': [f"{type(e).__name__}: {e}"],
                'skills_used': [],
                'steps_executed': 0,
                'outputs': {},
                'final_output': None,
            }
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            logger.error(f"AutoAgent unexpected error ({type(e).__name__}): {e}", exc_info=True)
            result = {
                'success': False,
                'errors': [f"{type(e).__name__}: {e}"],
                'skills_used': [],
                'steps_executed': 0,
                'outputs': {},
                'final_output': None,
            }

        # Convert to ExecutionResult
        execution_time = (datetime.now() - start_time).total_seconds()

        # Clean task for display
        display_task = _clean_for_display(task)

        is_success = result.get('success', False)

        exec_result = ExecutionResult(
            success=is_success,
            task=display_task,
            task_type=task_type_enum,
            skills_used=result.get('skills_used', []),
            steps_executed=result.get('steps_executed', 0),
            outputs=result.get('outputs', {}),
            final_output=result.get('final_output'),
            errors=result.get('errors', []),
            execution_time=execution_time,
            stopped_early=result.get('stopped_early', False),
        )

        # Update BaseAgent metrics
        wall_time = _time.time() - wall_start
        self._metrics["total_executions"] += 1
        self._metrics["total_execution_time"] += wall_time
        if is_success:
            self._metrics["successful_executions"] += 1
        else:
            self._metrics["failed_executions"] += 1

        # Run post-execution hooks (from BaseAgent)
        try:
            from .base.base_agent import AgentResult
            agent_result = AgentResult(
                success=is_success,
                output=result,
                agent_name=self.config.name,
                execution_time=wall_time,
            )
            await self._run_post_hooks(agent_result, task=task, **kwargs)
        except Exception as e:
            logger.warning(f"Post-hook failed: {e}")
        finally:
            # Clean up streaming listener
            if _streaming_listener is not None:
                for etype in ("tool_start", "tool_end", "step_start", "step_end", "status", "error", "streaming"):
                    _broadcaster.unsubscribe(etype, _streaming_listener)
            # Clean up DSPy callback
            if _dspy_callback is not None:
                try:
                    import dspy
                    if hasattr(dspy.settings, 'callbacks') and isinstance(dspy.settings.callbacks, list):
                        dspy.settings.callbacks.remove(_dspy_callback)
                except (ImportError, ValueError):
                    pass

        return exec_result

    async def execute_and_send(
        self,
        task: str,
        output_skill: Optional[str] = None,
        chat_id: Optional[str] = None
    ) -> ExecutionResult:
        """
        Execute task and send result to messaging platform.

        Args:
            task: Task description
            output_skill: Override output skill (telegram-sender, slack, discord)
            chat_id: Optional chat ID for messaging
        """
        # Execute task
        result = await self.execute(task)

        if not result.success:
            return result

        # Send output (if output skill configured)
        output_skill = output_skill or self.default_output_skill
        if not output_skill or not self.skills_registry:
            return result

        skill = self.skills_registry.get_skill(output_skill)
        if not skill or not skill.tools:
            return result

        # Find a send/message tool
        send_tool = None
        for tool_name, tool_func in skill.tools.items():
            if any(kw in tool_name.lower() for kw in ['send', 'message', 'post', 'notify']):
                send_tool = tool_func
                break

        if send_tool and result.final_output:
            # Format message
            if isinstance(result.final_output, str):
                message = f"Task: {task}\n\n{result.final_output[:3500]}"
            else:
                message = f"Task: {task}\n\nCompleted with {result.steps_executed} steps"

            params = {'text': message, 'message': message}
            if chat_id:
                params['chat_id'] = chat_id

            try:
                if inspect.iscoroutinefunction(send_tool):
                    await send_tool(params)
                else:
                    send_tool(params)
            except Exception as e:
                logger.warning(f"Failed to send output: {e}")

        return result


# Convenience function
async def run_task(task: str, send_output: bool = False) -> ExecutionResult:
    """
    Run a task with AutoAgent.

    Args:
        task: Task description
        send_output: Whether to send to messaging

    Returns:
        ExecutionResult
    """
    agent = AutoAgent(enable_output=send_output)
    return await agent.execute(task)
