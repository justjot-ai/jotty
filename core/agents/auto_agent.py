"""
AutoAgent - Autonomous task execution with fully agentic planning.

Uses AgenticPlanner for all planning decisions (no hardcoded logic).
Takes any open-ended task, discovers relevant skills, plans execution,
and runs the workflow automatically.

Refactored to inherit from AutonomousAgent for unified infrastructure.
"""
import asyncio
import logging
import inspect
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from .agentic_planner import AgenticPlanner
from .base import AutonomousAgent, AutonomousAgentConfig, ExecutionStep

logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Inferred task types."""
    RESEARCH = "research"           # Learn about something
    COMPARISON = "comparison"       # Compare things
    CREATION = "creation"           # Create content/document
    COMMUNICATION = "communication" # Send/share something
    ANALYSIS = "analysis"           # Analyze data
    AUTOMATION = "automation"       # Automate a workflow
    UNKNOWN = "unknown"


def _clean_for_display(text: str) -> str:
    """
    Remove internal context from text for user-facing display.

    Strips:
    - Transferable Learnings sections
    - Meta-Learning Advice
    - Multi-Perspective Analysis (keep short summary only)
    - Learned Insights
    - Relevant past experience
    """
    if not text:
        return text

    # Markers that indicate start of internal context
    internal_markers = [
        '# Transferable Learnings',
        '## Task Type Pattern',
        '## Role Advice',
        '## Meta-Learning Advice',
        '\n\nRelevant past experience:',
        '\n\nLearned Insights:',
        '\n\n[Multi-Perspective Analysis',
    ]

    result = text
    for marker in internal_markers:
        if marker in result:
            result = result.split(marker)[0]

    return result.strip()


@dataclass
class ExecutionResult:
    """Result of task execution."""
    success: bool
    task: str
    task_type: TaskType
    skills_used: List[str]
    steps_executed: int
    outputs: Dict[str, Any]
    final_output: Any
    errors: List[str] = field(default_factory=list)
    execution_time: float = 0.0


class AutoAgent(AutonomousAgent):
    """
    Autonomous agent that discovers and executes skills for any task.

    Inherits from AutonomousAgent for:
    - Skill discovery and selection
    - Execution planning via AgenticPlanner
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
        planner: Optional[AgenticPlanner] = None,
        skill_filter: Optional[str] = None
    ):
        """
        Initialize AutoAgent.

        Args:
            default_output_skill: Optional skill for final output (e.g., messaging skill)
            enable_output: Whether to send output via messaging (requires default_output_skill)
            skill_filter: Optional category filter for skill discovery
            max_steps: Maximum execution steps
            timeout: Default timeout for operations
            planner: Optional AgenticPlanner instance (creates new if None)
        """
        # Create config for base class
        config = AutonomousAgentConfig(
            name="AutoAgent",
            max_steps=max_steps,
            timeout=float(timeout),
            enable_replanning=True,
            max_replans=3,
            skill_filter=skill_filter,
            default_output_skill=default_output_skill,
            enable_output=enable_output and default_output_skill is not None,
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
        status_fn: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Execute prompt ensembling for multi-perspective analysis.

        Strategies:
        - self_consistency: Same prompt, N samples, synthesis
        - multi_perspective: Different expert personas (default)
        - gsa: Generative Self-Aggregation
        - debate: Multi-round argumentation
        """
        def _status(stage: str, detail: str = ""):
            if status_fn:
                status_fn(stage, detail)

        try:
            # Try to use the ensemble skill
            if self.skills_registry:
                skill = self.skills_registry.get_skill('claude-cli-llm')
                if skill:
                    ensemble_tool = skill.tools.get('ensemble_prompt_tool')
                    if ensemble_tool:
                        _status("Ensemble", f"using {strategy} strategy")
                        result = ensemble_tool({
                            'prompt': task,
                            'strategy': strategy,
                            'synthesis_style': 'structured'
                        })
                        return result

            # Fallback: Use DSPy directly
            import dspy
            if not hasattr(dspy.settings, 'lm') or dspy.settings.lm is None:
                return {'success': False, 'error': 'No LLM configured'}

            lm = dspy.settings.lm

            # Simple multi-perspective implementation
            perspectives = [
                ("analytical", "Analyze this from a data-driven, logical perspective:"),
                ("creative", "Consider unconventional angles and innovative solutions:"),
                ("critical", "Play devil's advocate - identify risks and problems:"),
                ("practical", "Focus on feasibility and actionable steps:"),
            ]

            responses = {}
            for name, prefix in perspectives:
                _status(f"  {name}", "analyzing...")
                try:
                    prompt = f"{prefix}\n\n{task}"
                    response = lm(prompt=prompt)
                    text = response[0] if isinstance(response, list) else str(response)
                    responses[name] = text
                except Exception as e:
                    logger.warning(f"Perspective '{name}' failed: {e}")

            if not responses:
                return {'success': False, 'error': 'All perspectives failed'}

            # Synthesize
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

    def _should_auto_ensemble(self, task: str) -> bool:
        """
        Determine if ensemble should be auto-enabled based on task type.

        BE CONSERVATIVE - ensemble adds significant latency (4x LLM calls).
        Only enable for tasks that genuinely benefit from multiple perspectives.
        """
        task_lower = task.lower()

        # EXCLUSION: Don't auto-ensemble for creation/generation tasks
        creation_keywords = [
            'create ', 'generate ', 'write ', 'build ', 'make ',
            'checklist', 'template', 'document', 'report',
            'draft ', 'prepare ', 'compile ',
        ]
        for keyword in creation_keywords:
            if keyword in task_lower:
                return False

        # Comparison indicators (STRONG signal)
        comparison_keywords = [
            ' vs ', ' versus ', 'compare ',
            'difference between', 'differences between',
            'pros and cons', 'advantages and disadvantages',
        ]

        # Decision indicators (STRONG signal)
        decision_keywords = [
            'should i ', 'should we ',
            'which is better', 'what is best',
            'choose between', 'decide between',
        ]

        for keyword in comparison_keywords + decision_keywords:
            if keyword in task_lower:
                return True

        return False

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
        ensemble = kwargs.pop('ensemble', None)  # None = auto-detect
        ensemble_strategy = kwargs.pop('ensemble_strategy', 'multi_perspective')

        def _status(stage: str, detail: str = ""):
            """Report progress if callback provided."""
            if status_callback:
                try:
                    status_callback(stage, detail)
                except Exception:
                    pass
            logger.info(f"  {stage}" + (f": {detail}" if detail else ""))

        # Ensure initialized
        self._ensure_initialized()

        # Run pre-execution hooks (from BaseAgent)
        try:
            await self._run_pre_hooks(task=task, **kwargs)
        except Exception as e:
            logger.warning(f"Pre-hook failed: {e}")

        # Auto-detect ensemble for certain task types
        if ensemble is None:
            ensemble = self._should_auto_ensemble(task)
            if ensemble:
                _status("Auto-ensemble", "enabled for analysis/comparison task")

        # Optional: Ensemble pre-analysis for enriched context
        enriched_task = task
        if ensemble:
            _status("Ensembling", f"strategy={ensemble_strategy}")
            ensemble_result = await self._execute_ensemble(task, ensemble_strategy, _status)
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
        task_type_enum = self._infer_task_type_enum(task)

        # Execute via AutonomousAgent._execute_impl (handles its own replanning)
        try:
            result = await super()._execute_impl(task=enriched_task, status_callback=status_callback, **kwargs)
        except Exception as e:
            logger.error(f"AutoAgent execution failed: {e}")
            result = {
                'success': False,
                'errors': [str(e)],
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
            execution_time=execution_time
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
