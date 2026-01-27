"""
AutoAgent - Autonomous task execution with fully agentic planning.

Uses AgenticPlanner for all planning decisions (no hardcoded logic).
Takes any open-ended task, discovers relevant skills, plans execution,
and runs the workflow automatically.
"""
import asyncio
import logging
import inspect
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from .agentic_planner import AgenticPlanner

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


@dataclass
class ExecutionStep:
    """A step in the execution plan."""
    skill_name: str
    tool_name: str
    params: Dict[str, Any]
    description: str
    depends_on: List[int] = field(default_factory=list)
    output_key: str = ""
    optional: bool = False


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


class AutoAgent:
    """
    Autonomous agent that discovers and executes skills for any task.

    Usage:
        agent = AutoAgent()
        result = await agent.execute("RNN vs CNN")
    """

    def __init__(
        self,
        default_output_skill: str = "telegram-sender",
        enable_output: bool = True,
        max_steps: int = 10,
        timeout: int = 300,
        planner: Optional[AgenticPlanner] = None
    ):
        """
        Initialize AutoAgent.

        Args:
            default_output_skill: Skill to use for final output (telegram, slack, etc.)
            enable_output: Whether to send output to messaging
            max_steps: Maximum execution steps
            timeout: Default timeout for operations
            planner: Optional AgenticPlanner instance (creates new if None)
        """
        self.default_output_skill = default_output_skill
        self.enable_output = enable_output
        self.max_steps = max_steps
        self.timeout = timeout

        # Use agentic planner for all planning decisions
        self.planner = planner or AgenticPlanner()

        self._registry = None
        self._manifest = None
        self._discovery = None

    def _init_dependencies(self):
        """Lazy load dependencies."""
        if self._registry is None:
            from core.registry.skills_registry import get_skills_registry
            self._registry = get_skills_registry()
            self._registry.init()

        if self._manifest is None:
            from core.registry.skills_manifest import get_skills_manifest
            self._manifest = get_skills_manifest()

    def _infer_task_type(self, task: str) -> TaskType:
        """Infer task type using agentic planner (semantic understanding)."""
        task_type, reasoning, confidence = self.planner.infer_task_type(task)
        logger.debug(f"Task type inference: {task_type.value} (confidence: {confidence:.2f})")
        return task_type

    def _discover_skills(self, task: str) -> List[Dict[str, Any]]:
        """Discover relevant skills for task."""
        import sys
        sys.path.insert(0, 'skills/skill-discovery')

        try:
            from skills.skill_discovery import tools as discovery
        except ImportError:
            # Direct import
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "discovery_tools",
                "skills/skill-discovery/tools.py"
            )
            discovery = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(discovery)

        result = discovery.find_skills_for_task_tool({'task': task, 'max_results': 8})
        return result.get('recommended_skills', [])

    def _plan_execution(
        self,
        task: str,
        task_type: TaskType,
        skills: List[Dict[str, Any]],
        previous_outputs: Optional[Dict[str, Any]] = None
    ) -> List[ExecutionStep]:
        """
        Plan execution steps using agentic planner (fully LLM-based).
        """
        steps, reasoning = self.planner.plan_execution(
            task=task,
            task_type=task_type,
            skills=skills,
            previous_outputs=previous_outputs,
            max_steps=self.max_steps
        )
        logger.debug(f"Execution plan reasoning: {reasoning}")
        return steps

    # All hardcoded planning methods removed - now using AgenticPlanner
    # _plan_comparison, _plan_research, _plan_creation, _plan_analysis
    # are replaced by planner.plan_execution() which uses LLM reasoning

    async def _execute_tool(
        self,
        skill_name: str,
        tool_name: str,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a single tool."""
        skill = self._registry.get_skill(skill_name)
        if not skill:
            return {'success': False, 'error': f'Skill not found: {skill_name}'}

        tool = skill.tools.get(tool_name)
        if not tool:
            return {'success': False, 'error': f'Tool not found: {tool_name}'}

        try:
            if inspect.iscoroutinefunction(tool):
                result = await tool(params)
            else:
                result = tool(params)
            return result
        except Exception as e:
            logger.error(f"Tool execution failed: {e}", exc_info=True)
            return {'success': False, 'error': str(e)}

    def _resolve_params(
        self,
        params: Dict[str, Any],
        outputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Resolve template variables in params."""
        resolved = {}

        for key, value in params.items():
            if isinstance(value, str) and '{{' in value:
                # Replace template variables
                for out_key, out_value in outputs.items():
                    placeholder = '{{' + out_key + '}}'
                    if placeholder in value:
                        # Convert output to string for insertion
                        if isinstance(out_value, dict):
                            if 'results' in out_value:
                                # Format search results
                                results = out_value['results']
                                formatted = '\n'.join([
                                    f"- {r.get('title', '')}: {r.get('snippet', '')}"
                                    for r in results[:5]
                                ])
                                value = value.replace(placeholder, formatted)
                            elif 'text' in out_value:
                                value = value.replace(placeholder, out_value['text'])
                            else:
                                value = value.replace(placeholder, str(out_value))
                        else:
                            value = value.replace(placeholder, str(out_value))
                resolved[key] = value
            else:
                resolved[key] = value

        return resolved

    async def execute(self, task: str) -> ExecutionResult:
        """
        Execute a task automatically.

        Args:
            task: Task description (can be minimal like "RNN vs CNN")

        Returns:
            ExecutionResult with outputs and status
        """
        start_time = datetime.now()

        # Initialize
        self._init_dependencies()

        logger.info(f"AutoAgent executing: {task}")

        # Step 1: Infer task type
        task_type = self._infer_task_type(task)
        logger.info(f"Task type: {task_type.value}")

        # Step 2: Discover skills
        all_skills = self._discover_skills(task)
        logger.info(f"Discovered {len(all_skills)} potential skills")

        # Step 2.5: Select best skills using agentic planner
        if all_skills:
            skills, selection_reasoning = self.planner.select_skills(
                task=task,
                available_skills=all_skills,
                max_skills=8
            )
            logger.info(f"Selected {len(skills)} skills: {[s.get('name') for s in skills]}")
            logger.debug(f"Selection reasoning: {selection_reasoning}")
        else:
            # Fallback: try to get skills from registry
            self._init_dependencies()
            if self._registry:
                all_skills_list = self._registry.list_skills()
                skills = [
                    {
                        'name': s['name'],
                        'description': s.get('description', ''),
                        'tools': s.get('tools', [])
                    }
                    for s in all_skills_list[:10]
                ]
                if skills:
                    skills, _ = self.planner.select_skills(task, skills, max_skills=5)
            else:
                skills = []

        # Step 3: Plan execution using agentic planner
        steps = self._plan_execution(task, task_type, skills)
        logger.info(f"Planned {len(steps)} steps")

        # Step 4: Execute steps
        outputs = {}
        errors = []
        skills_used = []
        steps_executed = 0

        for i, step in enumerate(steps):
            logger.info(f"Step {i+1}/{len(steps)}: {step.description}")

            # Resolve params with previous outputs
            resolved_params = self._resolve_params(step.params, outputs)

            # Execute
            result = await self._execute_tool(
                step.skill_name,
                step.tool_name,
                resolved_params
            )

            if result.get('success'):
                outputs[step.output_key or f'step_{i}'] = result
                skills_used.append(step.skill_name)
                steps_executed += 1
                logger.info(f"  ✅ Success")
                
                # If step failed but we have outputs, try replanning with new context
                # This enables adaptive planning based on intermediate results
            else:
                error_msg = result.get('error', 'Unknown error')
                errors.append(f"Step {i+1} ({step.skill_name}): {error_msg}")
                logger.warning(f"  ❌ Failed: {error_msg}")

                # For optional steps, continue; for required steps, consider replanning
                if not step.optional and i < len(steps) - 1:
                    # Try to replan remaining steps with current context
                    logger.info(f"  🔄 Attempting to replan remaining steps with current context")
                    try:
                        remaining_steps, _ = self.planner.plan_execution(
                            task=task,
                            task_type=task_type,
                            skills=skills,
                            previous_outputs=outputs,
                            max_steps=self.max_steps - i - 1
                        )
                        # Replace remaining steps with replanned ones
                        steps = steps[:i+1] + remaining_steps
                        logger.info(f"  ✅ Replanned {len(remaining_steps)} remaining steps")
                    except Exception as e:
                        logger.warning(f"  ⚠️  Replanning failed: {e}, continuing with original plan")

        # Determine final output
        final_output = None
        if 'comparison_text' in outputs:
            final_output = outputs['comparison_text'].get('text', '')
        elif 'research_text' in outputs:
            final_output = outputs['research_text'].get('text', '')
        elif 'content' in outputs:
            final_output = outputs['content'].get('text', '')
        elif 'slides' in outputs:
            final_output = outputs['slides']
        elif 'search_results' in outputs:
            # Format search results as fallback when LLM fails
            search_data = outputs['search_results']
            if search_data.get('results'):
                formatted = f"# Search Results for: {task}\n\n"
                for i, r in enumerate(search_data['results'][:10], 1):
                    formatted += f"## {i}. {r.get('title', 'No title')}\n"
                    formatted += f"{r.get('snippet', '')}\n"
                    formatted += f"Source: {r.get('url', '')}\n\n"
                final_output = formatted
        elif outputs:
            # Get last output
            final_output = list(outputs.values())[-1]

        execution_time = (datetime.now() - start_time).total_seconds()

        return ExecutionResult(
            success=steps_executed > 0,
            task=task,
            task_type=task_type,
            skills_used=list(set(skills_used)),
            steps_executed=steps_executed,
            outputs=outputs,
            final_output=final_output,
            errors=errors,
            execution_time=execution_time
        )

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

        # Send output
        output_skill = output_skill or self.default_output_skill
        skill = self._registry.get_skill(output_skill)

        if skill:
            send_tool = skill.tools.get('send_message_tool') or skill.tools.get('send_telegram_message_tool')

            if send_tool and result.final_output:
                # Format message
                if isinstance(result.final_output, str):
                    message = f"📋 Task: {task}\n\n{result.final_output[:3500]}"
                else:
                    message = f"📋 Task: {task}\n\n✅ Completed with {result.steps_executed} steps"

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
