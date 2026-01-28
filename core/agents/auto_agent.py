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
        self._init_dependencies()
        skill = self._registry.get_skill(skill_name) if self._registry else None
        if not skill:
            # Suggest available skills
            available_skills = []
            if self._registry:
                available_skills = [s['name'] for s in self._registry.list_skills()[:5]]
            error_msg = f'Skill not found: {skill_name}'
            if available_skills:
                error_msg += f'. Available skills: {available_skills}'
            return {'success': False, 'error': error_msg}

        available_tools = list(skill.tools.keys()) if hasattr(skill, 'tools') else []
        tool = skill.tools.get(tool_name) if hasattr(skill, 'tools') else None
        if not tool:
            error_msg = f'Tool not found: {tool_name}'
            if available_tools:
                error_msg += f'. Available tools in {skill_name}: {available_tools[:5]}'
            return {'success': False, 'error': error_msg}

        try:
            if inspect.iscoroutinefunction(tool):
                result = await tool(params)
            else:
                result = tool(params)
            return result
        except Exception as e:
            logger.error(f"Tool execution failed: {e}", exc_info=True)
            return {'success': False, 'error': str(e)}

    def _validate_and_filter_steps(
        self,
        steps: List[ExecutionStep],
        skills: List[Dict[str, Any]]
    ) -> List[ExecutionStep]:
        """
        Validate that all steps reference existing tools and filter out invalid ones.
        
        Args:
            steps: Planned execution steps
            skills: Available skills with their tools
            
        Returns:
            Filtered list of steps with valid tools
        """
        self._init_dependencies()
        if not self._registry:
            return steps
        
        # Build skill->tools mapping
        skill_tools = {}
        for skill in skills:
            skill_name = skill.get('name', '')
            if skill_name:
                skill_obj = self._registry.get_skill(skill_name)
                if skill_obj:
                    skill_tools[skill_name] = list(skill_obj.tools.keys()) if hasattr(skill_obj, 'tools') else []
        
        valid_steps = []
        invalid_count = 0
        
        for step in steps:
            skill_name = step.skill_name
            tool_name = step.tool_name
            
            # Check if skill exists
            if skill_name not in skill_tools:
                logger.warning(f"  ⚠️  Step '{step.description}' uses invalid skill '{skill_name}', skipping")
                invalid_count += 1
                continue
            
            # Check if tool exists in skill
            available_tools = skill_tools.get(skill_name, [])
            
            # Auto-fix: If tool_name matches skill_name, try to use first available tool
            if tool_name not in available_tools:
                # Try to auto-fix common mistakes
                fixed_tool = None
                
                # Case 1: Tool name matches skill name (common mistake)
                if tool_name == skill_name and available_tools:
                    # Use first available tool (usually the main one)
                    fixed_tool = available_tools[0]
                    logger.info(
                        f"  🔧 Auto-fixed: '{tool_name}' → '{fixed_tool}' "
                        f"(skill name used instead of tool name)"
                    )
                # Case 2: Common tool name mappings (LLM invents intuitive names)
                elif available_tools:
                    # Map common LLM-invented names to actual tool names
                    tool_mappings = {
                        # File operations
                        'create_file': 'write_file_tool',
                        'write_file': 'write_file_tool',
                        'create_file_tool': 'write_file_tool',
                        'write_file_tool': 'write_file_tool',  # Already correct
                        'read_file': 'read_file_tool',
                        'read_file_tool': 'read_file_tool',  # Already correct
                        # Document conversion
                        'convert_document': 'convert_to_pdf_tool',
                        'convert_to_pdf': 'convert_to_pdf_tool',
                        'convert': 'convert_to_pdf_tool',
                        # PDF tools
                        'validate_pdf': 'get_metadata_tool',  # Closest match
                        'check_pdf': 'get_metadata_tool',
                    }
                    
                    # Check mapping first
                    if tool_name in tool_mappings:
                        mapped_name = tool_mappings[tool_name]
                        if mapped_name in available_tools:
                            fixed_tool = mapped_name
                            logger.info(
                                f"  🔧 Auto-fixed: '{tool_name}' → '{fixed_tool}' "
                                f"(tool name mapping)"
                            )
                    
                    # Case 3: Try fuzzy matching if mapping didn't work
                    if not fixed_tool:
                        for avail_tool in available_tools:
                            # Check if any tool name contains the requested tool name
                            if tool_name.lower() in avail_tool.lower() or avail_tool.lower() in tool_name.lower():
                                fixed_tool = avail_tool
                                logger.info(
                                    f"  🔧 Auto-fixed: '{tool_name}' → '{fixed_tool}' "
                                    f"(fuzzy match)"
                                )
                                break
                
                if fixed_tool:
                    # Update step with fixed tool name
                    step.tool_name = fixed_tool
                    valid_steps.append(step)
                else:
                    logger.warning(
                        f"  ⚠️  Step '{step.description}' uses invalid tool '{tool_name}' "
                        f"in skill '{skill_name}'. Available tools: {available_tools[:5]}"
                    )
                    invalid_count += 1
                    continue
            else:
                valid_steps.append(step)
        
        if invalid_count > 0:
            logger.warning(f"⚠️  Filtered out {invalid_count} invalid steps (tools not found)")
        
        return valid_steps

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
        
        # Step 2.3: Ensure code generation skills are available for code tasks
        task_lower = task.lower()
        is_code_task = any(keyword in task_lower for keyword in [
            'generate code', 'write code', 'create code', 'code generation',
            'implement', 'develop', 'programming', 'write file', 'create file'
        ])
        
        if is_code_task:
            self._init_dependencies()
            if self._registry:
                # Ensure file-operations is in the candidate list
                file_ops = self._registry.get_skill('file-operations')
                if file_ops:
                    file_ops_dict = {
                        'name': 'file-operations',
                        'description': 'File system operations: read, write, create directories, search files',
                        'tools': list(file_ops.tools.keys()) if hasattr(file_ops, 'tools') else []
                    }
                    # Add if not already present
                    if not any(s.get('name') == 'file-operations' for s in all_skills):
                        all_skills.append(file_ops_dict)
                        logger.info("➕ Added file-operations skill for code generation task")
                
                # Also ensure skill-creator is available
                skill_creator = self._registry.get_skill('skill-creator')
                if skill_creator:
                    skill_creator_dict = {
                        'name': 'skill-creator',
                        'description': 'Create new Jotty skills and code templates',
                        'tools': list(skill_creator.tools.keys()) if hasattr(skill_creator, 'tools') else []
                    }
                    if not any(s.get('name') == 'skill-creator' for s in all_skills):
                        all_skills.append(skill_creator_dict)
                        logger.info("➕ Added skill-creator skill for code generation task")

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

        # Step 3.5: Validate tools exist before execution
        steps = self._validate_and_filter_steps(steps, skills)
        
        # Step 3.6: If no valid steps and task requires code generation, try adding file-operations
        if not steps and is_code_task:
            logger.info("🔧 No valid steps for code generation task, attempting to add file-operations")
            
            # Try to get file-operations from registry
            self._init_dependencies()
            if self._registry:
                file_ops = self._registry.get_skill('file-operations')
                if file_ops:
                    file_ops_dict = {
                        'name': 'file-operations',
                        'description': 'File system operations: read, write, create directories, search files',
                        'tools': list(file_ops.tools.keys()) if hasattr(file_ops, 'tools') else []
                    }
                    
                    # Add file-operations to skills if not already there
                    if not any(s.get('name') == 'file-operations' for s in skills):
                        logger.info("➕ Adding file-operations skill for code generation")
                        skills.append(file_ops_dict)
                        # Re-plan with file-operations included
                        steps = self._plan_execution(task, task_type, skills)
                        steps = self._validate_and_filter_steps(steps, skills)
        
        if not steps:
            logger.warning("⚠️  No valid steps after tool validation, cannot proceed")
            return ExecutionResult(
                success=False,
                task=task,
                task_type=task_type,
                skills_used=[],
                steps_executed=0,
                outputs={},
                final_output=None,
                errors=["No valid tools found for planned steps. For code generation tasks, ensure file-operations or skill-creator skills are available."],
                execution_time=(datetime.now() - start_time).total_seconds()
            )

        # Step 4: Execute steps
        outputs = {}
        errors = []
        skills_used = []
        steps_executed = 0
        replan_count = 0
        max_replans = 3  # Prevent infinite replanning loops

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
                
            else:
                error_msg = result.get('error', 'Unknown error')
                errors.append(f"Step {i+1} ({step.skill_name}): {error_msg}")
                logger.warning(f"  ❌ Failed: {error_msg}")

                # For optional steps, continue; for required steps, consider replanning
                if not step.optional and i < len(steps) - 1 and replan_count < max_replans:
                    # Check if tool exists - if not, don't replan (will generate same error)
                    if 'Tool not found' in error_msg or 'Skill not found' in error_msg:
                        logger.warning(f"  ⚠️  Tool/skill not found, skipping replan (would generate same error)")
                        # Mark step as optional and continue
                        step.optional = True
                        continue
                    
                    # Try to replan remaining steps with current context
                    logger.info(f"  🔄 Attempting to replan remaining steps with current context")
                    replan_count += 1
                    try:
                        remaining_steps, _ = self.planner.plan_execution(
                            task=task,
                            task_type=task_type,
                            skills=skills,
                            previous_outputs=outputs,
                            max_steps=self.max_steps - i - 1
                        )
                        # Validate replanned steps before using them
                        remaining_steps = self._validate_and_filter_steps(remaining_steps, skills)
                        if remaining_steps:
                            # Replace remaining steps with replanned ones
                            steps = steps[:i+1] + remaining_steps
                            logger.info(f"  ✅ Replanned {len(remaining_steps)} remaining steps")
                        else:
                            logger.warning(f"  ⚠️  Replanning produced no valid steps, continuing with original plan")
                    except Exception as e:
                        logger.warning(f"  ⚠️  Replanning failed: {e}, continuing with original plan")
                elif replan_count >= max_replans:
                    logger.warning(f"  ⚠️  Max replans ({max_replans}) reached, stopping replanning")

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
