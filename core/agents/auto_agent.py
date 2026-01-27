"""
AutoAgent - Autonomous task execution with skill discovery.

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
        timeout: int = 300
    ):
        """
        Initialize AutoAgent.

        Args:
            default_output_skill: Skill to use for final output (telegram, slack, etc.)
            enable_output: Whether to send output to messaging
            max_steps: Maximum execution steps
            timeout: Default timeout for operations
        """
        self.default_output_skill = default_output_skill
        self.enable_output = enable_output
        self.max_steps = max_steps
        self.timeout = timeout

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
        """Infer the type of task from description."""
        task_lower = task.lower()

        # Comparison patterns
        if any(w in task_lower for w in ['vs', 'versus', 'compare', 'difference', 'between']):
            return TaskType.COMPARISON

        # Creation patterns
        if any(w in task_lower for w in ['create', 'make', 'generate', 'build', 'write', 'design']):
            return TaskType.CREATION

        # Communication patterns
        if any(w in task_lower for w in ['send', 'share', 'post', 'notify', 'message']):
            return TaskType.COMMUNICATION

        # Analysis patterns
        if any(w in task_lower for w in ['analyze', 'analysis', 'report', 'statistics', 'data']):
            return TaskType.ANALYSIS

        # Research patterns (default for questions/topics)
        if any(w in task_lower for w in ['what', 'how', 'why', 'explain', 'learn', 'research']):
            return TaskType.RESEARCH

        # Short phrases without action words = research
        if len(task.split()) <= 5:
            return TaskType.RESEARCH

        return TaskType.UNKNOWN

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
        skills: List[Dict[str, Any]]
    ) -> List[ExecutionStep]:
        """
        Plan execution steps based on task type and available skills.
        """
        steps = []
        skill_names = [s['name'] for s in skills]

        # Build plan based on task type
        if task_type == TaskType.COMPARISON:
            steps = self._plan_comparison(task, skill_names)
        elif task_type == TaskType.RESEARCH:
            steps = self._plan_research(task, skill_names)
        elif task_type == TaskType.CREATION:
            steps = self._plan_creation(task, skill_names)
        elif task_type == TaskType.ANALYSIS:
            steps = self._plan_analysis(task, skill_names)
        else:
            # Default: research flow
            steps = self._plan_research(task, skill_names)

        return steps[:self.max_steps]

    def _plan_comparison(self, task: str, skills: List[str]) -> List[ExecutionStep]:
        """Plan for comparison tasks."""
        steps = []

        # Step 1: Web search
        if 'web-search' in skills:
            steps.append(ExecutionStep(
                skill_name='web-search',
                tool_name='search_web_tool',
                params={'query': task, 'max_results': 10},
                description='Search for comparison information',
                output_key='search_results'
            ))

        # Step 2: Generate comparison with LLM
        if 'claude-cli-llm' in skills:
            steps.append(ExecutionStep(
                skill_name='claude-cli-llm',
                tool_name='generate_text_tool',
                params={
                    'prompt': f'''Create a detailed comparison for: {task}

Include:
1. A comparison table with key differences
2. When to use each option
3. Pros and cons of each
4. Recommendation based on use case

Use the following research for context:
{{{{search_results}}}}''',
                    'model': 'sonnet',
                    'timeout': 120
                },
                description='Generate comparison analysis',
                depends_on=[0] if 'web-search' in skills else [],
                output_key='comparison_text'
            ))

        # Step 3: Create slides (optional)
        if 'slide-generator' in skills:
            steps.append(ExecutionStep(
                skill_name='slide-generator',
                tool_name='generate_slides_from_topic_tool',
                params={
                    'topic': task,
                    'n_slides': 8,
                    'template': 'dark',
                    'export_as': 'pdf',
                    'send_telegram': self.enable_output
                },
                description='Generate presentation slides',
                depends_on=[1] if 'claude-cli-llm' in skills else [],
                output_key='slides',
                optional=True
            ))

        return steps

    def _plan_research(self, task: str, skills: List[str]) -> List[ExecutionStep]:
        """Plan for research tasks."""
        steps = []

        # Step 1: Web search
        if 'web-search' in skills:
            steps.append(ExecutionStep(
                skill_name='web-search',
                tool_name='search_web_tool',
                params={'query': task, 'max_results': 10},
                description='Search for information',
                output_key='search_results'
            ))

        # Step 2: Summarize/analyze with LLM
        if 'claude-cli-llm' in skills:
            steps.append(ExecutionStep(
                skill_name='claude-cli-llm',
                tool_name='generate_text_tool',
                params={
                    'prompt': f'''Research and explain: {task}

Provide:
1. Clear explanation of the topic
2. Key concepts and terminology
3. Important points to understand
4. Practical examples if applicable

Use the following research for context:
{{{{search_results}}}}''',
                    'model': 'sonnet',
                    'timeout': 120
                },
                description='Generate research summary',
                depends_on=[0] if 'web-search' in skills else [],
                output_key='research_text'
            ))
        elif 'summarize' in skills:
            steps.append(ExecutionStep(
                skill_name='summarize',
                tool_name='summarize_text_tool',
                params={
                    'text': '{{search_results}}',
                    'length': 'medium',
                    'style': 'paragraph'
                },
                description='Summarize research',
                depends_on=[0] if 'web-search' in skills else [],
                output_key='research_text'
            ))

        return steps

    def _plan_creation(self, task: str, skills: List[str]) -> List[ExecutionStep]:
        """Plan for creation tasks."""
        steps = []

        # Determine what to create
        task_lower = task.lower()

        if any(w in task_lower for w in ['slide', 'presentation', 'ppt']):
            if 'slide-generator' in skills:
                steps.append(ExecutionStep(
                    skill_name='slide-generator',
                    tool_name='generate_slides_from_topic_tool',
                    params={
                        'topic': task,
                        'n_slides': 10,
                        'template': 'dark',
                        'export_as': 'both',
                        'send_telegram': self.enable_output
                    },
                    description='Generate presentation',
                    output_key='slides'
                ))

        elif any(w in task_lower for w in ['image', 'picture', 'art']):
            if 'openai-image-gen' in skills:
                steps.append(ExecutionStep(
                    skill_name='openai-image-gen',
                    tool_name='generate_image_tool',
                    params={'prompt': task, 'size': '1024x1024'},
                    description='Generate image',
                    output_key='image'
                ))
            elif 'image-generator' in skills:
                steps.append(ExecutionStep(
                    skill_name='image-generator',
                    tool_name='generate_image_tool',
                    params={'prompt': task},
                    description='Generate image',
                    output_key='image'
                ))

        elif any(w in task_lower for w in ['pdf', 'document', 'report']):
            # Research first, then create PDF
            if 'web-search' in skills:
                steps.append(ExecutionStep(
                    skill_name='web-search',
                    tool_name='search_web_tool',
                    params={'query': task, 'max_results': 10},
                    description='Research topic',
                    output_key='search_results'
                ))

            if 'claude-cli-llm' in skills:
                steps.append(ExecutionStep(
                    skill_name='claude-cli-llm',
                    tool_name='generate_text_tool',
                    params={
                        'prompt': f'Write a detailed document about: {task}\n\nResearch:\n{{{{search_results}}}}',
                        'model': 'sonnet',
                        'timeout': 180
                    },
                    description='Generate document content',
                    depends_on=[0] if 'web-search' in skills else [],
                    output_key='document_text'
                ))

        else:
            # Default: use LLM to create content
            if 'claude-cli-llm' in skills:
                steps.append(ExecutionStep(
                    skill_name='claude-cli-llm',
                    tool_name='generate_text_tool',
                    params={
                        'prompt': task,
                        'model': 'sonnet',
                        'timeout': 120
                    },
                    description='Generate content',
                    output_key='content'
                ))

        return steps

    def _plan_analysis(self, task: str, skills: List[str]) -> List[ExecutionStep]:
        """Plan for analysis tasks."""
        # Similar to research but with more data focus
        return self._plan_research(task, skills)

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
        skills = self._discover_skills(task)
        skill_names = [s['name'] for s in skills]
        logger.info(f"Discovered skills: {skill_names}")

        if not skills:
            # Fallback to basic skills
            skills = [
                {'name': 'web-search', 'category': 'search'},
                {'name': 'claude-cli-llm', 'category': 'llm'}
            ]
            skill_names = ['web-search', 'claude-cli-llm']

        # Step 3: Plan execution
        steps = self._plan_execution(task, task_type, skills)
        logger.info(f"Planned {len(steps)} steps")

        # Step 4: Execute steps
        outputs = {}
        errors = []
        skills_used = []
        steps_executed = 0

        for i, step in enumerate(steps):
            logger.info(f"Step {i+1}: {step.description}")

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

                if not step.optional:
                    # Continue anyway for robustness
                    pass

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
