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
        params: Dict[str, Any],
        max_retries: int = 2
    ) -> Dict[str, Any]:
        """
        Execute a single tool with retry logic for network errors.
        
        For web-related tools, will also try alternative tools if primary fails.
        """
        import time
        
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

        # Retry logic for network errors
        last_error = None
        for attempt in range(max_retries):
            try:
                if inspect.iscoroutinefunction(tool):
                    result = await tool(params)
                else:
                    result = tool(params)
                
                # If success, return immediately
                if result.get('success', False):
                    return result
                
                # If failure but not a network error, return immediately (no retry)
                error_msg = result.get('error', '')
                if error_msg and 'Network error' not in error_msg and 'timeout' not in error_msg.lower() and 'Request' not in error_msg:
                    return result
                
                # Network error - retry
                last_error = error_msg
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s
                    logger.debug(f"  Retrying {skill_name}.{tool_name} after {wait_time}s (attempt {attempt + 2}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                else:
                    return result
                    
            except Exception as e:
                error_str = str(e)
                # Check if it's a network-related error
                is_network_error = any(keyword in error_str.lower() for keyword in [
                    'network', 'timeout', 'connection', 'request', 'http', 'dns', 'socket'
                ]) or 'requests' in str(type(e))
                
                if is_network_error and attempt < max_retries - 1:
                    # Network exception - retry
                    last_error = error_str
                    wait_time = 2 ** attempt
                    logger.debug(f"  Retrying {skill_name}.{tool_name} after network error (attempt {attempt + 2}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                else:
                    # Non-network exception or retries exhausted - don't retry
                    logger.error(f"Tool execution failed: {e}", exc_info=True)
                    return {'success': False, 'error': error_str}
            except Exception as e:
                # Non-network exception - don't retry
                logger.error(f"Tool execution failed: {e}", exc_info=True)
                return {'success': False, 'error': str(e)}
        
        # All retries exhausted
        return {'success': False, 'error': last_error or f'Failed after {max_retries} attempts'}
    
    async def _try_alternative_web_tools(
        self,
        step: ExecutionStep,
        params: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Try alternative web tools when primary web-search tool fails.
        
        Alternatives:
        1. web-scraper (if URL is available)
        2. http-client (if URL is available)
        """
        self._init_dependencies()
        if not self._registry:
            return None
        
        # Extract URL from params if available
        url = params.get('url') or params.get('query')
        if not url or '{' in str(url) or '${' in str(url):
            # URL not available or has unresolved template variables
            return None
        
        # Try web-scraper as alternative
        scraper_skill = self._registry.get_skill('web-scraper')
        if scraper_skill and hasattr(scraper_skill, 'tools'):
            scraper_tool = scraper_skill.tools.get('scrape_website_tool')
            if scraper_tool:
                try:
                    logger.debug(f"  Trying web-scraper as alternative...")
                    scraper_params = {'url': url} if url.startswith('http') else params
                    if inspect.iscoroutinefunction(scraper_tool):
                        result = await scraper_tool(scraper_params)
                    else:
                        result = scraper_tool(scraper_params)
                    
                    if result.get('success'):
                        logger.info(f"  ✅ Alternative tool (web-scraper) succeeded")
                        return result
                except Exception as e:
                    logger.debug(f"  web-scraper alternative failed: {e}")
        
        # Try http-client as alternative
        http_skill = self._registry.get_skill('http-client')
        if http_skill and hasattr(http_skill, 'tools'):
            http_tool = http_skill.tools.get('http_get_tool')
            if http_tool and url.startswith('http'):
                try:
                    logger.debug(f"  Trying http-client as alternative...")
                    http_params = {'url': url}
                    if inspect.iscoroutinefunction(http_tool):
                        result = await http_tool(http_params)
                    else:
                        result = http_tool(http_params)
                    
                    if result.get('success'):
                        # Convert HTTP response to web-search format
                        logger.info(f"  ✅ Alternative tool (http-client) succeeded")
                        return {
                            'success': True,
                            'url': url,
                            'content': str(result.get('body', '')),
                            'title': 'Fetched Content'
                        }
                except Exception as e:
                    logger.debug(f"  http-client alternative failed: {e}")
        
        return None
    
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
        """
        Resolve template variables in params.
        
        Supports formats:
        - ${step_name.field} - get field from output
        - ${step_name.output.field} - get field from output dict
        - ${step_name[0].field} - array indexing
        - {step_name.field} - without $ prefix
        """
        import re
        
        resolved = {}

        def resolve_path(path: str, outputs: Dict[str, Any]) -> Any:
            """
            Resolve a path like:
            - 'search_results.results[0].url' (nested)
            - 'search_results[0].url' (direct array access - smart fallback)
            - 'search_results.url' (simple key)
            """
            import re
            
            # Handle direct array access: search_results[0].url
            # Split by . but preserve array indices
            parts = []
            current = ''
            i = 0
            while i < len(path):
                if path[i] == '[':
                    if current:
                        parts.append(current)
                        current = ''
                    # Find the closing ]
                    j = i + 1
                    while j < len(path) and path[j] != ']':
                        j += 1
                    if j < len(path):
                        parts.append(f'[{path[i+1:j]}]')
                        i = j + 1
                    else:
                        current += path[i]
                        i += 1
                elif path[i] == '.':
                    if current:
                        parts.append(current)
                        current = ''
                    i += 1
                else:
                    current += path[i]
                    i += 1
            if current:
                parts.append(current)
            
            value = outputs
            
            for idx, part in enumerate(parts):
                if value is None:
                    return None
                
                # Handle array indexing: [0] or key[0]
                if part.startswith('[') and part.endswith(']'):
                    # Direct array access: [0]
                    try:
                        index = int(part[1:-1])
                        
                        # Smart fallback: if value is a dict with 'results' key, use that
                        # This handles cases like {search_results[0].url} where search_results
                        # is a dict with a 'results' array inside
                        if isinstance(value, dict) and 'results' in value:
                            value = value['results']
                        
                        if isinstance(value, (list, tuple)) and index < len(value):
                            value = value[index]
                        else:
                            return None
                    except (ValueError, IndexError, TypeError):
                        return None
                elif '[' in part and ']' in part:
                    # Key with array: key[0]
                    key = part.split('[')[0]
                    index_str = part.split('[')[1].split(']')[0]
                    try:
                        index = int(index_str)
                        
                        # Get the value for the key
                        if isinstance(value, dict):
                            key_value = value.get(key)
                        else:
                            return None
                        
                        # Smart fallback: if key_value is a dict with 'results' key, use that
                        # This handles cases like {search_results[0].url} where search_results is a dict
                        # with a 'results' array inside
                        if isinstance(key_value, dict) and 'results' in key_value:
                            key_value = key_value['results']
                        
                        # Now try to index
                        if isinstance(key_value, (list, tuple)) and index < len(key_value):
                            value = key_value[index]
                        elif isinstance(key_value, dict):
                            # If it's still a dict, maybe it's a list-like dict? Try direct access
                            return None
                        else:
                            return None
                    except (ValueError, IndexError, TypeError):
                        return None
                else:
                    # Regular key access
                    if isinstance(value, dict):
                        value = value.get(part)
                    else:
                        return None
                
                if value is None:
                    break
            
            return value

        for key, value in params.items():
            if isinstance(value, str):
                # Handle ${...} format
                pattern = r'\$\{([^}]+)\}'
                def replacer(match):
                    path = match.group(1)
                    resolved_value = resolve_path(path, outputs)
                    return str(resolved_value) if resolved_value is not None else match.group(0)
                
                value = re.sub(pattern, replacer, value)
                
                # Handle {...} format (without $)
                pattern2 = r'\{([^}]+)\}'
                value = re.sub(pattern2, replacer, value)
                
                # Handle {{...}} format (double braces)
                pattern3 = r'\{\{([^}]+)\}\}'
                def replacer3(match):
                    out_key = match.group(1)
                    out_value = outputs.get(out_key)
                    if isinstance(out_value, dict):
                        if 'results' in out_value:
                            # Format search results
                            results = out_value['results']
                            formatted = '\n'.join([
                                f"- {r.get('title', '')}: {r.get('snippet', '')}"
                                for r in results[:5]
                            ])
                            return formatted
                        elif 'text' in out_value:
                            return out_value['text']
                        else:
                            return str(out_value)
                    return str(out_value) if out_value is not None else match.group(0)
                
                value = re.sub(pattern3, replacer3, value)
                
                resolved[key] = value
            elif isinstance(value, dict):
                resolved[key] = self._resolve_params(value, outputs)
            elif isinstance(value, list):
                resolved[key] = [
                    self._resolve_params(item, outputs) if isinstance(item, dict) else item
                    for item in value
                ]
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

                # Try alternative web tools for network errors
                if ('Network error' in error_msg or 'timeout' in error_msg.lower() or 
                    'Request' in error_msg) and step.skill_name == 'web-search':
                    logger.info(f"  🔄 Trying alternative web tools for network error...")
                    alternative_result = await self._try_alternative_web_tools(step, resolved_params)
                    if alternative_result and alternative_result.get('success'):
                        outputs[step.output_key or f'step_{i}'] = alternative_result
                        skills_used.append(step.skill_name)
                        steps_executed += 1
                        logger.info(f"  ✅ Success with alternative tool")
                        continue

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
