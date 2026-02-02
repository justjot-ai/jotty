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
        self.default_output_skill = default_output_skill
        self.enable_output = enable_output and default_output_skill is not None
        self.max_steps = max_steps
        self.timeout = timeout
        self.skill_filter = skill_filter  # Category filter for multi-agent mode

        # Use agentic planner for all planning decisions
        self.planner = planner or AgenticPlanner()

        self._registry = None
        self._manifest = None
        self._discovery = None

        if skill_filter:
            logger.info(f"AutoAgent initialized with skill filter: {skill_filter}")

    def _init_dependencies(self):
        """Lazy load dependencies."""
        if self._registry is None:
            from ..registry.skills_registry import get_skills_registry
            self._registry = get_skills_registry()
            self._registry.init()

        if self._manifest is None:
            from ..registry.skills_manifest import get_skills_manifest
            self._manifest = get_skills_manifest()

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

        Args:
            task: The task/question to analyze
            strategy: Ensembling strategy
            status_fn: Optional status callback function

        Returns:
            Dict with ensemble results including synthesized response
        """
        def _status(stage: str, detail: str = ""):
            if status_fn:
                status_fn(stage, detail)

        try:
            # Try to use the ensemble skill
            self._init_dependencies()
            if self._registry:
                skill = self._registry.get_skill('claude-cli-llm')
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

        Auto-enables for: comparisons, decisions (NOT general analysis/creation).
        Override with ensemble=False if not desired.

        Args:
            task: The task description

        Returns:
            True if ensemble should be auto-enabled
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
                logger.debug(f"Auto-ensemble SKIPPED for creation task: {keyword}")
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
                logger.debug(f"Auto-ensemble triggered by keyword: {keyword}")
                return True

        return False

    def _infer_task_type(self, task: str) -> TaskType:
        """Infer task type using agentic planner (semantic understanding)."""
        task_type, reasoning, confidence = self.planner.infer_task_type(task)
        logger.debug(f"Task type inference: {task_type.value} (confidence: {confidence:.2f})")
        return task_type

    def _discover_skills(self, task: str) -> List[Dict[str, Any]]:
        """
        Discover relevant skills for task using Jotty's SkillsRegistry.

        Uses semantic matching (task words vs skill name/description).
        No hardcoded categories or skill lists - purely data-driven.
        """
        try:
            from ..registry.skills_registry import get_skills_registry
            registry = get_skills_registry()

            if not registry.initialized:
                registry.init()

            task_lower = task.lower()
            # Filter out stop words for better matching
            stop_words = {'the', 'and', 'for', 'with', 'how', 'what', 'are', 'is', 'to', 'of', 'in', 'on', 'a', 'an'}
            task_words = [w for w in task_lower.split() if len(w) > 2 and w not in stop_words]

            skills = []

            for skill_name, skill_def in registry.loaded_skills.items():
                skill_name_lower = skill_name.lower()
                desc = getattr(skill_def, 'description', '') or ''
                desc_lower = desc.lower()

                # Calculate relevance score based on semantic matching
                score = 0

                # Match task words against skill name and description
                for word in task_words:
                    if word in skill_name_lower:
                        score += 3  # Strong match: word in skill name
                    if word in desc_lower:
                        score += 1  # Weak match: word in description

                # Include skills with positive relevance
                if score > 0:
                    skills.append({
                        'name': skill_name,
                        'description': desc or skill_name,
                        'category': getattr(skill_def, 'category', 'general'),
                        'relevance_score': score
                    })

            # Sort by relevance score
            skills.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)

            # If no matching skills, return all available (let LLM select)
            if not skills:
                for skill_name, skill_def in list(registry.loaded_skills.items())[:20]:
                    desc = getattr(skill_def, 'description', '') or ''
                    skills.append({
                        'name': skill_name,
                        'description': desc or skill_name,
                        'category': getattr(skill_def, 'category', 'general'),
                        'relevance_score': 0
                    })

            return skills[:15]  # Return top 15 for LLM to select from

        except Exception as e:
            logger.warning(f"Skill discovery failed: {e}, returning empty list")
            return []

    async def _ensure_skill_dependencies(
        self,
        skill_names: List[str],
        status_callback: Optional[Callable] = None
    ):
        """
        Ensure all selected skills have their dependencies installed.

        Args:
            skill_names: List of skill names to check
            status_callback: Optional callback for status updates
        """
        def _status(stage: str, detail: str = ""):
            if status_callback:
                try:
                    status_callback(stage, detail)
                except Exception:
                    pass

        try:
            from ..orchestration.v2.swarm_installer import SwarmInstaller
            installer = SwarmInstaller()

            for skill_name in skill_names:
                results = await installer.install_skill_dependencies(skill_name)
                if results:
                    installed = [r.package for r in results if r.success]
                    if installed:
                        _status("Dependencies installed", ", ".join(installed))

        except Exception as e:
            logger.debug(f"Dependency installation skipped: {e}")

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
        max_retries: int = 2,
        status_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Execute a single tool with retry logic for network errors.

        For web-related tools, will also try alternative tools if primary fails.
        """
        import time

        def _tool_status(msg: str):
            if status_callback:
                try:
                    status_callback("    Tool", msg)
                except Exception:
                    pass

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
                if attempt > 0:
                    _tool_status(f"retry attempt {attempt + 1}/{max_retries}")
                if inspect.iscoroutinefunction(tool):
                    result = await tool(params)
                else:
                    result = tool(params)

                # Handle None or non-dict results
                if result is None:
                    result = {'success': False, 'error': f'Tool {tool_name} returned None'}
                elif not isinstance(result, dict):
                    result = {'success': True, 'output': result}

                # Ensure error message exists for failed results
                if not result.get('success', False) and not result.get('error'):
                    # Try to extract error info from result
                    error_info = result.get('message') or result.get('reason') or result.get('details')
                    if error_info:
                        result['error'] = str(error_info)
                    else:
                        result['error'] = f'Tool {skill_name}.{tool_name} failed without error details'

                # If success, return immediately
                if result.get('success', False):
                    return result

                # If failure but not a network error, return immediately (no retry)
                error_msg = result.get('error', '')
                if error_msg and 'Network error' not in error_msg and 'timeout' not in error_msg.lower() and 'Request' not in error_msg:
                    return result

                # Network error - retry
                last_error = error_msg or f'Tool {tool_name} failed'
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

            if tool_name not in available_tools:
                fixed_tool = None

                # Case 1: Tool name matches skill name (LLM used skill name as tool)
                if tool_name == skill_name and available_tools:
                    fixed_tool = available_tools[0]

                # Case 2: Fuzzy substring matching
                if not fixed_tool and available_tools:
                    for avail_tool in available_tools:
                        if tool_name.lower() in avail_tool.lower() or avail_tool.lower() in tool_name.lower():
                            fixed_tool = avail_tool
                            break

                # Case 3: Use first available tool as last resort
                if not fixed_tool and available_tools:
                    fixed_tool = available_tools[0]

                if fixed_tool:
                    logger.info(f"  Auto-fixed: '{tool_name}' -> '{fixed_tool}' in skill '{skill_name}'")
                    step.tool_name = fixed_tool
                    valid_steps.append(step)
                else:
                    logger.warning(
                        f"  Step '{step.description}' uses invalid tool '{tool_name}' "
                        f"in skill '{skill_name}'. Available: {available_tools[:5]}"
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
        outputs: Dict[str, Any],
        step=None
    ) -> Dict[str, Any]:
        """
        Resolve template variables in params and auto-inject dependent outputs.

        Supports formats:
        - ${step_name.field} - get field from output
        - ${step_name.output.field} - get field from output dict
        - ${step_name[0].field} - array indexing
        - {step_name.field} - without $ prefix

        Auto-injection: If a step has depends_on but params contain no template
        references, the dependent outputs are formatted and injected into the
        first text/content/input param.
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
                    if resolved_value is None:
                        return match.group(0)
                    # Format complex objects (dicts/lists) into readable text
                    if isinstance(resolved_value, (dict, list)):
                        return self._format_output_as_text(resolved_value)
                    return str(resolved_value)

                value = re.sub(pattern, replacer, value)

                # Handle {...} format (without $) - only if it looks like a template var
                pattern2 = r'\{([a-zA-Z_][a-zA-Z0-9_.\[\]]*)\}'
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
                resolved[key] = self._resolve_params(value, outputs, step=step)
            elif isinstance(value, list):
                resolved[key] = [
                    self._resolve_params(item, outputs, step=step) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                resolved[key] = value

        # Auto-inject dependent outputs when no template vars were resolved
        if step and hasattr(step, 'depends_on') and step.depends_on and outputs:
            resolved = self._auto_inject_dependent_outputs(resolved, outputs, step)

        return resolved

    def _format_output_as_text(self, output: Any) -> str:
        """Format a step output (dict/list/str) into readable text for downstream tools."""
        if isinstance(output, str):
            return output
        if isinstance(output, dict):
            # Search results format
            if 'results' in output and isinstance(output['results'], list):
                parts = []
                for r in output['results']:
                    if isinstance(r, dict):
                        title = r.get('title', '')
                        snippet = r.get('snippet', '')
                        url = r.get('url', '')
                        parts.append(f"- {title}: {snippet}" + (f" ({url})" if url else ""))
                    else:
                        parts.append(f"- {r}")
                return '\n'.join(parts)
            # Text output
            if 'text' in output:
                return str(output['text'])
            if 'output' in output:
                return str(output['output'])
            if 'summary' in output:
                return str(output['summary'])
            if 'content' in output:
                return str(output['content'])
            # Generic dict - pick the most content-rich value
            best_key = None
            best_len = 0
            for k, v in output.items():
                if k in ('success', 'error', 'count', 'query', 'format', 'model', 'length', 'style'):
                    continue
                v_str = str(v)
                if len(v_str) > best_len:
                    best_key = k
                    best_len = len(v_str)
            if best_key:
                return self._format_output_as_text(output[best_key])
            return str(output)
        if isinstance(output, list):
            return '\n'.join([self._format_output_as_text(item) for item in output[:20]])
        return str(output)

    def _auto_inject_dependent_outputs(
        self,
        resolved: Dict[str, Any],
        outputs: Dict[str, Any],
        step
    ) -> Dict[str, Any]:
        """Inject dependent step outputs into params when template resolution missed them.

        If a step depends on previous steps but its resolved params don't contain
        any of the dependent output data, format the outputs and inject them into
        the first text-like parameter.
        """
        # Collect output keys from dependent steps
        dep_output_keys = []
        for dep_idx in step.depends_on:
            # Find the output_key for this dependency index
            # outputs dict uses output_key as key, so check which keys came from deps
            for key in outputs:
                dep_output_keys.append(key)

        if not dep_output_keys:
            return resolved

        # Check if any resolved param already contains dependent data
        resolved_str = str(resolved)
        has_dep_data = False
        for dep_key in dep_output_keys:
            dep_output = outputs.get(dep_key)
            if dep_output is None:
                continue
            # Check if the output data appears in resolved params (not just the key name)
            dep_text = self._format_output_as_text(dep_output)
            if len(dep_text) > 50 and dep_text[:50] in resolved_str:
                has_dep_data = True
                break

        if has_dep_data:
            return resolved  # Template resolution already injected the data

        # Format all dependent outputs into text
        dep_texts = []
        for dep_key in dep_output_keys:
            dep_output = outputs.get(dep_key)
            if dep_output is not None:
                formatted = self._format_output_as_text(dep_output)
                if formatted and len(formatted.strip()) > 10:
                    dep_texts.append(formatted)

        if not dep_texts:
            return resolved

        combined_text = '\n\n'.join(dep_texts)

        # Find the best param to inject into (text > content > input > query > first string param)
        inject_keys = ['text', 'content', 'input', 'query', 'prompt', 'data', 'source']
        injected = False
        for ik in inject_keys:
            if ik in resolved:
                current = resolved[ik]
                if isinstance(current, str) and len(current.strip()) < 100:
                    # Current value is a short placeholder - replace with actual data
                    resolved[ik] = combined_text
                    injected = True
                    logger.info(f"Auto-injected {len(combined_text)} chars from dependent steps into param '{ik}'")
                    break

        # If no known key found, try first string param
        if not injected:
            for k, v in resolved.items():
                if isinstance(v, str) and len(v.strip()) < 100 and k not in ('skill_name', 'tool_name', 'output_key'):
                    resolved[k] = combined_text
                    logger.info(f"Auto-injected {len(combined_text)} chars from dependent steps into param '{k}'")
                    break

        return resolved

    async def execute(self, task: str, **kwargs) -> ExecutionResult:
        """
        Execute a task automatically.

        Args:
            task: Task description (can be minimal like "RNN vs CNN")
            status_callback: Optional callback(stage, detail) for progress updates
            ensemble: Enable prompt ensembling for multi-perspective analysis
            ensemble_strategy: Strategy for ensembling:
                - 'self_consistency': Same prompt, N samples, synthesis
                - 'multi_perspective': Different expert personas (default)
                - 'gsa': Generative Self-Aggregation
                - 'debate': Multi-round argumentation

        Returns:
            ExecutionResult with outputs and status
        """
        start_time = datetime.now()

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
            logger.info(f"  🔧 {stage}" + (f": {detail}" if detail else ""))

        # Initialize
        self._init_dependencies()

        # Auto-detect ensemble for certain task types (if not explicitly set)
        if ensemble is None:
            ensemble = self._should_auto_ensemble(task)
            if ensemble:
                _status("Auto-ensemble", "enabled for analysis/comparison task")

        # Optional: Ensemble pre-analysis for enriched context
        ensemble_context = None
        enriched_task = task  # Start with original task
        if ensemble:
            _status("Ensembling", f"strategy={ensemble_strategy}")
            ensemble_result = await self._execute_ensemble(task, ensemble_strategy, _status)
            if ensemble_result.get('success'):
                ensemble_context = ensemble_result
                _status("Ensemble complete", f"{len(ensemble_result.get('perspectives_used', []))} perspectives")

                # ENRICH the task with ensemble synthesis as instructions
                synthesis = ensemble_result.get('response', '')
                if synthesis:
                    enriched_task = f"""{task}

[Multi-Perspective Analysis - Use these insights to guide your work]:
{synthesis[:3000]}"""
                    _status("Task enriched", "with multi-perspective synthesis")

                # Also pass context for skills that want raw perspectives
                kwargs['ensemble_context'] = ensemble_context

        _status("AutoAgent", f"starting task execution")

        # Step 1: Infer task type (use original task for classification)
        _status("Analyzing", "inferring task type")
        task_type = self._infer_task_type(task)
        _status("Task type", task_type.value)

        # Step 2: Discover skills (use original task for discovery)
        _status("Discovering", "finding relevant skills")
        all_skills = self._discover_skills(task)
        _status("Skills found", f"{len(all_skills)} potential skills")

        # Step 2.5: Select best skills using agentic planner
        # Use enriched_task so LLM has ensemble context for better selection
        if all_skills:
            _status("Selecting", "choosing best skills for task")
            skills, selection_reasoning = self.planner.select_skills(
                task=enriched_task,  # Use enriched task with ensemble insights
                available_skills=all_skills,
                max_skills=8
            )
            skill_names = [s.get('name') for s in skills]
            _status("Skills selected", ", ".join(skill_names[:5]) + ("..." if len(skill_names) > 5 else ""))
            logger.debug(f"Selection reasoning: {selection_reasoning}")

            # Step 2.6: Ensure skill dependencies are installed
            await self._ensure_skill_dependencies(skill_names, status_callback)
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
        # Use enriched_task so planner has ensemble insights for better execution plan
        _status("Planning", "creating execution plan")
        steps = self._plan_execution(enriched_task, task_type, skills)
        _status("Plan ready", f"{len(steps)} steps to execute")

        # Step 3.5: Validate tools exist before execution
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
                errors=["No valid tools found for planned steps"],
                execution_time=(datetime.now() - start_time).total_seconds()
            )

        # Step 4: Execute steps with parallel execution for independent steps
        outputs = {}
        errors = []
        skills_used = []
        steps_executed = 0
        replan_count = 0
        max_replans = 3  # Prevent infinite replanning loops
        excluded_skills = set()  # Skills that failed due to domain mismatch - exclude from replanning

        # Group steps by dependency level for parallel execution
        executed_indices = set()
        
        async def execute_step_with_error_handling(step_idx: int, step: ExecutionStep):
            """Execute a single step with error handling."""
            try:
                # Clean description for display (remove internal context)
                display_desc = _clean_for_display(step.description)[:80]
                _status(f"Step {step_idx+1}/{len(steps)}", f"{step.skill_name}: {display_desc}")
                _status(f"  Executing", f"{step.tool_name}")

                # Resolve params with current outputs
                resolved_params = self._resolve_params(step.params, outputs, step=step)

                # Execute
                result = await self._execute_tool(
                    step.skill_name,
                    step.tool_name,
                    resolved_params,
                    status_callback=status_callback
                )

                return step_idx, step, result, None
            except Exception as e:
                return step_idx, step, {'success': False, 'error': str(e)}, str(e)
        
        # Execute steps in dependency order with parallelization
        while len(executed_indices) < len(steps):
            # Find steps that can be executed now (dependencies satisfied or no dependencies)
            ready_steps = []
            for i, step in enumerate(steps):
                if i in executed_indices:
                    continue
                
                # Check if all dependencies are satisfied
                dependencies_satisfied = all(
                    dep_idx in executed_indices for dep_idx in step.depends_on
                )
                
                if dependencies_satisfied:
                    ready_steps.append((i, step))
            
            if not ready_steps:
                # No ready steps - check if we're stuck
                remaining = [i for i in range(len(steps)) if i not in executed_indices]
                if remaining:
                    _status("Fallback", f"executing {len(remaining)} remaining steps sequentially")
                    # Execute remaining steps sequentially as fallback
                    for i in remaining:
                        step = steps[i]
                        _status(f"Step {i+1}/{len(steps)}", f"{step.skill_name}: {step.description}")
                        _status(f"  Executing", f"{step.tool_name}")
                        resolved_params = self._resolve_params(step.params, outputs, step=step)
                        result = await self._execute_tool(
                            step.skill_name,
                            step.tool_name,
                            resolved_params,
                            status_callback=status_callback
                        )
                        if result.get('success'):
                            outputs[step.output_key or f'step_{i}'] = result
                            skills_used.append(step.skill_name)
                            steps_executed += 1
                            executed_indices.add(i)
                            _status(f"✓ Step {i+1}", f"{step.skill_name} succeeded")
                        else:
                            errors.append(f"Step {i+1} ({step.skill_name}): {result.get('error', 'Unknown error')}")
                            _status(f"✗ Step {i+1}", f"{step.skill_name} failed: {result.get('error', 'Unknown error')}")
                break
            
            # Execute ready steps in parallel
            if len(ready_steps) == 1:
                # Single step - execute directly
                i, step = ready_steps[0]
                _status(f"Step {i+1}/{len(steps)}", f"{step.skill_name}: {step.description}")
                _status(f"  Executing", f"{step.tool_name} with {len(step.params)} params")
                resolved_params = self._resolve_params(step.params, outputs, step=step)
                result = await self._execute_tool(
                    step.skill_name,
                    step.tool_name,
                    resolved_params,
                    status_callback=status_callback
                )
                results = [(i, step, result, None)]
            else:
                # Multiple steps - execute in parallel
                step_names = [s.skill_name for _, s in ready_steps[:3]]
                _status(f"Parallel execution", f"{len(ready_steps)} steps: {', '.join(step_names)}")
                tasks = [
                    execute_step_with_error_handling(i, step)
                    for i, step in ready_steps
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for result_item in results:
                if isinstance(result_item, Exception):
                    logger.error(f"Step execution exception: {result_item}")
                    continue
                
                step_idx, step, result, exception = result_item
                
                if result.get('success'):
                    outputs[step.output_key or f'step_{step_idx}'] = result
                    skills_used.append(step.skill_name)
                    steps_executed += 1
                    executed_indices.add(step_idx)

                    # Show success with output summary - Claude Code style
                    output_summary = ""
                    if isinstance(result, dict):
                        # Search results - show count
                        if 'results' in result and isinstance(result['results'], list):
                            count = len(result['results'])
                            query = result.get('query', step.description[:50])
                            _status(f"Search results", f"{query[:60]} → {count} results")
                            # Don't show generic success message for search
                            continue

                        # File outputs - show path
                        for key in ['pdf_path', 'md_path', 'output_path', 'file_path']:
                            if key in result and result[key]:
                                output_summary = f" → {result[key]}"
                                break

                        # Word count
                        if not output_summary and 'word_count' in result:
                            output_summary = f" → {result['word_count']} words"

                    _status(f"✓ Step {step_idx+1}", f"{step.skill_name} succeeded{output_summary}")
                else:
                    error_msg = result.get('error', 'Unknown error') or str(exception) if exception else 'Unknown error'
                    errors.append(f"Step {step_idx+1} ({step.skill_name}): {error_msg}")
                    _status(f"✗ Step {step_idx+1}", f"{step.skill_name} failed: {error_msg}")
                    
                    # CRITICAL: Always mark step as executed to prevent infinite loops
                    # Even if it failed, we don't want to retry it forever
                    executed_indices.add(step_idx)
                    
                    # Try alternative tools for network/timeout errors (any skill with web capability)
                    is_network_error = any(kw in error_msg.lower() for kw in ['network', 'timeout', 'request', 'connection'])
                    skill_has_web = any(kw in step.skill_name.lower() for kw in ['web', 'search', 'http', 'api', 'fetch'])
                    if is_network_error and skill_has_web:
                        logger.info(f"  🔄 Trying alternative tools for network error...")
                        resolved_params = self._resolve_params(step.params, outputs, step=step)
                        alternative_result = await self._try_alternative_web_tools(step, resolved_params)
                        if alternative_result and alternative_result.get('success'):
                            outputs[step.output_key or f'step_{step_idx}'] = alternative_result
                            skills_used.append(step.skill_name)
                            steps_executed += 1
                            executed_indices.add(step_idx)
                            logger.info(f"  ✅ Step {step_idx+1} succeeded with alternative tool")
                            continue
                    
                    # Note: step_idx is already marked as executed above
                    # For optional steps, just continue; for required steps, consider replanning
                    if step.optional:
                        continue  # Skip optional failed steps (already marked as executed)

                    # Detect domain-mismatch errors (skill used for wrong task type)
                    # These indicate the LLM selected an inappropriate skill
                    domain_mismatch_keywords = [
                        '404', 'not found', 'delisted', 'no data found',
                        'quote not found', 'symbol may be delisted',
                        'invalid ticker', 'unknown symbol', 'no price data',
                        'division by zero', 'float division',  # Data processing errors from wrong input
                    ]
                    is_domain_mismatch = any(kw in error_msg.lower() for kw in domain_mismatch_keywords)

                    if is_domain_mismatch:
                        # Skill is inappropriate for this task - exclude from future replanning
                        excluded_skills.add(step.skill_name)
                        logger.info(f"  🚫 Excluding skill '{step.skill_name}' from replanning (domain mismatch)")

                    if step_idx < len(steps) - 1 and replan_count < max_replans:
                        # Check if tool exists - if not, don't replan
                        if 'Tool not found' in error_msg or 'Skill not found' in error_msg:
                            logger.warning(f"  ⚠️  Tool/skill not found, skipping replan")
                            continue  # Already marked as executed above

                        # Try to replan remaining steps with excluded skills filtered out
                        logger.info(f"  🔄 Attempting to replan remaining steps")
                        replan_count += 1
                        try:
                            # Filter out excluded skills before replanning
                            available_skills = [s for s in skills if s.get('name') not in excluded_skills]
                            if not available_skills:
                                logger.warning(f"  ⚠️  No skills remaining after exclusions")
                                continue

                            remaining_steps, _ = self.planner.plan_execution(
                                task=task,
                                task_type=task_type,
                                skills=available_skills,  # Use filtered skills
                                previous_outputs=outputs,
                                max_steps=self.max_steps - len(executed_indices)
                            )
                            remaining_steps = self._validate_and_filter_steps(remaining_steps, available_skills)
                            if remaining_steps:
                                # Replace remaining steps
                                steps = steps[:step_idx+1] + remaining_steps
                                logger.info(f"  ✅ Replanned {len(remaining_steps)} remaining steps (excluded: {excluded_skills})")
                                break  # Restart execution loop with new steps
                            else:
                                logger.warning(f"  ⚠️  Replanning produced no valid steps")
                                # Already marked as executed above
                        except Exception as e:
                            logger.warning(f"  ⚠️  Replanning failed: {e}")
                            # Already marked as executed above
                    elif replan_count >= max_replans:
                        logger.warning(f"  ⚠️  Max replans ({max_replans}) reached")
                        # Already marked as executed above

        # Determine final output: use last step's output
        final_output = None
        if outputs:
            final_output = list(outputs.values())[-1]

        execution_time = (datetime.now() - start_time).total_seconds()

        # Use original task (not enriched) for result display
        # The enriched_task contains internal context not meant for user display
        display_task = _clean_for_display(task)

        return ExecutionResult(
            success=steps_executed > 0,
            task=display_task,
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

        # Send output (if output skill configured)
        output_skill = output_skill or self.default_output_skill
        if not output_skill:
            return result

        skill = self._registry.get_skill(output_skill)
        if not skill or not skill.tools:
            return result

        # Find a send/message tool (look for common patterns in tool names)
        send_tool = None
        for tool_name, tool_func in skill.tools.items():
            if any(kw in tool_name.lower() for kw in ['send', 'message', 'post', 'notify']):
                send_tool = tool_func
                break

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
