"""
Agentic Planner - Fully LLM-based planning (no hardcoded logic)

Replaces all rule-based planning with agentic LLM decisions.
No keyword matching, no hardcoded flows, fully adaptive.

Supports both:
- Raw string tasks (simple planning)
- TaskGraph tasks (structured planning with metadata)
"""

import json
import logging
import asyncio
import traceback
from typing import Dict, Any, List, Optional, Union, TYPE_CHECKING
from dataclasses import dataclass, field

try:
    from pydantic import BaseModel, Field
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = None

try:
    import dspy
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False


# Pydantic model for typed DSPy output - accepts common field name variations
if PYDANTIC_AVAILABLE:
    from pydantic import field_validator, model_validator

    class ExecutionStepSchema(BaseModel):
        """Schema for execution plan steps - accepts common LLM field name variations."""
        skill_name: str = Field(default="", description="Skill name from available_skills")
        tool_name: str = Field(default="", description="Tool name from that skill's tools list")
        params: Dict[str, Any] = Field(default_factory=dict, description="Tool parameters")
        description: str = Field(default="", description="What this step does")
        depends_on: List[int] = Field(default_factory=list, description="Indices of steps this depends on")
        output_key: str = Field(default="", description="Key to store output under")
        optional: bool = Field(default=False, description="Whether step is optional")

        class Config:
            extra = "allow"  # Allow extra fields from LLM

        @model_validator(mode='before')
        @classmethod
        def normalize_field_names(cls, data: Dict[str, Any]) -> Dict[str, Any]:
            """Normalize common LLM field name variations to expected names."""
            if not isinstance(data, dict):
                return data

            # skill_name aliases: skill, skill_name, skills_used (first item)
            if 'skill_name' not in data or not data.get('skill_name'):
                skill = data.get('skill', '')
                if not skill:
                    # Try skills_used array
                    skills_used = data.get('skills_used', [])
                    if skills_used and isinstance(skills_used, list):
                        skill = skills_used[0]
                data['skill_name'] = skill

            # tool_name aliases: tool, tool_name, tools_used (first item), action
            if 'tool_name' not in data or not data.get('tool_name'):
                tool = data.get('tool', '')
                if not tool:
                    # Try tools_used array
                    tools_used = data.get('tools_used', [])
                    if tools_used and isinstance(tools_used, list):
                        tool = tools_used[0]
                if not tool:
                    # Extract from action like "use write_file_tool to..."
                    action = data.get('action', '')
                    if action:
                        import re
                        tool_match = re.search(r'\b([a-z_]+_tool)\b', action)
                        if tool_match:
                            tool = tool_match.group(1)
                data['tool_name'] = tool

            # params aliases: parameters, params, tool_input, input, inputs, tool_params
            if 'params' not in data or not data.get('params'):
                data['params'] = (
                    data.get('parameters') or
                    data.get('tool_input') or
                    data.get('tool_params') or
                    data.get('inputs') or  # LLM often uses 'inputs' plural
                    data.get('input') or
                    {}
                )

            return data

# Import context utilities for error handling and compression
try:
    from ..utils.context_utils import (
        ContextCompressor,
        ErrorDetector,
        ErrorType,
        ExecutionTrajectory,
        detect_error_type,
    )
    CONTEXT_UTILS_AVAILABLE = True
except ImportError:
    CONTEXT_UTILS_AVAILABLE = False
    ContextCompressor = None
    ErrorDetector = None
    ErrorType = None

# Avoid circular import - use TYPE_CHECKING for type hints
if TYPE_CHECKING:
    from .auto_agent import ExecutionStep, TaskType, ExecutionResult

logger = logging.getLogger(__name__)

# Import TaskGraph if available (for enhanced planning)
try:
    from ..autonomous.intent_parser import TaskGraph
    TASK_GRAPH_AVAILABLE = True
except ImportError:
    TASK_GRAPH_AVAILABLE = False
    TaskGraph = None

# Import ExecutionStep and TaskType at runtime (after module initialization)
def _get_execution_step():
    """Lazy import to avoid circular dependency."""
    from .auto_agent import ExecutionStep
    return ExecutionStep

def _get_task_type():
    """Lazy import to avoid circular dependency."""
    from .auto_agent import TaskType
    return TaskType


# =============================================================================
# DSPy Signatures for Agentic Planning
# =============================================================================

class TaskTypeInferenceSignature(dspy.Signature):
    """Classify the task type from description.

    You are a CLASSIFIER. Analyze the task description and classify it into one of these types:

    CLASSIFICATION GUIDE:
    - creation: Create files, build apps, write code, generate content, make something new
      Examples: "Create a Python file", "Build a todo app", "Write a UI component"
    - research: Search web, find information, discover facts, investigate topics
      Examples: "Research best practices", "Find documentation", "Search for tutorials"
    - comparison: Compare options, vs analysis, evaluate alternatives
      Examples: "Compare React vs Vue", "Which database is better"
    - analysis: Analyze data, evaluate code, review content, assess quality
      Examples: "Analyze this code", "Review the architecture", "Evaluate performance"
    - communication: Send messages, notify, email, communicate with users
      Examples: "Send an email", "Notify the team", "Post an update"
    - automation: Automate workflows, schedule tasks, set up pipelines
      Examples: "Automate deployment", "Schedule backups", "Set up CI/CD"
    - unknown: Only if none of the above clearly fit

    IMPORTANT: Default to 'creation' for any task involving building, creating, or writing code/files.
    """
    task_description: str = dspy.InputField(desc="The task description to classify")

    task_type: str = dspy.OutputField(
        desc="Output EXACTLY one word: creation, research, comparison, analysis, communication, automation, or unknown. If task mentions 'create', 'build', 'write', or 'make', output 'creation'."
    )
    reasoning: str = dspy.OutputField(
        desc="Brief 1-2 sentence explanation of why this type was chosen."
    )
    confidence: float = dspy.OutputField(
        desc="A number between 0.0 and 1.0 indicating confidence."
    )


class ExecutionPlanningSignature(dspy.Signature):
    """Create executable plan with PRODUCTION-QUALITY, WORLD-CLASS code.

    YOU ARE A SENIOR ENGINEER AT A TOP TECH COMPANY.
    Generate code that would pass a rigorous code review.

    QUALITY STANDARDS:
    1. ARCHITECTURE: Clean separation of concerns, dependency injection, interfaces
    2. TYPE SAFETY: Full type hints, dataclasses/Pydantic models, enums for constants
    3. ERROR HANDLING: Graceful degradation, informative errors, logging
    4. UI/UX EXCELLENCE:
       - Terminal: Use 'rich' library for panels, tables, progress, live updates
       - Colors: Consistent theme, accessibility-friendly contrast
       - Interaction: Keyboard shortcuts, vim keybindings, smooth animations
    5. DOCUMENTATION: Docstrings, usage examples, inline comments for complex logic
    6. TESTABILITY: Pure functions, mockable dependencies, clear interfaces

    CONSOLIDATION REQUIREMENT:
    - If task mentions multiple similar apps (Jira, Asana, Any.do, Trello, etc.)
    - Create ONE UNIFIED app combining the BEST features of all
    - NOT separate implementations - a single comprehensive solution

    CODE COMPLETENESS:
    - Generate FULL, COMPLETE, WORKING code (300-500+ lines for apps)
    - NO placeholders, NO truncation, NO "..." or "[code here]"
    - Include ALL imports, ALL classes, ALL methods, main() entry point

    EXAMPLE OUTPUT QUALITY (for task management app):
    - Rich terminal UI with panels, tables, and colors
    - Kanban board view + List view + Calendar view
    - Keyboard navigation (j/k/h/l for vim users)
    - Data persistence with JSON/SQLite
    - Undo/redo support
    - Search and filtering
    - Export to multiple formats
    """

    task_description: str = dspy.InputField(desc="Task to accomplish")
    task_type: str = dspy.InputField(desc="Task type: research, analysis, creation, etc.")
    available_skills: str = dspy.InputField(desc="JSON array of skills with tools and their required params")
    previous_outputs: str = dspy.InputField(desc="JSON dict of outputs from previous steps")
    max_steps: int = dspy.InputField(desc="Maximum number of steps")

    # Use typed Pydantic model - DSPy JSONAdapter enforces schema
    if PYDANTIC_AVAILABLE:
        execution_plan: List[ExecutionStepSchema] = dspy.OutputField(
            desc="List of execution steps"
        )
    else:
        execution_plan: List[dict] = dspy.OutputField(
            desc='Steps array. Each: {"skill_name": "...", "tool_name": "...", "params": {...}, "description": "..."}'
        )
    reasoning: str = dspy.OutputField(desc="Brief explanation of the plan")
    estimated_complexity: str = dspy.OutputField(desc="simple, medium, or complex")


class SkillSelectionSignature(dspy.Signature):
    """Select the BEST skills needed to complete the task.

    You are a SKILL SELECTOR. Analyze the task description to identify required capabilities.

    CRITICAL: GENERATION vs RESEARCH distinction:
    - GENERATION tasks: User wants LLM to CREATE content from its knowledge
    - RESEARCH tasks: User wants to FIND/SEARCH for external information

    CRITICAL MATCHING RULES (in priority order):
    1. "checklist" / "todo" / "list of steps" / "compliance list" → GENERATION task
       → Use claude-cli-llm (LLM generates the checklist from knowledge)
       → Do NOT use research-to-pdf (that searches web, creates research reports)

    2. "create" / "generate" / "write" / "draft" content → GENERATION task
       → Use claude-cli-llm for content generation
       → Use docx-tools or file-operations for saving to file

    3. "research" / "search" / "find" / "look up" / "what's new" → RESEARCH task
       → Use web-search, research-to-pdf, last30days-claude-cli

    4. "convert" / "transform" format → use document-converter

    COMMON MISTAKES TO AVOID:
    - "Create checklist for X framework" = GENERATION (LLM knows frameworks)
      → Use claude-cli-llm, NOT research-to-pdf
    - "Research latest news about X" = RESEARCH (needs web search)
      → Use research-to-pdf or web-search
    - "Create report about X" = Usually GENERATION unless "research" is mentioned
      → Use claude-cli-llm

    Guidelines:
    1. Default to GENERATION (claude-cli-llm) unless task explicitly mentions search/research/find
    2. Match task verbs to skill capabilities
    3. Prefer simpler skills over complex multi-step skills

    You are NOT executing anything. You are ONLY selecting which skills are needed.
    """
    task_description: str = dspy.InputField(desc="The task to analyze - identify ALL required capabilities")
    available_skills: str = dspy.InputField(
        desc="JSON list of all available skills with their descriptions and tools"
    )
    max_skills: int = dspy.InputField(
        desc="Maximum number of skills to select"
    )

    selected_skills: str = dspy.OutputField(
        desc='Return ONLY a JSON array like ["skill-name-1", "skill-name-2"]. No markdown, no explanation, just the JSON array.'
    )
    reasoning: str = dspy.OutputField(
        desc="Explain what capabilities the task needs and which skill provides each"
    )
    skill_priorities: str = dspy.OutputField(
        desc="JSON dict mapping skill names to priority (0.0-1.0). Higher = execute earlier. Order by logical workflow."
    )


# =============================================================================
# Agentic Planner
# =============================================================================

class AgenticPlanner:
    """
    Fully agentic planner - no hardcoded logic.

    All planning decisions made by LLM:
    - Task type inference (semantic, not keyword matching)
    - Skill selection (capability-based matching)
    - Execution planning (adaptive, context-aware)
    - Dependency resolution (intelligent)
    """

    # Global semaphore to limit concurrent LLM calls (prevents rate limiting)
    _llm_semaphore = None
    _max_concurrent_llm_calls = 1  # Serialize LLM calls by default

    @classmethod
    def set_max_concurrent_llm_calls(cls, max_calls: int):
        """Set maximum concurrent LLM calls across all planner instances."""
        cls._max_concurrent_llm_calls = max(1, max_calls)
        cls._llm_semaphore = None  # Reset to recreate with new limit

    @classmethod
    def _get_semaphore(cls):
        """Get or create the global LLM semaphore."""
        if cls._llm_semaphore is None:
            import threading
            cls._llm_semaphore = threading.Semaphore(cls._max_concurrent_llm_calls)
        return cls._llm_semaphore

    def __init__(self):
        """Initialize agentic planner."""
        if not DSPY_AVAILABLE:
            raise RuntimeError("DSPy required for AgenticPlanner")

        # Use Predict for execution planning with JSON output enforcement via prompt
        self.execution_planner = dspy.Predict(ExecutionPlanningSignature)
        self._use_typed_predictor = False

        self.task_type_inferrer = dspy.ChainOfThought(TaskTypeInferenceSignature)
        self.skill_selector = dspy.ChainOfThought(SkillSelectionSignature)

        # Store signatures for JSON schema extraction
        self._signatures = {
            'task_type': TaskTypeInferenceSignature,
            'execution': ExecutionPlanningSignature,
            'skill_selection': SkillSelectionSignature,
        }

        # Context compression for handling context length errors
        self._compressor = ContextCompressor() if CONTEXT_UTILS_AVAILABLE else None
        self._max_compression_retries = 3

        logger.info("🧠 AgenticPlanner initialized (fully LLM-based, no hardcoded logic)")

    def _call_with_retry(
        self,
        module,
        kwargs: Dict[str, Any],
        compressible_fields: Optional[List[str]] = None,
        max_retries: int = 5
    ):
        """
        Call a DSPy module with automatic retry and context compression.

        Learned from BaseSwarmAgent pattern:
        - Detect error types (context length, timeout, parse, rate limit)
        - Compress context on context length errors
        - Exponential backoff on timeouts
        - Wait on rate limits (uses global semaphore to serialize calls)
        - Preserve trajectory/progress

        Args:
            module: DSPy module to call
            kwargs: Arguments to pass to module
            compressible_fields: Fields that can be compressed (e.g., 'available_skills')
            max_retries: Maximum retry attempts (default 5 for rate limit resilience)

        Returns:
            Module result or raises exception
        """
        import time

        compression_ratio = 0.7
        last_error = None
        semaphore = self._get_semaphore()

        for attempt in range(max_retries + 1):
            try:
                if attempt > 0:
                    logger.info(f"   Retry {attempt}/{max_retries}")

                # Use semaphore to serialize LLM calls (prevents rate limiting)
                with semaphore:
                    return module(**kwargs)

            except Exception as e:
                last_error = e

                # Detect error type
                if CONTEXT_UTILS_AVAILABLE:
                    error_type, strategy = detect_error_type(e)
                    logger.debug(f"   Error type detected: {error_type.value}")
                else:
                    # Fallback detection
                    error_str = str(e).lower()
                    error_type_str = type(e).__name__.lower()

                    if any(p in error_str for p in ['context', 'token', 'too long']):
                        error_type = 'context_length'
                        strategy = {'should_retry': True, 'action': 'compress'}
                    elif any(p in error_str for p in ['rate limit', 'rate_limit', 'ratelimit', 'too many requests', '429']) or 'ratelimit' in error_type_str:
                        # Rate limit error - wait longer before retry
                        error_type = 'rate_limit'
                        # Extract wait time from error message if available (e.g., "Try again in 60 seconds")
                        import re
                        wait_match = re.search(r'(\d+)\s*seconds?', error_str)
                        wait_time = int(wait_match.group(1)) if wait_match else 60
                        strategy = {'should_retry': True, 'action': 'wait', 'delay_seconds': wait_time}
                        logger.warning(f"Rate limit hit, will wait {wait_time}s before retry")
                    elif 'timeout' in error_str:
                        error_type = 'timeout'
                        strategy = {'should_retry': True, 'action': 'backoff', 'delay_seconds': 2}
                    else:
                        error_type = 'unknown'
                        strategy = {'should_retry': False}

                if not strategy.get('should_retry') or attempt >= max_retries:
                    raise

                action = strategy.get('action', 'fail')

                if action == 'compress' and compressible_fields and self._compressor:
                    # Compress specified fields
                    for field in compressible_fields:
                        if field in kwargs and kwargs[field]:
                            original = kwargs[field]
                            if isinstance(original, str) and len(original) > 1000:
                                result = self._compressor.compress(
                                    original,
                                    target_ratio=compression_ratio
                                )
                                kwargs[field] = result.content
                                logger.info(f"   Compressed {field}: {result.original_length} → {result.compressed_length} chars")

                    compression_ratio *= 0.7  # More aggressive next time

                elif action == 'backoff':
                    delay = strategy.get('delay_seconds', 1) * (2 ** attempt)
                    logger.info(f"   Backing off for {delay}s...")
                    time.sleep(min(delay, 30))  # Cap at 30s

                elif action == 'wait':
                    delay = strategy.get('delay_seconds', 30)
                    logger.info(f"   Rate limited, waiting {delay}s...")
                    time.sleep(delay)

        # Should not reach here, but just in case
        if last_error:
            raise last_error
        raise RuntimeError("Unexpected state in retry logic")
    
    def infer_task_type(self, task: str):
        """
        Infer task type using LLM semantic understanding.

        Args:
            task: Task description

        Returns:
            (TaskType, reasoning, confidence)
        """
        TaskType = _get_task_type()

        try:
            import dspy
            import asyncio
            import re

            # Prepare task for inference (preserves full context)
            task_for_inference = self._abstract_task_for_planning(task)

            # Check async context and call with appropriate LM binding
            try:
                asyncio.get_running_loop()
                lm = dspy.settings.lm
                with dspy.context(lm=lm):
                    result = self.task_type_inferrer(task_description=task_for_inference)
            except RuntimeError:
                # Not in async context — direct call
                result = self.task_type_inferrer(task_description=task_for_inference)

            # Parse task_type field
            task_type_str = str(result.task_type).lower().strip().split()[0] if result.task_type else 'unknown'

            task_type_map = {
                'research': TaskType.RESEARCH,
                'comparison': TaskType.COMPARISON,
                'creation': TaskType.CREATION,
                'communication': TaskType.COMMUNICATION,
                'analysis': TaskType.ANALYSIS,
                'automation': TaskType.AUTOMATION,
            }
            task_type = task_type_map.get(task_type_str, TaskType.UNKNOWN)

            # Override if LLM returned unknown but keywords clearly indicate creation
            if task_type == TaskType.UNKNOWN:
                task_lower = task.lower()
                if any(w in task_lower for w in ['create', 'build', 'make', 'write', 'generate', 'implement']):
                    task_type = TaskType.CREATION
                    logger.info(f"Overriding 'unknown' to 'creation' based on keywords")

            # Parse confidence
            try:
                confidence_match = re.search(r'(\d+\.?\d*)', str(result.confidence))
                confidence = float(confidence_match.group(1)) if confidence_match else 0.7
                confidence = max(0.0, min(1.0, confidence))
            except (ValueError, TypeError, AttributeError):
                confidence = 0.7

            reasoning = str(result.reasoning).strip() if result.reasoning else f"Inferred as {task_type_str}"

            logger.info(f"📋 Task type inferred: {task_type.value} (confidence: {confidence:.2f})")
            return task_type, reasoning, confidence

        except Exception as e:
            logger.warning(f"Task type inference failed: {e}, using keyword fallback")

            # Minimal keyword fallback
            task_lower = task.lower()
            if any(w in task_lower for w in ['compare', 'vs', 'versus', 'comparison']):
                return TaskType.COMPARISON, "Keyword fallback: comparison task", 0.6
            elif any(w in task_lower for w in ['research', 'find', 'search', 'discover']):
                return TaskType.RESEARCH, "Keyword fallback: research task", 0.6
            elif any(w in task_lower for w in ['create', 'generate', 'make', 'build']):
                return TaskType.CREATION, "Keyword fallback: creation task", 0.6
            elif any(w in task_lower for w in ['analyze', 'analysis', 'evaluate']):
                return TaskType.ANALYSIS, "Keyword fallback: analysis task", 0.6
            return TaskType.UNKNOWN, f"Inference failed: {str(e)[:100]}", 0.3

    # Skills that depend on external services that may be unreliable
    # These are deprioritized in skill selection to prefer more reliable alternatives
    DEPRIORITIZED_SKILLS = {
        'search-to-justjot-idea',      # JustJot API issues
        'mcp-justjot-mcp-client',      # MCP timeout issues
        'mcp-justjot',                 # MCP timeout issues
        'justjot-mcp-http',            # JustJot API issues
        'notion-research-documentation',  # Requires Notion API setup
        'reddit-trending-to-justjot',  # JustJot API issues
        'notebooklm-pdf',              # Requires browser sign-in
        'oauth-automation',            # Requires browser interaction
    }

    def select_skills(
        self,
        task: str,
        available_skills: List[Dict[str, Any]],
        max_skills: int = 8
    ) -> tuple[List[Dict[str, Any]], str]:
        """
        Select best skills for task using LLM semantic matching.

        Falls back to using first available skills if LLM fails.
        Deprioritizes skills that depend on unreliable external services.

        Args:
            task: Task description
            available_skills: List of available skills
            max_skills: Maximum skills to select

        Returns:
            (selected_skills, reasoning)
        """
        if not available_skills:
            return [], "No skills available"

        # Filter out deprioritized skills (move to end of list)
        reliable_skills = [s for s in available_skills if s.get('name') not in self.DEPRIORITIZED_SKILLS]
        deprioritized = [s for s in available_skills if s.get('name') in self.DEPRIORITIZED_SKILLS]
        available_skills = reliable_skills + deprioritized  # Reliable first

        if deprioritized:
            logger.debug(f"Deprioritized {len(deprioritized)} unreliable skills: {[s.get('name') for s in deprioritized]}")

        llm_selected_names = []
        llm_reasoning = ""

        # Try LLM selection
        try:
            skills_json = json.dumps([
                {
                    'name': s.get('name', ''),
                    'description': s.get('description', ''),
                    'tools': s.get('tools', [])
                }
                for s in available_skills[:50]
            ], indent=2)

            import dspy

            # Call skill selector with retry and context compression
            selector_kwargs = {
                'task_description': task,
                'available_skills': skills_json,
                'max_skills': max_skills
            }

            result = self._call_with_retry(
                module=self.skill_selector,
                kwargs=selector_kwargs,
                compressible_fields=['available_skills'],
                max_retries=self._max_compression_retries
            )

            # Parse selected skills
            try:
                selected_skills_str = str(result.selected_skills).strip()
                if selected_skills_str.startswith('```'):
                    import re
                    json_match = re.search(r'```(?:json)?\s*\n(.*?)\n```', selected_skills_str, re.DOTALL)
                    if json_match:
                        selected_skills_str = json_match.group(1).strip()

                llm_selected_names = json.loads(selected_skills_str)
                if not isinstance(llm_selected_names, list):
                    llm_selected_names = [llm_selected_names]
            except (json.JSONDecodeError, ValueError):
                llm_selected_names = self._extract_skill_names_from_text(result.selected_skills)

            llm_reasoning = result.reasoning or "LLM semantic matching"

            # Parse skill priorities for ordering
            try:
                priorities_str = str(result.skill_priorities).strip()
                if priorities_str.startswith('{'):
                    skill_priorities = json.loads(priorities_str)
                else:
                    skill_priorities = {}
            except (json.JSONDecodeError, ValueError):
                skill_priorities = {}

            logger.info(f"LLM selected {len(llm_selected_names)} skills: {llm_selected_names}")
            if skill_priorities:
                logger.info(f"Skill priorities: {skill_priorities}")

        except Exception as e:
            logger.warning(f"LLM selection failed: {e}")
            skill_priorities = {}

        # Build final selection
        if llm_selected_names:
            final_names = list(set(llm_selected_names))[:max_skills]
            reasoning = llm_reasoning
        else:
            # Smart fallback: match skills based on task keywords
            task_lower = task.lower()
            matched_skills = []

            # Priority keyword mappings for common tasks
            keyword_skill_map = {
                ('file', 'create', 'write', 'save'): 'file-operations',
                ('generate', 'llm', 'text', 'content'): 'claude-cli-llm',
                ('search', 'web', 'find', 'lookup'): 'web-search',
                ('terminal', 'shell', 'command', 'run'): 'terminal',
                ('research', 'report'): 'research-to-pdf',
                ('image', 'picture', 'photo'): 'image-generator',
                ('calculate', 'math', 'compute'): 'calculator',
            }

            for keywords, skill_name in keyword_skill_map.items():
                if any(kw in task_lower for kw in keywords):
                    # Check if this skill is available
                    for s in available_skills:
                        if s.get('name') == skill_name:
                            matched_skills.append(skill_name)
                            break

            if matched_skills:
                final_names = matched_skills[:max_skills]
                reasoning = f"Keyword-matched fallback: {matched_skills}"
                logger.info(f"Keyword fallback matched: {matched_skills}")
            else:
                # Last resort: use first available skills
                final_names = [s.get('name') for s in available_skills[:max_skills]]
                reasoning = "Fallback: using first available skills"

        # Filter to available skills
        selected_skills = [s for s in available_skills if s.get('name') in final_names]

        if not selected_skills and available_skills:
            selected_skills = available_skills[:max_skills]

        # Order skills by LLM-assigned priorities (no hardcoded flow order)
        def get_skill_order(skill):
            # Use LLM priority (higher priority = earlier execution)
            priority = skill_priorities.get(skill.get('name'), 0.5)
            return -priority  # Negate so higher priority comes first

        selected_skills = sorted(selected_skills, key=get_skill_order)
        logger.info(f"Ordered skills: {[s.get('name') for s in selected_skills]}")

        # Enrich skills with tools from registry
        selected_skills = self._enrich_skills_with_tools(selected_skills)

        selected_skills = selected_skills[:max_skills]

        logger.info(f"Selected {len(selected_skills)} skills: {[s.get('name') for s in selected_skills]}")
        return selected_skills, reasoning

    def _enrich_skills_with_tools(self, selected_skills: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enrich skill dicts with tool names and descriptions from registry."""
        try:
            from ..registry.skills_registry import get_skills_registry
            registry = get_skills_registry()
            if not registry:
                return selected_skills

            enriched = []
            for skill_dict in selected_skills:
                skill_name = skill_dict.get('name')
                skill_obj = registry.get_skill(skill_name)
                if skill_obj:
                    enriched_skill = skill_dict.copy()
                    if skill_obj.tools:
                        enriched_skill['tools'] = list(skill_obj.tools.keys())
                    else:
                        enriched_skill['tools'] = []
                    if not enriched_skill.get('description') and skill_obj.description:
                        enriched_skill['description'] = skill_obj.description
                    enriched.append(enriched_skill)
                else:
                    enriched.append(skill_dict)
            return enriched
        except Exception as e:
            logger.warning(f"Could not enrich skills: {e}")
            return selected_skills
    
    def plan_execution(
        self,
        task: str,
        task_type,
        skills: List[Dict[str, Any]],
        previous_outputs: Optional[Dict[str, Any]] = None,
        max_steps: int = 10
    ):
        """
        Plan execution steps using LLM reasoning.

        Args:
            task: Task description
            task_type: Inferred task type
            skills: Available skills (if empty, uses default file-operations)
            previous_outputs: Outputs from previous steps
            max_steps: Maximum steps

        Returns:
            (execution_steps, reasoning)
        """
        try:
            # If no skills provided, add default file-operations skill for creation tasks
            if not skills:
                task_type_value = task_type.value if hasattr(task_type, 'value') else str(task_type)
                if task_type_value in ['creation', 'unknown']:
                    logger.info("No skills provided, adding default file-operations skill")
                    skills = [{
                        'name': 'file-operations',
                        'description': 'Create, read, write files',
                        'tools': [
                            {'name': 'write_file_tool', 'params': {'path': 'string', 'content': 'string'}},
                            {'name': 'read_file_tool', 'params': {'path': 'string'}}
                        ]
                    }]
                else:
                    logger.warning(f"No skills available for task type '{task_type_value}'")
                    return [], f"No skills available for task type '{task_type_value}'"

            # Format skills for LLM WITH TOOL SCHEMAS
            # CRITICAL: Include parameter schemas so LLM knows what parameters each tool needs
            formatted_skills = []
            for s in skills:
                skill_dict = {
                    'name': s.get('name', ''),
                    'description': s.get('description', ''),
                    'tools': []
                }
                
                # Get tool names
                # Tools can be: list of strings, list of dicts, or dict
                tools_raw = s.get('tools', [])
                if isinstance(tools_raw, dict):
                    tool_names = list(tools_raw.keys())
                elif isinstance(tools_raw, list):
                    # Extract names if it's a list of dicts, otherwise use as-is (list of strings)
                    tool_names = [t.get('name') if isinstance(t, dict) else t for t in tools_raw]
                else:
                    tool_names = []
                
                # Enrich with tool schemas from registry
                try:
                    from ..registry.skills_registry import get_skills_registry
                    registry = get_skills_registry()
                    if registry:
                        skill_obj = registry.get_skill(skill_dict['name'])
                        if skill_obj and hasattr(skill_obj, 'tools') and skill_obj.tools:
                            for tool_name in tool_names:
                                tool_func = skill_obj.tools.get(tool_name)
                                if tool_func:
                                    # Extract parameter schema from docstring
                                    tool_schema = self._extract_tool_schema(tool_func, tool_name)
                                    skill_dict['tools'].append(tool_schema)
                                else:
                                    # Fallback: just name if tool not found
                                    skill_dict['tools'].append({'name': tool_name})
                        else:
                            # Fallback: just names if registry lookup fails
                            skill_dict['tools'] = [{'name': name} for name in tool_names]
                    else:
                        skill_dict['tools'] = [{'name': name} for name in tool_names]
                except Exception as e:
                    logger.debug(f"Could not enrich tool schemas for {skill_dict['name']}: {e}")
                    # Fallback: just names
                    skill_dict['tools'] = [{'name': name} for name in tool_names]
                
                formatted_skills.append(skill_dict)
            
            skills_json = json.dumps(formatted_skills, indent=2)
            
            # Format previous outputs
            outputs_json = json.dumps(previous_outputs or {}, indent=2)
            
            # Execute planning - signature is already baked into the module
            # No need to set it globally (which causes async task errors)
            import dspy
            
            # Abstract the task description to avoid LLM confusion
            # Clean task for planning (remove injected context)
            abstracted_task = self._abstract_task_for_planning(task)
            logger.debug(f"🔍 Task: '{abstracted_task[:80]}'")

            logger.info(f"📤 Calling LLM for execution plan...")
            logger.debug(f"   Task: {abstracted_task[:100]}")
            logger.debug(f"   Skills count: {len(skills)}")

            # Call execution planner with retry and context compression
            planner_kwargs = {
                'task_description': abstracted_task,
                'task_type': task_type.value,
                'available_skills': skills_json,
                'previous_outputs': outputs_json,
                'max_steps': max_steps,
                'config': {"response_format": {"type": "json_object"}}
            }

            result = self._call_with_retry(
                module=self.execution_planner,
                kwargs=planner_kwargs,
                compressible_fields=['available_skills', 'previous_outputs'],
                max_retries=self._max_compression_retries
            )

            # Debug: Log raw LLM response
            logger.info(f"📥 LLM response received")
            raw_plan = getattr(result, 'execution_plan', None)
            logger.debug(f"   Raw execution_plan type: {type(raw_plan)}")
            logger.debug(f"   Raw execution_plan (first 500 chars): {str(raw_plan)[:500] if raw_plan else 'NONE'}")

            # Parse execution plan - with JSONAdapter, this should be a list already
            plan_data = None

            # Method 0: Already a list (JSONAdapter working correctly)
            if isinstance(raw_plan, list):
                plan_data = raw_plan
                logger.info(f"   JSONAdapter returned list: {len(plan_data)} steps")
                # Log first step to debug skill_name issue
                if plan_data:
                    first_step = plan_data[0]
                    # Check if it's a Pydantic model or dict
                    if hasattr(first_step, 'skill_name'):
                        logger.info(f"   First step: skill_name='{first_step.skill_name}', tool_name='{first_step.tool_name}'")
                    elif isinstance(first_step, dict):
                        logger.info(f"   First step dict keys: {list(first_step.keys())}")
                        logger.info(f"   skill_name='{first_step.get('skill_name', '')}', skill='{first_step.get('skill', '')}'")
                    else:
                        logger.info(f"   First step type: {type(first_step)}")
            elif not raw_plan:
                logger.warning("LLM returned empty execution_plan field")
                plan_data = []
            else:
                # Fallback: String parsing for backwards compatibility
                import re
                plan_str = str(raw_plan).strip()

                # Method 1: Direct JSON parse (if LLM followed instructions)
                if plan_str.startswith('['):
                    try:
                        plan_data = json.loads(plan_str)
                        logger.info(f"   Direct JSON parse successful: {len(plan_data)} steps")
                    except json.JSONDecodeError:
                        pass

                # Method 2: Extract from markdown code block
                if plan_data is None and '```' in plan_str:
                    json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', plan_str, re.DOTALL)
                    if json_match:
                        try:
                            plan_data = json.loads(json_match.group(1).strip())
                            logger.info(f"   Extracted from code block: {len(plan_data)} steps")
                        except json.JSONDecodeError:
                            pass

                # Method 3: Find JSON array anywhere in text
                if plan_data is None:
                    # Look for JSON array pattern [...]
                    array_match = re.search(r'\[\s*\{.*?\}\s*\]', plan_str, re.DOTALL)
                    if array_match:
                        try:
                            plan_data = json.loads(array_match.group(0))
                            logger.info(f"   Extracted JSON array from text: {len(plan_data)} steps")
                        except json.JSONDecodeError:
                            pass

                # Method 4: Use _extract_plan_from_text helper
                if plan_data is None:
                    plan_data = self._extract_plan_from_text(str(raw_plan))
                    if plan_data:
                        logger.info(f"   Extracted via helper: {len(plan_data)} steps")

                # Method 5: Direct LLM retry with explicit JSON-only prompt
                if not plan_data:  # None or empty list
                    logger.warning(f"DSPy returned text, retrying with direct LLM call for JSON...")
                    try:
                        import dspy
                        lm = dspy.settings.lm
                        if lm:
                            # Get ALL skills with their first tool for the prompt
                            skill_info = []
                            for s in skills:
                                tools = s.get('tools', [])
                                if isinstance(tools, list) and tools:
                                    tool_name = tools[0].get('name', tools[0]) if isinstance(tools[0], dict) else tools[0]
                                    skill_info.append(f"{s.get('name')}/{tool_name}")
                                else:
                                    skill_info.append(s.get('name', ''))

                            direct_prompt = f"""Return ONLY a JSON array with 2-3 steps. Select ONLY the most relevant skills for this task.

Task: {task}
Task type: {task_type.value if hasattr(task_type, 'value') else task_type}
Available skills: {skill_info}

Select 2-3 most relevant skills. Return JSON array:
[{{"skill_name": "skill-name", "tool_name": "tool-name", "params": {{}}, "description": "what it does", "depends_on": [], "output_key": "step_0", "optional": false}}]

JSON:"""

                            response = lm(prompt=direct_prompt)
                            response_text = response[0] if isinstance(response, list) else str(response)
                            response_text = response_text.strip()
                            logger.debug(f"   Direct LLM response (first 200): {response_text[:200]}")

                            # Try to parse JSON from response
                            if response_text.startswith('['):
                                plan_data = json.loads(response_text)
                                logger.info(f"   Direct LLM retry successful: {len(plan_data)} steps")
                            elif '[' in response_text:
                                # Extract JSON array from response
                                start = response_text.find('[')
                                end = response_text.rfind(']') + 1
                                if end > start:
                                    plan_data = json.loads(response_text[start:end])
                                    logger.info(f"   Extracted from direct LLM: {len(plan_data)} steps")
                            else:
                                logger.warning(f"   Direct LLM returned no JSON array")
                        else:
                            logger.warning(f"   No LM configured for direct retry")
                    except Exception as e:
                        logger.warning(f"   Direct LLM retry failed: {e}")

                # If all methods failed, return empty list
                if plan_data is None:
                    logger.error(f"Could not parse execution plan. LLM returned text instead of JSON.")
                    logger.error(f"   Raw response (first 300 chars): {plan_str[:300]}")
                    plan_data = []

            # Ensure it's a list
            if plan_data and not isinstance(plan_data, list):
                plan_data = [plan_data]
            
            # Convert to ExecutionStep objects
            # Handle both LLM's preferred field names and our standard names
            ExecutionStep = _get_execution_step()
            steps = []

            # Build tool-to-skill mapping for inferring skill from tool name
            tool_to_skill = {}
            for s in skills:
                skill_name_map = s.get('name', '')
                for t in s.get('tools', []):
                    t_name = t.get('name') if isinstance(t, dict) else t
                    if t_name:
                        tool_to_skill[t_name] = skill_name_map

            # Build available skill names set for validation
            available_skill_names = {s.get('name', '') for s in skills if s.get('name')}
            logger.info(f"📋 Available skills for validation: {sorted(available_skill_names)}")

            # Helper to get value from dict or Pydantic model
            def get_val(obj, key, default=''):
                if isinstance(obj, dict):
                    return obj.get(key, default)
                return getattr(obj, key, default)

            # Helper for fuzzy skill matching (LLM might return slightly different names)
            def find_matching_skill(name: str) -> str:
                """Find matching skill using exact match, contains match, or prefix match."""
                if not name:
                    return ''
                name_lower = name.lower().strip()
                # Exact match first
                if name_lower in {s.lower() for s in available_skill_names}:
                    for s in available_skill_names:
                        if s.lower() == name_lower:
                            return s
                # Contains match (e.g., "file" matches "file-operations")
                for s in available_skill_names:
                    if name_lower in s.lower() or s.lower() in name_lower:
                        logger.debug(f"Fuzzy matched '{name}' -> '{s}'")
                        return s
                # Word overlap match (e.g., "file_operations" matches "file-operations")
                name_words = set(name_lower.replace('-', '_').replace(' ', '_').split('_'))
                for s in available_skill_names:
                    skill_words = set(s.lower().replace('-', '_').split('_'))
                    if name_words & skill_words:  # Any word overlap
                        logger.debug(f"Word overlap matched '{name}' -> '{s}'")
                        return s
                return ''

            for i, step_data in enumerate(plan_data[:max_steps]):
                try:
                    logger.debug(f"Processing step {i+1}: {step_data}")

                    # Handle both dict and Pydantic model objects
                    # Also handle LLM's field name variations
                    skill_name = get_val(step_data, 'skill_name') or get_val(step_data, 'skill', '')
                    tool_name = get_val(step_data, 'tool_name') or get_val(step_data, 'tool', '') or get_val(step_data, 'action', '')

                    # Infer skill from tool if skill is empty
                    if not skill_name and tool_name:
                        skill_name = tool_to_skill.get(tool_name, '')

                    # Infer skill from available skills if only one selected
                    # LLM often omits skill_name when there's an obvious single choice
                    if not skill_name and len(available_skill_names) == 1:
                        skill_name = list(available_skill_names)[0]
                        logger.info(f"Auto-inferred skill_name='{skill_name}' (only one skill available)")
                    elif not skill_name and len(available_skill_names) <= 3:
                        # With few skills, infer from task keywords
                        desc = str(get_val(step_data, 'description', task)).lower()
                        for candidate in available_skill_names:
                            if candidate.replace('-', ' ') in desc or any(w in desc for w in candidate.split('-')):
                                skill_name = candidate
                                logger.info(f"Inferred skill_name='{skill_name}' from description match")
                                break

                    # Try fuzzy matching if exact skill not found
                    if skill_name and skill_name not in available_skill_names:
                        matched = find_matching_skill(skill_name)
                        if matched:
                            logger.info(f"Skill name normalized: '{skill_name}' -> '{matched}'")
                            skill_name = matched

                    # VALIDATION: Skip steps without valid skill_name
                    if not skill_name or skill_name not in available_skill_names:
                        desc = str(get_val(step_data, 'description', f'Step {i+1}'))[:50]
                        logger.warning(f"Skipping step {i+1}: '{desc}' - skill='{skill_name}' not in available skills")
                        continue

                    description = get_val(step_data, 'description', f'Step {i+1}')

                    # Infer tool_name from skill if empty
                    if not tool_name:
                        for s in skills:
                            if s.get('name') == skill_name:
                                skill_tools = s.get('tools', [])
                                if skill_tools:
                                    tool_names = [t.get('name') if isinstance(t, dict) else t for t in skill_tools]
                                    desc_lower = description.lower()
                                    task_lower = task.lower()

                                    # For file-operations, select tool based on task keywords
                                    if skill_name == 'file-operations':
                                        # Check for directory operations FIRST (before file operations)
                                        if any(w in desc_lower for w in ['directory', 'folder', 'mkdir']):
                                            if 'create_directory_tool' in tool_names:
                                                tool_name = 'create_directory_tool'
                                        # File creation (must have file extension or explicit file reference)
                                        elif any(w in task_lower or w in desc_lower for w in ['create', 'write', 'generate', 'make']):
                                            # Only use write_file_tool if description mentions a file (has extension)
                                            if any(ext in desc_lower for ext in ['.py', '.js', '.ts', '.json', '.md', '.txt', '.html', '.css', 'file']):
                                                if 'write_file_tool' in tool_names:
                                                    tool_name = 'write_file_tool'
                                            elif 'write_file_tool' in tool_names:
                                                tool_name = 'write_file_tool'
                                        elif any(w in task_lower or w in desc_lower for w in ['read', 'load', 'get']):
                                            if 'read_file_tool' in tool_names:
                                                tool_name = 'read_file_tool'

                                    # Fallback to first tool if no specific match
                                    if not tool_name:
                                        first_tool = skill_tools[0]
                                        tool_name = first_tool.get('name') if isinstance(first_tool, dict) else first_tool

                                    logger.debug(f"Inferred tool_name='{tool_name}' from skill '{skill_name}'")
                                break

                    # Handle params - from dict, Pydantic model, or fallback
                    # Check multiple aliases including 'inputs' which LLMs often use
                    step_params = (
                        get_val(step_data, 'params') or
                        get_val(step_data, 'parameters') or
                        get_val(step_data, 'inputs') or  # LLM often uses 'inputs' instead of 'params'
                        get_val(step_data, 'tool_input') or
                        {}
                    )
                    if isinstance(step_params, str):
                        step_params = {}

                    if not step_params:
                        # Build params from skill schema
                        # Use step description for filename extraction (not full task)
                        prev_output = f'step_{i-1}' if i > 0 else None
                        # For file operations, use FULL TASK to extract file path (e.g., "save as /tmp/file.html")
                        # then use description for content generation hints
                        param_source = task if tool_name in ['write_file_tool', 'read_file_tool'] else task
                        step_params = self._build_skill_params(skill_name, param_source, prev_output, tool_name)
                        logger.debug(f"Built params for step {i+1}: {list(step_params.keys())}")

                    step = ExecutionStep(
                        skill_name=skill_name,
                        tool_name=tool_name,
                        params=step_params,
                        description=description,
                        depends_on=get_val(step_data, 'depends_on', []),
                        output_key=get_val(step_data, 'output_key', f'step_{i}'),
                        optional=get_val(step_data, 'optional', False)
                    )
                    steps.append(step)
                except Exception as e:
                    logger.warning(f"Failed to create step {i+1}: {e}")
                    continue

            # Validate: Ensure we have at least one step
            is_fallback_plan = False
            if not steps or len(steps) == 0:
                logger.warning("Execution plan resulted in 0 steps, using fallback plan")
                fallback_plan_data = self._create_fallback_plan(task, task_type, skills)
                ExecutionStep = _get_execution_step()
                steps = []
                for i, step_data in enumerate(fallback_plan_data[:max_steps]):
                    try:
                        step = ExecutionStep(
                            skill_name=step_data.get('skill_name', ''),
                            tool_name=step_data.get('tool_name', ''),
                            params=step_data.get('params', {}),
                            description=step_data.get('description', f'Step {i+1}'),
                            depends_on=step_data.get('depends_on', []),
                            output_key=step_data.get('output_key', f'step_{i}'),
                            optional=step_data.get('optional', False)
                        )
                        steps.append(step)
                    except Exception as e:
                        logger.warning(f"Failed to create fallback step {i+1}: {e}")
                        continue
                reasoning = f"Fallback plan created: {len(steps)} steps"
                is_fallback_plan = True
            else:
                reasoning = result.reasoning or f"Planned {len(steps)} steps"

            # DISABLED: Skill expansion logic
            # Trust the LLM's plan - if it only needs 1 step, that's correct
            # The old logic would add unnecessary steps for all selected skills
            # even when they weren't needed for the task
            used_skills = {step.skill_name for step in steps}
            if len(steps) > 0:
                logger.info(f"📋 Plan uses {len(used_skills)} skills: {used_skills}")

            logger.info(f"📝 Planned {len(steps)} execution steps")
            logger.debug(f"   Reasoning: {reasoning}")
            if hasattr(result, 'estimated_complexity'):
                logger.debug(f"   Complexity: {result.estimated_complexity}")
            
            return steps, reasoning
            
        except Exception as e:
            logger.error(f"Execution planning failed: {e}", exc_info=True)
            # Try fallback plan when LLM fails
            logger.warning("Attempting fallback plan due to execution planning failure")
            try:
                fallback_plan_data = self._create_fallback_plan(task, task_type, skills)
                logger.info(f"🔧 Fallback plan generated {len(fallback_plan_data)} steps: {fallback_plan_data}")
                
                if not fallback_plan_data:
                    logger.error("Fallback plan returned empty list!")
                    return [], f"Planning failed: {e}"
                
                ExecutionStep = _get_execution_step()
                steps = []
                for i, step_data in enumerate(fallback_plan_data[:max_steps]):
                    try:
                        step = ExecutionStep(
                            skill_name=step_data.get('skill_name', ''),
                            tool_name=step_data.get('tool_name', ''),
                            params=step_data.get('params', {}),
                            description=step_data.get('description', f'Step {i+1}'),
                            depends_on=step_data.get('depends_on', []),
                            output_key=step_data.get('output_key', f'step_{i}'),
                            optional=step_data.get('optional', False)
                        )
                        steps.append(step)
                        logger.debug(f"✅ Created fallback step {i+1}: {step.skill_name}.{step.tool_name}")
                    except Exception as step_e:
                        logger.warning(f"Failed to create fallback step {i+1}: {step_e}")
                        logger.debug(f"   Step data: {step_data}")
                        continue
                
                if steps:
                    logger.info(f"✅ Fallback plan created: {len(steps)} steps")
                    return steps, f"Fallback plan (planning failed: {str(e)[:100]})"
                else:
                    logger.error(f"❌ Fallback plan created {len(fallback_plan_data)} steps but 0 were converted to ExecutionStep objects")
            except Exception as fallback_e:
                logger.error(f"Fallback plan also failed: {fallback_e}", exc_info=True)
            
            return [], f"Planning failed: {e}"

    def _extract_tool_schema(self, tool_func, tool_name: str) -> Dict[str, Any]:
        """
        Extract parameter schema from tool function docstring.
        
        Args:
            tool_func: The tool function
            tool_name: Name of the tool
            
        Returns:
            Dictionary with tool name, parameters, and description
        """
        schema = {
            'name': tool_name,
            'parameters': [],
            'description': ''
        }
        
        if not tool_func or not hasattr(tool_func, '__doc__') or not tool_func.__doc__:
            return schema
        
        docstring = tool_func.__doc__
        lines = docstring.split('\n')
        
        # Extract description (first line)
        schema['description'] = lines[0].strip() if lines else ''
        
        # Extract parameters from Args section
        in_args = False
        for line in lines:
            line = line.strip()
            
            if 'Args:' in line or 'Parameters:' in line:
                in_args = True
                continue
            
            if in_args and line.startswith('-'):
                # Parse: "- path (str, required): Path to the file"
                parts = line[1:].strip().split(':', 1)
                if len(parts) == 2:
                    param_def = parts[0].strip()
                    desc = parts[1].strip()
                    
                    # Parse "path (str, required)" or "path (str, optional)"
                    param_name = param_def.split('(')[0].strip() if '(' in param_def else param_def.strip()
                    
                    if '(' in param_def:
                        type_info = param_def.split('(')[1].split(')')[0]
                        param_type = type_info.split(',')[0].strip()
                        required = 'required' in type_info.lower()
                    else:
                        param_type = 'str'
                        required = True  # Default to required if not specified
                    
                    schema['parameters'].append({
                        'name': param_name,
                        'type': param_type,
                        'required': required,
                        'description': desc
                    })
            
            elif in_args and ('Returns:' in line or 'Raises:' in line):
                break
        
        return schema
    
    def _abstract_task_for_planning(self, task: str) -> str:
        """
        Prepare task description for LLM planning calls.

        CRITICAL: Strips out context pollution markers that can corrupt:
        - Search queries (causing massive URLs)
        - PDF topic parameters
        - Skill tool inputs

        Context pollution markers include:
        - Q-Learning lessons
        - Transfer learning context
        - Multi-perspective analysis
        - Previous run metadata

        Truncates to 500 chars to stay within reasonable prompt limits.
        """
        # Clean the task first, then truncate
        cleaned = self._clean_task_for_query(task)
        return cleaned[:500].strip()

    def _clean_task_for_query(self, task: str) -> str:
        """
        Clean task description for use in search queries and skill inputs.

        CRITICAL: This prevents query pollution where enrichment context
        (Q-learning, transfer learning, etc.) gets passed to web search
        causing massive URLs and timeouts.

        Args:
            task: Task description (may contain enrichment context)

        Returns:
            Clean task description (original request only)
        """
        # Markers that indicate enrichment context (to be stripped)
        context_markers = [
            '\n[Multi-Perspective Analysis',
            '\n[Multi-Perspective',
            '\nLearned Insights:',
            '\n# Transferable Learnings',
            '\n# Q-Learning Lessons',
            '\n## Task Type Pattern',
            '\n## Role Advice',
            '\n## Meta-Learning Advice',
            '\n\n---\n',  # Common separator before context
            '\nBased on previous learnings:',
            '\nRecommended approach:',
            '\nPrevious success patterns:',
            '\n[Analysis]:',
            '\n[Consensus]:',
            '\n[Tensions]:',
            '\n[Blind Spots]:',
        ]

        cleaned = task
        for marker in context_markers:
            if marker in cleaned:
                # Keep only the part before the marker
                cleaned = cleaned.split(marker)[0]

        return cleaned.strip()

    def _build_skill_params(self, skill_name: str, task: str, prev_output_key: Optional[str] = None, tool_name: str = None) -> Dict[str, Any]:
        """Build params for a skill by looking up its tool schema from registry."""
        params = {}
        prev_ref = f"${{{prev_output_key}}}" if prev_output_key else task

        # Special handling for write_file_tool - generate actual content
        if skill_name == 'file-operations' and tool_name == 'write_file_tool':
            params = self._generate_file_content(task)
            if params.get('path') and params.get('content'):
                return params

        try:
            from ..registry.skills_registry import get_skills_registry
            registry = get_skills_registry()
            skill = registry.get_skill(skill_name)

            if skill and hasattr(skill, 'tools') and skill.tools:
                # Get the specific tool or first available
                tool_func = skill.tools.get(tool_name) if tool_name else list(skill.tools.values())[0]

                if tool_func:
                    # Extract parameter schema from docstring
                    schema = self._extract_tool_schema(tool_func, tool_name or list(skill.tools.keys())[0])
                    tool_params = schema.get('parameters', [])

                    # CRITICAL: Clean task for query params (strip enrichment context)
                    # The task may contain appended context like:
                    # - "[Multi-Perspective Analysis..."
                    # - "Learned Insights:..."
                    # - "# Transferable Learnings..."
                    # This context pollutes search queries causing timeouts
                    clean_task = self._clean_task_for_query(task)

                    # Build params based on actual schema
                    for param in tool_params:
                        param_name = param.get('name', '')
                        param_required = param.get('required', False)
                        param_desc = param.get('description', '').lower()

                        # Intelligently fill params based on description and name
                        # Use CLEAN task for search queries to avoid polluting URLs
                        if param_name in ['query', 'topic', 'search_query', 'q']:
                            params[param_name] = clean_task
                        elif param_name in ['message', 'text', 'content', 'body']:
                            # For write_file_tool, try to get generated content
                            if tool_name == 'write_file_tool':
                                file_content = self._generate_file_content(task)
                                if file_content and file_content.get('content'):
                                    params[param_name] = file_content['content']
                                else:
                                    params[param_name] = f"# TODO: Generated content for: {clean_task}"
                            else:
                                params[param_name] = prev_ref
                        elif param_name in ['file_path', 'pdf_path', 'path', 'input_path']:
                            # For file creation, extract actual filename/path from task
                            if tool_name == 'write_file_tool':
                                import re
                                # Try to extract file path from task (supports directory and absolute paths)
                                filepath_patterns = [
                                    # Match absolute paths: /tmp/file.html, /home/user/file.py
                                    r'(/(?:[\w\-\.]+/)*[\w\-\.]+\.(?:py|js|ts|html|css|json|md|txt|yaml|yml))',
                                    # Match "save as /path/file.ext" or "save to /path/file.ext"
                                    r'save\s+(?:as|to|it\s+as|it\s+to)\s+["\']?(/[\w\-\./]+\.(?:py|js|ts|html|css|json|md|txt|yaml|yml))["\']?',
                                    # Match paths with directories: pkg/subdir/file.py
                                    r'(?:create|write|make|generate)\s+(?:a\s+)?(?:file\s+)?(?:called\s+)?["\']?([\w\-]+(?:/[\w\-]+)*\.(?:py|js|ts|html|css|json|md|txt))["\']?',
                                    # Match quoted paths with directories
                                    r'["\']([\w\-]+(?:/[\w\-]+)*\.(?:py|js|ts|html|css|json|md|txt))["\']',
                                    # Match any path-like pattern: dir/file.ext
                                    r'([\w\-]+(?:/[\w\-]+)*\.(?:py|js|ts|html|css|json|md|txt))',
                                    # Fallback: simple filename
                                    r'(\w+\.(?:py|js|ts|html|css|json|md|txt))',
                                ]
                                extracted_filepath = None
                                for pattern in filepath_patterns:
                                    match = re.search(pattern, task, re.IGNORECASE)
                                    if match:
                                        extracted_filepath = match.group(1)
                                        break
                                params[param_name] = extracted_filepath or prev_ref
                                logger.debug(f"Extracted filepath: '{extracted_filepath}' from task")
                            else:
                                params[param_name] = prev_ref
                        elif param_name in ['max_results', 'limit', 'count']:
                            params[param_name] = 10
                        elif param_name in ['title', 'name']:
                            params[param_name] = clean_task[:50]
                        elif param_required:
                            # For other required params, use task or prev_ref based on description
                            if any(word in param_desc for word in ['input', 'content', 'data', 'result']):
                                params[param_name] = prev_ref
                            else:
                                params[param_name] = clean_task

                    if params:
                        logger.debug(f"Built params from schema for {skill_name}: {list(params.keys())}")
                        return params

        except Exception as e:
            logger.debug(f"Could not build params from registry for {skill_name}: {e}")

        # Fallback: generic params covering common required fields
        # Use CLEAN task for query/topic to avoid polluting search URLs
        clean_task = self._clean_task_for_query(task)
        return {
            'task': clean_task,
            'query': clean_task,
            'topic': clean_task,  # For research skills
            'input': prev_ref,
            'content': prev_ref,
            'text': prev_ref,
            'message': prev_ref,  # For notification skills
        }

    def _generate_file_content(self, task: str) -> Dict[str, Any]:
        """Generate actual file content using LLM for write_file_tool."""
        import re

        # Extract filename/path from task (supports directory and absolute paths)
        filepath_patterns = [
            # Match absolute paths: /tmp/file.html, /home/user/file.py
            r'(/(?:[\w\-\.]+/)*[\w\-\.]+\.(?:py|js|ts|html|css|json|md|txt|yaml|yml))',
            # Match "save as /path/file.ext" or "save to /path/file.ext"
            r'save\s+(?:as|to|it\s+as|it\s+to)\s+["\']?(/[\w\-\./]+\.(?:py|js|ts|html|css|json|md|txt|yaml|yml))["\']?',
            # Match paths with directories: pkg/subdir/file.py
            r'(?:create|write|make|generate)\s+(?:a\s+)?(?:file\s+)?(?:called\s+)?["\']?([\w\-]+(?:/[\w\-]+)*\.(?:py|js|ts|html|css|json|md|txt))["\']?',
            # Match quoted paths with directories
            r'["\']([\w\-]+(?:/[\w\-]+)*\.(?:py|js|ts|html|css|json|md|txt))["\']',
            # Match any path-like pattern: dir/file.ext
            r'([\w\-]+(?:/[\w\-]+)*\.(?:py|js|ts|html|css|json|md|txt))',
            # Fallback: simple filename
            r'(\w+\.(?:py|js|ts|html|css|json|md|txt))',
        ]

        filename = None
        for pattern in filepath_patterns:
            match = re.search(pattern, task, re.IGNORECASE)
            if match:
                filename = match.group(1)
                break

        # If no explicit filename, infer from task content
        if not filename:
            filename = self._infer_filename_from_task(task)

        if not filename:
            return {}

        # Use LLM to generate the actual content
        try:
            import dspy
            lm = dspy.settings.lm
            if not lm:
                return {}

            # Determine file type for appropriate code generation
            ext = filename.split('.')[-1].lower()
            lang_hint = {
                'py': 'Python',
                'js': 'JavaScript',
                'ts': 'TypeScript',
                'html': 'HTML',
                'css': 'CSS',
                'json': 'JSON',
                'md': 'Markdown',
            }.get(ext, 'code')

            prompt = f"""Generate the complete {lang_hint} code for this task. Return ONLY the code, no explanations.

Task: {task}
Filename: {filename}

{lang_hint} code:"""

            response = lm(prompt=prompt)
            content = response[0] if isinstance(response, list) else str(response)

            # Clean up response (remove markdown code blocks if present)
            content = content.strip()
            if content.startswith('```'):
                lines = content.split('\n')
                # Remove first line (```python) and last line (```)
                if lines[-1].strip() == '```':
                    lines = lines[1:-1]
                else:
                    lines = lines[1:]
                content = '\n'.join(lines)

            logger.debug(f"Generated {len(content)} chars of content for {filename}")
            return {'path': filename, 'content': content}

        except Exception as e:
            logger.warning(f"Failed to generate file content: {e}")
            return {}

    def _infer_filename_from_task(self, task: str) -> Optional[str]:
        """Infer appropriate filename from task description when not explicitly provided."""
        import re
        task_lower = task.lower()

        # Detect file type from task keywords
        file_ext = '.py'  # Default to Python
        if any(w in task_lower for w in ['html', 'webpage', 'website', 'web page', 'frontend', 'ui']):
            file_ext = '.html'
        elif any(w in task_lower for w in ['javascript', 'react', 'node', '.js']):
            file_ext = '.js'
        elif any(w in task_lower for w in ['typescript', '.ts']):
            file_ext = '.ts'
        elif any(w in task_lower for w in ['css', 'stylesheet', 'style']):
            file_ext = '.css'
        elif any(w in task_lower for w in ['json', 'config']):
            file_ext = '.json'
        elif any(w in task_lower for w in ['markdown', 'readme', 'documentation']):
            file_ext = '.md'

        # Try to extract a meaningful name from the task
        # Pattern 1: "Create/Build/Make X model/class/service/component"
        patterns = [
            r'(?:create|build|make|implement|write|add)\s+(?:a\s+)?(\w+)\s+(?:model|class|service|component|module|handler|controller|view)',
            r'(?:create|build|make|implement|write)\s+(?:a\s+)?(\w+)(?:Service|Model|Controller|Handler|Manager|Component)',
            r'(\w+)\s+(?:model|class|service|component|module)\s+with',
            r'(?:unit\s+)?tests?\s+for\s+(?:the\s+)?(\w+)',
            r'(\w+)\s+(?:implementation|functionality)',
        ]

        for pattern in patterns:
            match = re.search(pattern, task, re.IGNORECASE)
            if match:
                name = match.group(1).lower()
                # Clean up common prefixes
                name = re.sub(r'^(the|a|an)\s*', '', name)
                if name and len(name) > 1:
                    # Format filename based on type
                    if 'test' in task_lower:
                        return f"test_{name}{file_ext}"
                    elif 'model' in task_lower:
                        return f"models/{name}{file_ext}"
                    elif 'service' in task_lower:
                        return f"services/{name}_service{file_ext}"
                    elif 'api' in task_lower or 'endpoint' in task_lower:
                        return f"api/{name}{file_ext}"
                    else:
                        return f"{name}{file_ext}"

        # Fallback: Generate name from first meaningful word
        words = re.findall(r'\b([a-z]{3,})\b', task_lower)
        skip_words = {'create', 'build', 'make', 'implement', 'write', 'add', 'the', 'with', 'for', 'and', 'that', 'this'}
        for word in words:
            if word not in skip_words:
                if 'test' in task_lower:
                    return f"test_{word}{file_ext}"
                return f"{word}{file_ext}"

        # Last resort: app.py or index.html
        if file_ext == '.html':
            return 'index.html'
        return 'app.py'

    def _create_fallback_plan(
        self,
        task: str,
        task_type,
        skills: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Create a minimal fallback plan when LLM fails.

        Prioritizes skills by relevance to task type:
        - RESEARCH: web-search, http-client, web-scraper, claude-cli-llm
        - CREATION: claude-cli-llm, file-operations, docx-tools
        - ANALYSIS: calculator, claude-cli-llm
        - Default: claude-cli-llm, web-search
        """
        if not skills:
            return []

        TaskType = _get_task_type()

        # Priority order by task type
        priority_map = {
            TaskType.RESEARCH: ['web-search', 'http-client', 'web-scraper', 'claude-cli-llm', 'arxiv-downloader'],
            TaskType.COMPARISON: ['web-search', 'claude-cli-llm', 'calculator'],
            TaskType.CREATION: ['claude-cli-llm', 'file-operations', 'docx-tools', 'text-utils'],
            TaskType.ANALYSIS: ['calculator', 'claude-cli-llm', 'web-search'],
            TaskType.COMMUNICATION: ['claude-cli-llm', 'http-client'],
            TaskType.AUTOMATION: ['shell-exec', 'file-operations', 'process-manager'],
        }

        # Get priority list for this task type
        priority_skills = priority_map.get(task_type, ['claude-cli-llm', 'web-search'])

        # Sort skills by priority
        skill_names = {s.get('name', ''): s for s in skills}
        sorted_skills = []

        # Add priority skills first (if available)
        for ps in priority_skills:
            if ps in skill_names:
                sorted_skills.append(skill_names[ps])

        # Add remaining skills
        for s in skills:
            if s not in sorted_skills:
                sorted_skills.append(s)

        plan = []

        for skill in sorted_skills:
            skill_name = skill.get('name', '')
            tools = skill.get('tools', [])
            if isinstance(tools, dict):
                tools = list(tools.keys())
            if not tools:
                # Try loading from registry
                try:
                    from ..registry.skills_registry import get_skills_registry
                    registry = get_skills_registry()
                    skill_obj = registry.get_skill(skill_name)
                    if skill_obj and skill_obj.tools:
                        tools = list(skill_obj.tools.keys())
                except Exception:
                    pass

            if not tools:
                continue

            # Get tool names list
            tool_names = [t if isinstance(t, str) else t.get('name', '') for t in tools]
            tool_names = [t for t in tool_names if t]  # Filter empty

            # Select best tool based on task keywords
            task_lower = task.lower()
            tool_name = tool_names[0] if tool_names else ''  # Default to first

            # Smart tool selection for common skills
            if skill_name == 'file-operations':
                if any(w in task_lower for w in ['create', 'write', 'generate', 'make', 'save']):
                    if 'write_file_tool' in tool_names:
                        tool_name = 'write_file_tool'
                elif any(w in task_lower for w in ['read', 'load', 'get', 'open']):
                    if 'read_file_tool' in tool_names:
                        tool_name = 'read_file_tool'
                elif any(w in task_lower for w in ['delete', 'remove']):
                    if 'delete_file_tool' in tool_names:
                        tool_name = 'delete_file_tool'
                elif any(w in task_lower for w in ['directory', 'folder', 'mkdir']):
                    if 'create_directory_tool' in tool_names:
                        tool_name = 'create_directory_tool'
            elif skill_name == 'claude-cli-llm':
                if any(w in task_lower for w in ['generate', 'create', 'write', 'code']):
                    if 'generate_text_tool' in tool_names:
                        tool_name = 'generate_text_tool'
            elif skill_name == 'shell-exec':
                if 'run_command_tool' in tool_names:
                    tool_name = 'run_command_tool'

            logger.debug(f"Fallback selected tool '{tool_name}' for skill '{skill_name}'")

            prev_output_key = f'result_{len(plan) - 1}' if plan else None
            params = self._build_skill_params(skill_name, task, prev_output_key, tool_name)

            plan.append({
                'skill_name': skill_name,
                'tool_name': tool_name,
                'params': params,
                'description': f'Execute {skill_name}: {task}',
                'depends_on': [len(plan) - 1] if plan else [],  # Chain steps
                'output_key': f'result_{len(plan)}',
                'optional': len(plan) > 0  # First step required, rest optional
            })

            if len(plan) >= 3:
                break

        logger.info(f"Fallback plan: {len(plan)} steps from selected skills")
        return plan
    
    def _extract_skill_names_from_text(self, text: str) -> List[str]:
        """Extract skill names from LLM text output."""
        import re
        skill_names = []

        # Try to find JSON array pattern: ["skill1", "skill2"]
        json_array_match = re.search(r'\[(.*?)\]', text, re.DOTALL)
        if json_array_match:
            array_content = json_array_match.group(1)
            # Extract quoted strings
            matches = re.findall(r'"([^"]+)"', array_content)
            skill_names.extend(matches)

        # Try markdown list format: - skill-name or * skill-name
        if not skill_names:
            md_matches = re.findall(r'^[\-\*]\s*([a-z][a-z0-9\-_]+)', text, re.MULTILINE)
            skill_names.extend(md_matches)

        # Try numbered list: 1. skill-name
        if not skill_names:
            num_matches = re.findall(r'^\d+\.\s*([a-z][a-z0-9\-_]+)', text, re.MULTILINE)
            skill_names.extend(num_matches)

        # Also look for standalone quoted strings that might be skill names
        if not skill_names:
            matches = re.findall(r'"([^"]+)"', text)
            # Filter to likely skill names (lowercase, hyphens, common skill patterns)
            skill_names = [m for m in matches if ('-' in m or '_' in m) and m.islower()]

        # Last resort: find any word that looks like a skill name (has hyphen)
        if not skill_names:
            word_matches = re.findall(r'\b([a-z][a-z0-9]*-[a-z0-9\-]+)\b', text.lower())
            skill_names.extend(word_matches)

        # Remove duplicates and limit
        return list(dict.fromkeys(skill_names))[:10]
    
    def _extract_plan_from_text(self, text: str) -> List[Dict[str, Any]]:
        """Extract execution plan from LLM text output when JSON parsing fails.

        Returns empty list if extraction fails - fallback plan will be used.
        """
        import re
        steps = []

        # Try to find quoted skill names like "web-search" or 'web-search'
        skill_pattern = r'["\']([a-z][a-z0-9\-]+)["\']'
        skill_mentions = re.findall(skill_pattern, text.lower())

        # Filter to valid-looking skill names (must have hyphen or underscore, typical of skill names)
        valid_skills = [s for s in skill_mentions if '-' in s and len(s) > 4]

        if not valid_skills:
            # No valid skills found - return empty so fallback plan is used
            logger.debug("Could not extract valid skill names from text, using fallback plan")
            return []

        # Create a simple step for each mentioned skill
        for i, skill_name in enumerate(dict.fromkeys(valid_skills)):  # Deduplicate
            steps.append({
                'skill_name': skill_name,
                'tool_name': '',  # Will be resolved by validation
                'params': {},
                'description': f'Execute {skill_name}',
                'depends_on': [i-1] if i > 0 else [],
                'output_key': f'step_{i}',
                'optional': i > 0  # First step required, rest optional
            })

        return steps[:5]  # Limit to 5 steps
    
    # =============================================================================
    # Enhanced Planning with TaskGraph and Metadata
    # =============================================================================
    
    async def plan_with_metadata(
        self,
        task: Union[str, 'TaskGraph'],
        available_skills: Optional[List[Dict[str, Any]]] = None,
        max_steps: int = 15,
        convert_to_agents: bool = False
    ) -> 'ExecutionPlan':
        """
        Plan execution with enhanced metadata (ExecutionPlan).
        
        Works with both raw strings and TaskGraph.
        
        Args:
            task: Task description (str) or TaskGraph
            available_skills: Available skills (auto-discovers if None)
            max_steps: Maximum execution steps
            convert_to_agents: If True, also convert skills to AgentConfig for Conductor
            
        Returns:
            ExecutionPlan with steps and metadata
        """
        # Handle TaskGraph or raw string
        if TASK_GRAPH_AVAILABLE and isinstance(task, TaskGraph):
            task_string = task.metadata.get('original_request', '')
            task_type = task.task_type
            integrations = task.integrations
        else:
            task_string = str(task)
            task_type, _, _ = self.infer_task_type(task_string)
            integrations = []
        
        # Get TaskType for type checking
        TaskType = _get_task_type()
        
        # Discover skills if not provided
        if available_skills is None:
            available_skills = self._discover_available_skills()
        
        # Select best skills
        selected_skills, selection_reasoning = self.select_skills(
            task=task_string,
            available_skills=available_skills,
            max_skills=10
        )
        
        # Optionally convert skills to agents for Conductor
        agents = None
        if convert_to_agents:
            agents = await self._convert_skills_to_agents(selected_skills)
        
        # Plan execution
        steps, planning_reasoning = self.plan_execution(
            task=task_string,
            task_type=task_type,
            skills=selected_skills,
            max_steps=max_steps
        )
        
        # Extract metadata
        skill_names = [s.get('name') for s in selected_skills]
        required_tools = self._extract_required_tools(steps, skill_names)
        required_credentials = self._extract_required_credentials(integrations)
        estimated_time = self._estimate_time(steps)
        
        # Create ExecutionPlan
        if TASK_GRAPH_AVAILABLE and isinstance(task, TaskGraph):
            task_graph = task
        else:
            # Create minimal TaskGraph from raw string
            task_graph = self._create_task_graph_from_string(task_string, task_type)
        
        metadata = {
            'skills_discovered': skill_names,
            'selection_reasoning': selection_reasoning,
            'planning_reasoning': planning_reasoning
        }
        
        if agents:
            metadata['agents_created'] = [a.name for a in agents]
        
        return ExecutionPlan(
            task_graph=task_graph,
            steps=steps,
            estimated_time=estimated_time,
            required_tools=required_tools,
            required_credentials=required_credentials,
            metadata=metadata
        )
    
    async def _convert_skills_to_agents(self, skills: List[Dict[str, Any]]):
        """Convert skills to AgentConfig for Conductor."""
        try:
            from ..registry.skill_to_agent_converter import SkillToAgentConverter
            from ..registry.skills_registry import get_skills_registry
            
            converter = SkillToAgentConverter()
            registry = get_skills_registry()
            
            # Get SkillDefinition objects
            skill_defs = []
            for skill_dict in skills:
                skill_name = skill_dict.get('name')
                if skill_name:
                    skill_def = registry.get_skill(skill_name)
                    if skill_def:
                        skill_defs.append(skill_def)
            
            # Convert to agents
            agents = await converter.convert_skills_to_agents(skill_defs)
            return agents
            
        except Exception as e:
            logger.warning(f"Failed to convert skills to agents: {e}")
            return None
    
    def _discover_available_skills(self) -> List[Dict[str, Any]]:
        """Discover available skills from registry."""
        try:
            from ..registry.skills_registry import get_skills_registry
            registry = get_skills_registry()
            registry.init()
            
            all_skills_list = registry.list_skills()
            return [
                {
                    'name': s['name'],
                    'description': s.get('description', ''),
                    'tools': s.get('tools', [])
                }
                for s in all_skills_list
            ]
        except Exception as e:
            logger.warning(f"Failed to discover skills: {e}")
            return []
    
    def _extract_required_tools(self, steps: List[Any], skill_names: List[str]) -> List[str]:
        """Extract all required tools from steps."""
        tools = set(skill_names)
        for step in steps:
            if step.skill_name:
                tools.add(step.skill_name)
        return list(tools)
    
    def _extract_required_credentials(self, integrations: List[str]) -> List[str]:
        """Extract required credentials from integrations (any integration may need credentials)."""
        credentials = []
        for integration in integrations:
            # Assume any integration might need an API key
            credentials.append(f"{integration.lower()}_api_key")
        return credentials
    
    def _estimate_time(self, steps: List[Any]) -> str:
        """Estimate total execution time."""
        time_per_step = 2  # minutes average
        total_minutes = len(steps) * time_per_step
        
        if total_minutes < 60:
            return f"{total_minutes} minutes"
        else:
            hours = total_minutes // 60
            minutes = total_minutes % 60
            return f"{hours}h {minutes}m"
    
    def _create_task_graph_from_string(self, task_string: str, task_type) -> Optional[Any]:
        """Create minimal TaskGraph from raw string."""
        if not TASK_GRAPH_AVAILABLE:
            return None
        
        from ..autonomous.intent_parser import TaskGraph
        return TaskGraph(
            task_type=task_type,
            workflow=None,
            metadata={'original_request': task_string}
        )


# =============================================================================
# ExecutionPlan (moved here for unified planning)
# =============================================================================

@dataclass
class ExecutionPlan:
    """Execution plan with enhanced metadata."""
    task_graph: Optional[Any] = None  # TaskGraph if available
    steps: List[Any] = field(default_factory=list)  # List[ExecutionStep] - imported lazily
    estimated_time: Optional[str] = None
    required_tools: List[str] = field(default_factory=list)
    required_credentials: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Convenience Functions
# =============================================================================

def create_agentic_planner() -> AgenticPlanner:
    """Create a new agentic planner instance."""
    return AgenticPlanner()
