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
    import dspy
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False

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
    """Infer task type using semantic understanding (not keyword matching).
    
    CRITICAL: Output ONLY valid JSON with these exact fields: task_type, reasoning, confidence.
    Do NOT include any text before or after the JSON. Do NOT request permissions or mention tools.
    """
    task_description: str = dspy.InputField(
        desc="Natural language task description"
    )
    
    task_type: str = dspy.OutputField(
        desc="ONLY output one of these exact values (no other text): research, comparison, creation, communication, analysis, automation, unknown"
    )
    reasoning: str = dspy.OutputField(
        desc="Brief 1-2 sentence explanation of why this task type was inferred. Do NOT request permissions, mention tools, or include markdown formatting."
    )
    confidence: float = dspy.OutputField(
        desc="ONLY output a number between 0.0 and 1.0. No text, no explanation, just the number."
    )


class ExecutionPlanningSignature(dspy.Signature):
    """Plan execution steps using LLM reasoning (no hardcoded flows).
    
    CRITICAL: Output ONLY valid JSON. execution_plan must be valid JSON array.
    Do NOT include any text before or after the JSON.
    """
    task_description: str = dspy.InputField(
        desc="What needs to be accomplished"
    )
    task_type: str = dspy.InputField(
        desc="Inferred task type"
    )
    available_skills: str = dspy.InputField(
        desc="JSON list of available skills with their capabilities: "
        "[{'name': 'skill-name', 'description': '...', 'tools': ['tool1', 'tool2']}]"
    )
    previous_outputs: str = dspy.InputField(
        desc="JSON dict of outputs from previous steps: {'output_key': 'value'}"
    )
    max_steps: int = dspy.InputField(
        desc="Maximum number of steps allowed"
    )
    
    execution_plan: str = dspy.OutputField(
        desc="ONLY output a valid JSON array. Each step is an object with: "
        "skill_name (string), tool_name (string), params (object), "
        "description (string), depends_on (array of integers), output_key (string), optional (boolean). "
        "Do NOT include any text before or after the JSON array."
    )
    reasoning: str = dspy.OutputField(
        desc="Brief explanation (1-2 sentences) of why this plan was chosen. Do NOT include markdown or formatting."
    )
    estimated_complexity: str = dspy.OutputField(
        desc="ONLY output one of these exact values: simple, medium, complex. No other text."
    )


class SkillSelectionSignature(dspy.Signature):
    """Select best skills for task using semantic matching."""
    task_description: str = dspy.InputField(
        desc="What needs to be accomplished"
    )
    available_skills: str = dspy.InputField(
        desc="JSON list of all available skills"
    )
    max_skills: int = dspy.InputField(
        desc="Maximum number of skills to select"
    )
    
    selected_skills: str = dspy.OutputField(
        desc="JSON list of selected skill names"
    )
    reasoning: str = dspy.OutputField(
        desc="Why these skills were selected"
    )
    skill_priorities: str = dspy.OutputField(
        desc="JSON dict mapping skill names to priority scores (0.0-1.0)"
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
    
    def __init__(self):
        """Initialize agentic planner."""
        if not DSPY_AVAILABLE:
            raise RuntimeError("DSPy required for AgenticPlanner")
        
        # Initialize with explicit signature to ensure JSON schema is passed
        self.task_type_inferrer = dspy.ChainOfThought(TaskTypeInferenceSignature)
        self.execution_planner = dspy.ChainOfThought(ExecutionPlanningSignature)
        self.skill_selector = dspy.ChainOfThought(SkillSelectionSignature)
        
        # Store signatures for JSON schema extraction
        self._signatures = {
            'task_type': TaskTypeInferenceSignature,
            'execution': ExecutionPlanningSignature,
            'skill_selection': SkillSelectionSignature,
        }
        
        logger.info("🧠 AgenticPlanner initialized (fully LLM-based, no hardcoded logic)")
    
    def infer_task_type(self, task: str):
        """
        Infer task type using semantic understanding.
        
        Args:
            task: Task description
            
        Returns:
            (TaskType, reasoning, confidence)
        """
        try:
            # Set timeout for task type inference (30 seconds)
            import dspy
            if hasattr(dspy, 'settings') and dspy.settings.lm:
                lm = dspy.settings.lm
                # Temporarily set timeout if supported
                original_timeout = getattr(lm, 'timeout', None)
            
            # Call with explicit signature context for JSON schema
            # DSPy ChainOfThought should pass signature to LM, but we ensure it's available
            try:
                result = self.task_type_inferrer(task_description=task)
            except (TimeoutError, Exception) as e:
                if "timeout" in str(e).lower() or "timed out" in str(e).lower():
                    logger.warning(f"Task type inference timed out: {e}, using keyword fallback")
                    # Fall through to keyword-based inference
                    TaskType = _get_task_type()
                    task_lower = task.lower()
                    if any(word in task_lower for word in ['research', 'find', 'search', 'discover']):
                        return TaskType.RESEARCH, "Timeout fallback: task contains research keywords", 0.6
                    elif any(word in task_lower for word in ['create', 'generate', 'make', 'build']):
                        return TaskType.CREATION, "Timeout fallback: task contains creation keywords", 0.6
                    return TaskType.UNKNOWN, f"Timeout fallback: {str(e)[:100]}", 0.3
                raise
            
            # Parse task type
            TaskType = _get_task_type()
            
            # Get task_type_str - handle both direct access and fallback
            if hasattr(result, 'task_type') and result.task_type:
                task_type_str = str(result.task_type).lower().strip()
            else:
                # Try to extract from reasoning if available
                reasoning = str(result.reasoning) if hasattr(result, 'reasoning') else ''
                if 'research' in reasoning.lower():
                    task_type_str = 'research'
                elif 'create' in reasoning.lower() or 'generate' in reasoning.lower():
                    task_type_str = 'creation'
                elif 'compare' in reasoning.lower():
                    task_type_str = 'comparison'
                elif 'analyze' in reasoning.lower():
                    task_type_str = 'analysis'
                elif 'automate' in reasoning.lower():
                    task_type_str = 'automation'
                else:
                    task_type_str = 'unknown'
            
            # Clean task_type_str - remove any extra text
            task_type_str = task_type_str.split()[0] if task_type_str else 'unknown'
            
            task_type_map = {
                'research': TaskType.RESEARCH,
                'comparison': TaskType.COMPARISON,
                'creation': TaskType.CREATION,
                'communication': TaskType.COMMUNICATION,
                'analysis': TaskType.ANALYSIS,
                'automation': TaskType.AUTOMATION,
            }
            
            task_type = task_type_map.get(task_type_str, TaskType.UNKNOWN)
            
            # Parse confidence - handle string or number
            try:
                confidence_str = str(result.confidence).strip()
                # Extract number if there's text around it
                import re
                confidence_match = re.search(r'(\d+\.?\d*)', confidence_str)
                if confidence_match:
                    confidence = float(confidence_match.group(1))
                else:
                    confidence = float(confidence_str)
                confidence = max(0.0, min(1.0, confidence))
            except (ValueError, TypeError, AttributeError):
                confidence = 0.7  # Default if parsing fails
            
            # Clean reasoning - remove permission requests and markdown
            reasoning = str(result.reasoning) if hasattr(result, 'reasoning') else f"Inferred as {task_type_str}"
            reasoning = reasoning.strip()
            # Remove common unwanted phrases
            unwanted_phrases = [
                "I don't have permission",
                "I need permission",
                "Would you like me to",
                "Note:",
                "**",
                "##",
            ]
            for phrase in unwanted_phrases:
                if phrase.lower() in reasoning.lower():
                    # Extract only the relevant part
                    parts = reasoning.split(phrase)
                    reasoning = parts[0].strip() if parts else reasoning
            
            logger.info(f"📋 Task type inferred: {task_type.value} (confidence: {confidence:.2f})")
            logger.debug(f"   Reasoning: {reasoning[:200]}...")
            
            return task_type, reasoning, confidence
            
        except Exception as e:
            TaskType = _get_task_type()
            logger.warning(f"Task type inference failed: {e}, defaulting to UNKNOWN")
            logger.debug(f"   Error details: {traceback.format_exc()}")
            
            # Try to infer from task description as fallback
            task_lower = task.lower()
            if any(word in task_lower for word in ['research', 'find', 'search', 'discover']):
                return TaskType.RESEARCH, "Fallback inference: task contains research keywords", 0.6
            elif any(word in task_lower for word in ['create', 'generate', 'make', 'build']):
                return TaskType.CREATION, "Fallback inference: task contains creation keywords", 0.6
            
            return TaskType.UNKNOWN, f"Inference failed: {str(e)[:100]}", 0.0
    
    def select_skills(
        self,
        task: str,
        available_skills: List[Dict[str, Any]],
        max_skills: int = 8
    ) -> tuple[List[Dict[str, Any]], str]:
        """
        Select best skills for task using semantic matching.
        
        Args:
            task: Task description
            available_skills: List of available skills
            max_skills: Maximum skills to select
            
        Returns:
            (selected_skills, reasoning)
        """
        if not available_skills:
            return [], "No skills available"
        
        try:
            # Format skills for LLM
            skills_json = json.dumps([
                {
                    'name': s.get('name', ''),
                    'description': s.get('description', ''),
                    'tools': s.get('tools', [])
                }
                for s in available_skills[:50]  # Limit for context
            ], indent=2)
            
            result = self.skill_selector(
                task_description=task,
                available_skills=skills_json,
                max_skills=max_skills
            )
            
            # Parse selected skills
            try:
                selected_names = json.loads(result.selected_skills)
                if not isinstance(selected_names, list):
                    selected_names = [selected_names]
            except json.JSONDecodeError:
                # Try to extract from string
                selected_names = self._extract_skill_names_from_text(result.selected_skills)
            
            # If no skills extracted, try keyword matching as fallback
            if not selected_names:
                logger.warning("No skills extracted from LLM response, using keyword matching fallback")
                task_lower = task.lower()
                # Match skills based on task keywords
                for skill in available_skills:
                    skill_name = skill.get('name', '').lower()
                    skill_desc = skill.get('description', '').lower()
                    # Check if task keywords match skill name or description
                    if any(keyword in skill_name or keyword in skill_desc 
                           for keyword in ['research', 'search', 'web'] if 'research' in task_lower):
                        selected_names.append(skill.get('name'))
                    elif any(keyword in skill_name or keyword in skill_desc 
                             for keyword in ['code', 'generate', 'create'] if any(w in task_lower for w in ['code', 'generate', 'create'])):
                        selected_names.append(skill.get('name'))
                
                # If still no matches, use first few skills as fallback
                if not selected_names and available_skills:
                    selected_names = [s.get('name') for s in available_skills[:max_skills]]
            
            # Filter to available skills
            selected_skills = [
                s for s in available_skills
                if s.get('name') in selected_names
            ]
            
            # If still empty, use first few skills as last resort
            if not selected_skills and available_skills:
                logger.warning("Skill filtering resulted in empty list, using first available skills")
                selected_skills = available_skills[:max_skills]
            
            # Limit to max_skills
            selected_skills = selected_skills[:max_skills]
            
            reasoning = result.reasoning or f"Selected {len(selected_skills)} skills"
            
            logger.info(f"🎯 Selected {len(selected_skills)} skills: {[s.get('name') for s in selected_skills]}")
            logger.debug(f"   Reasoning: {reasoning}")
            
            return selected_skills, reasoning
            
        except Exception as e:
            logger.warning(f"Skill selection failed: {e}, using first available skills")
            # Return first few skills as fallback
            return available_skills[:max_skills], f"Selection failed: {e}"
    
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
            skills: Available skills
            previous_outputs: Outputs from previous steps
            max_steps: Maximum steps
            
        Returns:
            (execution_steps, reasoning)
        """
        try:
            # Format skills for LLM
            skills_json = json.dumps([
                {
                    'name': s.get('name', ''),
                    'description': s.get('description', ''),
                    'tools': list(s.get('tools', {}).keys()) if isinstance(s.get('tools'), dict) else s.get('tools', [])
                }
                for s in skills
            ], indent=2)
            
            # Format previous outputs
            outputs_json = json.dumps(previous_outputs or {}, indent=2)
            
            result = self.execution_planner(
                task_description=task,
                task_type=task_type.value,
                available_skills=skills_json,
                previous_outputs=outputs_json,
                max_steps=max_steps
            )
            
            # Parse execution plan
            try:
                # Try to parse as JSON
                plan_str = str(result.execution_plan).strip()
                
                # Remove markdown code blocks if present
                if plan_str.startswith('```'):
                    # Extract JSON from code block
                    import re
                    json_match = re.search(r'```(?:json)?\s*\n(.*?)\n```', plan_str, re.DOTALL)
                    if json_match:
                        plan_str = json_match.group(1).strip()
                
                plan_data = json.loads(plan_str)
                if not isinstance(plan_data, list):
                    plan_data = [plan_data]
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse execution plan JSON: {e}, trying to extract from text")
                logger.debug(f"   Plan string: {plan_str[:200]}...")
                plan_data = self._extract_plan_from_text(result.execution_plan)
                if not plan_data:
                    # Fallback: create minimal plan
                    logger.warning("Could not extract plan, creating minimal fallback plan")
                    plan_data = [{
                        'skill_name': 'unknown',
                        'tool_name': 'unknown',
                        'params': {},
                        'description': f"Execute: {task}",
                        'depends_on': [],
                        'output_key': 'result',
                        'optional': False
                    }]
            
            # Convert to ExecutionStep objects
            ExecutionStep = _get_execution_step()
            steps = []
            for i, step_data in enumerate(plan_data[:max_steps]):
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
                    logger.warning(f"Failed to create step {i+1}: {e}")
                    continue
            
            reasoning = result.reasoning or f"Planned {len(steps)} steps"
            
            logger.info(f"📝 Planned {len(steps)} execution steps")
            logger.debug(f"   Reasoning: {reasoning}")
            logger.debug(f"   Complexity: {result.estimated_complexity}")
            
            return steps, reasoning
            
        except Exception as e:
            logger.error(f"Execution planning failed: {e}", exc_info=True)
            return [], f"Planning failed: {e}"
    
    def _extract_skill_names_from_text(self, text: str) -> List[str]:
        """Extract skill names from LLM text output."""
        # Try to find JSON-like structures
        import re
        # Look for quoted strings that might be skill names
        matches = re.findall(r'"([^"]+)"', text)
        # Filter to likely skill names (lowercase, hyphens)
        skill_names = [m for m in matches if '-' in m or m.islower()]
        return skill_names[:10]  # Limit
    
    def _extract_plan_from_text(self, text: str) -> List[Dict[str, Any]]:
        """Extract execution plan from LLM text output when JSON parsing fails."""
        import re
        steps = []
        
        # Try to find step-like structures
        step_pattern = r'step\s+\d+[:\-]?\s*(.+?)(?=step\s+\d+|$)'
        matches = re.findall(step_pattern, text, re.IGNORECASE | re.DOTALL)
        
        for i, match in enumerate(matches):
            # Try to extract skill and tool names
            skill_match = re.search(r'skill[:\s]+([\w\-]+)', match, re.IGNORECASE)
            tool_match = re.search(r'tool[:\s]+([\w\-_]+)', match, re.IGNORECASE)
            
            if skill_match:
                steps.append({
                    'skill_name': skill_match.group(1),
                    'tool_name': tool_match.group(1) if tool_match else 'execute_tool',
                    'params': {},
                    'description': match.strip()[:100],
                    'depends_on': [],
                    'output_key': f'step_{i}',
                    'optional': False
                })
        
        return steps
    
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
        """Extract required credentials from integrations."""
        credentials = []
        for integration in integrations:
            if integration.lower() in ['reddit', 'notion', 'slack', 'twitter', 'github', 'telegram', 'discord']:
                credentials.append(f"{integration}_api_key")
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
