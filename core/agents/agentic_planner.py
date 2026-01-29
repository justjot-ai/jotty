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
    """Classify the task type from description.
    
    You are a CLASSIFIER. Analyze the task description and classify it into one of: research, comparison, creation, communication, analysis, automation, unknown.
    You are NOT executing anything. You are ONLY classifying the task type.
    
    IMPORTANT: Return ONLY a JSON object with fields: task_type, reasoning, confidence.
    Do NOT ask for permission. Do NOT execute anything. Just classify.
    """
    task_description: str = dspy.InputField(desc="The task description to classify - you are classifying it, not executing it")
    
    task_type: str = dspy.OutputField(
        desc="ONLY output one of these exact values: research, comparison, creation, communication, analysis, automation, unknown"
    )
    reasoning: str = dspy.OutputField(
        desc="Brief 1-2 sentence explanation. Do NOT request permissions. Do NOT mention tools. Just explain the classification."
    )
    confidence: float = dspy.OutputField(
        desc="ONLY output a number between 0.0 and 1.0. No text, no explanation, just the number."
    )


class ExecutionPlanningSignature(dspy.Signature):
    """Plan execution steps - output JSON ONLY, NO TEXT.
    
    You are a PLANNER. Your ONLY job is to PLAN steps, not execute them.
    
    CRITICAL INSTRUCTIONS:
    1. Output ONLY a JSON array - NO text, NO explanations, NO markdown
    2. Start your response with '[' and end with ']'
    3. Do NOT include any text before or after the JSON array
    4. Do NOT write explanations, comments, or descriptions outside the JSON
    
    The task_description describes WHAT needs to be done. Your job is to PLAN HOW to do it.
    You are NOT executing anything. You are NOT asking for permission. You are ONLY creating a plan.
    
    Return ONLY a JSON array of steps. Each step must have: skill_name, tool_name, params (object), 
    description, depends_on (array), output_key, optional (boolean).
    Use only tools from available_skills.
    
    REMEMBER: JSON ONLY. NO TEXT. NO EXPLANATIONS.
    """
    task_description: str = dspy.InputField(desc="Task to plan - you are PLANNING, not executing. Return JSON plan only.")
    task_type: str = dspy.InputField(
        desc="Inferred task type"
    )
    available_skills: str = dspy.InputField(
        desc="JSON list of available skills with their capabilities and tool schemas: "
        "[{'name': 'skill-name', 'description': '...', 'tools': [{'name': 'tool1', 'parameters': [...], 'description': '...'}]}]. "
        "Each tool object contains: 'name' (exact tool name), 'parameters' (list of parameter schemas with 'name', 'type', 'required', 'description'), and 'description'. "
        "CRITICAL: You MUST use ONLY the EXACT tool names provided. "
        "CRITICAL: You MUST provide ALL required parameters for each tool. Check the 'parameters' array for each tool to see which parameters are 'required: true'. "
        "IMPORTANT FOR DOCUMENT CONVERSION: Pandoc-based tools (convert_to_docx_tool, convert_to_html_tool, convert_to_pdf_tool) "
        "CANNOT convert FROM PDF files. They can convert FROM markdown, HTML, DOCX, EPUB, etc. "
        "If you need multiple formats, convert FROM the original source file (e.g., markdown) to all target formats, "
        "NOT from PDF to other formats. "
        "Example: If tool 'write_file_tool' has parameters [{'name': 'path', 'required': true}, {'name': 'content', 'required': true}], "
        "your plan MUST include both 'path' and 'content' in the params object. "
        "Do NOT invent, abbreviate, or modify tool names or parameter names. "
        "Copy tool names and parameter names EXACTLY as shown in the schema."
    )
    previous_outputs: str = dspy.InputField(
        desc="JSON dict of outputs from previous steps: {'output_key': 'value'}"
    )
    max_steps: int = dspy.InputField(
        desc="Maximum number of steps allowed"
    )
    
    execution_plan: str = dspy.OutputField(
        desc="CRITICAL: You MUST output ONLY a valid JSON array. NO TEXT. NO EXPLANATIONS. NO MARKDOWN. "
        "Start with '[' and end with ']'. Each step is an object with: "
        "skill_name (string), tool_name (string), params (object), "
        "description (string), depends_on (array of step indices), output_key (string), optional (boolean). "
        "CRITICAL: tool_name MUST be one of the exact tool names from the 'tools' array for that skill. "
        "Do NOT use the skill name as the tool name. "
        "DATA FLOW: When a step depends on previous step output, reference it in params using "
        "${output_key} syntax. For example, if step 0 has output_key='search_results', "
        "step 1 can use {\"text\": \"${search_results}\"} to receive that output. "
        "EXAMPLE: ["
        "{\"skill_name\": \"web-search\", \"tool_name\": \"search_web_tool\", \"params\": {\"query\": \"AI trends\"}, \"description\": \"Search\", \"depends_on\": [], \"output_key\": \"search_results\", \"optional\": false}, "
        "{\"skill_name\": \"summarize\", \"tool_name\": \"summarize_text_tool\", \"params\": {\"text\": \"${search_results}\"}, \"description\": \"Summarize\", \"depends_on\": [0], \"output_key\": \"summary\", \"optional\": false}"
        "]"
    )
    reasoning: str = dspy.OutputField(
        desc="Brief explanation (1-2 sentences) of why this plan was chosen. Do NOT include markdown or formatting."
    )
    estimated_complexity: str = dspy.OutputField(
        desc="ONLY output one of these exact values: simple, medium, complex. No other text."
    )


class SkillSelectionSignature(dspy.Signature):
    """Select the best skills for the task.
    
    You are a SKILL SELECTOR. Analyze the task and select relevant skills from available_skills. Return JSON with selected_skills array, reasoning, and skill_priorities dict.
    You are NOT executing anything. You are ONLY selecting which skills are needed.
    """
    task_description: str = dspy.InputField(desc="What needs to be accomplished - you are selecting skills for planning, not executing")
    available_skills: str = dspy.InputField(
        desc="JSON list of all available skills with their descriptions and tools"
    )
    max_skills: int = dspy.InputField(
        desc="Maximum number of skills to select"
    )
    
    selected_skills: str = dspy.OutputField(
        desc="ONLY output a valid JSON array of skill names. Example: [\"file-operations\", \"document-converter\"]. No other text."
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
        
        # Hybrid approach: ChainOfThought for reasoning, Predict for structured output
        # Reasoning tasks (task type, skill selection) benefit from ChainOfThought's transparency
        # Structured output (execution planning) needs Predict's format enforcement
        # 
        # Try using JSONAdapter for execution planning to enforce JSON output
        # This should prevent LLM from asking for permissions
        try:
            from dspy.adapters import JSONAdapter
            # Use JSONAdapter for execution planning to force JSON output
            self.execution_planner = dspy.Predict(ExecutionPlanningSignature)
            # Wrap with JSONAdapter if available
            # Note: JSONAdapter might need to be configured differently
            logger.debug("Using Predict with JSONAdapter support for execution planning")
        except ImportError:
            self.execution_planner = dspy.Predict(ExecutionPlanningSignature)
            logger.debug("JSONAdapter not available, using standard Predict")
        
        self.task_type_inferrer = dspy.ChainOfThought(TaskTypeInferenceSignature)
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
    
    def select_skills(
        self,
        task: str,
        available_skills: List[Dict[str, Any]],
        max_skills: int = 8
    ) -> tuple[List[Dict[str, Any]], str]:
        """
        Select best skills for task using LLM semantic matching.

        Falls back to using first available skills if LLM fails.

        Args:
            task: Task description
            available_skills: List of available skills
            max_skills: Maximum skills to select

        Returns:
            (selected_skills, reasoning)
        """
        if not available_skills:
            return [], "No skills available"

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

            result = self.skill_selector(
                task_description=task,
                available_skills=skills_json,
                max_skills=max_skills
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
            logger.info(f"LLM selected {len(llm_selected_names)} skills: {llm_selected_names}")

        except Exception as e:
            logger.warning(f"LLM selection failed: {e}")

        # Build final selection
        if llm_selected_names:
            final_names = list(set(llm_selected_names))[:max_skills]
            reasoning = llm_reasoning
        else:
            # Fallback: use first available skills
            final_names = [s.get('name') for s in available_skills[:max_skills]]
            reasoning = "Fallback: using first available skills"

        # Filter to available skills
        selected_skills = [s for s in available_skills if s.get('name') in final_names]

        if not selected_skills and available_skills:
            selected_skills = available_skills[:max_skills]

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
            skills: Available skills
            previous_outputs: Outputs from previous steps
            max_steps: Maximum steps
            
        Returns:
            (execution_steps, reasoning)
        """
        try:
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
            # The LLM sees "Create a file..." and thinks it needs to execute
            # So we abstract it to focus on planning, not execution
            abstracted_task = self._abstract_task_for_planning(task)
            logger.debug(f"🔍 Task abstraction: '{task[:80]}...' -> '{abstracted_task}'")
            
            # Prefix task description to make it clear this is PLANNING, not execution
            # This helps LLM understand it's creating a plan, not executing the task
            planning_task = f"PLAN HOW TO: {abstracted_task}"
            logger.debug(f"🔍 Planning task: {planning_task}")
            
            # Use dspy.context() if we're in an async context and need to set LM
            # Otherwise, just call the module directly (it already has the signature)
            result = self.execution_planner(
                task_description=planning_task,
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
                
                # Validate: Check if plan is empty
                if not plan_data or len(plan_data) == 0:
                    logger.warning("LLM returned empty execution plan, using fallback")
                    plan_data = self._create_fallback_plan(task, task_type, skills)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse execution plan JSON: {e}, trying to extract from text")
                logger.warning(f"   Plan string (first 500 chars): {plan_str[:500]}")
                logger.warning(f"   Plan string length: {len(plan_str)}")
                # Log the full response for debugging
                if hasattr(result, 'execution_plan'):
                    logger.debug(f"   Full execution_plan response: {str(result.execution_plan)[:1000]}")
                plan_data = self._extract_plan_from_text(result.execution_plan)
                if not plan_data:
                    # Fallback: create intelligent plan based on task type and available skills
                    logger.warning("Could not extract plan, creating intelligent fallback plan")
                    logger.warning(f"   Reason: LLM returned invalid JSON. This usually means:")
                    logger.warning(f"   1. LLM timed out or was interrupted")
                    logger.warning(f"   2. LLM returned text instead of JSON")
                    logger.warning(f"   3. LLM response was empty or malformed")
                    plan_data = self._create_fallback_plan(task, task_type, skills)
            
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
            
            # Validate: Ensure we have at least one step
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
            else:
                reasoning = result.reasoning or f"Planned {len(steps)} steps"
            
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

        Preserves the actual task context so the LLM can plan specific steps.
        The DSPy signatures already contain instructions like "You are a PLANNER"
        and "You are NOT executing anything", so no context stripping is needed.
        Truncates to 500 chars to stay within reasonable prompt limits.
        """
        return task[:500].strip()
    
    def _create_fallback_plan(
        self,
        task: str,
        task_type,
        skills: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Create a minimal fallback plan when LLM fails.

        Uses the first available tool from each selected skill.
        No keyword matching - the LLM already selected relevant skills.
        """
        if not skills:
            return []

        plan = []

        for skill in skills:
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

            plan.append({
                'skill_name': skill_name,
                'tool_name': tools[0],
                'params': {'task': task[:200]},
                'description': f'Execute {skill_name}: {task[:80]}',
                'depends_on': [],
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
        
        # Also look for standalone quoted strings that might be skill names
        if not skill_names:
            matches = re.findall(r'"([^"]+)"', text)
            # Filter to likely skill names (lowercase, hyphens, common skill patterns)
            skill_names = [m for m in matches if ('-' in m or '_' in m) and m.islower()]
        
        # Remove duplicates and limit
        return list(dict.fromkeys(skill_names))[:10]
    
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
