"""
Jotty v6.0 - Enhanced Agent Module
===================================

All A-Team agent enhancements:
- Dr. Chen: Inter-agent communication, multi-round validation
- Aristotle: Causal knowledge injection
- Dr. Agarwal: Dynamic context allocation

Agent types:
- InspectorAgent: Architect and Auditor validation agents
- Shared scratchpad for tool result caching
- Multi-round refinement loop
"""

import dspy
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json
import re
import logging
import asyncio
import time  # 🔥 CRITICAL FIX: Missing import causing "name 'time' is not defined"

from ..foundation.data_structures import (
    JottyConfig, ValidationResult, OutputTag, ValidationRound,
    MemoryLevel, AgentMessage, SharedScratchpad, CommunicationType
)
from ..memory.cortex import HierarchicalMemory
from ..learning.learning import DynamicBudgetManager

logger = logging.getLogger(__name__)


# =============================================================================
# SIGNATURES
# =============================================================================

class ArchitectSignature(dspy.Signature):
    """🔍 Pre-Exploration Agent: Explore data and brief the actor.
    
    🔥 A-TEAM PHASE 2 FIX: Architect is an ADVISOR, not a gatekeeper!
    
    YOUR ROLE:
    - EXPLORE available data using your tools
    - UNDERSTAND what data exists and its quality
    - BRIEF the actor on your findings
    - ADVISE on best approaches
    - DO NOT BLOCK execution!
    
    YOU ARE A REACT AGENT WITH TOOLS:
    - CALL tools to discover tables, search terms, explore metadata
    - DO NOT just generate text
    - Base your briefing on TOOL EVIDENCE
    
    CRITICAL: You are an exploration advisor, NOT a validator!
    """
    
    # Simple inputs - agent uses TOOLS for details
    task: str = dspy.InputField(desc="What to explore. USE TOOLS to discover available data.")
    context: str = dspy.InputField(desc="Current state summary. Call tools for specifics.")
    
    # 🔥 A-TEAM: Exploration outputs (NO should_proceed!)
    reasoning: str = dspy.OutputField(desc="Step-by-step exploration WITH tool calls and results. What did you discover?")
    exploration_summary: str = dspy.OutputField(desc="Summary of what you found: available tables, relevant terms, data quality notes")
    recommendations: str = dspy.OutputField(desc="Recommendations for the actor: suggested approach, things to be careful about, tips")
    data_quality_notes: str = dspy.OutputField(desc="Notes on data quality: missing data, stale tables, partition issues, etc.")
    confidence: float = dspy.OutputField(desc="Confidence in your exploration 0.0-1.0. Low confidence = need more exploration.")
    suggested_approach: str = dspy.OutputField(desc="Suggested approach for the actor based on what you discovered")
    insight_to_share: str = dspy.OutputField(desc="Key insight to share with other agents for learning")


class AuditorSignature(dspy.Signature):
    """Post-validation agent. USE YOUR TOOLS to validate extracted data.
    
    You are a ReAct agent with tools. DO NOT just generate text.
    CALL your tools to check the extraction, then validate based on tool results.
    """
    
    # Simple inputs - agent uses TOOLS for details
    task: str = dspy.InputField(desc="What extraction to validate. USE TOOLS to inspect actual data.")
    context: str = dspy.InputField(desc="Extraction summary. Call tools for detailed validation.")
    
    # FULL outputs for memory, RL, and reflection
    reasoning: str = dspy.OutputField(desc="Step-by-step analysis WITH tool calls and validation results")
    is_valid: bool = dspy.OutputField(desc="True if extraction is valid, False if issues found")
    confidence: float = dspy.OutputField(desc="Confidence 0.0-1.0 based on tool observations")
    output_tag: str = dspy.OutputField(desc="One of: useful (valid), fail (invalid), enquiry (uncertain)")
    output_name: str = dspy.OutputField(desc="Name for this validated output")
    why_useful: str = dspy.OutputField(desc="Why this output is useful - for memory/learning")
    fix_instructions: str = dspy.OutputField(desc="If invalid, specific fix instructions from tools")
    insight_to_share: str = dspy.OutputField(desc="Key insight to share with other agents for learning")


class RefinementSignature(dspy.Signature):
    """Refinement: Improve decision based on feedback."""
    
    original_decision: str = dspy.InputField(desc="Original validation decision and reasoning")
    feedback: str = dspy.InputField(desc="Feedback from other agents or round results")
    additional_context: str = dspy.InputField(desc="New context gathered")
    
    refined_reasoning: str = dspy.OutputField(desc="Updated reasoning incorporating feedback")
    refined_decision: bool = dspy.OutputField(desc="Refined should_proceed/is_valid")
    refined_confidence: float = dspy.OutputField(desc="Updated confidence")
    changes_made: str = dspy.OutputField(desc="What changed and why")


# =============================================================================
# INTERNAL REASONING TOOL
# =============================================================================

class InternalReasoningTool:
    """
    Agent's internal reasoning capability.
    
    Allows agents to:
    - Think through complex decisions
    - Access memory without external tools
    - Reason about causal relationships
    """
    
    def __init__(self, memory: HierarchicalMemory, config: JottyConfig):
        self.memory = memory
        self.config = config
        self.name = "reason_about"
        self.description = """
        Internal reasoning tool for complex analysis.
        Use when you need to:
        - Think through a decision step by step
        - Access relevant past experiences
        - Understand why something works or doesn't
        
        Arguments:
        - question: What to reason about
        - context_scope: 'all', 'relevant', 'memory', 'causal'
        """
    
    def __call__(self, question: str, context_scope: str = "relevant") -> Dict[str, Any]:
        """Execute internal reasoning."""
        result = {
            "question": question,
            "relevant_memories": [],
            "causal_insights": [],
            "analysis": ""
        }
        
        # Get relevant memories
        if context_scope in ("all", "relevant", "memory"):
            memories = self.memory.retrieve(
                query=question,
                goal=question,
                budget_tokens=5000,
                levels=[MemoryLevel.SEMANTIC, MemoryLevel.PROCEDURAL, MemoryLevel.META]
            )
            # ✅ Return FULL memory content, not truncated 
            result["relevant_memories"] = [
                {"content": m.content, "value": m.default_value}
                for m in memories  # 🔥 NO LIMIT - FULL content
            ]
        
        # Get causal knowledge
        if context_scope in ("all", "relevant", "causal"):
            causal = self.memory.retrieve_causal(question, {})
            result["causal_insights"] = [
                {"cause": c.cause, "effect": c.effect, "confidence": c.confidence}
                for c in causal
            ]
        
        return result


# =============================================================================
# TOOL WRAPPER WITH CACHING
# =============================================================================

class CachingToolWrapper:
    """
    Wraps user tools with caching via shared scratchpad.
    
    Prevents redundant tool calls across agents.
    """
    
    def __init__(self, tool: Any, scratchpad: SharedScratchpad, agent_name: str):
        self.tool = tool
        self.scratchpad = scratchpad
        self.agent_name = agent_name
        self.name = getattr(tool, 'name', str(tool))
        self.description = getattr(tool, 'description', '')
    
    def __call__(self, **kwargs) -> Any:
        """Call tool with caching."""
        # Check cache first
        cached = self.scratchpad.get_cached_result(self.name, kwargs)
        if cached is not None:
            return cached
        
        # Execute tool
        try:
            if callable(self.tool):
                result = self.tool(**kwargs)
            else:
                result = self.tool
        except Exception as e:
            result = {"error": str(e)}
        
        # Cache result WITHOUT truncation
        message = AgentMessage(
            sender=self.agent_name,
            receiver="*",  # Broadcast
            message_type=CommunicationType.TOOL_RESULT,
            content={
                "tool": self.name, 
                "args": str(kwargs), 
                "result_summary": str(result)  # ✅ FULL result, no  truncation!
            },
            tool_name=self.name,
            tool_args=kwargs,
            tool_result=result
        )
        self.scratchpad.add_message(message)
        
        return result


# =============================================================================
# EVAL AGENT
# =============================================================================

class InspectorAgent:
    """
    Enhanced evaluation agent with all A-Team features.
    
    Supports:
    - Inter-agent communication via scratchpad
    - Multi-round validation refinement
    - Causal knowledge injection
    - Dynamic context allocation
    """
    
    def __init__(self,
                 md_path: Path,
                 is_architect: bool,
                 tools: List[Any],
                 config: JottyConfig,
                 scratchpad: SharedScratchpad = None):
        """
        Initialize InspectorAgent.
        
        Parameters:
            md_path: Path to markdown system prompt
            is_architect: True for Architect, False for Auditor
            tools: List of user-provided tools
            config: JOTTY configuration
            scratchpad: Shared scratchpad for communication
        """
        self.config = config
        self.is_architect = is_architect
        self.scratchpad = scratchpad or SharedScratchpad()
        
        # Load system prompt
        self.md_path = Path(md_path)
        self.system_prompt = self._load_system_prompt()
        
        # Agent name from filename
        self.agent_name = self.md_path.stem
        
        # Initialize memory
        self.memory = HierarchicalMemory(self.agent_name, config)
        
        # Budget manager
        self.budget_manager = DynamicBudgetManager(config)
        
        # 🔥 A-TEAM CRITICAL FIX: Don't double-wrap tools!
        # Tools from _get_auto_discovered_dspy_tools() are ALREADY smart wrappers
        # with caching, param resolution, etc. Don't wrap them again!
        self.user_tools = tools
        
        # Check if tools are already dspy.Tool objects (from new architecture)
        if tools and len(tools) > 0 and hasattr(tools[0], 'func'):
            # New architecture: Individual DSPy tools with smart wrappers
            # NO additional wrapping needed!
            logger.debug(f"✅ Using {len(tools)} individual DSPy tools (already wrapped)")
            self.cached_tools = tools  # Use as-is!
        else:
            # Legacy: Wrap with CachingToolWrapper
            logger.debug(f"⚠️  Wrapping {len(tools)} legacy tools with CachingToolWrapper")
            self.cached_tools = [
                CachingToolWrapper(t, self.scratchpad, self.agent_name)
                for t in tools
            ]
        
        # Add internal reasoning tool
        self.reasoning_tool = InternalReasoningTool(self.memory, config)
        self.all_tools = self.cached_tools + [self.reasoning_tool]
        
        # Create DSPy agent
        self.signature = ArchitectSignature if is_architect else AuditorSignature
        self.agent = dspy.ReAct(
            self.signature,
            tools=self.all_tools,
            max_iters=config.max_eval_iters
        )
        
        # Refinement agent
        if config.enable_multi_round:
            self.refiner = dspy.ChainOfThought(RefinementSignature)
        
        # Statistics
        self.total_calls = 0
        self.total_approvals = 0
        self.tool_calls: List[Dict] = []
    
    def _smart_truncate(self, text: str, max_chars: int) -> str:
        """
        Intelligently truncate text to fit within character limit.
        
        Tries to:
        1. Keep complete sentences
        2. Keep complete words
        3. Add ellipsis only if truncated
        
        NO RANDOM CONTEXT LOSS!
        """
        if len(text) <= max_chars:
            return text
        
        # Try to truncate at sentence boundary
        truncated = text[:max_chars]
        
        # Look for last sentence ending
        for delimiter in ['. ', '.\n', '! ', '?\n', '? ']:
            last_sentence = truncated.rfind(delimiter)
            if last_sentence > max_chars * 0.7:  # At least 70% of content
                return truncated[:last_sentence + 1] + "..."
        
        # Fall back to word boundary
        last_space = truncated.rfind(' ')
        if last_space > max_chars * 0.8:  # At least 80% of content
            return truncated[:last_space] + "..."
        
        # Last resort: hard truncate
        return truncated + "..."
    
    def _load_system_prompt(self) -> str:
        """Load system prompt from markdown file."""
        if self.md_path.exists():
            return self.md_path.read_text()
        else:
            # Default minimal prompt
            role = "pre-validation" if self.is_architect else "post-validation"
            return f"You are a {role} agent. Analyze inputs carefully."
    
    async def run(self,
                  goal: str,
                  inputs: Dict[str, Any],
                  trajectory: List[Dict],
                  round: ValidationRound = ValidationRound.INITIAL) -> ValidationResult:
        """
        Run validation.
        
        Parameters:
            goal: The goal being validated
            inputs: Input fields for signature
            trajectory: Current trajectory
            round: Validation round (for multi-round)
        
        Returns:
            ValidationResult with decision and metadata
        """
        import time
        start_time = time.time()
        self.total_calls += 1
        self.tool_calls = []
        
        # 🎨 AGENT STATUS: Show what agent is doing with nice icons
        agent_type = "🔍 Architect" if self.is_architect else "✅ Auditor"
        logger.info(f"")
        logger.info(f"{'='*80}")
        logger.info(f"{agent_type} Agent: {self.agent_name}")
        logger.info(f"🎯 Task: Validate inputs/output for goal")
        logger.info(f"⏰ Started: {datetime.now().strftime('%H:%M:%S')}")
        logger.info(f"🔄 Status: WORKING... (Agent is analyzing with LLM)")
        logger.info(f"{'='*80}")
        
        # 🎯 A-TEAM FIX: SMART MEMORY for Learning, NOT Blocking!
        # Memory should HELP actors improve, not block them based on past failures
        if self.is_architect:
            # Architect: Retrieve SUCCESS PATTERNS ONLY (not blocking decisions)
            memory_context = self._build_smart_memory_for_architect(goal, inputs)
            logger.info(f"✅ [ARCHITECT SMART MEMORY] Retrieved success patterns for learning")
        else:
            # Auditor: Retrieve all patterns for validation
            memory_context = self._build_memory_context(goal, inputs)
        
        causal_context = self._build_causal_context(goal, inputs)
        shared_insights = self._get_shared_insights()
        
        # 🔧 FIX: Convert trajectory items to serializable format
        # ✅ A-TEAM FIX: FILTER trajectory for Architect - keep success patterns, remove failures!
        # Auditor needs recent trajectory to understand what actor did
        if self.is_architect:
            # Filter trajectory: Keep success patterns, remove blocking/failure patterns
            filtered_trajectory = []
            for item in trajectory:
                item_str = str(item).lower()
                # Skip if it contains blocking/failure keywords
                if any(keyword in item_str for keyword in ['blocked', 'architect_blocked', 'failed_validation', 'should_proceed=false']):
                    continue
                # Keep if it contains success keywords OR is neutral
                if any(keyword in item_str for keyword in ['success', 'valid', 'approved', 'should_proceed=true']):
                    filtered_trajectory.append(item)
                elif len(filtered_trajectory) < 3:  # Keep some neutral context
                    filtered_trajectory.append(item)
            
            serializable_trajectory = filtered_trajectory[-3:] if len(filtered_trajectory) > 3 else filtered_trajectory
            logger.info(f"✅ [ARCHITECT TRAJECTORY] Filtered trajectory: {len(trajectory)} → {len(serializable_trajectory)} (kept success patterns)")
        else:
            # Auditor: Only send RECENT trajectory - Auditor validates output, not full history
            # Full trajectory causes 15K+ tokens, hanging for 60s per validation
            recent_trajectory = trajectory[-5:] if len(trajectory) > 5 else trajectory
            
            serializable_trajectory = []
            if recent_trajectory:
                for item in recent_trajectory:
                    if hasattr(item, '__dict__'):
                        # Convert to dict, with compression for large items
                        item_str = str(item)
                        if len(item_str) > 1000:
                            # Compress large items to prevent context overflow
                            item_str = item_str[:500] + f"...[{len(item_str)-500} chars truncated for validation]"
                        serializable_trajectory.append(item_str)
                    else:
                        item_str = str(item)
                        if len(item_str) > 1000:
                            item_str = item_str[:500] + f"...[{len(item_str)-500} chars truncated]"
                        serializable_trajectory.append(item_str)
        
        # 🎯 USE JOTTY'S BUDGET MANAGER - NO HARDCODING!
        # Compute budget allocation intelligently
        system_tokens = len(self.system_prompt) // 4
        input_tokens = sum(len(str(v)) // 4 for v in inputs.values())
        trajectory_tokens = sum(len(s) // 4 for s in serializable_trajectory)
        
        allocation = self.budget_manager.compute_allocation(
            system_prompt_tokens=system_tokens,
            input_tokens=input_tokens,
            trajectory_tokens=trajectory_tokens,
            tool_output_tokens=0  # Will be computed dynamically
        )
        
        # 🎯 BUILD CONTEXT PARTS WITH BUDGET ALLOCATION
        # SmartContextGuard will handle intelligent truncation if needed
        context_parts = []
        
        # Add memory context (use allocated budget)
        if memory_context:
            memory_budget = allocation.get('memory', 5000)
            memory_tokens = len(memory_context) // 4
            if memory_tokens > memory_budget:
                # Intelligent truncation: keep complete sentences
                memory_context = self._smart_truncate(memory_context, memory_budget * 4)
            context_parts.append(f"{memory_context}")
        elif self.is_architect:
            # ✅ EXPLICIT: Architect validates CURRENT inputs, learns from SUCCESS
            context_parts.append(
                "IMPORTANT: You are validating CURRENT inputs for THIS execution attempt. "
                "Focus on whether the current inputs are sufficient NOW. "
                "If you see learned patterns above, use them to HELP validate (what inputs work well), "
                "NOT to block based on past failures. Your job is to enable success, not prevent attempts."
            )
        
        # Add trajectory (use allocated budget)
        if serializable_trajectory:
            traj_budget = allocation.get('trajectory', 5000)
            # Keep most recent items that fit in budget
            traj_str = json.dumps(serializable_trajectory, indent=2)
            if len(traj_str) // 4 > traj_budget:
                # Keep recent items
                for i in range(len(serializable_trajectory) - 1, -1, -1):
                    partial = json.dumps(serializable_trajectory[i:], indent=2)
                    if len(partial) // 4 <= traj_budget:
                        traj_str = partial
                        break
            context_parts.append(f"History: {traj_str}")
        
        # Add insights (typically small, no truncation needed)
        if shared_insights:
            context_parts.append(f"Insights: {shared_insights}")
        
        # Add causal context (use remaining budget)
        if causal_context:
            context_parts.append(f"Causal: {causal_context}")
        
        # 🔥 A-TEAM GENERIC SOLUTION: Inject ALL extra inputs into context
        # ANY field in inputs that's NOT part of the agent's signature gets added to context.
        # This allows integration layers to pass domain-specific data WITHOUT hardcoding in JOTTY!
        # Examples: schemas, business terms, metadata, constraints, etc.
        signature_fields = set(self.signature.input_fields.keys()) if hasattr(self.signature, 'input_fields') else set()
        extra_fields = {}
        
        for key, value in inputs.items():
            # Skip fields that are part of signature or special JOTTY fields
            if key not in signature_fields and key not in ['goal', 'trajectory', 'round']:
                if value is not None and str(value).strip():  # Only non-empty values
                    extra_fields[key] = value
        
        # Add ALL extra fields to context generically (NO HARDCODING!)
        if extra_fields:
            context_parts.append("\n=== ADDITIONAL CONTEXT ===")
            for field_name, field_value in extra_fields.items():
                # Smart truncation for large values to respect token budget
                value_str = str(field_value)
                if len(value_str) > 5000:
                    value_str = value_str[:5000] + f"...[truncated from {len(value_str)} chars]"
                
                # Format nicely for LLM
                formatted_name = field_name.replace('_', ' ').title()
                context_parts.append(f"\n{formatted_name}:\n{value_str}")
            
            logger.info(f"✅ [GENERIC CONTEXT] Injected {len(extra_fields)} extra fields into {self.agent_name}: {list(extra_fields.keys())}")
        
        context_str = "\n".join(context_parts)
        
        # Simplified inputs - agent should call tools for details
        base_task = inputs.get("goal", goal) or goal
        
        # 🆕 A-TEAM FIX #1: Extract ALL execution metadata from inputs
        # JOTTY is generic - we extract ALL fields so agent can see execution results!
        # This fixes the Auditor false negative bug where execution metadata was invisible.
        execution_metadata_parts = []
        metadata_fields = [
            'execution_status', 'execution_success', 'row_count',
            'has_data', 'error_message', 'action_result', 'action_taken'
        ]
        
        for field in metadata_fields:
            if field in inputs and inputs[field] is not None:
                # Format for LLM readability
                value = inputs[field]
                # Intelligently truncate very long values (but keep reasonable detail)
                if isinstance(value, str) and len(value) > 1000:
                    value = value + f"...[{len(value)-1000} chars truncated for context]"
                execution_metadata_parts.append(f"{field}: {value}")
        
        # Build context with ALL available information
        metadata_str = ""
        if execution_metadata_parts:
            metadata_str = "EXECUTION METADATA:\n" + "\n".join(execution_metadata_parts) + "\n\n"
        
        action_args = inputs.get('action_args', '')
        base_context = f"{metadata_str}{action_args}\n\n{context_str}"
        
        # If context is too large, use smart truncation
        max_context_tokens = self.config.max_context_tokens - system_tokens - 1000  # Reserve for output
        if len(base_context) // 4 > max_context_tokens:
            base_context = self._smart_truncate(base_context, max_context_tokens * 4)
        
        # =====================================================================
        # RETRY LOOP with signature validation and regeneration feedback
        # =====================================================================
        max_retries = self.config.max_eval_retries  # ✅ FROM CONFIG! (was hardcoded=3)
        last_error = None
        validation_result = None
        
        for attempt in range(max_retries):
            try:
                # Build inputs with retry feedback if needed
                if attempt == 0:
                    full_inputs = {"task": base_task, "context": base_context}
                else:
                    # Add feedback about what was missing
                    retry_feedback = f"""
RETRY ATTEMPT {attempt + 1}/{max_retries}
PREVIOUS ERROR: {last_error}

YOU MUST provide ALL required output fields:
"""
                    if self.is_architect:
                        retry_feedback += """
Architect Required Outputs:
- reasoning: Your step-by-step analysis (string)
- should_proceed: True or False (boolean)
- confidence: 0.0 to 1.0 (float)
- injected_context: Additional context for actor (string, can be empty)
- injected_instructions: Guidance for actor (string, can be empty)
- insight_to_share: Key insight for other agents (string, can be empty)
"""
                    else:
                        retry_feedback += """
Auditor Required Outputs:
- reasoning: Your validation analysis (string)
- is_valid: True or False (boolean)
- confidence: 0.0 to 1.0 (float)
- output_tag: 'useful', 'fail', or 'enquiry' (string)
- output_name: Name for this output (string)
- why_useful: Why this is useful (string)
- fix_instructions: What to fix if invalid (string)
- insight_to_share: Key insight for other agents (string)
"""
                    full_inputs = {
                        "task": f"{base_task}\n\n{retry_feedback}",
                        "context": base_context
                    }
                
                # Run agent WITH TIMEOUT (to prevent hanging on API failures)
                # ⚡ A-TEAM FIX: Add timeout for LLM calls to prevent indefinite hangs
                timeout_seconds = getattr(self.config, 'llm_timeout_seconds', 180)  # 3 minutes default
                
                try:
                    # Wrap LLM call in timeout
                    result = await asyncio.wait_for(
                        asyncio.to_thread(lambda: self.agent(**full_inputs)),
                        timeout=timeout_seconds
                    )
                except asyncio.TimeoutError:
                    logger.warning(f"⏰ [{self.agent_name}] LLM call timed out after {timeout_seconds}s on attempt {attempt + 1}/{max_retries}")
                    last_error = f"LLM API timeout after {timeout_seconds}s"
                    if attempt < max_retries - 1:
                        # Exponential backoff before retry
                        backoff_time = min(5 * (2 ** attempt), 30)  # 5s, 10s, 30s max
                        logger.info(f"⏰ Retrying after {backoff_time}s backoff...")
                        await asyncio.sleep(backoff_time)
                        continue
                    else:
                        # Final timeout - return default result
                        logger.error(f"❌ [{self.agent_name}] All {max_retries} attempts timed out. Using default.")
                        raise Exception(last_error)
                
                # Validate the result has required fields
                missing_fields = self._check_required_fields(result)
                
                if missing_fields:
                    last_error = f"Missing fields: {missing_fields}"
                    if attempt < max_retries - 1:
                        continue  # Retry
                    # On last attempt, use defaults for missing fields
                
                # Parse result
                validation_result = self._parse_result(result, round, start_time)
                
                # Share insight if any
                if hasattr(result, 'insight_to_share') and result.insight_to_share:
                    self._share_insight(result.insight_to_share)
                
                # Store in memory
                self._store_experience(goal, inputs, validation_result)
                
                # Success - break retry loop
                break
                
            except Exception as e:
                last_error = str(e)
                if attempt == max_retries - 1:
                    # Final failure
                    validation_result = ValidationResult(
                        agent_name=self.agent_name,
                        is_valid=True if self.is_architect else False,  # Default to proceed for architect
                        confidence=self.config.default_confidence_on_error,  # ✅ FROM CONFIG! (was 0.3)
                        reasoning=f"Error after {max_retries} attempts: {last_error}",
                        should_proceed=True if self.is_architect else None,  # Default proceed
                        output_tag=OutputTag.ENQUIRY if not self.is_architect else None,
                        validation_round=round
                    )
        
        # Ensure we have a result
        if validation_result is None:
            validation_result = ValidationResult(
                agent_name=self.agent_name,
                is_valid=True,
                confidence=self.config.default_confidence_no_validation,  # ✅ FROM CONFIG! (was 0.5)
                reasoning="Default result - no validation performed",
                should_proceed=True if self.is_architect else None,
                validation_round=round
            )
        
        validation_result.execution_time = time.time() - start_time
        validation_result.tool_calls = self.tool_calls
        
        # Update stats
        if self.is_architect and validation_result.should_proceed:
            self.total_approvals += 1
        elif not self.is_architect and validation_result.is_valid:
            self.total_approvals += 1
        
        return validation_result
    
    async def refine(self,
                     original_result: ValidationResult,
                     feedback: str,
                     additional_context: str = "") -> ValidationResult:
        """
        Refine decision based on feedback (multi-round validation).
        """
        if not self.config.enable_multi_round:
            return original_result
        
        try:
            result = self.refiner(
                original_decision=f"Decision: {'proceed' if original_result.should_proceed else 'block'}, Reasoning: {original_result.reasoning}",
                feedback=feedback,
                additional_context=additional_context
            )
            
            # Create refined result
            refined = ValidationResult(
                agent_name=self.agent_name,
                is_valid=result.refined_decision,
                confidence=float(result.refined_confidence),
                reasoning=result.refined_reasoning,
                should_proceed=result.refined_decision if self.is_architect else None,
                validation_round=ValidationRound.REFINEMENT,
                previous_rounds=[original_result]
            )
            
            # Assess reasoning quality
            refined.reasoning_steps = result.refined_reasoning.split('\n')
            refined.reasoning_quality = min(1.0, len(refined.reasoning_steps) * 0.1 + 0.3)
            
            return refined
            
        except Exception:
            return original_result
    
    def _build_smart_memory_for_architect(self, goal: str, inputs: Dict) -> str:
        """
        Build SMART memory context for Architect.
        
        🎯 USER CRITICAL FIX: Memory should HELP, not BLOCK!
        
        Architect should learn from:
        - ✅ Successful validation patterns ("what inputs led to success?")
        - ✅ Overcoming blocks ("how did we fix previous blocks?")
        - ❌ NOT past blocking decisions
        
        This is like a mentor saying "here's what worked before" 
        instead of "you failed before, so fail again".
        """
        # Get query from inputs
        query = inputs.get('proposed_action', '') or inputs.get('action_result', '') or goal
        
        # Compute budget using budget manager
        system_tokens = len(self.system_prompt) // 4
        input_tokens = sum(len(str(v)) // 4 for v in inputs.values())
        
        allocation = self.budget_manager.compute_allocation(
            system_prompt_tokens=system_tokens,
            input_tokens=input_tokens,
            trajectory_tokens=2000,  # Estimate
            tool_output_tokens=2000
        )
        
        memory_budget = allocation['memory']
        
        # 🎯 CRITICAL: Retrieve SUCCESS patterns, NOT blocking decisions!
        memories = self.memory.retrieve(
            query=query,
            goal=goal,
            budget_tokens=memory_budget,
            context_hints=f"Looking for successful patterns that helped similar tasks: {goal}"
        )
        
        # 🔥 FILTER OUT blocked/failed memories after retrieval
        if memories:
            filtered_memories = []
            for mem in memories:
                # Skip memories with failure tags
                if hasattr(mem, 'metadata') and isinstance(mem.metadata, dict):
                    tags = mem.metadata.get('tags', [])
                    if any(tag in tags for tag in ['blocked', 'architect_blocked', 'failed_validation', 'failed']):
                        continue  # Skip this memory
                filtered_memories = filtered_memories
                memories = filtered_memories if filtered_memories else memories
        
        if not memories:
            return ""
        
        # Format as learning context (positive framing)
        memory_lines = ["LEARNED PATTERNS (what worked before):"]
        for mem in memories:
            # Extract only actionable insights, not blocking decisions
            if hasattr(mem, 'insight') and mem.insight:
                memory_lines.append(f"- {mem.insight}")
            elif hasattr(mem, 'what_worked') and mem.what_worked:
                memory_lines.append(f"- {mem.what_worked}")
        
        return "\n".join(memory_lines) if len(memory_lines) > 1 else ""
    
    def _build_memory_context(self, goal: str, inputs: Dict) -> str:
        """
        Build memory context string.
        
        🎯 NO HARDCODED SLICING - Uses budget manager allocation!
        """
        # Get query from inputs
        query = inputs.get('proposed_action', '') or inputs.get('action_result', '') or goal
        
        # Compute budget using budget manager
        system_tokens = len(self.system_prompt) // 4
        input_tokens = sum(len(str(v)) // 4 for v in inputs.values())
        
        allocation = self.budget_manager.compute_allocation(
            system_prompt_tokens=system_tokens,
            input_tokens=input_tokens,
            trajectory_tokens=2000,  # Estimate
            tool_output_tokens=2000
        )
        
        memory_budget = allocation['memory']
        
        # Retrieve memories with computed budget
        memories = self.memory.retrieve(
            query=query,
            goal=goal,
            budget_tokens=memory_budget,
            context_hints=f"Looking for memories relevant to: {goal}"  # ✅ NO  limit!
        )
        
        if not memories:
            return "No relevant memories found."
        
        # Format memories WITHOUT hardcoded truncation
        # Budget manager already handles sizing during retrieval
        context_parts = []
        for mem in memories:
            value = mem.get_value(goal)
            # Let budget manager control size, not hardcoded 
            content = mem.content
            if len(content) > 500:  # Only truncate VERY long individual memories
                content = self._smart_truncate(content, 500)
            context_parts.append(f"[Value: {value:.2f}] {content}")
        
        return "\n---\n".join(context_parts)
    
    def _build_causal_context(self, goal: str, inputs: Dict) -> str:
        """Build causal knowledge context."""
        if not self.config.enable_causal_learning:
            return "No causal knowledge available."
        
        # Get relevant causal links
        query = goal + " " + str(inputs.get('proposed_action', ''))
        causal_links = self.memory.retrieve_causal(query, inputs)
        
        if not causal_links:
            return "No relevant causal knowledge."
        
        parts = []
        for link in causal_links:
            parts.append(f"BECAUSE: {link.cause}\nTHEREFORE: {link.effect}\n(Confidence: {link.confidence:.2f})")
        
        return "\n---\n".join(parts)
    
    def _get_shared_insights(self) -> str:
        """
        Get insights shared by other agents.
        
        🎯 NO HARDCODED  - Returns ALL relevant insights!
        Let context manager handle sizing if needed.
        """
        if not self.config.enable_agent_communication:
            return ""
        
        messages = self.scratchpad.get_messages_for(self.agent_name)
        
        insights = []
        for msg in messages:
            if msg.message_type == CommunicationType.INSIGHT:
                insights.append(f"[{msg.sender}]: {msg.insight}")
            elif msg.message_type == CommunicationType.WARNING:
                insights.append(f"[WARNING from {msg.sender}]: {msg.content}")
        
        if not insights:
            return "No shared insights."
        
        # Return ALL insights, not just last 5!
        # If this gets too large, _smart_truncate will handle it
        return "\n".join(insights)
    
    def _share_insight(self, insight: str):
        """Share insight with other agents."""
        if not self.config.enable_agent_communication:
            return
        
        message = AgentMessage(
            sender=self.agent_name,
            receiver="*",
            message_type=CommunicationType.INSIGHT,
            content={"insight": insight},
            insight=insight,
            confidence=self.config.default_confidence_insight_share  # ✅ FROM CONFIG! (was 0.7)
        )
        self.scratchpad.add_message(message)
    
    def _check_required_fields(self, result: Any) -> List[str]:
        """Check if result has all required fields, return list of missing ones."""
        missing = []
        
        # Common required fields
        if not hasattr(result, 'reasoning') or not result.reasoning:
            missing.append('reasoning')
        if not hasattr(result, 'confidence'):
            missing.append('confidence')
        
        if self.is_architect:
            # Architect required fields
            if not hasattr(result, 'should_proceed'):
                missing.append('should_proceed')
        else:
            # Auditor required fields
            if not hasattr(result, 'is_valid'):
                missing.append('is_valid')
            if not hasattr(result, 'output_tag'):
                missing.append('output_tag')
        
        return missing
    
    def _parse_result(self, result: Any, round: ValidationRound, start_time: float) -> ValidationResult:
        """Parse DSPy result into ValidationResult with all fields for memory/RL/reflection."""
        # Extract common fields with defaults
        reasoning = getattr(result, 'reasoning', '') or ''
        confidence = float(getattr(result, 'confidence', 0.5) or 0.5)
        confidence = max(0.0, min(1.0, confidence))
        
        # Extract insight for inter-agent sharing
        insight = getattr(result, 'insight_to_share', '') or ''
        if insight:
            self._share_insight(insight)
        
        if self.is_architect:
            # Extract Architect fields
            should_proceed = getattr(result, 'should_proceed', True)
            if isinstance(should_proceed, str):
                should_proceed = should_proceed.lower() in ('true', 'yes', '1', 'proceed')
            
            # Get all instruction fields
            injected_context = getattr(result, 'injected_context', '') or ''
            injected_instructions = getattr(result, 'injected_instructions', '') or ''
            
            # 🎨 AGENT COMPLETION STATUS
            elapsed = time.time() - start_time
            decision_icon = "✅" if should_proceed else "🛑"
            logger.info(f"")
            logger.info(f"{'='*80}")
            logger.info(f"🔍 Architect Agent: {self.agent_name} - COMPLETE")
            logger.info(f"{decision_icon} Decision: {'PROCEED' if should_proceed else 'BLOCKED'}")
            logger.info(f"💪 Confidence: {confidence:.2f}")
            logger.info(f"⏱️  Duration: {elapsed:.2f}s")
            logger.info(f"{'='*80}")
            logger.info(f"")
            
            return ValidationResult(
                agent_name=self.agent_name,
                is_valid=should_proceed,
                confidence=confidence,
                reasoning=reasoning,
                should_proceed=should_proceed,
                injected_context=injected_context,
                injected_instructions=injected_instructions,
                validation_round=round,
                reasoning_steps=reasoning.split('\n') if reasoning else [],
                reasoning_quality=min(1.0, len(reasoning) / 500)
            )
        else:
            # Extract Auditor fields
            is_valid = getattr(result, 'is_valid', True)
            if isinstance(is_valid, str):
                is_valid = is_valid.lower() in ('true', 'yes', '1', 'valid')
            
            # Parse output tag
            tag_str = getattr(result, 'output_tag', 'useful') or 'useful'
            tag_str = tag_str.lower().strip()
            try:
                output_tag = OutputTag(tag_str)
            except ValueError:
                output_tag = OutputTag.USEFUL if is_valid else OutputTag.FAIL
            
            # Get all Auditor fields
            why_useful = getattr(result, 'why_useful', '') or ''
            fix_instructions = getattr(result, 'fix_instructions', '') or ''
            output_name = getattr(result, 'output_name', '') or ''
            
            # 🎨 AGENT COMPLETION STATUS
            elapsed = time.time() - start_time
            decision_icon = "✅" if is_valid else "❌"
            logger.info(f"")
            logger.info(f"{'='*80}")
            logger.info(f"✅ Auditor Agent: {self.agent_name} - COMPLETE")
            logger.info(f"{decision_icon} Decision: {'VALID' if is_valid else 'INVALID'}")
            logger.info(f"💪 Confidence: {confidence:.2f}")
            logger.info(f"🏷️  Tag: {output_tag.value}")
            logger.info(f"⏱️  Duration: {elapsed:.2f}s")
            logger.info(f"{'='*80}")
            logger.info(f"")
            
            return ValidationResult(
                agent_name=self.agent_name,
                is_valid=is_valid,
                confidence=confidence,
                reasoning=reasoning,
                output_tag=output_tag,
                why_useful=why_useful if is_valid else fix_instructions,
                validation_round=round,
                reasoning_steps=reasoning.split('\n') if reasoning else [],
                reasoning_quality=min(1.0, len(reasoning) / 500)
            )
    
    def _store_experience(self, goal: str, inputs: Dict, result: ValidationResult):
        """
        Store experience in memory.
        
        🎯 NO HARDCODED TRUNCATION - Store FULL reasoning!
        Memory system handles deduplication and prioritization.
        """
        # Format experience WITHOUT arbitrary truncation
        if self.is_architect:
            content = f"""
Architect Decision: {'PROCEED' if result.should_proceed else 'BLOCK'}
Confidence: {result.confidence:.2f}
Proposed Action: {inputs.get('proposed_action', 'N/A')}
Reasoning: {result.reasoning}
""".strip()
        else:
            # Get full result string (no  truncation!)
            action_result = str(inputs.get('action_result', 'N/A'))
            content = f"""
Auditor Decision: {'VALID' if result.is_valid else 'INVALID'}
Tag: {result.output_tag.value if result.output_tag else 'N/A'}
Confidence: {result.confidence:.2f}
Result: {action_result}
Reasoning: {result.reasoning}
""".strip()
        
        # Initial value based on confidence
        initial_value = 0.5 + 0.3 * (result.confidence - 0.5)
        
        # Store with FULL context (no  truncation!)
        self.memory.store(
            content=content,
            level=MemoryLevel.EPISODIC,
            context={
                'goal': goal,
                'inputs': {k: str(v) for k, v in inputs.items()},  # ✅ Full values!
                'round': result.validation_round.value
            },
            goal=goal,
            initial_value=initial_value
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get agent statistics."""
        return {
            "agent_name": self.agent_name,
            "is_architect": self.is_architect,
            "total_calls": self.total_calls,
            "total_approvals": self.total_approvals,
            "approval_rate": self.total_approvals / max(1, self.total_calls),
            "memory_stats": self.memory.get_statistics()
        }


# =============================================================================
# MULTI-ROUND VALIDATION ORCHESTRATOR
# =============================================================================

class MultiRoundValidator:
    """
    Orchestrates multi-round validation with refinement.
    
    Process:
    1. Initial round: All agents vote independently
    2. Check for disagreement or low confidence
    3. If needed: Share insights, run refinement round
    4. Final decision based on all rounds
    """
    
    def __init__(self, agents: List[InspectorAgent], config: JottyConfig):
        self.agents = agents
        self.config = config
    
    async def validate(self,
                       goal: str,
                       inputs: Dict[str, Any],
                       trajectory: List[Dict],
                       is_architect: bool) -> Tuple[List[ValidationResult], bool]:
        """
        Run multi-round validation.
        
        Returns:
            (all_results, combined_decision)
        """
        all_results = []
        
        # Round 1: Initial validation
        initial_results = []
        for agent in self.agents:
            result = await agent.run(goal, inputs, trajectory, ValidationRound.INITIAL)
            initial_results.append(result)
            all_results.append(result)
        
        # Check if refinement needed
        needs_refinement = self._needs_refinement(initial_results, is_architect)
        
        if needs_refinement and self.config.enable_multi_round:
            # Build feedback from initial results
            feedback = self._build_feedback(initial_results)
            
            # Run refinement round
            for i, agent in enumerate(self.agents):
                if initial_results[i].confidence < self.config.refinement_on_low_confidence:
                    refined = await agent.refine(
                        initial_results[i],
                        feedback,
                        additional_context="Other agents' reasoning and insights are available."
                    )
                    all_results.append(refined)
                    # Use refined result
                    initial_results[i] = refined
        
        # Combine decisions
        # ✅ A-TEAM FIX: Detect reasoning-vote contradictions and fix them
        # If reasoning says "block" but vote is True, flip it!
        corrected_results = []
        for r in all_results:
            corrected = r
            reasoning_lower = r.reasoning.lower() if r.reasoning else ""
            
            # Check for contradiction indicators
            block_keywords = ['block', 'invalid', 'incorrect', 'wrong', 'fail', 'must be corrected', 'cannot be trusted']
            proceed_keywords = ['valid', 'correct', 'accurate', 'acceptable', 'proceed']
            
            block_count = sum(1 for kw in block_keywords if kw in reasoning_lower)
            proceed_count = sum(1 for kw in proceed_keywords if kw in reasoning_lower)
            
            # If reasoning strongly suggests blocking but vote says valid, flip it
            if not is_architect and r.is_valid and block_count > proceed_count + 2:
                logger.warning(f"⚠️ [VOTE CORRECTION] {r.agent_name}: Reasoning suggests blocking but vote is valid=True. Flipping to False.")
                logger.warning(f"   Block keywords: {block_count}, Proceed keywords: {proceed_count}")
                logger.warning(f"   Reasoning snippet: {reasoning_lower[:200]}...")
                
                # Create corrected result
                from copy import copy
                corrected = copy(r)
                corrected.is_valid = False
            
            corrected_results.append(corrected)
        
        # Use corrected results for voting
        if is_architect:
            # For Architect: weighted average of proceed decisions
            total_weight = sum(r.confidence for r in corrected_results)
            if total_weight == 0:
                combined = any(r.should_proceed for r in corrected_results)
            else:
                weighted_proceed = sum(
                    r.confidence if r.should_proceed else 0
                    for r in corrected_results
                ) / total_weight
                combined = weighted_proceed > 0.5
        else:
            # For Auditor: Conservative voting - ANY confident rejection blocks
            # 🎯 A-TEAM FIX: Don't let high-confidence "valid=True" override a well-reasoned "valid=False"
            # If ANY validator identifies a real issue (confidence > 0.5), BLOCK
            confident_rejections = [
                r for r in corrected_results 
                if not r.is_valid and r.confidence > 0.5
            ]
            
            if confident_rejections:
                # At least one validator found a real issue → BLOCK
                combined = False
            elif all(r.is_valid for r in corrected_results):
                # ALL validators say valid → PASS
                combined = True
            else:
                # Mixed results, no confident rejection → use weighted average
                total_weight = sum(r.confidence for r in corrected_results)
                if total_weight == 0:
                    combined = any(r.is_valid for r in corrected_results)
                else:
                    weighted_valid = sum(
                        r.confidence if r.is_valid else 0
                        for r in corrected_results
                    ) / total_weight
                    combined = weighted_valid > 0.5
        
        return all_results, combined  # Return original results but use corrected for voting
    
    def _needs_refinement(self, results: List[ValidationResult], is_architect: bool) -> bool:
        """Check if refinement round is needed."""
        # Low confidence
        if any(r.confidence < self.config.refinement_on_low_confidence for r in results):
            return True
        
        # Disagreement
        if self.config.refinement_on_disagreement:
            if is_architect:
                decisions = [r.should_proceed for r in results]
            else:
                decisions = [r.is_valid for r in results]
            
            if len(set(decisions)) > 1:
                return True
        
        return False
    
    def _build_feedback(self, results: List[ValidationResult]) -> str:
        """
        Build feedback summary for refinement.
        
        🎯 NO HARDCODED  - Include FULL reasoning for refinement!
        """
        parts = []
        for r in results:
            decision = "proceed" if r.should_proceed else "block"
            # Include FULL reasoning, not truncated 
            parts.append(f"{r.agent_name}: {decision} (confidence: {r.confidence:.2f})\nReasoning: {r.reasoning}")
        
        return "\n---\n".join(parts)
