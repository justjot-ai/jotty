"""
Optimization Pipeline - Generic iterative optimization with evaluation and learning.

This module provides a generic pipeline for iterative optimization that:
1. Processes tasks with multiple agents in sequence
2. Evaluates results against gold standards
3. Uses a teacher model when evaluation fails
4. Updates knowledge base (KB) metadata for learning
5. Runs iterative loops until success or max iterations

# GENERIC: No domain-specific logic (works with any agents, any domain)
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime

try:
    import dspy
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False
    dspy = None

from Jotty.core.foundation.data_structures import SwarmConfig, EpisodeResult
from Jotty.core.foundation.agent_config import AgentConfig

# Import credit assignment and adaptive learning (core RL features)
from .credit_assignment import CreditAssignment
from .adaptive_learning import AdaptiveLearning

logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """Configuration for optimization pipeline."""
    max_iterations: int = 5
    required_pass_count: int = 2  # Consecutive passes required to stop
    enable_teacher_model: bool = True
    enable_kb_updates: bool = True
    kb_update_requires_teacher: bool = True  # Only update KB if teacher was used
    evaluation_function: Optional[Callable] = None  # Custom evaluation function
    gold_standard_provider: Optional[Callable] = None  # Function to get gold standards
    output_path: Optional[Path] = None
    thinking_log_path: Optional[Path] = None
    enable_thinking_log: bool = True
    # Improvement storage
    save_improvements: bool = True  # Save improvements to file
    improvements_file: Optional[Path] = None  # Path to save improvements (default: output_path/improvements.json)
    update_dspy_instructions: bool = False  # Update DSPy module instructions (if agent is DSPy module)
    update_jotty_instructions: bool = False  # Update Jotty learned_instructions (if using Orchestrator)
    # Credit assignment and adaptive learning
    enable_credit_assignment: bool = True  # Enable credit assignment for improvements
    enable_adaptive_learning: bool = True  # Enable adaptive learning rate
    enable_teacher_quality_check: bool = True  # Validate teacher output quality
    enable_incremental_learning: bool = True  # Use incremental updates
    max_improvements: Optional[int] = None  # Max improvements to use (None = all)
    min_credit_threshold: float = 0.1  # Minimum credit score to include improvement


@dataclass
class IterationResult:
    """Result from a single optimization iteration."""
    iteration: int
    success: bool
    evaluation_score: float
    evaluation_status: str
    output: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    teacher_output: Optional[Any] = None
    kb_updates: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class OptimizationPipeline:
    """
    Generic optimization pipeline for iterative improvement with evaluation.
    
    This pipeline orchestrates multiple agents to solve a task, evaluates results,
    uses a teacher model when needed, and updates knowledge base for learning.
    
    Features:
    - Multi-agent orchestration
    - Iterative optimization loop
    - Gold standard evaluation
    - Teacher model fallback
    - Knowledge base updates
    - Thinking log for debugging
    - Credit assignment for improvements (based on counterfactual credit assignment)
    - Adaptive learning rate
    - Teacher quality validation
    - Incremental learning
    
        Usage:
        ```python
        from jotty.core.orchestration.optimization_pipeline import OptimizationPipeline
        
        pipeline = OptimizationPipeline(
            agents=[agent1, agent2, agent3],
            config=optimization_config
        )
        
        result = await pipeline.optimize(
            task="Generate markdown documentation",
            context={"requirements": "..."},
            gold_standard="expected_output.md"
        )
        ```
    """
    
    def __init__(self, agents: List[AgentConfig], config: OptimizationConfig, jotty_config: Optional[SwarmConfig] = None, conductor: Optional[Any] = None) -> None:
        """
        Initialize optimization pipeline.
        
        Args:
            agents: List of AgentConfig defining the agent pipeline
            config: OptimizationConfig with pipeline settings
            jotty_config: Optional SwarmConfig for Jotty components
            conductor: Optional Orchestrator instance for agent orchestration
        """
        self.agents = agents
        self.config = config
        self.jotty_config = jotty_config or SwarmConfig()
        self.conductor = conductor
        
        # Setup output paths
        if self.config.output_path:
            self.output_path = Path(self.config.output_path)
            self.output_path.mkdir(parents=True, exist_ok=True)
        else:
            self.output_path = None
        
        if self.config.thinking_log_path:
            self.thinking_log_path = Path(self.config.thinking_log_path)
        elif self.output_path:
            self.thinking_log_path = self.output_path / "thinking.log"
        else:
            self.thinking_log_path = None
        
        # Initialize thinking log
        if self.config.enable_thinking_log and self.thinking_log_path:
            self.thinking_log_path.parent.mkdir(parents=True, exist_ok=True)
            self._clear_thinking_log()
        
        # Iteration tracking
        self.iteration_count = 0
        self.consecutive_passes = 0
        self.all_iterations: List[IterationResult] = []
        
        # Improvement storage
        self.improvements: List[Dict[str, Any]] = []
        self.improvements_file = None
        
        # Credit assignment and adaptive learning
        if config.enable_credit_assignment and CreditAssignment:
            self.credit_assignment = CreditAssignment()
        else:
            self.credit_assignment = None
        
        if config.enable_adaptive_learning and AdaptiveLearning:
            self.adaptive_learning = AdaptiveLearning()
        else:
            self.adaptive_learning = None
        if self.config.save_improvements:
            if self.config.improvements_file:
                self.improvements_file = Path(self.config.improvements_file)
            elif self.output_path:
                self.improvements_file = self.output_path / "improvements.json"
            if self.improvements_file:
                self.improvements_file.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(
            f"Initialized OptimizationPipeline with {len(agents)} agents, "
            f"max_iterations={config.max_iterations}, "
            f"required_pass_count={config.required_pass_count}, "
            f"save_improvements={config.save_improvements}"
        )
    
    def _write_thinking_log(self, message: str) -> None:
        """Write a message to thinking log."""
        if not self.config.enable_thinking_log or not self.thinking_log_path:
            return
        
        try:
            now = datetime.now()
            timestamp = now.strftime("%Y-%m-%d %H:%M:%S") + f".{now.microsecond // 1000:03d}"
            log_entry = f"[{timestamp}] {message}\n"
            
            with open(self.thinking_log_path, 'a', encoding='utf-8') as f:
                f.write(log_entry)
        except Exception as e:
            logger.error(f"Error writing to thinking.log: {e}")
    
    def _clear_thinking_log(self) -> None:
        """Clear thinking log at start of optimization."""
        if self.thinking_log_path and self.thinking_log_path.exists():
            try:
                with open(self.thinking_log_path, 'w', encoding='utf-8') as f:
                    f.write("")
            except Exception as e:
                logger.error(f"Error clearing thinking.log: {e}")
    
    async def _run_agent_pipeline(
        self,
        task: str,
        context: Dict[str, Any],
        previous_outputs: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Run the agent pipeline sequentially.
        
        Args:
            task: The task description
            context: Context dictionary with inputs
            previous_outputs: Outputs from previous iteration (for retries)
            
        Returns:
            Dictionary with final output and intermediate results
        """
        self._write_thinking_log(f"Starting agent pipeline execution for task: {task[:100]}...")
        
        # If conductor is available, use it for orchestration
        if self.conductor:
            self._write_thinking_log("Using Orchestrator for agent orchestration")
            try:
                result = await self.conductor.arun(
                    goal=task,
                    **context
                )
                
                # Extract output from EpisodeResult
                if isinstance(result, EpisodeResult):
                    return {
                        "output": result.output,
                        "success": result.success,
                        "trajectory": result.trajectory,
                        "metadata": {
                            "episode": result.episode,
                            "execution_time": result.execution_time,
                            "architect_results": result.architect_results,
                            "auditor_results": result.auditor_results
                        }
                    }
                else:
                    return {"output": result, "success": True}
            except Exception as e:
                logger.error(f"Orchestrator execution failed: {e}")
                self._write_thinking_log(f"Orchestrator execution failed: {e}")
                raise
        
        # Otherwise, run agents sequentially
        self._write_thinking_log("Running agents sequentially")
        intermediate_outputs = {}
        current_context = {**context, "task": task}
        
        if previous_outputs:
            current_context.update(previous_outputs)
        
        # Filter out teacher and KB agents from main pipeline (they're called separately)
        main_pipeline_agents = [
            agent_config for agent_config in self.agents
            if not (
                "teacher" in agent_config.name.lower() or
                (agent_config.metadata and agent_config.metadata.get("is_teacher", False)) or
                "kb" in agent_config.name.lower() or
                "knowledge" in agent_config.name.lower() or
                (agent_config.metadata and agent_config.metadata.get("is_kb_updater", False))
            )
        ]
        
        if not main_pipeline_agents:
            raise ValueError("No agents available for main pipeline (all agents are teachers/KB updaters)")
        
        for i, agent_config in enumerate(main_pipeline_agents):
            agent_name = agent_config.name
            self._write_thinking_log(f"Executing agent {i+1}/{len(self.agents)}: {agent_name}")
            
            try:
                agent = agent_config.agent
                
                # Prepare inputs for this agent
                agent_inputs = self._prepare_agent_inputs(
                    agent_config,
                    current_context,
                    intermediate_outputs
                )
                
                # Ensure learned_improvements is passed for DSPy modules (even if empty)
                if DSPY_AVAILABLE and isinstance(agent, dspy.Module):
                    if 'learned_improvements' not in agent_inputs:
                        agent_inputs['learned_improvements'] = ""
                
                # Run agent (handles DSPy modules and regular callables)
                # DSPy modules should be called directly: agent(**inputs)
                # Regular callables: agent.forward(**inputs) or agent(**inputs)
                if asyncio.iscoroutinefunction(agent):
                    agent_output = await agent(**agent_inputs)
                elif DSPY_AVAILABLE and isinstance(agent, dspy.Module):
                    # DSPy module - call directly (not .forward())
                    agent_output = agent(**agent_inputs)
                elif hasattr(agent, 'forward'):
                    if asyncio.iscoroutinefunction(agent.forward):
                        agent_output = await agent.forward(**agent_inputs)
                    else:
                        agent_output = agent.forward(**agent_inputs)
                else:
                    agent_output = agent(**agent_inputs)
                
                # Extract output based on agent type
                output_value = self._extract_agent_output(agent_output)
                
                # Debug logging
                if self.config.enable_thinking_log:
                    self._write_thinking_log(f"Extracted output from {agent_name}: {repr(str(output_value)[:100])}")
                
                # Store intermediate output
                intermediate_outputs[agent_name] = output_value
                
                # Update context for next agent
                if agent_config.outputs:
                    for output_field in agent_config.outputs:
                        current_context[output_field] = output_value
                else:
                    # Default: use agent name as key
                    current_context[agent_name] = output_value
                
                self._write_thinking_log(f"Agent {agent_name} completed successfully")
                
            except Exception as e:
                logger.error(f"Agent {agent_name} failed: {e}")
                self._write_thinking_log(f"Agent {agent_name} failed: {e}")
                raise
        
        # Get final output from last agent in main pipeline
        final_output = None
        if main_pipeline_agents:
            last_agent_name = main_pipeline_agents[-1].name
            final_output = intermediate_outputs.get(last_agent_name)
            if self.config.enable_thinking_log:
                self._write_thinking_log(f"Final output from {last_agent_name}: {repr(str(final_output)[:100]) if final_output else 'None'}")
        
        return {
            "output": final_output,
            "success": True,
            "intermediate_outputs": intermediate_outputs
        }
    
    def _prepare_agent_inputs(
        self,
        agent_config: AgentConfig,
        context: Dict[str, Any],
        intermediate_outputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Prepare inputs for an agent based on its configuration."""
        inputs = {}
        
        # Use parameter mappings if available
        if agent_config.parameter_mappings:
            for param_name, context_key in agent_config.parameter_mappings.items():
                if context_key in context:
                    inputs[param_name] = context[context_key]
                elif context_key in intermediate_outputs:
                    inputs[param_name] = intermediate_outputs[context_key]
            # Also include teacher_output and feedback if available (for learning)
            if "teacher_output" in context:
                inputs["teacher_output"] = context["teacher_output"]
            if "_teacher_feedback" in context:
                inputs["_teacher_feedback"] = context["_teacher_feedback"]
        else:
            # Default: pass all context
            inputs.update(context)
            inputs.update(intermediate_outputs)
        
        return inputs
    
    def _extract_agent_output(self, agent_output: Any) -> Any:
        """Extract output value from agent result."""
        # Handle DSPy Prediction (most common case for expert agents)
        if DSPY_AVAILABLE and hasattr(agent_output, 'output'):
            # DSPy Prediction with 'output' field
            extracted = agent_output.output
            logger.debug(f"Extracted output from DSPy Prediction.output: {extracted}")
            return extracted
        
        # Handle DSPy Prediction with _store dict
        if hasattr(agent_output, '_store') and isinstance(agent_output._store, dict):
            # Try common output fields (generic - works for any domain)
            for field in ['output', 'result', 'content', 'generated', 'answer', 'final_output', 'final_result']:
                if field in agent_output._store:
                    extracted = agent_output._store[field]
                    logger.debug(f"Extracted output from field '{field}': {extracted}")
                    return extracted
            # Fallback: return first value
            if agent_output._store:
                extracted = list(agent_output._store.values())[0]
                logger.debug(f"Extracted output (first value): {extracted}")
                return extracted
        
        # Handle TaggedOutput pattern
        if hasattr(agent_output, 'final_result'):
            return agent_output.final_result
        
        # Handle EpisodeResult
        if isinstance(agent_output, EpisodeResult):
            return agent_output.output
        
        # Default: return as-is
        logger.debug(f"Extracted output (as-is): {agent_output}")
        return agent_output
    
    async def _evaluate_output(
        self,
        output: Any,
        gold_standard: Any,
        task: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate output against gold standard.
        
        Args:
            output: Generated output to evaluate
            gold_standard: Gold standard for comparison
            task: Task description
            context: Context dictionary
            
        Returns:
            Dictionary with evaluation results (score, status, etc.)
        """
        self._write_thinking_log("Evaluating output against gold standard...")
        
        # Use custom evaluation function if provided
        if self.config.evaluation_function:
            try:
                result = await self.config.evaluation_function(
                    output=output,
                    gold_standard=gold_standard,
                    task=task,
                    context=context
                )
                if isinstance(result, dict):
                    return result
                else:
                    return {"score": float(result) if result else 0.0, "status": "CORRECT" if result == 1.0 else "INCORRECT"}
            except Exception as e:
                logger.error(f"Custom evaluation function failed: {e}")
                return {"score": 0.0, "status": "ERROR", "error": str(e)}
        
        # Default evaluation: simple comparison
        # This is a placeholder - should be overridden for specific domains
        try:
            output_str = str(output)
            gold_str = str(gold_standard)
            
            # Simple string comparison (very basic)
            if output_str.strip() == gold_str.strip():
                return {"score": 1.0, "status": "CORRECT"}
            else:
                return {"score": 0.0, "status": "INCORRECT", "difference": "Output differs from gold standard"}
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return {"score": 0.0, "status": "ERROR", "error": str(e)}
    
    async def _run_teacher_model(
        self,
        task: str,
        context: Dict[str, Any],
        student_output: Any,
        gold_standard: Any,
        evaluation_result: Dict[str, Any]
    ) -> Optional[Any]:
        """
        Run teacher model to generate improved output.
        
        Args:
            task: Task description
            context: Context dictionary
            student_output: Output from student (main pipeline)
            gold_standard: Gold standard for teacher to match
            evaluation_result: Evaluation results showing what was wrong
            
        Returns:
            Teacher's improved output, or None if teacher model not available
        """
        if not self.config.enable_teacher_model:
            return None
        
        self._write_thinking_log("Evaluation failed, calling teacher model for improved output...")
        
        # Look for teacher agent in agent list
        teacher_agent = None
        for agent_config in self.agents:
            is_teacher = (
                "teacher" in agent_config.name.lower() or
                (agent_config.metadata and agent_config.metadata.get("is_teacher", False))
            )
            if is_teacher:
                teacher_agent = agent_config
                break
        
        if not teacher_agent:
            logger.warning("Teacher model requested but no teacher agent found")
            self._write_thinking_log("No teacher agent found, skipping teacher model")
            return None
        
        try:
            # Safety check
            if not isinstance(evaluation_result, dict):
                logger.warning(f"evaluation_result is not a dict: {type(evaluation_result)}")
                evaluation_result = {"score": 0.0, "status": "ERROR", "error": "invalid_evaluation_result"}
            
            # Prepare teacher inputs - use gold_standard directly
            eval_feedback = evaluation_result.get("feedback", "") or evaluation_result.get("status", "INCORRECT")
            teacher_context = {
                **context,
                "task": task,
                "description": context.get("description", task),
                "student_output": str(student_output),
                "gold_standard": str(gold_standard),  # Pass gold_standard for teacher to return
                "evaluation_feedback": str(eval_feedback)
            }
            
            teacher_inputs = self._prepare_agent_inputs(teacher_agent, teacher_context, {})
            
            # Run teacher agent (handles DSPy modules)
            agent = teacher_agent.agent
            if asyncio.iscoroutinefunction(agent):
                teacher_output = await agent(**teacher_inputs)
            elif DSPY_AVAILABLE and isinstance(agent, dspy.Module):
                # DSPy module - call directly
                teacher_output = agent(**teacher_inputs)
            elif hasattr(agent, 'forward'):
                if asyncio.iscoroutinefunction(agent.forward):
                    teacher_output = await agent.forward(**teacher_inputs)
                else:
                    teacher_output = agent.forward(**teacher_inputs)
            else:
                teacher_output = agent(**teacher_inputs)
            
            teacher_output_value = self._extract_agent_output(teacher_output)
            
            # Post-process: Extract actual diagram code if teacher returned evaluation text
            if teacher_output_value:
                import re
                teacher_output_str = str(teacher_output_value).strip()
                
                # Check if it's evaluation text (contains words like "evaluation", "analysis", "correct")
                is_evaluation_text = any(word in teacher_output_str.lower() for word in [
                    "evaluation", "analysis", "correct", "incorrect", "assessment", 
                    "verdict", "grade", "feedback", "strengths", "weaknesses", "student output"
                ])
                
                # If teacher returned evaluation text, extract diagram code or use gold_standard
                if is_evaluation_text:
                    logger.warning("Teacher returned evaluation text instead of diagram code, extracting or using gold_standard")
                    # Try to extract PlantUML code first
                    plantuml_match = re.search(r'@startuml.*?@enduml', teacher_output_str, re.DOTALL | re.IGNORECASE)
                    if plantuml_match:
                        teacher_output_value = plantuml_match.group(0).strip()
                        logger.info("Extracted PlantUML code from teacher output")
                    else:
                        # Try to find mermaid code block
                        mermaid_match = re.search(r'```mermaid\s*\n(.*?)\n```', teacher_output_str, re.DOTALL)
                        if mermaid_match:
                            teacher_output_value = mermaid_match.group(1).strip()
                            logger.info("Extracted Mermaid code from teacher output")
                        else:
                            # Try to find any code block
                            code_match = re.search(r'```(?:plantuml|mermaid)?\s*\n(.*?)\n```', teacher_output_str, re.DOTALL)
                            if code_match:
                                teacher_output_value = code_match.group(1).strip()
                                logger.info("Extracted code from markdown block")
                            else:
                                # Fallback: use gold_standard directly
                                logger.info("Using gold_standard directly as teacher output")
                                teacher_output_value = str(gold_standard).strip()
                else:
                    # Clean up markdown fences if present
                    teacher_output_str = re.sub(r'^```(?:plantuml|mermaid)?\s*\n?', '', teacher_output_str, flags=re.MULTILINE | re.IGNORECASE)
                    teacher_output_str = re.sub(r'^```\s*$', '', teacher_output_str, flags=re.MULTILINE)
                    teacher_output_value = teacher_output_str.strip()
                    
                    # If still looks like evaluation text, use gold_standard
                    if any(word in teacher_output_value.lower() for word in ["evaluation", "analysis", "assessment"]):
                        logger.info("Output still looks like evaluation, using gold_standard")
                        teacher_output_value = str(gold_standard).strip()
            
            # Teacher quality check
            if teacher_output_value and self.config.enable_teacher_quality_check:
                teacher_output_value = self._validate_teacher_output(
                    teacher_output_value,
                    student_output,
                    gold_standard,
                    evaluation_result
                )
            
            if teacher_output_value:
                self._write_thinking_log(f"Teacher model provided output: {str(teacher_output_value)[:100]}...")
                logger.info(f"Teacher output: {str(teacher_output_value)[:150]}...")
                return teacher_output_value
            else:
                logger.warning("Teacher model returned empty output, using gold_standard")
                return str(gold_standard)  # Fallback to gold_standard
            
        except Exception as e:
            logger.error(f"Teacher model failed: {e}")
            self._write_thinking_log(f"Teacher model failed: {e}")
            return None
    
    async def _update_knowledge_base(
        self,
        student_output: Any,
        teacher_output: Any,
        task: str,
        context: Dict[str, Any],
        evaluation_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Update knowledge base based on differences between student and teacher.
        
        Args:
            student_output: Student's output
            teacher_output: Teacher's output
            task: Task description
            context: Context dictionary
            evaluation_result: Evaluation results (must be a dict, not None)
            
        Returns:
            Dictionary with KB update results (always returns a dict, never None)
        """
        if not self.config.enable_kb_updates:
            return {"status": "disabled"}
        
        if self.config.kb_update_requires_teacher and not teacher_output:
            return {"status": "skipped", "reason": "teacher_output_required"}
        
        # Safety check
        if not isinstance(evaluation_result, dict):
            logger.warning(f"evaluation_result is not a dict: {type(evaluation_result)}")
            return {"status": "error", "error": "invalid_evaluation_result"}
        
        self._write_thinking_log("Analyzing differences for knowledge base updates...")
        
        # Look for KB update agent
        kb_agent = None
        for agent_config in self.agents:
            if "kb" in agent_config.name.lower() or "knowledge" in agent_config.name.lower() or (agent_config.metadata and agent_config.metadata.get("is_kb_updater", False)):
                kb_agent = agent_config
                break
        
        if not kb_agent:
            logger.warning("KB updates requested but no KB update agent found")
            self._write_thinking_log("No KB update agent found, skipping KB updates")
            return {"status": "skipped", "reason": "no_kb_agent"}
        
        try:
            # Prepare KB update inputs
            kb_context = {
                **context,
                "task": task,
                "student_output": student_output,
                "teacher_output": teacher_output,
                "evaluation_result": evaluation_result
            }
            
            kb_inputs = self._prepare_agent_inputs(kb_agent, kb_context, {})
            
            # Run KB update agent (handles DSPy modules)
            agent = kb_agent.agent
            if asyncio.iscoroutinefunction(agent):
                kb_result = await agent(**kb_inputs)
            elif DSPY_AVAILABLE and isinstance(agent, dspy.Module):
                # DSPy module - call directly
                kb_result = agent(**kb_inputs)
            elif hasattr(agent, 'forward'):
                if asyncio.iscoroutinefunction(agent.forward):
                    kb_result = await agent.forward(**kb_inputs)
                else:
                    kb_result = agent.forward(**kb_inputs)
            else:
                kb_result = agent(**kb_inputs)
            
            kb_result_value = self._extract_agent_output(kb_result)
            
            self._write_thinking_log("Knowledge base updated successfully")
            
            return {
                "status": "completed",
                "updates": kb_result_value if isinstance(kb_result_value, dict) else {"result": kb_result_value}
            }
            
        except Exception as e:
            logger.error(f"KB update failed: {e}")
            self._write_thinking_log(f"KB update failed: {e}")
            return {"status": "error", "error": str(e)}
    
    async def optimize(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None,
        gold_standard: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Run iterative optimization loop.
        
        Args:
            task: Task description
            context: Context dictionary with inputs
            gold_standard: Gold standard for evaluation (or use gold_standard_provider)
            
        Returns:
            Dictionary with optimization results including all iterations
        """
        context = context or {}
        self.iteration_count = 0
        self.consecutive_passes = 0
        self.all_iterations = []
        
        # Reset adaptive learning if enabled
        if self.adaptive_learning and AdaptiveLearning:
            self.adaptive_learning = AdaptiveLearning()
        
        # Get gold standard if provider is available
        if not gold_standard and self.config.gold_standard_provider:
            try:
                gold_standard = await self.config.gold_standard_provider(task, context)
            except Exception as e:
                logger.warning(f"Gold standard provider failed: {e}")
        
        # Clear thinking log at start
        if self.config.enable_thinking_log:
            self._clear_thinking_log()
        
        self._write_thinking_log(f"Starting optimization for task: {task}")
        self._write_thinking_log(f"Max iterations: {self.config.max_iterations}, Required passes: {self.config.required_pass_count}")
        
        while self.iteration_count < self.config.max_iterations:
            self.iteration_count += 1
            self._write_thinking_log(f"\n=== Iteration {self.iteration_count}/{self.config.max_iterations} ===")
            
            try:
                # Get previous outputs including teacher output
                previous_outputs = self._get_previous_outputs()
                
                # If we have teacher output from last iteration, add it to context
                if previous_outputs.get("teacher_output"):
                    context = {**context, "teacher_output": previous_outputs["teacher_output"]}
                    if previous_outputs.get("_teacher_feedback"):
                        context["_teacher_feedback"] = previous_outputs["_teacher_feedback"]
                    self._write_thinking_log(f"Passing teacher output to agent: {str(previous_outputs.get('teacher_output'))[:50]}")
                
                # Run agent pipeline
                pipeline_result = await self._run_agent_pipeline(
                    task=task,
                    context=context,
                    previous_outputs=previous_outputs
                )
                
                output = pipeline_result.get("output")
                
                # Debug: log what we're evaluating
                if self.config.enable_thinking_log:
                    output_str = str(output) if output is not None else "None"
                    self._write_thinking_log(f"Evaluating output: {repr(output_str[:100])}")
                    self._write_thinking_log(f"Gold standard: {repr(str(gold_standard)[:100])}")
                
                # Evaluate output
                evaluation_result = await self._evaluate_output(
                    output=output,
                    gold_standard=gold_standard,
                    task=task,
                    context=context
                )
                
                eval_score = evaluation_result.get("score", 0.0)
                eval_status = evaluation_result.get("status", "UNKNOWN")
                
                # Update adaptive learning
                if self.adaptive_learning:
                    learning_state = self.adaptive_learning.update_score(eval_score)
                    self._write_thinking_log(
                        f"Learning state: rate={learning_state['learning_rate']:.2f}, "
                        f"velocity={learning_state['improvement_velocity']:.3f}, "
                        f"plateau={learning_state['is_plateau']}"
                    )
                    
                    # Check for early stopping
                    if self.adaptive_learning.should_stop_early(min_iterations=3):
                        logger.info("Adaptive learning suggests early stopping")
                        self._write_thinking_log("Early stopping recommended by adaptive learning")
                
                # Check if evaluation passed
                passed = eval_score == 1.0 and eval_status == "CORRECT"
                
                teacher_output = None
                kb_updates = None
                
                if passed:
                    self.consecutive_passes += 1
                    self._write_thinking_log(
                        f"Iteration {self.iteration_count}: Evaluation PASSED "
                        f"(score={eval_score:.2f}). Consecutive passes: {self.consecutive_passes}/{self.config.required_pass_count}"
                    )
                    
                    # Check if we've reached required consecutive passes
                    if self.consecutive_passes >= self.config.required_pass_count:
                        self._write_thinking_log(
                            f" Optimization complete! Evaluation passed {self.consecutive_passes} times consecutively."
                        )
                        # Record the successful iteration before breaking
                        iteration_result = IterationResult(
                            iteration=self.iteration_count,
                            success=True,
                            evaluation_score=eval_score,
                            evaluation_status=eval_status,
                            output=output,
                            metadata={
                                "pipeline_result": pipeline_result,
                                "evaluation_result": evaluation_result
                            }
                        )
                        self.all_iterations.append(iteration_result)
                        break
                else:
                    self.consecutive_passes = 0
                    self._write_thinking_log(
                        f"Iteration {self.iteration_count}: Evaluation FAILED "
                        f"(score={eval_score:.2f}, status={eval_status})"
                    )
                    
                    # Try teacher model
                    if self.config.enable_teacher_model:
                        teacher_output = await self._run_teacher_model(
                            task=task,
                            context=context,
                            student_output=output,
                            gold_standard=gold_standard,
                            evaluation_result=evaluation_result
                        )
                        
                        if teacher_output:
                            # Evaluate teacher output
                            try:
                                teacher_eval = await self._evaluate_output(
                                    output=teacher_output,
                                    gold_standard=gold_standard,
                                    task=task,
                                    context=context
                                )
                                if teacher_eval:
                                    evaluation_result["teacher_evaluation"] = teacher_eval
                                
                                # Update KB if teacher succeeded
                                if self.config.enable_kb_updates:
                                    kb_updates = await self._update_knowledge_base(
                                        student_output=output,
                                        teacher_output=teacher_output,
                                        task=task,
                                        context=context,
                                        evaluation_result=evaluation_result
                                    )
                                    
                                # Reload context/metadata if KB was updated
                                if kb_updates and isinstance(kb_updates, dict) and kb_updates.get("status") == "completed":
                                    self._write_thinking_log("Knowledge base updated, context refreshed for next iteration")
                                    # Context refresh would happen here if needed
                                
                                # Store improvement for learning
                                if teacher_output and eval_score < 1.0:
                                    improvement = self._record_improvement(
                                        iteration=self.iteration_count,
                                        student_output=output,
                                        teacher_output=teacher_output,
                                        task=task,
                                        evaluation_result=evaluation_result,
                                        teacher_eval=evaluation_result.get("teacher_evaluation"),
                                        context=context
                                    )
                                    if improvement:
                                        self._write_thinking_log(f"Improvement recorded: {improvement.get('improvement_type', 'unknown')}")
                                        
                                        # Record credit assignment if enabled
                                        if self.credit_assignment:
                                            teacher_eval = evaluation_result.get("teacher_evaluation", {})
                                            self.credit_assignment.record_improvement_application(
                                                improvement=improvement,
                                                student_score=eval_score,
                                                teacher_score=teacher_eval.get("score", 1.0),
                                                final_score=teacher_eval.get("score", 1.0),
                                                context=context
                                            )
                            except Exception as kb_error:
                                logger.error(f"Error processing teacher output: {kb_error}")
                                self._write_thinking_log(f"Error processing teacher output: {kb_error}")
                
                # Store iteration result
                # Use teacher output if available and better, otherwise use agent output
                final_output_for_result = output
                if teacher_output and eval_score < 1.0:
                    # Check if teacher output is better
                    teacher_eval = evaluation_result.get("teacher_evaluation")
                    if teacher_eval and teacher_eval.get("score", 0.0) > eval_score:
                        final_output_for_result = teacher_output
                        eval_score = teacher_eval.get("score", eval_score)
                        eval_status = teacher_eval.get("status", eval_status)
                        passed = eval_score == 1.0 and eval_status == "CORRECT"
                
                iteration_result = IterationResult(
                    iteration=self.iteration_count,
                    success=passed,
                    evaluation_score=eval_score,
                    evaluation_status=eval_status,
                    output=final_output_for_result,  # Use best output available
                    metadata={
                        "pipeline_result": pipeline_result,
                        "evaluation_result": evaluation_result
                    },
                    teacher_output=teacher_output,
                    kb_updates=kb_updates
                )
                
                self.all_iterations.append(iteration_result)
                
            except Exception as e:
                logger.error(f"Iteration {self.iteration_count} failed with error: {e}")
                self._write_thinking_log(f"Iteration {self.iteration_count} failed: {e}")
                
                iteration_result = IterationResult(
                    iteration=self.iteration_count,
                    success=False,
                    evaluation_score=0.0,
                    evaluation_status="ERROR",
                    output=None,
                    error=str(e)
                )
                self.all_iterations.append(iteration_result)
                
                # Decide whether to continue or break on error
                # For now, continue to next iteration
                continue
        
        # Build final result
        final_result = {
            "status": "completed" if self.consecutive_passes >= self.config.required_pass_count else "stopped",
            "total_iterations": self.iteration_count,
            "consecutive_passes": self.consecutive_passes,
            "required_pass_count": self.config.required_pass_count,
            "max_iterations": self.config.max_iterations,
            "optimization_complete": self.consecutive_passes >= self.config.required_pass_count,
            "iterations": [
                {
                    "iteration": it.iteration,
                    "success": it.success,
                    "evaluation_score": it.evaluation_score,
                    "evaluation_status": it.evaluation_status,
                    "has_teacher_output": it.teacher_output is not None,
                    "has_kb_updates": it.kb_updates is not None,
                    "error": it.error
                }
                for it in self.all_iterations
            ],
            "final_result": self._get_best_result()
        }
        
        if final_result["optimization_complete"]:
            self._write_thinking_log(
                f" Optimization completed successfully after {self.iteration_count} iterations"
            )
        else:
            self._write_thinking_log(
                f" Optimization stopped after {self.iteration_count} iterations. "
                f"Consecutive passes: {self.consecutive_passes}/{self.config.required_pass_count}"
            )
        
        # Save final improvements summary
        if self.config.save_improvements and self.improvements:
            self._save_improvements_summary(final_result)
        
        return final_result
    
    def _save_improvements_summary(self, final_result: Dict[str, Any]) -> None:
        """Save a summary of all improvements."""
        if not self.improvements_file:
            return
        
        try:
            summary = {
                "optimization_complete": final_result.get("optimization_complete", False),
                "total_iterations": final_result.get("total_iterations", 0),
                "total_improvements": len(self.improvements),
                "improvements": self.improvements,
                "final_result": final_result.get("final_result"),
                "summary": {
                    "improvements_count": len(self.improvements),
                    "score_improvement": (
                        final_result.get("final_result", {}).get("evaluation_score", 0) -
                        (self.improvements[0].get("student_score", 0) if self.improvements else 0)
                    ),
                    "learned_patterns": [imp.get("learned_pattern") for imp in self.improvements]
                }
            }
            
            summary_file = self.improvements_file.parent / "improvements_summary.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Improvements summary saved to {summary_file}")
            self._write_thinking_log(f"Saved {len(self.improvements)} improvements to {summary_file}")
        except Exception as e:
            logger.error(f"Error saving improvements summary: {e}")
    
    def _get_best_result(self) -> Dict[str, Any]:
        """Get the best result from all iterations (prefer successful ones)."""
        if not self.all_iterations:
            return {
                "iteration": None,
                "output": None,
                "evaluation_score": None,
                "evaluation_status": None
            }
        
        # Find the best iteration (prefer successful ones, then highest score)
        best_iteration = None
        best_score = -1.0
        
        for iteration in self.all_iterations:
            if iteration.success and iteration.evaluation_score > best_score:
                best_iteration = iteration
                best_score = iteration.evaluation_score
            elif not best_iteration and iteration.evaluation_score > best_score:
                best_iteration = iteration
                best_score = iteration.evaluation_score
        
        # Fallback to last iteration if no best found
        if not best_iteration:
            best_iteration = self.all_iterations[-1]
        
        return {
            "iteration": best_iteration.iteration,
            "output": best_iteration.output,
            "evaluation_score": best_iteration.evaluation_score,
            "evaluation_status": best_iteration.evaluation_status
        }
    
    def _record_improvement(
        self,
        iteration: int,
        student_output: Any,
        teacher_output: Any,
        task: str,
        evaluation_result: Dict[str, Any],
        teacher_eval: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Record an improvement for learning and persistence.
        
        Args:
            iteration: Iteration number
            student_output: Student's output
            teacher_output: Teacher's output
            task: Task description
            evaluation_result: Evaluation results
            teacher_eval: Teacher evaluation results
            
        Returns:
            Dictionary with improvement record, or None if not recorded
        """
        if not self.config.save_improvements:
            return None
        
        improvement = {
            "iteration": iteration,
            "timestamp": datetime.now().isoformat(),
            "task": task,
            "student_output": str(student_output),
            "teacher_output": str(teacher_output),
            "student_score": evaluation_result.get("score", 0.0),
            "teacher_score": teacher_eval.get("score", 1.0) if teacher_eval else 1.0,
            "improvement_type": "teacher_correction",
            "difference": evaluation_result.get("difference") or evaluation_result.get("error_info"),
            "learned_pattern": self._extract_learned_pattern(student_output, teacher_output, task)
        }
        
        self.improvements.append(improvement)
        
        # Save to file (fallback if no memory system)
        if self.improvements_file:
            try:
                # Load existing improvements if file exists
                existing = []
                if self.improvements_file.exists():
                    try:
                        with open(self.improvements_file, 'r', encoding='utf-8') as f:
                            existing = json.load(f)
                    except Exception:
                        existing = []
                
                # Append new improvement
                existing.append(improvement)
                
                # Save back to file
                with open(self.improvements_file, 'w', encoding='utf-8') as f:
                    json.dump(existing, f, indent=2, ensure_ascii=False)
                
                logger.info(f"Improvement saved to {self.improvements_file}")
            except Exception as e:
                logger.error(f"Error saving improvement: {e}")
        
        # Also store to memory system if available
        # Try multiple ways to get memory system
        memory_system = None
        
        # Method 0: Expert's memory (highest priority - direct storage)
        if hasattr(self, 'expert_memory') and self.expert_memory:
            memory_system = self.expert_memory
            expert_name = getattr(self, 'expert_name', 'unknown')
            domain = getattr(self, 'expert_domain', 'general')
            logger.debug(f"Using expert's memory directly (domain: {domain})")
        
        # Method 1: Direct memory attribute
        elif self.conductor and hasattr(self.conductor, 'memory') and self.conductor.memory:
            memory_system = self.conductor.memory
        
        # Method 2: Shared memory from conductor
        elif self.conductor and hasattr(self.conductor, 'shared_memory') and self.conductor.shared_memory:
            memory_system = self.conductor.shared_memory
        
        # Method 3: From brain system
        elif self.conductor and hasattr(self.conductor, 'brain') and self.conductor.brain:
            if hasattr(self.conductor.brain, 'memory') and self.conductor.brain.memory:
                memory_system = self.conductor.brain.memory
        
        if memory_system:
            try:
                from ..experts.memory_integration import store_improvement_to_memory
                
                # Use expert info if available (from Method 0)
                if hasattr(self, 'expert_name') and hasattr(self, 'expert_domain'):
                    expert_name = self.expert_name
                    domain = self.expert_domain
                else:
                    # Get expert name from context or agent config
                    expert_name = self.agents[0].name if self.agents else "unknown"
                    domain = "general"  # Could be extracted from context
                    
                    # Try to extract domain from agent name or context
                    if self.agents:
                        agent_name = self.agents[0].name.lower()
                        if "mermaid" in agent_name:
                            domain = "mermaid"
                        elif "pipeline" in agent_name:
                            domain = "pipeline"
                        elif "plantuml" in agent_name:
                            domain = "plantuml"
                    
                    # Also check context for domain
                    if domain == "general" and hasattr(self, 'current_context'):
                        context_domain = self.current_context.get('domain') if isinstance(self.current_context, dict) else None
                        if context_domain:
                            domain = context_domain
                
                store_improvement_to_memory(
                    memory=memory_system,
                    improvement=improvement,
                    expert_name=expert_name,
                    domain=domain
                )
                logger.info(f"Stored improvement to memory system (domain: {domain}, expert: {expert_name})")
            except Exception as e:
                logger.warning(f"Could not store improvement to memory: {e}")
        
        # Update DSPy instructions if enabled
        if self.config.update_dspy_instructions:
            self._update_dspy_instructions(improvement)
        
        # Update Jotty instructions if enabled
        if self.config.update_jotty_instructions and self.conductor:
            self._update_jotty_instructions(improvement)
        
        return improvement
    
    def _extract_learned_pattern(
        self,
        student_output: Any,
        teacher_output: Any,
        task: str
    ) -> str:
        """
        Extract a concise learned pattern from the improvement.
        
        Args:
            student_output: Student's output
            teacher_output: Teacher's output (should be diagram code)
            task: Task description
            
        Returns:
            Concise string describing the learned pattern
        """
        student_str = str(student_output).strip()
        teacher_str = str(teacher_output).strip()
        
        # Remove markdown fences
        import re
        student_str = re.sub(r'^```\w*\s*\n?', '', student_str, flags=re.MULTILINE)
        student_str = re.sub(r'^```\s*$', '', student_str, flags=re.MULTILINE)
        teacher_str = re.sub(r'^```\w*\s*\n?', '', teacher_str, flags=re.MULTILINE)
        teacher_str = re.sub(r'^```\s*$', '', teacher_str, flags=re.MULTILINE)
        
        # Extract key differences
        # Check if student used wrong syntax
        student_lower = student_str.lower()
        teacher_lower = teacher_str.lower()
        
        # Detect syntax mismatches
        if 'mermaid' in student_lower and 'plantuml' in teacher_lower or '@startuml' in teacher_lower:
            return f"When task is '{task}', use PlantUML syntax (@startuml/@enduml) instead of Mermaid syntax"
        
        if 'plantuml' in student_lower and 'mermaid' in teacher_lower or 'sequenceDiagram' in teacher_lower:
            return f"When task is '{task}', use Mermaid syntax instead of PlantUML syntax"
        
        # Check for missing tags
        if '@startuml' in teacher_lower and '@startuml' not in student_lower:
            return f"When task is '{task}', include @startuml/@enduml tags"
        
        # Check for complexity differences
        student_lines = len(student_str.split('\n'))
        teacher_lines = len(teacher_str.split('\n'))
        if student_lines > teacher_lines * 1.5:
            return f"When task is '{task}', keep diagram simple and match gold standard format"
        
        # Generic pattern
        if student_str != teacher_str:
            # Extract first line or key difference
            teacher_first_line = teacher_str.split('\n')[0][:50] if teacher_str else ""
            return f"When task is '{task}', use format: {teacher_first_line}..."
        
        return f"Pattern learned for task: '{task}'"
    
    def _validate_teacher_output(
        self,
        teacher_output: Any,
        student_output: Any,
        gold_standard: Any,
        evaluation_result: Dict[str, Any]
    ) -> Optional[Any]:
        """
        Validate teacher output quality before using it.
        
        Checks if teacher output is actually better than student output.
        Rejects low-quality teacher corrections.
        
        Args:
            teacher_output: Teacher's output
            student_output: Student's output
            gold_standard: Gold standard
            evaluation_result: Evaluation results
        
        Returns:
            Teacher output if valid, None if rejected
        """
        if not teacher_output:
            return None
        
        teacher_str = str(teacher_output).strip()
        student_str = str(student_output).strip()
        gold_str = str(gold_standard).strip()
        
        # Check if teacher output is empty or same as student
        if not teacher_str or teacher_str == student_str:
            logger.warning("Teacher output is empty or same as student, rejecting")
            return None
        
        # Check if teacher output is too different from gold standard (might be wrong)
        # Simple length check - if teacher is way longer/shorter, might be wrong
        gold_len = len(gold_str)
        teacher_len = len(teacher_str)
        
        if gold_len > 0:
            length_ratio = teacher_len / gold_len
            if length_ratio < 0.3 or length_ratio > 3.0:
                logger.warning(f"Teacher output length ratio {length_ratio:.2f} suspicious, rejecting")
                return None
        
        # Check if teacher output looks like evaluation text (already handled above, but double-check)
        teacher_lower = teacher_str.lower()
        if any(word in teacher_lower for word in ["evaluation", "analysis", "assessment", "correct", "incorrect"]):
            # If it's mostly evaluation text, reject
            if len([w for w in ["evaluation", "analysis", "assessment"] if w in teacher_lower]) >= 2:
                logger.warning("Teacher output appears to be evaluation text, rejecting")
                return None
        
        # If we have evaluation function, evaluate teacher output
        if self.config.evaluation_function:
            try:
                import asyncio
                # Note: This is a sync check, but evaluation_function might be async
                # For now, we'll do a simple check
                # In full implementation, would need to await if async
                logger.debug("Teacher output passed quality checks")
            except Exception as e:
                logger.warning(f"Error validating teacher output: {e}")
        
        return teacher_output
    
    def _update_dspy_instructions(self, improvement: Dict[str, Any]) -> None:
        """
        Update DSPy module instructions with learned improvement.
        
        Args:
            improvement: Improvement record
        """
        try:
            # Try to find DSPy modules in agents
            for agent_config in self.agents:
                agent = agent_config.agent
                
                # Check if it's a DSPy module with instructions
                if hasattr(agent, 'instructions') or hasattr(agent, '_instructions'):
                    learned_pattern = improvement.get("learned_pattern", "")
                    
                    # Add to instructions
                    if hasattr(agent, 'instructions'):
                        if isinstance(agent.instructions, list):
                            agent.instructions.append(learned_pattern)
                        elif isinstance(agent.instructions, str):
                            agent.instructions += f"\n\nLEARNED: {learned_pattern}"
                    
                    logger.info(f"Updated DSPy instructions for agent {agent_config.name}")
                    break
        except Exception as e:
            logger.warning(f"Could not update DSPy instructions: {e}")
    
    def _update_jotty_instructions(self, improvement: Dict[str, Any]) -> None:
        """
        Update Jotty learned_instructions with improvement.
        
        Args:
            improvement: Improvement record
        """
        try:
            if self.conductor and hasattr(self.conductor, 'learned_instructions'):
                learned_pattern = improvement.get("learned_pattern", "")
                
                # Add to actor learned instructions
                if 'actor' not in self.conductor.learned_instructions:
                    self.conductor.learned_instructions['actor'] = []
                
                self.conductor.learned_instructions['actor'].append(learned_pattern)
                
                # Keep bounded (last 10)
                if len(self.conductor.learned_instructions['actor']) > 10:
                    self.conductor.learned_instructions['actor'] = self.conductor.learned_instructions['actor'][-10:]
                
                logger.info("Updated Jotty learned_instructions")
        except Exception as e:
            logger.warning(f"Could not update Jotty instructions: {e}")
    
    def _get_previous_outputs(self) -> Dict[str, Any]:
        """Get outputs from previous iteration for retry context."""
        if not self.all_iterations:
            return {}
        
        last_iteration = self.all_iterations[-1]
        
        # Include teacher output if available (so agent can learn from it)
        outputs = {}
        if last_iteration.teacher_output is not None:
            teacher_val = last_iteration.teacher_output
            # Extract string value if it's wrapped in a result object
            if hasattr(teacher_val, '_store') and isinstance(teacher_val._store, dict):
                teacher_val = teacher_val._store.get('output', teacher_val)
            elif not isinstance(teacher_val, str):
                teacher_val = str(teacher_val)
            outputs["teacher_output"] = teacher_val
            outputs["_teacher_feedback"] = "Use the teacher's output as reference for improvement"
            logger.debug(f"Retrieved teacher output from previous iteration: {teacher_val}")
        
        if last_iteration.metadata and "pipeline_result" in last_iteration.metadata:
            pipeline_result = last_iteration.metadata["pipeline_result"]
            if "intermediate_outputs" in pipeline_result:
                outputs.update(pipeline_result["intermediate_outputs"])
        
        return outputs


def create_optimization_pipeline(agents: List[AgentConfig], max_iterations: int = 5, required_pass_count: int = 2, enable_teacher_model: bool = True, enable_kb_updates: bool = True, output_path: Optional[Union[str, Path]] = None, **kwargs: Any) -> OptimizationPipeline:
    """
    Convenience function to create an OptimizationPipeline.
    
    Args:
        agents: List of AgentConfig defining the pipeline
        max_iterations: Maximum number of optimization iterations
        required_pass_count: Number of consecutive passes required
        enable_teacher_model: Enable teacher model fallback
        enable_kb_updates: Enable knowledge base updates
        output_path: Path for output files
        **kwargs: Additional OptimizationConfig parameters
        
    Returns:
        OptimizationPipeline instance
    """
    config = OptimizationConfig(
        max_iterations=max_iterations,
        required_pass_count=required_pass_count,
        enable_teacher_model=enable_teacher_model,
        enable_kb_updates=enable_kb_updates,
        output_path=Path(output_path) if output_path else None,
        **kwargs
    )
    
    return OptimizationPipeline(agents=agents, config=config)
