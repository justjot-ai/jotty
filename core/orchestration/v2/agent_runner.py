"""
V2 AgentRunner - Executes a single agent with validation and learning

Extracted from SingleAgentOrchestrator for reuse in unified Conductor.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from ...foundation.data_structures import JottyConfig, EpisodeResult
from ...agents.inspector import InspectorAgent, MultiRoundValidator
from ...memory.cortex import HierarchicalMemory
from ...learning.learning import (
    TDLambdaLearner, AdaptiveLearningRate, IntermediateRewardCalculator,
    ReasoningCreditAssigner, AdaptiveExploration
)

logger = logging.getLogger(__name__)


@dataclass
class AgentRunnerConfig:
    """Configuration for AgentRunner"""
    architect_prompts: List[str]
    auditor_prompts: List[str]
    config: JottyConfig
    agent_name: str = "agent"
    enable_learning: bool = True
    enable_memory: bool = True


class AgentRunner:
    """
    Executes ONE agent with validation and learning.
    
    V2 User-Friendly Component:
    - Wraps agent execution
    - Provides Architect (pre-execution)
    - Provides Auditor (post-execution)
    - Handles learning and memory
    """
    
    def __init__(
        self,
        agent: Any,  # AutoAgent or DSPy agent
        config: AgentRunnerConfig,
        task_planner=None,  # Shared TaskPlanner (V2)
        task_board=None,  # Shared TaskBoard (V2)
        swarm_memory=None,  # Shared SwarmMemory (V2)
        swarm_state_manager=None,  # SwarmStateManager for state tracking (V2)
    ):
        """
        Initialize AgentRunner.
        
        Args:
            agent: The agent to execute (AutoAgent or DSPy module)
            config: AgentRunner configuration
            task_planner: Shared TaskPlanner (optional)
            task_board: Shared TaskBoard (optional)
            swarm_memory: Shared SwarmMemory (optional)
            swarm_state_manager: SwarmStateManager for state tracking (optional)
        """
        self.agent = agent
        self.config = config
        self.agent_name = config.agent_name
        
        # Shared components (V2)
        self.task_planner = task_planner
        self.task_board = task_board
        self.swarm_memory = swarm_memory
        self.swarm_state_manager = swarm_state_manager
        
        # Get agent state tracker (creates if doesn't exist)
        if self.swarm_state_manager:
            self.agent_tracker = self.swarm_state_manager.get_agent_tracker(self.agent_name)
            logger.info(f"📊 AgentStateTracker initialized for '{self.agent_name}'")
        
        from pathlib import Path
        
        from ...foundation.data_structures import SharedScratchpad
        
        # Shared scratchpad for agent communication
        scratchpad = SharedScratchpad()
        
        # Architect (pre-execution planning)
        architect_agents = [
            InspectorAgent(
                md_path=Path(prompt),
                is_architect=True,
                tools=[],
                config=config.config,
                scratchpad=scratchpad
            )
            for prompt in config.architect_prompts
        ]
        
        # Auditor (post-execution validation)
        auditor_agents = [
            InspectorAgent(
                md_path=Path(prompt),
                is_architect=False,
                tools=[],
                config=config.config,
                scratchpad=scratchpad
            )
            for prompt in config.auditor_prompts
        ]
        
        # Multi-round validators
        self.architect_validator = MultiRoundValidator(architect_agents, config.config)
        self.auditor_validator = MultiRoundValidator(auditor_agents, config.config)
        
        # Per-agent memory
        self.agent_memory: Optional[HierarchicalMemory] = None
        if config.enable_memory:
            self.agent_memory = HierarchicalMemory(
                config=config.config,
                agent_name=self.agent_name
            )
        
        # Per-agent learning
        from ...learning.learning import AdaptiveLearningRate
        self.agent_learner: Optional[TDLambdaLearner] = None
        if config.enable_learning:
            adaptive_lr = AdaptiveLearningRate(config.config)
            self.agent_learner = TDLambdaLearner(
                config=config.config,
                adaptive_lr=adaptive_lr
            )
        
        logger.info(f"✅ AgentRunner initialized: {self.agent_name}")
    
    async def run(self, goal: str, **kwargs) -> EpisodeResult:
        """
        Run agent execution with validation and learning.
        
        Args:
            goal: Task goal/description
            **kwargs: Additional arguments for agent
            
        Returns:
            EpisodeResult with output and metadata
        """
        import time
        start_time = time.time()
        
        logger.info(f"🚀 AgentRunner.run: {self.agent_name} - {goal[:50]}...")
        
        # Start episode for TD(λ) learning (if enabled)
        if self.agent_learner:
            self.agent_learner.start_episode(goal)
        
        # 1. Architect (pre-execution planning)
        architect_results, proceed = await self.architect_validator.validate(
            goal=goal,
            inputs={'goal': goal, **kwargs},
            trajectory=[],
            is_architect=True
        )
        
        # Track architect validation in state
        if self.swarm_state_manager:
            avg_confidence = sum(r.confidence for r in architect_results) / len(architect_results) if architect_results else 0.0
            self.agent_tracker.record_validation(
                validation_type='architect',
                passed=proceed,
                confidence=avg_confidence,
                feedback=architect_results[0].reasoning if architect_results else None
            )
            # Record swarm-level step
            self.swarm_state_manager.record_swarm_step({
                'agent': self.agent_name,
                'step': 'architect',
                'proceed': proceed,
                'confidence': avg_confidence,
                'architect_confidence': avg_confidence
            })
        
        # Architect doesn't block (exploration only) but log confidence concerns
        if architect_results:
            avg_confidence = sum(r.confidence for r in architect_results) / len(architect_results)
            if avg_confidence < 0.3:
                logger.warning(
                    f"⚠️  Architect VERY LOW confidence: {avg_confidence:.2f} "
                    f"(Decision: {'PROCEED' if proceed else 'BLOCKED'}). "
                    f"Consider gathering more information before execution."
                )
            elif avg_confidence < 0.5 and proceed:
                logger.info(
                    f"ℹ️  Architect LOW confidence: {avg_confidence:.2f} but proceeding. "
                    f"Execution will proceed but results may be uncertain."
                )
        
        # 2. Agent execution
        try:
            # Execute agent
            if hasattr(self.agent, 'execute'):
                # AutoAgent
                agent_output = await self.agent.execute(goal, **kwargs)
            elif hasattr(self.agent, 'forward'):
                # DSPy module
                agent_output = self.agent(goal=goal, **kwargs)
            else:
                # Callable
                agent_output = await self.agent(goal, **kwargs) if asyncio.iscoroutinefunction(self.agent) else self.agent(goal, **kwargs)
            
            # Build trajectory BEFORE auditor validation (so auditor can see execution history)
            trajectory = []
            if hasattr(agent_output, '__dict__'):
                trajectory.append({
                    'step': 1,
                    'action': 'execute',
                    'output': str(agent_output)[:500],
                    'success': True  # Will be updated after validation
                })
            else:
                trajectory.append({
                    'step': 1,
                    'action': 'execute',
                    'output': str(agent_output)[:500] if agent_output else None,
                    'success': True  # Will be updated after validation
                })
            
            # 3. Auditor (post-execution validation) - now with trajectory
            auditor_results, passed = await self.auditor_validator.validate(
                goal=goal,
                inputs={'goal': goal, 'output': str(agent_output)},
                trajectory=trajectory,  # Pass trajectory so auditor can see execution history
                is_architect=False
            )
            
            success = passed
            # ValidationResult has 'reasoning', not 'feedback'
            auditor_reasoning = auditor_results[0].reasoning if auditor_results else "No feedback"
            auditor_confidence = auditor_results[0].confidence if auditor_results else 0.0
            
            # Track auditor validation in state
            if self.swarm_state_manager:
                self.agent_tracker.record_validation(
                    validation_type='auditor',
                    passed=passed,
                    confidence=auditor_confidence,
                    feedback=auditor_reasoning
                )
                # Record agent output
                output_type = type(agent_output).__name__
                self.agent_tracker.record_output(agent_output, output_type)
                # Record swarm-level step
                self.swarm_state_manager.record_swarm_step({
                    'agent': self.agent_name,
                    'step': 'auditor',
                    'success': success,
                    'validation_passed': passed,
                    'auditor_result': auditor_reasoning[:100],
                    'auditor_confidence': auditor_confidence
                })
            
            # Update trajectory with validation result
            if trajectory:
                trajectory[0]['success'] = success
                trajectory[0]['validation'] = {
                    'passed': passed,
                    'confidence': auditor_confidence,
                    'tag': auditor_results[0].output_tag.value if auditor_results and auditor_results[0].output_tag else None
                }
            
            # 4. Memory storage (if enabled) - store before learning so we can use it
            episode_memory_entry = None
            if self.agent_memory:
                from ...foundation.data_structures import MemoryLevel
                episode_memory_entry = self.agent_memory.store(
                    content=f"Goal: {goal}\nOutput: {str(agent_output)[:500]}",
                    level=MemoryLevel.EPISODIC,
                    context={'agent': self.agent_name, 'goal': goal},
                    goal=goal
                )
            
            # 5. Learning update using proper episodic interface (if enabled)
            if self.agent_learner:
                # Record memory access if we stored one (for TD(λ) traces)
                if episode_memory_entry:
                    self.agent_learner.record_access(episode_memory_entry, step_reward=0.0)
                
                # Calculate final reward: +1.0 if passed, -0.5 if failed
                final_reward = 1.0 if success else -0.5
                
                # Build memories dict from accessed memories (for end_episode)
                memories_dict = {}
                if episode_memory_entry:
                    memories_dict[episode_memory_entry.key] = episode_memory_entry
                
                # Perform TD(λ) updates at episode end
                updates = self.agent_learner.end_episode(
                    final_reward=final_reward,
                    memories=memories_dict
                )
                
                if updates:
                    logger.debug(f"📚 Learning: Updated {len(updates)} memory values")
            
            duration = time.time() - start_time
            
            # Extract tagged outputs from auditor results
            from ...foundation.types.learning_types import TaggedOutput
            from ...foundation.types.enums import OutputTag
            tagged_outputs = []
            if auditor_results:
                for result in auditor_results:
                    if result.output_tag:
                        tagged_outputs.append(TaggedOutput(
                            name=self.agent_name,
                            tag=result.output_tag,
                            why_useful=result.why_useful or result.reasoning,
                            content=agent_output
                        ))
            
            # Build agent contributions
            agent_contributions = {}
            if architect_results:
                for result in architect_results:
                    from ...foundation.types.agent_types import AgentContribution
                    agent_contributions[result.agent_name] = AgentContribution(
                        agent_name=result.agent_name,
                        contribution_score=result.confidence if result.should_proceed else -result.confidence,
                        decision="approve" if result.should_proceed else "reject",
                        decision_correct=success,
                        counterfactual_impact=0.5,  # Default
                        reasoning_quality=result.confidence,
                        evidence_used=[],
                        tools_used=result.tool_calls or [],
                        decision_timing=0.5,
                        temporal_weight=1.0
                    )
            
            return EpisodeResult(
                output=agent_output,
                success=success,
                trajectory=trajectory,
                tagged_outputs=tagged_outputs,
                episode=0,  # Episode number (could track this)
                execution_time=duration,
                architect_results=architect_results or [],
                auditor_results=auditor_results or [],
                agent_contributions=agent_contributions
            )
            
        except Exception as e:
            logger.error(f"❌ Agent execution failed: {e}", exc_info=True)
            import traceback
            logger.debug(traceback.format_exc())
            
            # Track error in state
            if self.swarm_state_manager:
                error_type = type(e).__name__
                self.agent_tracker.record_error(
                    error=str(e),
                    error_type=error_type,
                    context={'goal': goal, 'kwargs': kwargs}
                )
                # Record swarm-level error step
                self.swarm_state_manager.record_swarm_step({
                    'agent': self.agent_name,
                    'step': 'error',
                    'error': str(e),
                    'error_type': error_type,
                    'success': False
                })
            
            # Return failed EpisodeResult with correct structure
            duration = time.time() - start_time
            return EpisodeResult(
                output=None,
                success=False,
                trajectory=[{'step': 0, 'action': 'error', 'error': str(e)}],
                tagged_outputs=[],
                episode=0,
                execution_time=time.time() - start_time,
                architect_results=architect_results if 'architect_results' in locals() else [],
                auditor_results=[],
                agent_contributions={},
                alerts=[f"Execution failed: {str(e)[:100]}"]
            )
