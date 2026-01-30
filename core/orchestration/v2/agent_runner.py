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
from ...learning.shaped_rewards import ShapedRewardManager

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
        learning_manager=None,  # Swarm-level LearningManager (V1 pipeline)
        transfer_learning=None,  # TransferableLearningStore for cross-swarm learning
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
        self.learning_manager = learning_manager
        self.transfer_learning = transfer_learning
        
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

        # Shaped rewards for dense learning signal
        self.shaped_reward_manager: Optional[ShapedRewardManager] = None
        if config.enable_learning:
            self.shaped_reward_manager = ShapedRewardManager()

        logger.info(f"AgentRunner initialized: {self.agent_name}")
    
    async def run(self, goal: str, **kwargs) -> EpisodeResult:
        """
        Run agent execution with validation and learning.

        Args:
            goal: Task goal/description
            skip_validation: If True, skip architect validation (fast mode)
            status_callback: Optional callback(stage, detail) for progress updates
            **kwargs: Additional arguments for agent

        Returns:
            EpisodeResult with output and metadata
        """
        import time
        start_time = time.time()

        # Extract flags
        skip_validation = kwargs.pop('skip_validation', False)
        status_callback = kwargs.pop('status_callback', None)

        def _status(stage: str, detail: str = ""):
            """Report progress if callback provided."""
            if status_callback:
                try:
                    status_callback(stage, detail)
                except Exception:
                    pass
            logger.info(f"  📍 {stage}" + (f": {detail}" if detail else ""))

        logger.info(f"AgentRunner.run: {self.agent_name} - {goal[:50]}..." + (" [FAST]" if skip_validation else ""))

        # Start episode for TD(λ) learning (if enabled)
        if self.agent_learner:
            self.agent_learner.start_episode(goal)

        # Reset shaped rewards for new episode
        if self.shaped_reward_manager:
            self.shaped_reward_manager.reset()

        _status("Preparing", "retrieving context")

        # Retrieve relevant memories and enrich goal context
        enriched_goal = goal
        if self.agent_memory:
            try:
                from ...foundation.data_structures import MemoryLevel
                relevant_memories = self.agent_memory.retrieve(
                    query=goal,
                    goal=goal,
                    budget_tokens=3000,
                    levels=[MemoryLevel.SEMANTIC, MemoryLevel.PROCEDURAL, MemoryLevel.META]
                )
                if relevant_memories:
                    context = "\n".join([m.content for m in relevant_memories[:5]])
                    enriched_goal = f"{goal}\n\nRelevant past experience:\n{context}"
                    logger.info(f"Memory retrieval: {len(relevant_memories)} memories injected as context")
            except Exception as e:
                logger.debug(f"Memory retrieval skipped: {e}")

        # Inject Q-learning context from swarm-level learner
        if self.learning_manager:
            try:
                state = {'query': goal, 'agent': self.agent_name}
                q_context = self.learning_manager.get_learned_context(state)
                if q_context:
                    enriched_goal = f"{enriched_goal}\n\nLearned Insights:\n{q_context}"
            except Exception as e:
                logger.debug(f"Q-learning context injection skipped: {e}")

        # Inject transferable learning context (cross-swarm, cross-goal)
        if self.transfer_learning:
            try:
                transfer_context = self.transfer_learning.format_context_for_agent(goal, self.agent_name)
                if transfer_context and 'Transferable Learnings' in transfer_context:
                    enriched_goal = f"{enriched_goal}\n\n{transfer_context}"
            except Exception as e:
                logger.debug(f"Transfer learning context injection skipped: {e}")

        # 1. Architect (pre-execution planning) - skip in fast mode
        architect_results = []
        proceed = True
        architect_shaped_reward = 0.0

        if not skip_validation:
            _status("Architect", "validating approach")
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

            # Architect doesn't block (exploration only) - log confidence to debug (not user-facing)
            if architect_results:
                avg_confidence = sum(r.confidence for r in architect_results) / len(architect_results)
                # Log to debug only - not user-facing
                logger.debug(
                    f"Architect confidence: {avg_confidence:.2f} "
                    f"(Decision: {'PROCEED' if proceed else 'BLOCKED'})"
                )

            # Shaped reward: architect validation
            if self.shaped_reward_manager and architect_results:
                architect_shaped_reward = self.shaped_reward_manager.check_rewards(
                    event_type="actor_start",
                    state={'architect_results': architect_results, 'proceed': proceed, 'goal': goal},
                    trajectory=[]
                )
        else:
            logger.info("⚡ Fast mode: skipping architect validation")

        # 2. Agent execution (use enriched goal with memory context)
        _status("Agent", "executing task (this may take a while)")
        try:
            # Always use AutoAgent for skill execution (skip_validation only affects architect/auditor)
            if hasattr(self.agent, 'execute'):
                # AutoAgent - pass status_callback if agent supports it
                if status_callback:
                    kwargs['status_callback'] = status_callback
                agent_output = await self.agent.execute(enriched_goal, **kwargs)
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
            
            # 3. Auditor (post-execution validation) - skip in fast mode
            if not skip_validation:
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
            else:
                # Fast mode: skip auditor, assume success
                logger.info("⚡ Fast mode: skipping auditor validation")
                success = True
                auditor_reasoning = "Fast mode: validation skipped"
                auditor_confidence = 1.0
                auditor_results = []
                passed = True
            
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
            
            # 5. Shaped rewards after auditor validation
            auditor_shaped_reward = 0.0
            if self.shaped_reward_manager:
                auditor_shaped_reward = self.shaped_reward_manager.check_rewards(
                    event_type="validation",
                    state={
                        'auditor_results': auditor_results,
                        'passed': passed,
                        'goal': goal,
                        'output': str(agent_output)[:500]
                    },
                    trajectory=trajectory
                )
                # Also check actor_complete rewards
                self.shaped_reward_manager.check_rewards(
                    event_type="actor_complete",
                    state={
                        'output': str(agent_output)[:500],
                        'success': success,
                        'goal': goal
                    },
                    trajectory=trajectory
                )

            # 6. Learning update with dense shaped rewards (if enabled)
            if self.agent_learner:
                # Record memory access with intermediate shaped reward
                step_reward = architect_shaped_reward + auditor_shaped_reward
                if episode_memory_entry:
                    self.agent_learner.record_access(episode_memory_entry, step_reward=step_reward)

                # Final reward combines sparse terminal + accumulated shaped rewards
                terminal_reward = 1.0 if success else -0.5
                shaped_total = self.shaped_reward_manager.get_total_reward() if self.shaped_reward_manager else 0.0
                final_reward = terminal_reward + shaped_total

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
                    logger.debug(f"Learning: Updated {len(updates)} memory values (shaped={shaped_total:.3f})")

            # 7. Record experience into swarm-level Q-learner
            if self.learning_manager:
                try:
                    q_state = {'query': goal, 'agent': self.agent_name, 'success': success}
                    q_action = {'actor': self.agent_name, 'task': goal[:100]}
                    q_reward = final_reward if 'final_reward' in locals() else (1.0 if success else -0.5)
                    self.learning_manager.record_outcome(q_state, q_action, q_reward, done=True)
                except Exception as e:
                    logger.debug(f"Swarm Q-learning record skipped: {e}")

            # 8. Memory consolidation: promote episodic -> semantic/procedural
            if self.agent_memory:
                try:
                    await self.agent_memory.consolidate()
                    logger.debug("Memory consolidation completed")
                except Exception as e:
                    logger.debug(f"Memory consolidation skipped: {e}")
            
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
