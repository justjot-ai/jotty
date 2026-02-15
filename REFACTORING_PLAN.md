# Unified Swarm Architecture - Refactoring Plan
## Goal: World's Best Self-Learning Swarm Manager

**Date:** 2026-02-15
**Status:** Planning → Implementation → Testing

---

## 🎯 Objectives

1. **Single unified BaseSwarm** - Replaces DomainSwarm + SwarmTemplate
2. **AUTO pattern** - Swarm intelligently selects coordination pattern
3. **Comprehensive learning** - Integrate ALL 8 learning mechanisms
4. **Zero learning loss** - Preserve every learning capability
5. **Examples & tests** - Demonstrate and validate everything
6. **Backward compatible** - Old code still works

---

## 📦 Phase 1: Enhanced CoordinationPattern (1 hour)

### Files to modify:
- `core/infrastructure/foundation/types/execution_types.py`

### Changes:
```python
class CoordinationPattern(Enum):
    """Agent coordination patterns."""

    # Adaptive
    AUTO = "auto"              # Swarm auto-selects pattern

    # Custom
    CUSTOM = "custom"          # Multi-stage workflow with STAGES

    # Linear
    SEQUENTIAL = "sequential"  # A → B → C

    # Concurrent
    PARALLEL = "parallel"      # A | B | C

    # Collaborative
    CONSENSUS = "consensus"    # Vote & agree
    DEBATE = "debate"          # Multi-round + synthesize
    ITERATIVE = "iterative"    # Feedback loop

    # Hierarchical
    HIERARCHICAL = "hierarchical"  # Manager → workers

    # Shared
    BLACKBOARD = "blackboard"  # Shared workspace

class MergeStrategy(Enum):
    """How to combine results (mechanical)."""

    COMBINE = "combine"    # List of all results
    CONCAT = "concat"      # String concatenation
    FIRST = "first"        # First success
    BEST = "best"          # Highest score
    VOTE = "vote"          # Majority wins

class SynthesisStrategy(Enum):
    """How to synthesize results (intelligent)."""

    SYNTHESIZE = "synthesize"      # LLM creates new insight
    CONSOLIDATE = "consolidate"    # Merge + deduplicate
    REFINE = "refine"              # Improve best result
    BLEND = "blend"                # Weighted combination
```

### Tests:
- `tests/test_coordination_patterns.py` - Test all patterns
- `tests/test_merge_strategies.py` - Test combine vs synthesize

---

## 📦 Phase 2: Unified BaseSwarm (3 hours)

### Files to create:
- `core/intelligence/swarms/unified_swarm.py` (NEW)

### Files to modify:
- `core/intelligence/swarms/base_swarm.py` (enhance)
- `core/intelligence/swarms/base/domain_swarm.py` (make it alias)

### New Architecture:

```python
class BaseSwarm(SwarmLearningMixin, ABC):
    """
    Universal base for ALL swarms - integrates ALL learning.

    Learning Layers (all integrated):
    1. Memory (5 levels) - What we experienced/know/can-do
    2. TD-Lambda - Reinforcement learning from rewards
    3. Swarm Intelligence - Meta-learning for coordination
    4. Gold Standards - Self-improvement via evaluation
    5. Improvement Agents - Feedback loop (Expert, Reviewer, etc.)
    6. Pattern Learning - Which coordination works best
    7. Transfer Learning - Apply to similar tasks
    8. Adaptive Components - Dynamic hyperparameters
    """

    # ========== Configuration ==========
    AGENT_TEAM: ClassVar[AgentTeam]
    COORDINATION: ClassVar[CoordinationPattern] = CoordinationPattern.AUTO
    STAGES: ClassVar[List[StageConfig]] = []

    # ========== Learning Integration (ALL) ==========
    def __init__(self, config: SwarmConfig):
        super().__init__(config)

        # Layer 1: Memory (5 levels)
        self._memory: SwarmMemory = None

        # Layer 2: TD-Lambda
        self._td_learner: TDLambdaLearner = None

        # Layer 3: Swarm Intelligence
        self._swarm_intelligence: SwarmIntelligence = None

        # Layer 4: Gold Standards
        self._gold_db: GoldStandardDB = None
        self._eval_history: EvaluationHistory = None
        self._improvement_history: ImprovementHistory = None

        # Layer 5: Improvement Agents
        self._expert: ExpertAgent = None
        self._reviewer: ReviewerAgent = None
        self._planner: PlannerAgent = None
        self._actor: ActorAgent = None
        self._learner: LearnerAgent = None

        # Layer 6: Pattern Learning
        self._pattern_learner: PatternLearner = None

        # Layer 7: Transfer Learning
        self._transfer_learner: TransferLearner = None

        # Layer 8: Adaptive Components
        self._adaptive: AdaptiveComponents = None

    async def execute(self, **kwargs) -> SwarmResult:
        """Execute with full learning integration."""

        # === 1. Pre-execution: Retrieve learned knowledge ===
        learned_context = await self._retrieve_learned_context(kwargs)

        # === 2. Pattern selection (AUTO or specified) ===
        pattern = await self._select_pattern(kwargs)

        # === 3. Pre-execution learning hook ===
        await self._pre_execute_learning(**kwargs)

        # === 4. Execute with selected pattern ===
        result = await self._execute_with_pattern(pattern, kwargs)

        # === 5. Post-execution learning hook ===
        await self._post_execute_learning(
            result=result,
            pattern=pattern,
            **kwargs
        )

        # === 6. Update all learning layers ===
        await self._update_all_learning(result, pattern, kwargs)

        # === 7. Consolidate to long-term memory ===
        await self._consolidate_to_memory(result, kwargs)

        return result

    async def _select_pattern(self, kwargs) -> CoordinationPattern:
        """
        AUTO pattern selection using ALL learning.

        Decision sources (in priority order):
        1. Historical success (pattern_learner)
        2. Transfer learning (similar tasks)
        3. Swarm intelligence (meta-learning)
        4. Task analysis (LLM)
        5. Fallback (simple heuristics)
        """

        if self.COORDINATION != CoordinationPattern.AUTO:
            return self.COORDINATION

        # 1. Historical: What worked before?
        task_type = self._classify_task(kwargs)
        historical = await self._pattern_learner.get_best_pattern(
            task_type=task_type,
            agent_count=len(self.AGENT_TEAM.agents),
        )
        if historical:
            logger.info(f"🧠 Historical best: {historical.value}")
            return historical

        # 2. Transfer: What worked for similar tasks?
        similar_pattern = await self._transfer_learner.get_pattern(
            task=kwargs.get("task", ""),
            context=kwargs,
        )
        if similar_pattern:
            logger.info(f"🔄 Transfer learning: {similar_pattern.value}")
            return similar_pattern

        # 3. Swarm intelligence: Meta-learned preference
        si_pattern = await self._swarm_intelligence.suggest_pattern(
            task_type=task_type,
            agents=self.AGENT_TEAM.agents,
        )
        if si_pattern:
            logger.info(f"🎯 Swarm intelligence: {si_pattern.value}")
            return si_pattern

        # 4. Task analysis: Use LLM to analyze
        analyzed_pattern = await self._analyze_task_for_pattern(kwargs)
        if analyzed_pattern:
            logger.info(f"📊 Task analysis: {analyzed_pattern.value}")
            return analyzed_pattern

        # 5. Fallback: Simple heuristics
        fallback = self._fallback_pattern_selection(kwargs)
        logger.info(f"⚙️ Fallback: {fallback.value}")
        return fallback

    async def _update_all_learning(
        self,
        result: SwarmResult,
        pattern: CoordinationPattern,
        kwargs: Dict,
    ):
        """Update ALL 8 learning layers."""

        # Prepare reward signal
        reward = self._calculate_reward(result)

        # Layer 1: Store to memory (5 levels)
        await self._memory.store(
            content=f"Task: {kwargs.get('task')} | Pattern: {pattern.value} | Success: {result.success}",
            level=MemoryLevel.EPISODIC,
            metadata={
                "pattern": pattern.value,
                "reward": reward,
                "task_type": self._classify_task(kwargs),
            },
        )

        # Layer 2: TD-Lambda update
        state = self._build_state(kwargs)
        action = {"pattern": pattern.value}
        next_state = self._build_state({**kwargs, "completed": True})

        await self._td_learner.update(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
        )

        # Layer 3: Swarm intelligence update
        await self._swarm_intelligence.record_execution(
            pattern=pattern,
            agents=list(self.AGENT_TEAM.agents.keys()),
            success=result.success,
            metrics=result.metadata,
        )

        # Layer 4: Gold standard evaluation
        if result.success and reward > 0.9:
            # Auto-curate excellent results as gold standards
            await self._gold_db.add_gold_standard(
                task_type=self._classify_task(kwargs),
                input_data=kwargs,
                expected_output=result.output,
                score=reward,
            )

        # Layer 5: Improvement agents (if needed)
        if reward < 0.8:
            # Low performance - trigger improvement cycle
            improvement = await self._run_improvement_cycle(
                result=result,
                kwargs=kwargs,
            )
            if improvement:
                await self._improvement_history.add(improvement)

        # Layer 6: Pattern learning
        await self._pattern_learner.record_performance(
            pattern=pattern,
            task_type=self._classify_task(kwargs),
            agent_count=len(self.AGENT_TEAM.agents),
            success=result.success,
            quality=result.confidence,
            speed=result.execution_time,
        )

        # Layer 7: Transfer learning
        await self._transfer_learner.record_execution(
            task=kwargs.get("task", ""),
            pattern=pattern,
            success=result.success,
            embeddings=self._get_task_embeddings(kwargs),
        )

        # Layer 8: Adaptive components update
        await self._adaptive.update(
            success=result.success,
            reward=reward,
        )
```

### Tests:
- `tests/test_unified_swarm.py` - Test unified architecture
- `tests/test_learning_integration.py` - Test all 8 layers work together
- `tests/test_pattern_selection.py` - Test AUTO mode

---

## 📦 Phase 3: Pattern Implementations (2 hours)

### Files to create:
- `core/intelligence/swarms/patterns/` (NEW directory)
  - `sequential.py` - SEQUENTIAL pattern
  - `parallel.py` - PARALLEL pattern
  - `consensus.py` - CONSENSUS pattern
  - `debate.py` - DEBATE pattern (multi-round + synthesis)
  - `iterative.py` - ITERATIVE pattern (feedback loop)
  - `hierarchical.py` - HIERARCHICAL pattern
  - `blackboard.py` - BLACKBOARD pattern

### Example: DEBATE pattern

```python
# core/intelligence/swarms/patterns/debate.py

async def execute_debate(
    swarm: BaseSwarm,
    task: Any,
    agents: List[Agent],
    rounds: int = 3,
    **kwargs
) -> SwarmResult:
    """
    Multi-round debate with intelligent synthesis.

    Rounds:
      1. Propose: Each agent proposes solution
      2-N. Critique & Refine: Agents critique and improve
      Final. Synthesize: Create best-of-all solution

    Returns:
        SwarmResult with synthesized output
    """

    proposals = {}

    # Round 1: Proposals
    logger.info("🗣️ Round 1: Proposals")
    for agent in agents:
        proposals[agent.name] = await agent.propose(task)

    # Rounds 2-N: Critique & Refine
    for round_num in range(2, rounds + 1):
        logger.info(f"🗣️ Round {round_num}: Critique & Refine")

        critiques = {}
        for agent in agents:
            other_proposals = {
                name: prop
                for name, prop in proposals.items()
                if name != agent.name
            }
            critiques[agent.name] = await agent.critique(other_proposals)

        # Refine based on critiques
        for agent in agents:
            received = [c.get(agent.name) for c in critiques.values() if agent.name in c]
            proposals[agent.name] = await agent.refine(
                current=proposals[agent.name],
                critiques=received,
            )

    # Final: Synthesize (intelligent, not just combine)
    logger.info("🎯 Final: Synthesis")
    synthesizer = swarm._get_synthesizer_agent()
    final_output = await synthesizer.synthesize(
        proposals=proposals,
        task=task,
        criteria=kwargs.get("synthesis_criteria", "best_of_all"),
    )

    return SwarmResult(
        success=True,
        output=final_output,
        confidence=0.95,  # High confidence from debate
        metadata={
            "pattern": "debate",
            "rounds": rounds,
            "proposals_count": len(proposals),
        },
    )
```

### Tests:
- `tests/patterns/test_sequential.py`
- `tests/patterns/test_parallel.py`
- `tests/patterns/test_debate.py`
- `tests/patterns/test_iterative.py`

---

## 📦 Phase 4: Examples (2 hours)

### Create comprehensive examples:

```
examples/unified_swarm/
├── 01_auto_pattern_selection.py      # AUTO mode demo
├── 02_sequential_pattern.py          # SEQUENTIAL demo
├── 03_parallel_pattern.py            # PARALLEL demo
├── 04_consensus_pattern.py           # CONSENSUS demo
├── 05_debate_pattern.py              # DEBATE demo
├── 06_iterative_pattern.py           # ITERATIVE demo
├── 07_custom_stages.py               # CUSTOM with STAGES
├── 08_learning_progression.py        # Show learning over time
├── 09_pattern_transfer.py            # Transfer learning demo
└── 10_world_class_swarm.py           # Complete showcase
```

### Example: AUTO pattern selection

```python
# examples/unified_swarm/01_auto_pattern_selection.py

"""
Example: AUTO Pattern Selection

Demonstrates:
- Swarm automatically selects best coordination pattern
- Pattern selection improves with experience
- All 8 learning layers working together
"""

import asyncio
from Jotty.core.intelligence.swarms import UnifiedSwarm
from Jotty.core.intelligence.swarms.base.agent_team import AgentTeam, CoordinationPattern

class AutoSwarm(UnifiedSwarm):
    """Swarm with AUTO pattern selection."""

    AGENT_TEAM = AgentTeam.define(
        (ResearcherAgent, "Researcher"),
        (AnalystAgent, "Analyst"),
        (WriterAgent, "Writer"),
    )

    # Let swarm decide!
    COORDINATION = CoordinationPattern.AUTO

async def main():
    swarm = AutoSwarm()

    # Task 1: Research (swarm will choose PARALLEL)
    print("\n=== Task 1: Research ===")
    result1 = await swarm.execute(
        task="Research 3 AI companies: OpenAI, Anthropic, Google DeepMind"
    )
    print(f"Selected pattern: {result1.metadata['pattern']}")
    print(f"Rationale: Independent research tasks → PARALLEL")

    # Task 2: Analysis pipeline (swarm will choose SEQUENTIAL)
    print("\n=== Task 2: Analysis ===")
    result2 = await swarm.execute(
        task="Research AI trends, analyze implications, write report"
    )
    print(f"Selected pattern: {result2.metadata['pattern']}")
    print(f"Rationale: Dependent steps → SEQUENTIAL")

    # Task 3: Decision (swarm will choose DEBATE)
    print("\n=== Task 3: Decision ===")
    result3 = await swarm.execute(
        task="Debate: Should we use React or Vue for our new project?"
    )
    print(f"Selected pattern: {result3.metadata['pattern']}")
    print(f"Rationale: Keyword 'debate' + decision needed → DEBATE")

    # Show learning progression
    print("\n=== Learning Progression ===")
    performance = await swarm._pattern_learner.get_performance_stats()
    print(f"Pattern performances:")
    for pattern, stats in performance.items():
        print(f"  {pattern}: {stats['avg_success']:.2%} success, {stats['avg_quality']:.2f} quality")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 📦 Phase 5: Backward Compatibility (1 hour)

### Make old code work:

```python
# core/intelligence/swarms/base/domain_swarm.py

# Keep as alias for backward compatibility
class DomainSwarm(BaseSwarm):
    """
    DEPRECATED: Use BaseSwarm directly.

    This class is kept for backward compatibility.
    All functionality moved to BaseSwarm.
    """

    def __init__(self, config):
        warnings.warn(
            "DomainSwarm is deprecated. Use BaseSwarm instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(config)

# core/intelligence/orchestration/templates/base.py

class SwarmTemplate:
    """
    DEPRECATED: Use BaseSwarm with STAGES instead.

    Old:
        class MySwarm(SwarmTemplate):
            agents = {...}
            pipeline = [...]

    New:
        class MySwarm(BaseSwarm):
            AGENT_TEAM = AgentTeam.define(...)
            STAGES = [...]
            COORDINATION = CoordinationPattern.CUSTOM
    """
    pass
```

---

## 📦 Phase 6: Tests & Validation (2 hours)

### Test suite:

```
tests/unified_swarm/
├── test_all_patterns.py              # Test each pattern works
├── test_auto_selection.py            # Test AUTO mode
├── test_learning_layers.py           # Test all 8 layers
├── test_backward_compat.py           # Old code still works
├── test_examples.py                  # All examples run
└── test_no_regression.py             # Nothing broke
```

### Integration test:

```python
# tests/unified_swarm/test_comprehensive.py

async def test_world_class_swarm():
    """Test everything works together."""

    swarm = UnifiedSwarm()

    # 1. First execution (no learning yet)
    result1 = await swarm.execute(task="Research AI")

    # 2. Check all learning layers active
    assert swarm._memory is not None
    assert swarm._td_learner is not None
    assert swarm._swarm_intelligence is not None
    assert swarm._pattern_learner is not None

    # 3. Second execution (should use learning)
    result2 = await swarm.execute(task="Research AI")

    # 4. Verify learning improved performance
    assert result2.execution_time <= result1.execution_time

    # 5. Verify memory consolidation
    memories = await swarm._memory.retrieve("Research AI", top_k=5)
    assert len(memories) > 0

    # 6. Verify pattern learning
    stats = await swarm._pattern_learner.get_stats()
    assert stats["executions"] >= 2
```

---

## 🎯 Success Criteria

✅ Single unified BaseSwarm replaces 3 separate concepts
✅ AUTO pattern works and learns from experience
✅ All 8 learning layers integrated and active
✅ Examples demonstrate all patterns
✅ Tests validate everything works
✅ Old code still works (backward compat)
✅ No learning lost - everything preserved
✅ World-class swarm manager achieved!

---

## 📅 Timeline

- **Phase 1:** 1 hour - Enhanced patterns
- **Phase 2:** 3 hours - Unified BaseSwarm
- **Phase 3:** 2 hours - Pattern implementations
- **Phase 4:** 2 hours - Examples
- **Phase 5:** 1 hour - Backward compat
- **Phase 6:** 2 hours - Tests

**Total:** ~11 hours of focused work

---

## 🚀 Next Steps

1. Review this plan
2. Get approval
3. Implement phase by phase
4. Test each phase
5. Create examples
6. Document
7. Celebrate world-class swarm manager! 🎉
