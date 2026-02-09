"""
Base Swarm Infrastructure
=========================

Foundation for all Jotty swarms with:
- Self-improving feedback loop (Expert → Reviewer → Planner → Actor)
- Shared resources (memory, context, bus, learner)
- Gold standard evaluation and agent improvement
- Execution tracking and learning

This is the CORE infrastructure that all domain swarms inherit from.

Architecture:
┌─────────────────────────────────────────────────────────────────────────┐
│                         SELF-IMPROVING LOOP                              │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐          │
│  │  Expert  │───►│ Reviewer │───►│ Planner  │───►│  Actor   │          │
│  │  Agent   │    │  Agent   │    │  Agent   │    │  Agent   │          │
│  └────┬─────┘    └────┬─────┘    └────┬─────┘    └────┬─────┘          │
│       │               │               │               │                 │
│       ▼               ▼               ▼               ▼                 │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │                    SHARED RESOURCES                             │    │
│  │  Memory (5-level) | Context | Message Bus | TD-Lambda Learner  │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                              │                                          │
│                              ▼                                          │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │                      GOLD STANDARD DB                           │    │
│  │  Expected outputs, evaluation criteria, improvement history     │    │
│  └────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘

Author: Jotty Team
Date: February 2026
"""

import asyncio
import logging
import json
import hashlib
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple, Callable, Type
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)

# =============================================================================
# RE-EXPORTS FROM EXTRACTED MODULES
# =============================================================================

from .swarm_types import (
    AgentRole,
    EvaluationResult,
    ImprovementType,
    GoldStandard,
    Evaluation,
    ImprovementSuggestion,
    AgentConfig,
    ExecutionTrace,
    SwarmConfig,
    SwarmResult,
)

# Signatures are in swarm_signatures.py (loaded lazily by swarms/__init__.py)
# They're not imported here to avoid triggering DSPy at module load time.
# Direct importers should use: from core.swarms.swarm_signatures import ...

from .evaluation import (
    GoldStandardDB,
    ImprovementHistory,
    EvaluationHistory,
)

from .improvement_agents import (
    ExpertAgent,
    ReviewerAgent,
    PlannerAgent,
    ActorAgent,
    AuditorAgent,
    LearnerAgent,
)

from .registry import (
    SwarmRegistry,
    register_swarm,
)

from ._learning_mixin import SwarmLearningMixin

# =============================================================================
# BASE SWARM CLASS
# =============================================================================

class BaseSwarm(SwarmLearningMixin, ABC):
    """
    Base class for all Jotty swarms.

    Provides:
    - Self-improving feedback loop
    - Shared resource management
    - Execution tracking
    - Gold standard evaluation

    Subclasses implement domain-specific logic.
    """

    def __init__(self, config: SwarmConfig):
        self.config = config
        self._initialized = False

        # Shared resources (lazy init)
        self._memory = None
        self._context = None
        self._bus = None
        self._td_learner = None

        # Self-improvement components
        self._gold_db = None
        self._improvement_history = None
        self._expert = None
        self._reviewer = None
        self._planner = None
        self._actor = None
        self._auditor = None
        self._learner = None

        # Agent0 curriculum integration (SwarmIntelligence)
        self._swarm_intelligence = None
        self._training_mode = False

        # Execution tracking
        self._traces: List[ExecutionTrace] = []
        self._evaluation_history = EvaluationHistory()

        # Learning lifecycle (populated by _pre_execute_learning)
        self._learned_context: Optional[Dict[str, Any]] = None

    def _init_shared_resources(self):
        """Initialize shared swarm resources."""
        if self._initialized:
            return

        # Auto-configure DSPy if needed
        try:
            import dspy
            if not hasattr(dspy.settings, 'lm') or dspy.settings.lm is None:
                try:
                    from ..integration.direct_claude_cli_lm import DirectClaudeCLI
                    lm = DirectClaudeCLI(model="sonnet")
                    dspy.configure(lm=lm)
                    logger.info("🔧 Auto-configured DSPy with DirectClaudeCLI")
                except Exception as e:
                    logger.warning(f"Could not configure DSPy LM: {e}")
        except ImportError:
            logger.warning("DSPy not available, skipping LM auto-configuration")

        # Initialize shared resources
        try:
            from ..agents.dag_agents import SwarmResources
            from ..foundation.data_structures import JottyConfig

            jotty_config = JottyConfig()
            resources = SwarmResources.get_instance(jotty_config)

            self._memory = resources.memory
            self._context = resources.context
            self._bus = resources.bus
            self._td_learner = resources.learner

            logger.info("✅ Shared swarm resources initialized")
        except Exception as e:
            logger.warning(f"SwarmResources not available: {e}")

        # Initialize self-improvement components
        if self.config.enable_self_improvement:
            self._init_self_improvement()

        self._initialized = True

    def _init_self_improvement(self):
        """Initialize self-improvement loop components."""
        # Gold standard database
        gold_path = self.config.gold_standard_path or str(
            Path.home() / "jotty" / "gold_standards" / self.config.domain
        )
        self._gold_db = GoldStandardDB(gold_path)

        # Improvement history
        history_path = str(Path.home() / "jotty" / "improvements" / self.config.domain)
        self._improvement_history = ImprovementHistory(history_path)

        # Create agent configs
        expert_config = AgentConfig(
            role=AgentRole.EXPERT,
            name=f"{self.config.name}_expert",
            system_prompt="You are an expert evaluator for the {domain} domain."
        )
        reviewer_config = AgentConfig(
            role=AgentRole.REVIEWER,
            name=f"{self.config.name}_reviewer",
            system_prompt="You are a senior reviewer analyzing agent performance."
        )
        planner_config = AgentConfig(
            role=AgentRole.PLANNER,
            name=f"{self.config.name}_planner",
            system_prompt="You are a planning expert optimizing task execution."
        )
        actor_config = AgentConfig(
            role=AgentRole.ACTOR,
            name=f"{self.config.name}_actor",
            system_prompt="You are an expert executor applying learnings."
        )

        # Initialize agents
        self._expert = ExpertAgent(expert_config, self._gold_db)
        self._reviewer = ReviewerAgent(reviewer_config, self._improvement_history)
        self._planner = PlannerAgent(planner_config, self._improvement_history)
        self._actor = ActorAgent(actor_config, self._improvement_history)

        # Auditor and Learner agents
        auditor_config = AgentConfig(
            role=AgentRole.AUDITOR,
            name=f"{self.config.name}_auditor",
            system_prompt="You are an auditor verifying evaluation quality."
        )
        learner_config = AgentConfig(
            role=AgentRole.LEARNER,
            name=f"{self.config.name}_learner",
            system_prompt="You are a learner extracting patterns from excellent executions."
        )
        self._auditor = AuditorAgent(auditor_config)
        self._learner = LearnerAgent(learner_config)

        logger.info("✅ Self-improvement loop initialized")

    def _get_intelligence_save_path(self) -> str:
        """Get persistent file path for SwarmIntelligence state."""
        domain = self.config.domain or 'default'
        name = self.config.name or 'base_swarm'
        safe_name = name.replace(' ', '_').replace('/', '_')
        save_dir = Path.home() / "jotty" / "intelligence"
        save_dir.mkdir(parents=True, exist_ok=True)
        return str(save_dir / f"{safe_name}_{domain}.json")

    def connect_swarm_intelligence(self, swarm_intelligence=None, enable_training: bool = False):
        """
        Connect to SwarmIntelligence for Agent0 curriculum integration.

        When connected, the swarm will:
        - Auto-load previous learning state from disk
        - Send executor feedback after each task execution
        - Auto-save state after each feedback event
        - Optionally use curriculum-generated training tasks
        - Benefit from tool-aware weakness detection

        Args:
            swarm_intelligence: SwarmIntelligence instance (creates new if None)
            enable_training: Enable curriculum-based training mode
        """
        if swarm_intelligence is None:
            try:
                from ..orchestration.v2.swarm_intelligence import SwarmIntelligence
                swarm_intelligence = SwarmIntelligence()
            except ImportError:
                logger.warning("SwarmIntelligence not available")
                return

        self._swarm_intelligence = swarm_intelligence
        self._training_mode = enable_training

        # Auto-load previous learning state from disk
        save_path = self._get_intelligence_save_path()
        if Path(save_path).exists():
            loaded = self._swarm_intelligence.load(save_path)
            if loaded:
                stats = self._swarm_intelligence.curriculum_generator.get_curriculum_stats()
                logger.info(
                    f"📂 Loaded previous learning: {stats['feedback_count']} feedback events, "
                    f"{len(stats['tool_success_rates'])} tools tracked"
                )

        if enable_training:
            self._swarm_intelligence.enable_training_mode(True, memory_system=self._memory)

        # Register swarm as an agent
        swarm_name = self.config.name or 'base_swarm'
        self._swarm_intelligence.register_agent(swarm_name)

        logger.info(f"✅ SwarmIntelligence connected (training={enable_training})")

    def _send_executor_feedback(
        self,
        task_type: str,
        success: bool,
        tools_used: List[str] = None,
        execution_time: float = 0.0,
        error_type: str = None
    ):
        """
        Send executor feedback to SwarmIntelligence for curriculum adaptation.

        Agent0 closed-loop: Executor feedback → Curriculum adaptation.
        """
        if not self._swarm_intelligence:
            return

        try:
            import time
            swarm_name = self.config.name or 'base_swarm'

            # Record task result for profile building
            self._swarm_intelligence.record_task_result(
                agent_name=swarm_name,
                task_type=task_type,
                success=success,
                execution_time=execution_time
            )

            # Send executor feedback for curriculum
            self._swarm_intelligence.receive_executor_feedback(
                task_id=f"{swarm_name}_{task_type}_{int(time.time())}",
                success=success,
                tools_used=tools_used or [],
                execution_time=execution_time,
                error_type=error_type,
                task_type=task_type
            )

            logger.debug(f"Agent0 feedback sent: {task_type} success={success}")

            # Auto-save learning state to disk after each feedback
            try:
                save_path = self._get_intelligence_save_path()
                self._swarm_intelligence.save(save_path)
                logger.debug(f"Agent0 state saved to {save_path}")
            except Exception as save_err:
                logger.debug(f"Failed to save Agent0 state: {save_err}")

        except Exception as e:
            logger.debug(f"Failed to send Agent0 feedback: {e}")

    def get_training_task(self, tool_aware: bool = True):
        """
        Get a curriculum-generated training task targeting swarm weaknesses.

        Agent0: Returns task designed to improve weak areas.

        Args:
            tool_aware: Use tool-aware task generation

        Returns:
            SyntheticTask or None if training mode disabled
        """
        if not self._swarm_intelligence or not self._training_mode:
            return None

        swarm_name = self.config.name or 'base_swarm'
        return self._swarm_intelligence.get_training_task(
            target_agent=swarm_name,
            tool_aware=tool_aware
        )

    # =========================================================================
    # LEARNING LIFECYCLE (Agent0 + MorphAgent + Expert Knowledge Stitched)
    # =========================================================================

    async def _run_auto_warmup(self, num_episodes: int = 3) -> Dict:
        """Cold-start init. No fake data — first real executions build baselines."""
        si = self._swarm_intelligence
        if not si:
            return {'seeded': 0, 'mode': 'cold_start'}

        swarm_name = self.config.name or 'base_swarm'
        si.register_agent(swarm_name)

        # Mark cold-start so agents know there is no history
        si.collective_memory.append({
            'agent': swarm_name, 'task_type': 'cold_start',
            'success': True, 'execution_time': 0.0,
            'context': {'cold_start': True},
            'timestamp': __import__('time').time()
        })

        try:
            save_path = self._get_intelligence_save_path()
            si.save(save_path)
        except Exception:
            pass

        logger.info(f"Cold-start for {swarm_name}: real executions will build baselines")
        return {'seeded': 0, 'swarm': swarm_name, 'mode': 'cold_start'}

    def _manage_tools(self) -> Dict:
        """
        Analyze tool performance and log warnings for weak tools.

        Delegates to SwarmIntelligence.tool_manager.analyze_tools() using
        the curriculum generator's tracked tool success rates.

        Returns:
            Dict with weak_tools, strong_tools, suggested_removals, replacements
        """
        if not self._swarm_intelligence:
            return {'weak_tools': [], 'strong_tools': [], 'suggested_removals': [], 'replacements': {}}

        si = self._swarm_intelligence
        swarm_name = self.config.name or 'base_swarm'
        tool_rates = si.curriculum_generator._tool_success_rates

        # Auto-populate registry from tracked tool rates
        si.tool_manager.auto_register_from_rates(tool_rates)

        analysis = si.tool_manager.analyze_tools(tool_rates, swarm_name)

        # Log warnings for weak tools
        for weak in analysis.get('weak_tools', []):
            logger.warning(
                f"Weak tool detected: {weak['tool']} "
                f"({weak['success_rate']:.0%} success over {weak['total']} uses)"
            )

        # After logging warnings, ACTUALLY update tool assignments
        replacements = analysis.get('replacements', {})
        removals = analysis.get('suggested_removals', [])

        if replacements or removals:
            add_tools = []
            remove_tools = []

            for weak_tool, replacement_list in replacements.items():
                if replacement_list:
                    best = replacement_list[0]  # First replacement
                    add_tools.append(best['name'])
                    remove_tools.append(weak_tool)
                    logger.info(
                        f"Tool swap: {weak_tool} -> {best['name']} "
                        f"(reason: {best.get('reason', 'low success rate')})"
                    )

            for tool_name in removals:
                if tool_name not in remove_tools:
                    remove_tools.append(tool_name)

            if add_tools or remove_tools:
                si.tool_manager.update_assignments(
                    swarm_name,
                    add=add_tools,
                    remove=remove_tools
                )
                logger.info(f"Tool assignments updated: +{add_tools} -{remove_tools}")

        return analysis

    def _get_active_tools(self, default_tools: List[str] = None) -> List[str]:
        """
        Get dynamic tool list: defaults + additions - deactivated.

        Delegates to SwarmIntelligence.tool_manager.get_active_tools().

        Args:
            default_tools: Default tool names for this swarm

        Returns:
            List of active tool names
        """
        if not self._swarm_intelligence:
            return default_tools or []

        swarm_name = self.config.name or 'base_swarm'
        return self._swarm_intelligence.tool_manager.get_active_tools(
            swarm_name, default_tools or []
        )

    def _curate_gold_standard(self, task_type, input_data, output_data, evaluation):
        """Auto-curate gold standard from execution scoring >= 0.9."""
        if not self._gold_db:
            return
        existing = self._gold_db.find_similar(task_type, input_data)
        if existing and existing.version >= self.config.gold_standard_max_version:
            return  # Don't over-accumulate
        criteria = evaluation.scores if evaluation.scores else {'overall': 1.0}
        gold = GoldStandard(
            id="", domain=self.config.domain, task_type=task_type,
            input_data=input_data, expected_output=output_data,
            evaluation_criteria=criteria,
            version=(existing.version + 1) if existing else 1
        )
        gs_id = self._gold_db.add(gold)
        logger.info(f"Auto-curated gold standard {gs_id} for {task_type} "
                    f"(score={evaluation.overall_score:.2f})")

    async def _evaluate_output(
        self,
        output: Dict[str, Any],
        task_type: str,
        input_data: Dict[str, Any]
    ) -> Optional[Evaluation]:
        """Evaluate output against gold standard if available."""
        if not self.config.enable_self_improvement or not self._expert:
            return None

        # Find matching gold standard
        gold_standard = self._gold_db.find_similar(task_type, input_data)
        if not gold_standard:
            logger.debug(f"No gold standard found for task_type: {task_type}")
            return None

        evaluation = await self._expert.evaluate(
            gold_standard_id=gold_standard.id,
            actual_output=output,
            context=json.dumps({'task_type': task_type, 'input': input_data})
        )

        self._evaluation_history.record(evaluation)
        return evaluation

    def _agent_context(self, agent_name: str) -> str:
        """Build per-agent learned context string.
        Convenience wrapper for subclasses to use in _init_agents()."""
        if not self._learned_context:
            return ""
        return self._build_learned_context_string(agent_name=agent_name)

    def _trace_phase(
        self,
        agent_name: str,
        agent_role: AgentRole,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        success: bool,
        phase_start: datetime,
        tools_used: List[str] = None
    ):
        """Record a phase trace with automatic timing.
        Convenience wrapper for subclasses to call after each execution phase."""
        elapsed = (datetime.now() - phase_start).total_seconds()
        self._record_trace(
            agent_name=agent_name,
            agent_role=agent_role,
            input_data=input_data,
            output_data=output_data,
            execution_time=elapsed,
            success=success,
            tools_used=tools_used or []
        )

    # =========================================================================
    # ARXIV SWARM INTEGRATION (Handoff, Coalition, Smart Routing)
    # =========================================================================

    def _handoff_task(
        self,
        task_id: str,
        to_agent: str,
        task_type: str,
        context: Dict = None,
        partial_result: Any = None,
        progress: float = 0.0
    ):
        """
        Hand off task to another agent with context preservation.

        SwarmAgentic pattern integrated into BaseSwarm.
        """
        if not self._swarm_intelligence:
            logger.warning("Handoff requires SwarmIntelligence connection")
            return None

        from_agent = self.config.name or 'base_swarm'

        handoff = self._swarm_intelligence.initiate_handoff(
            task_id=task_id,
            from_agent=from_agent,
            to_agent=to_agent,
            task_type=task_type,
            context=context or {},
            partial_result=partial_result,
            progress=progress
        )

        logger.info(f"Task handoff: {from_agent} → {to_agent} ({task_type})")
        return handoff

    def _accept_handoff(self, task_id: str):
        """
        Accept a pending handoff for this swarm.

        Returns HandoffContext with preserved state.
        """
        if not self._swarm_intelligence:
            return None

        swarm_name = self.config.name or 'base_swarm'
        return self._swarm_intelligence.accept_handoff(task_id, swarm_name)

    def _get_pending_handoffs(self):
        """Get all pending handoffs for this swarm."""
        if not self._swarm_intelligence:
            return []

        swarm_name = self.config.name or 'base_swarm'
        return self._swarm_intelligence.get_pending_handoffs(swarm_name)

    def _form_coalition(
        self,
        task_type: str,
        required_roles: List[str] = None,
        min_agents: int = 2,
        max_agents: int = 5
    ):
        """
        Form coalition for complex multi-agent tasks.

        SwarmAgentic pattern: Dynamic team assembly.
        """
        if not self._swarm_intelligence:
            logger.warning("Coalition requires SwarmIntelligence connection")
            return None

        coalition = self._swarm_intelligence.form_coalition(
            task_type=task_type,
            required_roles=required_roles,
            min_agents=min_agents,
            max_agents=max_agents
        )

        if coalition:
            logger.info(f"Coalition formed: {coalition.coalition_id} with {len(coalition.members)} agents")

        return coalition

    def _dissolve_coalition(self, coalition_id: str):
        """Dissolve coalition after task completion."""
        if self._swarm_intelligence:
            self._swarm_intelligence.dissolve_coalition(coalition_id)

    def _smart_route(
        self,
        task_id: str,
        task_type: str,
        task_description: str = "",
        prefer_coalition: bool = False,
        use_auction: bool = False
    ) -> Dict[str, Any]:
        """
        Smart routing combining all arXiv swarm patterns.

        Integrates: handoff, hierarchy, auction, coalition.

        Returns:
            Dict with assigned_agent, coalition_id, method, confidence
        """
        if not self._swarm_intelligence:
            return {"assigned_agent": None, "method": "none", "confidence": 0.0}

        return self._swarm_intelligence.smart_route(
            task_id=task_id,
            task_type=task_type,
            task_description=task_description,
            prefer_coalition=prefer_coalition,
            use_auction=use_auction,
            use_hierarchy=True
        )

    def _gossip_broadcast(self, message_type: str, content: Dict[str, Any]):
        """
        Broadcast message via gossip protocol.

        SwarmSys O(log n) dissemination pattern.
        """
        if not self._swarm_intelligence:
            return None

        swarm_name = self.config.name or 'base_swarm'
        return self._swarm_intelligence.gossip_broadcast(
            origin_agent=swarm_name,
            message_type=message_type,
            content=content
        )

    def _gossip_receive(self) -> List:
        """Receive gossip messages for this swarm."""
        if not self._swarm_intelligence:
            return []

        swarm_name = self.config.name or 'base_swarm'
        return self._swarm_intelligence.gossip_receive(swarm_name)

    def _build_supervisor_tree(self, agents: List[str] = None):
        """
        Build hierarchical supervisor tree for O(log n) coordination.

        SwarmSys pattern.
        """
        if self._swarm_intelligence:
            self._swarm_intelligence.build_supervisor_tree(agents)

    def _get_supervisor(self, agent: str = None) -> str:
        """Get supervisor for an agent (or self)."""
        if not self._swarm_intelligence:
            return None

        agent = agent or self.config.name or 'base_swarm'
        return self._swarm_intelligence.get_supervisor(agent)

    # =========================================================================
    # WORK-STEALING & LOAD BALANCING
    # =========================================================================

    def _get_load(self) -> float:
        """Get current load of this swarm."""
        if not self._swarm_intelligence:
            return 0.0
        swarm_name = self.config.name or 'base_swarm'
        return self._swarm_intelligence.get_agent_load(swarm_name)

    def _balance_load(self) -> List[Dict]:
        """Rebalance work across the swarm."""
        if not self._swarm_intelligence:
            return []
        return self._swarm_intelligence.balance_load()

    def _work_steal(self) -> bool:
        """Attempt to steal work if idle."""
        if not self._swarm_intelligence:
            return False
        swarm_name = self.config.name or 'base_swarm'
        result = self._swarm_intelligence.work_steal(swarm_name)
        return result is not None

    # =========================================================================
    # FAILURE RECOVERY
    # =========================================================================

    def _record_failure(
        self,
        task_id: str,
        task_type: str,
        error_type: str = "unknown",
        context: Dict = None
    ) -> str:
        """
        Record task failure and get reassignment.

        Returns new agent name or None.
        """
        if not self._swarm_intelligence:
            return None
        swarm_name = self.config.name or 'base_swarm'
        return self._swarm_intelligence.record_failure(
            task_id=task_id,
            agent=swarm_name,
            task_type=task_type,
            error_type=error_type,
            context=context
        )

    # =========================================================================
    # PRIORITY QUEUE
    # =========================================================================

    def _enqueue_task(
        self,
        task_id: str,
        task_type: str,
        priority: int = 5,
        context: Dict = None
    ):
        """Add task to priority queue."""
        if self._swarm_intelligence:
            self._swarm_intelligence.enqueue_task(
                task_id=task_id,
                task_type=task_type,
                priority=priority,
                context=context
            )

    def _dequeue_task(self) -> Dict:
        """Get next priority task."""
        if not self._swarm_intelligence:
            return None
        return self._swarm_intelligence.dequeue_task()

    def _escalate_task(self, task_id: str, new_priority: int):
        """Escalate task priority."""
        if self._swarm_intelligence:
            self._swarm_intelligence.escalate_priority(task_id, new_priority)

    # =========================================================================
    # TASK DECOMPOSITION
    # =========================================================================

    def _decompose_task(
        self,
        task_id: str,
        task_type: str,
        subtasks: List[Dict],
        parallel: bool = True
    ) -> List[str]:
        """
        Decompose complex task into subtasks.

        Args:
            task_id: Parent task ID
            task_type: Type of parent task
            subtasks: List of {"type": str, "context": dict, "priority": int}
            parallel: Whether subtasks can run in parallel

        Returns:
            List of subtask IDs.
        """
        if not self._swarm_intelligence:
            return []
        return self._swarm_intelligence.decompose_task(
            task_id=task_id,
            task_type=task_type,
            subtasks=subtasks,
            parallel=parallel
        )

    def _aggregate_results(self, parent_task_id: str, results: Dict) -> Dict:
        """Aggregate subtask results."""
        if not self._swarm_intelligence:
            return {"results": results}
        return self._swarm_intelligence.aggregate_subtask_results(
            parent_task_id=parent_task_id,
            results=results
        )

    # =========================================================================
    # BYZANTINE CONSENSUS
    # =========================================================================

    def _byzantine_vote(
        self,
        question: str,
        options: List[str],
        voters: List[str] = None
    ) -> Dict:
        """
        Run Byzantine fault-tolerant vote.

        Requires 2/3 majority for consensus.
        """
        if not self._swarm_intelligence:
            return {"decision": options[0] if options else None, "consensus": False}
        return self._swarm_intelligence.byzantine_vote(
            question=question,
            options=options,
            voters=voters
        )

    # =========================================================================
    # SWARM STATUS
    # =========================================================================

    def _get_swarm_status(self) -> Dict:
        """Get comprehensive swarm status."""
        if not self._swarm_intelligence:
            return {"health_score": 0.5}
        return self._swarm_intelligence.get_swarm_status()

    # =========================================================================
    # CIRCUIT BREAKER
    # =========================================================================

    def _record_circuit_failure(self, agent: str = None):
        """Record failure for circuit breaker."""
        if self._swarm_intelligence:
            agent = agent or self.config.name or 'base_swarm'
            self._swarm_intelligence.record_circuit_failure(agent)

    def _record_circuit_success(self, agent: str = None):
        """Record success - resets circuit breaker."""
        if self._swarm_intelligence:
            agent = agent or self.config.name or 'base_swarm'
            self._swarm_intelligence.record_circuit_success(agent)

    def _check_circuit(self, agent: str = None) -> bool:
        """Check if agent circuit is open (blocked)."""
        if not self._swarm_intelligence:
            return True
        agent = agent or self.config.name or 'base_swarm'
        return self._swarm_intelligence.check_circuit(agent)

    # =========================================================================
    # BACKPRESSURE
    # =========================================================================

    def _get_backpressure(self) -> float:
        """Get current swarm backpressure (0-1)."""
        if not self._swarm_intelligence:
            return 0.0
        return self._swarm_intelligence.calculate_backpressure()

    def _should_accept_task(self, priority: int = 5) -> bool:
        """Check if swarm should accept new task."""
        if not self._swarm_intelligence:
            return True
        return self._swarm_intelligence.should_accept_task(priority)

    # =========================================================================
    # LEADERSHIP & LIFECYCLE
    # =========================================================================

    def _elect_leader(self, candidates: List[str] = None, task_type: str = None) -> str:
        """Elect leader for a task."""
        if not self._swarm_intelligence:
            return candidates[0] if candidates else None
        return self._swarm_intelligence.elect_leader(candidates, task_type)

    def _get_adaptive_timeout(self, agent: str = None, task_type: str = None) -> float:
        """Get adaptive timeout for agent."""
        if not self._swarm_intelligence:
            return 30.0
        agent = agent or self.config.name or 'base_swarm'
        return self._swarm_intelligence.get_adaptive_timeout(agent, task_type or "general")

    # =========================================================================
    # PARALLEL EXECUTION
    # =========================================================================

    async def _execute_parallel(self, tasks: List[Dict], timeout: float = 30.0) -> List[Dict]:
        """
        Execute multiple tasks in parallel.

        Args:
            tasks: List of {"task_id", "func", "args", "kwargs"}
            timeout: Timeout per task

        Returns:
            List of results
        """
        if not self._swarm_intelligence:
            # Fallback: sequential execution
            results = []
            for task in tasks:
                try:
                    func = task.get("func")
                    args = task.get("args", [])
                    kwargs = task.get("kwargs", {})
                    if asyncio.iscoroutinefunction(func):
                        result = await func(*args, **kwargs)
                    else:
                        result = func(*args, **kwargs)
                    results.append({"task_id": task.get("task_id"), "success": True, "result": result})
                except Exception as e:
                    results.append({"task_id": task.get("task_id"), "success": False, "error": str(e)})
            return results

        return await self._swarm_intelligence.execute_parallel(tasks, timeout)

    async def _parallel_map(self, items: List, func, max_concurrent: int = 5) -> List:
        """Apply function to items in parallel."""
        if not self._swarm_intelligence:
            # Fallback
            results = []
            for item in items:
                try:
                    if asyncio.iscoroutinefunction(func):
                        results.append(await func(item))
                    else:
                        results.append(func(item))
                except:
                    results.append(None)
            return results

        return await self._swarm_intelligence.parallel_map(items, func, max_concurrent)

    async def _process_in_chunks(
        self,
        items: List,
        chunk_size: int,
        process_func,
        delay: float = 0.1
    ) -> List:
        """Process items in chunks to avoid timeouts."""
        if not self._swarm_intelligence:
            return await process_func(items)

        return await self._swarm_intelligence.process_in_chunks(
            items, chunk_size, process_func, delay
        )

    # =========================================================================
    # SMART CACHING
    # =========================================================================

    def _cache_result(self, key: str, result: Any, ttl: float = 3600.0):
        """Cache a result."""
        if self._swarm_intelligence:
            self._swarm_intelligence.cache_result(key, result, ttl)

    def _get_cached(self, key: str) -> Any:
        """Get cached result or None."""
        if not self._swarm_intelligence:
            return None
        return self._swarm_intelligence.get_cached(key)

    def _get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        if not self._swarm_intelligence:
            return {"hits": 0, "misses": 0, "hit_rate": 0, "size": 0}
        return self._swarm_intelligence.get_cache_stats()

    @abstractmethod
    async def execute(self, *args, **kwargs) -> SwarmResult:
        """Execute the swarm's main task. Implemented by subclasses."""
        pass

    def add_gold_standard(
        self,
        task_type: str,
        input_data: Dict[str, Any],
        expected_output: Dict[str, Any],
        evaluation_criteria: Dict[str, float]
    ) -> str:
        """Add a gold standard for evaluation."""
        if not self._gold_db:
            self._init_shared_resources()

        gold_standard = GoldStandard(
            id="",  # Will be generated
            domain=self.config.domain,
            task_type=task_type,
            input_data=input_data,
            expected_output=expected_output,
            evaluation_criteria=evaluation_criteria
        )

        return self._gold_db.add(gold_standard)

    def get_improvement_suggestions(self) -> List[Dict]:
        """Get pending improvement suggestions."""
        if not self._improvement_history:
            return []
        return self._improvement_history.get_pending_suggestions()

    def apply_improvement(self, suggestion_id: str):
        """Mark an improvement as applied."""
        if self._improvement_history:
            self._improvement_history.mark_applied(suggestion_id)

