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
import dspy
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

from .swarm_signatures import (
    ExpertEvaluationSignature,
    ReviewerAnalysisSignature,
    PlannerOptimizationSignature,
    ActorExecutionSignature,
    AuditorVerificationSignature,
    LearnerExtractionSignature,
)

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

# =============================================================================
# BASE SWARM CLASS
# =============================================================================

class BaseSwarm(ABC):
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
        if not hasattr(dspy.settings, 'lm') or dspy.settings.lm is None:
            try:
                from ..integration.direct_claude_cli_lm import DirectClaudeCLI
                lm = DirectClaudeCLI(model="sonnet")
                dspy.configure(lm=lm)
                logger.info("🔧 Auto-configured DSPy with DirectClaudeCLI")
            except Exception as e:
                logger.warning(f"Could not configure DSPy LM: {e}")

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

    async def _pre_execute_learning(self) -> Dict[str, Any]:
        """
        Pre-execution learning hook. Called at start of execute().

        Auto-connects SwarmIntelligence, loads saved state, runs warmup
        on first run, computes MorphAgent scores, analyzes tool performance,
        and stitches Agent0 + MorphAgent findings into a learned context dict.

        Returns:
            Dict with learning context (has_learning, tool_performance,
            agent_scores, weak_tools, recommendations, etc.)
        """
        learned_context = {
            'has_learning': False,
            'tool_performance': {},
            'agent_scores': {},
            'weak_tools': [],
            'strong_tools': [],
            'recommendations': [],
            'warmup_completed': False,
        }

        try:
            # 1. Auto-connect SwarmIntelligence if not connected
            if not self._swarm_intelligence:
                self.connect_swarm_intelligence()

            si = self._swarm_intelligence
            if not si:
                self._learned_context = learned_context
                return learned_context

            # 2. Auto-warmup if first run (no feedback history yet)
            stats = si.curriculum_generator.get_curriculum_stats()
            save_path = self._get_intelligence_save_path()
            if stats['feedback_count'] == 0 and not Path(save_path).exists():
                warmup_result = await self._run_auto_warmup()
                learned_context['warmup_completed'] = True
                logger.info("Auto-warmup complete — seeded initial learning data")

            # 3. Compute MorphAgent scores for all registered agents
            if si.agent_profiles:
                morph_scores = si.morph_scorer.compute_all_scores(si.agent_profiles)
                for agent_name, scores in morph_scores.items():
                    profile = si.agent_profiles.get(agent_name)
                    learned_context['agent_scores'][agent_name] = {
                        'rcs': scores.rcs,
                        'rds': scores.rds,
                        'tras': scores.tras,
                        'consistency': scores.rcs_components.get('consistency', 0.5),
                        'focus': scores.rcs_components.get('focus', 0.5),
                        'specialization': scores.rcs_components.get('specialization', 0.5),
                        'total_tasks': profile.total_tasks if profile else 0,
                    }

            # 4. Analyze tool success rates via ToolManager
            tool_analysis = self._manage_tools()
            learned_context['tool_performance'] = stats.get('tool_success_rates', {})
            learned_context['weak_tools'] = tool_analysis.get('weak_tools', [])
            learned_context['strong_tools'] = tool_analysis.get('strong_tools', [])

            # 5. STITCH: Combine weak tool + inconsistent agent = PRIORITY
            recommendations = []
            for weak in learned_context['weak_tools']:
                tool_name = weak['tool']
                rate = weak['success_rate']
                for agent_name, agent_data in learned_context['agent_scores'].items():
                    consistency = agent_data.get('consistency', 0.5)
                    if consistency < 0.5:
                        recommendations.insert(0, {
                            'priority': 'HIGH',
                            'type': 'tool_and_agent',
                            'tool': tool_name,
                            'tool_rate': rate,
                            'agent': agent_name,
                            'consistency': consistency,
                            'action': f"PRIORITY: Replace {tool_name} ({rate:.0%} success) AND "
                                      f"stabilize {agent_name} (consistency={consistency:.2f})"
                        })
                    else:
                        recommendations.append({
                            'priority': 'MEDIUM',
                            'type': 'tool_only',
                            'tool': tool_name,
                            'tool_rate': rate,
                            'action': f"Replace {tool_name} ({rate:.0%} success) — agent {agent_name} is stable"
                        })

            # Add agent-only warnings (consistent tool but inconsistent agent)
            for agent_name, agent_data in learned_context['agent_scores'].items():
                consistency = agent_data.get('consistency', 0.5)
                if consistency < 0.5 and not any(r.get('agent') == agent_name for r in recommendations):
                    recommendations.append({
                        'priority': 'LOW',
                        'type': 'agent_only',
                        'agent': agent_name,
                        'consistency': consistency,
                        'action': f"Warn {agent_name}: outputs inconsistent (consistency={consistency:.2f})"
                    })

            learned_context['recommendations'] = recommendations

            # 6. Retrieve expert domain knowledge from HierarchicalMemory
            expert_knowledge = self._retrieve_expert_knowledge()
            learned_context['expert_knowledge'] = expert_knowledge

            # 7. Analyze prior failures for recovery
            prior_failures = self._analyze_prior_failures()
            learned_context['prior_failures'] = prior_failures

            # 8. Analyze morph score trends (improving vs declining)
            score_trends = {}
            if si and si.morph_score_history and len(si.morph_score_history) >= 2:
                latest = si.morph_score_history[-1].get('scores', {})
                # Compare with 3 runs ago (or earliest available)
                compare_idx = max(0, len(si.morph_score_history) - 4)
                earlier = si.morph_score_history[compare_idx].get('scores', {})
                for agent_name_key in latest:
                    curr_rcs = latest[agent_name_key].get('rcs', 0)
                    prev_rcs = earlier.get(agent_name_key, {}).get('rcs', 0)
                    if prev_rcs > 0:
                        delta = curr_rcs - prev_rcs
                        if abs(delta) > 0.02:  # Only report meaningful changes
                            score_trends[agent_name_key] = {
                                'current': curr_rcs,
                                'previous': prev_rcs,
                                'delta': delta,
                                'direction': 'improving' if delta > 0 else 'declining'
                            }
            learned_context['score_trends'] = score_trends

            learned_context['has_learning'] = bool(
                learned_context['tool_performance'] or
                learned_context['agent_scores'] or
                learned_context['warmup_completed'] or
                learned_context['expert_knowledge'] or
                learned_context['prior_failures'] or
                learned_context['score_trends']
            )

            self._learned_context = learned_context
            if learned_context['has_learning']:
                expert_count = len(learned_context.get('expert_knowledge', []))
                logger.info(
                    f"Pre-execution learning: {len(learned_context['tool_performance'])} tools tracked, "
                    f"{len(learned_context['agent_scores'])} agents scored, "
                    f"{len(recommendations)} recommendations, "
                    f"{expert_count} expert patterns loaded"
                )

        except Exception as e:
            logger.debug(f"Pre-execution learning skipped: {e}")
            self._learned_context = learned_context

        return learned_context

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

    def _build_learned_context_string(self, agent_name: str = None) -> str:
        """
        Convert self._learned_context into injectable prompt text.

        Produces a compact string suitable for appending to DSPy agent inputs,
        so agents are aware of prior tool performance, agent consistency, and
        priority actions.

        Args:
            agent_name: Optional specific agent to tailor context for

        Returns:
            String like:
            '## Prior Learning
            Tool Performance: arxiv_fetch 100% RELIABLE, content_generate 45% WEAK
            Agent Notes: ContentPolisher has inconsistent outputs (consistency=0.3)
            Action: Validate content_generate output carefully before using.'
        """
        if not self._learned_context or not self._learned_context.get('has_learning'):
            return ""

        ctx = self._learned_context
        lines = ["## Prior Learning"]

        # Tool performance summary
        tool_parts = []
        for tool_info in ctx.get('strong_tools', []):
            tool_parts.append(f"{tool_info.get('tool', '?')} {tool_info.get('success_rate', 0):.0%} RELIABLE")
        for tool_info in ctx.get('weak_tools', []):
            tool_parts.append(f"{tool_info.get('tool', '?')} {tool_info.get('success_rate', 0):.0%} WEAK")

        if tool_parts:
            lines.append(f"Tool Performance: {', '.join(tool_parts)}")

        # Agent-specific context (both positive reinforcement and warnings)
        agent_notes = []
        scores = ctx.get('agent_scores', {})
        if agent_name and agent_name in scores:
            agent_data = scores[agent_name]
            rcs = agent_data.get('rcs', 0)
            consistency = agent_data.get('consistency', 0.5)
            focus = agent_data.get('focus', 0.5)
            total_tasks = agent_data.get('total_tasks', 0)
            # Competence and focus feedback require enough history to be meaningful
            if total_tasks >= 2:
                # Tiered competence feedback — always push for higher
                if rcs >= 0.85:
                    agent_notes.append(
                        f"Competence {rcs:.2f} — excellent, maintain this standard"
                    )
                elif rcs >= 0.6:
                    agent_notes.append(
                        f"Competence {rcs:.2f} — good but target >0.85, push harder on quality"
                    )
                elif rcs >= 0.4:
                    agent_notes.append(
                        f"Competence {rcs:.2f} — needs improvement, aim for >0.6"
                    )
                elif rcs > 0:
                    agent_notes.append(
                        f"Competence {rcs:.2f} — critical, significant quality issues"
                    )
                # Focus feedback
                if focus >= 0.85:
                    agent_notes.append("Focus is excellent — stay specialized")
                elif focus >= 0.6:
                    agent_notes.append(f"Focus {focus:.2f} — good but tighten specialization")
                elif focus > 0 and focus < 0.4:
                    agent_notes.append(f"Focus {focus:.2f} — too scattered, narrow your scope")
            # Consistency warnings stay unguarded (always useful even for new agents)
            if consistency < 0.5:
                agent_notes.append(
                    f"Consistency {consistency:.2f} — outputs vary too much, "
                    f"be extra careful with accuracy"
                )
        elif scores:
            # Summary for orchestrator or unmatched agents
            high_performers = []
            needs_improvement = []
            low_performers = []
            for name, agent_data in scores.items():
                rcs = agent_data.get('rcs', 0)
                consistency = agent_data.get('consistency', 0.5)
                if rcs >= 0.85:
                    high_performers.append(name)
                elif rcs < 0.5 and rcs > 0:
                    needs_improvement.append(f"{name}({rcs:.2f})")
                if consistency < 0.5:
                    low_performers.append(
                        f"{name} inconsistent ({consistency:.2f})"
                    )
            if high_performers:
                agent_notes.append(f"Strong agents: {', '.join(high_performers)}")
            if needs_improvement:
                agent_notes.append(f"Need improvement: {', '.join(needs_improvement)}")
            if low_performers:
                agent_notes.extend(low_performers)

        # Specialization label from AgentProfile
        si = self._swarm_intelligence
        if si and agent_name and agent_name in getattr(si, 'agent_profiles', {}):
            from ..orchestration.v2.swarm_intelligence import AgentSpecialization
            profile = si.agent_profiles[agent_name]
            spec = profile.specialization
            if spec != AgentSpecialization.GENERALIST:
                agent_notes.append(f"Specialization: {spec.value} — leverage this strength")

            # Per-agent time budget from profile
            if profile.avg_execution_time > 0 and profile.total_tasks >= 2:
                avg_t = profile.avg_execution_time
                agent_notes.append(f"Avg execution: {avg_t:.0f}s over {profile.total_tasks} tasks")

        if agent_notes:
            lines.append(f"Agent Notes: {'; '.join(agent_notes)}")

        # Execution patterns from collective memory (what works, typical timings)
        if si and si.collective_memory:
            recent = si.collective_memory[-20:]
            successes = [m for m in recent if m.get('success')]
            if successes:
                # Build timing expectations per task type
                from collections import defaultdict
                task_times = defaultdict(list)
                for m in successes:
                    tt = m.get('task_type', '')
                    if tt and m.get('execution_time', 0) > 0:
                        task_times[tt].append(m['execution_time'])
                if task_times:
                    timing_parts = []
                    for tt, times in task_times.items():
                        avg = sum(times) / len(times)
                        timing_parts.append(f"{tt}~{avg:.0f}s")
                    if len(timing_parts) <= 6:
                        lines.append(f"Typical timing: {', '.join(timing_parts)}")
                # Success streak info
                total_recent = len(recent)
                success_rate = len(successes) / total_recent if total_recent else 0
                if success_rate >= 0.9 and total_recent >= 5:
                    lines.append(
                        f"Track record: {len(successes)}/{total_recent} recent successes — "
                        f"maintain this standard"
                    )

        # Priority recommendations (HIGH first)
        recommendations = ctx.get('recommendations', [])
        high_priority = [r for r in recommendations if r.get('priority') == 'HIGH']
        if high_priority:
            actions = [r['action'] for r in high_priority[:3]]
            lines.append(f"Action: {'; '.join(actions)}")
        elif recommendations:
            lines.append(f"Action: {recommendations[0]['action']}")

        # Evaluation quality bar from persistent history
        if hasattr(self, '_evaluation_history'):
            avg_score = self._evaluation_history.get_average_score(5)
            eval_count = len(self._evaluation_history.evaluations)
            if eval_count >= 2 and avg_score > 0:
                if avg_score >= 0.9:
                    lines.append(
                        f"Quality bar: avg {avg_score:.0%} over {eval_count} evaluations — "
                        f"excellent standard, don't regress"
                    )
                elif avg_score >= 0.7:
                    lines.append(
                        f"Quality bar: avg {avg_score:.0%} over {eval_count} evaluations — "
                        f"good but push for higher"
                    )
                else:
                    lines.append(
                        f"Quality bar: avg {avg_score:.0%} over {eval_count} evaluations — "
                        f"needs significant improvement"
                    )

        # Improvement suggestions from prior cycles (what to improve + what worked)
        if hasattr(self, '_improvement_history') and self._improvement_history:
            pending = self._improvement_history.get_pending_suggestions()
            successful = self._improvement_history.get_successful_improvements()
            if pending or successful:
                imp_lines = []
                # Show successful improvements so agents know what works
                for s in successful[-3:]:
                    suggestion = s.get('suggestion', {})
                    desc = suggestion.get('description', '')
                    if desc:
                        imp_lines.append(f"- Applied successfully: {desc[:120]}")
                # Show pending improvements as directives
                for p in pending[-3:]:
                    suggestion = p.get('suggestion', {})
                    desc = suggestion.get('description', '')
                    priority = suggestion.get('priority', 'MEDIUM')
                    target_agent = suggestion.get('agent_role', '')
                    # Only show agent-specific improvements to that agent
                    if agent_name and target_agent and target_agent != agent_name:
                        continue
                    if desc:
                        imp_lines.append(f"- [{priority}] TODO: {desc[:120]}")
                if imp_lines:
                    lines.append("## Improvement Directives")
                    lines.extend(imp_lines)

        # Expert domain knowledge from HierarchicalMemory
        expert_knowledge = ctx.get('expert_knowledge', [])
        if expert_knowledge:
            expert_lines = []
            for imp in expert_knowledge[:5]:  # Top 5 patterns
                pattern = imp.get('learned_pattern', '')
                if pattern:
                    # Truncate long patterns for prompt efficiency
                    if len(pattern) > 150:
                        pattern = pattern[:147] + "..."
                    expert_lines.append(f"- {pattern}")

            if expert_lines:
                lines.append("## Expert Knowledge")
                lines.extend(expert_lines)

        # Failure recovery from prior runs
        prior_failures = ctx.get('prior_failures', [])
        if prior_failures:
            failure_lines = ["## Prior Failures (Avoid Repeating)"]
            for f in prior_failures[:3]:
                if f.get('source') == 'evaluation':
                    feedback = f.get('feedback', '')
                    if feedback:
                        failure_lines.append(f"- Previous run scored {f.get('score', 0):.0%}: {feedback[:100]}")
                elif f.get('source') == 'collective_memory':
                    failure_lines.append(
                        f"- Agent {f.get('agent', '?')} failed task {f.get('task_type', '?')}"
                    )
                elif f.get('source') == 'memory':
                    failure_lines.append(f"- {f.get('pattern', 'unknown failure')}")
            if len(failure_lines) > 1:
                lines.extend(failure_lines)

        # Morph score trends — show improvement/decline direction
        score_trends = ctx.get('score_trends', {})
        if score_trends:
            trend_lines = []
            for trend_agent, trend_data in score_trends.items():
                # Show trend for this specific agent or all agents for orchestrator
                if agent_name and trend_agent != agent_name:
                    continue
                delta = trend_data['delta']
                direction = trend_data['direction']
                current = trend_data['current']
                if direction == 'improving':
                    trend_lines.append(
                        f"{trend_agent}: {current:.2f} (+{delta:.2f}) improving — keep pushing"
                    )
                else:
                    trend_lines.append(
                        f"{trend_agent}: {current:.2f} ({delta:.2f}) declining — investigate and fix"
                    )
            if not agent_name:
                # Orchestrator sees all trends
                for trend_agent, trend_data in score_trends.items():
                    delta = trend_data['delta']
                    current = trend_data['current']
                    direction = trend_data['direction']
                    if direction == 'improving':
                        trend_lines.append(
                            f"{trend_agent}: {current:.2f} (+{delta:.2f}) improving"
                        )
                    else:
                        trend_lines.append(
                            f"{trend_agent}: {current:.2f} ({delta:.2f}) DECLINING"
                        )
            if trend_lines:
                lines.append("Score trends: " + "; ".join(trend_lines))

        return "\n".join(lines) if len(lines) > 1 else ""

    @classmethod
    def test_learning_pathways(cls) -> Dict[str, Dict[str, Any]]:
        """Diagnostic: inject synthetic data to verify all 5 learning pathways produce prompt text."""
        import tempfile
        results = {}

        # Create concrete subclass to bypass ABC restriction
        class _TestSwarm(cls):
            async def execute(self, *args, **kwargs):
                pass

        # Create minimal swarm instance for testing (no disk I/O beyond tempdir)
        config = SwarmConfig(name='pathway_tester', enable_self_improvement=True)
        instance = _TestSwarm.__new__(_TestSwarm)
        instance.config = config
        instance._swarm_intelligence = None
        instance._memory = None
        instance._learned_context = None
        tmp = tempfile.mkdtemp()
        instance._evaluation_history = EvaluationHistory(path=tmp + '/eval')
        instance._improvement_history = ImprovementHistory(path=tmp + '/imp')

        # === Pathway 1: weak_tools ===
        instance._learned_context = {
            'has_learning': True,
            'tool_performance': {'bad_tool': 0.3},
            'agent_scores': {},
            'weak_tools': [{'tool': 'bad_tool', 'success_rate': 0.3, 'total': 5}],
            'strong_tools': [{'tool': 'good_tool', 'success_rate': 0.95, 'total': 10}],
            'recommendations': [],
            'warmup_completed': True,
            'expert_knowledge': [],
            'prior_failures': [],
            'score_trends': {},
        }
        text = instance._build_learned_context_string()
        results['weak_tools'] = {
            'triggered': 'WEAK' in text,
            'prompt_snippet': text[:200] if text else '(empty)',
        }

        # === Pathway 2: expert_knowledge ===
        instance._learned_context['expert_knowledge'] = [
            {'learned_pattern': 'Always validate API responses before processing'},
            {'learned_pattern': 'Use batch processing for >100 items'},
        ]
        text = instance._build_learned_context_string()
        results['expert_knowledge'] = {
            'triggered': 'Expert Knowledge' in text,
            'prompt_snippet': text[text.find('Expert'):text.find('Expert') + 150] if 'Expert' in text else '(empty)',
        }

        # === Pathway 3: prior_failures ===
        instance._learned_context['prior_failures'] = [
            {'source': 'evaluation', 'score': 0.3, 'feedback': 'Missing key concepts', 'timestamp': datetime.now().isoformat()},
            {'source': 'collective_memory', 'agent': 'ConceptExtractor', 'task_type': 'expert', 'timestamp': datetime.now().isoformat()},
        ]
        text = instance._build_learned_context_string()
        results['prior_failures'] = {
            'triggered': 'Prior Failures' in text,
            'prompt_snippet': text[text.find('Prior Failures'):text.find('Prior Failures') + 200] if 'Prior Failures' in text else '(empty)',
        }

        # === Pathway 4: improvement_directives ===
        # ImprovementHistory uses self.history (list of dicts)
        # get_pending_suggestions() checks status == 'pending'
        # get_successful_improvements() checks outcome == 'success'
        instance._improvement_history.history = [
            {
                'id': 'test_pending_1',
                'suggestion': {'description': 'Improve concept extraction depth', 'priority': 5, 'agent_role': ''},
                'status': 'pending',
                'outcome': None,
            },
            {
                'id': 'test_success_1',
                'suggestion': {'description': 'Use more examples in explanations', 'priority': 3, 'agent_role': ''},
                'status': 'completed',
                'outcome': 'success',
            },
        ]
        text = instance._build_learned_context_string()
        results['improvement_directives'] = {
            'triggered': 'Improvement Directives' in text,
            'prompt_snippet': text[text.find('Improvement'):text.find('Improvement') + 200] if 'Improvement' in text else '(empty)',
        }

        # === Pathway 5: recommendations ===
        instance._learned_context['recommendations'] = [
            {'priority': 'HIGH', 'type': 'tool_and_agent', 'tool': 'bad_tool', 'tool_rate': 0.3,
             'agent': 'SlowAgent', 'consistency': 0.3,
             'action': 'PRIORITY: Replace bad_tool (30% success) AND stabilize SlowAgent (consistency=0.30)'}
        ]
        text = instance._build_learned_context_string()
        results['recommendations'] = {
            'triggered': 'Action:' in text and 'PRIORITY' in text,
            'prompt_snippet': text[text.find('Action:'):text.find('Action:') + 150] if 'Action:' in text else '(empty)',
        }

        # === Pathway 6: new_agent_no_misleading_rcs ===
        instance._learned_context['agent_scores'] = {
            'BrandNewAgent': {
                'rcs': 0.5, 'rds': 0.5, 'tras': 0.5,
                'consistency': 0.5, 'focus': 0.5, 'specialization': 0.5,
                'total_tasks': 0,
            }
        }
        text = instance._build_learned_context_string(agent_name='BrandNewAgent')
        results['new_agent_no_misleading_rcs'] = {
            'triggered': 'needs improvement' not in text,
            'prompt_snippet': text[:200] if text else '(no misleading feedback — correct)',
        }

        # Summary
        all_passed = all(r['triggered'] for r in results.values())
        results['_summary'] = {
            'total': 6,
            'passed': sum(1 for r in results.values() if isinstance(r, dict) and r.get('triggered')),
            'all_passed': all_passed,
        }

        return results

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

    def _retrieve_expert_knowledge(self) -> List[Dict[str, Any]]:
        """
        Retrieve expert-learned domain patterns from HierarchicalMemory.

        Queries HierarchicalMemory for improvements stored by BaseExpert agents
        (via memory_integration.py). Returns patterns relevant to this swarm's
        domain so they can be injected into DSPy agent prompts.

        Returns:
            List of expert improvement dicts with 'learned_pattern', 'domain',
            'source', etc. Empty list if memory unavailable.
        """
        if not self._memory:
            # Try initializing shared resources to get memory
            if not self._initialized:
                self._init_shared_resources()
            if not self._memory:
                return []

        domain = getattr(self.config, 'domain', None) or self.config.name or 'general'
        swarm_name = self.config.name or 'base_swarm'

        try:
            from ..foundation.data_structures import MemoryLevel

            # Primary: domain-scoped retrieval (key-prefix filtering)
            memory_entries = self._memory.retrieve_by_domain(
                domain=domain,
                goal=f"expert_{domain}_improvements",
                budget_tokens=5000,
                levels=[MemoryLevel.PROCEDURAL, MemoryLevel.META, MemoryLevel.SEMANTIC]
            )

            if not memory_entries:
                # Fallback: context-aware retrieval prioritizing wisdom (META > SEMANTIC)
                memory_entries = self._memory.retrieve_for_context(
                    query=f"expert improvements for {swarm_name}",
                    goal=f"expert_{domain}_improvements",
                    context_type="planning",
                    budget_tokens=3000,
                )

            improvements = []
            for entry in memory_entries[:10]:  # Cap at 10 patterns
                try:
                    improvement_data = json.loads(entry.content)
                    if isinstance(improvement_data, dict):
                        improvements.append(improvement_data)
                    elif isinstance(improvement_data, list):
                        improvements.extend(improvement_data[:5])
                except (json.JSONDecodeError, TypeError):
                    # Raw text pattern from consolidation
                    if entry.content and len(entry.content) > 10:
                        improvements.append({
                            'learned_pattern': entry.content,
                            'domain': domain,
                            'source': 'expert_memory',
                            'memory_level': entry.level.value if hasattr(entry, 'level') else 'unknown',
                        })

            if improvements:
                logger.info(f"Retrieved {len(improvements)} expert patterns from memory for domain '{domain}'")

            return improvements

        except Exception as e:
            logger.debug(f"Expert knowledge retrieval skipped: {e}")
            return []

    def _analyze_prior_failures(self) -> List[Dict[str, Any]]:
        """Analyze prior execution failures from collective_memory and evaluation history.
        Returns list of failure patterns with avoidance suggestions."""
        failures = []

        # Source 1: Evaluation history failures
        if hasattr(self, '_evaluation_history'):
            eval_failures = self._evaluation_history.get_failures(20)
            for f in eval_failures[-5:]:  # Last 5 failures
                failures.append({
                    'source': 'evaluation',
                    'score': f.get('overall_score', 0),
                    'feedback': f.get('feedback', ''),
                    'timestamp': f.get('timestamp', ''),
                })

        # Source 2: Collective memory from SwarmIntelligence
        si = self._swarm_intelligence
        if si and si.collective_memory:
            failed_tasks = [
                m for m in si.collective_memory[-50:]
                if not m.get('success', True)
            ]
            for m in failed_tasks[-5:]:
                failures.append({
                    'source': 'collective_memory',
                    'agent': m.get('agent', 'unknown'),
                    'task_type': m.get('task_type', 'unknown'),
                    'timestamp': m.get('timestamp', ''),
                })

        # Source 3: Execution traces stored in memory
        if self._memory:
            try:
                from ..foundation.data_structures import MemoryLevel
                failure_entries = self._memory.retrieve(
                    query=f"failed execution error {self.config.name or 'swarm'}",
                    goal="failure_analysis",
                    budget_tokens=2000,
                    levels=[MemoryLevel.META]
                )
                for entry in failure_entries[:3]:
                    try:
                        data = json.loads(entry.content)
                        if isinstance(data, dict) and not data.get('success', True):
                            failures.append({
                                'source': 'memory',
                                'pattern': data.get('learned_pattern', entry.content[:100]),
                            })
                    except (json.JSONDecodeError, TypeError):
                        pass
            except Exception:
                pass

        return failures

    def _store_execution_as_improvement(
        self,
        success: bool,
        execution_time: float,
        tools_used: List[str],
        task_type: str
    ):
        """
        Store execution outcome as an expert improvement in HierarchicalMemory.

        This bridges swarm execution results back into the expert memory system,
        so future expert training and swarm executions can learn from outcomes.

        Args:
            success: Whether execution succeeded
            execution_time: Total execution time in seconds
            tools_used: List of tool names used
            task_type: Type of task executed
        """
        if not self._memory:
            return

        domain = getattr(self.config, 'domain', None) or self.config.name or 'general'
        swarm_name = self.config.name or 'base_swarm'

        try:
            from ..foundation.data_structures import MemoryLevel

            # Build improvement from execution outcome
            if success:
                pattern = (
                    f"Successful {task_type} execution by {swarm_name}: "
                    f"tools [{', '.join(tools_used)}] completed in {execution_time:.1f}s"
                )
                level = MemoryLevel.PROCEDURAL
            else:
                pattern = (
                    f"Failed {task_type} execution by {swarm_name}: "
                    f"tools [{', '.join(tools_used)}] failed after {execution_time:.1f}s — "
                    f"consider alternative approach or tool substitution"
                )
                level = MemoryLevel.META  # Failures are learning wisdom

            improvement = {
                'timestamp': datetime.now().isoformat(),
                'task': task_type,
                'learned_pattern': pattern,
                'improvement_type': 'execution_outcome',
                'source': f'swarm_{swarm_name}',
                'success': success,
                'execution_time': execution_time,
                'tools_used': tools_used,
            }

            context = {
                'expert_name': swarm_name,
                'domain': domain,
                'task': task_type,
                'improvement_type': 'execution_outcome',
                'source': 'swarm_lifecycle',
            }

            self._memory.store(
                content=json.dumps(improvement, ensure_ascii=False),
                level=level,
                context=context,
                goal=f"expert_{domain}_improvements",
                initial_value=0.8 if success else 1.0,  # Failures are more valuable for learning
            )

            logger.debug(f"Stored execution outcome to expert memory: {task_type} {'success' if success else 'failure'}")

        except Exception as e:
            logger.debug(f"Failed to store execution improvement: {e}")

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

    async def _post_execute_learning(
        self,
        success: bool,
        execution_time: float,
        tools_used: List[str],
        task_type: str,
        output_data: Dict[str, Any] = None,
        input_data: Dict[str, Any] = None
    ):
        """
        Post-execution learning hook. Called at end of execute().

        Sends executor feedback, recomputes MorphAgent scores, re-analyzes
        tools, evaluates output, runs improvement cycle, and saves all state.

        Args:
            success: Whether execution succeeded
            execution_time: Total execution time in seconds
            tools_used: List of tool names used during execution
            task_type: Type of task that was executed
            output_data: Optional dict of output metrics for evaluation
            input_data: Optional dict of input params for evaluation matching
        """
        try:
            # 1. Send executor feedback (tools, success, timing)
            self._send_executor_feedback(
                task_type=task_type,
                success=success,
                tools_used=tools_used,
                execution_time=execution_time,
                error_type=None if success else 'execution_failure'
            )

            si = self._swarm_intelligence
            if not si:
                return

            # 2. Recompute MorphAgent scores with new data
            if si.agent_profiles:
                morph_scores = si.morph_scorer.compute_all_scores(si.agent_profiles)
                si.morph_score_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'scores': {
                        name: {'rcs': s.rcs, 'rds': s.rds, 'tras': s.tras}
                        for name, s in morph_scores.items()
                    }
                })
                # Bound history
                if len(si.morph_score_history) > 50:
                    si.morph_score_history = si.morph_score_history[-50:]

            # 3. Re-analyze tools and update assignments
            self._manage_tools()

            # 4. Evaluate output against gold standard (centralized for all swarms)
            evaluation = None
            if success and output_data and self.config.enable_self_improvement:
                try:
                    evaluation = await self._evaluate_output(
                        output=output_data,
                        task_type=task_type,
                        input_data=input_data or {}
                    )
                    if evaluation:
                        logger.info(
                            f"Evaluation: {evaluation.result.value} "
                            f"(score: {evaluation.overall_score:.2f})"
                        )
                except Exception as eval_err:
                    logger.debug(f"Evaluation skipped: {eval_err}")

            # 4a. Audit evaluation quality (non-blocking)
            if evaluation and self._auditor and output_data:
                try:
                    audit_result = await self._auditor.audit_evaluation(
                        evaluation={'scores': evaluation.scores,
                                    'overall_score': evaluation.overall_score,
                                    'result': evaluation.result.value,
                                    'feedback': evaluation.feedback},
                        output_data=output_data,
                        context=json.dumps({'task_type': task_type})
                    )
                    if not audit_result.get('passed', True):
                        logger.warning(
                            f"Audit failed for evaluation: {audit_result.get('reasoning', 'unknown')}"
                        )
                except Exception:
                    pass  # Non-blocking

            # 4b. Record iteration in benchmarks (always, not just when evaluation exists)
            if si:
                try:
                    score = evaluation.overall_score if evaluation else (1.0 if success else 0.0)
                    si.benchmarks.record_iteration(
                        iteration_id=f"{task_type}_{int(__import__('time').time())}",
                        task_type=task_type,
                        score=score,
                        execution_time=execution_time,
                        success=success
                    )
                except Exception:
                    pass

            # 4c. Auto-curate gold standard from excellent outputs
            if (evaluation and evaluation.overall_score >= 0.9 and
                evaluation.result in (EvaluationResult.EXCELLENT, EvaluationResult.GOOD) and
                self._gold_db and output_data and input_data):
                try:
                    self._curate_gold_standard(task_type, input_data, output_data, evaluation)
                except Exception:
                    pass

            # 4d. Extract learnings from excellent executions
            if (evaluation and evaluation.overall_score >= 0.9 and
                self._learner and output_data and input_data):
                try:
                    learnings = await self._learner.extract_learnings(
                        input_data=input_data,
                        output_data=output_data,
                        evaluation={'scores': evaluation.scores,
                                    'overall_score': evaluation.overall_score,
                                    'feedback': evaluation.feedback},
                        domain=self.config.domain
                    )
                    if learnings and self._improvement_history:
                        now = datetime.now().isoformat()
                        for learning in learnings:
                            suggestion = ImprovementSuggestion(
                                agent_role=AgentRole.ACTOR,
                                improvement_type=ImprovementType.TRAINING_DATA,
                                description=learning,
                                priority=3,
                                expected_impact=0.5,
                                implementation_details={'source': 'learner_extraction'},
                                based_on_evaluations=[evaluation.gold_standard_id]
                            )
                            sid = hashlib.md5(
                                f"{suggestion.agent_role.value}:{suggestion.description}:{now}".encode()
                            ).hexdigest()[:12]
                            self._improvement_history.history.append({
                                'id': sid,
                                'suggestion': asdict(suggestion),
                                'status': 'completed',
                                'created_at': now,
                                'applied_at': now,
                                'outcome': 'success',
                                'impact_measured': 0.5,
                                'notes': 'Auto-extracted from excellent execution',
                            })
                        self._improvement_history._save_history()
                        logger.info(f"Extracted {len(learnings)} learnings from excellent execution")
                except Exception:
                    pass

            # 5. Run improvement cycle if evaluation below threshold
            if evaluation and evaluation.overall_score < self.config.improvement_threshold:
                try:
                    suggestions = await self._run_improvement_cycle()
                    if suggestions:
                        logger.info(f"Generated {len(suggestions)} improvement suggestions")
                except Exception as imp_err:
                    logger.debug(f"Improvement cycle skipped: {imp_err}")

            # 6. Save state to disk
            try:
                save_path = self._get_intelligence_save_path()
                si.save(save_path)
                logger.debug(f"Post-execution learning state saved to {save_path}")
            except Exception as save_err:
                logger.debug(f"Failed to save post-execution state: {save_err}")

            # 7. Store execution outcome as expert improvement in HierarchicalMemory
            self._store_execution_as_improvement(
                success=success,
                execution_time=execution_time,
                tools_used=tools_used,
                task_type=task_type
            )

        except Exception as e:
            logger.debug(f"Post-execution learning skipped: {e}")

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

    async def _run_improvement_cycle(self) -> List[ImprovementSuggestion]:
        """Run the self-improvement cycle."""
        if not self.config.enable_self_improvement or not self._reviewer:
            return []

        # Check if improvement is needed (persistent across sessions)
        recent_evals = self._evaluation_history.get_recent(10)
        avg_score = self._evaluation_history.get_average_score(10)
        if not recent_evals or avg_score >= self.config.improvement_threshold:
            logger.info(f"Performance good ({avg_score:.2f}), skipping improvement cycle")
            return []

        # Get suggestions from reviewer
        agent_configs = {
            AgentRole.EXPERT: self._expert.config if self._expert else None,
            AgentRole.REVIEWER: self._reviewer.config if self._reviewer else None,
            AgentRole.PLANNER: self._planner.config if self._planner else None,
            AgentRole.ACTOR: self._actor.config if self._actor else None,
            AgentRole.AUDITOR: self._auditor.config if self._auditor else None,
            AgentRole.LEARNER: self._learner.config if self._learner else None,
        }
        agent_configs = {k: v for k, v in agent_configs.items() if v}

        suggestions = await self._reviewer.analyze_and_suggest(
            recent_evals,
            agent_configs
        )

        # Record suggestions
        for suggestion in suggestions:
            self._improvement_history.record_suggestion(suggestion)

        return suggestions

    def _record_trace(
        self,
        agent_name: str,
        agent_role: AgentRole,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        execution_time: float,
        success: bool,
        error: Optional[str] = None,
        tools_used: List[str] = None
    ):
        """Record execution trace for learning and Agent0 feedback."""
        trace = ExecutionTrace(
            agent_name=agent_name,
            agent_role=agent_role,
            input_data=input_data,
            output_data=output_data,
            execution_time=execution_time,
            success=success,
            error=error
        )
        self._traces.append(trace)

        # Fire TUI trace callback if active
        try:
            from .coding_swarm import _active_trace_callback
            if _active_trace_callback is not None:
                _active_trace_callback({
                    "agent": agent_name,
                    "role": agent_role.value if agent_role else "",
                    "time": execution_time,
                    "success": success,
                    "error": error,
                    "output_summary": str(output_data)[:100] if output_data else "",
                })
        except Exception:
            pass

        # Agent0: Per-phase swarm-level feedback removed — swarm-level recording
        # is handled once by _post_execute_learning() at end of execute().
        # Only per-agent recording happens here (below).

        # MorphAgent: Update agent profile for per-agent tracking
        swarm_name = self.config.name or 'base_swarm'
        if self._swarm_intelligence and hasattr(self._swarm_intelligence, 'agent_profiles'):
            self._swarm_intelligence.register_agent(agent_name)
            # Record task result under individual agent name (not swarm name)
            # so per-agent profiles accumulate real task_success data
            if agent_name != swarm_name:
                task_type_label = agent_role.value if agent_role else 'unknown'
                self._swarm_intelligence.record_task_result(
                    agent_name=agent_name,
                    task_type=task_type_label,
                    success=success,
                    execution_time=execution_time
                )

        # Store in memory for learning
        if self._memory and self.config.enable_learning:
            try:
                from ..foundation.data_structures import MemoryLevel
                self._memory.store(
                    content=json.dumps(asdict(trace), default=str),
                    level=MemoryLevel.EPISODIC,
                    context={'swarm': self.config.name, 'agent': agent_name},
                    goal=f"Execution trace: {agent_name}"
                )
            except Exception as e:
                logger.debug(f"Failed to store trace in memory: {e}")

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

    def record_improvement_outcome(
        self,
        suggestion_id: str,
        success: bool,
        impact: float,
        notes: str = ""
    ):
        """Record the outcome of an applied improvement."""
        if self._improvement_history:
            self._improvement_history.record_outcome(suggestion_id, success, impact, notes)



# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    'AgentRole',
    'EvaluationResult',
    'ImprovementType',

    # Data classes
    'GoldStandard',
    'Evaluation',
    'ImprovementSuggestion',
    'AgentConfig',
    'ExecutionTrace',
    'SwarmConfig',
    'SwarmResult',

    # DSPy Signatures
    'ExpertEvaluationSignature',
    'ReviewerAnalysisSignature',
    'PlannerOptimizationSignature',
    'ActorExecutionSignature',
    'AuditorVerificationSignature',
    'LearnerExtractionSignature',

    # Core classes
    'GoldStandardDB',
    'ImprovementHistory',
    'ExpertAgent',
    'ReviewerAgent',
    'PlannerAgent',
    'ActorAgent',
    'AuditorAgent',
    'LearnerAgent',
    'BaseSwarm',
    'SwarmRegistry',
    'register_swarm',
]
