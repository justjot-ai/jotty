"""
Knowledge mixin for BaseSwarm.

Contains knowledge retrieval and storage methods: expert knowledge retrieval
from SwarmMemory, prior failure analysis, execution outcome storage, learned
context string building, and learning pathway diagnostics.

Extracted from _learning_mixin.py to separate knowledge concerns from
coordination and core learning lifecycle concerns.
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List

from Jotty.core.infrastructure.foundation.exceptions import (
    MemoryError,
    MemoryRetrievalError,
    MemoryStorageError,
)


logger = logging.getLogger(__name__)


class SwarmKnowledgeMixin:
    """Mixin providing knowledge retrieval/storage infrastructure for BaseSwarm.

    Methods in this mixin handle knowledge management:
    - Expert knowledge retrieval from SwarmMemory
    - Prior failure analysis from evaluation history and traces
    - Execution outcome storage as expert improvements
    - Learned context string building for agent prompt injection
    - Learning pathway diagnostics

    Expects to be mixed into BaseSwarm which provides:
    - self._memory (SwarmMemory instance)
    - self._initialized, self._init_shared_resources()
    - self._swarm_intelligence (SwarmIntelligence instance)
    - self._learned_context dict
    - self._evaluation_history, self._improvement_history
    - self.config with .name, .domain, .enable_self_improvement, .enable_learning
    """

    def _retrieve_expert_knowledge(self) -> List[Dict[str, Any]]:
        """
        Retrieve expert-learned domain patterns from SwarmMemory.

        Queries SwarmMemory for improvements stored by BaseExpert agents
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

        domain = getattr(self.config, "domain", None) or self.config.name or "general"
        swarm_name = self.config.name or "base_swarm"

        try:
            from Jotty.core.infrastructure.foundation.data_structures import MemoryLevel

            # Primary: domain-scoped retrieval (key-prefix filtering)
            memory_entries = self._memory.retrieve_by_domain(
                domain=domain,
                goal=f"expert_{domain}_improvements",
                budget_tokens=5000,
                levels=[MemoryLevel.PROCEDURAL, MemoryLevel.META, MemoryLevel.SEMANTIC],
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
                        improvements.append(
                            {
                                "learned_pattern": entry.content,
                                "domain": domain,
                                "source": "expert_memory",
                                "memory_level": (
                                    entry.level.value if hasattr(entry, "level") else "unknown"
                                ),
                            }
                        )

            if improvements:
                logger.info(
                    f"Retrieved {len(improvements)} expert patterns from memory for domain '{domain}'"
                )

            return improvements

        except (MemoryRetrievalError, MemoryError) as e:
            logger.warning(f"Expert knowledge retrieval failed (memory): {e}")
            return []
        except Exception as e:
            logger.debug(f"Expert knowledge retrieval skipped (unexpected): {e}")
            return []

    def _analyze_prior_failures(self) -> List[Dict[str, Any]]:
        """Analyze prior execution failures from collective_memory and evaluation history.
        Returns list of failure patterns with avoidance suggestions."""
        failures = []

        # Source 1: Evaluation history failures
        if hasattr(self, "_evaluation_history"):
            eval_failures = self._evaluation_history.get_failures(20)
            for f in eval_failures[-5:]:  # Last 5 failures
                failures.append(
                    {
                        "source": "evaluation",
                        "score": f.get("overall_score", 0),
                        "feedback": f.get("feedback", ""),
                        "timestamp": f.get("timestamp", ""),
                    }
                )

        # Source 2: Collective memory from SwarmIntelligence
        si = self._swarm_intelligence
        if si and si.collective_memory:
            failed_tasks = [
                m for m in list(si.collective_memory)[-50:] if not m.get("success", True)
            ]
            for m in failed_tasks[-5:]:
                failures.append(
                    {
                        "source": "collective_memory",
                        "agent": m.get("agent", "unknown"),
                        "task_type": m.get("task_type", "unknown"),
                        "timestamp": m.get("timestamp", ""),
                    }
                )

        # Source 3: Execution traces stored in memory
        if self._memory:
            try:
                from Jotty.core.infrastructure.foundation.data_structures import MemoryLevel

                failure_entries = self._memory.retrieve(
                    query=f"failed execution error {self.config.name or 'swarm'}",
                    goal="failure_analysis",
                    budget_tokens=2000,
                    levels=[MemoryLevel.META],
                )
                for entry in failure_entries[:3]:
                    try:
                        data = json.loads(entry.content)
                        if isinstance(data, dict) and not data.get("success", True):
                            failures.append(
                                {
                                    "source": "memory",
                                    "pattern": data.get("learned_pattern", entry.content[:100]),
                                }
                            )
                    except (json.JSONDecodeError, TypeError):
                        pass
            except (MemoryRetrievalError, MemoryError) as e:
                logger.warning(f"Failure analysis from memory failed (memory): {e}")
            except Exception as e:
                logger.debug(f"Failure analysis from memory failed (unexpected): {e}")

        return failures

    def _store_execution_as_improvement(
        self, success: bool, execution_time: float, tools_used: List[str], task_type: str
    ) -> Any:
        """
        Store execution outcome as an expert improvement in SwarmMemory.

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

        domain = getattr(self.config, "domain", None) or self.config.name or "general"
        swarm_name = self.config.name or "base_swarm"

        try:
            from Jotty.core.infrastructure.foundation.data_structures import MemoryLevel

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
                "timestamp": datetime.now().isoformat(),
                "task": task_type,
                "learned_pattern": pattern,
                "improvement_type": "execution_outcome",
                "source": f"swarm_{swarm_name}",
                "success": success,
                "execution_time": execution_time,
                "tools_used": tools_used,
            }

            context = {
                "expert_name": swarm_name,
                "domain": domain,
                "task": task_type,
                "improvement_type": "execution_outcome",
                "source": "swarm_lifecycle",
            }

            self._memory.store(
                content=json.dumps(improvement, ensure_ascii=False),
                level=level,
                context=context,
                goal=f"expert_{domain}_improvements",
                initial_value=0.8 if success else 1.0,  # Failures are more valuable for learning
            )

            logger.debug(
                f"Stored execution outcome to expert memory: {task_type} {'success' if success else 'failure'}"
            )

        except (MemoryStorageError, MemoryError) as e:
            logger.warning(f"Failed to store execution improvement (memory): {e}")
        except Exception as e:
            logger.debug(f"Failed to store execution improvement (unexpected): {e}")

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
        if not self._learned_context or not self._learned_context.get("has_learning"):
            return ""

        ctx = self._learned_context
        lines = ["## Prior Learning"]

        # Tool performance summary
        tool_parts = []
        for tool_info in ctx.get("strong_tools", []):
            tool_parts.append(
                f"{tool_info.get('tool', '?')} {tool_info.get('success_rate', 0):.0%} RELIABLE"
            )
        for tool_info in ctx.get("weak_tools", []):
            tool_parts.append(
                f"{tool_info.get('tool', '?')} {tool_info.get('success_rate', 0):.0%} WEAK"
            )

        if tool_parts:
            lines.append(f"Tool Performance: {', '.join(tool_parts)}")

        # Agent-specific context (both positive reinforcement and warnings)
        agent_notes = []
        scores = ctx.get("agent_scores", {})
        if agent_name and agent_name in scores:
            agent_data = scores[agent_name]
            rcs = agent_data.get("rcs", 0)
            consistency = agent_data.get("consistency", 0.5)
            focus = agent_data.get("focus", 0.5)
            total_tasks = agent_data.get("total_tasks", 0)
            # Competence and focus feedback require enough history to be meaningful
            if total_tasks >= 2:
                # Tiered competence feedback — always push for higher
                if rcs >= 0.85:
                    agent_notes.append(f"Competence {rcs:.2f} — excellent, maintain this standard")
                elif rcs >= 0.6:
                    agent_notes.append(
                        f"Competence {rcs:.2f} — good but target >0.85, push harder on quality"
                    )
                elif rcs >= 0.4:
                    agent_notes.append(f"Competence {rcs:.2f} — needs improvement, aim for >0.6")
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
                rcs = agent_data.get("rcs", 0)
                consistency = agent_data.get("consistency", 0.5)
                if rcs >= 0.85:
                    high_performers.append(name)
                elif rcs < 0.5 and rcs > 0:
                    needs_improvement.append(f"{name}({rcs:.2f})")
                if consistency < 0.5:
                    low_performers.append(f"{name} inconsistent ({consistency:.2f})")
            if high_performers:
                agent_notes.append(f"Strong agents: {', '.join(high_performers)}")
            if needs_improvement:
                agent_notes.append(f"Need improvement: {', '.join(needs_improvement)}")
            if low_performers:
                agent_notes.extend(low_performers)

        # Specialization label from AgentProfile
        si = self._swarm_intelligence
        if si and agent_name and agent_name in getattr(si, "agent_profiles", {}):
            from Jotty.core.intelligence.orchestration.swarm_intelligence import AgentSpecialization

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
            recent = list(si.collective_memory)[-20:]
            successes = [m for m in recent if m.get("success")]
            if successes:
                # Build timing expectations per task type
                from collections import defaultdict

                task_times = defaultdict(list)
                for m in successes:
                    tt = m.get("task_type", "")
                    if tt and m.get("execution_time", 0) > 0:
                        task_times[tt].append(m["execution_time"])
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

        # Coordination protocol status (circuit breakers, backpressure, etc.)
        coord = ctx.get("coordination", {})
        if coord:
            coord_parts = []
            blocked = coord.get("circuit_blocked", [])
            if blocked:
                coord_parts.append(f"Circuit-blocked agents: {', '.join(blocked)}")
            backpressure = coord.get("backpressure", 0)
            if backpressure > 0.5:
                coord_parts.append(
                    f"Backpressure: {backpressure:.0%} (high — prioritize efficiency)"
                )
            load_balanced = coord.get("load_balanced", 0)
            if load_balanced > 0:
                coord_parts.append(f"{load_balanced} tasks rebalanced via work-stealing")
            if coord_parts:
                lines.append(f"Swarm Status: {'; '.join(coord_parts)}")

        # Priority recommendations (HIGH first)
        recommendations = ctx.get("recommendations", [])
        high_priority = [r for r in recommendations if r.get("priority") == "HIGH"]
        if high_priority:
            actions = [r["action"] for r in high_priority[:3]]
            lines.append(f"Action: {'; '.join(actions)}")
        elif recommendations:
            lines.append(f"Action: {recommendations[0]['action']}")

        # Evaluation quality bar from persistent history
        if hasattr(self, "_evaluation_history"):
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
        if hasattr(self, "_improvement_history") and self._improvement_history:
            pending = self._improvement_history.get_pending_suggestions()
            successful = self._improvement_history.get_successful_improvements()
            if pending or successful:
                imp_lines = []
                # Show successful improvements so agents know what works
                for s in successful[-3:]:
                    suggestion = s.get("suggestion", {})
                    desc = suggestion.get("description", "")
                    if desc:
                        imp_lines.append(f"- Applied successfully: {desc[:120]}")
                # Show pending improvements as directives
                for p in pending[-3:]:
                    suggestion = p.get("suggestion", {})
                    desc = suggestion.get("description", "")
                    priority = suggestion.get("priority", "MEDIUM")
                    target_agent = suggestion.get("agent_role", "")
                    # Only show agent-specific improvements to that agent
                    if agent_name and target_agent and target_agent != agent_name:
                        continue
                    if desc:
                        imp_lines.append(f"- [{priority}] TASK: {desc[:120]}")
                if imp_lines:
                    lines.append("## Improvement Directives")
                    lines.extend(imp_lines)

        # Expert domain knowledge from SwarmMemory
        expert_knowledge = ctx.get("expert_knowledge", [])
        if expert_knowledge:
            expert_lines = []
            for imp in expert_knowledge[:5]:  # Top 5 patterns
                pattern = imp.get("learned_pattern", "")
                if pattern:
                    # Truncate long patterns for prompt efficiency
                    if len(pattern) > 150:
                        pattern = pattern[:147] + "..."
                    expert_lines.append(f"- {pattern}")

            if expert_lines:
                lines.append("## Expert Knowledge")
                lines.extend(expert_lines)

        # Failure recovery from prior runs
        prior_failures = ctx.get("prior_failures", [])
        if prior_failures:
            failure_lines = ["## Prior Failures (Avoid Repeating)"]
            for f in prior_failures[:3]:
                if f.get("source") == "evaluation":
                    feedback = f.get("feedback", "")
                    if feedback:
                        failure_lines.append(
                            f"- Previous run scored {f.get('score', 0):.0%}: {feedback[:100]}"
                        )
                elif f.get("source") == "collective_memory":
                    failure_lines.append(
                        f"- Agent {f.get('agent', '?')} failed task {f.get('task_type', '?')}"
                    )
                elif f.get("source") == "memory":
                    failure_lines.append(f"- {f.get('pattern', 'unknown failure')}")
            if len(failure_lines) > 1:
                lines.extend(failure_lines)

        # Morph score trends — show improvement/decline direction
        score_trends = ctx.get("score_trends", {})
        if score_trends:
            trend_lines = []
            for trend_agent, trend_data in score_trends.items():
                # Show trend for this specific agent or all agents for orchestrator
                if agent_name and trend_agent != agent_name:
                    continue
                delta = trend_data["delta"]
                direction = trend_data["direction"]
                current = trend_data["current"]
                if direction == "improving":
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
                    delta = trend_data["delta"]
                    current = trend_data["current"]
                    direction = trend_data["direction"]
                    if direction == "improving":
                        trend_lines.append(f"{trend_agent}: {current:.2f} (+{delta:.2f}) improving")
                    else:
                        trend_lines.append(f"{trend_agent}: {current:.2f} ({delta:.2f}) DECLINING")
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
            async def execute(self, *args: Any, **kwargs: Any) -> Any:
                pass

        # Create minimal swarm instance for testing (no disk I/O beyond tempdir)
        config = SwarmConfig(name="pathway_tester", enable_self_improvement=True)
        instance = _TestSwarm.__new__(_TestSwarm)
        instance.config = config
        instance._swarm_intelligence = None
        instance._memory = None
        instance._learned_context = None
        tmp = tempfile.mkdtemp()
        instance._evaluation_history = EvaluationHistory(path=tmp + "/eval")
        instance._improvement_history = ImprovementHistory(path=tmp + "/imp")

        # === Pathway 1: weak_tools ===
        instance._learned_context = {
            "has_learning": True,
            "tool_performance": {"bad_tool": 0.3},
            "agent_scores": {},
            "weak_tools": [{"tool": "bad_tool", "success_rate": 0.3, "total": 5}],
            "strong_tools": [{"tool": "good_tool", "success_rate": 0.95, "total": 10}],
            "recommendations": [],
            "warmup_completed": True,
            "expert_knowledge": [],
            "prior_failures": [],
            "score_trends": {},
        }
        text = instance._build_learned_context_string()
        results["weak_tools"] = {
            "triggered": "WEAK" in text,
            "prompt_snippet": text[:200] if text else "(empty)",
        }

        # === Pathway 2: expert_knowledge ===
        instance._learned_context["expert_knowledge"] = [
            {"learned_pattern": "Always validate API responses before processing"},
            {"learned_pattern": "Use batch processing for >100 items"},
        ]
        text = instance._build_learned_context_string()
        results["expert_knowledge"] = {
            "triggered": "Expert Knowledge" in text,
            "prompt_snippet": (
                text[text.find("Expert") : text.find("Expert") + 150]
                if "Expert" in text
                else "(empty)"
            ),
        }

        # === Pathway 3: prior_failures ===
        instance._learned_context["prior_failures"] = [
            {
                "source": "evaluation",
                "score": 0.3,
                "feedback": "Missing key concepts",
                "timestamp": datetime.now().isoformat(),
            },
            {
                "source": "collective_memory",
                "agent": "ConceptExtractor",
                "task_type": "expert",
                "timestamp": datetime.now().isoformat(),
            },
        ]
        text = instance._build_learned_context_string()
        results["prior_failures"] = {
            "triggered": "Prior Failures" in text,
            "prompt_snippet": (
                text[text.find("Prior Failures") : text.find("Prior Failures") + 200]
                if "Prior Failures" in text
                else "(empty)"
            ),
        }

        # === Pathway 4: improvement_directives ===
        instance._improvement_history.history = [
            {
                "id": "test_pending_1",
                "suggestion": {
                    "description": "Improve concept extraction depth",
                    "priority": 5,
                    "agent_role": "",
                },
                "status": "pending",
                "outcome": None,
            },
            {
                "id": "test_success_1",
                "suggestion": {
                    "description": "Use more examples in explanations",
                    "priority": 3,
                    "agent_role": "",
                },
                "status": "completed",
                "outcome": "success",
            },
        ]
        text = instance._build_learned_context_string()
        results["improvement_directives"] = {
            "triggered": "Improvement Directives" in text,
            "prompt_snippet": (
                text[text.find("Improvement") : text.find("Improvement") + 200]
                if "Improvement" in text
                else "(empty)"
            ),
        }

        # === Pathway 5: recommendations ===
        instance._learned_context["recommendations"] = [
            {
                "priority": "HIGH",
                "type": "tool_and_agent",
                "tool": "bad_tool",
                "tool_rate": 0.3,
                "agent": "SlowAgent",
                "consistency": 0.3,
                "action": "PRIORITY: Replace bad_tool (30% success) AND stabilize SlowAgent (consistency=0.30)",
            }
        ]
        text = instance._build_learned_context_string()
        results["recommendations"] = {
            "triggered": "Action:" in text and "PRIORITY" in text,
            "prompt_snippet": (
                text[text.find("Action:") : text.find("Action:") + 150]
                if "Action:" in text
                else "(empty)"
            ),
        }

        # === Pathway 6: new_agent_no_misleading_rcs ===
        instance._learned_context["agent_scores"] = {
            "BrandNewAgent": {
                "rcs": 0.5,
                "rds": 0.5,
                "tras": 0.5,
                "consistency": 0.5,
                "focus": 0.5,
                "specialization": 0.5,
                "total_tasks": 0,
            }
        }
        text = instance._build_learned_context_string(agent_name="BrandNewAgent")
        results["new_agent_no_misleading_rcs"] = {
            "triggered": "needs improvement" not in text,
            "prompt_snippet": text[:200] if text else "(no misleading feedback — correct)",
        }

        # Summary
        all_passed = all(r["triggered"] for r in results.values())
        results["_summary"] = {
            "total": 6,
            "passed": sum(
                1 for r in results.values() if isinstance(r, dict) and r.get("triggered")
            ),
            "all_passed": all_passed,
        }

        return results
