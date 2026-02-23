"""
Execution Tracer — Comprehensive planning & execution document generator.

Captures every decision point during Orchestrator.run() and Orchestrator.chat():
  - Goal decomposition and domain classification
  - Swarm/agent routing decisions
  - Skill selection and UCB re-ranking
  - Agent-to-agent communication (step results, handoffs)
  - Learning signals (Q-table updates, reflections, patterns)
  - Quality scores and timing
  - Budget consumption

Produces a single markdown report post-execution.

Usage:
    from Jotty.core.infrastructure.monitoring.execution_tracer import ExecutionTracer

    tracer = ExecutionTracer()
    result = await orchestrator.run("complex task", status_callback=tracer.callback)
    report = tracer.generate_report(result)
    tracer.save_report(report, "reports/run_001.md")

Or use the convenience wrapper:
    from Jotty.core.infrastructure.monitoring.execution_tracer import traced_run, traced_chat

    result, report_path = await traced_run(orchestrator, "complex task")
    result, report_path = await traced_chat(orchestrator, "hello")
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class TraceEvent:
    """Single event in the execution trace."""

    timestamp: float
    elapsed: float  # seconds since trace start
    stage: str
    detail: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LearningSnapshot:
    """Snapshot of learning state before/after execution."""

    q_table: Dict[str, Dict[str, float]] = field(default_factory=dict)
    q_counts: Dict[str, Dict[str, int]] = field(default_factory=dict)
    episode_count: int = 0
    recent_reflections: List[str] = field(default_factory=list)
    recent_patterns: List[Dict[str, Any]] = field(default_factory=list)
    value_estimates: List[Dict[str, Any]] = field(default_factory=list)


class ExecutionTracer:
    """Captures execution events and generates comprehensive planning documents."""

    def __init__(self) -> None:
        self.trace_id = uuid.uuid4().hex[:12]
        self.start_time = time.time()
        self.events: List[TraceEvent] = []
        self.goal: str = ""
        self.mode: str = ""  # "run" or "chat"
        self._pre_snapshot: Optional[LearningSnapshot] = None
        self._post_snapshot: Optional[LearningSnapshot] = None
        self._user_callback: Optional[Callable] = None

    def callback(self, stage: str, detail: str, **metadata: Any) -> None:
        """Status callback compatible with Orchestrator's status_callback signature."""
        event = TraceEvent(
            timestamp=time.time(),
            elapsed=time.time() - self.start_time,
            stage=str(stage),
            detail=str(detail)[:2000],
            metadata=metadata,
        )
        self.events.append(event)

        # Forward to user's original callback if present
        if self._user_callback:
            try:
                self._user_callback(stage, detail)
            except Exception:
                pass

    def take_pre_snapshot(self) -> None:
        """Capture learning state before execution."""
        self._pre_snapshot = self._capture_learning_snapshot()

    def take_post_snapshot(self) -> None:
        """Capture learning state after execution."""
        self._post_snapshot = self._capture_learning_snapshot()

    def _capture_learning_snapshot(self) -> LearningSnapshot:
        snap = LearningSnapshot()
        try:
            from Jotty.core.intelligence.learning.learning_store import LearningStore

            store = LearningStore.get_instance()
            conn = store._get_conn()

            # Episode count
            row = conn.execute("SELECT COUNT(*) FROM episodes").fetchone()
            snap.episode_count = row[0] if row else 0

            # Q-table from TDLambdaLearner
            try:
                from Jotty.core.intelligence.learning.facade import get_td_lambda

                td = get_td_lambda()
                snap.q_table = dict(td.skill_q._q)
                snap.q_counts = dict(td.skill_q._counts)
            except Exception:
                pass

            # Recent reflections
            try:
                rows = conn.execute(
                    "SELECT unit_name, observation, adjustment FROM reflections "
                    "ORDER BY created_at DESC LIMIT 5"
                ).fetchall()
                snap.recent_reflections = [f"[{r[0]}] {r[1]} → {r[2]}" for r in rows]
            except Exception:
                pass

            # Recent patterns
            try:
                rows = conn.execute(
                    "SELECT description, recommendation, confidence, evidence_count "
                    "FROM patterns ORDER BY confidence DESC LIMIT 5"
                ).fetchall()
                snap.recent_patterns = [
                    {
                        "description": r[0],
                        "recommendation": r[1],
                        "confidence": r[2],
                        "evidence": r[3],
                    }
                    for r in rows
                ]
            except Exception:
                pass

            # Top value estimates
            try:
                rows = conn.execute(
                    "SELECT state_key, action_key, domain, value, update_count "
                    "FROM value_estimates WHERE state_key NOT LIKE '__%' "
                    "ORDER BY update_count DESC LIMIT 10"
                ).fetchall()
                snap.value_estimates = [
                    {
                        "state": r[0],
                        "action": r[1],
                        "domain": r[2],
                        "value": round(r[3], 4),
                        "updates": r[4],
                    }
                    for r in rows
                ]
            except Exception:
                pass

        except Exception as e:
            logger.debug(f"Learning snapshot failed: {e}")
        return snap

    def _capture_step_q_snapshot(self) -> Dict[str, Any]:
        """Capture StepQTable data (position + role + plans) for the report.

        Uses get_all_domains() to iterate all composite keys
        (e.g., "coding", "coding:finance", "research").
        """
        try:
            from Jotty.core.intelligence.learning.facade import get_td_lambda

            td = get_td_lambda()
            snapshot: Dict[str, Any] = {
                "step_stats": {},
                "best_plans": {},
                "role_guidance": {},
                "revision_count": td.step_q._revision_count,
            }
            # Iterate all domain keys (plain task_types + composite task_type:domain)
            for domain_key in td.step_q.get_all_domains():
                # Parse composite key
                task_type, domain = td.step_q._split_key(domain_key)

                step_stats = td.step_q.get_step_stats(task_type, domain=domain)
                if step_stats:
                    snapshot["step_stats"][domain_key] = step_stats

                best_plans = td.step_q.get_best_plan(task_type, top_n=3, domain=domain)
                if best_plans:
                    snapshot["best_plans"][domain_key] = [
                        {"roles": list(roles), "reward": round(r, 3)} for roles, r in best_plans
                    ]

                role_guidance = td.step_q.get_role_guidance(task_type, domain=domain)
                if role_guidance:
                    snapshot["role_guidance"][domain_key] = role_guidance

            # Also capture SkillQTable domains for the skill-level section
            snapshot["skill_domains"] = td.skill_q.get_all_domains()

            return snapshot
        except Exception:
            return {}

    def generate_report(
        self,
        result: Any = None,
        goal: str = "",
        mode: str = "run",
    ) -> str:
        """Generate a comprehensive markdown planning/execution report."""
        goal = goal or self.goal
        mode = mode or self.mode
        total_time = time.time() - self.start_time

        lines: List[str] = []
        _a = lines.append

        # ── HEADER ──
        _a("# Jotty Execution Report")
        _a("")
        _a("| Field | Value |")
        _a("|-------|-------|")
        _a(f"| **Trace ID** | `{self.trace_id}` |")
        _a(f"| **Mode** | `{mode}` |")
        _a(f"| **Timestamp** | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} |")
        _a(f"| **Total Time** | {total_time:.1f}s |")
        _a(f"| **Events Captured** | {len(self.events)} |")
        _a("")
        _a("## Goal")
        _a("")
        _a(f"> {goal}")
        _a("")

        # ── RESULT SUMMARY ──
        _a("## Result Summary")
        _a("")
        success = False
        output_text = ""
        if result:
            success = getattr(result, "success", False)
            _a(f"- **Success**: {'Yes' if success else 'No'}")

            # Extract output
            for attr in ("output", "content", "final_output"):
                val = getattr(result, attr, None)
                if isinstance(val, str) and len(val) > len(output_text):
                    output_text = val
                elif isinstance(val, dict):
                    for k in ("response", "text", "content", "output"):
                        t = val.get(k, "")
                        if isinstance(t, str) and len(t) > len(output_text):
                            output_text = t
            if not output_text and hasattr(result, "output"):
                raw = result.output
                if hasattr(raw, "final_output"):
                    output_text = str(getattr(raw, "final_output", ""))[:5000]
                elif hasattr(raw, "outputs"):
                    outputs = getattr(raw, "outputs", {})
                    if isinstance(outputs, dict):
                        for v in outputs.values():
                            if isinstance(v, dict):
                                t = v.get("response", "") or v.get("output", "")
                                if isinstance(t, str) and len(t) > len(output_text):
                                    output_text = t

            quality = getattr(result, "quality_score", None)
            if quality is not None:
                _a(f"- **Quality Score**: {quality:.2f}")
            exec_time = getattr(result, "execution_time", None)
            if exec_time is not None:
                _a(f"- **Execution Time**: {exec_time:.1f}s")

            # Agent contributions
            contributions = getattr(result, "agent_contributions", {})
            if contributions:
                _a(f"- **Agent Contributions**: {len(contributions)} agents")
                for agent, contrib in contributions.items():
                    _a(f"  - `{agent}`: {contrib}")

            # Trajectory
            trajectory = getattr(result, "trajectory", [])
            if trajectory:
                _a(f"- **Trajectory Steps**: {len(trajectory)}")

            _a(f"- **Output Length**: {len(output_text):,} chars")
        else:
            _a("- **Result**: No result returned")
        _a("")

        # ── EXECUTION TIMELINE ──
        _a("## Execution Timeline")
        _a("")
        if self.events:
            _a("| Time | Stage | Detail |")
            _a("|------|-------|--------|")
            for ev in self.events:
                elapsed = f"{ev.elapsed:.1f}s"
                stage = ev.stage.replace("|", "\\|")[:40]
                detail = ev.detail.replace("|", "\\|").replace("\n", " ")[:120]
                _a(f"| {elapsed} | {stage} | {detail} |")
        else:
            _a("*No status events captured. The execution path may not emit status callbacks.*")
        _a("")

        # ── PHASE ANALYSIS ──
        _a("## Phase Analysis")
        _a("")
        phases = self._group_into_phases()
        if phases:
            for phase_name, phase_events in phases.items():
                if not phase_events:
                    continue
                start_t = phase_events[0].elapsed
                end_t = phase_events[-1].elapsed
                duration = end_t - start_t
                _a(f"### {phase_name} ({duration:.1f}s)")
                _a("")
                for ev in phase_events:
                    _a(f"- **[{ev.elapsed:.1f}s]** {ev.stage}: {ev.detail[:200]}")
                _a("")
        else:
            _a("*No phases detected.*")
            _a("")

        # ── SKILL SELECTION & Q-TABLE ──
        _a("## Skill Selection & Q-Table")
        _a("")
        skill_events = [
            e for e in self.events if "skill" in e.stage.lower() or "skill" in e.detail.lower()
        ]
        if skill_events:
            for ev in skill_events:
                _a(f"- **[{ev.elapsed:.1f}s]** {ev.detail[:300]}")
            _a("")

        if self._post_snapshot and self._post_snapshot.q_table:
            _a("### Q-Table State (Post-Execution)")
            _a("")
            _a("*Keys may be `task_type` or `task_type:domain` (domain-specific learning).*")
            _a("")
            import math

            for task_type in sorted(self._post_snapshot.q_table.keys()):
                skills = self._post_snapshot.q_table[task_type]
                counts = self._post_snapshot.q_counts.get(task_type, {})
                total_n = sum(counts.values()) if counts else 0
                label = task_type
                if ":" in task_type:
                    base, dom = task_type.split(":", 1)
                    label = f"{base} (domain: {dom})"
                _a(f"**{label}** (total visits={total_n})")
                _a("")
                _a("| Skill | Q-Value | Visits | UCB Score |")
                _a("|-------|---------|--------|-----------|")
                for skill, q in sorted(skills.items(), key=lambda x: -x[1]):
                    n = counts.get(skill, 0)
                    if n > 0 and total_n > 0:
                        ucb = q + 1.41 * math.sqrt(math.log(total_n) / n)
                    else:
                        ucb = float("inf")
                    ucb_str = f"{ucb:.3f}" if ucb != float("inf") else "∞"
                    _a(f"| {skill} | {q:.4f} | {n} | {ucb_str} |")
                _a("")

        # ── STEP Q-TABLE (Plan Structure Quality) ──
        step_q_data = self._capture_step_q_snapshot()
        if step_q_data.get("step_stats") or step_q_data.get("role_guidance"):
            _a("## Plan Structure Quality (Step & Role Q-Tables)")
            _a("")
            _a(
                "*Dual tracking: position-based (rigid) and role-based (transferable across plan lengths).*"
            )
            rev_count = step_q_data.get("revision_count", 0)
            if rev_count:
                _a(f"*Revisions applied: {rev_count}*")
            _a("")

            def _domain_label(key: str) -> str:
                """Format 'coding:finance' as 'coding (domain: finance)'."""
                if ":" in key:
                    base, dom = key.split(":", 1)
                    return f"{base} (domain: {dom})"
                return key

            # ── Role Q-Table (primary — transferable) ──
            role_guidance = step_q_data.get("role_guidance", {})
            if role_guidance:
                _a("### Role Q-Table (Primary)")
                _a("")
                _a(
                    "*Roles are inferred from skill + description (e.g. 'generate', 'save', 'execute', 'verify').*"
                )
                _a(
                    "*Transferable across plan lengths. Domain-specific keys (task_type:domain) shown when present.*"
                )
                _a("")
                for domain_key in sorted(role_guidance):
                    guidance = role_guidance[domain_key]
                    if not guidance:
                        continue
                    _a(f"#### {_domain_label(domain_key)}")
                    _a("")
                    _a("| Role | Best Skill | Q-Value | Visits | All Skills |")
                    _a("|------|-----------|---------|--------|------------|")
                    for g in guidance:
                        all_skills_str = ", ".join(
                            f"{sk}={info['q']:.2f}({info['n']})"
                            for sk, info in g["all_skills"].items()
                        )
                        _a(
                            f"| {g['role']} | {g['best_skill']} | {g['best_q']:.3f} "
                            f"| {g['total_visits']} | {all_skills_str} |"
                        )
                    _a("")

            # ── Position Q-Table (auxiliary) ──
            if step_q_data.get("step_stats"):
                _a("### Position Q-Table (Auxiliary)")
                _a("")
                _a(
                    "*Position-specific: 'step 0 should use X'. Only meaningful within same-length plans.*"
                )
                _a("")
                for domain_key in sorted(step_q_data["step_stats"]):
                    stats = step_q_data["step_stats"][domain_key]
                    _a(f"#### {_domain_label(domain_key)}")
                    _a("")
                    _a("| Step | Best Skill | Q-Value | Visits | All Skills |")
                    _a("|------|-----------|---------|--------|------------|")
                    for step_pos in sorted(stats):
                        s = stats[step_pos]
                        best = s["best_skill"]
                        best_q = s["best_q"]
                        all_skills = ", ".join(
                            f"{sk}={info['q']:.2f}({info['n']})" for sk, info in s["skills"].items()
                        )
                        total_visits = sum(info["n"] for info in s["skills"].values())
                        _a(
                            f"| Step {step_pos} | {best} | {best_q:.3f} "
                            f"| {total_visits} | {all_skills} |"
                        )
                    _a("")

            # ── Best plans (stored as role sequences now) ──
            best_plans = step_q_data.get("best_plans", {})
            any_plans = any(best_plans.get(tt) for tt in best_plans)
            if any_plans:
                _a("### Best Plan Templates (Role Sequences)")
                _a("")
                for domain_key in sorted(best_plans):
                    plans = best_plans[domain_key]
                    if not plans:
                        continue
                    _a(f"**{_domain_label(domain_key)}:**")
                    _a("")
                    for i, plan in enumerate(plans[:3], 1):
                        roles_str = " → ".join(plan["roles"])
                        _a(f"{i}. `{roles_str}` (reward={plan['reward']})")
                    _a("")

        # ── LEARNING SIGNALS ──
        _a("## Learning Signals")
        _a("")

        if self._pre_snapshot and self._post_snapshot:
            ep_before = self._pre_snapshot.episode_count
            ep_after = self._post_snapshot.episode_count
            new_episodes = ep_after - ep_before
            _a(f"- **Episodes Before**: {ep_before}")
            _a(f"- **Episodes After**: {ep_after} (+{new_episodes})")
            _a("")

            # Q-table diff
            if self._pre_snapshot.q_table and self._post_snapshot.q_table:
                _a("### Q-Table Changes")
                _a("")
                changes_found = False
                for tt in self._post_snapshot.q_table:
                    pre_skills = self._pre_snapshot.q_table.get(tt, {})
                    post_skills = self._post_snapshot.q_table[tt]
                    pre_counts = self._pre_snapshot.q_counts.get(tt, {})
                    post_counts = self._post_snapshot.q_counts.get(tt, {})
                    for skill in post_skills:
                        old_q = pre_skills.get(skill, 0.5)
                        new_q = post_skills[skill]
                        old_n = pre_counts.get(skill, 0)
                        new_n = post_counts.get(skill, 0)
                        if old_n != new_n:
                            delta = new_q - old_q
                            arrow = "↑" if delta > 0 else "↓" if delta < 0 else "→"
                            _a(
                                f"- `{tt}/{skill}`: {old_q:.4f} {arrow} {new_q:.4f} "
                                f"(Δ={delta:+.4f}, visits {old_n}→{new_n})"
                            )
                            changes_found = True
                if not changes_found:
                    _a("*No Q-table changes during this execution.*")
                _a("")

        # Reflections
        if self._post_snapshot and self._post_snapshot.recent_reflections:
            _a("### Recent Reflections (Reflexion)")
            _a("")
            for r in self._post_snapshot.recent_reflections[:5]:
                _a(f"- {r[:300]}")
            _a("")

        # Patterns (Voyager skill library)
        if self._post_snapshot and self._post_snapshot.recent_patterns:
            _a("### Proven Strategies (Voyager Skill Library)")
            _a("")
            _a("| Strategy | Confidence | Evidence |")
            _a("|----------|------------|----------|")
            for p in self._post_snapshot.recent_patterns[:5]:
                conf = p.get("confidence", 0)
                desc = p.get("description", "")[:80]
                rec = p.get("recommendation", "")[:80]
                ev = p.get("evidence", 0)
                _a(f"| {desc}: {rec} | {conf:.0%} | {ev} episodes |")
            _a("")

        # Value estimates
        if self._post_snapshot and self._post_snapshot.value_estimates:
            _a("### Top Value Estimates")
            _a("")
            _a("| State | Action | Domain | Value | Updates |")
            _a("|-------|--------|--------|-------|---------|")
            for ve in self._post_snapshot.value_estimates[:10]:
                _a(
                    f"| {ve['state'][:30]} | {ve['action'][:20]} | {ve['domain']} "
                    f"| {ve['value']:.4f} | {ve['updates']} |"
                )
            _a("")

        # ── OUTPUT EXCERPT ──
        if output_text:
            _a("## Output Excerpt")
            _a("")
            excerpt = output_text[:3000]
            if len(output_text) > 3000:
                excerpt += f"\n\n... ({len(output_text) - 3000:,} more chars)"
            _a("```")
            _a(excerpt)
            _a("```")
            _a("")

        # ── BUDGET ──
        _a("## Budget & Cost")
        _a("")
        cost_data = self._extract_cost_data(result)
        if cost_data["total_calls"] > 0:
            _a(f"- **Total LLM Calls**: {cost_data['total_calls']}")
            _a(f"- **Input Tokens**: {cost_data['input_tokens']:,}")
            _a(f"- **Output Tokens**: {cost_data['output_tokens']:,}")
            total_tokens = cost_data["input_tokens"] + cost_data["output_tokens"]
            _a(f"- **Total Tokens**: {total_tokens:,}")
            _a(f"- **Estimated Cost**: ${cost_data['estimated_cost']:.4f}")
            if cost_data.get("models"):
                _a(f"- **Models Used**: {', '.join(cost_data['models'])}")
            if cost_data.get("breakdown"):
                _a("")
                _a("### Per-Step Cost Breakdown")
                _a("")
                _a("| Step | Skill | Input Tokens | Output Tokens | Time |")
                _a("|------|-------|-------------|--------------|------|")
                for step in cost_data["breakdown"]:
                    _a(
                        f"| {step['step']} | {step['skill']} "
                        f"| {step.get('input_tokens', '?'):,} "
                        f"| {step.get('output_tokens', '?'):,} "
                        f"| {step.get('time', '?')} |"
                    )
        else:
            _a(
                "*No cost data captured. LLM calls may use providers that don't report token usage.*"
            )
        _a("")

        _a("---")
        _a("*Generated by Jotty ExecutionTracer v1.0*")

        return "\n".join(lines)

    def _group_into_phases(self) -> Dict[str, List[TraceEvent]]:
        """Group trace events into logical phases."""
        phases: Dict[str, List[TraceEvent]] = {
            "1. Classification & Routing": [],
            "2. Setup & Context": [],
            "3. Skill Discovery & Selection": [],
            "4. Planning": [],
            "5. Execution": [],
            "6. Post-Processing & Learning": [],
        }

        phase_keywords = {
            "1. Classification & Routing": [
                "model tier",
                "analyzing task",
                "single-agent",
                "multi-agent",
                "auto-detect",
                "validationgate",
                "fast path",
                "direct mode",
                "llm analyzing",
                "sequential",
                "parallel",
            ],
            "2. Setup & Context": [
                "autonomous setup",
                "parsing intent",
                "setup complete",
                "intelligence",
                "learned hints",
                "preparing",
                "retrieving context",
                "architect",
                "zero-config",
                "autoagent",
            ],
            "3. Skill Discovery & Selection": [
                "skills found",
                "skills selected",
                "selecting",
                "choosing best",
                "discovering skills",
                "pre-filter",
                "ucb",
                "q-table",
                "task type",
            ],
            "4. Planning": [
                "plan ready",
                "creating execution plan",
                "planning",
                "vvp",
                "visual verification",
                "auto-correct",
            ],
            "5. Execution": [
                "step ",
                "loading",
                "executing",
                "done",
                "completed",
                "failed",
                "retry",
                "replan",
                "replanned",
                "budget",
            ],
            "6. Post-Processing & Learning": [
                "complete",
                "record",
                "episode",
                "reflect",
                "judge",
                "quality",
                "persist",
                "post-execution",
                "overhead",
            ],
        }

        for ev in self.events:
            combined = f"{ev.stage} {ev.detail}".lower()
            placed = False
            for phase, keywords in phase_keywords.items():
                if any(kw in combined for kw in keywords):
                    phases[phase].append(ev)
                    placed = True
                    break
            if not placed:
                phases["5. Execution"].append(ev)

        return {k: v for k, v in phases.items() if v}

    def _extract_cost_data(self, result: Any) -> Dict[str, Any]:
        """Extract cost/token data from result object, DSPy LM history, and step outputs."""
        data: Dict[str, Any] = {
            "total_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "estimated_cost": 0.0,
            "models": set(),
            "breakdown": [],
        }

        # Source 1: result.usage (chat mode — LLMExecutionResult)
        if result and hasattr(result, "usage") and result.usage:
            u = result.usage
            inp = u.get("input_tokens", 0) or 0
            out = u.get("output_tokens", 0) or 0
            data["input_tokens"] += inp
            data["output_tokens"] += out
            data["total_calls"] += 1

        # Source 2: result.output.outputs (run mode — AgenticExecutionResult step outputs)
        raw_output = getattr(result, "output", None) if result else None
        outputs = getattr(raw_output, "outputs", None)
        if isinstance(outputs, dict):
            for step_key, step_val in outputs.items():
                if not isinstance(step_val, dict):
                    continue
                model = step_val.get("model", "")
                provider = step_val.get("provider", "")
                exec_time = step_val.get("_execution_time_ms", 0)
                skill = step_val.get("skill", step_key)

                # Some tool results include usage
                step_usage = step_val.get("usage", {})
                inp = step_usage.get("input_tokens", 0) if step_usage else 0
                out = step_usage.get("output_tokens", 0) if step_usage else 0

                if model:
                    data["models"].add(model)
                if provider or model:
                    data["total_calls"] += 1
                    data["input_tokens"] += inp
                    data["output_tokens"] += out
                    data["breakdown"].append(
                        {
                            "step": step_key,
                            "skill": skill,
                            "input_tokens": inp,
                            "output_tokens": out,
                            "time": f"{exec_time / 1000:.1f}s" if exec_time else "?",
                            "model": model,
                        }
                    )

        # Source 3: DSPy LM history (catches all calls made via dspy)
        try:
            import dspy

            lm = dspy.settings.lm
            if lm and hasattr(lm, "history"):
                history = lm.history
                if isinstance(history, list):
                    dspy_calls = 0
                    dspy_input = 0
                    dspy_output = 0
                    for entry in history:
                        if isinstance(entry, dict):
                            u = entry.get("usage", {}) or {}
                            dspy_input += u.get("input_tokens", 0) or u.get("prompt_tokens", 0) or 0
                            dspy_output += (
                                u.get("output_tokens", 0) or u.get("completion_tokens", 0) or 0
                            )
                            dspy_calls += 1
                            m = entry.get("model", "")
                            if m:
                                data["models"].add(m)
                    # Only use DSPy data if it's larger (avoids double-counting)
                    if dspy_input > data["input_tokens"]:
                        data["input_tokens"] = dspy_input
                        data["output_tokens"] = dspy_output
                        data["total_calls"] = max(data["total_calls"], dspy_calls)
        except Exception:
            pass

        # Source 4: Count LLM calls from trace events as fallback
        if data["total_calls"] == 0:
            llm_events = [
                e
                for e in self.events
                if any(
                    kw in e.detail.lower()
                    for kw in [
                        "claude-api-llm",
                        "claude-cli-llm",
                        "running claude",
                        "running openai",
                    ]
                )
            ]
            data["total_calls"] = len(llm_events)

        # Estimate cost (Sonnet pricing: $3/$15 per 1M tokens)
        cost_input = data["input_tokens"] * 3.0 / 1_000_000
        cost_output = data["output_tokens"] * 15.0 / 1_000_000
        data["estimated_cost"] = cost_input + cost_output

        data["models"] = sorted(data["models"]) if data["models"] else []
        return data

    def save_report(self, report: str, path: str) -> str:
        """Save report to file. Creates parent directories."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(report, encoding="utf-8")
        logger.info(f"Execution report saved to {p}")
        return str(p.resolve())


# =============================================================================
# CONVENIENCE WRAPPERS — traced_run() and traced_chat()
# =============================================================================


async def traced_run(
    orchestrator: Any,
    goal: str,
    *,
    report_dir: str = "reports",
    user_callback: Optional[Callable] = None,
    **kwargs: Any,
) -> Tuple[Any, str]:
    """Run orchestrator.run() with full tracing.

    Returns:
        (result, report_path)
    """
    tracer = ExecutionTracer()
    tracer.goal = goal
    tracer.mode = "run"
    tracer._user_callback = user_callback

    tracer.take_pre_snapshot()

    result = await orchestrator.run(
        goal,
        status_callback=tracer.callback,
        **kwargs,
    )

    tracer.take_post_snapshot()

    report = tracer.generate_report(result, goal=goal, mode="run")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = goal[:40].lower().replace(" ", "_").replace("/", "_")
    filename = f"{report_dir}/{ts}_{slug}.md"
    path = tracer.save_report(report, filename)

    return result, path


async def traced_chat(
    orchestrator: Any,
    message: str,
    *,
    history: Optional[List[Dict[str, Any]]] = None,
    report_dir: str = "reports",
    user_callback: Optional[Callable] = None,
    **kwargs: Any,
) -> Tuple[Any, str]:
    """Run orchestrator.chat() with full tracing.

    Returns:
        (result, report_path)
    """
    tracer = ExecutionTracer()
    tracer.goal = message
    tracer.mode = "chat"
    tracer._user_callback = user_callback

    tracer.take_pre_snapshot()

    result = await orchestrator.chat(
        message,
        history=history,
        status_callback=tracer.callback,
        **kwargs,
    )

    tracer.take_post_snapshot()

    report = tracer.generate_report(result, goal=message, mode="chat")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = message[:40].lower().replace(" ", "_").replace("/", "_")
    filename = f"{report_dir}/{ts}_chat_{slug}.md"
    path = tracer.save_report(report, filename)

    return result, path
