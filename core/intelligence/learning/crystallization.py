"""
Agent Crystallization — auto-promote learned knowledge into fixed SOPs.

When a domain's learning curve plateaus (high success, stable Q-values,
consistent plan templates), we "crystallize" the proven knowledge into a
config that overrides the AutonomousAgent's exploration with hardened:
  - skill whitelist (proven skills only)
  - SOP role sequence (proven plan template)
  - role→skill bindings (best skill per role)
  - prompt guidance (distilled from episodes)

The AutonomousAgent checks for a crystallized config before planning.
If found, it uses the SOP instead of exploring.  If the crystallized
skills aren't available, it declines — returning control to the caller.

Persistence: JSON files in ~/jotty/learning/crystallized/<key>.json
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Where crystallized configs live on disk
_CRYSTAL_DIR = Path.home() / "jotty" / "learning" / "crystallized"


# =============================================================================
# Config — the crystallized knowledge for one task_type[:domain]
# =============================================================================


@dataclass
class CrystallizedConfig:
    """Hardened domain knowledge extracted from converged Q-tables."""

    domain_key: str  # e.g. "coding" or "coding:finance"
    task_type: str  # base task type
    domain: str = ""  # business domain (optional)
    skills: List[str] = field(default_factory=list)  # proven skill whitelist
    sop_roles: Tuple[str, ...] = ()  # proven role sequence e.g. ("generate","save","verify")
    role_skill_map: Dict[str, str] = field(default_factory=dict)  # role → best skill
    prompt_guidance: str = ""  # baked-in prompt from distilled lessons
    success_rate: float = 0.0
    total_episodes: int = 0
    created_at: str = ""
    role_confidence: Dict[str, float] = field(default_factory=dict)  # role → confidence [0,1]
    consecutive_failures: int = 0  # staleness canary counter

    def to_plan_hint(self, goal: str = "") -> str:
        """Format as a planner-consumable instruction string.

        Args:
            goal: Current task goal — used to retrieve sub-domain-specific
                  lessons instead of dumping all lessons (avoids
                  cross-contamination between e.g. ER and sequence diagrams).
        """
        parts = []
        if self.sop_roles:
            parts.append(
                "CRYSTALLIZED SOP (proven high-reward sequence, follow this):\n"
                + " → ".join(self.sop_roles)
            )
        if self.role_skill_map:
            bindings = "\n".join(
                f"  - {role}: use {skill}" for role, skill in self.role_skill_map.items()
            )
            parts.append(f"SKILL BINDINGS:\n{bindings}")

        # Prefer DSPy-optimized few-shot demos over raw text lessons.
        # The optimized module contains bootstrapped examples selected by
        # BootstrapFewShot — much more effective than plain text injection.
        dspy_available = False
        try:
            from .advanced_learning import DomainDSPyOptimizer

            optimizer = DomainDSPyOptimizer.get_instance()
            if optimizer.has_optimized(self.domain or self.task_type):
                dspy_available = True
                parts.append(
                    "DSPY OPTIMIZED: A BootstrapFewShot-optimized DSPy module is "
                    f"available for domain '{self.domain or self.task_type}'. "
                    "Use it for generation tasks — it contains proven few-shot "
                    "examples from successful episodes."
                )
        except Exception:
            pass

        # Still provide text lessons as fallback/supplement
        if not dspy_available:
            live_lessons = ""
            if goal and self.domain:
                try:
                    from .facade import get_learning_service

                    svc = get_learning_service()
                    lessons = svc.retrieve_distilled_lessons(
                        domain=self.domain or self.task_type,
                        goal=goal,
                        top_k=5,
                    )
                    if lessons:
                        live_lessons = "\n".join(
                            f"- {l['lesson']}" for l in lessons if isinstance(l, dict)
                        )
                except Exception:
                    pass

            if live_lessons:
                parts.append(f"DOMAIN LESSONS (for this specific task):\n{live_lessons}")
            elif self.prompt_guidance:
                parts.append(f"DOMAIN LESSONS:\n{self.prompt_guidance[:800]}")

        if self.role_confidence:
            low = [r for r, c in self.role_confidence.items() if c < 0.7]
            if low:
                parts.append(f"LOW-CONFIDENCE ROLES (explore alternatives): {', '.join(low)}")
        return "\n\n".join(parts)

    def get_dspy_module(self):
        """Load the DSPy-optimized module for this domain, if available.

        Returns the optimized module or None. Callers can use this to
        directly invoke the domain-specific DSPy module instead of
        a raw LLM call.
        """
        try:
            from .advanced_learning import DomainDSPyOptimizer

            optimizer = DomainDSPyOptimizer.get_instance()
            return optimizer.load_optimized(self.domain or self.task_type)
        except Exception:
            return None


# =============================================================================
# Convergence check — should we crystallize?
# =============================================================================


# Default thresholds (derived from LearningConfig — single source of truth)
def _build_default_thresholds() -> Dict[str, float]:
    try:
        from Jotty.core.infrastructure.foundation.configs.learning import LearningConfig

        cfg = LearningConfig()
        return {
            "min_episodes": cfg.crystal_min_episodes,
            "min_success_rate": cfg.crystal_min_success_rate,
            "min_plan_consistency": cfg.crystal_min_plan_consistency,
            "min_role_q": cfg.crystal_min_role_q,
            "min_plans": cfg.crystal_min_plans,
        }
    except Exception:
        return {
            "min_episodes": 25,
            "min_success_rate": 0.85,
            "min_plan_consistency": 0.60,
            "min_role_q": 0.65,
            "min_plans": 8,
        }


DEFAULT_THRESHOLDS: Dict[str, float] = _build_default_thresholds()

# Module-level aliases for backward compat
MIN_EPISODES = int(DEFAULT_THRESHOLDS["min_episodes"])
MIN_SUCCESS_RATE = DEFAULT_THRESHOLDS["min_success_rate"]
MIN_PLAN_CONSISTENCY = DEFAULT_THRESHOLDS["min_plan_consistency"]
MIN_ROLE_Q = DEFAULT_THRESHOLDS["min_role_q"]
MIN_PLANS = int(DEFAULT_THRESHOLDS["min_plans"])


def should_crystallize(
    task_type: str,
    domain: str = "",
    thresholds: Optional[Dict[str, float]] = None,
) -> Tuple[bool, Dict[str, Any]]:
    """Check if a domain's learning has converged enough to crystallize.

    Reads existing Q-table data — no new tracking needed.

    Args:
        thresholds: Override default thresholds. Keys:
            min_episodes, min_success_rate, min_plan_consistency,
            min_role_q, min_plans.

    Returns (should_crystallize, stats_dict).
    """
    t = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    _min_episodes = int(t["min_episodes"])
    _min_success = t["min_success_rate"]
    _min_consistency = t["min_plan_consistency"]
    _min_role_q = t["min_role_q"]
    _min_plans = int(t["min_plans"])

    from .facade import get_td_lambda

    td = get_td_lambda()
    key = td.skill_q._make_key(task_type, domain)
    stats: Dict[str, Any] = {"domain_key": key, "reasons": [], "thresholds": t}

    # 1. Enough observations? (fallback to base task_type if domain-specific is sparse,
    #    then merge sibling task_types for the same domain)
    skill_counts = td.skill_q._counts.get(key, {})
    if not skill_counts and ":" in key:
        base = td.skill_q._base_key(key)
        skill_counts = td.skill_q._counts.get(base, {})

    # Merge sibling task_types for the same domain (e.g. research:travel + creation:travel)
    if domain:
        for other_key in list(td.skill_q._counts.keys()):
            if other_key != key and other_key.endswith(f":{domain}"):
                for skill, cnt in td.skill_q._counts[other_key].items():
                    skill_counts[skill] = skill_counts.get(skill, 0) + cnt

    total_obs = sum(skill_counts.values()) if skill_counts else 0
    if total_obs < _min_episodes:
        stats["reasons"].append(f"too few observations ({total_obs} < {_min_episodes})")
        return False, stats
    stats["total_obs"] = total_obs

    # 2. Plan history — same-type plans for consistency, merged for count/success rate
    # Same-type plans (for consistency check): only the target key + base key
    same_type_plans = list(td.step_q._plan_history.get(key, []))
    if not same_type_plans:
        base = td.step_q._base_key(key)
        same_type_plans = list(td.step_q._plan_history.get(base, []))

    # All plans (for count + success rate): merge sibling task_types for same domain
    all_plans = list(same_type_plans)
    if domain:
        for other_key in list(td.step_q._plan_history.keys()):
            if other_key != key and (other_key.endswith(f":{domain}") or ":" not in other_key):
                all_plans.extend(td.step_q._plan_history[other_key])

    recent_all = all_plans[-20:] if len(all_plans) > 20 else all_plans
    if len(recent_all) < _min_plans:
        stats["reasons"].append(f"too few plans ({len(recent_all)} < {_min_plans})")
        return False, stats
    avg_reward = sum(r for _, r in recent_all) / len(recent_all)
    stats["success_rate"] = avg_reward
    if avg_reward < _min_success:
        stats["reasons"].append(f"success rate too low ({avg_reward:.0%} < {_min_success:.0%})")
        return False, stats

    # 3. Plan template consistency — use same-type plans only (don't mix
    #    "creation" 2-step templates with "research" 7-step templates)
    from collections import Counter

    recent_same = same_type_plans[-20:] if len(same_type_plans) > 20 else same_type_plans
    if len(recent_same) < 3:
        recent_same = recent_all  # fallback if too few same-type
    template_counts = Counter(roles for roles, _ in recent_same)
    top_template, top_count = template_counts.most_common(1)[0]
    consistency = top_count / len(recent_same)
    stats["plan_consistency"] = consistency
    stats["top_template"] = top_template
    if consistency < _min_consistency:
        stats["reasons"].append(f"plan inconsistent ({consistency:.0%} < {_min_consistency:.0%})")
        return False, stats

    # 4. Role Q-values stable and high (ignore roles with < 3 visits — noise)
    role_guidance = td.step_q.get_role_guidance(task_type, domain=domain)
    # Merge role guidance from sibling task types for same domain
    if domain and not role_guidance:
        for alt_type in ("research", "creation", "coding", "analysis"):
            if alt_type != task_type:
                alt_guidance = td.step_q.get_role_guidance(alt_type, domain=domain)
                if not alt_guidance:
                    alt_guidance = td.step_q.get_role_guidance(alt_type)
                if alt_guidance:
                    role_guidance = alt_guidance
                    break
    if not role_guidance:
        role_guidance = td.step_q.get_role_guidance(task_type)
    if not role_guidance:
        stats["reasons"].append("no role guidance data")
        return False, stats
    significant_roles = [g for g in role_guidance if g["total_visits"] >= 3]
    low_roles = [g for g in significant_roles if g["best_q"] < _min_role_q]
    if low_roles:
        stats["reasons"].append(f"roles with low Q: {[r['role'] for r in low_roles]}")
        return False, stats

    stats["reasons"].append("all checks passed")
    return True, stats


# =============================================================================
# Crystallize — extract config from Q-tables
# =============================================================================


def crystallize(
    task_type: str,
    domain: str = "",
    thresholds: Optional[Dict[str, float]] = None,
) -> Optional[CrystallizedConfig]:
    """Extract a CrystallizedConfig from current Q-table state.

    Only succeeds if should_crystallize() passes.
    Saves the config to disk automatically.
    """
    ok, stats = should_crystallize(task_type, domain, thresholds=thresholds)
    if not ok:
        logger.info(f"Cannot crystallize {task_type}:{domain}: {stats['reasons']}")
        return None

    from datetime import datetime

    from .facade import get_td_lambda

    td = get_td_lambda()
    key = stats["domain_key"]

    # Extract top skills
    top_skills = td.skill_q.get_top_skills(task_type, n=6, domain=domain)
    skill_names = [s for s, _ in top_skills]

    # Extract best plan template (role sequence)
    sop_roles = stats.get("top_template", ())

    # Extract role → best skill bindings + confidence scores
    role_guidance = td.step_q.get_role_guidance(task_type, domain=domain)
    role_skill_map = {g["role"]: g["best_skill"] for g in role_guidance}
    role_confidence = {}
    for g in role_guidance:
        q, visits = g["best_q"], g["total_visits"]
        role_confidence[g["role"]] = round(q * min(1.0, visits / 10), 3)

    # Extract prompt guidance from distilled lessons (hierarchical)
    prompt = ""
    try:
        from .facade import get_learning_service

        svc = get_learning_service()
        # Pull all lessons for this domain hierarchy
        lessons = svc.retrieve_distilled_lessons(
            domain=domain or task_type,
            top_k=8,
        )
        if lessons:
            parts = [f"- {l['lesson']}" for l in lessons if isinstance(l, dict)]
            if parts:
                prompt = "\n".join(parts)
    except Exception:
        pass

    config = CrystallizedConfig(
        domain_key=key,
        task_type=task_type,
        domain=domain,
        skills=skill_names,
        sop_roles=sop_roles,
        role_skill_map=role_skill_map,
        prompt_guidance=prompt,
        success_rate=stats.get("success_rate", 0),
        total_episodes=stats.get("total_obs", 0),
        created_at=datetime.now().isoformat(),
        role_confidence=role_confidence,
    )

    _save(config)
    logger.info(
        f"Crystallized {key}: {len(skill_names)} skills, "
        f"SOP={' → '.join(sop_roles)}, success={config.success_rate:.0%}"
    )

    # Run DSPy BootstrapFewShot optimization for this domain.
    # Uses successful episodes + gold data + distilled lessons as training set.
    try:
        from .advanced_learning import DomainDSPyOptimizer

        optimizer = DomainDSPyOptimizer.get_instance()
        optimized = optimizer.optimize(domain or task_type)
        if optimized:
            logger.info(f"DSPy module optimized for {domain or task_type}")
    except Exception as e:
        logger.debug(f"DSPy optimization during crystallization failed (non-fatal): {e}")

    return config


# =============================================================================
# Persistence — JSON files, one per domain key
# =============================================================================


def _key_to_filename(domain_key: str) -> str:
    return domain_key.replace(":", "_") + ".json"


def _save(config: CrystallizedConfig) -> Path:
    _CRYSTAL_DIR.mkdir(parents=True, exist_ok=True)
    path = _CRYSTAL_DIR / _key_to_filename(config.domain_key)
    data = asdict(config)
    data["sop_roles"] = list(config.sop_roles)
    path.write_text(json.dumps(data, indent=2))
    return path


def load(task_type: str, domain: str = "") -> Optional[CrystallizedConfig]:
    """Load a crystallized config from disk if it exists."""
    from .td_lambda import _DomainKeyMixin

    key = _DomainKeyMixin._make_key(task_type, domain)
    path = _CRYSTAL_DIR / _key_to_filename(key)
    if not path.exists():
        # Try base task_type if domain-specific doesn't exist
        if domain:
            base_path = _CRYSTAL_DIR / _key_to_filename(task_type)
            if base_path.exists():
                path = base_path
            else:
                return None
        else:
            return None
    try:
        data = json.loads(path.read_text())
        data["sop_roles"] = tuple(data.get("sop_roles", []))
        return CrystallizedConfig(**data)
    except Exception as e:
        logger.debug(f"Failed to load crystallized config {path}: {e}")
        return None


def decrystallize(task_type: str, domain: str = "") -> bool:
    """Remove a crystallized config (revert to exploration)."""
    from .td_lambda import _DomainKeyMixin

    key = _DomainKeyMixin._make_key(task_type, domain)
    path = _CRYSTAL_DIR / _key_to_filename(key)
    if path.exists():
        path.unlink()
        logger.info(f"Decrystallized {key}")
        return True
    return False


def record_crystallized_outcome(
    task_type: str,
    domain: str = "",
    success: bool = True,
    max_failures: Optional[int] = None,
) -> Optional[str]:
    """Record execution outcome for a crystallized config (staleness canary).

    On success: resets consecutive_failures to 0.
    On failure: increments counter. If >= max_failures, decrystallizes
    and returns "decrystallized". Otherwise returns None.

    Args:
        task_type: The task type
        domain: Business domain
        success: Whether the execution succeeded
        max_failures: Override for max consecutive failures before decrystallize.
            Defaults to LearningConfig.crystal_staleness_failures (3).

    Returns:
        "decrystallized" if config was removed, None otherwise.
    """
    config = load(task_type, domain)
    if config is None:
        return None

    if max_failures is None:
        try:
            from Jotty.core.infrastructure.foundation.configs.learning import LearningConfig

            max_failures = LearningConfig().crystal_staleness_failures
        except Exception:
            max_failures = 3

    if success:
        if config.consecutive_failures != 0:
            config.consecutive_failures = 0
            _save(config)
        return None

    config.consecutive_failures += 1
    if config.consecutive_failures >= max_failures:
        decrystallize(task_type, domain)
        logger.warning(
            f"Staleness canary: decrystallized {config.domain_key} after "
            f"{config.consecutive_failures} consecutive failures"
        )
        return "decrystallized"

    _save(config)
    return None


def list_crystallized() -> List[CrystallizedConfig]:
    """List all crystallized domain configs."""
    configs: list[CrystallizedConfig] = []
    if not _CRYSTAL_DIR.exists():
        return configs
    for path in sorted(_CRYSTAL_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text())
            data["sop_roles"] = tuple(data.get("sop_roles", []))
            configs.append(CrystallizedConfig(**data))
        except Exception:
            pass
    return configs


# =============================================================================
# Auto-crystallize check — call after successful execution
# =============================================================================


def maybe_crystallize(
    task_type: str,
    domain: str = "",
    thresholds: Optional[Dict[str, float]] = None,
) -> Optional[CrystallizedConfig]:
    """Check convergence and crystallize if ready. Idempotent.

    Call this after recording learning outcomes. If already crystallized,
    returns the existing config without re-checking.
    """
    existing = load(task_type, domain)
    if existing:
        return existing
    return crystallize(task_type, domain, thresholds=thresholds)


# =============================================================================
# Probation runner — curriculum → execute → learn → graduate
# =============================================================================


async def run_probation(
    task_type: str,
    domain: str = "",
    max_tasks: int = 15,
    goals: Optional[List[str]] = None,
    thresholds: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Run a probation loop: generate curriculum, execute, learn, crystallize.

    This is the bridge between CurriculumGenerator and Crystallization.
    Each iteration:
      1. Generate or consume a goal (curriculum or user-supplied)
      2. Execute via Orchestrator.run()
      3. Learning records automatically (Q-tables, episodes)
      4. CurriculumGenerator.update_from_result() closes feedback
      5. Check maybe_crystallize() — stop if graduated

    Args:
        task_type: The task type to train (e.g. "coding", "research")
        domain: Business domain (e.g. "finance", "devops")
        max_tasks: Max curriculum tasks before giving up
        goals: Optional explicit goals (skips curriculum generation)
        thresholds: Override graduation thresholds. Keys:
            min_episodes, min_success_rate, min_plan_consistency,
            min_role_q, min_plans.

    Returns:
        Dict with graduated, config, stats
    """
    from Jotty.core.infrastructure.foundation.dspy_init import load_api_keys

    load_api_keys()

    from Jotty.core.intelligence.orchestration.core.swarm_manager import Orchestrator
    from Jotty.core.intelligence.orchestration.intelligence.curriculum_generator import (
        CurriculumGenerator,
    )

    # Already graduated?
    existing = load(task_type, domain)
    if existing:
        return {
            "graduated": True,
            "config": existing,
            "tasks_run": 0,
            "succeeded": 0,
            "success_rate": existing.success_rate,
            "domain_key": existing.domain_key,
            "reason": "already crystallized",
        }

    orch = Orchestrator()
    curriculum = CurriculumGenerator()

    # If user supplied explicit goals, use those; otherwise generate
    goal_queue: List[str] = list(goals) if goals else []

    # Curriculum ordering: sort goals by estimated complexity (easy → hard)
    if len(goal_queue) > 2:

        def _complexity_score(goal: str) -> float:
            score = 0.0
            score += min(len(goal) / 500, 1.0) * 0.3
            tech_terms = {
                "implement",
                "architect",
                "optimize",
                "distributed",
                "concurrent",
                "algorithm",
                "benchmark",
                "migrate",
                "refactor",
                "security",
                "scale",
                "integrate",
            }
            words = goal.lower().split()
            score += min(sum(1 for w in words if w in tech_terms) / 3, 1.0) * 0.4
            for phrase in ("and then", "followed by", "after that", "finally"):
                if phrase in goal.lower():
                    score += 0.1
            return min(score, 1.0)

        goal_queue.sort(key=_complexity_score)

    results: List[Dict[str, Any]] = []
    graduated_config: Optional[CrystallizedConfig] = None

    from .td_lambda import _DomainKeyMixin

    domain_key = _DomainKeyMixin._make_key(task_type, domain) if domain else task_type

    logger.info(
        f"Probation started for {domain_key}: "
        f"max_tasks={max_tasks}, supplied_goals={len(goal_queue)}"
    )

    task = None  # Only set when using curriculum-generated goals

    for i in range(max_tasks):
        # 1. Pick a goal
        if goal_queue:
            goal = goal_queue.pop(0)
            task = None
        else:
            task = curriculum.generate_domain_task(task_type, domain)
            goal = task.description

        logger.info(f"Probation [{i+1}/{max_tasks}] {goal[:80]}")

        # 2. Execute (pass domain_hint so learning records use correct domain)
        try:
            result = await orch.run(goal=goal, domain_hint=domain or "")
            success = (
                result.get("success", False)
                if isinstance(result, dict)
                else getattr(result, "success", False)
            )
        except Exception as e:
            logger.warning(f"Probation task failed: {e}")
            success = False
            result = {"success": False, "error": str(e)}

        results.append({"goal": goal, "success": success, "iteration": i})

        # 3. Feed result back to curriculum (closes the feedback loop)
        if task is not None:
            curriculum.update_from_result(task, success, execution_time=0)

        # 4. Check graduation — check both the target task_type and the
        # classifier's actual task_type (they can diverge: "research travel
        # itinerary" may classify as "creation" instead of "research").
        graduated_config = maybe_crystallize(task_type, domain, thresholds=thresholds)
        if not graduated_config:
            # Also check the base task_type without domain
            graduated_config = maybe_crystallize(task_type, thresholds=thresholds)
        if graduated_config:
            logger.info(
                f"Probation GRADUATED after {i+1} tasks: {domain_key} "
                f"SOP={' → '.join(graduated_config.sop_roles)}"
            )
            break

    succeeded = sum(1 for r in results if r["success"])
    stats = {
        "graduated": graduated_config is not None,
        "config": graduated_config,
        "tasks_run": len(results),
        "succeeded": succeeded,
        "success_rate": succeeded / len(results) if results else 0,
        "domain_key": domain_key,
        "reason": "crystallized" if graduated_config else f"max_tasks reached ({max_tasks})",
    }

    logger.info(
        f"Probation complete for {domain_key}: "
        f"{'GRADUATED' if stats['graduated'] else 'NOT YET'}, "
        f"{succeeded}/{len(results)} succeeded"
    )
    return stats
