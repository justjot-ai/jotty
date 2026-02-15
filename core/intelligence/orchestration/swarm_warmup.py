"""
SwarmWarmup - Extracted from Orchestrator
==========================================

DrZero-inspired zero-data bootstrapping for cold-start mitigation.
"""

import logging
import time as time_module
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from .swarm_manager import Orchestrator

logger = logging.getLogger(__name__)


class SwarmWarmup:
    """
    DrZero-inspired warmup: runs synthetic training episodes
    to bootstrap agent learning before real user tasks.
    """

    def __init__(self, swarm: "Orchestrator") -> None:
        self.swarm = swarm

    async def warmup(
        self,
        num_episodes: int = 10,
        target_agent: Optional[str] = None,
        difficulty_range: Tuple[float, float] = (0.2, 0.6),
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Run synthetic training episodes for cold-start mitigation.

        Args:
            num_episodes: Number of synthetic training episodes
            target_agent: Optional specific agent to train (None = all)
            difficulty_range: (min, max) difficulty for curriculum
            verbose: Log progress

        Returns:
            Dict with warmup statistics
        """
        if verbose:
            logger.info(f" Starting DrZero warmup: {num_episodes} synthetic episodes")

        stats = {
            "episodes_run": 0,
            "successes": 0,
            "failures": 0,
            "agent_results": defaultdict(lambda: {"success": 0, "total": 0}),
            "task_type_results": defaultdict(lambda: {"success": 0, "total": 0}),
            "initial_baselines": dict(
                self.swarm.swarm_intelligence.curriculum_generator.difficulty_by_type
            ),
        }

        curriculum = self.swarm.swarm_intelligence.curriculum_generator

        for episode in range(num_episodes):
            task = curriculum.generate_training_task(
                profiles=self.swarm.swarm_intelligence.agent_profiles,
                target_agent=target_agent,
            )

            if task.difficulty < difficulty_range[0]:
                task.difficulty = difficulty_range[0]
            elif task.difficulty > difficulty_range[1]:
                task.difficulty = difficulty_range[1]

            if verbose:
                logger.info(
                    f"  [{episode + 1}/{num_episodes}] "
                    f"Task: {task.task_type} (difficulty: {task.difficulty:.1%})"
                )

            try:
                result = await self._run_synthetic_episode(task, target_agent)
                success = result.get("success", False)
                execution_time = result.get("execution_time", 0.0)
            except Exception as e:
                logger.warning(f"  Warmup episode {episode + 1} failed: {e}")
                success = False
                execution_time = 0.0

            curriculum.update_from_result(task, success, execution_time)

            stats["episodes_run"] += 1
            if success:
                stats["successes"] += 1
            else:
                stats["failures"] += 1

            agent_name = task.target_agent or "swarm"
            stats["agent_results"][agent_name]["total"] += 1
            if success:
                stats["agent_results"][agent_name]["success"] += 1

            stats["task_type_results"][task.task_type]["total"] += 1
            if success:
                stats["task_type_results"][task.task_type]["success"] += 1

        stats["success_rate"] = stats["successes"] / max(1, stats["episodes_run"])
        stats["final_baselines"] = dict(curriculum.difficulty_by_type)
        stats["curriculum_stats"] = curriculum.get_curriculum_stats()

        stats["agent_improvements"] = {}
        for agent_name, results in stats["agent_results"].items():
            rate = results["success"] / max(1, results["total"])
            stats["agent_improvements"][agent_name] = rate

        stats["agent_results"] = dict(stats["agent_results"])
        stats["task_type_results"] = dict(stats["task_type_results"])

        if verbose:
            logger.info(f" Warmup complete: {stats['success_rate']:.1%} success rate")
            logger.info(f"   Episodes: {stats['episodes_run']}, Successes: {stats['successes']}")
            for task_type, results in stats["task_type_results"].items():
                rate = results["success"] / max(1, results["total"])
                logger.info(f"   {task_type}: {rate:.1%} ({results['total']} episodes)")

        self.swarm._auto_save_learnings()
        return stats

    async def _run_synthetic_episode(
        self,
        task: Any,
        target_agent: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run a single synthetic training episode."""
        start_time = time_module.time()

        if target_agent:
            agent_name = target_agent
        elif self.swarm.mode == "single":
            agent_name = self.swarm.agents[0].name
        else:
            best = self.swarm.swarm_intelligence.get_best_agent_for_task(
                task.task_type,
                [a.name for a in self.swarm.agents],
            )
            agent_name = best or self.swarm.agents[0].name

        runner = self.swarm.runners.get(agent_name)
        if not runner:
            return {"success": False, "error": "No runner", "execution_time": 0.0}

        try:
            result = await runner.run(goal=task.description)
            success = getattr(result, "success", False)
            execution_time = time_module.time() - start_time

            self.swarm.swarm_intelligence.record_task_result(
                agent_name=agent_name,
                task_type=task.task_type,
                success=success,
                execution_time=execution_time,
                context={
                    "synthetic": True,
                    "warmup": True,
                    "difficulty": task.difficulty,
                    "curriculum_round": task.metadata.get("curriculum_round", 0),
                },
            )

            try:
                self.swarm.transfer_learning.record_experience(
                    query=task.description[:200],
                    agent=agent_name,
                    action=task.task_type,
                    reward=1.0 if success else 0.0,
                    success=success,
                    error="",
                    context={"synthetic": True, "warmup": True},
                )
            except Exception:
                pass

            return {
                "success": success,
                "agent": agent_name,
                "task_type": task.task_type,
                "difficulty": task.difficulty,
                "execution_time": execution_time,
            }

        except Exception as e:
            execution_time = time_module.time() - start_time
            logger.debug(f"Synthetic episode failed: {e}")

            self.swarm.swarm_intelligence.record_task_result(
                agent_name=agent_name,
                task_type=task.task_type,
                success=False,
                execution_time=execution_time,
                context={"synthetic": True, "warmup": True, "error": str(e)},
            )

            return {
                "success": False,
                "agent": agent_name,
                "task_type": task.task_type,
                "difficulty": task.difficulty,
                "execution_time": execution_time,
                "error": str(e),
            }

    def get_recommendation(self) -> Dict[str, Any]:
        """Check if warmup would be beneficial."""
        profiles = self.swarm.swarm_intelligence.agent_profiles
        curriculum = self.swarm.swarm_intelligence.curriculum_generator

        total_tasks = sum(p.total_tasks for p in profiles.values())

        if total_tasks < 5:
            return {
                "should_warmup": True,
                "reason": "Cold start - no learning history",
                "recommended_episodes": 15,
                "weak_areas": list(curriculum.task_templates.keys()),
            }

        weak_areas = []
        for agent_name, profile in profiles.items():
            for task_type, (success, total) in profile.task_success.items():
                if total >= 3 and success / total < 0.5:
                    weak_areas.append(task_type)

        weak_areas = list(set(weak_areas))

        if weak_areas:
            return {
                "should_warmup": True,
                "reason": f'Weak performance in: {", ".join(weak_areas)}',
                "recommended_episodes": len(weak_areas) * 5,
                "weak_areas": weak_areas,
            }

        total_success = sum(sum(s for s, t in p.task_success.values()) for p in profiles.values())
        total_attempts = sum(sum(t for s, t in p.task_success.values()) for p in profiles.values())

        if total_attempts > 0 and total_success / total_attempts < 0.7:
            return {
                "should_warmup": True,
                "reason": f"Overall success rate low: {total_success / total_attempts:.1%}",
                "recommended_episodes": 10,
                "weak_areas": weak_areas,
            }

        return {
            "should_warmup": False,
            "reason": "Learning state is healthy",
            "recommended_episodes": 0,
            "weak_areas": [],
        }
