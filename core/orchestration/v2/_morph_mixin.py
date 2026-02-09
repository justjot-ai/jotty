"""
SwarmIntelligence Morph Mixin
==============================

Extracted from swarm_intelligence.py — handles MorphAgent-inspired
profile optimization (RCS, RDS, TRAS scoring).
"""

import time
import logging
from typing import Dict, Any

from .swarm_data_structures import AgentSpecialization
from .morph_scoring import MorphScores

logger = logging.getLogger(__name__)


class MorphMixin:
    """Mixin for MorphAgent scoring and profile optimization."""

    def compute_morph_scores(self, task: str = None, task_type: str = None) -> Dict[str, MorphScores]:
        """
        Compute MorphAgent scores (RCS/RDS/TRAS) for all agents.

        Args:
            task: Optional task description for TRAS computation
            task_type: Optional task type for TRAS computation

        Returns:
            Dict of agent_name -> MorphScores
        """
        if not self.morph_scorer:
            return {}

        scores = self.morph_scorer.compute_all_scores(
            profiles=self.agent_profiles,
            task=task,
            task_type=task_type
        )

        self.morph_score_history.append({
            'timestamp': time.time(),
            'scores': {name: {'rcs': s.rcs, 'rds': s.rds, 'tras': s.tras} for name, s in scores.items()},
            'task_context': task[:50] if task else ''
        })

        if len(self.morph_score_history) > 100:
            self.morph_score_history = self.morph_score_history[-100:]

        return scores

    def get_swarm_health(self) -> Dict[str, Any]:
        """
        Get overall swarm health using MorphAgent metrics.

        Returns comprehensive health assessment.
        """
        health = {
            'avg_rcs': 0.5,
            'rds': 0.5,
            'avg_trust': 0.5,
            'specialization_coverage': 0.0,
            'agent_count': len(self.agent_profiles),
            'total_tasks': sum(p.total_tasks for p in self.agent_profiles.values()),
            'recommendations': []
        }

        if not self.agent_profiles:
            health['recommendations'].append("No agents registered - add agents to swarm")
            return health

        if self.morph_scorer:
            rcs_scores = []
            for profile in self.agent_profiles.values():
                rcs, _ = self.morph_scorer.compute_rcs(profile)
                rcs_scores.append(rcs)
            health['avg_rcs'] = sum(rcs_scores) / len(rcs_scores) if rcs_scores else 0.5
            health['rds'] = self.morph_scorer.compute_rds(self.agent_profiles)

        trust_scores = [p.trust_score for p in self.agent_profiles.values()]
        health['avg_trust'] = sum(trust_scores) / len(trust_scores) if trust_scores else 0.5

        unique_specs = set(p.specialization for p in self.agent_profiles.values())
        health['specialization_coverage'] = len(unique_specs) / len(AgentSpecialization)

        if health['avg_rcs'] < 0.4:
            health['recommendations'].append(
                "Low role clarity - consider warmup training to specialize agents")
        if health['rds'] < 0.4:
            health['recommendations'].append(
                "Low role differentiation - agents are too similar, consider diversifying")
        if health['avg_trust'] < 0.5:
            health['recommendations'].append(
                "Low average trust - some agents have inconsistent performance")
        if health['total_tasks'] < 10:
            health['recommendations'].append(
                "Limited task history - consider warmup() for self-training")
        if not health['recommendations']:
            health['recommendations'].append("Swarm health is good - no issues detected")

        return health

    def optimize_profiles_morph(self, num_iterations: int = 5, threshold: float = 0.1) -> Dict[str, Any]:
        """
        MorphAgent-inspired profile optimization.

        Iteratively improves agent profiles by computing RCS/RDS scores,
        identifying low-scoring agents, and generating curriculum tasks.
        """
        if not self.agent_profiles:
            return {'success': False, 'reason': 'No agents to optimize'}

        results = {
            'iterations': 0,
            'initial_rds': 0.0,
            'final_rds': 0.0,
            'initial_avg_rcs': 0.0,
            'final_avg_rcs': 0.0,
            'improvements': []
        }

        if self.morph_scorer:
            results['initial_rds'] = self.morph_scorer.compute_rds(self.agent_profiles)
            rcs_scores = [self.morph_scorer.compute_rcs(p)[0] for p in self.agent_profiles.values()]
            results['initial_avg_rcs'] = sum(rcs_scores) / len(rcs_scores) if rcs_scores else 0.5

        prev_score = results['initial_rds'] + results['initial_avg_rcs']

        for iteration in range(num_iterations):
            results['iterations'] = iteration + 1

            low_rcs_agents = []
            for name, profile in self.agent_profiles.items():
                if self.morph_scorer:
                    rcs, components = self.morph_scorer.compute_rcs(profile)
                    if rcs < 0.5:
                        low_rcs_agents.append((name, rcs, components))

            for agent_name, rcs, components in low_rcs_agents[:3]:
                if components.get('focus', 1.0) < 0.5:
                    profile = self.agent_profiles[agent_name]
                    if profile.task_success:
                        best_type = max(
                            profile.task_success.keys(),
                            key=lambda t: profile.task_success[t][0] / max(1, profile.task_success[t][1])
                        )
                        results['improvements'].append(
                            f"{agent_name}: Focus training on {best_type} (RCS: {rcs:.2f})")

            if self.morph_scorer:
                new_rds = self.morph_scorer.compute_rds(self.agent_profiles)
                new_rcs_scores = [self.morph_scorer.compute_rcs(p)[0] for p in self.agent_profiles.values()]
                new_avg_rcs = sum(new_rcs_scores) / len(new_rcs_scores) if new_rcs_scores else 0.5
                new_score = new_rds + new_avg_rcs

                if abs(new_score - prev_score) < threshold:
                    break

                prev_score = new_score
                results['final_rds'] = new_rds
                results['final_avg_rcs'] = new_avg_rcs

        return results

    def format_morph_report(self) -> str:
        """Generate human-readable MorphAgent scores report."""
        lines = [
            "# MorphAgent Scores Report",
            "=" * 50,
            ""
        ]

        if not self.agent_profiles:
            lines.append("No agents registered.")
            return "\n".join(lines)

        if self.morph_scorer:
            rds = self.morph_scorer.compute_rds(self.agent_profiles)
            lines.append(f"## Swarm Role Differentiation (RDS): {rds:.2f}")
            lines.append(f"   {'✓ Good diversity' if rds >= 0.5 else '⚠️ Agents too similar'}")
            lines.append("")

        lines.append("## Per-Agent Role Clarity (RCS)")
        lines.append("-" * 40)

        for name, profile in self.agent_profiles.items():
            if self.morph_scorer:
                rcs, components = self.morph_scorer.compute_rcs(profile)
                status = "✓" if rcs >= 0.5 else "⚠️"
                lines.append(f"  {status} {name}: RCS={rcs:.2f}")
                lines.append(f"      Focus: {components.get('focus', 0):.2f}, "
                           f"Consistency: {components.get('consistency', 0):.2f}, "
                           f"Specialization: {components.get('specialization', 0):.2f}")

        lines.append("")
        health = self.get_swarm_health()
        lines.append("## Health Summary")
        lines.append(f"  - Average RCS: {health['avg_rcs']:.2f}")
        lines.append(f"  - RDS: {health['rds']:.2f}")
        lines.append(f"  - Average Trust: {health['avg_trust']:.2f}")
        lines.append(f"  - Specialization Coverage: {health['specialization_coverage']:.1%}")

        if health['recommendations']:
            lines.append("")
            lines.append("## Recommendations")
            for rec in health['recommendations']:
                lines.append(f"  - {rec}")

        return "\n".join(lines)
