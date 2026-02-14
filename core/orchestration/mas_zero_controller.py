"""
MASZeroController - MAS-ZERO Capabilities (Composed)
=====================================================

Integrates MAS-ZERO inspired improvements:
1. Building blocks phase (MAS-Init)
2. Meta-feedback evaluation (solvability + completeness)
3. Candidate verification (MAS-Verify)
4. Dynamic reduction (multi -> single)
5. Iterative refinement (MAS-Evolve)
6. Per-problem experience library

Standalone composed class, replaces MASZeroMixin.
"""

import logging
from Jotty.core.utils.async_utils import StatusReporter
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from Jotty.core.foundation.data_structures import EpisodeResult
    from Jotty.core.foundation.agent_config import AgentConfig

logger = logging.getLogger(__name__)


class MASZeroController:
    """
    MAS-ZERO capabilities for Orchestrator.

    Composed into Orchestrator instead of mixed in.
    Takes explicit references to agents and runners.
    """

    def __init__(self, get_agents, get_runners):
        """
        Args:
            get_agents: Callable returning List[AgentConfig]
            get_runners: Callable returning Dict[str, AgentRunner]
        """
        self._get_agents = get_agents
        self._get_runners = get_runners
        self._experience_library = None

    @property
    def agents(self) -> List['AgentConfig']:
        return self._get_agents()

    @property
    def runners(self) -> dict:
        return self._get_runners()

    def _get_experience_library(self):
        """Lazy-load per-problem ExperienceLibrary."""
        if self._experience_library is None:
            from .mas_zero import ExperienceLibrary
            self._experience_library = ExperienceLibrary()
        return self._experience_library

    def reset_experience(self) -> None:
        """Reset experience library for a new problem."""
        if self._experience_library:
            self._experience_library.clear()

    # =========================================================================
    # MAS-VERIFY: Intelligent candidate answer selection
    # =========================================================================

    def verify(self, goal: str, results: Dict[str, Any]) -> Any:
        """
        Select the best output from multiple agent results.

        Uses LLM-based verification when multiple candidates exist.

        Args:
            goal: Original task goal
            results: Dict of agent_name -> EpisodeResult

        Returns:
            Best output (selected by LLM verifier), or None
        """
        from .mas_zero import get_candidate_verifier, CandidateAnswer

        if len(results) <= 1:
            return None

        candidates = []
        for name, result in results.items():
            success = result.success if hasattr(result, 'success') else False
            output = result.output if hasattr(result, 'output') else result
            candidates.append(CandidateAnswer(
                source=f"agent_{name}",
                agent_name=name,
                output=output,
                success=success,
                confidence=0.7 if success else 0.2,
                execution_time=getattr(result, 'execution_time', 0.0),
            ))

        experience = self._get_experience_library()
        for c in candidates:
            experience.add_candidate(
                source=c.source, agent_name=c.agent_name,
                output=c.output, success=c.success,
                confidence=c.confidence, execution_time=c.execution_time,
            )

        verifier = get_candidate_verifier()
        best = verifier.verify(goal, candidates)

        if best:
            logger.info(f"MAS-Verify: selected '{best.agent_name}' ({best.source})")
            return best.output
        return None

    # =========================================================================
    # META-FEEDBACK: Solvability + Completeness evaluation
    # =========================================================================

    def evaluate(self, goal: str, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate MAS design quality after multi-agent execution.

        Returns feedback on solvability, completeness, and whether
        refinement is needed.
        """
        from .mas_zero import get_meta_feedback_evaluator

        sub_tasks = []
        for agent_config in self.agents:
            sub_goal = (
                agent_config.capabilities[0]
                if agent_config.capabilities
                else goal
            )
            sub_tasks.append({
                'agent_name': agent_config.name,
                'sub_goal': sub_goal,
                'strategy': 'direct',
            })

        evaluator = get_meta_feedback_evaluator()
        feedback = evaluator.evaluate(goal, sub_tasks, results)

        experience = self._get_experience_library()
        for st in sub_tasks:
            result = results.get(st['agent_name'])
            success = False
            if result:
                success = result.success if hasattr(result, 'success') else result.get('success', False)
            experience.record(
                agent_name=st['agent_name'],
                sub_goal=st['sub_goal'],
                strategy=st['strategy'],
                output_summary=str(getattr(result, 'output', ''))[:300] if result else '',
                success=success,
                feedback=feedback.get('feedback', ''),
                solvability=feedback.get('solvability', 0.0),
                completeness=feedback.get('completeness', 0.0),
            )

        if feedback.get('should_refine'):
            logger.info(
                f"Meta-feedback: refinement needed "
                f"(solvability={feedback['solvability']:.0%}, "
                f"completeness={feedback['completeness']:.0%})"
            )

        return feedback

    # =========================================================================
    # DYNAMIC REDUCTION: Multi -> Single when simpler is better
    # =========================================================================

    def should_reduce(self, goal: str) -> bool:
        """
        Check if multi-agent should be reduced to single-agent.
        """
        from .mas_zero import should_reduce_to_single

        agents_count = len(self.agents)
        experience = self._get_experience_library()
        return should_reduce_to_single(goal, agents_count, experience)

    # =========================================================================
    # MAS-EVOLVE: Iterative refinement of MAS design
    # =========================================================================

    async def evolve(
        self,
        goal: str,
        initial_results: Dict[str, Any],
        max_iterations: int = 2,
        status_callback=None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Iteratively refine MAS design based on meta-feedback.
        """
        _status = StatusReporter(status_callback)

        experience = self._get_experience_library()
        best_results = initial_results

        for iteration in range(max_iterations):
            feedback = self.evaluate(goal, best_results)

            if not feedback.get('should_refine'):
                _status("MAS-Evolve", f"design acceptable after iteration {iteration}")
                break

            _status(
                "MAS-Evolve",
                f"iteration {iteration + 1}: refining "
                f"(solvability={feedback['solvability']:.0%})",
            )
            experience.next_iteration()

            too_hard = feedback.get('too_hard_tasks', [])
            if too_hard:
                _status("MAS-Evolve", f"re-routing {len(too_hard)} too-hard tasks")

            refinement_needed = False
            for agent_config in self.agents:
                name = agent_config.name
                result = best_results.get(name)
                success = result.success if hasattr(result, 'success') else False

                if success:
                    continue

                refinement_needed = True
                runner = self.runners.get(name)
                if not runner:
                    continue

                context = experience.get_context_summary()
                enriched_goal = (
                    agent_config.capabilities[0]
                    if agent_config.capabilities
                    else goal
                )
                if context:
                    enriched_goal = f"{enriched_goal}\n\n{context}"

                _status("MAS-Evolve", f"re-running {name} with feedback context")
                try:
                    new_result = await runner.run(goal=enriched_goal, **kwargs)
                    best_results[name] = new_result
                except Exception as e:
                    logger.warning(f"MAS-Evolve re-run failed for {name}: {e}")

            if not refinement_needed:
                break

        return best_results
