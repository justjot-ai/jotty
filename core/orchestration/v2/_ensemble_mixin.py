"""
SwarmManager Ensemble Mixin
============================

Extracted from swarm_manager.py — handles prompt ensembling
for multi-perspective analysis.
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class EnsembleMixin:
    """Mixin for prompt ensemble execution."""

    async def _execute_ensemble(
        self,
        goal: str,
        strategy: str = 'multi_perspective',
        status_callback=None
    ) -> Dict[str, Any]:
        """
        Execute prompt ensembling for multi-perspective analysis.

        Strategies:
        - self_consistency: Same prompt, N samples, synthesis
        - multi_perspective: Different expert personas (default)
        - gsa: Generative Self-Aggregation
        - debate: Multi-round argumentation
        """
        def _status(stage: str, detail: str = ""):
            if status_callback:
                try:
                    status_callback(stage, detail)
                except Exception:
                    pass

        try:
            # Try to use the skill
            try:
                from Jotty.core.registry.skills_registry import get_skills_registry
                registry = get_skills_registry()
                registry.init()
                skill = registry.get_skill('claude-cli-llm')

                if skill:
                    ensemble_tool = skill.tools.get('ensemble_prompt_tool')
                    if ensemble_tool:
                        _status("Ensemble", f"using {strategy} strategy (domain-aware)")
                        result = ensemble_tool({
                            'prompt': goal,
                            'strategy': strategy,
                            'synthesis_style': 'structured',
                            'verbose': True
                        })
                        if result.get('success') and result.get('quality_scores'):
                            for name, score in result['quality_scores'].items():
                                _status(f"  {name}", f"quality={score:.0%}")
                        return result
            except ImportError:
                pass

            # Fallback: Use DSPy directly for multi-perspective
            import dspy
            if not hasattr(dspy.settings, 'lm') or dspy.settings.lm is None:
                return {'success': False, 'error': 'No LLM configured'}

            lm = dspy.settings.lm

            perspectives = [
                ("analytical", "Analyze this from a data-driven, logical perspective:"),
                ("creative", "Consider unconventional angles and innovative solutions:"),
                ("critical", "Play devil's advocate - identify risks and problems:"),
                ("practical", "Focus on feasibility and actionable steps:"),
            ]

            responses = {}
            for name, prefix in perspectives:
                _status(f"  {name}", "analyzing...")
                try:
                    prompt = f"{prefix}\n\n{goal}"
                    response = lm(prompt=prompt)
                    text = response[0] if isinstance(response, list) else str(response)
                    responses[name] = text
                except Exception as e:
                    logger.warning(f"Perspective '{name}' failed: {e}")

            if not responses:
                return {'success': False, 'error': 'All perspectives failed'}

            _status("Synthesizing", f"{len(responses)} perspectives")
            synthesis_prompt = f"""Synthesize these {len(responses)} expert perspectives into a comprehensive analysis:

Question: {goal}

{chr(10).join(f'**{k.upper()}:** {v[:600]}' for k, v in responses.items())}

Provide a structured synthesis with:
1. **Consensus**: Where perspectives agree
2. **Tensions**: Where they diverge
3. **Blind Spots**: Unique insights from each
4. **Recommendation**: Balanced conclusion"""

            synthesis = lm(prompt=synthesis_prompt)
            final_response = synthesis[0] if isinstance(synthesis, list) else str(synthesis)

            return {
                'success': True,
                'response': final_response,
                'perspectives_used': list(responses.keys()),
                'individual_responses': responses,
                'strategy': strategy,
                'confidence': len(responses) / len(perspectives)
            }

        except Exception as e:
            logger.error(f"Ensemble execution failed: {e}", exc_info=True)
            return {'success': False, 'error': str(e)}

    def _should_auto_ensemble(self, goal: str) -> bool:
        """Decide if auto-ensemble is beneficial for this task."""
        analysis_keywords = ['compare', 'analyze', 'evaluate', 'assess', 'review',
                             'pros and cons', 'vs', 'versus', 'trade-off']
        goal_lower = goal.lower()
        return any(kw in goal_lower for kw in analysis_keywords)
