"""
Orchestrator Ensemble Mixin
============================

Extracted from swarm_manager.py — handles prompt ensembling
for multi-perspective analysis.
"""

import logging
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)


class EnsembleMixin:
    """Mixin for prompt ensemble execution."""

    async def _execute_ensemble(self, goal: str, strategy: str = 'multi_perspective', status_callback: Any = None, max_perspectives: int = 4) -> Dict[str, Any]:
        """
        Execute prompt ensembling for multi-perspective analysis.

        Optima-inspired (Chen et al., 2024): max_perspectives controls
        cost vs. quality tradeoff. Fewer perspectives = faster + cheaper.

        Strategies:
        - self_consistency: Same prompt, N samples, synthesis
        - multi_perspective: Different expert personas (default)
        - gsa: Generative Self-Aggregation
        - debate: Multi-round argumentation
        """
        def _status(stage: str, detail: str = '') -> Any:
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
                        _status("Ensemble", f"{strategy} ({max_perspectives} perspectives)")
                        result = ensemble_tool({
                            'prompt': goal,
                            'strategy': strategy,
                            'synthesis_style': 'structured',
                            'verbose': True,
                            'max_perspectives': max_perspectives,
                        })
                        if result.get('success') and result.get('quality_scores'):
                            for name, score in result['quality_scores'].items():
                                _status(f"  {name}", f"quality={score:.0%}")
                        return result
            except ImportError:
                pass

            # Fallback: Use DSPy directly for multi-perspective (also parallel)
            import dspy
            if not hasattr(dspy.settings, 'lm') or dspy.settings.lm is None:
                return {'success': False, 'error': 'No LLM configured'}

            lm = dspy.settings.lm

            all_perspectives = [
                ("analytical", "Analyze this from a data-driven, logical perspective:"),
                ("creative", "Consider unconventional angles and innovative solutions:"),
                ("critical", "Play devil's advocate - identify risks and problems:"),
                ("practical", "Focus on feasibility and actionable steps:"),
            ]
            perspectives = all_perspectives[:max_perspectives]

            # Optima-inspired: parallel perspective generation
            from concurrent.futures import ThreadPoolExecutor, as_completed
            responses = {}

            def _gen_perspective(name: Any, prefix: Any) -> Tuple:
                prompt = f"{prefix}\n\n{goal}"
                response = lm(prompt=prompt)
                return name, response[0] if isinstance(response, list) else str(response)

            with ThreadPoolExecutor(max_workers=min(len(perspectives), 4)) as executor:
                futures = {
                    executor.submit(_gen_perspective, name, prefix): name
                    for name, prefix in perspectives
                }
                for future in as_completed(futures):
                    try:
                        name, text = future.result()
                        responses[name] = text
                        _status(f"  {name}", "done")
                    except Exception as e:
                        name = futures[future]
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

    def _should_auto_ensemble(self, goal: str) -> tuple[bool, int]:
        """
        Decide if auto-ensemble is beneficial and with how many perspectives.

        Returns:
            (bool, int) tuple: (should_ensemble, max_perspectives)
        """
        from .swarm_ensemble import should_auto_ensemble
        return should_auto_ensemble(goal)
