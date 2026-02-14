"""
EnsembleManager - Prompt Ensembling (Composed)
===============================================

Handles prompt ensembling for multi-perspective analysis.
Standalone composed class, replaces EnsembleMixin.
"""

import logging
from typing import Dict, Any, Tuple

from Jotty.core.utils.async_utils import StatusReporter

logger = logging.getLogger(__name__)


class EnsembleManager:
    """
    Manages prompt ensemble execution.

    Composed into Orchestrator instead of mixed in.
    Stateless — depends only on DSPy global LM config and skills registry.
    """

    async def execute_ensemble(self, goal: str, strategy: str = 'multi_perspective', status_callback: Any = None, max_perspectives: int = 4) -> Dict[str, Any]:
        """
        Execute prompt ensembling for multi-perspective analysis.

        Strategies:
        - self_consistency: Same prompt, N samples, synthesis
        - multi_perspective: Different expert personas (default)
        - gsa: Generative Self-Aggregation
        - debate: Multi-round argumentation
        """
        _status = StatusReporter(status_callback)

        try:
            # Try skill-based ensemble first
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

            # Fallback: Use DSPy directly for multi-perspective
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

            responses = {}

            def _call_with_retry(prompt: Any, max_retries: Any = 4, base_delay: Any = 8.0) -> Any:
                """Call LM with exponential backoff on rate-limit errors."""
                import time
                for attempt in range(max_retries + 1):
                    try:
                        return lm(prompt=prompt)
                    except Exception as e:
                        err_str = str(e)
                        is_rate_limit = ('429' in err_str or 'RateLimit' in err_str or
                                         'rate limit' in err_str.lower())
                        if is_rate_limit and attempt < max_retries:
                            delay = base_delay * (2 ** attempt)
                            logger.info(f"Rate limited, retrying in {delay:.0f}s...")
                            time.sleep(delay)
                        else:
                            raise

            # Sequential execution with retry (safe for rate-limited providers)
            for name, prefix in perspectives:
                prompt = f"{prefix}\n\n{goal}"
                try:
                    response = _call_with_retry(prompt)
                    text = response[0] if isinstance(response, list) else str(response)
                    responses[name] = text
                    _status(f"  {name}", "done")
                except Exception as e:
                    logger.warning(f"Perspective '{name}' failed: {e}")

            if not responses:
                return {'success': False, 'error': 'All perspectives failed'}

            _status("Synthesizing", f"{len(responses)} perspectives")
            synthesis_prompt = (
                f"Synthesize these {len(responses)} expert perspectives "
                f"into a comprehensive analysis:\n\n"
                f"Question: {goal}\n\n"
                + "\n".join(f"**{k.upper()}:** {v[:600]}" for k, v in responses.items())
                + "\n\nProvide a structured synthesis with:\n"
                "1. **Consensus**: Where perspectives agree\n"
                "2. **Tensions**: Where they diverge\n"
                "3. **Blind Spots**: Unique insights from each\n"
                "4. **Recommendation**: Balanced conclusion"
            )

            synthesis = _call_with_retry(synthesis_prompt)
            final_response = synthesis[0] if isinstance(synthesis, list) else str(synthesis)

            return {
                'success': True,
                'response': final_response,
                'perspectives_used': list(responses.keys()),
                'individual_responses': responses,
                'strategy': strategy,
                'confidence': len(responses) / len(perspectives),
            }

        except Exception as e:
            logger.error(f"Ensemble execution failed: {e}", exc_info=True)
            return {'success': False, 'error': str(e)}

    def should_auto_ensemble(self, goal: str) -> Tuple[bool, int]:
        """
        Decide if auto-ensemble is beneficial and with how many perspectives.

        Returns:
            (should_ensemble, max_perspectives) tuple
        """
        from .swarm_ensemble import should_auto_ensemble
        return should_auto_ensemble(goal)
