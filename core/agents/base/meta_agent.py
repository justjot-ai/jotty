"""
MetaAgent - Self-Improvement Agent Base Class

Base class for agents that evaluate, analyze, and improve other agents:
- ExpertAgent: Evaluates outputs against gold standards
- ReviewerAgent: Analyzes patterns and suggests improvements
- PlannerAgent: Plans task execution with optimization
- ActorAgent: Executes tasks with learned improvements
- AuditorAgent: Verifies evaluation quality
- LearnerAgent: Extracts reusable learnings

Provides:
- Gold standard evaluation against a database
- Cross-agent visibility via shared context
- Improvement suggestion generation
- Learning extraction from excellent outputs

Author: A-Team
Date: February 2026
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Type

from .base_agent import BaseAgent, AgentRuntimeConfig, AgentResult

logger = logging.getLogger(__name__)


# =============================================================================
# META AGENT CONFIG
# =============================================================================

@dataclass
class MetaAgentConfig(AgentRuntimeConfig):
    """Configuration specific to MetaAgent."""
    enable_gold_db: bool = True
    enable_improvement_history: bool = True
    improvement_threshold: float = 0.7
    max_learnings_per_run: int = 5


# =============================================================================
# META AGENT
# =============================================================================

class MetaAgent(BaseAgent):
    """
    Base class for self-improvement agents.

    Provides shared infrastructure for:
    - Evaluating outputs against gold standards
    - Analyzing patterns across evaluations
    - Suggesting improvements for other agents
    - Extracting reusable learnings

    Subclasses (ExpertAgent, ReviewerAgent, etc.) implement specific logic.
    """

    def __init__(
        self,
        signature: Type = None,
        config: MetaAgentConfig = None,
        gold_db=None,
        improvement_history=None,
    ):
        """
        Initialize MetaAgent.

        Args:
            signature: Optional DSPy signature for the agent's task
            config: Agent configuration
            gold_db: GoldStandardDB instance for evaluation
            improvement_history: ImprovementHistory for tracking suggestions
        """
        config = config or MetaAgentConfig(name=self.__class__.__name__)
        super().__init__(config)

        self.signature = signature
        self.gold_db = gold_db
        self.improvement_history = improvement_history

        # DSPy module (created from signature)
        self._dspy_module = None

    def _ensure_initialized(self):
        """Initialize DSPy module if signature provided."""
        super()._ensure_initialized()

        if self._dspy_module is None and self.signature is not None:
            try:
                import dspy
                self._dspy_module = dspy.ChainOfThought(self.signature)
                logger.debug(f"Initialized DSPy module for {self.config.name}")
            except Exception as e:
                logger.warning(f"Failed to initialize DSPy module: {e}")

    # =========================================================================
    # GOLD STANDARD EVALUATION
    # =========================================================================

    async def evaluate_against_gold(
        self,
        gold_id: str,
        output: Dict[str, Any],
        context: str = ""
    ) -> Dict[str, Any]:
        """
        Evaluate output against a gold standard.

        Args:
            gold_id: Gold standard ID to compare against
            output: Actual output to evaluate
            context: Additional context for evaluation

        Returns:
            Evaluation dict with scores, feedback, and result
        """
        if self.gold_db is None:
            logger.warning("No gold_db configured, returning default evaluation")
            return {
                "gold_standard_id": gold_id,
                "scores": {},
                "overall_score": 0.5,
                "result": "needs_improvement",
                "feedback": ["No gold standard database configured"],
            }

        try:
            gold_standard = self.gold_db.get(gold_id)
            if not gold_standard:
                return {
                    "gold_standard_id": gold_id,
                    "scores": {},
                    "overall_score": 0.0,
                    "result": "failed",
                    "feedback": [f"Gold standard not found: {gold_id}"],
                }

            # Use DSPy module for evaluation if available
            if self._dspy_module is not None:
                result = self._dspy_module(
                    gold_standard=json.dumps({
                        'expected_output': gold_standard.expected_output,
                        'criteria': gold_standard.evaluation_criteria
                    }),
                    actual_output=json.dumps(output),
                    context=context
                )

                # Parse scores
                try:
                    scores = json.loads(result.scores) if hasattr(result, 'scores') else {}
                except (json.JSONDecodeError, TypeError):
                    scores = {}

                overall_score = float(result.overall_score) if hasattr(result, 'overall_score') else 0.5
                feedback = str(result.feedback).split('|') if hasattr(result, 'feedback') else []

                # Determine result based on score
                if overall_score >= 0.9:
                    eval_result = "excellent"
                elif overall_score >= 0.8:
                    eval_result = "good"
                elif overall_score >= 0.7:
                    eval_result = "acceptable"
                elif overall_score >= 0.5:
                    eval_result = "needs_improvement"
                else:
                    eval_result = "failed"

                return {
                    "gold_standard_id": gold_id,
                    "actual_output": output,
                    "scores": scores,
                    "overall_score": overall_score,
                    "result": eval_result,
                    "feedback": [f.strip() for f in feedback if f.strip()],
                }

            # Fallback: simple comparison
            return self._simple_evaluation(gold_standard, output)

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return {
                "gold_standard_id": gold_id,
                "scores": {},
                "overall_score": 0.0,
                "result": "failed",
                "feedback": [f"Evaluation error: {str(e)}"],
            }

    def _simple_evaluation(self, gold_standard, output: Dict[str, Any]) -> Dict[str, Any]:
        """Simple fallback evaluation without LLM."""
        expected = gold_standard.expected_output
        scores = {}

        # Compare each expected field
        for key, expected_value in expected.items():
            actual_value = output.get(key)
            if actual_value == expected_value:
                scores[key] = 1.0
            elif actual_value is not None:
                # Partial match for strings
                if isinstance(expected_value, str) and isinstance(actual_value, str):
                    # Simple word overlap score
                    expected_words = set(expected_value.lower().split())
                    actual_words = set(actual_value.lower().split())
                    if expected_words:
                        overlap = len(expected_words & actual_words)
                        scores[key] = overlap / len(expected_words)
                    else:
                        scores[key] = 0.5
                else:
                    scores[key] = 0.5
            else:
                scores[key] = 0.0

        overall = sum(scores.values()) / len(scores) if scores else 0.0

        return {
            "gold_standard_id": gold_standard.id,
            "actual_output": output,
            "scores": scores,
            "overall_score": overall,
            "result": "good" if overall >= 0.8 else "needs_improvement",
            "feedback": [],
        }

    # =========================================================================
    # IMPROVEMENT SUGGESTION
    # =========================================================================

    async def analyze_and_suggest_improvements(
        self,
        evaluations: List[Dict[str, Any]],
        agent_configs: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Analyze evaluations and suggest improvements.

        Args:
            evaluations: List of evaluation dicts
            agent_configs: Optional dict of agent configurations

        Returns:
            List of improvement suggestion dicts
        """
        if not evaluations:
            return []

        if self._dspy_module is None:
            logger.warning("No DSPy module configured, returning empty suggestions")
            return []

        try:
            # Get past successful improvements
            past_improvements = []
            if self.improvement_history is not None:
                past_improvements = self.improvement_history.get_successful_improvements()

            result = self._dspy_module(
                evaluations=json.dumps(evaluations, default=str),
                agent_configs=json.dumps(agent_configs or {}, default=str),
                improvement_history=json.dumps(past_improvements, default=str)
            )

            # Parse suggestions
            try:
                suggestions = json.loads(result.suggestions) if hasattr(result, 'suggestions') else []
            except (json.JSONDecodeError, TypeError):
                suggestions = []

            return suggestions

        except Exception as e:
            logger.error(f"Improvement analysis failed: {e}")
            return []

    # =========================================================================
    # LEARNING EXTRACTION
    # =========================================================================

    async def extract_learnings(
        self,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        evaluation: Dict[str, Any],
        domain: str = "general"
    ) -> List[str]:
        """
        Extract reusable learnings from an excellent execution.

        Args:
            input_data: Task input parameters
            output_data: Task output
            evaluation: Evaluation data
            domain: Domain of the task

        Returns:
            List of learning strings
        """
        # Only extract from excellent outputs
        if evaluation.get("result") not in ("excellent", "good"):
            return []

        if self._dspy_module is None:
            # Return simple learnings without LLM
            return [
                f"Successful execution in {domain} domain",
                f"Output matched expectations with score {evaluation.get('overall_score', 0):.2f}",
            ]

        try:
            result = self._dspy_module(
                input_data=json.dumps(input_data, default=str),
                output_data=json.dumps(output_data, default=str),
                evaluation_data=json.dumps(evaluation, default=str),
                domain=domain
            )

            learnings = []
            if hasattr(result, 'learnings'):
                learnings = [
                    l.strip() for l in str(result.learnings).split('|')
                    if l.strip()
                ]

            config: MetaAgentConfig = self.config
            return learnings[:config.max_learnings_per_run]

        except Exception as e:
            logger.debug(f"Learning extraction failed: {e}")
            return []

    # =========================================================================
    # CROSS-AGENT VISIBILITY
    # =========================================================================

    def get_agent_state(self, agent_name: str) -> Optional[Dict[str, Any]]:
        """
        Get state of another agent via shared context.

        Args:
            agent_name: Name of the agent to query

        Returns:
            Agent state dict or None
        """
        if self.context is not None:
            states = self.context.get("agent_states", {})
            return states.get(agent_name)
        return None

    def publish_state(self, state: Dict[str, Any]) -> None:
        """
        Publish this agent's state to shared context.

        Args:
            state: State dict to publish
        """
        if self.context is not None:
            states = self.context.get("agent_states", {})
            states[self.config.name] = state
            self.context.set("agent_states", states)

    def get_all_agent_states(self) -> Dict[str, Any]:
        """Get states of all agents."""
        if self.context is not None:
            return self.context.get("agent_states", {})
        return {}

    # =========================================================================
    # DEFAULT IMPLEMENTATION
    # =========================================================================

    async def _execute_impl(self, **kwargs) -> Dict[str, Any]:
        """
        Default implementation using DSPy module.

        Subclasses can override for custom behavior.
        """
        if self._dspy_module is None:
            raise NotImplementedError(
                f"{self.__class__.__name__} requires either a signature "
                "or an overridden _execute_impl method"
            )

        # Execute DSPy module
        import asyncio
        result = await asyncio.to_thread(self._dspy_module, **kwargs)

        # Convert result to dict
        output = {}
        for attr in dir(result):
            if not attr.startswith('_'):
                value = getattr(result, attr, None)
                if value is not None and not callable(value):
                    output[attr] = value

        return output


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_meta_agent(
    signature: Type = None,
    gold_db=None,
    improvement_history=None,
    model: str = "",
) -> MetaAgent:
    """
    Factory function to create a MetaAgent.

    Args:
        signature: Optional DSPy Signature class
        gold_db: GoldStandardDB instance
        improvement_history: ImprovementHistory instance
        model: LLM model to use

    Returns:
        Configured MetaAgent
    """
    from Jotty.core.foundation.config_defaults import DEFAULT_MODEL_ALIAS
    model = model or DEFAULT_MODEL_ALIAS
    name = f"MetaAgent[{signature.__name__}]" if signature else "MetaAgent"
    config = MetaAgentConfig(name=name, model=model)
    return MetaAgent(
        signature=signature,
        config=config,
        gold_db=gold_db,
        improvement_history=improvement_history,
    )


__all__ = [
    'MetaAgent',
    'MetaAgentConfig',
    'create_meta_agent',
]
