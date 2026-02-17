"""
LearningCapability - Mixin for Gold Standard Learning

Adds learning capabilities to any agent, swarm, or workflow:
- Gold standard training data loading
- Domain-specific evaluation
- Optimization pipeline integration
- Learned improvements injection

Usage:
    from Jotty.core.intelligence.reasoning.base import BaseAgent
    from Jotty.core.intelligence.reasoning.capabilities import LearningCapability

    class MermaidAgent(BaseAgent, LearningCapability):
        def __init__(self, config=None, enable_learning=True):
            BaseAgent.__init__(self, config)

            if enable_learning:
                gold_standards = load_gold_standards("mermaid")
                LearningCapability.__init__(
                    self,
                    domain="mermaid",
                    gold_standards=gold_standards,
                    domain_validator=self._validate_diagram
                )

        async def _execute_impl(self, task: str, **kwargs):
            result = await self._generate_diagram(task)

            # Learn and improve if capability enabled
            if hasattr(self, 'learn_from_gold_standards'):
                result = await self.learn_from_gold_standards(task, result)

            return result
"""

import logging
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class LearningCapability:
    """
    Mixin that adds gold standard learning to any execution pattern.

    Provides:
    - Gold standard storage and retrieval
    - Domain validation integration
    - Learned improvements tracking
    - Optimization pipeline hooks

    This mixin can be added to BaseAgent, BaseSwarm, or any workflow.
    """

    def __init__(
        self,
        domain: str,
        gold_standards: Optional[List[Dict[str, Any]]] = None,
        validation_cases: Optional[List[Dict[str, Any]]] = None,
        domain_validator: Optional[Callable] = None,
        improvements: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """
        Initialize learning capability.

        Args:
            domain: Domain name (e.g., "mermaid", "latex", "coding")
            gold_standards: List of gold standard examples for training
            validation_cases: List of validation test cases
            domain_validator: Function to validate domain-specific outputs
            improvements: Previously learned improvements to inject
        """
        self.learning_domain = domain
        self.gold_standards = gold_standards or []
        self.validation_cases = validation_cases or []
        self.domain_validator = domain_validator
        self.learned_improvements = improvements or []

        # Learning statistics
        self._learning_stats = {
            "training_examples": len(self.gold_standards),
            "validation_examples": len(self.validation_cases),
            "improvements_count": len(self.learned_improvements),
            "learning_runs": 0,
            "success_rate": 0.0,
        }

        logger.info(
            f"LearningCapability initialized for domain '{domain}' "
            f"({len(self.gold_standards)} training examples, "
            f"{len(self.learned_improvements)} improvements)"
        )

    def get_gold_standards(self) -> List[Dict[str, Any]]:
        """Get all gold standard training examples."""
        return self.gold_standards

    def add_gold_standard(self, example: Dict[str, Any]) -> None:
        """Add a new gold standard example."""
        self.gold_standards.append(example)
        self._learning_stats["training_examples"] = len(self.gold_standards)
        logger.debug(f"Added gold standard to {self.learning_domain}")

    def get_improvements(self) -> List[Dict[str, Any]]:
        """Get all learned improvements."""
        return self.learned_improvements

    def add_improvement(self, improvement: Dict[str, Any]) -> None:
        """Add a learned improvement."""
        self.learned_improvements.append(improvement)
        self._learning_stats["improvements_count"] = len(self.learned_improvements)
        logger.info(f"Learned new improvement for {self.learning_domain}: {improvement}")

    async def learn_from_gold_standards(self, task: str, current_output: Any, **kwargs) -> Any:
        """
        Learn from gold standards and improve output.

        This is a template method that:
        1. Retrieves relevant gold standards
        2. Compares current output against gold standards
        3. Identifies improvements
        4. Applies improvements to output

        Subclasses can override for domain-specific learning.

        Args:
            task: The input task/query
            current_output: Current output to improve
            **kwargs: Additional context

        Returns:
            Improved output
        """
        self._learning_stats["learning_runs"] += 1

        if not self.gold_standards:
            logger.debug(f"No gold standards available for {self.learning_domain}")
            return current_output

        # Find relevant gold standards (similarity search)
        relevant_examples = self._find_relevant_standards(task)

        if not relevant_examples:
            logger.debug(f"No relevant gold standards found for task: {task[:50]}...")
            return current_output

        # Validate current output
        if self.domain_validator:
            try:
                validation_result = await self.domain_validator(
                    current_output, relevant_examples[0], task, kwargs  # Use most relevant example
                )

                if validation_result.get("valid", True):
                    logger.debug(f"Output validated successfully for {self.learning_domain}")
                else:
                    logger.warning(
                        f"Validation failed for {self.learning_domain}: "
                        f"{validation_result.get('error', 'Unknown error')}"
                    )
            except Exception as e:
                logger.error(f"Domain validation failed: {e}")

        # Apply learned improvements (if any)
        improved_output = self._apply_improvements(current_output, task)

        return improved_output

    def _find_relevant_standards(self, task: str) -> List[Dict[str, Any]]:
        """
        Find gold standards relevant to the task.

        Uses simple keyword matching. Can be overridden for more
        sophisticated similarity search (embeddings, etc.).

        Args:
            task: Input task

        Returns:
            List of relevant gold standard examples
        """
        if not self.gold_standards:
            return []

        # Simple implementation: return all standards
        # TODO: Implement semantic similarity search
        return self.gold_standards[:3]  # Return top 3

    def _apply_improvements(self, output: Any, task: str) -> Any:
        """
        Apply learned improvements to output.

        Args:
            output: Current output
            task: Input task

        Returns:
            Improved output
        """
        if not self.learned_improvements:
            return output

        # Simple implementation: log improvements
        # TODO: Implement actual improvement application logic
        logger.debug(
            f"Applying {len(self.learned_improvements)} improvements "
            f"to {self.learning_domain} output"
        )

        return output

    async def validate_output(
        self,
        output: Any,
        expected: Optional[Any] = None,
        task: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Validate output against domain criteria.

        Args:
            output: Output to validate
            expected: Expected output (optional)
            task: Input task (optional)
            context: Additional context (optional)

        Returns:
            Validation result dict with 'valid', 'score', 'errors'
        """
        if not self.domain_validator:
            return {"valid": True, "score": 1.0, "errors": []}

        try:
            result = await self.domain_validator(output, expected, task or "", context or {})
            return result  # type: ignore[no-any-return]
        except Exception as e:
            logger.error(f"Validation error in {self.learning_domain}: {e}")
            return {"valid": False, "score": 0.0, "errors": [str(e)]}

    def get_learning_stats(self) -> Dict[str, Any]:
        """Get learning statistics."""
        return self._learning_stats.copy()
