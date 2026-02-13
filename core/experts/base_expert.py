"""
BaseExpert - Abstract base class for domain experts (DRY Pattern)

Extracted to eliminate duplicate code across expert implementations.
Provides template methods and common patterns for all expert agents.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class BaseExpert(ABC):
    """
    Abstract base class for all domain expert agents.

    Eliminates duplicate code by providing template methods.
    Subclasses only need to implement domain-specific logic.

    Template Method Pattern:
    - __init__() is provided (calls abstract methods)
    - _create_agent() delegates to _create_domain_agent()
    - _create_teacher() delegates to _create_domain_teacher()
    - _get_training_data() delegates to _get_default_training_cases()
    - _get_validation_data() delegates to _get_default_validation_cases()

    Subclasses must implement:
    - _create_domain_agent() - Create DSPy agent for this domain
    - _create_domain_teacher() - Create teacher agent for this domain
    - _get_default_training_cases() - Return training examples
    - _get_default_validation_cases() - Return validation examples
    - _evaluate_domain() - Evaluate agent output
    - domain property - Return domain name (e.g., "mermaid", "latex")
    """

    def __init__(
        self,
        config=None,
        memory=None,
        improvements: Optional[List[Dict[str, Any]]] = None
    ):
        """
        Initialize base expert with domain-specific configuration.

        Args:
            config: Optional ExpertAgentConfig (will create default if None)
            memory: Optional memory instance
            improvements: Optional list of learned improvements
        """
        # Import here to avoid circular dependency
        from .expert_agent import ExpertAgentConfig

        self.improvements = improvements or []

        # Create default config if not provided
        if config is None:
            config = ExpertAgentConfig(
                name=f"{self.domain}_expert",
                domain=self.domain,
                description=self.description,
                training_gold_standards=self._get_default_training_cases(),
                validation_cases=self._get_default_validation_cases(),
                evaluation_function=self._evaluate_domain,
                agent_module=self._create_agent_wrapper,
                teacher_module=self._create_teacher_wrapper
            )

        # Store config and memory
        self.config = config
        self.memory = memory

        logger.info(f" {self.domain.title()} Expert initialized")

    # =========================================================================
    # ABSTRACT METHODS - Must be implemented by subclasses
    # =========================================================================

    @property
    @abstractmethod
    def domain(self) -> str:
        """
        Domain name for this expert.

        Returns:
            Domain string (e.g., "mermaid", "latex", "plantuml")
        """
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """
        Human-readable description of this expert.

        Returns:
            Description string
        """
        pass

    @abstractmethod
    def _create_domain_agent(self, improvements: Optional[List[Dict[str, Any]]] = None) -> Any:
        """
        Create domain-specific DSPy agent.

        Args:
            improvements: Optional list of learned improvements to inject

        Returns:
            DSPy agent instance
        """
        pass

    @abstractmethod
    def _create_domain_teacher(self) -> Any:
        """
        Create domain-specific teacher agent.

        Returns:
            Teacher agent instance
        """
        pass

    @staticmethod
    @abstractmethod
    def _get_default_training_cases() -> List[Dict[str, Any]]:
        """
        Get default training cases for this domain.

        Returns:
            List of training examples with 'task' and 'gold_standard'
        """
        pass

    @staticmethod
    @abstractmethod
    def _get_default_validation_cases() -> List[Dict[str, Any]]:
        """
        Get default validation cases for this domain.

        Returns:
            List of validation examples with 'task' and 'expected_output'
        """
        pass

    @abstractmethod
    async def _evaluate_domain(
        self,
        output: Any,
        gold_standard: str,
        task: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate agent output against expected result (domain-specific).

        Args:
            output: Agent-generated output
            gold_standard: Expected/gold standard output
            task: Task description
            context: Context dictionary with additional info

        Returns:
            Dict with evaluation results (score, status, is_valid, error, metadata)
        """
        pass

    # =========================================================================
    # TEMPLATE METHODS - Provided by base class (DRY)
    # =========================================================================

    def _create_agent_wrapper(self, improvements: Optional[List[Dict[str, Any]]] = None) -> Any:
        """
        Wrapper for agent creation (template method).

        Delegates to domain-specific implementation.
        """
        return self._create_domain_agent(improvements=improvements or self.improvements)

    def _create_teacher_wrapper(self) -> Any:
        """
        Wrapper for teacher creation (template method).

        Delegates to domain-specific implementation.
        """
        return self._create_domain_teacher()

    def _create_default_agent(self, improvements: Optional[List[Dict[str, Any]]] = None) -> Any:
        """
        Create default agent (required by ExpertAgent interface).

        Delegates to domain-specific implementation.
        This eliminates the duplicate wrapper in each expert subclass.
        """
        return self._create_domain_agent(improvements=improvements)

    def get_training_data(self) -> List[Dict[str, Any]]:
        """
        Get training data for this expert.

        Returns:
            List of training cases
        """
        if self.config and self.config.training_gold_standards:
            return self.config.training_gold_standards
        return self._get_default_training_cases()

    def get_validation_data(self) -> List[Dict[str, Any]]:
        """
        Get validation data for this expert.

        Returns:
            List of validation cases
        """
        if self.config and self.config.validation_cases:
            return self.config.validation_cases
        return self._get_default_validation_cases()

    # =========================================================================
    # COMMON UTILITY METHODS - Shared across all experts (DRY)
    # =========================================================================

    def _is_dspy_available(self) -> bool:
        """Check if DSPy is available."""
        try:
            import dspy
            return True
        except ImportError:
            return False

    def _inject_improvements(self, base_signature, improvements: List[Dict[str, Any]]):
        """
        Inject learned improvements into DSPy signature.

        Args:
            base_signature: Base DSPy signature class
            improvements: List of improvements to inject

        Returns:
            Modified signature class or original if injection fails
        """
        if not improvements:
            return base_signature

        try:
            from .dspy_improvements import inject_improvements_into_signature
            return inject_improvements_into_signature(base_signature, improvements)
        except ImportError:
            logger.warning("Could not inject improvements into signature")
            return base_signature

    def get_stats(self) -> Dict[str, Any]:
        """
        Get expert statistics.

        Returns:
            Dict with expert metrics
        """
        return {
            "expert_type": self.__class__.__name__,
            "domain": self.domain,
            "improvements_count": len(self.improvements),
            "training_cases": len(self.get_training_data()),
            "validation_cases": len(self.get_validation_data())
        }

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(domain={self.domain})"


class SimpleDomainExpert(BaseExpert):
    """
    Simplified base for experts that don't use DSPy.

    For simple rule-based or template-based experts.
    """

    def _create_domain_agent(self, improvements: Optional[List[Dict[str, Any]]] = None) -> Any:
        """Create simple agent (override for custom logic)."""
        return None

    def _create_domain_teacher(self) -> Any:
        """No teacher for simple experts."""
        return None

    @staticmethod
    def _get_default_training_cases() -> List[Dict[str, Any]]:
        """No training for simple experts."""
        return []

    @staticmethod
    def _get_default_validation_cases() -> List[Dict[str, Any]]:
        """No validation for simple experts."""
        return []

    async def _evaluate_domain(
        self,
        output: Any,
        gold_standard: str,
        task: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Simple string comparison for non-DSPy experts."""
        output_str = str(output).strip()
        gold_str = str(gold_standard).strip()
        passed = output_str == gold_str

        return {
            "score": 1.0 if passed else 0.0,
            "status": "CORRECT" if passed else "FAIL",
            "is_valid": passed,
            "error": "" if passed else "Output does not match expected",
            "metadata": {}
        }
