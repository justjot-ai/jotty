"""
ML Skill Base Classes
=====================

Foundation for all ML skills in Jotty.

Design Philosophy:
- Skills are ATOMIC - do ONE thing well
- Skills are STATELESS - no side effects between calls
- Skills are COMPOSABLE - can be chained by templates
- Skills are TESTABLE - clear inputs/outputs

Every skill:
1. Takes inputs (X, y, context)
2. Performs ONE operation
3. Returns SkillResult with data and metrics
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union
from enum import Enum
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class SkillCategory(Enum):
    """Categories of ML skills."""
    DATA_PROFILING = "data_profiling"
    DATA_CLEANING = "data_cleaning"
    FEATURE_ENGINEERING = "feature_engineering"
    FEATURE_SELECTION = "feature_selection"
    MODEL_SELECTION = "model_selection"
    HYPERPARAMETER_OPTIMIZATION = "hyperparameter_optimization"
    ENSEMBLE = "ensemble"
    EVALUATION = "evaluation"
    EXPLANATION = "explanation"
    LLM_REASONING = "llm_reasoning"


@dataclass
class SkillResult:
    """
    Result from skill execution.

    All skills return this standardized result format.
    """
    skill_name: str
    category: SkillCategory
    success: bool

    # Main output data
    data: Any = None

    # Metrics (e.g., score, n_features)
    metrics: Dict[str, float] = field(default_factory=dict)

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Error information
    error: Optional[str] = None

    # Execution time
    execution_time_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'skill_name': self.skill_name,
            'category': self.category.value,
            'success': self.success,
            'metrics': self.metrics,
            'metadata': self.metadata,
            'error': self.error,
            'execution_time_seconds': self.execution_time_seconds,
        }


class MLSkill(ABC):
    """
    Base class for ML skills.

    Subclass this to create new skills:

    class MySkill(MLSkill):
        name = "my_skill"
        category = SkillCategory.FEATURE_ENGINEERING

        async def execute(self, X, y, **context) -> SkillResult:
            # Do something
            return SkillResult(...)
    """

    # Skill metadata (override in subclass)
    name: str = "base_skill"
    version: str = "1.0.0"
    description: str = "Base ML skill"
    category: SkillCategory = SkillCategory.DATA_PROFILING

    # Input/output schema (for validation)
    required_inputs: List[str] = ["X"]
    optional_inputs: List[str] = ["y"]
    outputs: List[str] = ["data"]

    # Resource hints
    requires_llm: bool = False
    requires_gpu: bool = False
    estimated_memory_gb: float = 1.0

    def __init__(self, config: Dict[str, Any] = None) -> None:
        """
        Initialize skill with optional configuration.

        Args:
            config: Skill-specific configuration
        """
        self.config = config or {}
        self._initialized = False

    async def init(self) -> Any:
        """
        Initialize any resources needed by the skill.

        Override in subclass if needed.
        """
        self._initialized = True

    @abstractmethod
    async def execute(self, X: pd.DataFrame, y: Optional[pd.Series] = None, **context: Any) -> SkillResult:
        """
        Execute the skill.

        Args:
            X: Input features DataFrame
            y: Target variable (optional for some skills)
            **context: Additional context (problem_type, business_context, etc.)

        Returns:
            SkillResult with output data and metrics
        """
        pass

    def validate_inputs(self, X: Any, y: Any = None, **context: Any) -> bool:
        """
        Validate inputs before execution.

        Override in subclass for custom validation.
        """
        if X is None:
            return False

        if not isinstance(X, (pd.DataFrame, np.ndarray)):
            return False

        return True

    def _create_result(self,
                       success: bool,
                       data: Any = None,
                       metrics: Dict = None,
                       metadata: Dict = None,
                       error: str = None,
                       execution_time: float = 0.0) -> SkillResult:
        """Helper to create standardized result."""
        return SkillResult(
            skill_name=self.name,
            category=self.category,
            success=success,
            data=data,
            metrics=metrics or {},
            metadata=metadata or {},
            error=error,
            execution_time_seconds=execution_time,
        )

    def _create_error_result(self, error: str) -> SkillResult:
        """Helper to create error result."""
        return self._create_result(
            success=False,
            error=error,
        )

    def get_info(self) -> Dict[str, Any]:
        """Get skill information."""
        return {
            'name': self.name,
            'version': self.version,
            'description': self.description,
            'category': self.category.value,
            'required_inputs': self.required_inputs,
            'optional_inputs': self.optional_inputs,
            'outputs': self.outputs,
            'requires_llm': self.requires_llm,
            'requires_gpu': self.requires_gpu,
        }


class SkillPipeline:
    """
    Execute multiple skills in sequence.

    Handles data flow between skills automatically.
    """

    def __init__(self, skills: List[MLSkill]) -> None:
        self.skills = skills
        self._results: List[SkillResult] = []

    async def execute(self, X: pd.DataFrame, y: pd.Series = None, **context: Any) -> List[SkillResult]:
        """
        Execute all skills in sequence.

        Args:
            X: Initial input features
            y: Target variable
            **context: Additional context

        Returns:
            List of SkillResult from each skill
        """
        self._results = []
        current_X = X

        for skill in self.skills:
            # Initialize skill if needed
            if not skill._initialized:
                await skill.init()

            # Execute skill
            result = await skill.execute(current_X, y, **context)
            self._results.append(result)

            # If skill produced new data, use it for next skill
            if result.success and result.data is not None:
                if isinstance(result.data, pd.DataFrame):
                    current_X = result.data

            # Update context with skill metadata
            context[f'{skill.name}_result'] = result

        return self._results

    def get_final_result(self) -> Optional[SkillResult]:
        """Get the result from the last skill."""
        return self._results[-1] if self._results else None

    def get_all_metrics(self) -> Dict[str, Dict]:
        """Aggregate metrics from all skills."""
        return {
            result.skill_name: result.metrics
            for result in self._results
            if result.success
        }


class SkillRegistry:
    """
    Registry of available ML skills.

    Allows discovery and instantiation of skills by name.
    """

    _skills: Dict[str, type] = {}

    @classmethod
    def register(cls, skill_class: type) -> None:
        """Register a skill class."""
        cls._skills[skill_class.name] = skill_class
        logger.debug(f"Registered skill: {skill_class.name}")

    @classmethod
    def get(cls, name: str, config: Dict = None) -> MLSkill:
        """Get a skill instance by name."""
        if name not in cls._skills:
            raise KeyError(f"Skill '{name}' not found. Available: {list(cls._skills.keys())}")
        return cls._skills[name](config)

    @classmethod
    def list_skills(cls) -> List[Dict]:
        """List all registered skills."""
        return [
            skill_class({}).get_info()
            for skill_class in cls._skills.values()
        ]

    @classmethod
    def get_by_category(cls, category: SkillCategory) -> List[str]:
        """Get skill names by category."""
        return [
            name for name, skill_class in cls._skills.items()
            if skill_class.category == category
        ]
