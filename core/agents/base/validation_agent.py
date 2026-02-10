"""
ValidationAgent - Base Class for Validation/Inspection Agents
==============================================================

Specialized MetaAgent for pre/post validation tasks:
- InspectorAgent (Architect/Auditor)
- VerificationAgents
- QA/Review Agents

Provides:
- Multi-round validation support
- Shared scratchpad for inter-agent communication
- Tool caching to prevent redundant calls
- Memory-based learning from past validations
- Configurable validation thresholds

Author: A-Team
Date: February 2026
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type

from .meta_agent import MetaAgent, MetaAgentConfig

from Jotty.core.foundation.types.enums import OutputTag, ValidationRound
from Jotty.core.foundation.types.validation_types import ValidationResult

logger = logging.getLogger(__name__)


# =============================================================================
# VALIDATION TYPES
# =============================================================================


@dataclass
class ValidationConfig(MetaAgentConfig):
    """Configuration for validation agents."""
    # Validation thresholds
    confidence_threshold: float = 0.7
    refinement_threshold: float = 0.5

    # Multi-round settings
    enable_multi_round: bool = True
    max_validation_rounds: int = 2
    refinement_on_disagreement: bool = True
    refinement_on_low_confidence: float = 0.5

    # Timeouts
    llm_timeout_seconds: float = 180.0

    # Memory settings
    store_validation_history: bool = True

    # Default values for errors
    default_confidence_on_error: float = 0.3
    default_confidence_no_validation: float = 0.5


# =============================================================================
# SHARED SCRATCHPAD
# =============================================================================

@dataclass
class AgentMessage:
    """Message exchanged between agents."""
    sender: str
    receiver: str  # "*" for broadcast
    message_type: str  # "insight", "warning", "tool_result"
    content: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)

    # Optional fields
    insight: str = ""
    confidence: float = 0.0
    tool_name: str = ""
    tool_args: Dict = field(default_factory=dict)
    tool_result: Any = None


class SharedScratchpad:
    """
    Shared scratchpad for inter-agent communication.

    Enables agents to:
    - Share insights and warnings
    - Cache tool results
    - Coordinate validation decisions
    """

    def __init__(self):
        self.messages: List[AgentMessage] = []
        self._tool_cache: Dict[str, Any] = {}

    def add_message(self, message: AgentMessage):
        """Add a message to the scratchpad."""
        self.messages.append(message)

    def get_messages_for(self, agent_name: str) -> List[AgentMessage]:
        """Get messages intended for a specific agent or broadcast."""
        return [
            m for m in self.messages
            if m.receiver == "*" or m.receiver == agent_name
        ]

    def get_cached_result(self, tool_name: str, args: Dict) -> Optional[Any]:
        """Get cached tool result if available."""
        cache_key = f"{tool_name}:{json.dumps(args, sort_keys=True, default=str)}"
        return self._tool_cache.get(cache_key)

    def cache_result(self, tool_name: str, args: Dict, result: Any):
        """Cache a tool result."""
        cache_key = f"{tool_name}:{json.dumps(args, sort_keys=True, default=str)}"
        self._tool_cache[cache_key] = result

    def clear(self):
        """Clear the scratchpad."""
        self.messages.clear()
        self._tool_cache.clear()


# =============================================================================
# VALIDATION AGENT BASE CLASS
# =============================================================================

class ValidationAgent(MetaAgent):
    """
    Base class for validation/inspection agents.

    Extends MetaAgent with:
    - Multi-round validation support
    - Shared scratchpad for coordination
    - Tool caching
    - Validation-specific metrics

    Subclasses (InspectorAgent, etc.) implement specific validation logic.

    Usage:
        class InspectorAgent(ValidationAgent):
            async def validate(self, goal, inputs, trajectory):
                # Custom validation logic
                result = await self._run_validation(goal, inputs)
                return result
    """

    def __init__(
        self,
        config: ValidationConfig = None,
        scratchpad: SharedScratchpad = None,
        is_pre_validation: bool = True,
    ):
        """
        Initialize ValidationAgent.

        Args:
            config: Validation configuration
            scratchpad: Shared scratchpad for inter-agent communication
            is_pre_validation: True for Architect (pre), False for Auditor (post)
        """
        config = config or ValidationConfig(name=self.__class__.__name__)
        super().__init__(config=config)

        self.validation_config: ValidationConfig = config
        self.scratchpad = scratchpad or SharedScratchpad()
        self.is_pre_validation = is_pre_validation

        # Validation-specific metrics
        self._validation_metrics = {
            "total_validations": 0,
            "approvals": 0,
            "rejections": 0,
            "refinements": 0,
            "avg_confidence": 0.0,
        }

        # Cached tools
        self._cached_tools: List[Any] = []

    def _ensure_initialized(self):
        """Initialize validation-specific resources."""
        super()._ensure_initialized()

        # Initialize DSPy module for validation if signature provided
        if self._dspy_module is None and self.signature is not None:
            try:
                import dspy
                self._dspy_module = dspy.ChainOfThought(self.signature)
                logger.debug(f"Initialized validation module for {self.config.name}")
            except Exception as e:
                logger.warning(f"Failed to initialize validation module: {e}")

    # =========================================================================
    # INTER-AGENT COMMUNICATION
    # =========================================================================

    def share_insight(self, insight: str, confidence: float = 0.7):
        """Share an insight with other agents."""
        message = AgentMessage(
            sender=self.config.name,
            receiver="*",
            message_type="insight",
            content={"insight": insight},
            insight=insight,
            confidence=confidence,
        )
        self.scratchpad.add_message(message)

    def share_warning(self, warning: str):
        """Share a warning with other agents."""
        message = AgentMessage(
            sender=self.config.name,
            receiver="*",
            message_type="warning",
            content={"warning": warning},
        )
        self.scratchpad.add_message(message)

    def get_shared_insights(self) -> List[str]:
        """Get insights shared by other agents."""
        messages = self.scratchpad.get_messages_for(self.config.name)
        return [
            m.insight for m in messages
            if m.message_type == "insight" and m.insight
        ]

    # =========================================================================
    # TOOL CACHING
    # =========================================================================

    def wrap_tool_with_cache(self, tool: Any) -> Callable:
        """Wrap a tool with caching logic."""
        tool_name = getattr(tool, 'name', str(tool))

        def cached_tool(**kwargs) -> Any:
            # Check cache
            cached = self.scratchpad.get_cached_result(tool_name, kwargs)
            if cached is not None:
                return cached

            # Execute and cache
            if callable(tool):
                result = tool(**kwargs)
            else:
                result = tool

            self.scratchpad.cache_result(tool_name, kwargs, result)

            # Record in scratchpad
            message = AgentMessage(
                sender=self.config.name,
                receiver="*",
                message_type="tool_result",
                content={"tool": tool_name, "args": str(kwargs)},
                tool_name=tool_name,
                tool_args=kwargs,
                tool_result=result,
            )
            self.scratchpad.add_message(message)

            return result

        return cached_tool

    # =========================================================================
    # VALIDATION HELPERS
    # =========================================================================

    def _build_memory_context(self, goal: str, inputs: Dict) -> str:
        """Build context from memory for validation."""
        if self.memory is None:
            return ""

        query = inputs.get('proposed_action', '') or inputs.get('action_result', '') or goal

        try:
            memories = self.memory.retrieve(
                query=query,
                goal=goal,
                budget_tokens=2000
            )

            if not memories:
                return ""

            parts = []
            for mem in memories:
                value = mem.get_value(goal) if hasattr(mem, 'get_value') else 0.5
                content = mem.content[:500] if len(mem.content) > 500 else mem.content
                parts.append(f"[Value: {value:.2f}] {content}")

            return "\n---\n".join(parts)
        except Exception as e:
            logger.warning(f"Failed to build memory context: {e}")
            return ""

    def _store_validation_result(self, goal: str, inputs: Dict, result: ValidationResult):
        """Store validation result in memory for learning."""
        if self.memory is None or not self.validation_config.store_validation_history:
            return

        try:
            from Jotty.core.foundation.data_structures import MemoryLevel

            decision = "PROCEED" if result.should_proceed else "BLOCK"
            if not self.is_pre_validation:
                decision = "VALID" if result.is_valid else "INVALID"

            content = f"""
Validation Decision: {decision}
Confidence: {result.confidence:.2f}
Agent: {result.agent_name}
Reasoning: {result.reasoning[:500]}
""".strip()

            self.memory.store(
                content=content,
                level=MemoryLevel.EPISODIC,
                context={
                    'goal': goal,
                    'round': result.validation_round.value,
                    'is_valid': result.is_valid,
                    'confidence': result.confidence,
                },
                goal=goal,
                initial_value=0.5 + 0.3 * (result.confidence - 0.5)
            )
        except Exception as e:
            logger.warning(f"Failed to store validation result: {e}")

    def _update_validation_metrics(self, result: ValidationResult):
        """Update validation-specific metrics."""
        self._validation_metrics["total_validations"] += 1

        if self.is_pre_validation:
            if result.should_proceed:
                self._validation_metrics["approvals"] += 1
            else:
                self._validation_metrics["rejections"] += 1
        else:
            if result.is_valid:
                self._validation_metrics["approvals"] += 1
            else:
                self._validation_metrics["rejections"] += 1

        # Update average confidence
        total = self._validation_metrics["total_validations"]
        prev_avg = self._validation_metrics["avg_confidence"]
        self._validation_metrics["avg_confidence"] = (
            (prev_avg * (total - 1) + result.confidence) / total
        )

    # =========================================================================
    # MULTI-ROUND VALIDATION
    # =========================================================================

    def needs_refinement(self, result: ValidationResult) -> bool:
        """Check if result needs refinement round."""
        if not self.validation_config.enable_multi_round:
            return False

        return result.confidence < self.validation_config.refinement_on_low_confidence

    async def refine_result(
        self,
        original: ValidationResult,
        feedback: str,
        additional_context: str = ""
    ) -> ValidationResult:
        """Refine a validation result based on feedback."""
        if not self.validation_config.enable_multi_round:
            return original

        self._validation_metrics["refinements"] += 1

        # Default: return original if no refinement module
        # Subclasses should override with actual refinement logic
        return original

    # =========================================================================
    # METRICS
    # =========================================================================

    def get_validation_metrics(self) -> Dict[str, Any]:
        """Get validation-specific metrics."""
        base_metrics = self.get_metrics()

        metrics = self._validation_metrics.copy()
        if metrics["total_validations"] > 0:
            metrics["approval_rate"] = (
                metrics["approvals"] / metrics["total_validations"]
            )
        else:
            metrics["approval_rate"] = 0.0

        return {**base_metrics, **metrics}

    # =========================================================================
    # DEFAULT IMPLEMENTATION
    # =========================================================================

    async def _execute_impl(self, **kwargs) -> Dict[str, Any]:
        """
        Default validation execution.

        Subclasses should override validate() or this method.
        """
        goal = kwargs.get('goal', '')
        inputs = kwargs.get('inputs', {})
        trajectory = kwargs.get('trajectory', [])

        result = await self.validate(goal, inputs, trajectory)

        return result.to_dict()

    async def validate(
        self,
        goal: str,
        inputs: Dict[str, Any],
        trajectory: List[Dict] = None
    ) -> ValidationResult:
        """
        Run validation.

        Override this method for custom validation logic.

        Args:
            goal: Goal being validated
            inputs: Input data for validation
            trajectory: Execution trajectory so far

        Returns:
            ValidationResult with decision and metadata
        """
        start_time = time.time()
        trajectory = trajectory or []

        # Build context
        memory_context = self._build_memory_context(goal, inputs)
        shared_insights = self.get_shared_insights()

        # Default validation using DSPy module
        if self._dspy_module is not None:
            try:
                task = inputs.get('task', goal)
                context = f"{memory_context}\n\nInsights: {shared_insights}"

                result = await asyncio.wait_for(
                    asyncio.to_thread(
                        self._dspy_module,
                        task=task,
                        context=context
                    ),
                    timeout=self.validation_config.llm_timeout_seconds
                )

                # Parse result
                is_valid = getattr(result, 'is_valid', True)
                if isinstance(is_valid, str):
                    is_valid = is_valid.lower() in ('true', 'yes', '1')

                should_proceed = getattr(result, 'should_proceed', True)
                if isinstance(should_proceed, str):
                    should_proceed = should_proceed.lower() in ('true', 'yes', '1')

                confidence = float(getattr(result, 'confidence', 0.5))
                confidence = max(0.0, min(1.0, confidence))

                validation_result = ValidationResult(
                    agent_name=self.config.name,
                    is_valid=is_valid if not self.is_pre_validation else should_proceed,
                    confidence=confidence,
                    reasoning=getattr(result, 'reasoning', ''),
                    should_proceed=should_proceed if self.is_pre_validation else None,
                    execution_time=time.time() - start_time,
                )

            except asyncio.TimeoutError:
                validation_result = ValidationResult(
                    agent_name=self.config.name,
                    is_valid=True,  # Default to proceed on timeout
                    confidence=self.validation_config.default_confidence_on_error,
                    reasoning="Validation timed out - defaulting to proceed",
                    should_proceed=True if self.is_pre_validation else None,
                    execution_time=time.time() - start_time,
                )
            except Exception as e:
                logger.error(f"Validation failed: {e}")
                validation_result = ValidationResult(
                    agent_name=self.config.name,
                    is_valid=True,
                    confidence=self.validation_config.default_confidence_on_error,
                    reasoning=f"Validation error: {e}",
                    should_proceed=True if self.is_pre_validation else None,
                    execution_time=time.time() - start_time,
                )
        else:
            # No DSPy module - default to pass
            validation_result = ValidationResult(
                agent_name=self.config.name,
                is_valid=True,
                confidence=self.validation_config.default_confidence_no_validation,
                reasoning="No validation module configured",
                should_proceed=True if self.is_pre_validation else None,
                execution_time=time.time() - start_time,
            )

        # Store and track
        self._store_validation_result(goal, inputs, validation_result)
        self._update_validation_metrics(validation_result)

        return validation_result


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_validation_agent(
    signature: Type = None,
    is_pre_validation: bool = True,
    model: str = "",
) -> ValidationAgent:
    """
    Factory function to create a ValidationAgent.

    Args:
        signature: Optional DSPy Signature class
        is_pre_validation: True for pre-validation (Architect), False for post (Auditor)
        model: LLM model to use

    Returns:
        Configured ValidationAgent
    """
    from Jotty.core.foundation.config_defaults import DEFAULT_MODEL_ALIAS
    model = model or DEFAULT_MODEL_ALIAS
    name = f"ValidationAgent[{signature.__name__}]" if signature else "ValidationAgent"
    config = ValidationConfig(name=name, model=model)

    agent = ValidationAgent(
        config=config,
        is_pre_validation=is_pre_validation,
    )
    agent.signature = signature

    return agent


__all__ = [
    'ValidationAgent',
    'ValidationConfig',
    'ValidationResult',
    'ValidationRound',
    'OutputTag',
    'SharedScratchpad',
    'AgentMessage',
    'create_validation_agent',
]
