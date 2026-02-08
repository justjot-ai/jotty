"""
DomainAgent - Single-Task Executor with DSPy Signatures

Wraps DSPy ChainOfThought or Predict modules for domain-specific tasks.
Provides:
- Automatic input/output field extraction from signatures
- Streaming support with progress callbacks
- Timeout handling
- Result parsing and validation

Usage:
    class MySignature(dspy.Signature):
        task: str = dspy.InputField()
        result: str = dspy.OutputField()

    agent = DomainAgent(MySignature)
    result = await agent.execute(task="do something")
    print(result.output)  # {'result': '...'}

Author: A-Team
Date: February 2026
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Type, Union

from .base_agent import BaseAgent, AgentConfig

logger = logging.getLogger(__name__)


# =============================================================================
# DOMAIN AGENT CONFIG
# =============================================================================

@dataclass
class DomainAgentConfig(AgentConfig):
    """Configuration specific to DomainAgent."""
    use_chain_of_thought: bool = True
    streaming: bool = False
    progress_callback: Optional[Callable[[str, float], None]] = None


# =============================================================================
# DOMAIN AGENT
# =============================================================================

class DomainAgent(BaseAgent):
    """
    Single-task executor that wraps a DSPy signature.

    Automatically extracts input/output fields from the signature and
    creates a ChainOfThought or Predict module.

    Example:
        class AnalysisSignature(dspy.Signature):
            '''Analyze the given text.'''
            text: str = dspy.InputField(desc="Text to analyze")
            analysis: str = dspy.OutputField(desc="Analysis result")
            confidence: float = dspy.OutputField(desc="Confidence score")

        agent = DomainAgent(AnalysisSignature)
        result = await agent.execute(text="Hello world")
        # result.output = {'analysis': '...', 'confidence': 0.95}
    """

    def __init__(
        self,
        signature: Type,
        config: DomainAgentConfig = None,
    ):
        """
        Initialize DomainAgent with a DSPy signature.

        Args:
            signature: DSPy Signature class defining inputs and outputs
            config: Optional configuration
        """
        config = config or DomainAgentConfig(
            name=f"DomainAgent[{signature.__name__}]"
        )
        super().__init__(config)

        self.signature = signature
        self._module = None

        # Auto-extract field names from signature
        self._input_fields: List[str] = []
        self._output_fields: List[str] = []
        self._extract_fields()

    def _extract_fields(self):
        """Extract input and output field names from the signature."""
        try:
            import dspy

            # Get fields from signature
            if hasattr(self.signature, 'model_fields'):
                # Pydantic-style signature
                for name, field_info in self.signature.model_fields.items():
                    # Check if it's an input or output field
                    if hasattr(field_info, 'json_schema_extra'):
                        extra = field_info.json_schema_extra or {}
                        if extra.get('__dspy_field_type') == 'input':
                            self._input_fields.append(name)
                        elif extra.get('__dspy_field_type') == 'output':
                            self._output_fields.append(name)
                    else:
                        # Fallback: check the field type
                        if isinstance(field_info.default, dspy.InputField):
                            self._input_fields.append(name)
                        elif isinstance(field_info.default, dspy.OutputField):
                            self._output_fields.append(name)

            # Fallback: scan class attributes
            if not self._input_fields and not self._output_fields:
                for name in dir(self.signature):
                    if name.startswith('_'):
                        continue
                    attr = getattr(self.signature, name, None)
                    if isinstance(attr, dspy.InputField):
                        self._input_fields.append(name)
                    elif isinstance(attr, dspy.OutputField):
                        self._output_fields.append(name)

            # Last resort: use signature's input_fields and output_fields
            if not self._input_fields:
                if hasattr(self.signature, 'input_fields'):
                    self._input_fields = list(self.signature.input_fields.keys())
            if not self._output_fields:
                if hasattr(self.signature, 'output_fields'):
                    self._output_fields = list(self.signature.output_fields.keys())

            logger.debug(
                f"Extracted fields - inputs: {self._input_fields}, "
                f"outputs: {self._output_fields}"
            )

        except Exception as e:
            logger.warning(f"Failed to extract fields from signature: {e}")

    def _ensure_initialized(self):
        """Initialize DSPy module."""
        super()._ensure_initialized()

        # Only create module if signature was provided
        # Swarm agents may provide signature=None and create their own modules
        if self._module is None and self.signature is not None:
            try:
                import dspy

                config: DomainAgentConfig = self.config
                if config.use_chain_of_thought:
                    self._module = dspy.ChainOfThought(self.signature)
                else:
                    self._module = dspy.Predict(self.signature)

                logger.debug(
                    f"Initialized {'ChainOfThought' if config.use_chain_of_thought else 'Predict'} "
                    f"module for {self.signature.__name__}"
                )
            except Exception as e:
                logger.error(f"Failed to initialize DSPy module: {e}")
                raise

    async def _execute_impl(self, **kwargs) -> Dict[str, Any]:
        """
        Execute the DSPy signature with the provided inputs.

        Args:
            **kwargs: Input field values matching the signature

        Returns:
            Dict with output field values
        """
        config: DomainAgentConfig = self.config

        # Filter inputs to only include signature input fields
        inputs = {
            k: v for k, v in kwargs.items()
            if k in self._input_fields
        }

        # Validate required inputs
        missing = [f for f in self._input_fields if f not in inputs]
        if missing:
            logger.warning(f"Missing input fields: {missing}")

        # Execute with timeout
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(self._module, **inputs),
                timeout=config.timeout
            )

            # Report progress if callback provided
            if config.progress_callback:
                config.progress_callback("completed", 1.0)

            # Extract output fields
            output = {}
            for field_name in self._output_fields:
                value = getattr(result, field_name, None)
                if value is not None:
                    output[field_name] = value

            # Include reasoning if ChainOfThought
            if hasattr(result, 'reasoning') and result.reasoning:
                output['_reasoning'] = result.reasoning

            return output

        except asyncio.TimeoutError:
            raise TimeoutError(
                f"DSPy execution timed out after {config.timeout}s"
            )

    @property
    def input_fields(self) -> List[str]:
        """Get list of input field names."""
        return self._input_fields.copy()

    @property
    def output_fields(self) -> List[str]:
        """Get list of output field names."""
        return self._output_fields.copy()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_domain_agent(
    signature: Type,
    use_chain_of_thought: bool = True,
    model: str = "sonnet",
    timeout: float = 120.0,
) -> DomainAgent:
    """
    Factory function to create a DomainAgent.

    Args:
        signature: DSPy Signature class
        use_chain_of_thought: Use ChainOfThought (True) or Predict (False)
        model: LLM model to use
        timeout: Execution timeout

    Returns:
        Configured DomainAgent
    """
    config = DomainAgentConfig(
        name=f"DomainAgent[{signature.__name__}]",
        model=model,
        timeout=timeout,
        use_chain_of_thought=use_chain_of_thought,
    )
    return DomainAgent(signature, config)


__all__ = [
    'DomainAgent',
    'DomainAgentConfig',
    'create_domain_agent',
]
