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

from .base_agent import AgentRuntimeConfig, BaseAgent

logger = logging.getLogger(__name__)


# =============================================================================
# DOMAIN AGENT CONFIG
# =============================================================================


@dataclass
class DomainAgentConfig(AgentRuntimeConfig):
    """Configuration specific to DomainAgent."""

    use_chain_of_thought: bool = True
    use_react: bool = False  # ReAct loop with tools (multi-step)
    max_react_iters: int = 5  # Max ReAct iterations before stopping
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

    def __init__(self, signature: Optional[Type] = None, config: DomainAgentConfig = None) -> None:
        """
        Initialize DomainAgent with an optional DSPy signature.

        When signature is None, the agent relies on skill-based fallback
        for execution (via SkillPlanExecutor).

        Args:
            signature: DSPy Signature class defining inputs and outputs, or None
            config: Optional configuration
        """
        sig_name = signature.__name__ if signature is not None else "NoSignature"
        config = config or DomainAgentConfig(name=f"DomainAgent[{sig_name}]")
        super().__init__(config)

        self.signature = signature
        self._module = None
        self._skill_executor = None

        # Auto-extract field names from signature
        self._input_fields: List[str] = []
        self._output_fields: List[str] = []
        if self.signature is not None:
            self._extract_fields()

    def _extract_fields(self) -> Any:
        """Extract input and output field names from the signature."""
        try:
            import dspy

            # Get fields from signature
            if hasattr(self.signature, "model_fields"):
                # Pydantic-style signature
                for name, field_info in self.signature.model_fields.items():
                    # Check if it's an input or output field
                    if hasattr(field_info, "json_schema_extra"):
                        extra = field_info.json_schema_extra or {}
                        if extra.get("__dspy_field_type") == "input":
                            self._input_fields.append(name)
                        elif extra.get("__dspy_field_type") == "output":
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
                    if name.startswith("_"):
                        continue
                    attr = getattr(self.signature, name, None)
                    if isinstance(attr, dspy.InputField):
                        self._input_fields.append(name)
                    elif isinstance(attr, dspy.OutputField):
                        self._output_fields.append(name)

            # Last resort: use signature's input_fields and output_fields
            if not self._input_fields:
                if hasattr(self.signature, "input_fields"):
                    self._input_fields = list(self.signature.input_fields.keys())
            if not self._output_fields:
                if hasattr(self.signature, "output_fields"):
                    self._output_fields = list(self.signature.output_fields.keys())

            logger.debug(
                f"Extracted fields - inputs: {self._input_fields}, "
                f"outputs: {self._output_fields}"
            )

        except Exception as e:
            logger.warning(f"Failed to extract fields from signature: {e}")

    def _ensure_initialized(self) -> Any:
        """Initialize DSPy module."""
        super()._ensure_initialized()

        # Only create module if signature was provided
        # Swarm agents may provide signature=None and create their own modules
        if self._module is None and self.signature is not None:
            try:
                import dspy

                config: DomainAgentConfig = self.config
                if config.use_react:
                    # ReAct mode: multi-step tool-using agent loop
                    tools = self._get_react_tools()
                    if hasattr(dspy, "ReAct"):
                        self._module = dspy.ReAct(
                            self.signature,
                            tools=tools,
                            max_iters=config.max_react_iters,
                        )
                    else:
                        # DSPy version without ReAct — fall back to ChainOfThought
                        logger.warning("dspy.ReAct not available, falling back to ChainOfThought")
                        self._module = dspy.ChainOfThought(self.signature)
                elif config.use_chain_of_thought:
                    self._module = dspy.ChainOfThought(self.signature)
                else:
                    self._module = dspy.Predict(self.signature)

                mode = (
                    "ReAct"
                    if config.use_react
                    else ("ChainOfThought" if config.use_chain_of_thought else "Predict")
                )
                logger.debug(f"Initialized {mode} module for {self.signature.__name__}")
            except Exception as e:
                logger.error(f"Failed to initialize DSPy module: {e}")
                raise

    def _enrich_module_for_call(
        self, learning_context: Optional[str] = None, workspace_dir: Optional[str] = None
    ) -> Any:
        """
        Enrich the DSPy module's signature instructions using PromptComposer.

        This is the bridge between PromptComposer and DSPy:
        - Takes the signature's existing docstring (instructions)
        - Enriches it with learning context, project rules, trust levels
        - Formats for the model family (XML for Claude, Markdown for GPT, etc.)
        - Returns a temporary module with enriched instructions

        If no extra context is available, returns self._module unchanged (zero cost).
        """
        if not learning_context and not workspace_dir:
            return self._module

        try:
            import dspy

            from Jotty.core.capabilities.prompts import PromptComposer

            # Get existing signature instructions (docstring)
            base_instructions = getattr(self.signature, "instructions", "") or ""

            # Detect model for family-aware formatting
            model = getattr(self.config, "model", "") or ""

            composer = PromptComposer(model=model)

            # Build enriched instructions: identity=base_instructions + learning + rules
            enriched = composer.compose(
                identity=base_instructions,
                learning_context=learning_context.split("\n\n") if learning_context else None,
                workspace_dir=workspace_dir,
            )

            # Create enriched signature + module
            enriched_sig = self.signature.with_instructions(enriched)
            config: DomainAgentConfig = self.config
            if config.use_chain_of_thought:
                return dspy.ChainOfThought(enriched_sig)
            else:
                return dspy.Predict(enriched_sig)

        except Exception as e:
            logger.debug(f"Module enrichment skipped: {e}")
            return self._module

    def _get_react_tools(self) -> list:
        """Build DSPy-compatible tool list from skills registry for ReAct mode.

        Returns list of callables that DSPy ReAct can use as tools.
        Empty list if no skills registry available.
        """
        if self.skills_registry is None:
            return []
        try:
            tools = []
            for skill_dict in self.skills_registry.list_skills():
                skill_name = skill_dict.get("name", "")
                skill_def = self.skills_registry.get_skill(skill_name)
                if skill_def and hasattr(skill_def, "tool_function") and skill_def.tool_function:
                    tools.append(skill_def.tool_function)
            return tools[:10]  # Cap at 10 tools to avoid context bloat
        except Exception as e:
            logger.debug(f"Could not build ReAct tools: {e}")
            return []

    async def _execute_impl(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Execute the DSPy signature with the provided inputs.

        Falls back to skill-based planning when:
        - self._module is None (no signature or initialization failed)
        - Required input fields are not present in kwargs
        - DSPy execution fails

        Args:
            **kwargs: Input field values matching the signature

        Returns:
            Dict with output field values
        """
        config: DomainAgentConfig = self.config

        # Fast path: no module means immediate fallback
        if self._module is None:
            task = kwargs.get("task", "") or self._build_task_from_kwargs(kwargs)
            if task:
                logger.info(
                    f"No DSPy module available, falling back to skill execution for: {task[:80]}"
                )
                return await self._execute_with_skills(task, **kwargs)
            # No module and no task — nothing we can do
            return {"error": "No DSPy module and no task provided", "success": False}

        # Filter inputs to only include signature input fields
        inputs = {k: v for k, v in kwargs.items() if k in self._input_fields}

        # Check if required inputs are present — fallback if not
        missing = [f for f in self._input_fields if f not in inputs]
        if missing and not inputs:
            task = kwargs.get("task", "") or self._build_task_from_kwargs(kwargs)
            if task:
                logger.info(
                    f"Missing all input fields {missing}, "
                    f"falling back to skill execution for: {task[:80]}"
                )
                return await self._execute_with_skills(task, **kwargs)

        if missing:
            logger.warning(f"Missing input fields: {missing}")

        # Enrich DSPy signature instructions with PromptComposer context.
        # This is the bridge: PromptComposer → DSPy → LLM call.
        # learning_context and workspace_dir flow from AgentRunner kwargs.
        _module = self._enrich_module_for_call(
            kwargs.get("learning_context"),
            kwargs.get("workspace_dir"),
        )

        # Primary path: execute DSPy signature
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(_module, **inputs), timeout=config.timeout
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
            if hasattr(result, "reasoning") and result.reasoning:
                output["_reasoning"] = result.reasoning

            return output

        except asyncio.TimeoutError:
            raise TimeoutError(f"DSPy execution timed out after {config.timeout}s")
        except Exception as e:
            # DSPy execution failed — attempt skill fallback
            task = kwargs.get("task", "") or self._build_task_from_kwargs(kwargs)
            if task:
                logger.warning(
                    f"DSPy execution failed ({e}), "
                    f"falling back to skill execution for: {task[:80]}"
                )
                return await self._execute_with_skills(task, **kwargs)
            raise

    # =========================================================================
    # SKILL-BASED FALLBACK
    # =========================================================================

    async def _execute_with_skills(self, task: str, **kwargs: Any) -> Dict[str, Any]:
        """
        Execute a task using skill discovery and planning.

        Lazy-creates a SkillPlanExecutor and delegates the full pipeline:
        discover skills -> select -> plan -> execute steps.

        Args:
            task: Task description
            **kwargs: Additional arguments (status_callback, etc.)

        Returns:
            Dict with execution results
        """
        if self._skill_executor is None:
            if self.skills_registry is None:
                return {
                    "success": False,
                    "error": "Skills registry not available for fallback execution",
                }
            from .skill_plan_executor import SkillPlanExecutor

            self._skill_executor = SkillPlanExecutor(
                skills_registry=self.skills_registry,
            )

        # Discover skills using BaseAgent.discover_skills()
        discovered = self.discover_skills(task)
        if not discovered:
            return {
                "success": False,
                "error": f"No skills discovered for task: {task[:100]}",
                "task": task,
            }

        status_callback = kwargs.get("status_callback")
        return await self._skill_executor.plan_and_execute(
            task=task,
            discovered_skills=discovered,
            status_callback=status_callback,
        )

    def _build_task_from_kwargs(self, kwargs: Dict[str, Any]) -> str:
        """
        Build a task description string from keyword arguments.

        Used when 'task' is not explicitly provided but other kwargs
        contain useful context for skill-based execution.

        Args:
            kwargs: Keyword arguments to extract task from

        Returns:
            Task description string, or empty string
        """
        # Try common task-like keys
        for key in ("query", "prompt", "question", "description", "text", "input"):
            value = kwargs.get(key)
            if value and isinstance(value, str):
                return value
        # Last resort: concatenate all string values
        parts = [
            f"{k}: {v}"
            for k, v in kwargs.items()
            if isinstance(v, str) and v and k not in ("status_callback",)
        ]
        return "; ".join(parts) if parts else ""

    @property
    def input_fields(self) -> List[str]:
        """Get list of input field names."""
        return self._input_fields.copy()

    @property
    def output_fields(self) -> List[str]:
        """Get list of output field names."""
        return self._output_fields.copy()

    def get_io_schema(self) -> Any:
        """Get typed AgentIOSchema built from the DSPy Signature.

        Returns an ``AgentIOSchema`` with input/output field names,
        types, and descriptions extracted from the Signature class.
        Cached after first call.
        """
        if hasattr(self, "_io_schema") and self._io_schema is not None:
            return self._io_schema

        from Jotty.core.modes.agent._execution_types import AgentIOSchema, ToolParam

        agent_name = getattr(self.config, "name", self.__class__.__name__)
        if self.signature is not None:
            schema = AgentIOSchema.from_dspy_signature(agent_name, self.signature)
        else:
            # No signature — build from extracted field names
            schema = AgentIOSchema(
                agent_name=agent_name,
                inputs=[ToolParam(name=f) for f in self._input_fields],
                outputs=[ToolParam(name=f) for f in self._output_fields],
            )
        self._io_schema = schema
        return schema


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def create_domain_agent(
    signature: Type,
    use_chain_of_thought: bool = True,
    model: str = "",
    timeout: float = 0.0,
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
    from Jotty.core.infrastructure.foundation.config_defaults import (
        DEFAULT_MODEL_ALIAS,
        LLM_TIMEOUT_SECONDS,
    )

    model = model or DEFAULT_MODEL_ALIAS
    timeout = timeout or float(LLM_TIMEOUT_SECONDS)
    config = DomainAgentConfig(
        name=f"DomainAgent[{signature.__name__}]",
        model=model,
        timeout=timeout,
        use_chain_of_thought=use_chain_of_thought,
    )
    return DomainAgent(signature, config)


__all__ = [
    "DomainAgent",
    "DomainAgentConfig",
    "create_domain_agent",
]
