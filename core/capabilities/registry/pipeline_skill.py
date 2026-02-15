"""
Generic Pipeline Skill Framework

Supports declarative Source → Processor → Sink pattern.
Composite skills defined as arrays of pipeline steps with types.
"""

import logging
import re
from enum import Enum
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class StepType(Enum):
    """Pipeline step types."""

    SOURCE = "source"  # Data retrieval/generation
    PROCESSOR = "processor"  # Data transformation
    SINK = "sink"  # Data output/delivery


class PipelineSkill:
    """
    Generic pipeline skill following Source → Processor → Sink pattern.

    Supports declarative configuration:
    ```python
    pipeline = [
        {
            "type": "source",
            "skill": "web-search",
            "tool": "search_web_tool",
            "params": {"query": "{{topic}}"}
        },
        {
            "type": "processor",
            "skill": "claude-cli-llm",
            "tool": "summarize_text_tool",
            "params": {"content": "{{source.results}}"}
        },
        {
            "type": "sink",
            "skill": "telegram-sender",
            "tool": "send_telegram_file_tool",
            "params": {"file_path": "{{processor.pdf_path}}"}
        }
    ]
    ```
    """

    def __init__(self, name: str, description: str, pipeline: List[Dict[str, Any]]) -> None:
        """
        Initialize pipeline skill.

        Args:
            name: Skill name
            description: Skill description
            pipeline: List of pipeline steps, each with:
                - type: "source", "processor", or "sink"
                - skill: Skill name
                - tool: Tool name in that skill
                - params: Parameters (dict or template string with {{vars}})
                - output_key: Key to store result (default: step index or skill name)
                - required: Whether step is required (default: True)
        """
        self.name = name
        self.description = description
        self.pipeline = pipeline
        self._validate_pipeline()

    def _validate_pipeline(self) -> Any:
        """Validate pipeline structure."""
        if not self.pipeline:
            raise ValueError("Pipeline must have at least one step")

        # Check that we have at least one source and one sink
        has_source = any(step.get("type") == StepType.SOURCE.value for step in self.pipeline)
        has_sink = any(step.get("type") == StepType.SINK.value for step in self.pipeline)

        if not has_source:
            raise ValueError("Pipeline must have at least one source step")
        if not has_sink:
            logger.warning(
                f"Pipeline '{self.name}' has no sink step - output will only be stored locally"
            )

    async def execute(self, initial_params: Dict[str, Any], registry: Any) -> Dict[str, Any]:
        """
        Execute pipeline workflow.

        Args:
            initial_params: Initial parameters
            registry: Skills registry instance

        Returns:
            Dictionary with results from all steps
        """
        results = {}
        results["_initial"] = initial_params
        current_params = initial_params.copy()

        # Execute steps sequentially (sources → processors → sinks)
        for i, step in enumerate(self.pipeline):
            step_type = step.get("type", "processor")  # Default to processor

            # Resolve template variables in params
            step_params = self._resolve_params(step.get("params", {}), current_params, results)

            # Execute step
            step_result = await self._execute_step(step, step_params, registry, results)

            # Store result
            output_key = step.get("output_key", f"{step_type}_{i}")
            results[output_key] = step_result

            # Update current_params with step result for next steps
            if step_result.get("success"):
                # Merge result into params for next steps
                current_params.update(step_result)
            else:
                # If step failed and required, stop
                if step.get("required", True):
                    results["_success"] = False
                    results["_error"] = f"Step {i} ({step_type}) failed: {step_result.get('error')}"
                    return results

        results["_success"] = True
        return results

    def _resolve_params(
        self, params: Any, current_params: Dict[str, Any], results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Resolve template variables in params.

        Supports:
        - {{variable}} - from current_params
        - {{step.field}} - from results[step]
        - {{step.field.nested}} - nested access

        Args:
            params: Parameters (dict, string with templates, or function)
            current_params: Current parameter context
            results: Results from previous steps

        Returns:
            Resolved parameters dict
        """
        # If params is a function, call it
        if callable(params):
            return params(current_params, results)

        # If params is a string, treat as template
        if isinstance(params, str):
            return self._resolve_template_string(params, current_params, results)

        # If params is a dict, resolve recursively
        if isinstance(params, dict):
            resolved = {}
            for key, value in params.items():
                if isinstance(value, str) and "{{" in value:
                    resolved[key] = self._resolve_template_string(value, current_params, results)
                elif isinstance(value, dict):
                    resolved[key] = self._resolve_params(value, current_params, results)
                else:
                    resolved[key] = value
            return resolved

        # Otherwise return as-is
        return params

    def _resolve_template_string(
        self, template: str, current_params: Dict[str, Any], results: Dict[str, Any]
    ) -> Any:
        """
        Resolve template string like "{{variable}}" or "{{step.field}}".

        Examples:
        - "{{topic}}" -> current_params['topic']
        - "{{source.results}}" -> results['source']['results']
        - "{{processor.pdf_path}}" -> results['processor']['pdf_path']
        """
        # Find all template variables
        pattern = r"\{\{([^}]+)\}\}"
        matches = re.findall(pattern, template)

        if not matches:
            return template

        # If template is just a single variable, return the value directly
        if (
            len(matches) == 1
            and template.strip() == f"{{{{{matches[0]}}}}}"
            and not matches[0].startswith("_")
        ):
            var_name = matches[0].strip()
            value = self._get_template_value(var_name, current_params, results)
            return value

        # Otherwise, substitute in template
        resolved = template
        for var_name in matches:
            var_name = var_name.strip()
            value = self._get_template_value(var_name, current_params, results)
            resolved = resolved.replace(f"{{{{{var_name}}}}}", str(value))

        return resolved

    def _get_template_value(
        self, var_name: str, current_params: Dict[str, Any], results: Dict[str, Any]
    ) -> Any:
        """
        Get value from template variable name.

        Supports:
        - "variable" -> current_params['variable']
        - "step.field" -> results['step']['field']
        - "step.field.nested" -> results['step']['field']['nested']
        """
        # Check if it's a step reference (contains dot)
        if "." in var_name:
            parts = var_name.split(".")
            step_key = parts[0]
            field_path = parts[1:]

            # Get step result
            step_result = results.get(step_key)
            if not step_result:
                raise ValueError(f"Step '{step_key}' not found in results")

            # Navigate nested path
            value = step_result
            for field in field_path:
                if isinstance(value, dict):
                    value = value.get(field)
                else:
                    raise ValueError(f"Cannot access '{field}' on non-dict value from '{step_key}'")

            if value is None:
                raise ValueError(f"Field '{var_name}' not found or is None")

            return value

        # Otherwise, check current_params
        if var_name in current_params:
            return current_params[var_name]

        # Check results (for backwards compatibility)
        if var_name in results:
            return results[var_name]

        raise ValueError(f"Template variable '{var_name}' not found")

    async def _execute_step(
        self, step: Dict[str, Any], params: Dict[str, Any], registry: Any, results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a single pipeline step."""
        skill_name = step.get("skill")
        tool_name = step.get("tool")

        if not skill_name or not tool_name:
            return {"success": False, "error": "skill and tool required for step"}

        # Get skill and tool
        skill = registry.get_skill(skill_name)
        if not skill:
            return {"success": False, "error": f"Skill not found: {skill_name}"}

        tool_func = skill.tools.get(tool_name)
        if not tool_func:
            return {"success": False, "error": f"Tool not found: {skill_name}.{tool_name}"}

        # Execute tool
        try:
            import inspect

            # Resolve params if they contain template variables
            resolved_params = self._resolve_params(params, params, results)

            if inspect.iscoroutinefunction(tool_func):
                result = await tool_func(resolved_params)
            else:
                result = tool_func(resolved_params)

            return result
        except Exception as e:
            logger.error(f"Step execution error: {e}", exc_info=True)
            return {"success": False, "error": f"Step execution failed: {str(e)}"}


def create_pipeline_skill(
    name: str, description: str, pipeline: List[Dict[str, Any]]
) -> PipelineSkill:
    """
    Factory function to create pipeline skills.

    Example:
        skill = create_pipeline_skill(
            name='search-summarize-pdf-telegram',
            description='Search → Summarize → PDF → Telegram',
            pipeline=[
                {
                    "type": "source",
                    "skill": "web-search",
                    "tool": "search_web_tool",
                    "params": {"query": "{{topic}}", "max_results": 10}
                },
                {
                    "type": "processor",
                    "skill": "claude-cli-llm",
                    "tool": "summarize_text_tool",
                    "params": {
                        "content": "{{source.results}}",
                        "prompt": "Summarize these search results"
                    }
                },
                {
                    "type": "processor",
                    "skill": "document-converter",
                    "tool": "convert_to_pdf_tool",
                    "params": {
                        "input_file": "{{processor.summary_path}}",
                        "output_file": "{{output_dir}}/summary.pdf"
                    }
                },
                {
                    "type": "sink",
                    "skill": "telegram-sender",
                    "tool": "send_telegram_file_tool",
                    "params": {
                        "file_path": "{{processor.pdf_path}}",
                        "chat_id": "{{telegram_chat_id}}"
                    }
                }
            ]
        )
    """
    return PipelineSkill(name, description, pipeline)
