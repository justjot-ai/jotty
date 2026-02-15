"""
Domain-Specific Validators for Expert Agents

Provides domain-specific validation rules for different expert types.
"""

import logging
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class DomainValidator(ABC):
    """Base class for domain-specific validators."""

    @abstractmethod
    def validate(
        self, output: str, expected_type: str, context: Dict[str, Any]
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Validate output for domain-specific rules.

        Args:
            output: Generated output to validate
            expected_type: Expected diagram/code type
            context: Context dictionary

        Returns:
            Tuple of (is_valid, error_message, metadata)
        """
        pass

    @abstractmethod
    def detect_type(self, output: str) -> str:
        """Detect the type of output."""
        pass


class MermaidValidator(DomainValidator):
    """Validator for Mermaid diagrams."""

    def validate(
        self, output: str, expected_type: str, context: Dict[str, Any]
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """Validate Mermaid diagram."""
        output = output.strip()

        # Remove markdown fences
        output = re.sub(r"^```mermaid\s*\n?", "", output, flags=re.MULTILINE)
        output = re.sub(r"^```\s*$", "", output, flags=re.MULTILINE)
        output = output.strip()

        if not output:
            return False, "Empty diagram", {}

        # Detect actual type
        actual_type = self.detect_type(output)

        # Check type match
        type_match = self._check_type_match(actual_type, expected_type)

        # Domain-specific checks
        errors = []
        metadata = {
            "actual_type": actual_type,
            "expected_type": expected_type,
            "type_match": type_match,
            "lines": len(output.split("\n")),
        }

        # Check for gitGraph specifically
        if expected_type.lower() == "gitgraph":
            if actual_type != "gitGraph":
                errors.append(f"Expected gitGraph but got {actual_type}")
                metadata["type_error"] = True

        # Check for required elements based on context
        required_elements = context.get("required_elements", [])
        if required_elements:
            found_elements = []
            for element in required_elements:
                if element.lower() in output.lower():
                    found_elements.append(element)
            metadata["required_elements_found"] = found_elements
            metadata["required_elements_total"] = required_elements
            metadata["element_coverage"] = (
                len(found_elements) / len(required_elements) if required_elements else 0.0
            )

        # Check syntax basics
        if actual_type == "unknown":
            errors.append("Unknown diagram type")

        # Check for balanced brackets (basic)
        open_brackets = output.count("[") + output.count("(") + output.count("{")
        close_brackets = output.count("]") + output.count(")") + output.count("}")
        if abs(open_brackets - close_brackets) > 5:  # Allow some imbalance for complex diagrams
            errors.append(
                f"Significant bracket imbalance: {open_brackets} open, {close_brackets} close"
            )

        is_valid = type_match and len(errors) == 0
        error_msg = "; ".join(errors) if errors else "Valid"

        return is_valid, error_msg, metadata

    def detect_type(self, output: str) -> str:
        """Detect Mermaid diagram type."""
        if not output:
            return "unknown"

        first_line = output.split("\n")[0].strip().lower()

        # Check for gitGraph specifically first (it's a special case)
        if first_line.startswith("gitgraph") or "gitgraph" in first_line:
            return "gitGraph"

        # Other types
        type_map = {
            "graph": ["graph", "flowchart"],
            "sequenceDiagram": ["sequencediagram", "sequence"],
            "stateDiagram-v2": ["statediagram-v2", "statediagram"],
            "gantt": ["gantt"],
            "erDiagram": ["erdiagram", "er diagram"],
            "journey": ["journey"],
            "classDiagram": ["classdiagram", "class diagram"],
        }

        for mermaid_type, keywords in type_map.items():
            if any(keyword in first_line for keyword in keywords):
                return mermaid_type

        return "unknown"

    def _check_type_match(self, actual: str, expected: str) -> bool:
        """Check if actual type matches expected."""
        if actual == expected:
            return True

        # Special cases
        if expected.lower() == "flowchart" and actual in ["graph", "flowchart"]:
            return True

        if expected.lower() == "graph" and actual in ["graph", "flowchart"]:
            return True

        return False


class PlantUMLValidator(DomainValidator):
    """Validator for PlantUML diagrams."""

    def validate(
        self, output: str, expected_type: str, context: Dict[str, Any]
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """Validate PlantUML diagram."""
        output = output.strip()

        # Remove markdown fences
        output = re.sub(r"^```plantuml\s*\n?", "", output, flags=re.MULTILINE)
        output = re.sub(r"^```\s*$", "", output, flags=re.MULTILINE)
        output = output.strip()

        if not output:
            return False, "Empty diagram", {}

        # Detect actual type
        actual_type = self.detect_type(output)

        # Check for @startuml/@enduml tags
        has_start_tag = "@startuml" in output.lower()
        has_end_tag = "@enduml" in output.lower()

        metadata = {
            "actual_type": actual_type,
            "expected_type": expected_type,
            "has_start_tag": has_start_tag,
            "has_end_tag": has_end_tag,
            "has_tags": has_start_tag and has_end_tag,
            "lines": len(output.split("\n")),
        }

        errors = []

        # Check tags
        if not has_start_tag or not has_end_tag:
            errors.append("Missing @startuml/@enduml tags")
            metadata["missing_tags"] = True

        # Check type match
        type_match = actual_type == expected_type or expected_type.lower() in actual_type.lower()
        if not type_match:
            errors.append(f"Type mismatch: expected {expected_type}, got {actual_type}")

        is_valid = len(errors) == 0 or (has_start_tag and has_end_tag)  # Tags are critical
        error_msg = "; ".join(errors) if errors else "Valid"

        return is_valid, error_msg, metadata

    def detect_type(self, output: str) -> str:
        """Detect PlantUML diagram type."""
        if not output:
            return "unknown"

        output_lower = output.lower()

        # Check for diagram type keywords
        if "sequence" in output_lower or "participant" in output_lower:
            return "sequence"
        elif "class" in output_lower and "diagram" in output_lower:
            return "class"
        elif "state" in output_lower or "statechart" in output_lower:
            return "state"
        elif "activity" in output_lower:
            return "activity"
        elif "component" in output_lower:
            return "component"
        else:
            return "unknown"


class MathLaTeXValidator(DomainValidator):
    """Validator for Math LaTeX expressions."""

    def validate(
        self, output: str, expected_type: str, context: Dict[str, Any]
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Validate Math LaTeX expression.

        Args:
            output: LaTeX expression to validate
            expected_type: Expected type (inline, display, equation, formula)
            context: Context dictionary

        Returns:
            Tuple of (is_valid, error_message, metadata)
        """
        errors = []
        metadata = {}

        # Remove markdown fences
        output = output.strip()
        if output.startswith("```"):
            lines = output.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            output = "\n".join(lines).strip()

        # Check for math delimiters
        has_inline = "$" in output
        has_display = "$$" in output or "\\[" in output or "\\begin{equation}" in output.lower()
        has_commands = bool(re.search(r"\\[a-zA-Z]+", output))

        if not (has_inline or has_display or has_commands):
            errors.append("Missing math delimiters ($, $$, \\[, etc.) or LaTeX commands")

        # Check type match
        actual_type = self.detect_type(output)
        type_match = actual_type == expected_type or expected_type == "any"

        if not type_match and expected_type != "any":
            errors.append(f"Type mismatch: expected {expected_type}, got {actual_type}")

        # Check balanced braces
        open_braces = output.count("{")
        close_braces = output.count("}")
        if open_braces != close_braces:
            errors.append(f"Unbalanced braces: {open_braces} open, {close_braces} close")

        # Check for required elements
        required_elements = context.get("required_elements", [])
        found_elements = []
        if required_elements:
            for element in required_elements:
                # Check for LaTeX commands or symbols
                if element.lower() in output.lower() or f"\\{element}" in output:
                    found_elements.append(element)

        metadata = {
            "actual_type": actual_type,
            "expected_type": expected_type,
            "type_match": type_match,
            "has_inline": has_inline,
            "has_display": has_display,
            "has_commands": has_commands,
            "found_elements": found_elements,
            "required_elements": required_elements,
            "element_coverage": (
                len(found_elements) / len(required_elements) if required_elements else 1.0
            ),
        }

        is_valid = len(errors) == 0
        error_msg = "; ".join(errors) if errors else ""

        return is_valid, error_msg, metadata

    def detect_type(self, output: str) -> str:
        """Detect Math LaTeX expression type."""
        output_lower = output.lower()

        if (
            "$$" in output
            or "\\[" in output
            or "\\begin{equation}" in output_lower
            or "\\begin{align}" in output_lower
        ):
            return "display"
        elif "$" in output:
            return "inline"
        elif "\\begin{" in output_lower:
            return "equation"
        else:
            return "formula"


def get_validator(domain: str) -> Optional[DomainValidator]:
    """Get validator for a domain."""
    validators = {
        "mermaid": MermaidValidator(),
        "plantuml": PlantUMLValidator(),
        "plantml": PlantUMLValidator(),  # Alias
        "math_latex": MathLaTeXValidator(),
        "latex": MathLaTeXValidator(),  # Alias
        "math": MathLaTeXValidator(),  # Alias
    }

    return validators.get(domain.lower())
