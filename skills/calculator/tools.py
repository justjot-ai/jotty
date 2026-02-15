"""
Calculator Skill - V2 (Migrated to BaseToolSkill)

Perform mathematical calculations and unit conversions.
Uses unified base skill architecture.
"""

import math
from typing import Any, Dict

from Jotty.core.infrastructure.utils.tool_helpers import tool_wrapper
from skills._base import BaseToolSkill, validate_params

# Safe evaluation context with math functions
SAFE_MATH = {
    "__builtins__": {},
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "pow": pow,
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "sinh": math.sinh,
    "cosh": math.cosh,
    "tanh": math.tanh,
    "log": math.log,
    "log10": math.log10,
    "log2": math.log2,
    "exp": math.exp,
    "floor": math.floor,
    "ceil": math.ceil,
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
    "inf": float("inf"),
    "nan": float("nan"),
}

# Unit conversion tables
LENGTH_TO_METERS = {
    "km": 1000,
    "m": 1,
    "cm": 0.01,
    "mm": 0.001,
    "miles": 1609.34,
    "yards": 0.9144,
    "feet": 0.3048,
    "inches": 0.0254,
}

WEIGHT_TO_KG = {"kg": 1, "g": 0.001, "lbs": 0.453592, "oz": 0.0283495}


def _extract_math_expression(text: str) -> str:
    """Extract a math expression from natural language."""
    import re

    cleaned = text.strip()

    # If it already looks like a math expression, return as-is
    if re.match(r"^[\d\s+\-*/().,%e^]+$", cleaned):
        return cleaned

    # Strip dollar signs, commas from numbers
    cleaned = re.sub(r"\$([0-9,.]+)", r"\1", cleaned)
    cleaned = cleaned.replace(",", "")

    # Try to find an embedded math expression
    math_match = re.search(r"([\d.]+\s*[+\-*/^%]\s*[\d.]+(?:\s*[+\-*/^%]\s*[\d.]+)*)", cleaned)
    if math_match:
        return math_match.group(1).replace("^", "**")

    # Extract all numbers from the text
    numbers = [float(x) for x in re.findall(r"[\d]+\.?\d*", cleaned)]
    lower = cleaned.lower()

    if len(numbers) >= 2:
        a, b = numbers[0], numbers[1]

        # Percentage gain/change
        if any(
            kw in lower for kw in ("percentage gain", "percent gain", "% gain", "percentage change")
        ):
            return f"({b}-{a})/{a}*100"

        # Percentage decrease
        if any(
            kw in lower for kw in ("percentage decrease", "percent decrease", "percentage loss")
        ):
            return f"({a}-{b})/{a}*100"

        # Conversion / multiply
        if any(kw in lower for kw in ("convert", "at rate", "multiply", "times")):
            return f"{a}*{b}"

        # Division
        if any(kw in lower for kw in ("divide", "ratio of", "divided by")):
            return f"{a}/{b}"

        # Difference
        if any(kw in lower for kw in ("difference", "subtract", "minus")):
            return f"{a}-{b}"

        # Sum
        if any(kw in lower for kw in ("sum", "add", "plus", "total")):
            return f"{a}+{b}"

    return text


class CalculatorSkill(BaseToolSkill):
    """Calculate mathematical expressions."""

    @validate_params(required=["expression"])
    def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        raw_expression = params["expression"]
        expression = _extract_math_expression(raw_expression)

        try:
            result = eval(expression, SAFE_MATH)

            if isinstance(result, (int, float)):
                response = {"result": float(result), "expression": expression}
                if expression != raw_expression:
                    response["original_input"] = raw_expression
                    response["parsed_expression"] = expression
                return self.success(**response)
            else:
                return self.error(f"Expression did not evaluate to a number: {result}")

        except ZeroDivisionError:
            return self.error(f"Division by zero error. Expression: {expression}")
        except NameError as e:
            return self.error(
                f"Unknown function or variable: {str(e)}. "
                f"Expression: {expression}. "
                f"Available functions: sqrt, sin, cos, tan, log, exp, abs, round"
            )
        except SyntaxError as e:
            return self.error(f"Invalid expression syntax: {str(e)}. Expression: {expression}")


class UnitConverterSkill(BaseToolSkill):
    """Convert between different units."""

    @validate_params(required=["value", "from_unit", "to_unit"])
    def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            value = float(params["value"])
        except ValueError:
            return self.error(f'Parameter "value" must be a number, got: {params.get("value")}')

        from_unit = params["from_unit"].lower()
        to_unit = params["to_unit"].lower()

        # Temperature conversions
        temp_map = {("celsius", "c"): "c", ("fahrenheit", "f"): "f", ("kelvin", "k"): "k"}

        from_t = next((v for k, v in temp_map.items() if from_unit in k), None)
        to_t = next((v for k, v in temp_map.items() if to_unit in k), None)

        if from_t and to_t:
            if from_t == to_t:
                result = value
            elif from_t == "c" and to_t == "f":
                result = (value * 9 / 5) + 32
            elif from_t == "f" and to_t == "c":
                result = (value - 32) * 5 / 9
            elif from_t == "c" and to_t == "k":
                result = value + 273.15
            elif from_t == "k" and to_t == "c":
                result = value - 273.15
            elif from_t == "f" and to_t == "k":
                result = ((value - 32) * 5 / 9) + 273.15
            elif from_t == "k" and to_t == "f":
                result = ((value - 273.15) * 9 / 5) + 32
        elif from_unit == to_unit:
            result = value
        elif from_unit in LENGTH_TO_METERS and to_unit in LENGTH_TO_METERS:
            meters = value * LENGTH_TO_METERS[from_unit]
            result = meters / LENGTH_TO_METERS[to_unit]
        elif from_unit in WEIGHT_TO_KG and to_unit in WEIGHT_TO_KG:
            kg = value * WEIGHT_TO_KG[from_unit]
            result = kg / WEIGHT_TO_KG[to_unit]
        else:
            return self.error(
                f"Unsupported conversion: {from_unit} to {to_unit}. "
                "Supported: length, weight, temperature"
            )

        return self.success(
            result=round(result, 6), from_unit=from_unit, to_unit=to_unit, value=value
        )


# Create skill instances
calculator = CalculatorSkill("calculator")
unit_converter = UnitConverterSkill("unit_converter")


# Backward compatibility - maintain old function signatures
@tool_wrapper(required_params=["expression"])
def calculate_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Perform basic mathematical calculations.

    Backward compatible wrapper for calculator skill.
    """
    return calculator(params)


@tool_wrapper(required_params=["value", "from_unit", "to_unit"])
def convert_units_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert between different units.

    Backward compatible wrapper for unit converter skill.
    """
    return unit_converter(params)


__all__ = ["calculate_tool", "convert_units_tool", "calculator", "unit_converter"]
