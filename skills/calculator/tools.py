"""
Calculator Skill

Perform mathematical calculations and unit conversions.
Refactored to use Jotty core utilities.
"""

import math
from typing import Dict, Any

from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, tool_wrapper

from Jotty.core.infrastructure.utils.skill_status import SkillStatus

# Safe evaluation context with math functions
SAFE_MATH = {
    '__builtins__': {},
    'abs': abs, 'round': round, 'min': min, 'max': max, 'sum': sum, 'pow': pow,
    'sqrt': math.sqrt, 'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
    'asin': math.asin, 'acos': math.acos, 'atan': math.atan,
    'sinh': math.sinh, 'cosh': math.cosh, 'tanh': math.tanh,
    'log': math.log, 'log10': math.log10, 'log2': math.log2, 'exp': math.exp,
    'floor': math.floor, 'ceil': math.ceil,
    'pi': math.pi, 'e': math.e, 'tau': math.tau,
    'inf': float('inf'), 'nan': float('nan')
}

# Unit conversion tables

# Status emitter for progress updates
status = SkillStatus("calculator")

LENGTH_TO_METERS = {
    'km': 1000, 'm': 1, 'cm': 0.01, 'mm': 0.001,
    'miles': 1609.34, 'yards': 0.9144, 'feet': 0.3048, 'inches': 0.0254
}

WEIGHT_TO_KG = {'kg': 1, 'g': 0.001, 'lbs': 0.453592, 'oz': 0.0283495}


def _extract_math_expression(text: str) -> str:
    """Extract a math expression from natural language.

    LLM planners often pass descriptions like 'Calculate the percentage gain
    if stock went from $500 to $850' instead of '(850-500)/500*100'.
    This method extracts numbers and infers the operation.
    """
    import re

    cleaned = text.strip()

    # If it already looks like a math expression, return as-is
    if re.match(r'^[\d\s+\-*/().,%e^]+$', cleaned):
        return cleaned

    # Strip dollar signs, commas from numbers
    cleaned = re.sub(r'\$([0-9,.]+)', r'\1', cleaned)
    cleaned = cleaned.replace(',', '')

    # Try to find an embedded math expression (e.g. "(850-500)/500*100")
    math_match = re.search(r'([\d.]+\s*[+\-*/^%]\s*[\d.]+(?:\s*[+\-*/^%]\s*[\d.]+)*)', cleaned)
    if math_match:
        expr = math_match.group(1).replace('^', '**')
        return expr

    # Extract all numbers from the text
    numbers = [float(x) for x in re.findall(r'[\d]+\.?\d*', cleaned)]
    lower = cleaned.lower()

    if len(numbers) >= 2:
        a, b = numbers[0], numbers[1]

        # Percentage gain/change
        if any(kw in lower for kw in ('percentage gain', 'percent gain', '% gain',
                                       'percentage change', 'percent change',
                                       'percentage increase', 'percent increase')):
            return f'({b}-{a})/{a}*100'

        # Percentage decrease
        if any(kw in lower for kw in ('percentage decrease', 'percent decrease',
                                       'percentage loss', 'percent loss')):
            return f'({a}-{b})/{a}*100'

        # Conversion / multiply
        if any(kw in lower for kw in ('convert', 'at rate', 'at a rate', 'multiply',
                                       'times', 'rate of')):
            return f'{a}*{b}'

        # P/E × EPS = price (or similar product relationships)
        if any(kw in lower for kw in ('p/e', 'pe ratio', 'implied price',
                                       'implied stock price')):
            return f'{a}*{b}'

        # Division
        if any(kw in lower for kw in ('divide', 'ratio of', 'divided by')):
            return f'{a}/{b}'

        # Difference
        if any(kw in lower for kw in ('difference', 'subtract', 'minus', 'less')):
            return f'{a}-{b}'

        # Sum
        if any(kw in lower for kw in ('sum', 'add', 'plus', 'total', 'combined')):
            return f'{a}+{b}'

    if len(numbers) == 3:
        a, b, c = numbers[0], numbers[1], numbers[2]
        # Three numbers with percentage context: (b-a)/a * 100
        if 'percent' in lower or '%' in lower:
            return f'({b}-{a})/{a}*100'
        # P/E, EPS, implied price: P/E * EPS
        if 'p/e' in lower or 'pe ratio' in lower:
            return f'{a}*{b}'

    # If nothing matched, return original — let eval try and fail gracefully
    return text


@tool_wrapper(required_params=['expression'])
def calculate_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Perform basic mathematical calculations.

    Supports:
    - Basic arithmetic: +, -, *, /, **, %
    - Functions: sqrt, sin, cos, tan, log, log10, exp, abs, round, floor, ceil
    - Constants: pi, e
    - Natural language: 'percentage gain from 500 to 850' → (850-500)/500*100

    Args:
        params: Dictionary containing:
            - expression (str, required): Mathematical expression to evaluate

    Returns:
        Dictionary with success, result, expression
    """
    status.set_callback(params.pop('_status_callback', None))

    raw_expression = params['expression']
    expression = _extract_math_expression(raw_expression)

    try:
        result = eval(expression, SAFE_MATH)

        if isinstance(result, (int, float)):
            response = tool_response(result=float(result), expression=expression)
            if expression != raw_expression:
                response['original_input'] = raw_expression
                response['parsed_expression'] = expression
            return response
        else:
            return tool_error(f'Expression did not evaluate to a number: {result}')

    except ZeroDivisionError:
        return tool_error(
            'Division by zero error. Check your expression for division operations. '
            f'Expression attempted: {expression}'
        )
    except NameError as e:
        return tool_error(
            f'Unknown function or variable: {str(e)}. '
            f'Expression: {expression}. '
            f'Available functions: sqrt, sin, cos, tan, log, exp, abs, round. '
            f'Example: "sqrt(16)" or "sin(pi/2)"'
        )
    except SyntaxError as e:
        return tool_error(
            f'Invalid expression syntax: {str(e)}. '
            f'Expression: {expression}. '
            f'Use standard math notation. Examples: "2 + 2", "(5 * 3) - 2", "sqrt(16)"'
        )


@tool_wrapper(required_params=['value', 'from_unit', 'to_unit'])
def convert_units_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert between different units.

    Supported: length (km, m, cm, mm, miles, yards, feet, inches),
    weight (kg, g, lbs, oz), temperature (celsius, fahrenheit, kelvin)

    Args:
        params: Dictionary containing:
            - value (float, required): Value to convert
            - from_unit (str, required): Source unit
            - to_unit (str, required): Target unit

    Returns:
        Dictionary with success, result, from_unit, to_unit, value
    """
    status.set_callback(params.pop('_status_callback', None))

    try:
        value = float(params['value'])
    except ValueError:
        return tool_error(
            f'Parameter "value" must be a number, got: {params.get("value")}. '
            f'Example: {{"value": 100, "from_unit": "km", "to_unit": "miles"}}'
        )

    from_unit = params['from_unit'].lower()
    to_unit = params['to_unit'].lower()

    # Temperature conversions
    temp_map = {
        ('celsius', 'c'): 'c', ('fahrenheit', 'f'): 'f', ('kelvin', 'k'): 'k'
    }

    from_t = next((v for k, v in temp_map.items() if from_unit in k), None)
    to_t = next((v for k, v in temp_map.items() if to_unit in k), None)

    if from_t and to_t:
        if from_t == to_t:
            result = value
        elif from_t == 'c' and to_t == 'f':
            result = (value * 9/5) + 32
        elif from_t == 'f' and to_t == 'c':
            result = (value - 32) * 5/9
        elif from_t == 'c' and to_t == 'k':
            result = value + 273.15
        elif from_t == 'k' and to_t == 'c':
            result = value - 273.15
        elif from_t == 'f' and to_t == 'k':
            result = ((value - 32) * 5/9) + 273.15
        elif from_t == 'k' and to_t == 'f':
            result = ((value - 273.15) * 9/5) + 32
    elif from_unit == to_unit:
        result = value
    elif from_unit in LENGTH_TO_METERS and to_unit in LENGTH_TO_METERS:
        meters = value * LENGTH_TO_METERS[from_unit]
        result = meters / LENGTH_TO_METERS[to_unit]
    elif from_unit in WEIGHT_TO_KG and to_unit in WEIGHT_TO_KG:
        kg = value * WEIGHT_TO_KG[from_unit]
        result = kg / WEIGHT_TO_KG[to_unit]
    else:
        return tool_error(
            f'Unsupported conversion: {from_unit} to {to_unit}. '
            'Supported: length, weight, temperature'
        )

    return tool_response(
        result=round(result, 6),
        from_unit=from_unit,
        to_unit=to_unit,
        value=value
    )


__all__ = ['calculate_tool', 'convert_units_tool']
