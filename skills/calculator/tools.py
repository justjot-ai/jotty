"""
Calculator Skill

Perform mathematical calculations and unit conversions.
Refactored to use Jotty core utilities.
"""

import math
from typing import Dict, Any

from Jotty.core.utils.tool_helpers import tool_response, tool_error, tool_wrapper

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
LENGTH_TO_METERS = {
    'km': 1000, 'm': 1, 'cm': 0.01, 'mm': 0.001,
    'miles': 1609.34, 'yards': 0.9144, 'feet': 0.3048, 'inches': 0.0254
}

WEIGHT_TO_KG = {'kg': 1, 'g': 0.001, 'lbs': 0.453592, 'oz': 0.0283495}


@tool_wrapper(required_params=['expression'])
def calculate_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Perform basic mathematical calculations.

    Supports:
    - Basic arithmetic: +, -, *, /, **, %
    - Functions: sqrt, sin, cos, tan, log, log10, exp, abs, round, floor, ceil
    - Constants: pi, e

    Args:
        params: Dictionary containing:
            - expression (str, required): Mathematical expression to evaluate

    Returns:
        Dictionary with success, result, expression
    """
    expression = params['expression']

    try:
        result = eval(expression, SAFE_MATH)

        if isinstance(result, (int, float)):
            return tool_response(result=float(result), expression=expression)
        else:
            return tool_error(f'Expression did not evaluate to a number: {result}')

    except ZeroDivisionError:
        return tool_error('Division by zero')
    except NameError as e:
        return tool_error(f'Unknown function or variable: {str(e)}')
    except SyntaxError as e:
        return tool_error(f'Invalid expression syntax: {str(e)}')


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
    try:
        value = float(params['value'])
    except ValueError:
        return tool_error('value must be a number')

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
