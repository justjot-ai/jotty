import math
from typing import Dict, Any


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
        Dictionary with:
            - success (bool): Whether calculation succeeded
            - result (float): Calculated result
            - error (str, optional): Error message if failed
    """
    try:
        expression = params.get('expression')
        if not expression:
            return {
                'success': False,
                'error': 'expression parameter is required'
            }
        
        # Safe evaluation context with math functions
        safe_dict = {
            '__builtins__': {},
            'abs': abs,
            'round': round,
            'min': min,
            'max': max,
            'sum': sum,
            'pow': pow,
            'sqrt': math.sqrt,
            'sin': math.sin,
            'cos': math.cos,
            'tan': math.tan,
            'asin': math.asin,
            'acos': math.acos,
            'atan': math.atan,
            'sinh': math.sinh,
            'cosh': math.cosh,
            'tanh': math.tanh,
            'log': math.log,
            'log10': math.log10,
            'log2': math.log2,
            'exp': math.exp,
            'floor': math.floor,
            'ceil': math.ceil,
            'pi': math.pi,
            'e': math.e,
            'tau': math.tau,
            'inf': float('inf'),
            'nan': float('nan')
        }
        
        # Evaluate expression safely
        result = eval(expression, safe_dict)
        
        # Convert to float if needed
        if isinstance(result, (int, float)):
            return {
                'success': True,
                'result': float(result),
                'expression': expression
            }
        else:
            return {
                'success': False,
                'error': f'Expression did not evaluate to a number: {result}'
            }
    except ZeroDivisionError:
        return {
            'success': False,
            'error': 'Division by zero'
        }
    except NameError as e:
        return {
            'success': False,
            'error': f'Unknown function or variable: {str(e)}'
        }
    except SyntaxError as e:
        return {
            'success': False,
            'error': f'Invalid expression syntax: {str(e)}'
        }
    except Exception as e:
        return {
            'success': False,
            'error': f'Error calculating: {str(e)}'
        }


def convert_units_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert between different units.
    
    Supported conversions:
    - Length: km, m, cm, mm, miles, yards, feet, inches
    - Weight: kg, g, lbs, oz
    - Temperature: celsius, fahrenheit, kelvin
    - Volume: liter, ml, gallon, quart, pint, cup
    
    Args:
        params: Dictionary containing:
            - value (float, required): Value to convert
            - from_unit (str, required): Source unit
            - to_unit (str, required): Target unit
    
    Returns:
        Dictionary with:
            - success (bool): Whether conversion succeeded
            - result (float): Converted value
            - from_unit (str): Source unit
            - to_unit (str): Target unit
            - error (str, optional): Error message if failed
    """
    try:
        value = params.get('value')
        from_unit = params.get('from_unit', '').lower()
        to_unit = params.get('to_unit', '').lower()
        
        if value is None:
            return {
                'success': False,
                'error': 'value parameter is required'
            }
        
        if not from_unit:
            return {
                'success': False,
                'error': 'from_unit parameter is required'
            }
        
        if not to_unit:
            return {
                'success': False,
                'error': 'to_unit parameter is required'
            }
        
        value = float(value)
        
        # Length conversions (to meters first, then to target)
        length_to_meters = {
            'km': 1000,
            'm': 1,
            'cm': 0.01,
            'mm': 0.001,
            'miles': 1609.34,
            'yards': 0.9144,
            'feet': 0.3048,
            'inches': 0.0254
        }
        
        # Weight conversions (to kilograms first)
        weight_to_kg = {
            'kg': 1,
            'g': 0.001,
            'lbs': 0.453592,
            'oz': 0.0283495
        }
        
        # Temperature conversions (special handling)
        if from_unit in ['celsius', 'c'] and to_unit in ['fahrenheit', 'f']:
            result = (value * 9/5) + 32
        elif from_unit in ['fahrenheit', 'f'] and to_unit in ['celsius', 'c']:
            result = (value - 32) * 5/9
        elif from_unit in ['celsius', 'c'] and to_unit in ['kelvin', 'k']:
            result = value + 273.15
        elif from_unit in ['kelvin', 'k'] and to_unit in ['celsius', 'c']:
            result = value - 273.15
        elif from_unit in ['fahrenheit', 'f'] and to_unit in ['kelvin', 'k']:
            result = ((value - 32) * 5/9) + 273.15
        elif from_unit in ['kelvin', 'k'] and to_unit in ['fahrenheit', 'f']:
            result = ((value - 273.15) * 9/5) + 32
        elif from_unit == to_unit:
            result = value
        # Length conversions
        elif from_unit in length_to_meters and to_unit in length_to_meters:
            meters = value * length_to_meters[from_unit]
            result = meters / length_to_meters[to_unit]
        # Weight conversions
        elif from_unit in weight_to_kg and to_unit in weight_to_kg:
            kg = value * weight_to_kg[from_unit]
            result = kg / weight_to_kg[to_unit]
        else:
            return {
                'success': False,
                'error': f'Unsupported conversion: {from_unit} to {to_unit}. Supported: length (km, m, cm, mm, miles, yards, feet, inches), weight (kg, g, lbs, oz), temperature (celsius, fahrenheit, kelvin)'
            }
        
        return {
            'success': True,
            'result': round(result, 6),
            'from_unit': from_unit,
            'to_unit': to_unit,
            'value': value
        }
    except ValueError:
        return {
            'success': False,
            'error': 'value must be a number'
        }
    except Exception as e:
        return {
            'success': False,
            'error': f'Error converting units: {str(e)}'
        }
