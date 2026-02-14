---
name: calculator
description: "Provides mathematical calculation capabilities including basic arithmetic, scientific calculations, and unit conversions. Use when the user wants to calculate, math, add."
---

# Calculator Skill

## Description
Provides mathematical calculation capabilities including basic arithmetic, scientific calculations, and unit conversions.


## Type
base


## Capabilities
- analyze


## Triggers
- "calculate"
- "math"
- "add"
- "subtract"
- "multiply"
- "compute"
- "what is the result of"
- "how much is"

## Category
workflow-automation

## Tools

### calculate_tool
Performs basic mathematical calculations.

**Parameters:**
- `expression` (str, required): Mathematical expression to evaluate (e.g., "2 + 2", "sqrt(16)", "sin(pi/2)")

**Returns:**
- `success` (bool): Whether calculation succeeded
- `result` (float): Calculated result
- `error` (str, optional): Error message if failed

### convert_units_tool
Converts between different units.

**Parameters:**
- `value` (float, required): Value to convert
- `from_unit` (str, required): Source unit (e.g., "km", "miles", "celsius", "fahrenheit", "kg", "lbs")
- `to_unit` (str, required): Target unit

**Returns:**
- `success` (bool): Whether conversion succeeded
- `result` (float): Converted value
- `from_unit` (str): Source unit
- `to_unit` (str): Target unit
- `error` (str, optional): Error message if failed
