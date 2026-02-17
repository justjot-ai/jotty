---
name: math-toolkit
description: "Unified math, finance, and conversion toolkit. Calculate expressions, convert units/currencies, compute BMI, tips, mortgages, loan amortization, statistics, and number base conversions."
---

# Math Toolkit

Unified mathematics, finance, and conversion skill. Consolidates calculator, unit-converter,
bmi-calculator, tip-calculator, mortgage-calculator, loan-amortization-calculator,
statistics-calculator, currency-converter, number-base-converter, binary-converter,
and roman-numeral-converter into one coherent toolkit.

## Type
base

## Capabilities
- calculate
- math
- convert-units
- finance
- statistics
- number-conversion

## Triggers
- "calculate"
- "math"
- "convert"
- "unit conversion"
- "bmi"
- "tip"
- "mortgage"
- "loan"
- "amortization"
- "statistics"
- "mean"
- "median"
- "standard deviation"
- "currency"
- "exchange rate"
- "binary"
- "hex"
- "octal"
- "roman numeral"

## Category
math

## Tools

### calculate_tool
Evaluate mathematical expressions with natural language support.

**Parameters:**
- `expression` (str, required): Mathematical expression or natural language math query

**Returns:**
- `result` (float): Calculated result
- `expression` (str): Parsed expression

### convert_units_tool
Convert between length, weight, temperature, volume, and speed units.

**Parameters:**
- `value` (float, required): Value to convert
- `from_unit` (str, required): Source unit
- `to_unit` (str, required): Target unit

**Returns:**
- `result` (float): Converted value

### bmi_tool
Calculate BMI from weight and height, with category and healthy range.

**Parameters:**
- `weight_kg` (float, required): Weight in kilograms
- `height_m` (float, required): Height in meters

**Returns:**
- `bmi` (float): BMI value
- `category` (str): Weight category
- `healthy_weight_range_kg` (object): Min/max healthy weight

### tip_tool
Calculate tip amount, total bill, and per-person split.

**Parameters:**
- `bill_amount` (float, required): Bill amount
- `tip_percent` (float, optional): Tip percentage (default: 18)
- `num_people` (int, optional): Number of people (default: 1)

**Returns:**
- `tip_amount` (float): Tip amount
- `total` (float): Total with tip
- `per_person` (float): Per-person amount
- `suggestions` (object): Tip suggestions at 15%, 18%, 20%, 25%

### mortgage_tool
Calculate monthly mortgage payment and amortization summary.

**Parameters:**
- `principal` (float, required): Loan principal amount
- `annual_rate` (float, required): Annual interest rate (percentage)
- `term_years` (int, required): Loan term in years

**Returns:**
- `monthly_payment` (float): Monthly payment
- `total_paid` (float): Total amount paid
- `total_interest` (float): Total interest paid
- `amortization_summary` (array): Yearly breakdown

### amortization_schedule_tool
Generate detailed loan amortization schedule with optional extra payments.

**Parameters:**
- `principal` (float, required): Loan principal
- `annual_rate` (float, required): Annual interest rate (percentage)
- `years` (int, required): Loan term in years
- `extra_payment` (float, optional): Extra monthly payment (default: 0)

**Returns:**
- `monthly_payment` (float): Monthly payment
- `schedule` (array): Month-by-month breakdown
- `payoff_months` (int): Actual payoff duration
- `interest_savings` (float): Savings from extra payments

### statistics_tool
Compute descriptive statistics on a dataset.

**Parameters:**
- `data` (array, required): List of numbers
- `include_percentiles` (bool, optional): Include percentiles (default: false)

**Returns:**
- `count` (int): Number of values
- `mean` (float): Average
- `median` (float): Median
- `mode` (float): Most common value
- `std_dev` (float): Standard deviation
- `variance` (float): Variance
- `min` (float): Minimum
- `max` (float): Maximum

### convert_number_base_tool
Convert numbers between bases (binary, octal, decimal, hex, any base 2-36).

**Parameters:**
- `value` (str, required): Number to convert
- `from_base` (int, optional): Source base (default: 10)
- `to_base` (int, optional): Target base (default: 2)

**Returns:**
- `result` (str): Converted number
- `all_bases` (object): Value in binary, octal, decimal, hex

### roman_numeral_tool
Convert between Roman numerals and integers.

**Parameters:**
- `value` (str, required): Integer or Roman numeral string

**Returns:**
- `roman` (str): Roman numeral representation
- `integer` (int): Integer representation

## Dependencies
None
