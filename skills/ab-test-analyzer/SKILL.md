---
name: analyzing-ab-tests
description: "Calculate statistical significance, p-values, and confidence intervals for A/B tests. Use when the user wants to analyze A/B test, p-value, significance."
---

# Ab Test Analyzer Skill

Calculate statistical significance, p-values, and confidence intervals for A/B tests. Use when the user wants to analyze A/B test, p-value, significance.

## Type
base

## Capabilities
- analyze

## Reference
For detailed tool documentation, see [REFERENCE.md](REFERENCE.md).

## Workflow

```
Task Progress:
- [ ] Step 1: Parse input parameters
- [ ] Step 2: Execute operation
- [ ] Step 3: Return results
```

## Triggers
- "a/b test"
- "p-value"
- "significance"
- "conversion rate"
- "hypothesis test"

## Category
data-analysis

## Tools

### ab_test_tool
Analyze A/B test results for statistical significance.

**Parameters:**
- `visitors_a` (int, required): Visitors in control group
- `conversions_a` (int, required): Conversions in control group
- `visitors_b` (int, required): Visitors in test group
- `conversions_b` (int, required): Conversions in test group
- `confidence_level` (float, optional): Confidence level (default: 0.95)

**Returns:**
- `success` (bool)
- `significant` (bool): Whether result is statistically significant
- `p_value` (float): Two-tailed p-value
- `lift` (float): Relative improvement percentage

## Dependencies
None
