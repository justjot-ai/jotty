---
name: building-decision-matrices
description: "Build weighted decision matrices to compare options across multiple criteria. Use when the user wants to compare options, decision matrix, weighted scoring, pros cons."
---

# Decision Matrix Builder Skill

Build weighted decision matrices to compare options across multiple criteria. Use when the user wants to compare options, decision matrix, weighted scoring, pros cons.

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
- "decision matrix"
- "compare options"
- "weighted scoring"
- "decision analysis"
- "pros cons"

## Category
workflow-automation

## Tools

### build_decision_matrix_tool
Build a weighted decision matrix.

**Parameters:**
- `options` (list, required): List of option names
- `criteria` (list, required): List of {name, weight} dicts (weight 1-10)
- `scores` (dict, required): {option: {criterion: score}} (scores 1-10)

**Returns:**
- `success` (bool)
- `results` (list): Ranked options with weighted scores
- `matrix` (str): Text-formatted matrix
- `winner` (str): Best option

## Dependencies
None
