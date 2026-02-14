---
name: calculating-taxes
description: "Calculate US federal income tax using current brackets, standard deduction, and effective rates. Use when the user wants to calculate tax, income tax, tax brackets, federal tax."
---

# Tax Calculator Skill

Calculate US federal income tax using current brackets, standard deduction, and effective rates. Use when the user wants to calculate tax, income tax, tax brackets, federal tax.

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
- "tax"
- "income tax"
- "tax bracket"
- "federal tax"
- "tax calculator"
- "effective rate"

## Category
data-analysis

## Tools

### calculate_federal_tax_tool
Calculate US federal income tax.

**Parameters:**
- `income` (float, required): Gross annual income
- `filing_status` (str, optional): single, married_joint, married_separate, head_of_household (default: single)
- `deductions` (float, optional): Itemized deductions (uses standard if less)
- `year` (int, optional): Tax year (default: 2024)

**Returns:**
- `success` (bool)
- `tax_owed` (float): Total federal tax
- `effective_rate` (float): Effective tax rate percentage
- `marginal_rate` (float): Marginal tax bracket
- `breakdown` (list): Tax by bracket

## Dependencies
None
