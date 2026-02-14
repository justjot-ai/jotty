---
name: calculating-loan-amortization
description: "Generate loan amortization schedules with monthly payments, interest, and principal breakdown. Use when the user wants to calculate loan, amortization schedule, mortgage payment."
---

# Loan Amortization Calculator Skill

Generate loan amortization schedules with monthly payments, interest, and principal breakdown. Use when the user wants to calculate loan, amortization schedule, mortgage payment.

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
- "loan"
- "amortization"
- "mortgage"
- "monthly payment"
- "loan calculator"
- "interest"

## Category
data-analysis

## Tools

### amortization_schedule_tool
Generate a loan amortization schedule.

**Parameters:**
- `principal` (float, required): Loan principal amount
- `annual_rate` (float, required): Annual interest rate (percentage)
- `years` (int, required): Loan term in years
- `extra_payment` (float, optional): Extra monthly payment (default: 0)

**Returns:**
- `success` (bool)
- `monthly_payment` (float): Monthly payment amount
- `total_interest` (float): Total interest paid
- `total_paid` (float): Total amount paid
- `schedule` (list): Monthly breakdown (first 12 + last month)
- `payoff_months` (int): Actual months to payoff

## Dependencies
None
