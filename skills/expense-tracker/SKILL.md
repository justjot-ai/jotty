---
name: tracking-expenses
description: "Track expenses with categorization, budget limits, and summary reports. Uses JSON file storage. Use when the user wants to track expense, log spending, budget tracker."
---

# Expense Tracker Skill

Track expenses with categorization, budget limits, and summary reports. Uses JSON file storage. Use when the user wants to track expense, log spending, budget tracker.

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
- "expense"
- "spending"
- "budget"
- "track expense"
- "log expense"
- "expense report"

## Category
workflow-automation

## Tools

### add_expense_tool
Add an expense entry.

**Parameters:**
- `amount` (float, required): Expense amount
- `category` (str, required): Category (food, transport, utilities, entertainment, shopping, health, other)
- `description` (str, optional): Description of expense
- `date` (str, optional): Date YYYY-MM-DD (default: today)
- `storage_file` (str, optional): JSON storage path (default: expenses.json)

**Returns:**
- `success` (bool)
- `expense` (dict): Added expense record
- `running_total` (float): Total expenses

### expense_summary_tool
Get expense summary and breakdown.

**Parameters:**
- `storage_file` (str, optional): JSON storage path
- `month` (str, optional): Filter by month YYYY-MM
- `category` (str, optional): Filter by category

**Returns:**
- `success` (bool)
- `total` (float): Total expenses
- `by_category` (dict): Breakdown by category
- `count` (int): Number of expenses

## Dependencies
None
