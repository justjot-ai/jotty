---
name: generating-invoices
description: "Generate invoice data as structured JSON with line items, taxes, and totals. Use when the user wants to create invoice, generate bill, invoice template."
---

# Invoice Generator Skill

Generate invoice data as structured JSON with line items, taxes, and totals. Use when the user wants to create invoice, generate bill, invoice template.

## Type
base

## Capabilities
- generate

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
- "invoice"
- "bill"
- "receipt"
- "generate invoice"
- "create invoice"

## Category
workflow-automation

## Tools

### generate_invoice_tool
Generate a structured invoice.

**Parameters:**
- `client_name` (str, required): Client/company name
- `items` (list, required): Line items [{description, quantity, rate}]
- `invoice_number` (str, optional): Invoice number (auto-generated if omitted)
- `tax_rate` (float, optional): Tax rate percentage (default: 0)
- `currency` (str, optional): Currency code (default: USD)
- `due_days` (int, optional): Payment due in days (default: 30)
- `notes` (str, optional): Additional notes

**Returns:**
- `success` (bool)
- `invoice` (dict): Complete invoice data with totals

## Dependencies
None
