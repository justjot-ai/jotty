---
name: converting-currency
description: "Convert currencies using live exchange rates from frankfurter.app. Use when the user wants to convert currency, exchange rate, USD to EUR."
---

# Currency Converter Skill

Convert currencies using live exchange rates from frankfurter.app. Use when the user wants to convert currency, exchange rate, USD to EUR.

## Type
base

## Capabilities
- data-fetch

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
- "currency"
- "exchange rate"
- "convert currency"
- "USD"
- "EUR"
- "forex"

## Category
data-analysis

## Tools

### convert_currency_tool
Convert between currencies using live rates.

**Parameters:**
- `amount` (float, required): Amount to convert
- `from_currency` (str, required): Source currency code (e.g. USD)
- `to_currency` (str, required): Target currency code (e.g. EUR)

**Returns:**
- `success` (bool)
- `converted` (float): Converted amount
- `rate` (float): Exchange rate used
- `from_currency` (str): Source currency
- `to_currency` (str): Target currency

## Dependencies
None
