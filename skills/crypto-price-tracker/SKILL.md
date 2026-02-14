---
name: tracking-crypto-prices
description: "Fetch cryptocurrency prices from CoinGecko free API. Use when the user wants to check crypto price, bitcoin price, ethereum price."
---

# Crypto Price Tracker Skill

Fetch cryptocurrency prices from CoinGecko free API. Use when the user wants to check crypto price, bitcoin price, ethereum price.

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
- "crypto"
- "bitcoin"
- "ethereum"
- "crypto price"
- "BTC"
- "ETH"
- "cryptocurrency"

## Category
data-analysis

## Tools

### crypto_price_tool
Get current cryptocurrency price.

**Parameters:**
- `coin` (str, required): Coin ID or symbol (bitcoin, ethereum, BTC, ETH)
- `currency` (str, optional): Fiat currency (default: usd)

**Returns:**
- `success` (bool)
- `coin` (str): Coin name
- `price` (float): Current price
- `change_24h` (float): 24h price change percentage

## Dependencies
None
