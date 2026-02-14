---
name: sending-commodities-to-telegram
description: "Fetches latest commodities prices from investing.com and sends to Telegram. Use when the user wants to send to telegram, telegram message, notify via telegram."
---

# Investing.com Commodities to Telegram

Fetches latest commodities prices from investing.com and sends to Telegram.


## Type
composite

## Base Skills
- investing-commodities
- telegram-sender

## Execution
sequential


## Capabilities
- data-fetch
- communicate

## Tools

### `commodities_to_telegram_tool`

Fetches commodities prices and sends formatted report to Telegram.

**Parameters:**
- `category` (str, optional): Category filter - 'energy', 'metals', 'agriculture', or 'all' (default: 'all')
- `send_telegram` (bool, optional): Whether to send to Telegram (default: True)
- `telegram_chat_id` (str, optional): Telegram chat ID (uses env var if not provided)
- `format` (str, optional): Message format - 'html', 'markdown', or 'text' (default: 'html')

**Returns:**
- `success` (bool): Whether operation succeeded
- `commodities` (list): List of commodities fetched
- `telegram_sent` (bool): Whether sent to Telegram
- `telegram_message_id` (int, optional): Telegram message ID if sent
- `error` (str, optional): Error message if failed

## Workflow

```
Task Progress:
- [ ] Step 1: Fetch commodities prices
- [ ] Step 2: Format report
- [ ] Step 3: Send to Telegram
```

**Step 1: Fetch commodities prices**
Retrieve latest commodities data from investing.com.

**Step 2: Format report**
Structure the price data into a formatted HTML/markdown report.

**Step 3: Send to Telegram**
Deliver the commodities report to the specified Telegram chat.

## Triggers
- "investing commodities to telegram"
- "send to telegram"
- "telegram message"
- "notify via telegram"

## Category
communication
