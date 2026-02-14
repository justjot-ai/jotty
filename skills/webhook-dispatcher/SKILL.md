---
name: dispatching-webhooks
description: "Send HTTP webhook payloads with retry logic and authentication. Use when the user wants to send webhook, POST payload, HTTP callback."
---

# Webhook Dispatcher Skill

Send HTTP webhook payloads with retry logic and authentication. Use when the user wants to send webhook, POST payload, HTTP callback.

## Type
base

## Capabilities
- data-fetch
- code

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
- "webhook"
- "send webhook"
- "POST"
- "callback"
- "http post"

## Category
development

## Tools

### send_webhook_tool
Send an HTTP webhook with retry logic.

**Parameters:**
- `url` (str, required): Webhook URL
- `payload` (dict, required): JSON payload
- `method` (str, optional): HTTP method (default: POST)
- `headers` (dict, optional): Custom headers
- `retries` (int, optional): Max retries (default: 3)
- `timeout` (int, optional): Timeout seconds (default: 10)

**Returns:**
- `success` (bool)
- `status_code` (int): HTTP status code
- `attempts` (int): Number of attempts made

## Dependencies
None
