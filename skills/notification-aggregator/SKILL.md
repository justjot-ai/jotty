---
name: aggregating-notifications
description: "Route and format notifications for multiple channels: email, webhook, log, console. Use when the user wants to send notifications, route alerts, aggregate messages."
---

# Notification Aggregator Skill

Route and format notifications for multiple channels: email, webhook, log, console. Use when the user wants to send notifications, route alerts, aggregate messages.

## Type
base

## Capabilities
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
- "notification"
- "alert"
- "notify"
- "webhook"
- "send alert"
- "push notification"

## Category
workflow-automation

## Tools

### send_notification_tool
Route a notification to specified channels.

**Parameters:**
- `message` (str, required): Notification message
- `channels` (list, optional): Channels: console, log, webhook, file (default: ["console"])
- `level` (str, optional): info, warning, error, critical (default: info)
- `title` (str, optional): Notification title
- `webhook_url` (str, optional): Webhook URL for webhook channel
- `file_path` (str, optional): File path for file channel

**Returns:**
- `success` (bool)
- `delivered` (list): Channels where delivery succeeded
- `failed` (list): Channels where delivery failed

## Dependencies
None
