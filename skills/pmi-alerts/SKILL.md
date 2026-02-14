---
name: alerting-pmi
description: "Create, manage, and monitor stock price alerts with configurable conditions and notification channels."
---

# PMI Alerts

Price and event alert management via PlanMyInvesting API.

## Description

Create, manage, and monitor stock price alerts with configurable conditions and notification channels.

## Type
derived

## Base Skills
- pmi-market-data

## Capabilities
- finance
- communicate

## Use When
User wants to create price alerts, list alerts, delete alerts, or check alert statistics

## Tools

### list_alerts_tool
List all active alerts.

**Parameters:**
- `symbol` (str, optional): Filter by symbol
- `alert_type` (str, optional): Filter by type (price, volume, news)

### create_alert_tool
Create a new price/event alert.

**Parameters:**
- `symbol` (str, required): Stock symbol
- `condition` (str, required): Condition (above, below, crosses, percent_change)
- `value` (float, required): Trigger value
- `alert_type` (str, optional): Type (price, volume)
- `notify_via` (str, optional): Channel (telegram, email)

### delete_alert_tool
Delete an alert.

**Parameters:**
- `alert_id` (str, required): Alert ID to delete

### get_alert_stats_tool
Get alert statistics.

## Triggers
- "pmi alerts"

## Category
financial-analysis
