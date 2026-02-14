---
name: brokering-pmi
description: "List connected brokers, check connection status, and refresh authentication tokens."
---

# PMI Broker

Broker connection management via PlanMyInvesting API.

## Description

List connected brokers, check connection status, and refresh authentication tokens.

## Type
base

## Capabilities
- finance
- devops

## Use When
User wants to list brokers, check broker status, or refresh broker tokens

## Tools

### list_brokers_tool
List all connected brokers and their status.

### get_broker_status_tool
Get detailed status for a specific broker.

**Parameters:**
- `broker` (str, required): Broker name (e.g. "zerodha", "angel")

### refresh_tokens_tool
Refresh authentication tokens for a broker.

**Parameters:**
- `broker` (str, required): Broker name to refresh

## Triggers
- "pmi broker"

## Category
financial-analysis
