---
name: monitoring-uptime
description: "Check HTTP endpoint availability, response time, status codes. Use when the user wants to check uptime, ping website, monitor endpoint."
---

# Uptime Monitor Skill

Check HTTP endpoint availability, response time, status codes. Use when the user wants to check uptime, ping website, monitor endpoint.

## Type
base

## Capabilities
- data-fetch
- devops

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
- "uptime"
- "ping"
- "health check"
- "endpoint status"
- "website up"

## Category
development

## Tools

### check_endpoint_tool
Check HTTP endpoint availability.

**Parameters:**
- `url` (str, required): URL to check
- `method` (str, optional): HTTP method (default: GET)
- `timeout` (int, optional): Timeout seconds (default: 10)
- `expected_status` (int, optional): Expected HTTP status (default: 200)

**Returns:**
- `success` (bool)
- `status_code` (int): HTTP status code
- `response_time_ms` (float): Response time in milliseconds
- `available` (bool): Whether endpoint is available

## Dependencies
None
