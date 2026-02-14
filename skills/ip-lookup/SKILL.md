---
name: looking-up-ip
description: "Look up IP address geolocation, ISP, and network info. Use when the user wants to lookup IP, geolocate IP, find IP location, my IP."
---

# Ip Lookup Skill

Look up IP address geolocation, ISP, and network info. Use when the user wants to lookup IP, geolocate IP, find IP location, my IP.

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
- "ip"
- "ip lookup"
- "geolocate"
- "my ip"
- "ip address"
- "whois"

## Category
development

## Tools

### ip_lookup_tool
Look up geolocation and network info for an IP address.

**Parameters:**
- `ip` (str, optional): IP address (default: your public IP)

**Returns:**
- `success` (bool)
- `ip` (str): IP address
- `country` (str): Country
- `city` (str): City
- `isp` (str): Internet Service Provider

## Dependencies
None
