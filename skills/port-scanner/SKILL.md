---
name: port-scanner
description: "Check if specific ports are open on a host using socket connections"
---

# Port Scanner Skill

Check if specific ports are open on a host using socket connections

## Type
base

## Capabilities
- Check if ports are open
- Scan port ranges
- Look up common port services

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
- "scan ports"
- "check if port is open"
- "port scanner"

## Category
networking/diagnostics

## Tools

### scan_ports
Check if ports are open on a host.
**Params:** host (str), ports (list[int]), timeout (float)

## Dependencies
None
