---
name: looking-up-dns
description: "Perform DNS lookups — A, AAAA, MX, CNAME, TXT, NS records. Use when the user wants to DNS lookup, resolve domain, check DNS records, MX records."
---

# Dns Lookup Skill

Perform DNS lookups — A, AAAA, MX, CNAME, TXT, NS records. Use when the user wants to DNS lookup, resolve domain, check DNS records, MX records.

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
- "dns"
- "dns lookup"
- "resolve domain"
- "mx records"
- "nameserver"
- "dig"

## Category
development

## Tools

### dns_lookup_tool
Perform DNS lookup for a domain.

**Parameters:**
- `domain` (str, required): Domain to look up
- `record_type` (str, optional): A, AAAA, MX, CNAME, TXT, NS, SOA (default: A)

**Returns:**
- `success` (bool)
- `domain` (str): Queried domain
- `records` (list): DNS records found

## Dependencies
None
