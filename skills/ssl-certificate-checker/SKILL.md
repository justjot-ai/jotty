---
name: checking-ssl-certificates
description: "Check SSL certificate validity, expiry date, issuer, and chain. Use when the user wants to check SSL, certificate expiry, HTTPS cert."
---

# Ssl Certificate Checker Skill

Check SSL certificate validity, expiry date, issuer, and chain. Use when the user wants to check SSL, certificate expiry, HTTPS cert.

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
- "ssl"
- "certificate"
- "cert expiry"
- "https check"
- "tls"

## Category
development

## Tools

### check_ssl_tool
Check SSL certificate for a hostname.

**Parameters:**
- `hostname` (str, required): Domain to check
- `port` (int, optional): Port (default: 443)

**Returns:**
- `success` (bool)
- `subject` (str): Certificate subject
- `issuer` (str): Certificate issuer
- `expires` (str): Expiry date
- `days_remaining` (int): Days until expiry
- `valid` (bool): Whether cert is currently valid

## Dependencies
None
