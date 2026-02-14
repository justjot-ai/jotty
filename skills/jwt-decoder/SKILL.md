---
name: decoding-jwt
description: "Decode and inspect JWT tokens — view header, payload, claims, and expiry. Use when the user wants to decode JWT, inspect token, check token expiry."
---

# Jwt Decoder Skill

Decode and inspect JWT tokens — view header, payload, claims, and expiry. Use when the user wants to decode JWT, inspect token, check token expiry.

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
- "jwt"
- "decode jwt"
- "token"
- "inspect jwt"
- "json web token"

## Category
development

## Tools

### decode_jwt_tool
Decode a JWT token without verification.

**Parameters:**
- `token` (str, required): JWT token string

**Returns:**
- `success` (bool)
- `header` (dict): Token header (algorithm, type)
- `payload` (dict): Token payload (claims)
- `expired` (bool): Whether token is expired

## Dependencies
None
