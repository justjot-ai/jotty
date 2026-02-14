---
name: generating-passwords
description: "Generate cryptographically secure passwords, passphrases, and PIN codes. Use when the user wants to generate password, passphrase, PIN, secret."
---

# Password Generator Skill

Generate cryptographically secure passwords, passphrases, and PIN codes. Use when the user wants to generate password, passphrase, PIN, secret.

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
- "password"
- "passphrase"
- "PIN"
- "generate secret"
- "random string"

## Category
development

## Tools

### generate_password_tool
Generate a secure random password.

**Parameters:**
- `length` (int, optional): Password length (default: 16, min: 8, max: 128)
- `uppercase` (bool, optional): Include uppercase (default: true)
- `lowercase` (bool, optional): Include lowercase (default: true)
- `digits` (bool, optional): Include digits (default: true)
- `symbols` (bool, optional): Include symbols (default: true)
- `count` (int, optional): Number of passwords (default: 1)

**Returns:**
- `success` (bool)
- `passwords` (list): Generated passwords
- `strength` (str): Estimated strength

## Dependencies
None
