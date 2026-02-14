---
name: checksum-verifier
description: "Calculate and verify MD5, SHA1, and SHA256 checksums for files and strings"
---

# Checksum Verifier Skill

Calculate and verify MD5, SHA1, and SHA256 checksums for files and strings

## Type
base

## Capabilities
- Calculate MD5/SHA1/SHA256 checksums
- Verify checksums against expected values
- Hash files and strings

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
- "calculate checksum"
- "verify sha256"
- "md5 hash"

## Category
security/verification

## Tools

### calculate_checksum
Calculate checksum of text or file.
### verify_checksum
Verify a checksum matches.

## Dependencies
None
