---
name: hashing-data
description: "Compute and verify MD5, SHA-256, SHA-512, and bcrypt hashes for files and strings. Use when the user wants to hash, checksum, verify integrity."
---

# Hash Calculator Skill

Compute and verify MD5, SHA-256, SHA-512, and bcrypt hashes for files and strings. Use when the user wants to hash, checksum, verify integrity.

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
- "hash"
- "checksum"
- "md5"
- "sha256"
- "sha512"
- "verify hash"

## Category
development

## Tools

### hash_tool
Compute hash of text or file.

**Parameters:**
- `text` (str, optional): Text to hash
- `file_path` (str, optional): File to hash
- `algorithm` (str, optional): md5, sha1, sha256, sha512 (default: sha256)

**Returns:**
- `success` (bool): Whether hashing succeeded
- `hash` (str): Hex digest
- `algorithm` (str): Algorithm used

### verify_hash_tool
Verify a hash matches expected value.

**Parameters:**
- `text` (str, optional): Text to verify
- `file_path` (str, optional): File to verify
- `expected_hash` (str, required): Expected hash value
- `algorithm` (str, optional): Algorithm (default: sha256)

**Returns:**
- `success` (bool): Whether verification succeeded
- `match` (bool): Whether hashes match

## Dependencies
None
