---
name: encoding-toolkit
description: "Unified encoding, hashing, security, and identity toolkit. Encode/decode Base64, compute hashes, decode JWTs, generate passwords/passphrases, generate UUIDs/ULIDs, verify checksums, and apply educational ciphers."
---

# Encoding Toolkit

Unified encoding, hashing, security, and identity skill. Consolidates base64-encoder,
hash-calculator, jwt-decoder, password-generator, uuid-generator, checksum-verifier,
and encryption-tool into one coherent toolkit.

## Type
base

## Capabilities
- encode
- decode
- hash
- security
- identity
- crypto

## Triggers
- "encode"
- "decode"
- "base64"
- "hash"
- "sha256"
- "md5"
- "jwt"
- "token"
- "password"
- "passphrase"
- "uuid"
- "ulid"
- "checksum"
- "encrypt"
- "decrypt"
- "cipher"

## Category
security

## Tools

### base64_encode_tool
Encode text to Base64, URL-safe Base64, or hex.

**Parameters:**
- `text` (str, required): Text to encode
- `encoding` (str, optional): Encoding type: base64, base64url, hex (default: base64)

### base64_decode_tool
Decode Base64, URL-safe Base64, or hex string.

**Parameters:**
- `encoded` (str, required): Encoded string
- `encoding` (str, optional): Encoding type: base64, base64url, hex (default: base64)

### hash_tool
Compute hash of text or file (md5, sha1, sha256, sha512, sha384, sha224).

**Parameters:**
- `text` (str, optional): Text to hash
- `file_path` (str, optional): File to hash
- `algorithm` (str, optional): Hash algorithm (default: sha256)

### verify_hash_tool
Verify a hash matches expected value.

**Parameters:**
- `expected_hash` (str, required): Expected hash value
- `text` (str, optional): Text to hash
- `file_path` (str, optional): File to hash
- `algorithm` (str, optional): Hash algorithm (default: sha256)

### decode_jwt_tool
Decode a JWT token without verification, showing header, payload, expiry.

**Parameters:**
- `token` (str, required): JWT token string

### generate_password_tool
Generate cryptographically secure random passwords.

**Parameters:**
- `length` (int, optional): Password length (default: 16, min: 4, max: 128)
- `count` (int, optional): Number of passwords (default: 1)
- `uppercase` (bool, optional): Include uppercase (default: true)
- `lowercase` (bool, optional): Include lowercase (default: true)
- `digits` (bool, optional): Include digits (default: true)
- `symbols` (bool, optional): Include symbols (default: true)

### generate_passphrase_tool
Generate a passphrase from random dictionary words.

**Parameters:**
- `words` (int, optional): Number of words (default: 5, min: 3, max: 12)
- `separator` (str, optional): Word separator (default: -)
- `capitalize` (bool, optional): Capitalize words (default: true)

### generate_uuid_tool
Generate UUID identifiers (v1, v4, v5).

**Parameters:**
- `version` (int, optional): UUID version: 1, 4, 5 (default: 4)
- `count` (int, optional): Number of UUIDs (default: 1, max: 100)
- `namespace` (str, optional): Namespace for v5: dns, url, oid, x500 (default: dns)
- `name` (str, optional): Name for v5 (default: example.com)

### generate_ulid_tool
Generate ULID identifiers (sortable, 128-bit, Crockford Base32).

**Parameters:**
- `count` (int, optional): Number of ULIDs (default: 1, max: 100)

### encryption_tool
Apply educational ciphers: Caesar, Vigenere, ROT13, XOR.

**Parameters:**
- `operation` (str, required): Cipher: caesar, rot13, vigenere, xor
- `text` (str, required): Text to encrypt/decrypt
- `decrypt` (bool, optional): Decrypt mode (default: false)
- `shift` (int, optional): Caesar shift (default: 3)
- `key` (str, optional): Key for Vigenere/XOR

## Dependencies
None
