---
name: anonymizing-data
description: "Anonymize PII by masking emails, phone numbers, names, IPs, and credit card numbers. Use when the user wants to anonymize, mask PII, redact data."
---

# Data Anonymizer Skill

Anonymize PII by masking emails, phone numbers, names, IPs, and credit card numbers. Use when the user wants to anonymize, mask PII, redact data.

## Type
base

## Capabilities
- analyze

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
- "anonymize"
- "mask"
- "redact"
- "PII"
- "remove personal data"

## Category
data-analysis

## Tools

### anonymize_text_tool
Anonymize PII in text.

**Parameters:**
- `text` (str, required): Text containing PII
- `mask_emails` (bool, optional): Mask email addresses (default: true)
- `mask_phones` (bool, optional): Mask phone numbers (default: true)
- `mask_ips` (bool, optional): Mask IP addresses (default: true)
- `mask_credit_cards` (bool, optional): Mask credit card numbers (default: true)

**Returns:**
- `success` (bool)
- `anonymized` (str): Text with PII masked
- `detections` (dict): Count of each PII type found

## Dependencies
None
