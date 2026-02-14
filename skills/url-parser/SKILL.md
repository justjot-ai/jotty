---
name: parsing-urls
description: "Parse, build, and manipulate URLs — extract components, add query params. Use when the user wants to parse URL, extract domain, build URL."
---

# Url Parser Skill

Parse, build, and manipulate URLs — extract components, add query params. Use when the user wants to parse URL, extract domain, build URL.

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
- "url"
- "parse url"
- "domain"
- "query string"
- "url encode"

## Category
development

## Tools

### parse_url_tool
Parse a URL into its components.

**Parameters:**
- `url` (str, required): URL to parse

**Returns:**
- `success` (bool)
- `scheme` (str): Protocol
- `host` (str): Hostname
- `port` (int): Port number
- `path` (str): URL path
- `query` (dict): Query parameters
- `fragment` (str): Fragment identifier

## Dependencies
None
