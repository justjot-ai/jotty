---
name: testing-regex
description: "Test, match, and explain regular expressions. Use when the user wants to test regex, match pattern, extract matches, regex."
---

# Regex Tester Skill

Test, match, and explain regular expressions. Use when the user wants to test regex, match pattern, extract matches, regex.

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
- "regex"
- "regular expression"
- "pattern match"
- "test regex"
- "re.match"

## Category
development

## Tools

### regex_match_tool
Test a regex pattern against text.

**Parameters:**
- `pattern` (str, required): Regular expression pattern
- `text` (str, required): Text to match against
- `flags` (str, optional): Flags: i=ignorecase, m=multiline, s=dotall

**Returns:**
- `success` (bool)
- `matches` (list): All matches with groups and positions

## Dependencies
None
