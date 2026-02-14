---
name: building-email-templates
description: "Create HTML email templates with inline CSS for cross-client compatibility. Use when the user wants to build email, create email template, HTML email."
---

# Email Template Builder Skill

Create HTML email templates with inline CSS for cross-client compatibility. Use when the user wants to build email, create email template, HTML email.

## Type
base

## Capabilities
- generate
- document

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
- "email template"
- "html email"
- "newsletter"
- "email builder"
- "inline css email"

## Category
content-creation

## Tools

### build_email_template_tool
Build an HTML email template with inline CSS.

**Parameters:**
- `subject` (str, required): Email subject
- `body` (str, required): Email body text or HTML
- `template` (str, optional): Template style: basic, newsletter, promotional, transactional (default: basic)
- `brand_color` (str, optional): Primary brand color hex (default: #007bff)
- `footer_text` (str, optional): Footer text
- `preheader` (str, optional): Preheader text

**Returns:**
- `success` (bool)
- `html` (str): Complete HTML email
- `subject` (str): Email subject

## Dependencies
None
