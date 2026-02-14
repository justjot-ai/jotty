---
name: mcp-justjot
description: "Create, manage, and organize ideas on JustJot.ai via REST API — CRUD for ideas, templates, sections, and tags. Use when the user wants to create."
---

# mcp-justjot

Create, manage, and organize ideas on JustJot.ai via REST API — CRUD for ideas, templates, sections, and tags.

## Type
base

## Capabilities
- data-fetch
- document
- communicate

## Use When
User wants to create, list, update, delete, or organize ideas on JustJot.ai

## Tools

### Ideas
- `list_ideas_tool` — List all ideas
- `create_idea_tool` — Create a new idea
- `get_idea_tool` — Get idea by ID
- `update_idea_tool` — Update an existing idea
- `delete_idea_tool` — Delete an idea
- `get_ideas_by_tag_tool` — Get ideas filtered by tag

### Templates
- `list_templates_tool` — List templates
- `get_template_tool` — Get template by ID or name

### Sections
- `add_section_tool` — Add section to idea
- `update_section_tool` — Update section in idea

### Tags
- `list_tags_tool` — List all tags

## Configuration

| Variable | Description | Required |
|---|---|---|
| `JUSTJOT_API_URL` | API base URL (default: https://justjot.ai) | No |
| `JUSTJOT_API_KEY` or `CLERK_SECRET_KEY` | Clerk API key for service auth | No |
| `JUSTJOT_USER_ID` | User ID for service auth | No |
| `JUSTJOT_AUTH_TOKEN` | Bearer token (fallback) | No |

## Triggers
- "mcp justjot"
- "create"

## Category
development
