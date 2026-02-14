---
name: notion-meeting-intelligence
description: "This skill helps prepare for meetings by gathering context from Notion, enriching with research, and creating both internal pre-reads and external-facing agendas saved to Notion. Use when the user wants to save to notion, notion page, create notion."
---

# Notion Meeting Intelligence Skill

Prepares meeting materials by gathering context from Notion and creating comprehensive pre-reads and agendas.

## Description

This skill helps prepare for meetings by gathering context from Notion, enriching with research, and creating both internal pre-reads and external-facing agendas saved to Notion.


## Type
composite

## Base Skills
- voice
- claude-cli-llm
- notion

## Execution
sequential


## Capabilities
- analyze
- document

## Tools

### `prepare_meeting_materials_tool`

Prepare meeting materials from Notion context.

**Parameters:**
- `meeting_topic` (str, required): Meeting topic/title
- `meeting_type` (str, optional): Type - 'decision', 'brainstorm', 'status_update', 'customer', 'one_on_one' (default: 'status_update')
- `attendees` (list, optional): List of attendee names
- `related_project` (str, optional): Related project name or page ID
- `search_queries` (list, optional): Queries to search Notion for context
- `create_pre_read` (bool, optional): Create internal pre-read (default: True)
- `create_agenda` (bool, optional): Create external agenda (default: True)

**Returns:**
- `success` (bool): Whether preparation succeeded
- `pre_read_page_id` (str, optional): Pre-read page ID
- `agenda_page_id` (str, optional): Agenda page ID
- `context_found` (dict): Context gathered from Notion
- `error` (str, optional): Error message if failed

## Usage Examples

### Prepare Decision Meeting

```python
result = await prepare_meeting_materials_tool({
    'meeting_topic': 'Q4 Product Roadmap',
    'meeting_type': 'decision',
    'related_project': 'product-roadmap',
    'create_pre_read': True,
    'create_agenda': True
})
```

## Dependencies

- `notion`: For Notion API integration
- `claude-cli-llm`: For enriching with research

## Triggers
- "notion meeting intelligence"
- "save to notion"
- "notion page"
- "create notion"
- "meeting notes"
- "meeting summary"
- "meeting insights"

## Category
communication
