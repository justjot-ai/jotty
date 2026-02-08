# Internal Communications Skill

Helps write internal communications using company-standard formats for status reports, updates, newsletters, FAQs, and more.

## Description

This skill provides templates and guidelines for writing various types of internal communications including 3P updates (Progress/Plans/Problems), company newsletters, FAQs, status reports, leadership updates, project updates, and incident reports.


## Type
base

## Tools

### `write_internal_comm_tool`

Write internal communications using standard formats.

**Parameters:**
- `comm_type` (str, required): Type - '3p_update', 'newsletter', 'faq', 'status_report', 'leadership_update', 'project_update', 'incident_report'
- `content` (dict, required): Content data for the communication
- `format` (str, optional): Output format - 'markdown', 'html', 'plain' (default: 'markdown')
- `tone` (str, optional): Tone - 'professional', 'casual', 'formal' (default: 'professional')

**Returns:**
- `success` (bool): Whether writing succeeded
- `communication` (str): Generated communication content
- `format` (str): Format used
- `error` (str, optional): Error message if failed

## Usage Examples

### 3P Update

```python
result = await write_internal_comm_tool({
    'comm_type': '3p_update',
    'content': {
        'progress': ['Completed feature X', 'Launched beta'],
        'plans': ['Ship feature Y next week', 'Plan Q4 roadmap'],
        'problems': ['Facing scaling challenges', 'Need more resources']
    }
})
```

### Status Report

```python
result = await write_internal_comm_tool({
    'comm_type': 'status_report',
    'content': {
        'project': 'Q4 Initiative',
        'status': 'On track',
        'highlights': ['Milestone 1 complete', 'Team performing well'],
        'risks': ['Resource constraints'],
        'next_steps': ['Complete milestone 2']
    }
})
```

## Dependencies

- `claude-cli-llm`: For generating communications
