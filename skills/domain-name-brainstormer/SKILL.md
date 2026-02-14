---
name: domain-name-brainstormer
description: "This skill helps you find the perfect domain name for your project by generating creative options and checking what's actually available to register. Saves hours of brainstorming and manual checking. Use when the user wants to generate."
---

# Domain Name Brainstormer Skill

Generates creative domain name ideas for your project and checks availability across multiple TLDs (.com, .io, .dev, .ai, etc.).

## Description

This skill helps you find the perfect domain name for your project by generating creative options and checking what's actually available to register. Saves hours of brainstorming and manual checking.


## Type
derived

## Base Skills
- web-search


## Capabilities
- research

## Tools

### `brainstorm_domains_tool`

Generate domain name suggestions and check availability.

**Parameters:**
- `project_description` (str, required): Description of your project/product
- `keywords` (list, optional): Specific keywords to include
- `preferred_tlds` (list, optional): Preferred TLDs (default: ['.com', '.io', '.dev', '.ai'])
- `max_suggestions` (int, optional): Maximum suggestions (default: 15)
- `check_availability` (bool, optional): Check domain availability (default: True)

**Returns:**
- `success` (bool): Whether generation succeeded
- `suggestions` (list): List of domain suggestions with availability
- `recommendations` (dict): Top recommendations with reasoning
- `error` (str, optional): Error message if failed

## Usage Examples

### Basic Usage

```python
result = await brainstorm_domains_tool({
    'project_description': 'AI-powered code review tool for developers',
    'max_suggestions': 15
})
```

### With Keywords

```python
result = await brainstorm_domains_tool({
    'project_description': 'Design agency',
    'keywords': ['pixel', 'studio'],
    'preferred_tlds': ['.com', '.design']
})
```

## Dependencies

- `claude-cli-llm`: For creative domain name generation
- `http-client`: For checking domain availability (optional)

## Triggers
- "domain name brainstormer"
- "generate"

## Category
workflow-automation
