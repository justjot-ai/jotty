---
name: content-research-writer
description: "This skill acts as your writing partner, helping you research, outline, draft, and refine content while maintaining your unique voice and style. Provides collaborative outlining, research assistance, hook improvement, and section-by-section feedback. Use when the user wants to write article, research and write, create content."
---

# Content Research Writer Skill

Assists in writing high-quality content by conducting research, adding citations, improving hooks, and providing real-time feedback.

## Description

This skill acts as your writing partner, helping you research, outline, draft, and refine content while maintaining your unique voice and style. Provides collaborative outlining, research assistance, hook improvement, and section-by-section feedback.


## Type
derived

## Base Skills
- web-search


## Capabilities
- research
- document


## Triggers
- "write article"
- "research and write"
- "create content"
- "write blog post"

## Category
document-creation

## Tools

### `write_content_with_research_tool`

Assist in writing content with research and feedback.

**Parameters:**
- `topic` (str, required): Topic or title of the content
- `content_type` (str, optional): Type - 'blog_post', 'article', 'newsletter', 'tutorial', 'case_study' (default: 'article')
- `target_audience` (str, optional): Target audience description
- `target_length` (str, optional): Target length - 'short', 'medium', 'long' (default: 'medium')
- `writing_style` (str, optional): Style - 'formal', 'conversational', 'technical' (default: 'conversational')
- `draft_content` (str, optional): Existing draft content to improve
- `action` (str, optional): Action - 'outline', 'research', 'improve_hook', 'review_section', 'full_review' (default: 'outline')
- `section_to_review` (str, optional): Specific section to review
- `research_topics` (list, optional): Topics to research

**Returns:**
- `success` (bool): Whether operation succeeded
- `outline` (str, optional): Generated outline
- `research` (dict, optional): Research findings with citations
- `improved_content` (str, optional): Improved content
- `feedback` (dict, optional): Feedback on content
- `citations` (list, optional): List of citations
- `error` (str, optional): Error message if failed

## Usage Examples

### Create Outline

```python
result = await write_content_with_research_tool({
    'topic': 'AI Impact on Product Management',
    'content_type': 'article',
    'action': 'outline'
})
```

### Research Topics

```python
result = await write_content_with_research_tool({
    'topic': 'AI Impact on Product Management',
    'action': 'research',
    'research_topics': ['productivity gains', 'adoption rates', 'expert opinions']
})
```

### Improve Hook

```python
result = await write_content_with_research_tool({
    'topic': 'AI Impact on Product Management',
    'action': 'improve_hook',
    'draft_content': 'Product management is changing because of AI...'
})
```

## Dependencies

- `web-search`: For research
- `claude-cli-llm`: For content generation and feedback
