# Research → Content → Notion Composite Skill

Complete research workflow that combines research, content writing, and Notion documentation.

## Description

This composite skill combines:
1. **Research** (Source): lead-research-assistant, competitive-ads-extractor, or web-search
2. **Content Writing** (Processor): content-research-writer
3. **Notion Documentation** (Sink): notion-research-documentation

## Usage

```python
from skills.research_to_notion.tools import research_to_notion_tool

# Research leads and document
result = await research_to_notion_tool({
    'research_type': 'leads',
    'research_query': 'AI code review tools',
    'product_description': 'AI-powered code review tool',
    'content_action': 'outline',
    'create_notion_page': True
})

# Competitive research
result = await research_to_notion_tool({
    'research_type': 'competitive',
    'research_query': 'Competitor Inc',
    'competitor_name': 'Competitor Inc',
    'content_action': 'draft'
})
```

## Parameters

- `research_type` (str, required): 'leads', 'competitive', or 'topic'
- `research_query` (str, required): Query for research
- `product_description` (str, optional): For lead research
- `competitor_name` (str, optional): For competitive research
- `content_action` (str, optional): 'outline', 'draft', 'full' (default: 'outline')
- `notion_output_format` (str, optional): 'brief', 'detailed', 'comprehensive' (default: 'detailed')
- `max_leads` (int, optional): Max leads (default: 10)
- `max_ads` (int, optional): Max ads (default: 5)
- `create_notion_page` (bool, optional): Create Notion page (default: True)

## Architecture

Source → Processor → Sink pattern:
- **Source**: Research skills (leads/competitive/topic)
- **Processor**: Content research writer
- **Sink**: Notion research documentation

No code duplication - reuses existing skills.
