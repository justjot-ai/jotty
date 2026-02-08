# Product Launch Pipeline Composite Skill

Complete product launch workflow: domain brainstorming → lead research → competitor analysis → content creation.

## Description

This composite skill combines:
1. **Domain Brainstorming** (Source): domain-name-brainstormer
2. **Lead Research** (Processor): lead-research-assistant
3. **Competitor Analysis** (Processor): competitive-ads-extractor
4. **Content Creation** (Sink): content-research-writer


## Type
composite

## Base Skills
- web-search
- claude-cli-llm
- slack
- telegram-sender

## Execution
sequential


## Capabilities
- document
- code

## Usage

```python
from skills.product_launch_pipeline.tools import product_launch_pipeline_tool

result = await product_launch_pipeline_tool({
    'product_description': 'AI-powered code review tool',
    'industry': 'Software Development',
    'competitor_names': ['CodeRabbit', 'DeepCode'],
    'content_action': 'outline',
    'max_leads': 20
})
```

## Parameters

- `product_description` (str, required): Product description
- `product_name` (str, optional): Product name (will brainstorm if not provided)
- `industry` (str, optional): Target industry
- `location` (str, optional): Geographic location
- `competitor_names` (list, optional): List of competitor names
- `content_action` (str, optional): 'outline', 'draft', 'full' (default: 'outline')
- `max_domain_suggestions` (int, optional): Max domain suggestions (default: 10)
- `max_leads` (int, optional): Max leads (default: 10)
- `max_ads_per_competitor` (int, optional): Max ads per competitor (default: 5)
- `brainstorm_domains` (bool, optional): Brainstorm domains (default: True)
- `research_leads` (bool, optional): Research leads (default: True)
- `analyze_competitors` (bool, optional): Analyze competitors (default: True)
- `write_content` (bool, optional): Write content (default: True)

## Architecture

Source → Processor → Processor → Sink pattern:
- **Source**: Domain name brainstormer
- **Processor**: Lead research → Competitor analysis
- **Sink**: Content research writer

No code duplication - reuses existing skills.
