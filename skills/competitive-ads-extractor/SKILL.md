# Competitive Ads Extractor Skill

Extracts and analyzes competitors' ads from ad libraries to understand messaging and creative approaches that resonate.

## Description

This skill helps you understand your competitors' advertising strategies by extracting ads from ad libraries (Facebook, Google, etc.) and analyzing messaging, creative approaches, and targeting strategies.


## Type
derived

## Base Skills
- web-search

## Tools

### `extract_competitive_ads_tool`

Extract and analyze competitor ads.

**Parameters:**
- `competitor_name` (str, required): Name of competitor company
- `platforms` (list, optional): Platforms to search (default: ['facebook', 'google'])
- `max_ads` (int, optional): Maximum ads to extract (default: 20)
- `analysis_depth` (str, optional): Analysis depth - 'basic', 'detailed' (default: 'detailed')
- `output_file` (str, optional): Path to save analysis report

**Returns:**
- `success` (bool): Whether extraction succeeded
- `ads` (list): List of extracted ads with metadata
- `insights` (dict): Analysis insights (messaging, creative, targeting)
- `output_file` (str, optional): Path to saved report
- `error` (str, optional): Error message if failed

## Usage Examples

### Basic Usage

```python
result = await extract_competitive_ads_tool({
    'competitor_name': 'Competitor Inc',
    'max_ads': 20
})
```

### Multi-Platform Analysis

```python
result = await extract_competitive_ads_tool({
    'competitor_name': 'Competitor Inc',
    'platforms': ['facebook', 'google', 'linkedin'],
    'analysis_depth': 'detailed'
})
```

## Dependencies

- `web-search`: For finding ad libraries
- `web-scraper`: For extracting ad content
- `claude-cli-llm`: For analyzing ad strategies
