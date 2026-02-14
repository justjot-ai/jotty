---
name: newsdailydigest
description: "Aggregate news, summarize with AI, format beautifully, send via email. Use when user wants daily news digest."
---

# News Daily Digest Skill

## Description
Composite skill: News aggregation → AI summarization → Beautiful formatting → Email delivery

## Type
composite

## Capabilities
- data-fetch
- analyze
- communicate

## Triggers
- "send daily news"
- "news digest"
- "summarize today's news"

## Category
communication

## Base Skills
- web-search
- claude-cli-llm
- document-converter
- email-sender

## Execution Mode
sequential

## Tools

### news_daily_digest_tool
Aggregate news, summarize, format, send email (all-in-one).

**Parameters:**
- `topics` (list, required): News topics to cover (e.g., ["AI", "Tech", "Business"])
- `email` (str, required): Recipient email address
- `max_articles_per_topic` (int, optional): Max articles per topic. Default: 5

**Returns:**
- `success` (bool): Whether digest was sent
- `topics_covered` (int): Number of topics
- `articles_analyzed` (int): Total articles analyzed
- `email_sent` (bool): Whether email was delivered

## Usage Examples
```python
result = news_daily_digest_tool({
    'topics': ['AI', 'Startups', 'Climate Tech'],
    'email': 'user@example.com',
    'max_articles_per_topic': 10
})
```
