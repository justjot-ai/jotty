---
name: reading-rss-feeds
description: "Fetch and parse RSS/Atom feeds to extract articles, titles, and summaries. Use when the user wants to read RSS, parse feed, get news feed, Atom feed."
---

# Rss Feed Reader Skill

Fetch and parse RSS/Atom feeds to extract articles, titles, and summaries. Use when the user wants to read RSS, parse feed, get news feed, Atom feed.

## Type
base

## Capabilities
- data-fetch

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
- "rss"
- "atom feed"
- "news feed"
- "parse feed"
- "syndication"

## Category
data-analysis

## Tools

### fetch_rss_tool
Fetch and parse an RSS/Atom feed.

**Parameters:**
- `url` (str, required): Feed URL
- `limit` (int, optional): Max items to return (default: 10)

**Returns:**
- `success` (bool)
- `feed_title` (str): Feed title
- `items` (list): Feed entries with title, link, description, date
- `item_count` (int): Number of items returned

## Dependencies
requests
