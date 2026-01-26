# Reddit Trending → Markdown → JustJot Idea Pipeline

## Overview

This pipeline demonstrates the Source → Processor → Sink pattern:
1. **Source**: Search Reddit for trending topics
2. **Processor**: Format results as markdown
3. **Sink**: Create JustJot idea with the content

## Architecture

```
┌─────────────────┐     ┌──────────────┐     ┌─────────────────┐
│   Source        │ --> │  Processor   │ --> │     Sink        │
│ web-search      │     │ Format MD    │     │ mcp-justjot     │
│ (Reddit filter) │     │              │     │ (create_idea)   │
└─────────────────┘     └──────────────┘     └─────────────────┘
```

## Usage

```python
from skills.reddit_trending_to_justjot.tools import reddit_trending_to_justjot_tool

result = await reddit_trending_to_justjot_tool({
    'topic': 'multi agent systems',
    'title': 'Multi-Agent Systems: Reddit Trends',
    'max_results': 10,
    'tags': ['reddit', 'ai', 'trending']
})
```

## Configuration

### JustJot.ai URL

Set the correct JustJot.ai URL:

```bash
# For cmd.dev (check actual domain)
export JUSTJOT_API_URL="https://justjot.cmd.dev"  # or actual domain

# For local development
export JUSTJOT_API_URL="http://localhost:3000"

# For Docker
export JUSTJOT_API_URL="http://justjot-ai-blue:3000"
```

### Authentication (if required)

```bash
export JUSTJOT_AUTH_TOKEN="your-auth-token"
```

## Test Results

✅ **Reddit Search**: Working
- Successfully searches Reddit using `site:reddit.com` filter
- Filters results to Reddit URLs only
- Falls back to alternative search if needed

✅ **Markdown Formatting**: Working
- Formats Reddit posts as structured markdown
- Includes titles, URLs, and snippets
- Ready for JustJot idea sections

⚠️ **JustJot Creation**: Requires correct URL
- Pipeline structure is correct
- Needs proper JustJot.ai API URL configuration
- Check DNS resolution for cmd.dev domain

## Example Output

The pipeline creates a JustJot idea with:

**Title**: "Multi-Agent Systems: Reddit Trends"

**Section**: "Reddit Trending Posts"
- Markdown formatted list of Reddit posts
- Each post includes title, URL, and snippet
- Organized with headers and separators

**Tags**: ['reddit', 'multi-agent-systems', 'ai', 'trending']

## Troubleshooting

### DNS Resolution Error
```
Failed to resolve 'justjot.ai.cmd.dev'
```
**Solution**: Check the actual cmd.dev domain:
- Try `justjot.cmd.dev`
- Try `justjot-ai.cmd.dev`
- Or check your cmd.dev dashboard for the correct URL

### No Reddit Results
**Solution**: 
- Try broader search terms
- Check if DuckDuckGo search is working
- Verify network connectivity

### Authentication Errors
**Solution**:
- Set `JUSTJOT_AUTH_TOKEN` environment variable
- Verify token has create_idea permission
- Check if token has expired

## Next Steps

1. ✅ Pipeline created and tested
2. ✅ Reddit search working
3. ✅ Markdown formatting working
4. 🔄 Configure correct JustJot.ai URL
5. 🔄 Test end-to-end with actual JustJot.ai instance
