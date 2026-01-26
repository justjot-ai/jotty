"""
last30days-claude-cli: Research topics using Claude CLI LM (no API keys required).

This skill uses Claude CLI's WebSearch tool to research topics from the last 30 days
across Reddit, X/Twitter, and the web.
"""

import asyncio
import json
import os
import subprocess
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


async def last30days_claude_cli(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Research a topic from the last 30 days using Claude CLI LM.
    
    Args:
        params: Dictionary containing:
            - topic (required): Topic to research
            - tool (optional): Target tool (e.g., "ChatGPT", "Midjourney")
            - quick (optional): Faster research, fewer sources
            - deep (optional): Comprehensive research
            - emit (optional): Output format: "compact", "json", "md"
    
    Returns:
        Dict with research results
    """
    topic = params.get('topic', '')
    if not topic:
        return {
            'success': False,
            'error': 'Topic is required'
        }
    
    tool = params.get('tool', '')
    quick = params.get('quick', False)
    deep = params.get('deep', False)
    emit = params.get('emit', 'compact')
    
    # Calculate date range (last 30 days)
    to_date = datetime.now()
    from_date = to_date - timedelta(days=30)
    date_range = f"{from_date.strftime('%Y-%m-%d')} to {to_date.strftime('%Y-%m-%d')}"
    
    # Determine search depth
    if quick:
        num_searches = 3
        search_depth = "quick"
    elif deep:
        num_searches = 8
        search_depth = "deep"
    else:
        num_searches = 5
        search_depth = "default"
    
    try:
        # Build research query for Claude CLI
        research_prompt = f"""Research the topic "{topic}" from the last 30 days ({date_range}).

Find discussions, posts, and articles about this topic across:
1. Reddit (site:reddit.com)
2. X/Twitter (site:x.com OR site:twitter.com)
3. Web (blogs, news, documentation)

For each source, provide:
- Title/headline
- URL
- Key insights or quotes
- Date (if available)
- Engagement metrics (upvotes, likes, etc.) if visible

Search queries to use:
- "{topic} site:reddit.com"
- "{topic} site:x.com OR site:twitter.com"
- "{topic} 2026"
- "{topic} latest"
- "{'best ' + topic if 'best' not in topic.lower() else topic}"

Return a structured summary with:
1. Key findings from Reddit
2. Key findings from X/Twitter
3. Key findings from Web
4. Common patterns across all sources
5. Top recommendations or insights

Format the output as JSON with this structure:
{{
  "topic": "{topic}",
  "date_range": "{date_range}",
  "reddit": [
    {{
      "title": "...",
      "url": "...",
      "insights": "...",
      "date": "...",
      "engagement": "..."
    }}
  ],
  "x": [
    {{
      "text": "...",
      "url": "...",
      "author": "...",
      "insights": "...",
      "date": "..."
    }}
  ],
  "web": [
    {{
      "title": "...",
      "url": "...",
      "insights": "...",
      "date": "..."
    }}
  ],
  "patterns": ["pattern1", "pattern2", ...],
  "recommendations": ["rec1", "rec2", ...]
}}"""
        
        # Call Claude CLI with WebSearch tool
        # Use stdin for prompt input
        cmd = [
            'claude',
            '--model', 'sonnet',
            '--print',
            '--output-format', 'json',
            '--tools', 'WebSearch',
            '--'
        ]
        
        logger.info(f"Calling Claude CLI for topic: {topic}")
        
        # Prepare environment (let Claude use credentials file, not API key)
        env = os.environ.copy()
        api_key = env.get('ANTHROPIC_API_KEY', '')
        if api_key.startswith('sk-ant-oat'):
            # Remove OAuth token - let Claude use credentials file
            env.pop('ANTHROPIC_API_KEY', None)
        
        result = subprocess.run(
            cmd,
            input=research_prompt,
            capture_output=True,
            text=True,
            timeout=180,  # 3 minutes timeout
            env=env
        )
        
        if result.returncode != 0:
            logger.error(f"Claude CLI error: {result.stderr}")
            return {
                'success': False,
                'error': f"Claude CLI failed: {result.stderr}",
                'stdout': result.stdout
            }
        
        # Parse response
        try:
            response_data = json.loads(result.stdout.strip())
            
            # Extract structured output if available
            if 'structured_output' in response_data:
                research_data = response_data['structured_output']
            elif 'result' in response_data:
                # Try to parse result as JSON
                result_text = response_data['result']
                try:
                    research_data = json.loads(result_text)
                except (json.JSONDecodeError, TypeError):
                    # If result is text, try to extract JSON from it
                    import re
                    json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
                    if json_match:
                        try:
                            research_data = json.loads(json_match.group())
                        except json.JSONDecodeError:
                            # Fallback: create structured data from text
                            research_data = _parse_text_response(result_text, topic)
                    else:
                        research_data = _parse_text_response(result_text, topic)
            else:
                # Try to extract JSON from raw stdout
                import re
                json_match = re.search(r'\{.*\}', result.stdout, re.DOTALL)
                if json_match:
                    try:
                        research_data = json.loads(json_match.group())
                    except json.JSONDecodeError:
                        research_data = _parse_text_response(result.stdout, topic)
                else:
                    research_data = _parse_text_response(result.stdout, topic)
            
            # Format output based on emit mode
            if emit == 'json':
                return {
                    'success': True,
                    'output': research_data,
                    'format': 'json'
                }
            elif emit == 'md':
                # Convert to markdown
                markdown = _format_as_markdown(topic, date_range, research_data, tool)
                return {
                    'success': True,
                    'output': markdown,
                    'format': 'md'
                }
            else:  # compact
                compact = _format_as_compact(topic, date_range, research_data, tool)
                return {
                    'success': True,
                    'output': compact,
                    'format': 'compact'
                }
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Claude CLI response: {e}")
            return {
                'success': False,
                'error': f"Failed to parse response: {e}",
                'raw_output': result.stdout
            }
            
    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'error': 'Research timed out after 3 minutes'
        }
    except Exception as e:
        logger.error(f"Error in last30days_claude_cli: {e}", exc_info=True)
        return {
            'success': False,
            'error': str(e)
        }


def _format_as_markdown(topic: str, date_range: str, data: Dict[str, Any], tool: str = '') -> str:
    """Format research data as markdown."""
    lines = [
        f"# Research Results: \"{topic}\"",
        f"",
        f"**Date Range:** {date_range}",
        f"**Mode:** Claude CLI WebSearch",
        f"",
    ]
    
    if tool:
        lines.append(f"**Target Tool:** {tool}")
        lines.append("")
    
    # Reddit section
    if 'reddit' in data and data['reddit']:
        lines.append("## Reddit Discussions")
        lines.append("")
        for item in data['reddit'][:10]:  # Limit to top 10
            lines.append(f"### {item.get('title', 'Untitled')}")
            if item.get('url'):
                lines.append(f"**URL:** {item['url']}")
            if item.get('insights'):
                lines.append(f"**Insights:** {item['insights']}")
            if item.get('engagement'):
                lines.append(f"**Engagement:** {item['engagement']}")
            lines.append("")
    
    # X section
    if 'x' in data and data['x']:
        lines.append("## X/Twitter Posts")
        lines.append("")
        for item in data['x'][:10]:  # Limit to top 10
            if item.get('text'):
                lines.append(f"**{item.get('author', 'Unknown')}:** {item['text'][:200]}...")
            if item.get('url'):
                lines.append(f"**URL:** {item['url']}")
            lines.append("")
    
    # Web section
    if 'web' in data and data['web']:
        lines.append("## Web Sources")
        lines.append("")
        for item in data['web'][:10]:  # Limit to top 10
            lines.append(f"### {item.get('title', 'Untitled')}")
            if item.get('url'):
                lines.append(f"**URL:** {item['url']}")
            if item.get('insights'):
                lines.append(f"**Insights:** {item['insights']}")
            lines.append("")
    
    # Patterns
    if 'patterns' in data and data['patterns']:
        lines.append("## Key Patterns")
        lines.append("")
        for pattern in data['patterns']:
            lines.append(f"- {pattern}")
        lines.append("")
    
    # Recommendations
    if 'recommendations' in data and data['recommendations']:
        lines.append("## Recommendations")
        lines.append("")
        for rec in data['recommendations']:
            lines.append(f"- {rec}")
        lines.append("")
    
    return "\n".join(lines)


def _parse_text_response(text: str, topic: str) -> Dict[str, Any]:
    """Parse Claude CLI text response into structured data."""
    # Extract URLs and content from text
    import re
    
    reddit_items = []
    x_items = []
    web_items = []
    patterns = []
    
    # Look for Reddit URLs
    reddit_urls = re.findall(r'https?://(?:www\.)?reddit\.com/r/[^\s\)]+', text)
    for url in reddit_urls[:10]:
        reddit_items.append({
            'url': url,
            'title': f'Reddit discussion about {topic}',
            'insights': 'Found via WebSearch'
        })
    
    # Look for X/Twitter URLs
    x_urls = re.findall(r'https?://(?:www\.)?(?:x\.com|twitter\.com)/[^\s\)]+', text)
    for url in x_urls[:10]:
        x_items.append({
            'url': url,
            'text': f'Post about {topic}',
            'author': 'Unknown'
        })
    
    # Extract key sentences as patterns
    sentences = re.split(r'[.!?]\s+', text)
    for sentence in sentences[:10]:
        if len(sentence) > 20 and topic.lower() in sentence.lower():
            patterns.append(sentence.strip())
    
    return {
        'topic': topic,
        'reddit': reddit_items,
        'x': x_items,
        'web': web_items,
        'patterns': patterns[:5],
        'recommendations': patterns[:3],
        'raw_text': text[:1000]  # Keep first 1000 chars for reference
    }


def _format_as_compact(topic: str, date_range: str, data: Dict[str, Any], tool: str = '') -> str:
    """Format research data as compact text."""
    lines = [
        f"Research Results: \"{topic}\"",
        f"Date Range: {date_range}",
        f"Mode: Claude CLI WebSearch",
        "",
    ]
    
    if tool:
        lines.append(f"Target Tool: {tool}")
        lines.append("")
    
    # Summary counts
    reddit_count = len(data.get('reddit', []))
    x_count = len(data.get('x', []))
    web_count = len(data.get('web', []))
    
    lines.append(f"Sources found: {reddit_count} Reddit, {x_count} X, {web_count} Web")
    lines.append("")
    
    # Key patterns
    if 'patterns' in data and data['patterns']:
        lines.append("Key Patterns:")
        for pattern in data['patterns'][:5]:
            lines.append(f"  • {pattern}")
        lines.append("")
    
    # Top recommendations
    if 'recommendations' in data and data['recommendations']:
        lines.append("Top Recommendations:")
        for rec in data['recommendations'][:5]:
            lines.append(f"  • {rec}")
        lines.append("")
    
    lines.append("---")
    lines.append("Use --emit=json or --emit=md for detailed output")
    
    return "\n".join(lines)


# Export tool function
__all__ = ['last30days_claude_cli']
