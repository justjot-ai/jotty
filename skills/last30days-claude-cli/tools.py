"""
last30days-claude-cli: Research topics using Jotty's web search tools (no API keys required).

This skill uses Jotty's built-in web-search skill to research topics from the last 30 days
across Reddit, X/Twitter, and the web.
"""

import asyncio
import json
import logging
import re
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# Import Jotty's web search tool
search_web_tool = None
fetch_webpage_tool = None

try:
    # Method 1: Try registry-based import
    from Jotty.core.registry.skills_registry import get_skills_registry
    registry = get_skills_registry()
    if registry:
        web_skill = registry.get_skill('web-search')
        if web_skill and hasattr(web_skill, 'tools'):
            search_web_tool = web_skill.tools.get('search_web_tool')
            fetch_webpage_tool = web_skill.tools.get('fetch_webpage_tool')
            if search_web_tool:
                logger.info("Loaded web-search tools from registry")
except Exception as e:
    logger.debug(f"Registry import failed: {e}")

if not search_web_tool:
    try:
        # Method 2: Direct file import
        import importlib.util
        from pathlib import Path as PathLib
        web_tools_path = PathLib(__file__).parent.parent / 'web-search' / 'tools.py'
        if web_tools_path.exists():
            spec = importlib.util.spec_from_file_location("web_search_tools", web_tools_path)
            web_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(web_module)
            search_web_tool = getattr(web_module, 'search_web_tool', None)
            fetch_webpage_tool = getattr(web_module, 'fetch_webpage_tool', None)
            if search_web_tool:
                logger.info("Loaded web-search tools via importlib")
    except Exception as e:
        logger.debug(f"Importlib import failed: {e}")

if not search_web_tool:
    logger.warning("Could not import web-search tools, will use fallback")


async def last30days_claude_cli(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Research a topic from the last 30 days using Jotty's web search tools.
    
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
    
    if not search_web_tool:
        return {
            'success': False,
            'error': 'Web search tool not available. Ensure web-search skill is installed.'
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
        max_results_per_query = 5
        num_queries = 3
    elif deep:
        max_results_per_query = 15
        num_queries = 8
    else:
        max_results_per_query = 10
        num_queries = 5
    
    try:
        logger.info(f"Researching topic: {topic} (last 30 days)")
        
        # Build search queries - use simpler queries that work better with DuckDuckGo
        queries = []
        
        # Reddit searches (simpler format)
        queries.append({
            'query': f'{topic} reddit',
            'type': 'reddit',
            'max_results': max_results_per_query
        })
        
        # X/Twitter searches
        queries.append({
            'query': f'{topic} twitter',
            'type': 'x',
            'max_results': max_results_per_query
        })
        queries.append({
            'query': f'{topic} x.com',
            'type': 'x',
            'max_results': max_results_per_query
        })
        
        # Web searches
        queries.append({
            'query': f'{topic} 2026',
            'type': 'web',
            'max_results': max_results_per_query
        })
        queries.append({
            'query': f'{topic} latest',
            'type': 'web',
            'max_results': max_results_per_query
        })
        
        if not quick:
            queries.append({
                'query': f'best {topic}' if 'best' not in topic.lower() else topic,
                'type': 'web',
                'max_results': max_results_per_query
            })
            queries.append({
                'query': f'{topic} news',
                'type': 'web',
                'max_results': max_results_per_query
            })
        
        # Limit number of queries based on depth
        queries = queries[:num_queries]
        
        # Execute searches in parallel
        reddit_items = []
        x_items = []
        web_items = []
        
        search_tasks = []
        for q in queries:
            search_tasks.append(_search_async(q))
        
        # Wait for all searches to complete
        results = await asyncio.gather(*search_tasks, return_exceptions=True)
        
        # Process results
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Search failed: {result}")
                continue
            
            if not result.get('success'):
                continue
            
            query_type = queries[i]['type']
            search_results = result.get('results', [])
            
            for item in search_results:
                url = item.get('url', '')
                title = item.get('title', '')
                snippet = item.get('snippet', '')
                
                # Categorize by URL and query type
                # Priority: URL pattern > query type
                if 'reddit.com' in url.lower() or '/r/' in url.lower():
                    reddit_items.append({
                        'title': title,
                        'url': url,
                        'insights': snippet,
                        'date': _extract_date_from_url(url),
                        'engagement': None
                    })
                elif 'x.com' in url.lower() or 'twitter.com' in url.lower() or '/status/' in url.lower():
                    x_items.append({
                        'text': snippet or title,
                        'url': url,
                        'author': _extract_author_from_url(url),
                        'insights': snippet,
                        'date': _extract_date_from_url(url)
                    })
                elif query_type == 'reddit' and ('reddit' in title.lower() or 'reddit' in snippet.lower()):
                    # Query was for Reddit but URL doesn't match - still categorize as Reddit
                    reddit_items.append({
                        'title': title,
                        'url': url,
                        'insights': snippet,
                        'date': _extract_date_from_url(url),
                        'engagement': None
                    })
                elif query_type == 'x' and ('twitter' in title.lower() or 'x.com' in title.lower() or 'twitter' in snippet.lower()):
                    # Query was for X but URL doesn't match - still categorize as X
                    x_items.append({
                        'text': snippet or title,
                        'url': url,
                        'author': _extract_author_from_url(url),
                        'insights': snippet,
                        'date': _extract_date_from_url(url)
                    })
                else:
                    # Everything else is web
                    web_items.append({
                        'title': title,
                        'url': url,
                        'insights': snippet,
                        'date': _extract_date_from_url(url)
                    })
        
        # Remove duplicates by URL
        reddit_items = _deduplicate_by_url(reddit_items)
        x_items = _deduplicate_by_url(x_items)
        web_items = _deduplicate_by_url(web_items)
        
        # Extract patterns and recommendations
        patterns = _extract_patterns(reddit_items + x_items + web_items, topic)
        recommendations = patterns[:5]  # Top 5 patterns as recommendations
        
        # Build structured data
        research_data = {
            'topic': topic,
            'date_range': date_range,
            'reddit': reddit_items[:20],  # Limit to top 20
            'x': x_items[:20],
            'web': web_items[:20],
            'patterns': patterns[:10],
            'recommendations': recommendations
        }
        
        # Format output based on emit mode
        if emit == 'json':
            return {
                'success': True,
                'output': research_data,
                'format': 'json'
            }
        elif emit == 'md':
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
            
    except Exception as e:
        logger.error(f"Error in last30days_claude_cli: {e}", exc_info=True)
        return {
            'success': False,
            'error': str(e)
        }


async def _search_async(query_info: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a web search asynchronously."""
    if not search_web_tool:
        return {'success': False, 'error': 'Web search tool not available'}
    
    # Run in thread pool since search_web_tool is synchronous
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        search_web_tool,
        {'query': query_info['query'], 'max_results': query_info['max_results']}
    )


def _extract_date_from_url(url: str) -> Optional[str]:
    """Try to extract date from URL (e.g., reddit URLs sometimes have dates)."""
    # Reddit URLs: /r/subreddit/comments/id/title/
    # X URLs: /username/status/id
    # Most URLs don't have dates, return None
    return None


def _extract_author_from_url(url: str) -> str:
    """Extract author/username from URL."""
    # X/Twitter: https://x.com/username/status/...
    match = re.search(r'(?:x\.com|twitter\.com)/([^/]+)', url)
    if match:
        return match.group(1)
    return 'Unknown'


def _deduplicate_by_url(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove duplicate items by URL."""
    seen_urls = set()
    unique_items = []
    for item in items:
        url = item.get('url', '')
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique_items.append(item)
    return unique_items


def _extract_patterns(items: List[Dict[str, Any]], topic: str) -> List[str]:
    """Extract common patterns and insights from search results."""
    patterns = []
    
    # Collect all insights/snippets and titles
    texts = []
    for item in items:
        text = item.get('insights', '') or item.get('text', '') or item.get('title', '')
        if text:
            texts.append(text.lower())
    
    # Extract meaningful phrases (2-3 word combinations)
    phrase_freq = {}
    stop_words = {'that', 'this', 'with', 'from', 'have', 'been', 'will', 'would', 'your', 'their', 'there', 'these', 'those', 'which', 'what', 'when', 'where', 'about', 'after', 'before'}
    
    for text in texts:
        # Extract 2-word phrases
        words = re.findall(r'\b\w{3,}\b', text)
        for i in range(len(words) - 1):
            word1, word2 = words[i], words[i+1]
            if word1 not in stop_words and word2 not in stop_words:
                phrase = f"{word1} {word2}"
                phrase_freq[phrase] = phrase_freq.get(phrase, 0) + 1
    
    # Get top phrases
    top_phrases = sorted(phrase_freq.items(), key=lambda x: x[1], reverse=True)[:15]
    
    # Filter out topic-related phrases (they're obvious)
    topic_words = set(topic.lower().split())
    
    for phrase, count in top_phrases:
        phrase_words = set(phrase.split())
        # Skip if phrase is just topic words
        if phrase_words.issubset(topic_words):
            continue
        if count >= 2:  # Appears in at least 2 sources
            patterns.append(f"{phrase} (mentioned {count}x)")
    
    # If we don't have enough patterns, add single important words
    if len(patterns) < 5:
        word_freq = {}
        for text in texts:
            words = re.findall(r'\b\w{5,}\b', text)  # Words with 5+ chars
            for word in words:
                if word not in stop_words and word not in topic_words:
                    word_freq[word] = word_freq.get(word, 0) + 1
        
        top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        for word, count in top_words:
            if count >= 2 and word not in [p.split()[0] for p in patterns]:
                patterns.append(f"{word} (mentioned {count}x)")
    
    return patterns[:10]  # Return top 10 patterns


def _format_as_markdown(topic: str, date_range: str, data: Dict[str, Any], tool: str = '') -> str:
    """Format research data as markdown."""
    lines = [
        f"# Research Results: \"{topic}\"",
        f"",
        f"**Date Range:** {date_range}",
        f"**Mode:** Jotty Web Search (DuckDuckGo)",
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




def _format_as_compact(topic: str, date_range: str, data: Dict[str, Any], tool: str = '') -> str:
    """Format research data as compact text."""
    lines = [
        f"Research Results: \"{topic}\"",
        f"Date Range: {date_range}",
        f"Mode: Jotty Web Search (DuckDuckGo)",
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


# Export tool function - registry looks for functions ending in _tool or callable attributes
last30days_claude_cli_tool = last30days_claude_cli

__all__ = ['last30days_claude_cli', 'last30days_claude_cli_tool']
