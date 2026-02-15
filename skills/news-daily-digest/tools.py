"""News Daily Digest Skill — aggregate news, summarize, format, and prepare digest."""

import json
import re
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from html import unescape
from pathlib import Path
from typing import Any, Dict, List, Optional

from Jotty.core.infrastructure.utils.skill_status import SkillStatus
from Jotty.core.infrastructure.utils.tool_helpers import tool_error, tool_response, tool_wrapper

status = SkillStatus("news-daily-digest")

# RSS feed sources per topic (free, no API key required)
TOPIC_FEEDS: Dict[str, List[Dict[str, str]]] = {
    "ai": [
        {"name": "MIT Tech Review AI", "url": "https://www.technologyreview.com/feed/"},
        {
            "name": "AI News",
            "url": "https://news.google.com/rss/search?q=artificial+intelligence&hl=en",
        },
    ],
    "tech": [
        {"name": "TechCrunch", "url": "https://techcrunch.com/feed/"},
        {"name": "Ars Technica", "url": "https://feeds.arstechnica.com/arstechnica/index"},
    ],
    "business": [
        {"name": "Business News", "url": "https://news.google.com/rss/search?q=business&hl=en"},
        {
            "name": "Reuters Business",
            "url": "https://news.google.com/rss/search?q=reuters+business&hl=en",
        },
    ],
    "science": [
        {"name": "Science News", "url": "https://news.google.com/rss/search?q=science&hl=en"},
        {"name": "Nature", "url": "https://news.google.com/rss/search?q=nature+science&hl=en"},
    ],
    "startups": [
        {
            "name": "Startup News",
            "url": "https://news.google.com/rss/search?q=startups+funding&hl=en",
        },
    ],
    "climate": [
        {"name": "Climate News", "url": "https://news.google.com/rss/search?q=climate+tech&hl=en"},
    ],
}

# Fallback: unknown topics use Google News RSS search
GOOGLE_NEWS_RSS = "https://news.google.com/rss/search?q={query}&hl=en"


def _clean_html(text: str) -> str:
    """Strip HTML tags and decode entities from text."""
    clean = re.sub(r"<[^>]+>", "", text)
    clean = unescape(clean)
    return " ".join(clean.split()).strip()


def _fetch_rss_articles(feed_url: str, max_articles: int = 5) -> List[Dict[str, str]]:
    """Fetch articles from an RSS feed URL.

    Returns list of dicts with title, link, description, published.
    """
    try:
        req = urllib.request.Request(
            feed_url,
            headers={"User-Agent": "Jotty-NewsDigest/1.0"},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = resp.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, OSError):
        return []

    articles: List[Dict[str, str]] = []
    try:
        root = ET.fromstring(data)
    except ET.ParseError:
        return []

    # Handle RSS 2.0 format
    items = root.findall(".//item")
    # Handle Atom format if no RSS items found
    if not items:
        atom_ns = "http://www.w3.org/2005/Atom"
        items = root.findall(f".//{{{atom_ns}}}entry")

    for item in items[:max_articles]:
        article: Dict[str, str] = {}

        # RSS 2.0 fields
        title_el = item.find("title")
        link_el = item.find("link")
        desc_el = item.find("description")
        pub_el = item.find("pubDate")

        # Atom fallback
        if title_el is None:
            atom_ns = "http://www.w3.org/2005/Atom"
            title_el = item.find(f"{{{atom_ns}}}title")
            link_el_atom = item.find(f"{{{atom_ns}}}link")
            desc_el = item.find(f"{{{atom_ns}}}summary")
            pub_el = item.find(f"{{{atom_ns}}}published")
            if link_el_atom is not None:
                article["link"] = link_el_atom.get("href", "")

        article["title"] = (
            _clean_html(title_el.text) if title_el is not None and title_el.text else "Untitled"
        )
        if "link" not in article:
            article["link"] = link_el.text.strip() if link_el is not None and link_el.text else ""
        article["description"] = (
            _clean_html(desc_el.text)[:300] if desc_el is not None and desc_el.text else ""
        )
        article["published"] = pub_el.text.strip() if pub_el is not None and pub_el.text else ""

        if article["title"] != "Untitled":
            articles.append(article)

    return articles


def _get_feeds_for_topic(topic: str) -> List[Dict[str, str]]:
    """Get feed sources for a given topic, falling back to Google News search."""
    normalized = topic.lower().strip()
    if normalized in TOPIC_FEEDS:
        return TOPIC_FEEDS[normalized]
    # Fallback: construct a Google News RSS search
    query = urllib.request.quote(topic)
    return [{"name": f"Google News: {topic}", "url": GOOGLE_NEWS_RSS.format(query=query)}]


def _summarize_articles(articles: List[Dict[str, str]], topic: str) -> Dict[str, Any]:
    """Create a summary for a topic's articles.

    Uses extractive summarization (first sentences of descriptions).
    """
    if not articles:
        return {
            "topic": topic,
            "article_count": 0,
            "headlines": [],
            "summary": f"No articles found for topic: {topic}",
        }

    headlines = [a["title"] for a in articles]
    descriptions = [a["description"] for a in articles if a.get("description")]

    # Build a brief summary from available descriptions
    summary_parts = []
    for article in articles[:3]:
        desc = article.get("description", "")
        if desc:
            # Take the first sentence
            first_sentence = re.split(r"(?<=[.!?])\s", desc)[0]
            summary_parts.append(f"- {article['title']}: {first_sentence}")
        else:
            summary_parts.append(f"- {article['title']}")

    return {
        "topic": topic,
        "article_count": len(articles),
        "headlines": headlines,
        "summary": "\n".join(summary_parts) if summary_parts else "No summaries available.",
        "articles": articles,
    }


def _format_digest_markdown(
    topic_summaries: List[Dict[str, Any]],
    email: str,
) -> str:
    """Format the full digest as Markdown."""
    now = datetime.now()
    date_str = now.strftime("%A, %B %d, %Y")
    total_articles = sum(s["article_count"] for s in topic_summaries)

    sections = [
        f"# Daily News Digest",
        f"**Date:** {date_str}  ",
        f"**Prepared for:** {email}  ",
        f"**Topics:** {len(topic_summaries)} | **Articles:** {total_articles}",
        "",
        "---",
        "",
    ]

    for ts in topic_summaries:
        sections.append(f"## {ts['topic']}")
        sections.append(f"*{ts['article_count']} article(s)*")
        sections.append("")

        if ts.get("articles"):
            for article in ts["articles"]:
                link = article.get("link", "")
                title = article["title"]
                if link:
                    sections.append(f"### [{title}]({link})")
                else:
                    sections.append(f"### {title}")
                if article.get("description"):
                    sections.append(f"> {article['description'][:200]}")
                if article.get("published"):
                    sections.append(f"*Published: {article['published']}*")
                sections.append("")
        else:
            sections.append(ts["summary"])
            sections.append("")

        sections.append("---")
        sections.append("")

    sections.extend(
        [
            "*This digest was generated by Jotty AI Agent Framework.*",
            "",
        ]
    )
    return "\n".join(sections)


def _format_digest_html(markdown_content: str, email: str) -> str:
    """Wrap digest content in minimal HTML for email delivery."""
    return (
        "<!DOCTYPE html>\n<html><head>"
        "<meta charset='utf-8'>"
        "<title>Daily News Digest</title>"
        "<style>"
        "body{font-family:'Segoe UI',Tahoma,Geneva,Verdana,sans-serif;"
        "max-width:700px;margin:0 auto;padding:20px;color:#333;line-height:1.6;}"
        "h1{color:#1a1a2e;border-bottom:2px solid #e94560;padding-bottom:10px;}"
        "h2{color:#16213e;margin-top:25px;}"
        "h3{color:#0f3460;}"
        "blockquote{border-left:3px solid #e94560;margin:10px 0;padding:5px 15px;"
        "color:#555;background:#f9f9f9;}"
        "hr{border:none;border-top:1px solid #eee;margin:20px 0;}"
        "a{color:#e94560;text-decoration:none;}"
        "a:hover{text-decoration:underline;}"
        ".footer{color:#888;font-size:0.85em;margin-top:30px;}"
        "</style></head><body>\n"
        f"<pre style='white-space:pre-wrap;font-family:inherit;'>{markdown_content}</pre>\n"
        "<div class='footer'>Generated by Jotty AI Agent Framework</div>\n"
        "</body></html>"
    )


def _save_digest(content: str, email: str, fmt: str = "markdown") -> str:
    """Save digest to disk and return the file path."""
    output_dir = Path.home() / "jotty" / "reports" / "news-digest"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_email = re.sub(r"[^a-zA-Z0-9]", "_", email)[:30]
    ext = "html" if fmt == "html" else "md"
    file_path = output_dir / f"digest_{safe_email}_{timestamp}.{ext}"
    file_path.write_text(content, encoding="utf-8")
    return str(file_path)


@tool_wrapper(required_params=["topics", "email"])
def news_daily_digest_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Aggregate news, summarize, format, and prepare digest for email delivery.

    Parameters:
        topics (list, required): News topics to cover (e.g., ["AI", "Tech", "Business"])
        email (str, required): Recipient email address
        max_articles_per_topic (int, optional): Max articles per topic. Default: 5

    Returns:
        success, topics_covered, articles_analyzed, digest_path, email_sent,
        topic_summaries
    """
    status.set_callback(params.pop("_status_callback", None))

    topics = params["topics"]
    email = params["email"]
    max_articles = int(params.get("max_articles_per_topic", 5))

    # Validate topics
    if not isinstance(topics, list) or len(topics) == 0:
        return tool_error("topics must be a non-empty list of strings")

    # Validate email (basic check)
    if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email):
        return tool_error(f"Invalid email address: '{email}'")

    # Clamp max_articles
    max_articles = max(1, min(max_articles, 50))

    topic_summaries: List[Dict[str, Any]] = []
    total_articles = 0

    for topic in topics:
        status.searching(topic)
        feeds = _get_feeds_for_topic(topic)

        all_articles: List[Dict[str, str]] = []
        for feed_info in feeds:
            status.fetching(feed_info["url"])
            articles = _fetch_rss_articles(feed_info["url"], max_articles=max_articles)
            for article in articles:
                article["source"] = feed_info["name"]
            all_articles.extend(articles)

        # Deduplicate by title
        seen_titles: set = set()
        unique_articles: List[Dict[str, str]] = []
        for article in all_articles:
            title_key = article["title"].lower().strip()
            if title_key not in seen_titles:
                seen_titles.add(title_key)
                unique_articles.append(article)

        # Limit to max_articles
        unique_articles = unique_articles[:max_articles]
        total_articles += len(unique_articles)

        # Summarize
        status.analyzing(topic)
        summary = _summarize_articles(unique_articles, topic)
        topic_summaries.append(summary)

    # Format digest
    status.creating("digest")
    digest_markdown = _format_digest_markdown(topic_summaries, email)
    digest_html = _format_digest_html(digest_markdown, email)

    # Save digest locally
    digest_path = _save_digest(digest_html, email, fmt="html")

    # Note: actual email sending would require an email-sender skill or SMTP config.
    # This tool prepares the digest; email delivery is delegated to the composite pipeline.
    email_sent = False
    email_note = (
        "Digest prepared and saved locally. Email delivery requires "
        "email-sender skill or SMTP configuration."
    )

    status.done(f"Digest ready: {len(topic_summaries)} topics, {total_articles} articles")

    return tool_response(
        topics_covered=len(topic_summaries),
        articles_analyzed=total_articles,
        digest_path=digest_path,
        email_sent=email_sent,
        email_note=email_note,
        topic_summaries=[
            {
                "topic": s["topic"],
                "article_count": s["article_count"],
                "headlines": s["headlines"],
            }
            for s in topic_summaries
        ],
    )


__all__ = ["news_daily_digest_tool"]
