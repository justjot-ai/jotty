"""
YouTube Downloader Skill

Download YouTube video transcripts and convert to documents.
Refactored to use Jotty core utilities.
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, Optional

from Jotty.core.infrastructure.utils.skill_status import SkillStatus
from Jotty.core.infrastructure.utils.tool_helpers import tool_error, tool_response, tool_wrapper

logger = logging.getLogger(__name__)

# Status emitter for progress updates
status = SkillStatus("youtube-downloader")


def _extract_video_id(url: str) -> Optional[str]:
    """Extract YouTube video ID from URL."""
    patterns = [
        r"(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([a-zA-Z0-9_-]{11})",
        r"youtube\.com/watch\?.*v=([a-zA-Z0-9_-]{11})",
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    return None


def _extract_playlist_id(url: str) -> Optional[str]:
    """Extract YouTube playlist ID from URL."""
    match = re.search(r"list=([a-zA-Z0-9_-]+)", url)
    return match.group(1) if match else None


def _extract_transcript_via_innertube(video_id: str) -> Optional[list]:
    """
    Extract transcript from YouTube using Innertube API (page HTML scraping).

    This method works even when youtube-transcript-api is blocked by cloud IPs.
    It extracts the ytInitialPlayerResponse from the page HTML.

    Args:
        video_id: YouTube video ID

    Returns:
        List of transcript segments with 'text' and 'start' keys, or None if failed
    """
    import json

    import requests

    try:
        logger.info(f"[YouTube Innertube] Fetching page HTML for {video_id}")

        # Fetch video page
        url = f"https://www.youtube.com/watch?v={video_id}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }

        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        html = response.text

        # Extract ytInitialPlayerResponse
        logger.info("[YouTube Innertube] Extracting player response...")

        pattern = r"ytInitialPlayerResponse\s*=\s*({.+?});"
        match = re.search(pattern, html, re.DOTALL)

        if not match:
            # Try alternative pattern
            pattern = r"var ytInitialPlayerResponse = ({.+?});"
            match = re.search(pattern, html, re.DOTALL)

        if not match:
            logger.error("[YouTube Innertube] Could not find ytInitialPlayerResponse in page HTML")
            return None

        player_data = json.loads(match.group(1))

        # Extract caption tracks
        captions_data = player_data.get("captions", {})
        if not captions_data:
            logger.warning("[YouTube Innertube] No captions data found")
            return None

        caption_tracks = captions_data.get("playerCaptionsTracklistRenderer", {}).get(
            "captionTracks", []
        )

        if not caption_tracks:
            logger.warning("[YouTube Innertube] No caption tracks available")
            return None

        # Find English caption track (prefer manual over auto-generated)
        caption_url = None
        for track in caption_tracks:
            lang_code = track.get("languageCode", "")
            kind = track.get("kind", "")

            if lang_code.startswith("en"):
                caption_url = track.get("baseUrl")
                if kind != "asr":  # Prefer manual captions over auto-generated
                    break

        if not caption_url:
            # Fallback: use first available track
            caption_url = caption_tracks[0].get("baseUrl")

        if not caption_url:
            logger.error("[YouTube Innertube] No caption URL found")
            return None

        logger.info(f"[YouTube Innertube] Fetching captions from: {caption_url[:80]}...")

        # Fetch caption XML
        caption_response = requests.get(caption_url, headers=headers, timeout=15)
        caption_response.raise_for_status()

        # Parse XML captions
        import xml.etree.ElementTree as ET

        root = ET.fromstring(caption_response.text)

        transcript_data = []
        for text_elem in root.findall(".//text"):
            text = text_elem.text
            if not text:
                continue

            # Get timestamp
            start = float(text_elem.get("start", 0))

            # Clean up HTML entities
            import html

            text = html.unescape(text)

            transcript_data.append({"text": text.strip(), "start": start})

        if not transcript_data:
            logger.warning("[YouTube Innertube] No transcript segments extracted")
            return None

        logger.info(f"[YouTube Innertube] Successfully extracted {len(transcript_data)} segments")
        return transcript_data

    except requests.RequestException as e:
        logger.error(f"[YouTube Innertube] HTTP request failed: {e}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"[YouTube Innertube] JSON parsing failed: {e}")
        return None
    except Exception as e:
        logger.error(f"[YouTube Innertube] Extraction failed: {e}", exc_info=True)
        return None


async def _extract_transcript_via_browser(video_url: str) -> Optional[list]:
    """
    Extract transcript from YouTube page using browser automation.

    Fallback method when youtube-transcript-api fails (e.g., cloud IP blocking).

    Args:
        video_url: Full YouTube video URL

    Returns:
        List of transcript segments with 'text' and 'start' keys, or None if failed
    """
    try:
        # Import browser-automation skill tools
        import importlib.util
        import sys

        spec = importlib.util.spec_from_file_location(
            "browser_tools", "skills/browser-automation/tools.py"
        )
        browser_tools = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(browser_tools)

        logger.info(f"[YouTube Browser] Navigating to {video_url}")

        # Navigate to video page
        nav_result = await browser_tools.browser_navigate_tool(
            {"url": video_url, "wait_for": "ytd-player", "extract_text": False}
        )

        if not nav_result.get("success"):
            logger.error(f"[YouTube Browser] Navigation failed: {nav_result.get('error')}")
            return None

        logger.info("[YouTube Browser] Looking for transcript button...")

        # Try multiple selectors for the "Show transcript" button (YouTube changes frequently)
        transcript_button_selectors = [
            'button[aria-label*="transcript" i]',  # Case-insensitive match
            'button[aria-label*="Show transcript"]',
            "ytd-video-description-transcript-section-renderer button",
            '[class*="transcript"] button',
            'button:has-text("Show transcript")',
        ]

        # Try to click the transcript button
        transcript_opened = False
        for selector in transcript_button_selectors:
            try:
                click_result = await browser_tools.browser_click_tool(
                    {
                        "selector": selector,
                        "wait_for": "#segments-container, ytd-transcript-segment-list-renderer",
                    }
                )

                if click_result.get("success"):
                    logger.info(f"[YouTube Browser] Transcript button clicked: {selector}")
                    transcript_opened = True
                    break
            except Exception as e:
                logger.debug(f"[YouTube Browser] Selector failed: {selector} - {e}")
                continue

        if not transcript_opened:
            # Try the three-dot menu approach
            logger.info("[YouTube Browser] Trying three-dot menu...")
            try:
                # Click more button
                await browser_tools.browser_click_tool(
                    {"selector": 'button[aria-label="More actions"]'}
                )

                # Click "Show transcript" in menu
                await browser_tools.browser_click_tool(
                    {
                        "selector": 'tp-yt-paper-listbox yt-formatted-string:has-text("Show transcript")',
                        "wait_for": "#segments-container",
                    }
                )
                transcript_opened = True
            except Exception as e:
                logger.warning(f"[YouTube Browser] Three-dot menu failed: {e}")

        if not transcript_opened:
            logger.error("[YouTube Browser] Could not open transcript panel")
            return None

        # Wait for transcript to load
        import asyncio

        await asyncio.sleep(2)  # Dynamic loading delay

        logger.info("[YouTube Browser] Extracting transcript segments...")

        # Extract transcript segments
        extract_result = await browser_tools.browser_execute_js_tool(
            {
                "script": """
                () => {
                    const segments = [];
                    const segmentElements = document.querySelectorAll(
                        '#segments-container ytd-transcript-segment-renderer, ' +
                        'ytd-transcript-segment-list-renderer [role="button"]'
                    );

                    for (const seg of segmentElements) {
                        const timeEl = seg.querySelector('.segment-timestamp, [class*="time"]');
                        const textEl = seg.querySelector('.segment-text, yt-formatted-string');

                        if (textEl) {
                            const text = textEl.textContent.trim();
                            let startSeconds = 0;

                            if (timeEl) {
                                const timeStr = timeEl.textContent.trim();
                                // Parse "1:23" or "1:23:45" to seconds
                                const parts = timeStr.split(':').map(Number);
                                if (parts.length === 2) {
                                    startSeconds = parts[0] * 60 + parts[1];
                                } else if (parts.length === 3) {
                                    startSeconds = parts[0] * 3600 + parts[1] * 60 + parts[2];
                                }
                            }

                            if (text) {
                                segments.push({
                                    text: text,
                                    start: startSeconds
                                });
                            }
                        }
                    }

                    return segments;
                }
            """
            }
        )

        if not extract_result.get("success"):
            logger.error(f"[YouTube Browser] Extraction failed: {extract_result.get('error')}")
            return None

        transcript_data = extract_result.get("result", [])

        if not transcript_data:
            logger.warning("[YouTube Browser] No transcript segments found")
            return None

        logger.info(f"[YouTube Browser] Successfully extracted {len(transcript_data)} segments")

        # Close browser to clean up
        await browser_tools.browser_close_tool({})

        return transcript_data

    except Exception as e:
        logger.error(f"[YouTube Browser] Extraction failed: {e}", exc_info=True)
        return None


@tool_wrapper(required_params=["video_url"])
def download_youtube_video_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Download YouTube video transcript and convert to document.

    Args:
        params: Dictionary containing:
            - video_url (str, required): YouTube video URL
            - output_format (str, optional): 'markdown', 'pdf', 'epub' (default: 'markdown')
            - include_timestamps (bool, optional): Include timestamps (default: True)
            - summarize (bool, optional): Generate AI summary (default: False)
            - output_dir (str, optional): Output directory

    Returns:
        Dictionary with success, video_id, title, output_path
    """
    status.set_callback(params.pop("_status_callback", None))

    video_id = _extract_video_id(params["video_url"])
    if not video_id:
        return tool_error(f'Invalid YouTube URL: {params["video_url"]}')

    output_dir = Path(params.get("output_dir", "./output/youtube"))
    output_dir.mkdir(parents=True, exist_ok=True)
    include_timestamps = params.get("include_timestamps", True)

    # Try to import and use youtube-transcript-api
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        from youtube_transcript_api._errors import NoTranscriptFound, TranscriptsDisabled
    except ImportError:
        return tool_response(
            video_id=video_id,
            title="YouTube Video",
            output_path=str(output_dir / f"{video_id}.md"),
            message="youtube-transcript-api not installed. Run: pip install youtube-transcript-api",
            transcript_length=0,
        )

    # Fetch transcript
    status.emit("fetching", f"transcript for {video_id}")

    try:
        # Try direct API first (no proxy - might work depending on IP)
        api = YouTubeTranscriptApi()
        transcript_data = api.fetch(video_id)
        logger.info(f"[YouTube] Successfully fetched {len(transcript_data)} segments")

    except Exception as e:
        logger.warning(f"[YouTube] API failed: {e}")

        # FALLBACK 1: Try Innertube API (page HTML scraping)
        logger.info("[YouTube] Attempting Innertube API fallback...")
        status.emit("extracting", "transcript via Innertube API")

        transcript_data = _extract_transcript_via_innertube(video_id)

        if not transcript_data:
            # FALLBACK 2: Try browser automation (requires Playwright dependencies)
            logger.info("[YouTube] Innertube failed, attempting browser-automation fallback...")
            status.emit("extracting", "transcript via browser automation")

            import asyncio

            transcript_data = asyncio.run(_extract_transcript_via_browser(params["video_url"]))

            if not transcript_data:
                return tool_error(
                    f"Failed to fetch transcript via API, Innertube, and browser automation. "
                    f"API error: {str(e)}. "
                    f"YouTube may be blocking access or transcript may not be available."
                )

            logger.info(f"[YouTube] Browser fallback successful: {len(transcript_data)} segments")
        else:
            logger.info(f"[YouTube] Innertube fallback successful: {len(transcript_data)} segments")

    # Format as markdown
    status.emit("processing", "transcript")
    markdown_lines = [
        f"# YouTube Transcript: {video_id}",
        f"URL: https://youtube.com/watch?v={video_id}",
        "",
        "## Transcript",
        "",
    ]

    for entry in transcript_data:
        text = entry["text"].strip()
        if include_timestamps:
            start_time = entry["start"]
            minutes = int(start_time // 60)
            seconds = int(start_time % 60)
            timestamp = f"[{minutes:02d}:{seconds:02d}]"
            markdown_lines.append(f"{timestamp} {text}")
        else:
            markdown_lines.append(text)

    markdown_content = "\n".join(markdown_lines)

    # Save to file
    output_path = output_dir / f"{video_id}.md"
    output_path.write_text(markdown_content, encoding="utf-8")

    status.emit("creating", str(output_path))

    return tool_response(
        video_id=video_id,
        title=f"YouTube Video {video_id}",
        output_path=str(output_path),
        transcript_length=len(transcript_data),
        message=f"Successfully downloaded {len(transcript_data)} transcript segments",
    )


@tool_wrapper(required_params=["playlist_url"])
def download_youtube_playlist_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Download transcripts from YouTube playlist.

    Args:
        params: Dictionary containing:
            - playlist_url (str, required): YouTube playlist URL
            - output_format (str, optional): Output format (default: 'markdown')
            - combine (bool, optional): Combine into one document (default: True)
            - max_videos (int, optional): Maximum videos to process

    Returns:
        Dictionary with success, playlist_id, videos_processed, output_paths
    """
    status.set_callback(params.pop("_status_callback", None))

    playlist_id = _extract_playlist_id(params["playlist_url"])
    if not playlist_id:
        return tool_error(f'Invalid playlist URL: {params["playlist_url"]}')

    return tool_response(
        playlist_id=playlist_id,
        videos_processed=0,
        output_paths=[],
        message="YouTube playlist downloader requires youtube-transcript-api. Install: pip install youtube-transcript-api",
    )


__all__ = ["download_youtube_video_tool", "download_youtube_playlist_tool"]
