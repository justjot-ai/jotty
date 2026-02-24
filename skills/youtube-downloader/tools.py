"""
YouTube Downloader Skill

Download YouTube video transcripts and convert to documents.
Uses a 5-level fallback chain to handle YouTube's aggressive cloud-IP blocking:

  Level 1: youtube_transcript_api (direct API, works on non-cloud IPs)
  Level 2: Innertube HTML scraping (extract captionTracks from page source)
  Level 3: yt-dlp subtitle extraction (requires deno JS runtime)
  Level 4: Third-party transcript API (Supadata.ai, configurable via env)
  Level 5: Playwright browser automation (last resort)
"""

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

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


# ---------------------------------------------------------------------------
# Level 2: Innertube HTML scraping
# ---------------------------------------------------------------------------


def _extract_transcript_via_innertube(video_id: str) -> Optional[List[dict]]:
    """
    Extract transcript from YouTube using Innertube API (page HTML scraping).

    This method works when youtube-transcript-api is blocked but the page HTML
    still contains captionTracks data. Cloud IPs often strip this data, so this
    level may also fail on cloud servers.
    """
    import html as html_mod
    import json
    import xml.etree.ElementTree as ET

    import requests

    try:
        logger.info(f"[YouTube L2-Innertube] Fetching page HTML for {video_id}")

        url = f"https://www.youtube.com/watch?v={video_id}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }

        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        page_html = response.text

        # Extract ytInitialPlayerResponse
        pattern = r"ytInitialPlayerResponse\s*=\s*({.+?});"
        match = re.search(pattern, page_html, re.DOTALL)
        if not match:
            pattern = r"var ytInitialPlayerResponse = ({.+?});"
            match = re.search(pattern, page_html, re.DOTALL)
        if not match:
            logger.warning("[YouTube L2-Innertube] No ytInitialPlayerResponse in page HTML")
            return None

        player_data = json.loads(match.group(1))
        captions_data = player_data.get("captions", {})
        if not captions_data:
            logger.warning(
                "[YouTube L2-Innertube] No captions data (cloud IP may have stripped it)"
            )
            return None

        caption_tracks = captions_data.get("playerCaptionsTracklistRenderer", {}).get(
            "captionTracks", []
        )
        if not caption_tracks:
            logger.warning("[YouTube L2-Innertube] 0 caption tracks available")
            return None

        # Find English caption track (prefer manual over auto-generated)
        caption_url = None
        for track in caption_tracks:
            lang_code = track.get("languageCode", "")
            kind = track.get("kind", "")
            if lang_code.startswith("en"):
                caption_url = track.get("baseUrl")
                if kind != "asr":
                    break

        if not caption_url:
            caption_url = caption_tracks[0].get("baseUrl")
        if not caption_url:
            logger.error("[YouTube L2-Innertube] No caption URL found")
            return None

        logger.info(f"[YouTube L2-Innertube] Fetching captions from: {caption_url[:80]}...")
        caption_response = requests.get(caption_url, headers=headers, timeout=15)
        caption_response.raise_for_status()

        root = ET.fromstring(caption_response.text)
        transcript_data = []
        for text_elem in root.findall(".//text"):
            text = text_elem.text
            if not text:
                continue
            start = float(text_elem.get("start", 0))
            text = html_mod.unescape(text)
            transcript_data.append({"text": text.strip(), "start": start})

        if not transcript_data:
            logger.warning("[YouTube L2-Innertube] No transcript segments extracted")
            return None

        logger.info(f"[YouTube L2-Innertube] Extracted {len(transcript_data)} segments")
        return transcript_data

    except Exception as e:
        logger.warning(f"[YouTube L2-Innertube] Failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Level 3: yt-dlp subtitle extraction
# ---------------------------------------------------------------------------


def _extract_transcript_via_ytdlp(video_id: str) -> Optional[List[dict]]:
    """
    Extract transcript using yt-dlp command line tool.

    Tries multiple player client variants. Requires yt-dlp installed and
    optionally deno JS runtime for solving YouTube's JavaScript challenges.
    """
    import shutil
    import subprocess
    import tempfile

    ytdlp_path = shutil.which("yt-dlp")
    if not ytdlp_path:
        logger.info("[YouTube L3-ytdlp] yt-dlp not installed, skipping")
        return None

    # Check for deno (needed for JS challenges)
    deno_path = shutil.which("deno") or os.path.expanduser("~/.deno/bin/deno")
    js_args = []
    if os.path.isfile(deno_path):
        js_args = ["--js-runtimes", f"deno:{deno_path}"]

    # Check for cookie file
    cookie_args = []
    cookie_file = os.environ.get("YOUTUBE_COOKIES_FILE", "")
    if cookie_file and os.path.isfile(cookie_file):
        cookie_args = ["--cookies", cookie_file]

    # Try different player clients
    clients = ["web_creator", "android", "web"]
    video_url = f"https://www.youtube.com/watch?v={video_id}"

    with tempfile.TemporaryDirectory() as tmpdir:
        for client in clients:
            try:
                output_template = os.path.join(tmpdir, f"yt_{client}_%(id)s")
                cmd = [
                    ytdlp_path,
                    *js_args,
                    *cookie_args,
                    "--extractor-args",
                    f"youtube:player_client={client}",
                    "--write-sub",
                    "--write-auto-sub",
                    "--sub-lang",
                    "en",
                    "--skip-download",
                    "--sub-format",
                    "vtt",
                    "-o",
                    output_template,
                    video_url,
                ]

                logger.info(f"[YouTube L3-ytdlp] Trying client={client}")
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

                # Check for VTT files
                for f in Path(tmpdir).glob("*.vtt"):
                    transcript_data = _parse_vtt_file(str(f))
                    if transcript_data:
                        logger.info(
                            f"[YouTube L3-ytdlp] client={client} extracted "
                            f"{len(transcript_data)} segments"
                        )
                        return transcript_data

                if "Sign in to confirm" in result.stderr:
                    logger.info(f"[YouTube L3-ytdlp] client={client}: bot detection triggered")
                    continue

            except subprocess.TimeoutExpired:
                logger.warning(f"[YouTube L3-ytdlp] client={client}: timeout")
            except Exception as e:
                logger.warning(f"[YouTube L3-ytdlp] client={client}: {e}")

    logger.info("[YouTube L3-ytdlp] All clients failed")
    return None


def _parse_vtt_file(vtt_path: str) -> Optional[List[dict]]:
    """Parse a WebVTT subtitle file into transcript segments."""
    try:
        content = Path(vtt_path).read_text(encoding="utf-8")
        segments = []
        seen_texts = set()

        # Parse VTT timestamps and text
        # Format: 00:00:01.234 --> 00:00:04.567\nText here
        pattern = r"(\d{2}:\d{2}:\d{2}\.\d{3})\s*-->\s*\d{2}:\d{2}:\d{2}\.\d{3}\s*\n(.+?)(?=\n\n|\n\d{2}:|\Z)"
        matches = re.findall(pattern, content, re.DOTALL)

        for timestamp, text in matches:
            # Parse timestamp to seconds
            parts = timestamp.split(":")
            seconds = int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])

            # Clean text (remove VTT formatting tags)
            clean_text = re.sub(r"<[^>]+>", "", text).strip()
            clean_text = re.sub(r"\s+", " ", clean_text)

            if clean_text and clean_text not in seen_texts:
                seen_texts.add(clean_text)
                segments.append({"text": clean_text, "start": seconds})

        return segments if segments else None

    except Exception as e:
        logger.warning(f"[YouTube VTT] Parse error: {e}")
        return None


# ---------------------------------------------------------------------------
# Level 4: Third-party transcript API
# ---------------------------------------------------------------------------


def _extract_transcript_via_third_party(video_id: str) -> Optional[List[dict]]:
    """
    Extract transcript using a third-party transcript API.

    Supported services (configure via environment variables):
      - SUPADATA_API_KEY: Supadata.ai (100 free req/month)
        Sign up at https://supadata.ai

    These services have their own residential IP infrastructure and are not
    affected by YouTube's cloud-IP blocking.
    """
    import json

    import requests

    # Try Supadata.ai
    supadata_key = os.environ.get("SUPADATA_API_KEY", "")
    if supadata_key:
        try:
            logger.info("[YouTube L4-API] Trying Supadata.ai")
            video_url = f"https://www.youtube.com/watch?v={video_id}"
            r = requests.get(
                "https://api.supadata.ai/v1/transcript",
                params={"url": video_url, "lang": "en"},
                headers={"x-api-key": supadata_key},
                timeout=30,
            )

            if r.status_code == 200:
                data = r.json()
                # Supadata returns {lang, content: [{text, offset (ms), duration, lang}]}
                content = data.get("content", data.get("transcript", []))
                if isinstance(content, list) and content:
                    transcript_data = []
                    for item in content:
                        text = item.get("text", "").strip()
                        # offset is in milliseconds, convert to seconds
                        offset_ms = item.get("offset", item.get("start", 0))
                        start = float(offset_ms) / 1000.0 if offset_ms > 100 else float(offset_ms)
                        if text:
                            transcript_data.append({"text": text, "start": start})

                    if transcript_data:
                        logger.info(
                            f"[YouTube L4-API] Supadata extracted "
                            f"{len(transcript_data)} segments"
                        )
                        return transcript_data

                # Supadata may return text as a single string
                if isinstance(content, str) and content:
                    logger.info(
                        f"[YouTube L4-API] Supadata returned plain text ({len(content)} chars)"
                    )
                    return [{"text": content, "start": 0.0}]

            elif r.status_code == 401:
                logger.warning("[YouTube L4-API] Supadata: invalid API key")
            elif r.status_code == 429:
                logger.warning("[YouTube L4-API] Supadata: rate limit exceeded")
            else:
                logger.warning(f"[YouTube L4-API] Supadata: HTTP {r.status_code}")

        except Exception as e:
            logger.warning(f"[YouTube L4-API] Supadata failed: {e}")

    # Try custom transcript API endpoint (user-configurable)
    custom_api_url = os.environ.get("YOUTUBE_TRANSCRIPT_API_URL", "")
    custom_api_key = os.environ.get("YOUTUBE_TRANSCRIPT_API_KEY", "")
    if custom_api_url:
        try:
            logger.info(f"[YouTube L4-API] Trying custom API: {custom_api_url[:50]}")
            headers = {}
            if custom_api_key:
                headers["Authorization"] = f"Bearer {custom_api_key}"
                headers["x-api-key"] = custom_api_key

            r = requests.get(
                custom_api_url,
                params={"video_id": video_id, "lang": "en"},
                headers=headers,
                timeout=30,
            )

            if r.status_code == 200:
                data = r.json()
                # Flexible response parsing
                content = (
                    data.get("transcript", [])
                    or data.get("content", [])
                    or data.get("segments", [])
                    or data.get("data", [])
                )
                if isinstance(content, list) and content:
                    transcript_data = []
                    for item in content:
                        if isinstance(item, dict):
                            text = item.get("text", item.get("content", "")).strip()
                            start = float(
                                item.get("start", item.get("offset", item.get("time", 0)))
                            )
                        else:
                            text = str(item).strip()
                            start = 0.0
                        if text:
                            transcript_data.append({"text": text, "start": start})
                    if transcript_data:
                        logger.info(
                            f"[YouTube L4-API] Custom API extracted "
                            f"{len(transcript_data)} segments"
                        )
                        return transcript_data

        except Exception as e:
            logger.warning(f"[YouTube L4-API] Custom API failed: {e}")

    if not supadata_key and not custom_api_url:
        logger.info(
            "[YouTube L4-API] No API keys configured. "
            "Set SUPADATA_API_KEY in .env (free at https://supadata.ai)"
        )

    return None


# ---------------------------------------------------------------------------
# Level 5: Playwright browser automation
# ---------------------------------------------------------------------------


async def _extract_transcript_via_browser(video_url: str) -> Optional[List[dict]]:
    """
    Extract transcript from YouTube using Playwright browser automation.

    Last resort: uses a full headless browser to navigate to YouTube,
    open the transcript panel, and extract text from the DOM.

    Note: On cloud IPs, YouTube may block even browser-based caption access.
    In that case, this method will attempt to extract whatever page content
    is available as context for the video.
    """
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        logger.info("[YouTube L5-Browser] Playwright not installed, skipping")
        return None

    import asyncio

    try:
        logger.info(f"[YouTube L5-Browser] Navigating to {video_url}")

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                ),
                viewport={"width": 1280, "height": 720},
                locale="en-US",
            )
            page = await context.new_page()

            await page.goto(video_url, wait_until="domcontentloaded", timeout=30000)

            # Accept cookies consent
            try:
                consent = page.locator('button:has-text("Accept all")')
                if await consent.count() > 0:
                    await consent.first.click(timeout=3000)
                    await asyncio.sleep(1)
            except Exception:
                pass

            await asyncio.sleep(2)

            # Try to expand description and click "Show transcript"
            try:
                expand_btn = page.locator("#expand")
                if await expand_btn.count() > 0:
                    await expand_btn.first.click(timeout=3000)
                    await asyncio.sleep(1)
            except Exception:
                pass

            try:
                transcript_btn = page.locator('button:has-text("Show transcript")')
                if await transcript_btn.count() > 0:
                    await transcript_btn.first.click(timeout=5000)
                    await asyncio.sleep(3)

                    # Click "Transcript" tab (not "Chapters")
                    try:
                        tab = page.locator('[role="tab"]:has-text("Transcript")')
                        if await tab.count() > 0:
                            await tab.first.click(timeout=3000)
                            await asyncio.sleep(2)
                    except Exception:
                        pass

                    # Extract transcript segments from DOM
                    segments = await page.evaluate(
                        """() => {
                        var results = [];
                        var selectors = [
                            "ytd-transcript-segment-renderer",
                            "#segments-container [class*='segment']",
                        ];

                        for (var i = 0; i < selectors.length; i++) {
                            var elements = document.querySelectorAll(selectors[i]);
                            if (elements.length > 0) {
                                for (var j = 0; j < elements.length; j++) {
                                    var el = elements[j];
                                    var timeEl = el.querySelector("[class*='time'], .segment-timestamp");
                                    var textEl = el.querySelector(".segment-text, yt-formatted-string");

                                    var text = textEl ? textEl.textContent.trim() : el.textContent.trim();
                                    var startSeconds = 0;

                                    if (timeEl) {
                                        var timeStr = timeEl.textContent.trim();
                                        var parts = timeStr.split(":").map(Number);
                                        if (parts.length === 2) startSeconds = parts[0] * 60 + parts[1];
                                        else if (parts.length === 3) startSeconds = parts[0] * 3600 + parts[1] * 60 + parts[2];
                                    }

                                    if (text) results.push({text: text, start: startSeconds});
                                }
                                break;
                            }
                        }
                        return results;
                    }"""
                    )

                    if segments:
                        logger.info(
                            f"[YouTube L5-Browser] Extracted {len(segments)} segments from DOM"
                        )
                        await browser.close()
                        return segments

            except Exception as e:
                logger.warning(f"[YouTube L5-Browser] Transcript panel failed: {e}")

            # Fallback: extract video description as minimal context
            description = await page.evaluate(
                """() => {
                var desc = document.querySelector("#description-inline-expander, #description");
                return desc ? desc.textContent.trim() : "";
            }"""
            )

            title = await page.title()
            await browser.close()

            if description and len(description) > 50:
                logger.info(
                    f"[YouTube L5-Browser] No transcript, extracted description ({len(description)} chars)"
                )
                return [{"text": f"[Video: {title}]\n{description}", "start": 0.0}]

            logger.warning("[YouTube L5-Browser] Could not extract any content")
            return None

    except Exception as e:
        logger.warning(f"[YouTube L5-Browser] Failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Main transcript extraction with 5-level fallback
# ---------------------------------------------------------------------------


def _extract_transcript(video_id: str, video_url: str) -> Optional[List[dict]]:
    """
    Extract YouTube transcript using 5-level fallback chain.

    Level 1: youtube_transcript_api (direct Innertube API)
    Level 2: HTML scraping (extract captionTracks from page source)
    Level 3: yt-dlp subtitle download (with JS runtime + cookie support)
    Level 4: Third-party API (Supadata.ai or custom, needs API key)
    Level 5: Playwright browser automation (headless Chromium)

    Returns:
        List of dicts with 'text' and 'start' keys, or None if all levels fail.
    """
    errors = []

    # Level 1: youtube_transcript_api
    try:
        from youtube_transcript_api import YouTubeTranscriptApi

        status.emit("fetching", f"transcript via API (L1)")
        api = YouTubeTranscriptApi()
        transcript_obj = api.fetch(video_id)
        # Convert to list of dicts
        segments = [{"text": seg.text, "start": seg.start} for seg in transcript_obj.snippets]
        if segments:
            logger.info(f"[YouTube] L1 (API) succeeded: {len(segments)} segments")
            return segments
    except ImportError:
        errors.append("L1: youtube_transcript_api not installed")
    except Exception as e:
        errors.append(f"L1: {str(e)[:80]}")
        logger.info(f"[YouTube] L1 (API) failed: {e}")

    # Level 2: Innertube HTML scraping
    status.emit("extracting", "transcript via Innertube (L2)")
    result = _extract_transcript_via_innertube(video_id)
    if result:
        logger.info(f"[YouTube] L2 (Innertube) succeeded: {len(result)} segments")
        return result
    errors.append("L2: Innertube scraping failed (captionTracks empty or blocked)")

    # Level 3: yt-dlp
    status.emit("extracting", "transcript via yt-dlp (L3)")
    result = _extract_transcript_via_ytdlp(video_id)
    if result:
        logger.info(f"[YouTube] L3 (yt-dlp) succeeded: {len(result)} segments")
        return result
    errors.append("L3: yt-dlp failed (bot detection or not installed)")

    # Level 4: Third-party API
    status.emit("extracting", "transcript via API service (L4)")
    result = _extract_transcript_via_third_party(video_id)
    if result:
        logger.info(f"[YouTube] L4 (third-party API) succeeded: {len(result)} segments")
        return result
    errors.append("L4: No third-party API configured or all failed")

    # Level 5: Playwright browser
    status.emit("extracting", "transcript via browser (L5)")
    import asyncio

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as pool:
                result = pool.submit(
                    asyncio.run, _extract_transcript_via_browser(video_url)
                ).result(timeout=60)
        else:
            result = asyncio.run(_extract_transcript_via_browser(video_url))
    except Exception as e:
        result = None
        errors.append(f"L5: Playwright failed ({e})")

    if result:
        logger.info(f"[YouTube] L5 (Playwright) succeeded: {len(result)} segments")
        return result
    errors.append("L5: Playwright browser extraction failed")

    # All levels failed
    logger.error(
        f"[YouTube] All 5 fallback levels failed for {video_id}:\n"
        + "\n".join(f"  {e}" for e in errors)
    )
    return None


# ---------------------------------------------------------------------------
# Public tool functions
# ---------------------------------------------------------------------------


@tool_wrapper(required_params=["video_url"])
def download_youtube_video_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Download YouTube video transcript and convert to document.

    Uses 5-level fallback chain to extract transcripts even from cloud servers
    where YouTube blocks direct API access.

    Args:
        params: Dictionary containing:
            - video_url (str, required): YouTube video URL
            - output_format (str, optional): 'markdown', 'pdf', 'epub' (default: 'markdown')
            - include_timestamps (bool, optional): Include timestamps (default: True)
            - summarize (bool, optional): Generate AI summary (default: False)
            - output_dir (str, optional): Output directory

    Returns:
        Dictionary with success, video_id, title, output_path

    Environment variables for cloud-IP bypass:
        SUPADATA_API_KEY: Supadata.ai API key (free at https://supadata.ai)
        YOUTUBE_TRANSCRIPT_API_URL: Custom transcript API endpoint
        YOUTUBE_TRANSCRIPT_API_KEY: API key for custom endpoint
        YOUTUBE_COOKIES_FILE: Path to Netscape-format cookie file for yt-dlp
    """
    status.set_callback(params.pop("_status_callback", None))

    video_id = _extract_video_id(params["video_url"])
    if not video_id:
        return tool_error(f'Invalid YouTube URL: {params["video_url"]}')

    output_dir = Path(params.get("output_dir", "./output/youtube"))
    output_dir.mkdir(parents=True, exist_ok=True)
    include_timestamps = params.get("include_timestamps", True)

    # Extract transcript using 5-level fallback
    status.emit("fetching", f"transcript for {video_id}")
    transcript_data = _extract_transcript(video_id, params["video_url"])

    if not transcript_data:
        return tool_error(
            f"Failed to extract transcript for {video_id}. "
            f"All 5 fallback levels failed. "
            f"For cloud servers, set SUPADATA_API_KEY in .env "
            f"(free at https://supadata.ai, 100 req/month)."
        )

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
            start_time = entry.get("start", 0)
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
        message="YouTube playlist downloader requires youtube-transcript-api. "
        "Install: pip install youtube-transcript-api",
    )


__all__ = ["download_youtube_video_tool", "download_youtube_playlist_tool"]
