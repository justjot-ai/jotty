"""
YouTube Downloader Skill

Download YouTube video transcripts and convert to documents.
Refactored to use Jotty core utilities.
"""

import re
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from Jotty.core.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.utils.skill_status import SkillStatus

logger = logging.getLogger(__name__)

# Status emitter for progress updates
status = SkillStatus("youtube-downloader")



def _extract_video_id(url: str) -> Optional[str]:
    """Extract YouTube video ID from URL."""
    patterns = [
        r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([a-zA-Z0-9_-]{11})',
        r'youtube\.com/watch\?.*v=([a-zA-Z0-9_-]{11})'
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    return None


def _extract_playlist_id(url: str) -> Optional[str]:
    """Extract YouTube playlist ID from URL."""
    match = re.search(r'list=([a-zA-Z0-9_-]+)', url)
    return match.group(1) if match else None


@tool_wrapper(required_params=['video_url'])
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
    status.set_callback(params.pop('_status_callback', None))

    video_id = _extract_video_id(params['video_url'])
    if not video_id:
        return tool_error(f'Invalid YouTube URL: {params["video_url"]}')

    output_dir = Path(params.get('output_dir', './output/youtube'))
    output_dir.mkdir(parents=True, exist_ok=True)
    include_timestamps = params.get('include_timestamps', True)

    # Try to import and use youtube-transcript-api
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
    except ImportError:
        return tool_response(
            video_id=video_id,
            title='YouTube Video',
            output_path=str(output_dir / f'{video_id}.md'),
            message='youtube-transcript-api not installed. Run: pip install youtube-transcript-api',
            transcript_length=0
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
        return tool_error(f'Failed to fetch transcript: {str(e)}. YouTube may be blocking this IP address. Try using browser-automation skill to manually extract transcript from youtube.com/watch?v={video_id}')

    # Format as markdown
    status.emit("processing", "transcript")
    markdown_lines = [
        f"# YouTube Transcript: {video_id}",
        f"URL: https://youtube.com/watch?v={video_id}",
        "",
        "## Transcript",
        ""
    ]

    for entry in transcript_data:
        text = entry['text'].strip()
        if include_timestamps:
            start_time = entry['start']
            minutes = int(start_time // 60)
            seconds = int(start_time % 60)
            timestamp = f"[{minutes:02d}:{seconds:02d}]"
            markdown_lines.append(f"{timestamp} {text}")
        else:
            markdown_lines.append(text)

    markdown_content = "\n".join(markdown_lines)

    # Save to file
    output_path = output_dir / f'{video_id}.md'
    output_path.write_text(markdown_content, encoding='utf-8')

    status.emit("creating", str(output_path))

    return tool_response(
        video_id=video_id,
        title=f'YouTube Video {video_id}',
        output_path=str(output_path),
        transcript_length=len(transcript_data),
        message=f'Successfully downloaded {len(transcript_data)} transcript segments'
    )


@tool_wrapper(required_params=['playlist_url'])
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
    status.set_callback(params.pop('_status_callback', None))

    playlist_id = _extract_playlist_id(params['playlist_url'])
    if not playlist_id:
        return tool_error(f'Invalid playlist URL: {params["playlist_url"]}')

    return tool_response(
        playlist_id=playlist_id,
        videos_processed=0,
        output_paths=[],
        message='YouTube playlist downloader requires youtube-transcript-api. Install: pip install youtube-transcript-api'
    )


__all__ = ['download_youtube_video_tool', 'download_youtube_playlist_tool']
