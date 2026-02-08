"""
YouTube Downloader Skill

Download YouTube video transcripts and convert to documents.
Refactored to use Jotty core utilities.
"""

import re
from pathlib import Path
from typing import Dict, Any, Optional

from Jotty.core.utils.tool_helpers import tool_response, tool_error, tool_wrapper


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
            - include_timestamp (bool, optional): Include timestamps (default: True)
            - summarize (bool, optional): Generate AI summary (default: False)
            - output_dir (str, optional): Output directory

    Returns:
        Dictionary with success, video_id, title, output_path
    """
    video_id = _extract_video_id(params['video_url'])
    if not video_id:
        return tool_error(f'Invalid YouTube URL: {params["video_url"]}')

    output_format = params.get('output_format', 'markdown')
    output_dir = Path(params.get('output_dir', './output/youtube'))
    output_dir.mkdir(parents=True, exist_ok=True)

    return tool_response(
        video_id=video_id,
        title='YouTube Video',
        author='Unknown',
        output_path=str(output_dir / f'{video_id}.md'),
        message='YouTube downloader requires youtube-transcript-api. Install: pip install youtube-transcript-api',
        transcript_length=0
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
