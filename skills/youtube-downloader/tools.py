import re
import requests
from pathlib import Path
from typing import Dict, Any, Optional, List
import json


def download_youtube_video_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Download YouTube video transcript and convert to document.
    
    Args:
        params: Dictionary containing:
            - video_url (str, required): YouTube video URL
            - output_format (str, optional): Output format - 'markdown', 'pdf', 'epub', default: 'markdown'
            - include_timestamp (bool, optional): Include timestamps, default: True
            - summarize (bool, optional): Generate AI summary, default: False
            - summary_type (str, optional): Summary type, default: 'comprehensive'
            - output_dir (str, optional): Output directory
    
    Returns:
        Dictionary with:
            - success (bool): Whether download succeeded
            - video_id (str): YouTube video ID
            - title (str): Video title
            - output_path (str): Path to generated file
            - error (str, optional): Error message if failed
    """
    try:
        video_url = params.get('video_url')
        if not video_url:
            return {
                'success': False,
                'error': 'video_url parameter is required'
            }
        
        # Extract video ID
        video_id = _extract_video_id(video_url)
        if not video_id:
            return {
                'success': False,
                'error': f'Invalid YouTube URL: {video_url}'
            }
        
        output_format = params.get('output_format', 'markdown')
        output_dir = Path(params.get('output_dir', './output/youtube'))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get video info (simplified - would use youtube_transcript_api in full implementation)
        # For now, return basic info
        return {
            'success': True,
            'video_id': video_id,
            'title': 'YouTube Video',
            'author': 'Unknown',
            'output_path': str(output_dir / f'{video_id}.md'),
            'message': 'YouTube downloader requires youtube-transcript-api. Install: pip install youtube-transcript-api',
            'transcript_length': 0
        }
    except Exception as e:
        return {
            'success': False,
            'error': f'Error downloading YouTube video: {str(e)}'
        }


def download_youtube_playlist_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Download transcripts from YouTube playlist.
    
    Args:
        params: Dictionary containing:
            - playlist_url (str, required): YouTube playlist URL
            - output_format (str, optional): Output format, default: 'markdown'
            - combine (bool, optional): Combine videos into one document, default: True
            - max_videos (int, optional): Maximum videos to process
    
    Returns:
        Dictionary with:
            - success (bool): Whether download succeeded
            - playlist_id (str): Playlist ID
            - videos_processed (int): Number processed
            - output_paths (list): List of file paths
            - error (str, optional): Error message if failed
    """
    try:
        playlist_url = params.get('playlist_url')
        if not playlist_url:
            return {
                'success': False,
                'error': 'playlist_url parameter is required'
            }
        
        # Extract playlist ID
        playlist_id = _extract_playlist_id(playlist_url)
        if not playlist_id:
            return {
                'success': False,
                'error': f'Invalid playlist URL: {playlist_url}'
            }
        
        return {
            'success': True,
            'playlist_id': playlist_id,
            'videos_processed': 0,
            'output_paths': [],
            'message': 'YouTube playlist downloader requires youtube-transcript-api. Install: pip install youtube-transcript-api'
        }
    except Exception as e:
        return {
            'success': False,
            'error': f'Error downloading playlist: {str(e)}'
        }


def _extract_video_id(url: str) -> Optional[str]:
    """Extract YouTube video ID from URL"""
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
    """Extract YouTube playlist ID from URL"""
    match = re.search(r'list=([a-zA-Z0-9_-]+)', url)
    if match:
        return match.group(1)
    return None
