"""
Video Downloader Skill - Download videos from YouTube and other platforms.

Supports various formats, quality options, and batch downloads.
"""
import asyncio
import logging
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import os
import json

from Jotty.core.infrastructure.utils.skill_status import SkillStatus
from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, async_tool_wrapper

logger = logging.getLogger(__name__)

# Check for yt-dlp or youtube-dl

# Status emitter for progress updates
status = SkillStatus("video-downloader")

try:
    import yt_dlp
    YT_DLP_AVAILABLE = True
except ImportError:
    YT_DLP_AVAILABLE = False
    logger.warning("yt-dlp not available, checking for youtube-dl...")
    try:
        import youtube_dl
        YOUTUBE_DL_AVAILABLE = True
    except ImportError:
        YOUTUBE_DL_AVAILABLE = False
        logger.warning("Neither yt-dlp nor youtube-dl available")


@async_tool_wrapper()
async def download_video_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Download video from URL.
    
    Args:
        params:
            - video_url (str): URL of video
            - output_path (str, optional): Output directory
            - quality (str, optional): Quality preference
            - format (str, optional): Format preference
            - extract_audio (bool, optional): Audio only
            - download_thumbnail (bool, optional): Download thumbnail
            - download_metadata (bool, optional): Save metadata
    
    Returns:
        Dictionary with download results
    """
    status.set_callback(params.pop('_status_callback', None))

    video_url = params.get('video_url', '')
    output_path = params.get('output_path', None)
    quality = params.get('quality', 'best')
    format_pref = params.get('format', 'auto')
    extract_audio = params.get('extract_audio', False)
    download_thumbnail = params.get('download_thumbnail', True)
    download_metadata = params.get('download_metadata', True)
    
    if not video_url:
        return {
            'success': False,
            'error': 'video_url is required'
        }
    
    if not YT_DLP_AVAILABLE and not YOUTUBE_DL_AVAILABLE:
        return {
            'success': False,
            'error': 'yt-dlp or youtube-dl not available. Install with: pip install yt-dlp'
        }
    
    # Determine output directory
    if not output_path:
        output_path = Path.home() / 'Downloads'
    else:
        output_path = Path(os.path.expanduser(output_path))
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        if YT_DLP_AVAILABLE:
            result = await _download_with_ytdlp(
                video_url, output_path, quality, format_pref,
                extract_audio, download_thumbnail, download_metadata
            )
        else:
            result = await _download_with_youtubedl(
                video_url, output_path, quality, format_pref,
                extract_audio, download_thumbnail, download_metadata
            )
        
        return result
        
    except Exception as e:
        logger.error(f"Video download failed: {e}", exc_info=True)
        return {
            'success': False,
            'error': str(e)
        }


async def _download_with_ytdlp(
    video_url: str,
    output_path: Path,
    quality: str,
    format_pref: str,
    extract_audio: bool,
    download_thumbnail: bool,
    download_metadata: bool
) -> Dict:
    """Download using yt-dlp."""
    
    import yt_dlp
    
    # Configure options
    ydl_opts = {
        'outtmpl': str(output_path / '%(title)s.%(ext)s'),
        'quiet': False,
        'no_warnings': False,
    }
    
    # Quality selection
    if extract_audio:
        ydl_opts['format'] = 'bestaudio/best'
        ydl_opts['postprocessors'] = [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': format_pref if format_pref != 'auto' else 'mp3',
            'preferredquality': '192',
        }]
    else:
        if quality == 'best':
            ydl_opts['format'] = 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best'
        elif quality == 'worst':
            ydl_opts['format'] = 'worst'
        elif quality in ['720p', '1080p', '4k']:
            ydl_opts['format'] = f'bestvideo[height<={quality.replace("p", "")}]+bestaudio/best[height<={quality.replace("p", "")}]'
        else:
            ydl_opts['format'] = 'best'
    
    # Thumbnail
    if download_thumbnail:
        ydl_opts['writethumbnail'] = True
    
    # Metadata
    if download_metadata:
        ydl_opts['writedescription'] = True
        ydl_opts['writeinfojson'] = True
    
    # Download
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=True)
        
        # Get downloaded file
        title = info.get('title', 'video')
        ext = info.get('ext', 'mp4')
        if extract_audio:
            ext = format_pref if format_pref != 'auto' else 'mp3'
        
        video_file = output_path / f"{title}.{ext}"
        
        # Find thumbnail
        thumbnail_file = None
        if download_thumbnail:
            thumbnail_ext = info.get('thumbnail', '').split('.')[-1] if info.get('thumbnail') else 'jpg'
            thumbnail_file = output_path / f"{title}.{thumbnail_ext}"
            if not thumbnail_file.exists():
                # Try common extensions
                for ext in ['jpg', 'png', 'webp']:
                    thumb = output_path / f"{title}.{ext}"
                    if thumb.exists():
                        thumbnail_file = thumb
                        break
        
        # Get metadata
        metadata = {
            'title': info.get('title', ''),
            'description': info.get('description', ''),
            'duration': info.get('duration', 0),
            'uploader': info.get('uploader', ''),
            'upload_date': info.get('upload_date', ''),
            'view_count': info.get('view_count', 0),
            'like_count': info.get('like_count', 0)
        }
        
        file_size = video_file.stat().st_size if video_file.exists() else 0
        
        return {
            'success': True,
            'video_path': str(video_file),
            'thumbnail_path': str(thumbnail_file) if thumbnail_file and thumbnail_file.exists() else None,
            'metadata': metadata,
            'file_size': file_size
        }


async def _download_with_youtubedl(
    video_url: str,
    output_path: Path,
    quality: str,
    format_pref: str,
    extract_audio: bool,
    download_thumbnail: bool,
    download_metadata: bool
) -> Dict:
    """Download using youtube-dl (fallback)."""
    
    import youtube_dl
    
    ydl_opts = {
        'outtmpl': str(output_path / '%(title)s.%(ext)s'),
        'quiet': False,
    }
    
    if extract_audio:
        ydl_opts['format'] = 'bestaudio/best'
        ydl_opts['postprocessors'] = [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': format_pref if format_pref != 'auto' else 'mp3',
        }]
    else:
        ydl_opts['format'] = 'best' if quality == 'best' else 'worst'
    
    if download_thumbnail:
        ydl_opts['writethumbnail'] = True
    
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=True)
        
        title = info.get('title', 'video')
        ext = info.get('ext', 'mp4')
        if extract_audio:
            ext = format_pref if format_pref != 'auto' else 'mp3'
        
        video_file = output_path / f"{title}.{ext}"
        
        return {
            'success': True,
            'video_path': str(video_file),
            'metadata': {
                'title': info.get('title', ''),
                'duration': info.get('duration', 0)
            },
            'file_size': video_file.stat().st_size if video_file.exists() else 0
        }
