# Video Downloader Skill

Downloads videos from YouTube and other platforms for offline viewing, editing, or archival.

## Description

This skill downloads videos from YouTube and other platforms directly to your computer. Supports various formats, quality options, and batch downloads.


## Type
base


## Capabilities
- media

## Tools

### `download_video_tool`

Download video from URL.

**Parameters:**
- `video_url` (str, required): URL of video to download
- `output_path` (str, optional): Output directory (default: ~/Downloads)
- `quality` (str, optional): Quality - 'best', 'worst', '720p', '1080p', '4k', 'audio' (default: 'best')
- `format` (str, optional): Format - 'mp4', 'webm', 'mp3', 'auto' (default: 'auto')
- `extract_audio` (bool, optional): Extract audio only (default: False)
- `download_thumbnail` (bool, optional): Download thumbnail (default: True)
- `download_metadata` (bool, optional): Save metadata (default: True)

**Returns:**
- `success` (bool): Whether download succeeded
- `video_path` (str): Path to downloaded video
- `thumbnail_path` (str, optional): Path to thumbnail
- `metadata` (dict, optional): Video metadata
- `file_size` (int, optional): File size in bytes
- `error` (str, optional): Error message if failed

## Usage Examples

### Basic Download

```python
result = await download_video_tool({
    'video_url': 'https://youtube.com/watch?v=...',
    'quality': '1080p'
})
```

### Audio Only

```python
result = await download_video_tool({
    'video_url': 'https://youtube.com/watch?v=...',
    'extract_audio': True,
    'format': 'mp3'
})
```

## Dependencies

- `yt-dlp`: For video downloading (recommended over youtube-dl)

## Important Notes

⚠️ **Copyright & Fair Use**
- Only download videos you have permission to download
- Respect copyright laws and platform terms of service
- Use for personal, educational, or fair use purposes
