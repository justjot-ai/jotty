# YouTube Downloader Skill

## Description
Downloads YouTube video transcripts and converts them to markdown/PDF/EPUB. Supports single videos and playlists with optional AI summarization.

## Tools

### download_youtube_video_tool
Downloads transcript from a YouTube video and converts to document.

**Parameters:**
- `video_url` (str, required): YouTube video URL
- `output_format` (str, optional): Output format - 'markdown', 'pdf', 'epub', default: 'markdown'
- `include_timestamps` (bool, optional): Include timestamps in transcript, default: True
- `summarize` (bool, optional): Generate AI summary, default: False
- `summary_type` (str, optional): Summary type - 'brief', 'comprehensive', 'detailed', default: 'comprehensive'
- `output_dir` (str, optional): Output directory, default: './output/youtube'

**Returns:**
- `success` (bool): Whether download succeeded
- `video_id` (str): YouTube video ID
- `title` (str): Video title
- `author` (str): Channel name
- `output_path` (str): Path to generated file
- `transcript_length` (int): Number of transcript segments
- `error` (str, optional): Error message if failed

### download_youtube_playlist_tool
Downloads transcripts from all videos in a YouTube playlist.

**Parameters:**
- `playlist_url` (str, required): YouTube playlist URL
- `output_format` (str, optional): Output format, default: 'markdown'
- `include_timestamps` (bool, optional): Include timestamps, default: True
- `combine` (bool, optional): Combine all videos into one document, default: True
- `max_videos` (int, optional): Maximum videos to process, default: None (all)
- `output_dir` (str, optional): Output directory

**Returns:**
- `success` (bool): Whether download succeeded
- `playlist_id` (str): Playlist ID
- `videos_processed` (int): Number of videos processed
- `output_paths` (list): List of generated file paths
- `error` (str, optional): Error message if failed
