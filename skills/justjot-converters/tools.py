"""
JustJot Converters Skill Tools

Wraps JustJot.ai's utility converters as Jotty skill tools.
Provides conversion and distribution tools for various content types.
"""
import os
import sys
import logging
from typing import Dict, Any, Optional
from pathlib import Path

from Jotty.core.utils.skill_status import SkillStatus

logger = logging.getLogger(__name__)

# Import validation utilities
try:
    from Jotty.core.validation import ParamValidator, ValidationError
except ImportError:
    from Jotty.core.validation import ParamValidator, ValidationError

# Valid output formats
ARXIV_OUTPUT_FORMATS = ['markdown', 'pdf', 'remarkable', 'a4', 'letter', 'epub', 'kindle']
YOUTUBE_OUTPUT_FORMATS = ['markdown', 'pdf', 'remarkable', 'epub']
SUMMARY_TYPES = ['short', 'medium', 'comprehensive', 'study_guide']
EMAIL_PROVIDERS = ['gmail', 'outlook', 'yahoo', 'custom']

# Lazy import tracking

# Status emitter for progress updates
status = SkillStatus("justjot-converters")

_JUSTJOT_PATH_ADDED = False


def _ensure_justjot_path():
    """Add JustJot.ai to sys.path if needed."""
    global _JUSTJOT_PATH_ADDED

    if _JUSTJOT_PATH_ADDED:
        return

    _JUSTJOT_PATH_ADDED = True

    justjot_paths = [
        Path('/var/www/sites/personal/stock_market/JustJot.ai'),
        Path.home() / 'JustJot.ai',
        Path.cwd().parent / 'JustJot.ai',
    ]

    for justjot_path in justjot_paths:
        if justjot_path.exists():
            if str(justjot_path) not in sys.path:
                sys.path.insert(0, str(justjot_path))
            logger.info(f"Added JustJot.ai to path: {justjot_path}")
            break


# ============================================================================
# ArXiv Converter Tools
# ============================================================================

def arxiv_to_markdown_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert an arXiv paper to formatted markdown or PDF.

    Args:
        params: Dictionary containing:
            - arxiv_id (str, required): ArXiv paper ID or URL (e.g., "1706.03762" or "https://arxiv.org/abs/1706.03762")
            - output_format (str, optional): "markdown", "pdf", "remarkable", "epub", "kindle" (default: "remarkable")
            - output_dir (str, optional): Output directory path

    Returns:
        Dictionary with success, output_path, title, error
    """
    status.set_callback(params.pop('_status_callback', None))
    try:
        # Validate parameters
        validator = ParamValidator(params)
        try:
            arxiv_input = validator.require('arxiv_id', str)
            output_format = validator.optional('output_format', str, default='remarkable',
                                               choices=ARXIV_OUTPUT_FORMATS)
            output_dir = validator.optional('output_dir', str)
        except ValidationError as e:
            return e.to_dict()

        _ensure_justjot_path()

        from utils.arxiv_converter import ArxivConverter, extract_arxiv_id
        from Jotty.core.config import PageSize

        # Extract clean arxiv ID
        arxiv_id = extract_arxiv_id(arxiv_input)
        if not arxiv_id:
            arxiv_id = arxiv_input  # Use as-is if extraction fails

        # Map format to PageSize
        format_map = {
            'remarkable': PageSize.REMARKABLE,
            'pdf': PageSize.A4,
            'a4': PageSize.A4,
            'letter': PageSize.LETTER,
            'epub': PageSize.EPUB,
            'kindle': PageSize.KINDLE,
        }

        page_size = format_map.get(output_format, PageSize.REMARKABLE)

        # Create converter and convert
        output_path_dir = Path(output_dir) if output_dir else Path.cwd()
        converter = ArxivConverter(arxiv_id, output_dir=output_path_dir)
        result_path = converter.convert(page_size=page_size)

        if result_path:
            return {
                'success': True,
                'output_path': str(result_path),
                'title': f"arXiv:{arxiv_id}",
                'arxiv_id': arxiv_id
            }
        else:
            return {
                'success': False,
                'error': 'Conversion failed - check LaTeX compilation errors'
            }

    except ImportError as e:
        return {'success': False, 'error': f'ArXiv converter not available: {e}'}
    except Exception as e:
        logger.error(f"ArXiv conversion failed: {e}", exc_info=True)
        return {'success': False, 'error': str(e)}


# ============================================================================
# YouTube Converter Tools
# ============================================================================

def youtube_to_markdown_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a YouTube video transcript to formatted markdown or PDF.

    Args:
        params: Dictionary containing:
            - url (str, required): YouTube video URL
            - output_format (str, optional): "markdown", "pdf", "remarkable", "epub" (default: "markdown")
            - include_timestamps (bool, optional): Include timestamps (default: true)
            - summarize (bool, optional): Generate AI summary (default: false)
            - summary_type (str, optional): "short", "medium", "comprehensive", "study_guide" (default: "comprehensive")
            - output_dir (str, optional): Output directory
            - webshare_username (str, optional): Proxy username
            - webshare_password (str, optional): Proxy password

    Returns:
        Dictionary with success, output_path, title, author, duration, error
    """
    status.set_callback(params.pop('_status_callback', None))
    try:
        # Validate parameters
        validator = ParamValidator(params)
        try:
            url = validator.validate_pattern('url', 'youtube_url',
                                            message="'url' must be a valid YouTube URL")
            output_format = validator.optional('output_format', str, default='markdown',
                                               choices=YOUTUBE_OUTPUT_FORMATS)
            include_timestamps = validator.optional('include_timestamps', bool, default=True)
            summarize = validator.optional('summarize', bool, default=False)
            summary_type = validator.optional('summary_type', str, default='comprehensive',
                                              choices=SUMMARY_TYPES)
            output_dir = validator.optional('output_dir', str)
            webshare_username = validator.optional('webshare_username', str)
            webshare_password = validator.optional('webshare_password', str)
        except ValidationError as e:
            return e.to_dict()

        _ensure_justjot_path()

        from utils.youtube_converter import YouTubeConverter

        # Create converter with optional proxy
        converter_kwargs = {}
        if output_dir:
            converter_kwargs['output_dir'] = Path(output_dir)
        if webshare_username and webshare_password:
            converter_kwargs['webshare_username'] = webshare_username
            converter_kwargs['webshare_password'] = webshare_password

        converter = YouTubeConverter(**converter_kwargs)

        # Get video info first
        video_id = converter.extract_video_id(url)
        video_info = converter.get_video_info(video_id)

        # Convert
        output_path = converter.convert_video(
            url,
            output_format=output_format,
            include_timestamps=include_timestamps,
            summarize=summarize,
            summary_type=summary_type
        )

        return {
            'success': True,
            'output_path': str(output_path),
            'title': video_info.get('title', 'Unknown'),
            'author': video_info.get('author', 'Unknown'),
            'duration': converter.format_timestamp(video_info.get('duration', 0)),
            'video_id': video_id
        }

    except ImportError as e:
        return {'success': False, 'error': f'YouTube converter not available: {e}'}
    except Exception as e:
        logger.error(f"YouTube conversion failed: {e}", exc_info=True)
        return {'success': False, 'error': str(e)}


# ============================================================================
# HTML Converter Tools
# ============================================================================

def html_to_markdown_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert an HTML page to clean markdown.

    Args:
        params: Dictionary containing:
            - url (str, optional): URL to fetch and convert
            - content (str, optional): Direct HTML content to convert
            - title (str, optional): Document title (default: "HTML Document")
            - output_path (str, optional): Output file path

    Returns:
        Dictionary with success, content, output_path, error
    """
    try:
        # Validate parameters
        validator = ParamValidator(params)
        try:
            validator.require_one_of('url', 'content',
                                    message="Either 'url' or 'content' is required")
            url = validator.optional('url', str)
            if url:
                validator.validate_url('url', required=False)
            html_content = validator.optional('content', str)
            title = validator.optional('title', str, default='HTML Document')
            output_path = validator.optional('output_path', str)
        except ValidationError as e:
            return e.to_dict()

        _ensure_justjot_path()

        # Fetch URL if provided
        if url:
            import requests
            resp = requests.get(url, timeout=30, headers={
                'User-Agent': 'Mozilla/5.0 (compatible; JustJot/1.0)'
            })
            resp.raise_for_status()
            html_content = resp.text

        # Convert to markdown
        try:
            import html2text
            h = html2text.HTML2Text()
            h.ignore_links = False
            h.ignore_images = False
            h.body_width = 0  # No line wrapping
            markdown = h.handle(html_content)
        except ImportError:
            # Fallback: basic tag stripping
            import re
            # Remove scripts and styles
            html_content = re.sub(r'<script[^>]*>.*?</script>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
            html_content = re.sub(r'<style[^>]*>.*?</style>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
            # Strip remaining tags
            markdown = re.sub(r'<[^>]+>', '', html_content)
            # Clean up whitespace
            markdown = re.sub(r'\n\s*\n', '\n\n', markdown)
            markdown = markdown.strip()

        result = {
            'success': True,
            'content': markdown,
            'title': title
        }

        # Save to file if requested
        if output_path:
            Path(output_path).write_text(markdown, encoding='utf-8')
            result['output_path'] = output_path

        return result

    except Exception as e:
        logger.error(f"HTML conversion failed: {e}", exc_info=True)
        return {'success': False, 'error': str(e)}


# ============================================================================
# Kindle Email Tools
# ============================================================================

def send_to_kindle_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Send a document to Kindle via email.

    Args:
        params: Dictionary containing:
            - file_path (str, required): Path to document (PDF or EPUB)
            - subject (str, optional): Email subject
            - convert (bool, optional): Request Kindle conversion (default: true)

    Returns:
        Dictionary with success, kindle_email, error
    """
    status.set_callback(params.pop('_status_callback', None))

    try:
        # Validate parameters
        validator = ParamValidator(params)
        try:
            file_path = validator.validate_file_exists('file_path', required=True)
            subject = validator.optional('subject', str)
            convert = validator.optional('convert', bool, default=True)
        except ValidationError as e:
            return e.to_dict()

        _ensure_justjot_path()

        from utils.kindle_email import KindleEmailer

        emailer = KindleEmailer()

        if not emailer.is_configured():
            return {
                'success': False,
                'error': 'Kindle email not configured. Run kindle_configure_tool first.'
            }

        success = emailer.send(file_path, subject=subject, convert=convert)

        if success:
            status = emailer.get_status()
            return {
                'success': True,
                'kindle_email': status.get('kindle_email'),
                'message': 'Document sent successfully'
            }
        else:
            return {
                'success': False,
                'error': 'Failed to send email - check SMTP credentials'
            }

    except ImportError as e:
        return {'success': False, 'error': f'Kindle emailer not available: {e}'}
    except Exception as e:
        logger.error(f"Kindle send failed: {e}", exc_info=True)
        return {'success': False, 'error': str(e)}


def kindle_configure_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Configure Kindle email delivery settings.

    Args:
        params: Dictionary containing:
            - kindle_email (str, required): Your Kindle email (xxxxx@kindle.com)
            - smtp_email (str, required): Your email address
            - smtp_password (str, required): SMTP password (use App Password for Gmail)
            - provider (str, optional): "gmail", "outlook", "yahoo", "custom" (default: "gmail")
            - smtp_server (str, optional): Custom SMTP server (required if provider="custom")
            - smtp_port (int, optional): SMTP port (default: 587)

    Returns:
        Dictionary with success, message, error
    """
    try:
        # Validate parameters
        validator = ParamValidator(params)
        try:
            kindle_email = validator.validate_pattern('kindle_email', 'kindle_email',
                                                      message="'kindle_email' must be a valid @kindle.com address")
            smtp_email = validator.validate_email('smtp_email', required=True)
            smtp_password = validator.require('smtp_password', str)
            provider = validator.optional('provider', str, default='gmail', choices=EMAIL_PROVIDERS)
            smtp_server = validator.optional('smtp_server', str)
            smtp_port = validator.validate_range('smtp_port', min_val=1, max_val=65535, default=587)
        except ValidationError as e:
            return e.to_dict()

        _ensure_justjot_path()

        from utils.kindle_email import KindleEmailer

        emailer = KindleEmailer()
        success = emailer.configure(
            kindle_email=kindle_email,
            smtp_email=smtp_email,
            smtp_password=smtp_password,
            provider=provider,
            smtp_server=smtp_server,
            smtp_port=smtp_port
        )

        if success:
            return {
                'success': True,
                'message': f"Kindle email configured. Add {smtp_email} to your Kindle's approved email list at Amazon."
            }
        else:
            return {
                'success': False,
                'error': 'Configuration failed - check your credentials'
            }

    except ImportError as e:
        return {'success': False, 'error': f'Kindle emailer not available: {e}'}
    except Exception as e:
        logger.error(f"Kindle configure failed: {e}", exc_info=True)
        return {'success': False, 'error': str(e)}


def kindle_status_tool(params: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Check Kindle email configuration status.

    Returns:
        Dictionary with configured, kindle_email, smtp_email, provider
    """
    status.set_callback(params.pop('_status_callback', None))

    try:
        _ensure_justjot_path()

        from utils.kindle_email import KindleEmailer

        emailer = KindleEmailer()
        status = emailer.get_status()

        return {
            'success': True,
            'configured': status.get('configured', False),
            'kindle_email': status.get('kindle_email'),
            'smtp_email': status.get('smtp_email'),
            'provider': status.get('provider'),
            'config_file': status.get('config_file')
        }

    except ImportError as e:
        return {'success': False, 'error': f'Kindle emailer not available: {e}'}
    except Exception as e:
        logger.error(f"Kindle status check failed: {e}", exc_info=True)
        return {'success': False, 'error': str(e)}


# ============================================================================
# reMarkable Sync Tools
# ============================================================================

def sync_to_remarkable_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Upload a document to reMarkable device via cloud.

    Args:
        params: Dictionary containing:
            - file_path (str, required): Path to PDF file
            - folder_name (str, optional): Folder name on reMarkable (creates if doesn't exist)
            - document_name (str, optional): Document name (uses filename if not provided)

    Returns:
        Dictionary with success, document_name, error
    """
    status.set_callback(params.pop('_status_callback', None))

    try:
        # Validate parameters
        validator = ParamValidator(params)
        try:
            file_path = validator.validate_file_exists('file_path', required=True)
            folder_name = validator.optional('folder_name', str)
            document_name = validator.optional('document_name', str)
        except ValidationError as e:
            return e.to_dict()

        _ensure_justjot_path()

        from utils.remarkable_sync import RemarkableSync

        sync = RemarkableSync()

        if not sync.is_registered():
            return {
                'success': False,
                'error': 'Not registered with reMarkable cloud. Run remarkable_register_tool first.'
            }

        success = sync.upload(
            file_path,
            folder_name=folder_name,
            document_name=document_name
        )

        if success:
            return {
                'success': True,
                'document_name': document_name or file_path.stem,
                'folder': folder_name or 'Root',
                'message': 'Document uploaded successfully'
            }
        else:
            return {
                'success': False,
                'error': 'Upload failed - check reMarkable cloud connection'
            }

    except ImportError as e:
        return {'success': False, 'error': f'reMarkable sync not available: {e}'}
    except Exception as e:
        logger.error(f"reMarkable sync failed: {e}", exc_info=True)
        return {'success': False, 'error': str(e)}


def remarkable_register_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Register device with reMarkable cloud.

    Args:
        params: Dictionary containing:
            - one_time_code (str, required): 8-character code from https://my.remarkable.com/device/browser/connect

    Returns:
        Dictionary with success, message, error
    """
    status.set_callback(params.pop('_status_callback', None))

    try:
        # Validate parameters
        validator = ParamValidator(params)
        try:
            one_time_code = validator.require('one_time_code', str)
            # Validate it's 8 characters
            if len(one_time_code) != 8:
                raise ValidationError("'one_time_code' must be exactly 8 characters", param='one_time_code')
        except ValidationError as e:
            return e.to_dict()

        _ensure_justjot_path()

        from utils.remarkable_sync import RemarkableSync

        sync = RemarkableSync()
        success = sync.register(one_time_code)

        if success:
            return {
                'success': True,
                'message': 'Device registered with reMarkable cloud successfully'
            }
        else:
            return {
                'success': False,
                'error': 'Registration failed - check one-time code and try again'
            }

    except ImportError as e:
        return {'success': False, 'error': f'reMarkable sync not available: {e}'}
    except Exception as e:
        logger.error(f"reMarkable registration failed: {e}", exc_info=True)
        return {'success': False, 'error': str(e)}


def remarkable_status_tool(params: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Check reMarkable cloud registration status.

    Returns:
        Dictionary with registered, rmapy_installed, config_file
    """
    status.set_callback(params.pop('_status_callback', None))

    try:
        _ensure_justjot_path()

        from utils.remarkable_sync import RemarkableSync

        sync = RemarkableSync()
        status = sync.get_status()

        return {
            'success': True,
            'registered': status.get('registered', False),
            'rmapy_installed': status.get('rmapy_installed', False),
            'config_file': status.get('config_file')
        }

    except ImportError as e:
        return {'success': False, 'error': f'reMarkable sync not available: {e}'}
    except Exception as e:
        logger.error(f"reMarkable status check failed: {e}", exc_info=True)
        return {'success': False, 'error': str(e)}
