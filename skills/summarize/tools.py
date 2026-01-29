"""
Text Summarization Skill

Summarize text content, URLs, and files using Claude CLI.
Supports multiple formats (txt, md, pdf) and output styles.
"""
import logging
import requests
from pathlib import Path
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class SummarizationService:
    """Service class for text summarization using Claude CLI."""

    def __init__(self):
        self._claude_skill = None
        self._registry_initialized = False

    def _get_claude_skill(self):
        """Lazy load Claude CLI skill from registry."""
        if self._claude_skill is None:
            try:
                try:
            from Jotty.core.registry.skills_registry import get_skills_registry
        except ImportError:
            from core.registry.skills_registry import get_skills_registry
                registry = get_skills_registry()
                registry.init()
                self._claude_skill = registry.get_skill('claude-cli-llm')
                self._registry_initialized = True
            except Exception as e:
                logger.error(f"Failed to load claude-cli-llm skill: {e}")
                raise RuntimeError(f"Failed to load claude-cli-llm skill: {e}")
        return self._claude_skill

    def _call_llm(self, prompt: str, model: str = 'sonnet', timeout: int = 120) -> Dict[str, Any]:
        """Call Claude CLI LLM for text generation."""
        claude_skill = self._get_claude_skill()
        if not claude_skill:
            return {'success': False, 'error': 'Claude CLI skill not available'}

        generate_tool = claude_skill.tools.get('generate_text_tool')
        if not generate_tool:
            return {'success': False, 'error': 'generate_text_tool not found'}

        result = generate_tool({
            'prompt': prompt,
            'model': model,
            'timeout': timeout
        })

        return result

    def _build_summary_prompt(
        self,
        text: str,
        length: str = 'medium',
        style: str = 'paragraph'
    ) -> str:
        """Build the summarization prompt based on parameters."""
        length_instructions = {
            'short': 'Provide a very brief summary in 2-3 sentences (approximately 50-75 words).',
            'medium': 'Provide a comprehensive summary in 1-2 paragraphs (approximately 150-250 words).',
            'long': 'Provide a detailed summary covering all main points (approximately 400-600 words).'
        }

        style_instructions = {
            'bullet': 'Format the summary as bullet points with clear, concise items.',
            'paragraph': 'Format the summary as flowing prose paragraphs.',
            'numbered': 'Format the summary as a numbered list of key points.'
        }

        length_inst = length_instructions.get(length, length_instructions['medium'])
        style_inst = style_instructions.get(style, style_instructions['paragraph'])

        prompt = f"""Summarize the following content accurately and comprehensively.

Instructions:
- {length_inst}
- {style_inst}
- Capture the main ideas, key arguments, and important details.
- Maintain factual accuracy and avoid adding information not in the original.
- Use clear, professional language.

Content to summarize:
---
{text}
---

Summary:"""

        return prompt

    def _build_key_points_prompt(self, text: str, max_points: int = 5) -> str:
        """Build prompt for extracting key points."""
        prompt = f"""Extract the {max_points} most important key points from the following content.

Instructions:
- Identify the {max_points} most significant and actionable points.
- Each point should be a clear, standalone statement.
- Prioritize points by importance and relevance.
- Be concise but complete in each point.
- Format as a numbered list.

Content:
---
{text}
---

Key Points (up to {max_points}):"""

        return prompt

    def fetch_url_content(self, url: str, timeout: int = 30) -> Dict[str, Any]:
        """Fetch and extract text content from a URL."""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            }

            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()

            # Try to extract text content
            content_type = response.headers.get('content-type', '')

            if 'text/html' in content_type:
                from bs4 import BeautifulSoup
                import html2text

                soup = BeautifulSoup(response.content, 'html.parser')

                # Remove unwanted elements
                for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'iframe']):
                    tag.decompose()

                # Get title
                title = ''
                title_tag = soup.find('title')
                if title_tag:
                    title = title_tag.get_text(strip=True)

                # Extract main content
                main = soup.find('main') or soup.find('article') or soup.find('body')
                if main:
                    h2t = html2text.HTML2Text()
                    h2t.ignore_links = True
                    h2t.ignore_images = True
                    h2t.body_width = 0
                    text_content = h2t.handle(str(main))
                else:
                    text_content = soup.get_text(separator='\n', strip=True)

                return {
                    'success': True,
                    'title': title,
                    'content': text_content[:50000],  # Limit content size
                    'content_length': len(text_content)
                }

            elif 'text/plain' in content_type:
                return {
                    'success': True,
                    'title': url.split('/')[-1],
                    'content': response.text[:50000],
                    'content_length': len(response.text)
                }

            else:
                return {
                    'success': False,
                    'error': f'Unsupported content type: {content_type}'
                }

        except requests.RequestException as e:
            return {
                'success': False,
                'error': f'Network error fetching URL: {str(e)}'
            }
        except Exception as e:
            return {
                'success': False,
                'error': f'Error fetching URL: {str(e)}'
            }

    def read_file_content(self, file_path: str) -> Dict[str, Any]:
        """Read and extract text content from a file."""
        try:
            path = Path(file_path)

            if not path.exists():
                return {'success': False, 'error': f'File not found: {file_path}'}

            suffix = path.suffix.lower()

            if suffix in ['.txt', '.md', '.markdown', '.rst', '.text']:
                # Plain text files
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                return {
                    'success': True,
                    'title': path.name,
                    'content': content[:100000],
                    'content_length': len(content),
                    'file_type': suffix[1:]
                }

            elif suffix == '.pdf':
                # PDF files using PyPDF2
                try:
                    import PyPDF2

                    content_parts = []
                    with open(path, 'rb') as f:
                        reader = PyPDF2.PdfReader(f)
                        for page in reader.pages:
                            text = page.extract_text()
                            if text:
                                content_parts.append(text)

                    content = '\n\n'.join(content_parts)
                    return {
                        'success': True,
                        'title': path.name,
                        'content': content[:100000],
                        'content_length': len(content),
                        'file_type': 'pdf',
                        'page_count': len(reader.pages)
                    }
                except ImportError:
                    return {
                        'success': False,
                        'error': 'PyPDF2 not installed. Install with: pip install PyPDF2'
                    }
                except Exception as e:
                    return {
                        'success': False,
                        'error': f'Error reading PDF: {str(e)}'
                    }

            elif suffix == '.html':
                # HTML files
                from bs4 import BeautifulSoup
                import html2text

                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    html_content = f.read()

                soup = BeautifulSoup(html_content, 'html.parser')
                h2t = html2text.HTML2Text()
                h2t.ignore_links = True
                h2t.body_width = 0
                content = h2t.handle(html_content)

                return {
                    'success': True,
                    'title': path.name,
                    'content': content[:100000],
                    'content_length': len(content),
                    'file_type': 'html'
                }

            else:
                return {
                    'success': False,
                    'error': f'Unsupported file type: {suffix}. Supported: txt, md, pdf, html'
                }

        except Exception as e:
            return {
                'success': False,
                'error': f'Error reading file: {str(e)}'
            }

    def summarize(
        self,
        text: str,
        length: str = 'medium',
        style: str = 'paragraph',
        model: str = 'sonnet',
        timeout: int = 120
    ) -> Dict[str, Any]:
        """
        Summarize text content.

        Args:
            text: Text content to summarize
            length: Summary length - 'short', 'medium', 'long'
            style: Output style - 'bullet', 'paragraph', 'numbered'
            model: Claude model to use
            timeout: Request timeout in seconds

        Returns:
            Dictionary with success status and summary
        """
        if not text or not text.strip():
            return {'success': False, 'error': 'Text content is required'}

        prompt = self._build_summary_prompt(text, length, style)
        result = self._call_llm(prompt, model=model, timeout=timeout)

        if not result.get('success'):
            return result

        return {
            'success': True,
            'summary': result.get('text', ''),
            'length': length,
            'style': style,
            'model': model,
            'input_length': len(text)
        }

    def extract_key_points(
        self,
        text: str,
        max_points: int = 5,
        model: str = 'sonnet',
        timeout: int = 120
    ) -> Dict[str, Any]:
        """
        Extract key points from text.

        Args:
            text: Text content to analyze
            max_points: Maximum number of key points to extract
            model: Claude model to use
            timeout: Request timeout in seconds

        Returns:
            Dictionary with success status and key points
        """
        if not text or not text.strip():
            return {'success': False, 'error': 'Text content is required'}

        if max_points < 1:
            max_points = 5
        elif max_points > 20:
            max_points = 20

        prompt = self._build_key_points_prompt(text, max_points)
        result = self._call_llm(prompt, model=model, timeout=timeout)

        if not result.get('success'):
            return result

        return {
            'success': True,
            'key_points': result.get('text', ''),
            'max_points': max_points,
            'model': model,
            'input_length': len(text)
        }


# Create singleton instance
_service = SummarizationService()


def summarize_text_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Summarize text content using Claude CLI.

    Args:
        params: Dictionary containing:
            - text (str, required): Text content to summarize
            - length (str, optional): Summary length - 'short', 'medium', 'long' (default: 'medium')
            - style (str, optional): Output style - 'bullet', 'paragraph', 'numbered' (default: 'paragraph')
            - model (str, optional): Claude model - 'sonnet', 'opus', 'haiku' (default: 'sonnet')

    Returns:
        Dictionary with:
            - success (bool): Whether summarization succeeded
            - summary (str): Generated summary
            - length (str): Summary length used
            - style (str): Output style used
            - error (str, optional): Error message if failed
    """
    try:
        text = params.get('text')
        if not text:
            return {'success': False, 'error': 'text parameter is required'}

        length = params.get('length', 'medium')
        style = params.get('style', 'paragraph')
        model = params.get('model', 'sonnet')

        # Validate parameters
        if length not in ['short', 'medium', 'long']:
            length = 'medium'
        if style not in ['bullet', 'paragraph', 'numbered']:
            style = 'paragraph'

        return _service.summarize(text, length=length, style=style, model=model)

    except Exception as e:
        logger.error(f"summarize_text_tool error: {e}", exc_info=True)
        return {'success': False, 'error': f'Summarization failed: {str(e)}'}


def summarize_url_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Summarize webpage content from a URL using Claude CLI.

    Args:
        params: Dictionary containing:
            - url (str, required): URL to fetch and summarize
            - length (str, optional): Summary length - 'short', 'medium', 'long' (default: 'medium')
            - style (str, optional): Output style - 'bullet', 'paragraph', 'numbered' (default: 'paragraph')
            - model (str, optional): Claude model (default: 'sonnet')
            - timeout (int, optional): URL fetch timeout in seconds (default: 30)

    Returns:
        Dictionary with:
            - success (bool): Whether summarization succeeded
            - summary (str): Generated summary
            - url (str): Source URL
            - title (str): Page title
            - error (str, optional): Error message if failed
    """
    try:
        url = params.get('url')
        if not url:
            return {'success': False, 'error': 'url parameter is required'}

        length = params.get('length', 'medium')
        style = params.get('style', 'paragraph')
        model = params.get('model', 'sonnet')
        fetch_timeout = params.get('timeout', 30)

        # Validate parameters
        if length not in ['short', 'medium', 'long']:
            length = 'medium'
        if style not in ['bullet', 'paragraph', 'numbered']:
            style = 'paragraph'

        # Fetch URL content
        logger.info(f"Fetching URL: {url}")
        fetch_result = _service.fetch_url_content(url, timeout=fetch_timeout)

        if not fetch_result.get('success'):
            return fetch_result

        content = fetch_result.get('content', '')
        title = fetch_result.get('title', '')

        if not content:
            return {'success': False, 'error': 'No content extracted from URL'}

        # Summarize content
        summary_result = _service.summarize(content, length=length, style=style, model=model)

        if not summary_result.get('success'):
            return summary_result

        return {
            'success': True,
            'summary': summary_result.get('summary', ''),
            'url': url,
            'title': title,
            'length': length,
            'style': style,
            'content_length': fetch_result.get('content_length', 0)
        }

    except Exception as e:
        logger.error(f"summarize_url_tool error: {e}", exc_info=True)
        return {'success': False, 'error': f'URL summarization failed: {str(e)}'}


def summarize_file_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Summarize file content (txt, md, pdf) using Claude CLI.

    Args:
        params: Dictionary containing:
            - file_path (str, required): Path to the file to summarize
            - length (str, optional): Summary length - 'short', 'medium', 'long' (default: 'medium')
            - style (str, optional): Output style - 'bullet', 'paragraph', 'numbered' (default: 'paragraph')
            - model (str, optional): Claude model (default: 'sonnet')

    Returns:
        Dictionary with:
            - success (bool): Whether summarization succeeded
            - summary (str): Generated summary
            - file_path (str): Source file path
            - file_type (str): Detected file type
            - error (str, optional): Error message if failed
    """
    try:
        file_path = params.get('file_path')
        if not file_path:
            return {'success': False, 'error': 'file_path parameter is required'}

        length = params.get('length', 'medium')
        style = params.get('style', 'paragraph')
        model = params.get('model', 'sonnet')

        # Validate parameters
        if length not in ['short', 'medium', 'long']:
            length = 'medium'
        if style not in ['bullet', 'paragraph', 'numbered']:
            style = 'paragraph'

        # Read file content
        logger.info(f"Reading file: {file_path}")
        read_result = _service.read_file_content(file_path)

        if not read_result.get('success'):
            return read_result

        content = read_result.get('content', '')
        file_type = read_result.get('file_type', 'unknown')

        if not content:
            return {'success': False, 'error': 'No content extracted from file'}

        # Summarize content
        summary_result = _service.summarize(content, length=length, style=style, model=model)

        if not summary_result.get('success'):
            return summary_result

        result = {
            'success': True,
            'summary': summary_result.get('summary', ''),
            'file_path': file_path,
            'file_type': file_type,
            'length': length,
            'style': style,
            'content_length': read_result.get('content_length', 0)
        }

        # Add page count for PDFs
        if 'page_count' in read_result:
            result['page_count'] = read_result['page_count']

        return result

    except Exception as e:
        logger.error(f"summarize_file_tool error: {e}", exc_info=True)
        return {'success': False, 'error': f'File summarization failed: {str(e)}'}


def extract_key_points_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract key points from text using Claude CLI.

    Args:
        params: Dictionary containing:
            - text (str, required): Text content to analyze
            - max_points (int, optional): Maximum number of key points (default: 5, max: 20)
            - model (str, optional): Claude model (default: 'sonnet')

    Returns:
        Dictionary with:
            - success (bool): Whether extraction succeeded
            - key_points (str): Extracted key points as numbered list
            - max_points (int): Maximum points requested
            - error (str, optional): Error message if failed
    """
    try:
        text = params.get('text')
        if not text:
            return {'success': False, 'error': 'text parameter is required'}

        max_points = params.get('max_points', 5)
        model = params.get('model', 'sonnet')

        # Validate max_points
        try:
            max_points = int(max_points)
        except (TypeError, ValueError):
            max_points = 5

        return _service.extract_key_points(text, max_points=max_points, model=model)

    except Exception as e:
        logger.error(f"extract_key_points_tool error: {e}", exc_info=True)
        return {'success': False, 'error': f'Key point extraction failed: {str(e)}'}


__all__ = [
    'summarize_text_tool',
    'summarize_url_tool',
    'summarize_file_tool',
    'extract_key_points_tool'
]
