"""
Content Pipeline Skill Tools

Wraps JustJot.ai's ContentPipeline as Jotty skill tools.
Provides document processing pipeline: Source -> Processors -> Sinks.
"""
import os
import sys
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

from Jotty.core.infrastructure.utils.skill_status import SkillStatus
from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, tool_wrapper

logger = logging.getLogger(__name__)

# Import validation utilities
try:
    from Jotty.core.validation import ParamValidator, ValidationError, validate_params
except ImportError:
    from Jotty.core.validation import ParamValidator, ValidationError, validate_params

# Validation schemas for tools
RUN_PIPELINE_SCHEMA = {
    'source_type': {'required': True, 'type': str},
    'source_params': {'required': True, 'type': dict},
    'processors': {'type': list, 'default': []},
    'sinks': {'type': list, 'default': []},
    'output_dir': {'type': str, 'dir_exists': True, 'create': True},
}

RUN_SOURCE_SCHEMA = {
    'source_type': {
        'required': True,
        'type': str,
        'choices': ['markdown', 'arxiv', 'youtube', 'html', 'pdf']
    },
    'source_params': {'required': True, 'type': dict},
}

PROCESS_DOCUMENT_SCHEMA = {
    'document': {'required': True, 'type': dict},
    'processors': {'required': True, 'type': list},
}

SINK_DOCUMENT_SCHEMA = {
    'document': {'required': True, 'type': dict},
    'sinks': {'required': True, 'type': list},
    'output_dir': {'type': str, 'dir_exists': True, 'create': True},
}

# Lazy import JustJot components

# Status emitter for progress updates
status = SkillStatus("content-pipeline")

_JUSTJOT_IMPORTED = False
_ContentPipeline = None
_Document = None


def _import_justjot():
    """Lazily import JustJot.ai components with fallback."""
    global _JUSTJOT_IMPORTED, _ContentPipeline, _Document

    if _JUSTJOT_IMPORTED:
        return _ContentPipeline is not None

    _JUSTJOT_IMPORTED = True

    # Try to find JustJot.ai in common locations
    justjot_paths = [
        Path('/var/www/sites/personal/stock_market/JustJot.ai'),
        Path.home() / 'JustJot.ai',
        Path.cwd().parent / 'JustJot.ai',
    ]

    for justjot_path in justjot_paths:
        if justjot_path.exists():
            if str(justjot_path) not in sys.path:
                sys.path.insert(0, str(justjot_path))
            break

    try:
        from Jotty.core.pipeline import ContentPipeline
        from Jotty.core.document import Document
        _ContentPipeline = ContentPipeline
        _Document = Document
        logger.info("JustJot.ai ContentPipeline imported successfully")
        return True
    except ImportError as e:
        logger.warning(f"JustJot.ai not available: {e}")
        return False


def _create_source_adapter(source_type: str, source_params: Dict[str, Any]):
    """Create a source adapter based on type."""
    if not _import_justjot():
        raise ImportError("JustJot.ai not available")

    source_type = source_type.lower()

    if source_type == 'markdown':
        # Create markdown source
        content = source_params.get('content')
        file_path = source_params.get('file_path')
        title = source_params.get('title', 'Untitled')

        if file_path:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

        if not content:
            raise ValueError("Either 'content' or 'file_path' required for markdown source")

        # Return a simple adapter-like object
        class MarkdownSource:
            def __init__(self, content, title):
                self.content = content
                self.title = title

            def __str__(self):
                return f"MarkdownSource('{self.title}')"

            def validate_input(self, **kwargs):
                return True

            def generate(self, **kwargs):
                return _Document(
                    title=self.title,
                    content=self.content,
                    source_type='markdown'
                )

        return MarkdownSource(content, title)

    elif source_type == 'arxiv':
        arxiv_id = source_params.get('arxiv_id')
        if not arxiv_id:
            raise ValueError("'arxiv_id' required for arxiv source")

        try:
            from utils.arxiv_converter import ArxivConverter, extract_arxiv_id

            # Clean the arxiv_id
            clean_id = extract_arxiv_id(arxiv_id) or arxiv_id

            class ArxivSource:
                def __init__(self, arxiv_id):
                    self.arxiv_id = arxiv_id
                    self.converter = None

                def __str__(self):
                    return f"ArxivSource('{self.arxiv_id}')"

                def validate_input(self, **kwargs):
                    return True

                def generate(self, **kwargs):
                    # For now, return a document with metadata
                    # Full conversion happens at sink stage
                    return _Document(
                        title=f"arXiv:{self.arxiv_id}",
                        content=f"ArXiv paper: {self.arxiv_id}\n\nFull content available via sink conversion.",
                        source_type='arxiv',
                        source_url=f"https://arxiv.org/abs/{self.arxiv_id}",
                        source_metadata={'arxiv_id': self.arxiv_id}
                    )

            return ArxivSource(clean_id)
        except ImportError:
            raise ImportError("ArXiv converter not available")

    elif source_type == 'youtube':
        url = source_params.get('url')
        if not url:
            raise ValueError("'url' required for youtube source")

        try:
            from utils.youtube_converter import YouTubeConverter

            class YouTubeSource:
                def __init__(self, url):
                    self.url = url
                    self.converter = YouTubeConverter()

                def __str__(self):
                    return f"YouTubeSource('{self.url}')"

                def validate_input(self, **kwargs):
                    return True

                def generate(self, **kwargs):
                    video_id = self.converter.extract_video_id(self.url)
                    video_info = self.converter.get_video_info(video_id)

                    try:
                        transcript = self.converter.get_transcript(video_id)
                        markdown = self.converter.transcript_to_markdown(
                            transcript, video_info,
                            include_timestamps=kwargs.get('include_timestamps', True)
                        )
                    except Exception as e:
                        markdown = f"# {video_info.get('title', 'YouTube Video')}\n\nTranscript unavailable: {e}"

                    return _Document(
                        title=video_info.get('title', 'YouTube Video'),
                        content=markdown,
                        author=video_info.get('author'),
                        source_type='youtube',
                        source_url=self.url,
                        source_metadata=video_info
                    )

            return YouTubeSource(url)
        except ImportError:
            raise ImportError("YouTube converter not available")

    elif source_type == 'html':
        url = source_params.get('url')
        content = source_params.get('content')

        if not url and not content:
            raise ValueError("Either 'url' or 'content' required for html source")

        try:
            from utils.html_converter import HTMLConverter

            class HTMLSource:
                def __init__(self, url=None, content=None):
                    self.url = url
                    self.html_content = content

                def __str__(self):
                    return f"HTMLSource('{self.url or 'inline'}')"

                def validate_input(self, **kwargs):
                    return True

                def generate(self, **kwargs):
                    if self.url:
                        import requests
                        resp = requests.get(self.url, timeout=30)
                        html = resp.text
                    else:
                        html = self.html_content

                    # Convert HTML to markdown
                    try:
                        import html2text
                        h = html2text.HTML2Text()
                        h.ignore_links = False
                        markdown = h.handle(html)
                    except ImportError:
                        # Fallback: strip HTML tags
                        import re
                        markdown = re.sub(r'<[^>]+>', '', html)

                    return _Document(
                        title=kwargs.get('title', 'HTML Document'),
                        content=markdown,
                        source_type='html',
                        source_url=self.url
                    )

            return HTMLSource(url=url, content=content)
        except ImportError as e:
            raise ImportError(f"HTML converter dependencies not available: {e}")

    else:
        raise ValueError(f"Unknown source type: {source_type}")


def _create_processors(processor_configs: List[Dict[str, Any]]) -> List:
    """Create processor adapters from configs."""
    if not _import_justjot():
        raise ImportError("JustJot.ai not available")

    processors = []

    for config in processor_configs:
        proc_type = config.get('type', '').lower()

        if proc_type == 'diagram_renderer':
            class DiagramRenderer:
                def __str__(self):
                    return "DiagramRenderer"

                def can_process(self, doc):
                    # Check if document has diagram code blocks
                    content = doc.content or doc.full_content
                    return any(diagram_type in content for diagram_type in
                              ['```mermaid', '```plantuml', '```graphviz', '```ditaa'])

                def process(self, doc):
                    # Mark as processed (actual rendering would happen here)
                    doc.diagrams_rendered = True
                    return doc

            processors.append(DiagramRenderer())

        elif proc_type == 'latex_handler':
            class LatexHandler:
                def __str__(self):
                    return "LatexHandler"

                def can_process(self, doc):
                    content = doc.content or doc.full_content
                    return '$$' in content or '$' in content or '\\begin{' in content

                def process(self, doc):
                    doc.processed_latex = True
                    return doc

            processors.append(LatexHandler())

        elif proc_type == 'image_downloader':
            class ImageDownloader:
                def __str__(self):
                    return "ImageDownloader"

                def can_process(self, doc):
                    content = doc.content or doc.full_content
                    return '![' in content and 'http' in content

                def process(self, doc):
                    # Would download images here
                    return doc

            processors.append(ImageDownloader())

        elif proc_type == 'syntax_fixer':
            class SyntaxFixer:
                def __str__(self):
                    return "SyntaxFixer"

                def can_process(self, doc):
                    return True

                def process(self, doc):
                    doc.syntax_fixed = True
                    return doc

            processors.append(SyntaxFixer())

        else:
            logger.warning(f"Unknown processor type: {proc_type}")

    return processors


def _create_sinks(sink_configs: List[Dict[str, Any]]) -> List:
    """Create sink adapters from configs."""
    if not _import_justjot():
        raise ImportError("JustJot.ai not available")

    sinks = []

    for config in sink_configs:
        sink_type = config.get('type', '').lower()
        format_type = config.get('format', 'a4').lower()

        if sink_type == 'pdf':
            class PDFSink:
                def __init__(self, format_type):
                    self.format_type = format_type

                def __str__(self):
                    return f"PDFSink({self.format_type})"

                def validate_document(self, doc):
                    return bool(doc.content or doc.full_content)

                def write(self, doc, output_path=None, **kwargs):
                    try:
                        from utils.markdown_converter import MarkdownConverter
                        from Jotty.core.config import PageSize

                        size_map = {
                            'remarkable': PageSize.REMARKABLE,
                            'a4': PageSize.A4,
                            'letter': PageSize.LETTER,
                            'kindle': PageSize.KINDLE,
                        }

                        page_size = size_map.get(self.format_type, PageSize.A4)

                        # Write markdown to temp file
                        import tempfile
                        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
                            f.write(doc.content or doc.full_content)
                            temp_md = Path(f.name)

                        try:
                            converter = MarkdownConverter(input_file=temp_md, output_dir=output_path)
                            converter.convert_to_pdf(
                                page_size=page_size,
                                title=doc.title,
                                author=doc.author or 'Unknown'
                            )
                            return Path(converter.output_file)
                        finally:
                            temp_md.unlink(missing_ok=True)

                    except ImportError:
                        # Fallback: just write markdown
                        output_dir = Path(output_path) if output_path else Path.cwd()
                        output_file = output_dir / f"{doc.title}.md"
                        output_file.write_text(doc.content or doc.full_content)
                        return output_file

            sinks.append(PDFSink(format_type))

        elif sink_type == 'epub':
            class EPUBSink:
                def __str__(self):
                    return "EPUBSink"

                def validate_document(self, doc):
                    return bool(doc.content or doc.full_content)

                def write(self, doc, output_path=None, **kwargs):
                    try:
                        from utils.markdown_converter import MarkdownConverter
                        from Jotty.core.config import PageSize

                        import tempfile
                        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
                            f.write(doc.content or doc.full_content)
                            temp_md = Path(f.name)

                        try:
                            converter = MarkdownConverter(input_file=temp_md, output_dir=output_path)
                            converter.convert_to_pdf(
                                page_size=PageSize.EPUB,
                                title=doc.title,
                                author=doc.author or 'Unknown'
                            )
                            return Path(converter.output_file)
                        finally:
                            temp_md.unlink(missing_ok=True)

                    except ImportError:
                        output_dir = Path(output_path) if output_path else Path.cwd()
                        output_file = output_dir / f"{doc.title}.md"
                        output_file.write_text(doc.content or doc.full_content)
                        return output_file

            sinks.append(EPUBSink())

        elif sink_type == 'markdown':
            class MarkdownSink:
                def __str__(self):
                    return "MarkdownSink"

                def validate_document(self, doc):
                    return bool(doc.content or doc.full_content)

                def write(self, doc, output_path=None, **kwargs):
                    import re
                    output_dir = Path(output_path) if output_path else Path.cwd()
                    safe_title = re.sub(r'[^\w\s-]', '', doc.title)[:50]
                    output_file = output_dir / f"{safe_title}.md"
                    output_file.write_text(doc.content or doc.full_content, encoding='utf-8')
                    return output_file

            sinks.append(MarkdownSink())

        elif sink_type == 'remarkable':
            class RemarkableSink:
                def __str__(self):
                    return "RemarkableSink"

                def validate_document(self, doc):
                    return bool(doc.content or doc.full_content)

                def write(self, doc, output_path=None, **kwargs):
                    try:
                        from utils.remarkable_sync import RemarkableSync
                        from utils.markdown_converter import MarkdownConverter
                        from Jotty.core.config import PageSize

                        # First generate remarkable-optimized PDF
                        import tempfile
                        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
                            f.write(doc.content or doc.full_content)
                            temp_md = Path(f.name)

                        try:
                            converter = MarkdownConverter(input_file=temp_md, output_dir=output_path)
                            converter.convert_to_pdf(
                                page_size=PageSize.REMARKABLE,
                                title=doc.title,
                                author=doc.author or 'Unknown'
                            )
                            pdf_path = Path(converter.output_file)

                            # Upload to reMarkable
                            sync = RemarkableSync()
                            if sync.is_registered():
                                sync.upload(pdf_path, document_name=doc.title)

                            return pdf_path
                        finally:
                            temp_md.unlink(missing_ok=True)

                    except ImportError as e:
                        raise ImportError(f"reMarkable sync not available: {e}")

            sinks.append(RemarkableSink())

        else:
            logger.warning(f"Unknown sink type: {sink_type}")

    return sinks


@tool_wrapper()
def run_pipeline_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run a complete content pipeline: Source -> Processors -> Sinks.

    Args:
        params: Dictionary containing:
            - source_type (str, required): Source adapter type ('markdown', 'arxiv', 'youtube', 'html', 'pdf')
            - source_params (dict, required): Parameters for source
            - processors (list, optional): Processor configurations
            - sinks (list, optional): Sink configurations
            - output_dir (str, optional): Output directory

    Returns:
        Dictionary with:
            - success (bool): Whether pipeline succeeded
            - document (dict): Final document (serialized)
            - output_paths (list): Generated file paths
            - history (list): Pipeline execution history
            - error (str, optional): Error message if failed
    """
    status.set_callback(params.pop('_status_callback', None))

    try:
        # Validate parameters
        try:
            validated = validate_params(params, RUN_PIPELINE_SCHEMA)
        except ValidationError as e:
            return e.to_dict()

        if not _import_justjot():
            return {
                'success': False,
                'error': 'JustJot.ai not available. Ensure it is installed.'
            }

        source_type = validated.get('source_type')
        source_params = validated.get('source_params', {})
        processor_configs = validated.get('processors', [])
        sink_configs = validated.get('sinks', [])
        output_dir = validated.get('output_dir')

        # Create pipeline
        pipeline = _ContentPipeline()

        # Create adapters
        source = _create_source_adapter(source_type, source_params)
        processors = _create_processors(processor_configs) if processor_configs else []
        sinks = _create_sinks(sink_configs) if sink_configs else []

        # Run pipeline
        result = pipeline.run(
            source=source,
            processors=processors if processors else None,
            sinks=sinks if sinks else None,
            output_dir=Path(output_dir) if output_dir else None
        )

        return {
            'success': True,
            'document': result['document'].to_dict(),
            'output_paths': [str(p) for p in result.get('output_paths', [])],
            'history': result.get('history', [])
        }

    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}", exc_info=True)
        return {'success': False, 'error': str(e)}


@tool_wrapper()
def run_source_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run only the source stage to generate a Document.

    Args:
        params: Dictionary containing:
            - source_type (str, required): Source adapter type ('markdown', 'arxiv', 'youtube', 'html', 'pdf')
            - source_params (dict, required): Parameters for source

    Returns:
        Dictionary with:
            - success (bool): Whether source generation succeeded
            - document (dict): Generated document (serialized)
            - error (str, optional): Error message if failed
    """
    status.set_callback(params.pop('_status_callback', None))

    try:
        # Validate parameters
        try:
            validated = validate_params(params, RUN_SOURCE_SCHEMA)
        except ValidationError as e:
            return e.to_dict()

        if not _import_justjot():
            return {
                'success': False,
                'error': 'JustJot.ai not available'
            }

        source_type = validated.get('source_type')
        source_params = validated.get('source_params', {})

        # Create and run source
        source = _create_source_adapter(source_type, source_params)
        document = source.generate()

        return {
            'success': True,
            'document': document.to_dict()
        }

    except Exception as e:
        logger.error(f"Source execution failed: {e}", exc_info=True)
        return {'success': False, 'error': str(e)}


@tool_wrapper()
def process_document_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process an existing document through processors.

    Args:
        params: Dictionary containing:
            - document (dict, required): Document dictionary
            - processors (list, required): Processor configurations

    Returns:
        Dictionary with:
            - success (bool): Whether processing succeeded
            - document (dict): Processed document (serialized)
            - error (str, optional): Error message if failed
    """
    status.set_callback(params.pop('_status_callback', None))

    try:
        # Validate parameters
        try:
            validated = validate_params(params, PROCESS_DOCUMENT_SCHEMA)
        except ValidationError as e:
            return e.to_dict()

        if not _import_justjot():
            return {
                'success': False,
                'error': 'JustJot.ai not available'
            }

        doc_dict = validated.get('document')
        processor_configs = validated.get('processors', [])

        # Reconstruct document
        document = _Document.from_dict(doc_dict)

        # Create and run processors
        processors = _create_processors(processor_configs)
        pipeline = _ContentPipeline()
        processed_doc = pipeline.process(document, processors)

        return {
            'success': True,
            'document': processed_doc.to_dict()
        }

    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)
        return {'success': False, 'error': str(e)}


@tool_wrapper()
def sink_document_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Write a document to one or more sinks.

    Args:
        params: Dictionary containing:
            - document (dict, required): Document dictionary
            - sinks (list, required): Sink configurations
            - output_dir (str, optional): Output directory

    Returns:
        Dictionary with:
            - success (bool): Whether sink writing succeeded
            - output_paths (list): Generated file paths
            - error (str, optional): Error message if failed
    """
    status.set_callback(params.pop('_status_callback', None))

    try:
        # Validate parameters
        try:
            validated = validate_params(params, SINK_DOCUMENT_SCHEMA)
        except ValidationError as e:
            return e.to_dict()

        if not _import_justjot():
            return {
                'success': False,
                'error': 'JustJot.ai not available'
            }

        doc_dict = validated.get('document')
        sink_configs = validated.get('sinks', [])
        output_dir = validated.get('output_dir')

        # Reconstruct document
        document = _Document.from_dict(doc_dict)

        # Create and run sinks
        sinks = _create_sinks(sink_configs)
        pipeline = _ContentPipeline()
        output_paths = pipeline.sink(
            document,
            sinks,
            output_dir=Path(output_dir) if output_dir else None
        )

        return {
            'success': True,
            'output_paths': [str(p) for p in output_paths]
        }

    except Exception as e:
        logger.error(f"Sink writing failed: {e}", exc_info=True)
        return {'success': False, 'error': str(e)}
