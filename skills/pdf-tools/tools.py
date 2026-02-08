"""
PDF Tools Skill

Comprehensive PDF toolkit using pdfplumber, pypdf, and reportlab.
Provides tools for extracting, manipulating, and creating PDF files.
"""
import os
import logging
import json
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

from Jotty.core.utils.skill_status import SkillStatus

# Status emitter for progress updates
status = SkillStatus("pdf-tools")


logger = logging.getLogger(__name__)


class PDFToolsError(Exception):
    """Base exception for PDF tools errors."""
    pass


class PDFExtractor:
    """Handles PDF text and table extraction using pdfplumber."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self._validate_file()

    def _validate_file(self):
        """Validate that the file exists and is a PDF."""
        if not os.path.exists(self.file_path):
            raise PDFToolsError(f"File not found: {self.file_path}")
        if not self.file_path.lower().endswith('.pdf'):
            raise PDFToolsError(f"File is not a PDF: {self.file_path}")

    def extract_text(self, pages: Optional[List[int]] = None, layout: bool = False) -> Dict[str, Any]:
        """Extract text from PDF pages."""
        try:
            import pdfplumber
        except ImportError:
            raise PDFToolsError("pdfplumber not installed. Install with: pip install pdfplumber")

        extracted_text = []
        page_count = 0

        with pdfplumber.open(self.file_path) as pdf:
            page_count = len(pdf.pages)
            target_pages = pages if pages else range(page_count)

            for page_num in target_pages:
                if page_num >= page_count or page_num < 0:
                    continue

                page = pdf.pages[page_num]
                if layout:
                    text = page.extract_text(layout=True) or ""
                else:
                    text = page.extract_text() or ""

                extracted_text.append({
                    'page': page_num + 1,
                    'text': text
                })

        return {
            'success': True,
            'file_path': self.file_path,
            'total_pages': page_count,
            'extracted_pages': len(extracted_text),
            'content': extracted_text
        }

    def extract_tables(self, pages: Optional[List[int]] = None,
                       output_format: str = 'json') -> Dict[str, Any]:
        """Extract tables from PDF pages."""
        try:
            import pdfplumber
        except ImportError:
            raise PDFToolsError("pdfplumber not installed. Install with: pip install pdfplumber")

        all_tables = []

        with pdfplumber.open(self.file_path) as pdf:
            page_count = len(pdf.pages)
            target_pages = pages if pages else range(page_count)

            for page_num in target_pages:
                if page_num >= page_count or page_num < 0:
                    continue

                page = pdf.pages[page_num]
                tables = page.extract_tables()

                for table_idx, table in enumerate(tables):
                    if not table or len(table) < 2:
                        continue

                    headers = table[0]
                    rows = table[1:]

                    if output_format == 'json':
                        table_data = [
                            {headers[i] if i < len(headers) else f'col_{i}':
                             cell for i, cell in enumerate(row)}
                            for row in rows
                        ]
                    elif output_format == 'csv':
                        csv_lines = [','.join(str(h) for h in headers)]
                        for row in rows:
                            csv_lines.append(','.join(str(cell) if cell else '' for cell in row))
                        table_data = '\n'.join(csv_lines)
                    elif output_format == 'dataframe':
                        table_data = {'headers': headers, 'rows': rows}
                    else:
                        table_data = {'headers': headers, 'rows': rows}

                    all_tables.append({
                        'page': page_num + 1,
                        'table_index': table_idx,
                        'data': table_data
                    })

        return {
            'success': True,
            'file_path': self.file_path,
            'table_count': len(all_tables),
            'output_format': output_format,
            'tables': all_tables
        }


class PDFManipulator:
    """Handles PDF manipulation using pypdf."""

    @staticmethod
    def merge_pdfs(file_paths: List[str], output_path: str) -> Dict[str, Any]:
        """Merge multiple PDF files into one."""
        try:
            from pypdf import PdfWriter, PdfReader
        except ImportError:
            raise PDFToolsError("pypdf not installed. Install with: pip install pypdf")

        for path in file_paths:
            if not os.path.exists(path):
                raise PDFToolsError(f"File not found: {path}")
            if not path.lower().endswith('.pdf'):
                raise PDFToolsError(f"File is not a PDF: {path}")

        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

        writer = PdfWriter()
        total_pages = 0

        for path in file_paths:
            reader = PdfReader(path)
            for page in reader.pages:
                writer.add_page(page)
                total_pages += 1

        with open(output_path, 'wb') as output_file:
            writer.write(output_file)

        return {
            'success': True,
            'output_path': output_path,
            'merged_files': len(file_paths),
            'total_pages': total_pages
        }

    @staticmethod
    def split_pdf(file_path: str, pages: Union[List[int], str],
                  output_dir: str) -> Dict[str, Any]:
        """Split PDF into separate files."""
        try:
            from pypdf import PdfWriter, PdfReader
        except ImportError:
            raise PDFToolsError("pypdf not installed. Install with: pip install pypdf")

        if not os.path.exists(file_path):
            raise PDFToolsError(f"File not found: {file_path}")

        os.makedirs(output_dir, exist_ok=True)

        reader = PdfReader(file_path)
        total_pages = len(reader.pages)

        if isinstance(pages, str):
            if '-' in pages:
                start, end = map(int, pages.split('-'))
                page_list = list(range(start - 1, min(end, total_pages)))
            else:
                page_list = [int(p) - 1 for p in pages.split(',')]
        else:
            page_list = [p - 1 if p > 0 else p for p in pages]

        output_files = []
        base_name = os.path.splitext(os.path.basename(file_path))[0]

        for page_num in page_list:
            if page_num < 0 or page_num >= total_pages:
                continue

            writer = PdfWriter()
            writer.add_page(reader.pages[page_num])

            output_path = os.path.join(output_dir, f"{base_name}_page_{page_num + 1}.pdf")
            with open(output_path, 'wb') as output_file:
                writer.write(output_file)

            output_files.append(output_path)

        return {
            'success': True,
            'source_file': file_path,
            'output_dir': output_dir,
            'files_created': len(output_files),
            'output_files': output_files
        }

    @staticmethod
    def rotate_pages(file_path: str, rotation: int,
                     pages: Optional[List[int]] = None,
                     output_path: Optional[str] = None) -> Dict[str, Any]:
        """Rotate pages in a PDF."""
        try:
            from pypdf import PdfWriter, PdfReader
        except ImportError:
            raise PDFToolsError("pypdf not installed. Install with: pip install pypdf")

        if rotation not in [90, 180, 270]:
            raise PDFToolsError("Rotation must be 90, 180, or 270 degrees")

        if not os.path.exists(file_path):
            raise PDFToolsError(f"File not found: {file_path}")

        reader = PdfReader(file_path)
        writer = PdfWriter()
        total_pages = len(reader.pages)

        target_pages = set(p - 1 if p > 0 else p for p in pages) if pages else set(range(total_pages))
        rotated_count = 0

        for i, page in enumerate(reader.pages):
            if i in target_pages:
                page.rotate(rotation)
                rotated_count += 1
            writer.add_page(page)

        if not output_path:
            base, ext = os.path.splitext(file_path)
            output_path = f"{base}_rotated{ext}"

        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

        with open(output_path, 'wb') as output_file:
            writer.write(output_file)

        return {
            'success': True,
            'output_path': output_path,
            'rotation': rotation,
            'pages_rotated': rotated_count,
            'total_pages': total_pages
        }

    @staticmethod
    def get_metadata(file_path: str) -> Dict[str, Any]:
        """Get metadata from a PDF file."""
        try:
            from pypdf import PdfReader
        except ImportError:
            raise PDFToolsError("pypdf not installed. Install with: pip install pypdf")

        if not os.path.exists(file_path):
            raise PDFToolsError(f"File not found: {file_path}")

        reader = PdfReader(file_path)
        metadata = reader.metadata

        meta_dict = {}
        if metadata:
            for key in metadata:
                value = metadata.get(key)
                if value:
                    clean_key = key.lstrip('/')
                    meta_dict[clean_key] = str(value)

        file_stats = os.stat(file_path)

        return {
            'success': True,
            'file_path': file_path,
            'page_count': len(reader.pages),
            'file_size_bytes': file_stats.st_size,
            'file_size_mb': round(file_stats.st_size / (1024 * 1024), 2),
            'metadata': meta_dict,
            'is_encrypted': reader.is_encrypted
        }


class PDFCreator:
    """Handles PDF creation using reportlab."""

    @staticmethod
    def create_pdf(content: str, output_path: str, title: Optional[str] = None,
                   page_size: str = 'A4') -> Dict[str, Any]:
        """Create a PDF from text or markdown content."""
        try:
            from reportlab.lib.pagesizes import A4, LETTER, LEGAL, landscape, portrait
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
            from reportlab.lib.colors import HexColor
        except ImportError:
            raise PDFToolsError("reportlab not installed. Install with: pip install reportlab")

        page_sizes = {
            'A4': A4,
            'LETTER': LETTER,
            'LEGAL': LEGAL,
            'A4-LANDSCAPE': landscape(A4),
            'LETTER-LANDSCAPE': landscape(LETTER)
        }

        selected_size = page_sizes.get(page_size.upper(), A4)

        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

        doc = SimpleDocTemplate(
            output_path,
            pagesize=selected_size,
            rightMargin=0.75 * inch,
            leftMargin=0.75 * inch,
            topMargin=0.75 * inch,
            bottomMargin=0.75 * inch
        )

        styles = getSampleStyleSheet()

        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=20,
            textColor=HexColor('#1e1e1e')
        )

        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            spaceBefore=15,
            spaceAfter=10,
            textColor=HexColor('#333333')
        )

        body_style = ParagraphStyle(
            'CustomBody',
            parent=styles['Normal'],
            fontSize=11,
            leading=16,
            spaceAfter=10
        )

        story = []

        if title:
            story.append(Paragraph(title, title_style))
            story.append(Spacer(1, 20))

        lines = content.split('\n')

        for line in lines:
            line = line.strip()

            if not line:
                story.append(Spacer(1, 10))
                continue

            if line.startswith('# '):
                story.append(Paragraph(line[2:], title_style))
            elif line.startswith('## '):
                story.append(Paragraph(line[3:], heading_style))
            elif line.startswith('### '):
                heading3_style = ParagraphStyle(
                    'Heading3',
                    parent=styles['Heading3'],
                    fontSize=14,
                    spaceBefore=12,
                    spaceAfter=8
                )
                story.append(Paragraph(line[4:], heading3_style))
            elif line.startswith('- ') or line.startswith('* '):
                bullet_text = f"\u2022  {line[2:]}"
                story.append(Paragraph(bullet_text, body_style))
            elif line.startswith('---') or line.startswith('***'):
                story.append(Spacer(1, 10))
                story.append(PageBreak())
            else:
                safe_line = line.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                if line.startswith('**') and line.endswith('**'):
                    safe_line = f"<b>{safe_line[2:-2]}</b>"
                elif line.startswith('*') and line.endswith('*'):
                    safe_line = f"<i>{safe_line[1:-1]}</i>"

                story.append(Paragraph(safe_line, body_style))

        doc.build(story)

        file_stats = os.stat(output_path)

        return {
            'success': True,
            'output_path': output_path,
            'title': title,
            'page_size': page_size,
            'file_size_bytes': file_stats.st_size,
            'file_size_mb': round(file_stats.st_size / (1024 * 1024), 2)
        }


def extract_text_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract text from a PDF file.

    Args:
        params: Dictionary containing:
            - file_path (str, required): Path to the PDF file
            - pages (list, optional): List of page numbers (0-indexed) to extract
            - layout (bool, optional): Preserve layout/formatting (default: False)

    Returns:
        Dictionary with success status, extracted text content by page
    """
    status.set_callback(params.pop('_status_callback', None))

    file_path = params.get('file_path')
    if not file_path:
        return {'success': False, 'error': 'file_path parameter is required'}

    pages = params.get('pages')
    layout = params.get('layout', False)

    try:
        extractor = PDFExtractor(file_path)
        return extractor.extract_text(pages=pages, layout=layout)
    except PDFToolsError as e:
        return {'success': False, 'error': str(e)}
    except Exception as e:
        logger.error(f"Text extraction failed: {e}", exc_info=True)
        return {'success': False, 'error': f'Text extraction failed: {str(e)}'}


def extract_tables_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract tables from a PDF file.

    Args:
        params: Dictionary containing:
            - file_path (str, required): Path to the PDF file
            - pages (list, optional): List of page numbers (0-indexed) to extract from
            - output_format (str, optional): 'json', 'csv', or 'dataframe' (default: 'json')

    Returns:
        Dictionary with success status, extracted tables
    """
    status.set_callback(params.pop('_status_callback', None))

    file_path = params.get('file_path')
    if not file_path:
        return {'success': False, 'error': 'file_path parameter is required'}

    pages = params.get('pages')
    output_format = params.get('output_format', 'json')

    if output_format not in ['json', 'csv', 'dataframe']:
        return {'success': False, 'error': "output_format must be 'json', 'csv', or 'dataframe'"}

    try:
        extractor = PDFExtractor(file_path)
        return extractor.extract_tables(pages=pages, output_format=output_format)
    except PDFToolsError as e:
        return {'success': False, 'error': str(e)}
    except Exception as e:
        logger.error(f"Table extraction failed: {e}", exc_info=True)
        return {'success': False, 'error': f'Table extraction failed: {str(e)}'}


def merge_pdfs_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple PDF files into a single PDF.

    Args:
        params: Dictionary containing:
            - file_paths (list, required): List of PDF file paths to merge
            - output_path (str, required): Path for the merged output PDF

    Returns:
        Dictionary with success status, output path, file count
    """
    status.set_callback(params.pop('_status_callback', None))

    file_paths = params.get('file_paths')
    output_path = params.get('output_path')

    if not file_paths:
        return {'success': False, 'error': 'file_paths parameter is required (list of PDF paths)'}
    if not isinstance(file_paths, list):
        return {'success': False, 'error': 'file_paths must be a list'}
    if len(file_paths) < 2:
        return {'success': False, 'error': 'At least 2 PDF files are required for merging'}
    if not output_path:
        return {'success': False, 'error': 'output_path parameter is required'}

    try:
        return PDFManipulator.merge_pdfs(file_paths, output_path)
    except PDFToolsError as e:
        return {'success': False, 'error': str(e)}
    except Exception as e:
        logger.error(f"PDF merge failed: {e}", exc_info=True)
        return {'success': False, 'error': f'PDF merge failed: {str(e)}'}


def split_pdf_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Split a PDF into separate page files.

    Args:
        params: Dictionary containing:
            - file_path (str, required): Path to the PDF file to split
            - pages (list/str, required): Page numbers to extract (1-indexed)
              Can be a list like [1, 3, 5] or a range string like "1-5"
            - output_dir (str, required): Directory to save split pages

    Returns:
        Dictionary with success status, output files list
    """
    status.set_callback(params.pop('_status_callback', None))
    file_path = params.get('file_path')
    pages = params.get('pages')
    output_dir = params.get('output_dir')

    if not file_path:
        return {'success': False, 'error': 'file_path parameter is required'}
    if not pages:
        return {'success': False, 'error': 'pages parameter is required (list or range string)'}
    if not output_dir:
        return {'success': False, 'error': 'output_dir parameter is required'}

    try:
        return PDFManipulator.split_pdf(file_path, pages, output_dir)
    except PDFToolsError as e:
        return {'success': False, 'error': str(e)}
    except Exception as e:
        logger.error(f"PDF split failed: {e}", exc_info=True)
        return {'success': False, 'error': f'PDF split failed: {str(e)}'}


def get_metadata_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get metadata from a PDF file.

    Args:
        params: Dictionary containing:
            - file_path (str, required): Path to the PDF file

    Returns:
        Dictionary with success status, page count, file size, metadata
    """
    status.set_callback(params.pop('_status_callback', None))

    file_path = params.get('file_path')
    if not file_path:
        return {'success': False, 'error': 'file_path parameter is required'}

    try:
        return PDFManipulator.get_metadata(file_path)
    except PDFToolsError as e:
        return {'success': False, 'error': str(e)}
    except Exception as e:
        logger.error(f"Metadata extraction failed: {e}", exc_info=True)
        return {'success': False, 'error': f'Metadata extraction failed: {str(e)}'}


def rotate_pages_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Rotate pages in a PDF file.

    Args:
        params: Dictionary containing:
            - file_path (str, required): Path to the PDF file
            - rotation (int, required): Rotation angle (90, 180, or 270 degrees)
            - pages (list, optional): Page numbers to rotate (1-indexed).
              If not provided, rotates all pages.
            - output_path (str, optional): Output file path.
              If not provided, creates a new file with '_rotated' suffix.

    Returns:
        Dictionary with success status, output path, rotation info
    """
    status.set_callback(params.pop('_status_callback', None))

    file_path = params.get('file_path')
    rotation = params.get('rotation')

    if not file_path:
        return {'success': False, 'error': 'file_path parameter is required'}
    if not rotation:
        return {'success': False, 'error': 'rotation parameter is required (90, 180, or 270)'}

    try:
        rotation = int(rotation)
    except (ValueError, TypeError):
        return {'success': False, 'error': 'rotation must be an integer (90, 180, or 270)'}

    pages = params.get('pages')
    output_path = params.get('output_path')

    try:
        return PDFManipulator.rotate_pages(file_path, rotation, pages, output_path)
    except PDFToolsError as e:
        return {'success': False, 'error': str(e)}
    except Exception as e:
        logger.error(f"Page rotation failed: {e}", exc_info=True)
        return {'success': False, 'error': f'Page rotation failed: {str(e)}'}


def create_pdf_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a PDF from text or markdown content.

    Args:
        params: Dictionary containing:
            - content (str, required): Text or markdown content to convert
            - output_path (str, required): Output file path for the PDF
            - title (str, optional): Document title
            - page_size (str, optional): Page size - 'A4', 'LETTER', 'LEGAL',
              'A4-LANDSCAPE', 'LETTER-LANDSCAPE' (default: 'A4')

    Returns:
        Dictionary with success status, output path, file size

    Supported Markdown:
        - # Heading 1, ## Heading 2, ### Heading 3
        - Bullet points with - or *
        - **bold** and *italic* text
        - --- or *** for page breaks
    """
    status.set_callback(params.pop('_status_callback', None))

    content = params.get('content')
    output_path = params.get('output_path')

    if not content:
        return {'success': False, 'error': 'content parameter is required'}
    if not output_path:
        return {'success': False, 'error': 'output_path parameter is required'}

    title = params.get('title')
    page_size = params.get('page_size', 'A4')

    try:
        return PDFCreator.create_pdf(content, output_path, title, page_size)
    except PDFToolsError as e:
        return {'success': False, 'error': str(e)}
    except Exception as e:
        logger.error(f"PDF creation failed: {e}", exc_info=True)
        return {'success': False, 'error': f'PDF creation failed: {str(e)}'}
