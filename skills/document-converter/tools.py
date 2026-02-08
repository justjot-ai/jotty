import subprocess
from pathlib import Path
from typing import Dict, Any, Optional
import shutil

from Jotty.core.utils.skill_status import SkillStatus

# Status emitter for progress updates
status = SkillStatus("document-converter")



def convert_to_pdf_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a document to PDF format using Pandoc.

    IMPORTANT: Pandoc can convert FROM markdown, HTML, DOCX, EPUB, LaTeX, etc.
    but CANNOT convert FROM PDF. Always use the original source file (markdown/HTML/DOCX)
    as input, not a PDF file.

    Supported input formats: markdown (.md), HTML (.html), DOCX (.docx), EPUB (.epub),
    LaTeX (.tex), reStructuredText (.rst), Textile, MediaWiki, etc.

    Args:
        params: Dictionary containing:
            - input_file (str, required): Path to input file (markdown, HTML, DOCX, etc. - NOT PDF)
            - output_file (str, optional): Output PDF path
            - page_size (str, optional): Page size (a4, a5, a6, letter, remarkable), default: 'a4'
            - title (str, optional): Document title
            - author (str, optional): Document author

    Returns:
        Dictionary with:
            - success (bool): Whether conversion succeeded
            - output_path (str): Path to generated PDF
            - error (str, optional): Error message if failed
    """
    status.set_callback(params.pop('_status_callback', None))
    status.emit("Converting", "📄 Converting to PDF...")

    try:
        input_file = params.get('input_file')
        if not input_file:
            return {
                'success': False,
                'error': 'input_file parameter is required'
            }
        
        input_path = Path(input_file)
        if not input_path.exists():
            return {
                'success': False,
                'error': f'Input file not found: {input_file}'
            }
        
        # Check if trying to convert from PDF (not supported)
        if input_path.suffix.lower() == '.pdf':
            return {
                'success': False,
                'error': 'Pandoc cannot convert FROM PDF. Use the original source file (markdown, HTML, DOCX, etc.) as input instead of a PDF file.'
            }
        
        # Determine output path
        output_file = params.get('output_file')
        if output_file:
            output_path = Path(output_file)
        else:
            output_path = input_path.with_suffix('.pdf')
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Page size mapping
        page_size = params.get('page_size', 'a4').lower()
        page_size_map = {
            'a4': 'a4paper',
            'a5': 'a5paper',
            'a6': 'a6paper',
            'letter': 'letterpaper',
            'remarkable': 'a5paper'  # reMarkable uses A5-like aspect ratio
        }
        pandoc_size = page_size_map.get(page_size, 'a4paper')
        
        # Build pandoc command - use geometry package with proper syntax
        # Add wrapping and formatting options for better PDF rendering
        cmd = [
            'pandoc',
            str(input_path),
            '-o', str(output_path),
            '--pdf-engine=xelatex',
            '--standalone',  # Generate standalone document
            '-V', f'geometry:{pandoc_size},margin=1in',
            '-V', 'fontsize=11pt',  # Ensure readable font size
            '-V', 'linestretch=1.15',  # Better line spacing (1.15 = 15% extra)
            '-V', 'urlcolor=blue',  # Make URLs blue
            '-V', 'linkcolor=blue',  # Make links blue
            '--toc',  # Add table of contents
            '--toc-depth=3',  # TOC depth
            '--highlight-style=tango',  # Code highlighting style
            '-V', 'colorlinks=true',  # Colored links
            '-V', 'breakurl=true',  # Break long URLs
        ]
        
        # Add URL wrapping support via geometry package
        # Pandoc will automatically wrap text, but we ensure URLs break properly
        
        # Add metadata
        title = params.get('title')
        if title:
            cmd.extend(['-M', f'title={title}'])
        
        author = params.get('author')
        if author:
            cmd.extend(['-M', f'author={author}'])
        
        # Execute pandoc
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode != 0:
            return {
                'success': False,
                'error': f'Pandoc conversion failed: {result.stderr}'
            }
        
        if not output_path.exists():
            return {
                'success': False,
                'error': 'PDF file was not created'
            }
        
        return {
            'success': True,
            'output_path': str(output_path),
            'file_size': output_path.stat().st_size
        }
    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'error': 'Conversion timed out after 60 seconds'
        }
    except Exception as e:
        return {
            'success': False,
            'error': f'Error converting to PDF: {str(e)}'
        }


def convert_to_epub_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a document to EPUB format using Pandoc.
    
    Args:
        params: Dictionary containing:
            - input_file (str, required): Path to input file
            - output_file (str, optional): Output EPUB path
            - title (str, optional): Document title
            - author (str, optional): Document author
    
    Returns:
        Dictionary with:
            - success (bool): Whether conversion succeeded
            - output_path (str): Path to generated EPUB
            - error (str, optional): Error message if failed
    """
    status.set_callback(params.pop('_status_callback', None))

    try:
        input_file = params.get('input_file')
        if not input_file:
            return {
                'success': False,
                'error': 'input_file parameter is required'
            }
        
        input_path = Path(input_file)
        if not input_path.exists():
            return {
                'success': False,
                'error': f'Input file not found: {input_file}'
            }
        
        output_file = params.get('output_file')
        if output_file:
            output_path = Path(output_file)
        else:
            output_path = input_path.with_suffix('.epub')
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        cmd = ['pandoc', str(input_path), '-o', str(output_path)]
        
        title = params.get('title')
        if title:
            cmd.extend(['-M', f'title={title}'])
        
        author = params.get('author')
        if author:
            cmd.extend(['-M', f'author={author}'])
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode != 0:
            return {
                'success': False,
                'error': f'Pandoc conversion failed: {result.stderr}'
            }
        
        if not output_path.exists():
            return {
                'success': False,
                'error': 'EPUB file was not created'
            }
        
        return {
            'success': True,
            'output_path': str(output_path),
            'file_size': output_path.stat().st_size
        }
    except Exception as e:
        return {
            'success': False,
            'error': f'Error converting to EPUB: {str(e)}'
        }


def convert_to_docx_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a document to DOCX format using Pandoc.
    
    IMPORTANT: Pandoc CANNOT convert FROM PDF. Use the original source file 
    (markdown, HTML, etc.) as input, not a PDF file.
    
    Supported input formats: markdown (.md), HTML (.html), EPUB (.epub), 
    LaTeX (.tex), reStructuredText (.rst), Textile, MediaWiki, etc.
    NOT supported: PDF (.pdf) - Pandoc cannot read PDF files.
    
    Args:
        params: Dictionary containing:
            - input_file (str, required): Path to input file (markdown, HTML, etc. - NOT PDF)
            - output_file (str, optional): Output DOCX path
            - title (str, optional): Document title
    
    Returns:
        Dictionary with:
            - success (bool): Whether conversion succeeded
            - output_path (str): Path to generated DOCX
            - error (str, optional): Error message if failed
    """
    status.set_callback(params.pop('_status_callback', None))

    try:
        input_file = params.get('input_file')
        if not input_file:
            return {
                'success': False,
                'error': 'input_file parameter is required'
            }
        
        input_path = Path(input_file)
        
        # Check if trying to convert from PDF (not supported) - check BEFORE file existence
        if input_path.suffix.lower() == '.pdf':
            return {
                'success': False,
                'error': 'Pandoc cannot convert FROM PDF. Use the original source file (markdown, HTML, DOCX, etc.) as input instead of a PDF file.'
            }
        
        if not input_path.exists():
            return {
                'success': False,
                'error': f'Input file not found: {input_file}'
            }
        
        output_file = params.get('output_file')
        if output_file:
            output_path = Path(output_file)
        else:
            output_path = input_path.with_suffix('.docx')
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        cmd = ['pandoc', str(input_path), '-o', str(output_path)]
        
        title = params.get('title')
        if title:
            cmd.extend(['-M', f'title={title}'])
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode != 0:
            return {
                'success': False,
                'error': f'Pandoc conversion failed: {result.stderr}'
            }
        
        if not output_path.exists():
            return {
                'success': False,
                'error': 'DOCX file was not created'
            }
        
        return {
            'success': True,
            'output_path': str(output_path),
            'file_size': output_path.stat().st_size
        }
    except Exception as e:
        return {
            'success': False,
            'error': f'Error converting to DOCX: {str(e)}'
        }


def convert_to_html_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a document to HTML format using Pandoc.
    
    IMPORTANT: Pandoc CANNOT convert FROM PDF. Use the original source file 
    (markdown, DOCX, etc.) as input, not a PDF file.
    
    Supported input formats: markdown (.md), DOCX (.docx), EPUB (.epub), 
    LaTeX (.tex), reStructuredText (.rst), Textile, MediaWiki, etc.
    NOT supported: PDF (.pdf) - Pandoc cannot read PDF files.
    
    Args:
        params: Dictionary containing:
            - input_file (str, required): Path to input file (markdown, DOCX, etc. - NOT PDF)
            - output_file (str, optional): Output HTML path
            - title (str, optional): Document title
            - standalone (bool, optional): Generate standalone HTML, default: True
    
    Returns:
        Dictionary with:
            - success (bool): Whether conversion succeeded
            - output_path (str): Path to generated HTML
            - error (str, optional): Error message if failed
    """
    status.set_callback(params.pop('_status_callback', None))

    try:
        input_file = params.get('input_file')
        if not input_file:
            return {
                'success': False,
                'error': 'input_file parameter is required'
            }
        
        input_path = Path(input_file)
        
        # Check if trying to convert from PDF (not supported) - check BEFORE file existence
        if input_path.suffix.lower() == '.pdf':
            return {
                'success': False,
                'error': 'Pandoc cannot convert FROM PDF. Use the original source file (markdown, DOCX, HTML, etc.) as input instead of a PDF file.'
            }
        
        if not input_path.exists():
            return {
                'success': False,
                'error': f'Input file not found: {input_file}'
            }
        
        output_file = params.get('output_file')
        if output_file:
            output_path = Path(output_file)
        else:
            output_path = input_path.with_suffix('.html')
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        cmd = ['pandoc', str(input_path), '-o', str(output_path)]
        
        if params.get('standalone', True):
            cmd.append('--standalone')
        
        title = params.get('title')
        if title:
            cmd.extend(['-M', f'title={title}'])
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode != 0:
            return {
                'success': False,
                'error': f'Pandoc conversion failed: {result.stderr}'
            }
        
        if not output_path.exists():
            return {
                'success': False,
                'error': 'HTML file was not created'
            }
        
        return {
            'success': True,
            'output_path': str(output_path),
            'file_size': output_path.stat().st_size
        }
    except Exception as e:
        return {
            'success': False,
            'error': f'Error converting to HTML: {str(e)}'
        }


def convert_to_markdown_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a document to Markdown format using Pandoc.
    
    Args:
        params: Dictionary containing:
            - input_file (str, required): Path to input file
            - output_file (str, optional): Output Markdown path
    
    Returns:
        Dictionary with:
            - success (bool): Whether conversion succeeded
            - output_path (str): Path to generated Markdown
            - error (str, optional): Error message if failed
    """
    status.set_callback(params.pop('_status_callback', None))

    try:
        input_file = params.get('input_file')
        if not input_file:
            return {
                'success': False,
                'error': 'input_file parameter is required'
            }
        
        input_path = Path(input_file)
        if not input_path.exists():
            return {
                'success': False,
                'error': f'Input file not found: {input_file}'
            }
        
        output_file = params.get('output_file')
        if output_file:
            output_path = Path(output_file)
        else:
            output_path = input_path.with_suffix('.md')
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        cmd = ['pandoc', str(input_path), '-o', str(output_path), '-t', 'markdown']
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode != 0:
            return {
                'success': False,
                'error': f'Pandoc conversion failed: {result.stderr}'
            }
        
        if not output_path.exists():
            return {
                'success': False,
                'error': 'Markdown file was not created'
            }
        
        return {
            'success': True,
            'output_path': str(output_path),
            'file_size': output_path.stat().st_size
        }
    except Exception as e:
        return {
            'success': False,
            'error': f'Error converting to Markdown: {str(e)}'
        }
