import subprocess
from pathlib import Path
from typing import Dict, Any, Optional
import shutil


def convert_to_pdf_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a document to PDF format using Pandoc.
    
    Args:
        params: Dictionary containing:
            - input_file (str, required): Path to input file
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
        cmd = [
            'pandoc',
            str(input_path),
            '-o', str(output_path),
            '--pdf-engine=xelatex',
            '--standalone',  # Generate standalone document
            '-V', f'geometry:{pandoc_size},margin=1in',
            '-V', 'fontsize=11pt'  # Ensure readable font size
        ]
        
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
    
    Args:
        params: Dictionary containing:
            - input_file (str, required): Path to input file
            - output_file (str, optional): Output DOCX path
            - title (str, optional): Document title
    
    Returns:
        Dictionary with:
            - success (bool): Whether conversion succeeded
            - output_path (str): Path to generated DOCX
            - error (str, optional): Error message if failed
    """
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
    
    Args:
        params: Dictionary containing:
            - input_file (str, required): Path to input file
            - output_file (str, optional): Output HTML path
            - title (str, optional): Document title
            - standalone (bool, optional): Generate standalone HTML, default: True
    
    Returns:
        Dictionary with:
            - success (bool): Whether conversion succeeded
            - output_path (str): Path to generated HTML
            - error (str, optional): Error message if failed
    """
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
