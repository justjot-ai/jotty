"""
DOCX Tools Skill

Word document toolkit using python-docx for reading, creating, and manipulating Word documents.
"""
import os
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


def read_docx_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Read text content from a Word document.

    Args:
        params: Dictionary containing:
            - file_path (str, required): Path to the .docx file
            - include_tables (bool, optional): Include table content (default: True)

    Returns:
        Dictionary with:
            - success (bool): Whether reading succeeded
            - text (str): Extracted text content
            - paragraphs (list): List of paragraph texts
            - tables (list): List of tables (if include_tables is True)
            - error (str, optional): Error message if failed
    """
    try:
        from docx import Document
    except ImportError:
        return {
            'success': False,
            'error': 'python-docx not installed. Install with: pip install python-docx'
        }

    file_path = params.get('file_path')
    include_tables = params.get('include_tables', True)

    if not file_path:
        return {'success': False, 'error': 'file_path parameter is required'}

    if not os.path.exists(file_path):
        return {'success': False, 'error': f'File not found: {file_path}'}

    try:
        doc = Document(file_path)

        paragraphs = [para.text for para in doc.paragraphs]
        full_text = '\n'.join(paragraphs)

        tables_data = []
        if include_tables:
            for table in doc.tables:
                table_rows = []
                for row in table.rows:
                    row_data = [cell.text for cell in row.cells]
                    table_rows.append(row_data)
                tables_data.append(table_rows)

        logger.info(f"Document read: {file_path}")

        return {
            'success': True,
            'text': full_text,
            'paragraphs': paragraphs,
            'tables': tables_data,
            'paragraph_count': len(paragraphs),
            'table_count': len(tables_data)
        }

    except Exception as e:
        logger.error(f"Failed to read document: {e}", exc_info=True)
        return {'success': False, 'error': f'Failed to read document: {str(e)}'}


def create_docx_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a new Word document.

    Args:
        params: Dictionary containing:
            - content (str or list, required): Text content or list of paragraphs
            - output_path (str, required): Output file path
            - title (str, optional): Document title (added as Heading1)

    Returns:
        Dictionary with:
            - success (bool): Whether creation succeeded
            - file_path (str): Path to created document
            - error (str, optional): Error message if failed
    """
    try:
        from docx import Document
    except ImportError:
        return {
            'success': False,
            'error': 'python-docx not installed. Install with: pip install python-docx'
        }

    content = params.get('content')
    output_path = params.get('output_path')
    title = params.get('title')

    if not content:
        return {'success': False, 'error': 'content parameter is required'}

    if not output_path:
        return {'success': False, 'error': 'output_path parameter is required'}

    try:
        doc = Document()

        if title:
            doc.add_heading(title, level=0)

        if isinstance(content, str):
            paragraphs = content.split('\n')
        elif isinstance(content, list):
            paragraphs = content
        else:
            paragraphs = [str(content)]

        for para_text in paragraphs:
            if para_text.strip():
                doc.add_paragraph(para_text)

        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        doc.save(output_path)
        logger.info(f"Document created: {output_path}")

        return {
            'success': True,
            'file_path': output_path,
            'paragraph_count': len(paragraphs)
        }

    except Exception as e:
        logger.error(f"Failed to create document: {e}", exc_info=True)
        return {'success': False, 'error': f'Failed to create document: {str(e)}'}


def add_paragraph_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add a paragraph to an existing Word document.

    Args:
        params: Dictionary containing:
            - file_path (str, required): Path to the .docx file
            - text (str, required): Paragraph text to add
            - style (str, optional): Paragraph style (Normal, Heading1, Heading2, etc.)

    Returns:
        Dictionary with:
            - success (bool): Whether addition succeeded
            - file_path (str): Path to modified document
            - error (str, optional): Error message if failed
    """
    try:
        from docx import Document
    except ImportError:
        return {
            'success': False,
            'error': 'python-docx not installed. Install with: pip install python-docx'
        }

    file_path = params.get('file_path')
    text = params.get('text')
    style = params.get('style', 'Normal')

    if not file_path:
        return {'success': False, 'error': 'file_path parameter is required'}

    if not text:
        return {'success': False, 'error': 'text parameter is required'}

    if not os.path.exists(file_path):
        return {'success': False, 'error': f'File not found: {file_path}'}

    try:
        doc = Document(file_path)

        if style.startswith('Heading'):
            level = int(style[-1]) if style[-1].isdigit() else 1
            doc.add_heading(text, level=level)
        else:
            para = doc.add_paragraph(text)
            try:
                para.style = style
            except KeyError:
                para.style = 'Normal'
                logger.warning(f"Style '{style}' not found, using 'Normal'")

        doc.save(file_path)
        logger.info(f"Paragraph added to: {file_path}")

        return {
            'success': True,
            'file_path': file_path,
            'style_applied': style
        }

    except Exception as e:
        logger.error(f"Failed to add paragraph: {e}", exc_info=True)
        return {'success': False, 'error': f'Failed to add paragraph: {str(e)}'}


def add_table_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add a table to an existing Word document.

    Args:
        params: Dictionary containing:
            - file_path (str, required): Path to the .docx file
            - data (list, required): List of lists representing table rows
            - headers (list, optional): List of header strings (if provided, adds header row)

    Returns:
        Dictionary with:
            - success (bool): Whether addition succeeded
            - file_path (str): Path to modified document
            - rows (int): Number of rows added
            - cols (int): Number of columns
            - error (str, optional): Error message if failed
    """
    try:
        from docx import Document
        from docx.shared import Inches
    except ImportError:
        return {
            'success': False,
            'error': 'python-docx not installed. Install with: pip install python-docx'
        }

    file_path = params.get('file_path')
    data = params.get('data')
    headers = params.get('headers')

    if not file_path:
        return {'success': False, 'error': 'file_path parameter is required'}

    if not data:
        return {'success': False, 'error': 'data parameter is required'}

    if not isinstance(data, list) or not all(isinstance(row, list) for row in data):
        return {'success': False, 'error': 'data must be a list of lists'}

    if not os.path.exists(file_path):
        return {'success': False, 'error': f'File not found: {file_path}'}

    try:
        doc = Document(file_path)

        all_rows = []
        if headers:
            all_rows.append(headers)
        all_rows.extend(data)

        if not all_rows:
            return {'success': False, 'error': 'No data to add'}

        num_cols = max(len(row) for row in all_rows)
        num_rows = len(all_rows)

        table = doc.add_table(rows=num_rows, cols=num_cols)
        table.style = 'Table Grid'

        for i, row_data in enumerate(all_rows):
            row = table.rows[i]
            for j, cell_text in enumerate(row_data):
                if j < num_cols:
                    row.cells[j].text = str(cell_text)

        if headers:
            for cell in table.rows[0].cells:
                for paragraph in cell.paragraphs:
                    for run in paragraph.runs:
                        run.bold = True

        doc.save(file_path)
        logger.info(f"Table added to: {file_path}")

        return {
            'success': True,
            'file_path': file_path,
            'rows': num_rows,
            'cols': num_cols
        }

    except Exception as e:
        logger.error(f"Failed to add table: {e}", exc_info=True)
        return {'success': False, 'error': f'Failed to add table: {str(e)}'}


def add_image_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add an image to an existing Word document.

    Args:
        params: Dictionary containing:
            - file_path (str, required): Path to the .docx file
            - image_path (str, required): Path to the image file
            - width (float, optional): Image width in inches (default: 6.0)

    Returns:
        Dictionary with:
            - success (bool): Whether addition succeeded
            - file_path (str): Path to modified document
            - image_path (str): Path to the image that was added
            - error (str, optional): Error message if failed
    """
    try:
        from docx import Document
        from docx.shared import Inches
    except ImportError:
        return {
            'success': False,
            'error': 'python-docx not installed. Install with: pip install python-docx'
        }

    file_path = params.get('file_path')
    image_path = params.get('image_path')
    width = params.get('width', 6.0)

    if not file_path:
        return {'success': False, 'error': 'file_path parameter is required'}

    if not image_path:
        return {'success': False, 'error': 'image_path parameter is required'}

    if not os.path.exists(file_path):
        return {'success': False, 'error': f'Document not found: {file_path}'}

    if not os.path.exists(image_path):
        return {'success': False, 'error': f'Image not found: {image_path}'}

    try:
        doc = Document(file_path)

        doc.add_picture(image_path, width=Inches(width))

        doc.save(file_path)
        logger.info(f"Image added to: {file_path}")

        return {
            'success': True,
            'file_path': file_path,
            'image_path': image_path,
            'width_inches': width
        }

    except Exception as e:
        logger.error(f"Failed to add image: {e}", exc_info=True)
        return {'success': False, 'error': f'Failed to add image: {str(e)}'}


def replace_text_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Find and replace text in a Word document.

    Args:
        params: Dictionary containing:
            - file_path (str, required): Path to the .docx file
            - find (str, required): Text to find
            - replace (str, required): Text to replace with

    Returns:
        Dictionary with:
            - success (bool): Whether replacement succeeded
            - file_path (str): Path to modified document
            - replacements (int): Number of replacements made
            - error (str, optional): Error message if failed
    """
    try:
        from docx import Document
    except ImportError:
        return {
            'success': False,
            'error': 'python-docx not installed. Install with: pip install python-docx'
        }

    file_path = params.get('file_path')
    find_text = params.get('find')
    replace_text = params.get('replace')

    if not file_path:
        return {'success': False, 'error': 'file_path parameter is required'}

    if not find_text:
        return {'success': False, 'error': 'find parameter is required'}

    if replace_text is None:
        return {'success': False, 'error': 'replace parameter is required'}

    if not os.path.exists(file_path):
        return {'success': False, 'error': f'File not found: {file_path}'}

    try:
        doc = Document(file_path)
        replacements = 0

        for para in doc.paragraphs:
            if find_text in para.text:
                for run in para.runs:
                    if find_text in run.text:
                        count = run.text.count(find_text)
                        run.text = run.text.replace(find_text, replace_text)
                        replacements += count

        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    for para in cell.paragraphs:
                        if find_text in para.text:
                            for run in para.runs:
                                if find_text in run.text:
                                    count = run.text.count(find_text)
                                    run.text = run.text.replace(find_text, replace_text)
                                    replacements += count

        doc.save(file_path)
        logger.info(f"Text replaced in: {file_path}, {replacements} replacements")

        return {
            'success': True,
            'file_path': file_path,
            'replacements': replacements,
            'find': find_text,
            'replace': replace_text
        }

    except Exception as e:
        logger.error(f"Failed to replace text: {e}", exc_info=True)
        return {'success': False, 'error': f'Failed to replace text: {str(e)}'}


def get_styles_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    List available styles in a Word document.

    Args:
        params: Dictionary containing:
            - file_path (str, required): Path to the .docx file

    Returns:
        Dictionary with:
            - success (bool): Whether listing succeeded
            - styles (list): List of style names with their types
            - paragraph_styles (list): List of paragraph style names
            - character_styles (list): List of character style names
            - error (str, optional): Error message if failed
    """
    try:
        from docx import Document
        from docx.enum.style import WD_STYLE_TYPE
    except ImportError:
        return {
            'success': False,
            'error': 'python-docx not installed. Install with: pip install python-docx'
        }

    file_path = params.get('file_path')

    if not file_path:
        return {'success': False, 'error': 'file_path parameter is required'}

    if not os.path.exists(file_path):
        return {'success': False, 'error': f'File not found: {file_path}'}

    try:
        doc = Document(file_path)

        all_styles = []
        paragraph_styles = []
        character_styles = []
        table_styles = []

        for style in doc.styles:
            style_info = {
                'name': style.name,
                'type': str(style.type).split('.')[-1].replace('>', '')
            }
            all_styles.append(style_info)

            if style.type == WD_STYLE_TYPE.PARAGRAPH:
                paragraph_styles.append(style.name)
            elif style.type == WD_STYLE_TYPE.CHARACTER:
                character_styles.append(style.name)
            elif style.type == WD_STYLE_TYPE.TABLE:
                table_styles.append(style.name)

        logger.info(f"Styles listed from: {file_path}")

        return {
            'success': True,
            'styles': all_styles,
            'paragraph_styles': sorted(paragraph_styles),
            'character_styles': sorted(character_styles),
            'table_styles': sorted(table_styles),
            'total_count': len(all_styles)
        }

    except Exception as e:
        logger.error(f"Failed to list styles: {e}", exc_info=True)
        return {'success': False, 'error': f'Failed to list styles: {str(e)}'}
