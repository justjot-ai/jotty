"""
PowerPoint Editor Skill

Editing toolkit for existing PowerPoint (.pptx) files using python-pptx.
Different from slide-generator which creates new presentations.
"""
import os
import logging
from typing import Dict, Any, List, Optional

from Jotty.core.utils.skill_status import SkillStatus

# Status emitter for progress updates
status = SkillStatus("pptx-editor")


logger = logging.getLogger(__name__)


class PPTXEditorError(Exception):
    """Custom exception for PPTX editor operations."""
    pass


class PPTXEditor:
    """Class for handling PowerPoint editing operations."""

    @staticmethod
    def _validate_file_path(file_path: str) -> None:
        """Validate that file exists and is a PPTX file."""
        if not file_path:
            raise PPTXEditorError("file_path parameter is required")
        if not os.path.exists(file_path):
            raise PPTXEditorError(f"File not found: {file_path}")
        if not file_path.lower().endswith('.pptx'):
            raise PPTXEditorError("File must be a .pptx file")

    @staticmethod
    def _get_presentation(file_path: str):
        """Load and return a Presentation object."""
        try:
            from pptx import Presentation
        except ImportError:
            raise PPTXEditorError("python-pptx not installed. Install with: pip install python-pptx")
        return Presentation(file_path)

    @staticmethod
    def _extract_shape_text(shape) -> str:
        """Extract text from a shape."""
        if hasattr(shape, 'text'):
            return shape.text
        if hasattr(shape, 'text_frame'):
            return '\n'.join([p.text for p in shape.text_frame.paragraphs])
        return ''

    @staticmethod
    def _get_slide_content(slide, include_notes: bool = False) -> Dict[str, Any]:
        """Extract content from a single slide."""
        content = {
            'title': '',
            'content': [],
            'shapes_count': len(slide.shapes)
        }

        for shape in slide.shapes:
            text = PPTXEditor._extract_shape_text(shape)
            if text:
                if shape.has_text_frame and shape == slide.shapes.title:
                    content['title'] = text
                else:
                    content['content'].append(text)

        if include_notes and slide.has_notes_slide:
            notes_slide = slide.notes_slide
            notes_text = notes_slide.notes_text_frame.text if notes_slide.notes_text_frame else ''
            content['notes'] = notes_text

        return content


def read_pptx_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Read presentation content from a PPTX file.

    Args:
        params: Dictionary containing:
            - file_path (str, required): Path to the PPTX file
            - include_notes (bool, optional): Include slide notes (default: False)

    Returns:
        Dictionary with:
            - success (bool): Whether operation succeeded
            - slides (list): List of slide content dicts
            - slide_count (int): Total number of slides
            - error (str, optional): Error message if failed
    """
    status.set_callback(params.pop('_status_callback', None))

    file_path = params.get('file_path')
    include_notes = params.get('include_notes', False)

    try:
        PPTXEditor._validate_file_path(file_path)
        prs = PPTXEditor._get_presentation(file_path)

        slides = []
        for idx, slide in enumerate(prs.slides):
            slide_content = PPTXEditor._get_slide_content(slide, include_notes)
            slide_content['index'] = idx
            slides.append(slide_content)

        logger.info(f"Read {len(slides)} slides from: {file_path}")

        return {
            'success': True,
            'slides': slides,
            'slide_count': len(slides),
            'file_path': file_path
        }

    except PPTXEditorError as e:
        return {'success': False, 'error': str(e)}
    except Exception as e:
        logger.error(f"Failed to read PPTX: {e}", exc_info=True)
        return {'success': False, 'error': f'Failed to read presentation: {str(e)}'}


def get_slide_layouts_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get available slide layouts from a PPTX file.

    Args:
        params: Dictionary containing:
            - file_path (str, required): Path to the PPTX file

    Returns:
        Dictionary with:
            - success (bool): Whether operation succeeded
            - layouts (list): List of layout dicts with index and name
            - error (str, optional): Error message if failed
    """
    status.set_callback(params.pop('_status_callback', None))

    file_path = params.get('file_path')

    try:
        PPTXEditor._validate_file_path(file_path)
        prs = PPTXEditor._get_presentation(file_path)

        layouts = []
        for idx, layout in enumerate(prs.slide_layouts):
            layouts.append({
                'index': idx,
                'name': layout.name
            })

        logger.info(f"Found {len(layouts)} layouts in: {file_path}")

        return {
            'success': True,
            'layouts': layouts,
            'layout_count': len(layouts)
        }

    except PPTXEditorError as e:
        return {'success': False, 'error': str(e)}
    except Exception as e:
        logger.error(f"Failed to get layouts: {e}", exc_info=True)
        return {'success': False, 'error': f'Failed to get layouts: {str(e)}'}


def add_slide_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add a new slide to a presentation.

    Args:
        params: Dictionary containing:
            - file_path (str, required): Path to the PPTX file
            - layout_index (int, optional): Index of slide layout to use (default: 1)
            - title (str, optional): Slide title
            - content (str or list, optional): Slide content (text or list of bullets)
            - position (int, optional): Position to insert slide (default: end)

    Returns:
        Dictionary with:
            - success (bool): Whether operation succeeded
            - slide_index (int): Index of the new slide
            - slide_count (int): Total slides after addition
            - error (str, optional): Error message if failed
    """
    status.set_callback(params.pop('_status_callback', None))

    file_path = params.get('file_path')
    layout_index = params.get('layout_index', 1)
    title = params.get('title', '')
    content = params.get('content', '')
    position = params.get('position')

    try:
        from pptx.util import Inches, Pt

        PPTXEditor._validate_file_path(file_path)
        prs = PPTXEditor._get_presentation(file_path)

        if layout_index >= len(prs.slide_layouts):
            layout_index = 1 if len(prs.slide_layouts) > 1 else 0

        slide_layout = prs.slide_layouts[layout_index]
        slide = prs.slides.add_slide(slide_layout)

        # Set title if shape exists
        if slide.shapes.title and title:
            slide.shapes.title.text = title

        # Add content
        if content:
            # Find content placeholder or create text box
            content_placeholder = None
            for shape in slide.shapes:
                if shape.has_text_frame and shape != slide.shapes.title:
                    content_placeholder = shape
                    break

            if content_placeholder:
                tf = content_placeholder.text_frame
                if isinstance(content, list):
                    for i, bullet in enumerate(content):
                        if i == 0:
                            tf.paragraphs[0].text = bullet
                        else:
                            p = tf.add_paragraph()
                            p.text = bullet
                            p.level = 0
                else:
                    tf.text = content
            else:
                # Create text box
                left = Inches(0.5)
                top = Inches(1.5)
                width = Inches(9)
                height = Inches(5)
                textbox = slide.shapes.add_textbox(left, top, width, height)
                tf = textbox.text_frame
                tf.word_wrap = True

                if isinstance(content, list):
                    for i, bullet in enumerate(content):
                        if i == 0:
                            tf.paragraphs[0].text = f"- {bullet}"
                        else:
                            p = tf.add_paragraph()
                            p.text = f"- {bullet}"
                else:
                    tf.text = content

        # Move slide to position if specified
        if position is not None and position < len(prs.slides) - 1:
            # Get the slide element
            slide_elem = slide._element
            slides_elem = prs.slides._sldIdLst
            # Move to position
            slides_elem.remove(slides_elem[-1])
            slides_elem.insert(position, slide._element.getparent())

        prs.save(file_path)
        new_index = len(prs.slides) - 1 if position is None else position

        logger.info(f"Added slide at index {new_index} to: {file_path}")

        return {
            'success': True,
            'slide_index': new_index,
            'slide_count': len(prs.slides),
            'file_path': file_path
        }

    except PPTXEditorError as e:
        return {'success': False, 'error': str(e)}
    except Exception as e:
        logger.error(f"Failed to add slide: {e}", exc_info=True)
        return {'success': False, 'error': f'Failed to add slide: {str(e)}'}


def update_slide_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update content of an existing slide.

    Args:
        params: Dictionary containing:
            - file_path (str, required): Path to the PPTX file
            - slide_index (int, required): Index of slide to update (0-based)
            - title (str, optional): New slide title
            - content (str or list, optional): New slide content

    Returns:
        Dictionary with:
            - success (bool): Whether operation succeeded
            - slide_index (int): Index of updated slide
            - error (str, optional): Error message if failed
    """
    status.set_callback(params.pop('_status_callback', None))

    file_path = params.get('file_path')
    slide_index = params.get('slide_index')
    title = params.get('title')
    content = params.get('content')

    try:
        from pptx.util import Inches

        PPTXEditor._validate_file_path(file_path)

        if slide_index is None:
            return {'success': False, 'error': 'slide_index parameter is required'}

        prs = PPTXEditor._get_presentation(file_path)

        if slide_index < 0 or slide_index >= len(prs.slides):
            return {'success': False, 'error': f'Invalid slide_index: {slide_index}. Presentation has {len(prs.slides)} slides.'}

        slide = prs.slides[slide_index]

        # Update title
        if title is not None and slide.shapes.title:
            slide.shapes.title.text = title

        # Update content
        if content is not None:
            content_placeholder = None
            for shape in slide.shapes:
                if shape.has_text_frame and shape != slide.shapes.title:
                    content_placeholder = shape
                    break

            if content_placeholder:
                tf = content_placeholder.text_frame
                # Clear existing paragraphs
                for para in tf.paragraphs:
                    para.clear()

                if isinstance(content, list):
                    for i, bullet in enumerate(content):
                        if i == 0:
                            tf.paragraphs[0].text = bullet
                        else:
                            p = tf.add_paragraph()
                            p.text = bullet
                            p.level = 0
                else:
                    tf.paragraphs[0].text = content

        prs.save(file_path)

        logger.info(f"Updated slide {slide_index} in: {file_path}")

        return {
            'success': True,
            'slide_index': slide_index,
            'file_path': file_path
        }

    except PPTXEditorError as e:
        return {'success': False, 'error': str(e)}
    except Exception as e:
        logger.error(f"Failed to update slide: {e}", exc_info=True)
        return {'success': False, 'error': f'Failed to update slide: {str(e)}'}


def delete_slide_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Delete a slide from a presentation.

    Args:
        params: Dictionary containing:
            - file_path (str, required): Path to the PPTX file
            - slide_index (int, required): Index of slide to delete (0-based)

    Returns:
        Dictionary with:
            - success (bool): Whether operation succeeded
            - deleted_index (int): Index of deleted slide
            - slide_count (int): Total slides after deletion
            - error (str, optional): Error message if failed
    """
    status.set_callback(params.pop('_status_callback', None))

    file_path = params.get('file_path')
    slide_index = params.get('slide_index')

    try:
        PPTXEditor._validate_file_path(file_path)

        if slide_index is None:
            return {'success': False, 'error': 'slide_index parameter is required'}

        prs = PPTXEditor._get_presentation(file_path)

        if slide_index < 0 or slide_index >= len(prs.slides):
            return {'success': False, 'error': f'Invalid slide_index: {slide_index}. Presentation has {len(prs.slides)} slides.'}

        # Get slide ID and remove
        slide_id = prs.slides._sldIdLst[slide_index]
        prs.part.drop_rel(slide_id.rId)
        prs.slides._sldIdLst.remove(slide_id)

        prs.save(file_path)

        logger.info(f"Deleted slide {slide_index} from: {file_path}")

        return {
            'success': True,
            'deleted_index': slide_index,
            'slide_count': len(prs.slides),
            'file_path': file_path
        }

    except PPTXEditorError as e:
        return {'success': False, 'error': str(e)}
    except Exception as e:
        logger.error(f"Failed to delete slide: {e}", exc_info=True)
        return {'success': False, 'error': f'Failed to delete slide: {str(e)}'}


def reorder_slides_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Reorder slides in a presentation.

    Args:
        params: Dictionary containing:
            - file_path (str, required): Path to the PPTX file
            - new_order (list, required): List of slide indices in new order
              e.g., [2, 0, 1] moves slide 2 to first, slide 0 to second, etc.

    Returns:
        Dictionary with:
            - success (bool): Whether operation succeeded
            - new_order (list): The applied order
            - slide_count (int): Total number of slides
            - error (str, optional): Error message if failed
    """
    status.set_callback(params.pop('_status_callback', None))

    file_path = params.get('file_path')
    new_order = params.get('new_order')

    try:
        PPTXEditor._validate_file_path(file_path)

        if not new_order:
            return {'success': False, 'error': 'new_order parameter is required'}

        if not isinstance(new_order, list):
            return {'success': False, 'error': 'new_order must be a list of indices'}

        prs = PPTXEditor._get_presentation(file_path)

        if len(new_order) != len(prs.slides):
            return {'success': False, 'error': f'new_order length ({len(new_order)}) must match slide count ({len(prs.slides)})'}

        if set(new_order) != set(range(len(prs.slides))):
            return {'success': False, 'error': 'new_order must contain each slide index exactly once'}

        # Reorder by manipulating the slide ID list
        sldIdLst = prs.slides._sldIdLst
        slide_ids = list(sldIdLst)

        # Clear and re-add in new order
        for slide_id in slide_ids:
            sldIdLst.remove(slide_id)

        for idx in new_order:
            sldIdLst.append(slide_ids[idx])

        prs.save(file_path)

        logger.info(f"Reordered slides in: {file_path}")

        return {
            'success': True,
            'new_order': new_order,
            'slide_count': len(prs.slides),
            'file_path': file_path
        }

    except PPTXEditorError as e:
        return {'success': False, 'error': str(e)}
    except Exception as e:
        logger.error(f"Failed to reorder slides: {e}", exc_info=True)
        return {'success': False, 'error': f'Failed to reorder slides: {str(e)}'}


def add_image_to_slide_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add an image to a slide.

    Args:
        params: Dictionary containing:
            - file_path (str, required): Path to the PPTX file
            - slide_index (int, required): Index of slide to add image to (0-based)
            - image_path (str, required): Path to the image file
            - position (dict, optional): Position with 'left', 'top', 'width', 'height' in inches
              Default: {'left': 1, 'top': 2, 'width': 6, 'height': 4}

    Returns:
        Dictionary with:
            - success (bool): Whether operation succeeded
            - slide_index (int): Index of slide with image
            - image_path (str): Path to added image
            - error (str, optional): Error message if failed
    """
    status.set_callback(params.pop('_status_callback', None))

    file_path = params.get('file_path')
    slide_index = params.get('slide_index')
    image_path = params.get('image_path')
    position = params.get('position', {})

    try:
        from pptx.util import Inches

        PPTXEditor._validate_file_path(file_path)

        if slide_index is None:
            return {'success': False, 'error': 'slide_index parameter is required'}

        if not image_path:
            return {'success': False, 'error': 'image_path parameter is required'}

        if not os.path.exists(image_path):
            return {'success': False, 'error': f'Image file not found: {image_path}'}

        prs = PPTXEditor._get_presentation(file_path)

        if slide_index < 0 or slide_index >= len(prs.slides):
            return {'success': False, 'error': f'Invalid slide_index: {slide_index}. Presentation has {len(prs.slides)} slides.'}

        slide = prs.slides[slide_index]

        # Position defaults
        left = Inches(position.get('left', 1))
        top = Inches(position.get('top', 2))
        width = Inches(position.get('width', 6)) if 'width' in position else None
        height = Inches(position.get('height', 4)) if 'height' in position else None

        # Add image
        if width and height:
            slide.shapes.add_picture(image_path, left, top, width, height)
        elif width:
            slide.shapes.add_picture(image_path, left, top, width=width)
        elif height:
            slide.shapes.add_picture(image_path, left, top, height=height)
        else:
            slide.shapes.add_picture(image_path, left, top)

        prs.save(file_path)

        logger.info(f"Added image to slide {slide_index} in: {file_path}")

        return {
            'success': True,
            'slide_index': slide_index,
            'image_path': image_path,
            'file_path': file_path
        }

    except PPTXEditorError as e:
        return {'success': False, 'error': str(e)}
    except Exception as e:
        logger.error(f"Failed to add image: {e}", exc_info=True)
        return {'success': False, 'error': f'Failed to add image: {str(e)}'}


def extract_text_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract all text from a presentation.

    Args:
        params: Dictionary containing:
            - file_path (str, required): Path to the PPTX file

    Returns:
        Dictionary with:
            - success (bool): Whether operation succeeded
            - text (str): All extracted text joined by newlines
            - slides_text (list): List of text per slide
            - total_characters (int): Total character count
            - error (str, optional): Error message if failed
    """
    status.set_callback(params.pop('_status_callback', None))

    file_path = params.get('file_path')

    try:
        PPTXEditor._validate_file_path(file_path)
        prs = PPTXEditor._get_presentation(file_path)

        slides_text = []
        all_text = []

        for idx, slide in enumerate(prs.slides):
            slide_texts = []
            for shape in slide.shapes:
                text = PPTXEditor._extract_shape_text(shape)
                if text:
                    slide_texts.append(text)

            slide_combined = '\n'.join(slide_texts)
            slides_text.append({
                'index': idx,
                'text': slide_combined
            })
            all_text.append(slide_combined)

        combined_text = '\n\n---\n\n'.join(all_text)

        logger.info(f"Extracted text from {len(prs.slides)} slides in: {file_path}")

        return {
            'success': True,
            'text': combined_text,
            'slides_text': slides_text,
            'slide_count': len(prs.slides),
            'total_characters': len(combined_text),
            'file_path': file_path
        }

    except PPTXEditorError as e:
        return {'success': False, 'error': str(e)}
    except Exception as e:
        logger.error(f"Failed to extract text: {e}", exc_info=True)
        return {'success': False, 'error': f'Failed to extract text: {str(e)}'}
