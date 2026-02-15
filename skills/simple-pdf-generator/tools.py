"""
Simple PDF Generator Template

Common use case: Convert text/markdown content to PDF.

Workflow:
1. Write content to file (if needed)
2. Convert to PDF
3. Return PDF path

Customizable for:
- Different input formats (text, markdown, HTML)
- Output options (page size, margins, styling)
- File handling (temp files, cleanup)
"""

import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict

from Jotty.core.infrastructure.utils.skill_status import SkillStatus
from Jotty.core.infrastructure.utils.tool_helpers import (
    async_tool_wrapper,
    tool_error,
    tool_response,
)

# Status emitter for progress updates
status = SkillStatus("simple-pdf-generator")


logger = logging.getLogger(__name__)


@async_tool_wrapper()
async def generate_pdf_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate PDF from text/markdown content.

    Template for: Content → File → PDF workflow

    Args:
        params: Dictionary containing:
            - content (str, required): Text or markdown content
            - output_file (str, optional): Output PDF path (default: auto-generated)
            - input_format (str, optional): 'text' or 'markdown' (default: 'markdown')
            - page_size (str, optional): 'a4', 'letter', etc. (default: 'a4')
            - cleanup_temp (bool, optional): Delete temp files (default: True)

    Returns:
        Dictionary with PDF file path and metadata
    """
    status.set_callback(params.pop("_status_callback", None))

    try:
        try:
            from Jotty.core.capabilities.registry.skills_registry import get_skills_registry
        except ImportError:
            from Jotty.core.capabilities.registry.skills_registry import get_skills_registry

        content = params.get("content", "")
        if not content:
            return {"success": False, "error": "content parameter is required"}

        output_file = params.get("output_file")
        input_format = params.get("input_format", "markdown")
        page_size = params.get("page_size", "a4")
        cleanup_temp = params.get("cleanup_temp", True)

        import tempfile

        # Auto-generate output file if not provided
        if not output_file:
            output_file = tempfile.mktemp(suffix=".pdf", prefix="generated_")

        registry = get_skills_registry()
        registry.init()

        composer_skill = registry.get_skill("skill-composer")
        if not composer_skill:
            return {"success": False, "error": "skill-composer not available"}

        compose_tool = composer_skill.tools.get("compose_skills_tool")

        # Determine input file extension
        input_ext = ".md" if input_format == "markdown" else ".txt"
        temp_input_file = tempfile.mktemp(suffix=input_ext, prefix="temp_")

        # Build workflow using skill-composer
        workflow = {
            "workflow": [
                {
                    "type": "single",
                    "name": "write_file",
                    "skill": "file-operations",
                    "tool": "write_file_tool",
                    "params": {"path": temp_input_file, "content": content},
                },
                {
                    "type": "single",
                    "name": "convert_pdf",
                    "skill": "document-converter",
                    "tool": "convert_to_pdf_tool",
                    "params": {
                        "input_file": "${write_file.path}",
                        "output_file": output_file,
                        "page_size": page_size,
                    },
                },
            ]
        }

        # Execute workflow
        result = await compose_tool(workflow)

        if not result.get("success"):
            # Cleanup on failure
            if cleanup_temp and os.path.exists(temp_input_file):
                try:
                    os.remove(temp_input_file)
                except OSError:
                    # File removal failed, ignore
                    pass
            return {"success": False, "error": f'PDF generation failed: {result.get("error")}'}

        # Cleanup temp file if requested
        if cleanup_temp and os.path.exists(temp_input_file):
            try:
                os.remove(temp_input_file)
            except OSError:
                # File removal failed, ignore
                pass

        return {
            "success": True,
            "pdf_path": output_file,
            "file_size": os.path.getsize(output_file) if os.path.exists(output_file) else 0,
            "input_format": input_format,
            "page_size": page_size,
        }

    except Exception as e:
        logger.error(f"PDF generation failed: {e}", exc_info=True)
        return {"success": False, "error": str(e)}
