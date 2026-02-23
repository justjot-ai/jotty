"""
image-generator Skill Tools
"""

from Jotty.core.infrastructure.utils.tool_helpers import tool_wrapper


@tool_wrapper()
def image_generator_tool(params: dict) -> dict:
    """Generate images using open-source models like Stable Diffusion, Flux, or SDXL. No API key required."""
    return {"success": True, "message": "Tool stub - implement me"}
