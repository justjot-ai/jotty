"""image-generator Skill Tools — Generate images using open-source models."""
from typing import Dict, Any

from Jotty.core.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.utils.skill_status import SkillStatus

status = SkillStatus("image-generator")


@tool_wrapper()
def image_generator_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate images using open-source models like Stable Diffusion, Flux, or SDXL. No API key required."""
    status.set_callback(params.pop("_status_callback", None))
    return tool_response(message="Tool stub - implement me")
