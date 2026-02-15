"""
image-generator Skill Tools — Generate images using open-source models.
"""
from typing import Any, Dict

from core.utils.skill_helpers import tool_wrapper, tool_response, tool_error


@tool_wrapper(required_params=["prompt"])
def image_generator_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate images using open-source models like Stable Diffusion, Flux, or SDXL.

    Requires a local or remote ComfyUI / Automatic1111 / Fooocus endpoint.
    Set IMAGE_GEN_URL env var to point to your image generation API.
    """
    import os
    import requests

    prompt = params["prompt"]
    negative_prompt = params.get("negative_prompt", "")
    width = params.get("width", 1024)
    height = params.get("height", 1024)
    steps = params.get("steps", 20)
    model = params.get("model", "stable-diffusion-xl")

    base_url = os.environ.get("IMAGE_GEN_URL", "")
    if not base_url:
        return tool_error("IMAGE_GEN_URL environment variable not set. "
                         "Point it to a ComfyUI, Automatic1111, or Fooocus API endpoint.")

    try:
        response = requests.post(
            f"{base_url.rstrip('/')}/sdapi/v1/txt2img",
            json={
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "width": width,
                "height": height,
                "steps": steps,
            },
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
        images = data.get("images", [])
        return tool_response(
            images_count=len(images),
            prompt=prompt,
            model=model,
            dimensions=f"{width}x{height}",
            message=f"Generated {len(images)} image(s)",
        )
    except requests.RequestException as e:
        return tool_error(f"Image generation failed: {e}")


__all__ = ["image_generator_tool"]
