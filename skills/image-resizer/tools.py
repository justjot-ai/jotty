"""Image Resizer Skill - resize images using Pillow."""
from pathlib import Path
from typing import Dict, Any
from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.infrastructure.utils.skill_status import SkillStatus

status = SkillStatus("image-resizer")


@tool_wrapper(required_params=["input_path"])
def resize_image_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Resize an image to specified dimensions."""
    status.set_callback(params.pop("_status_callback", None))

    try:
        from PIL import Image
    except ImportError:
        return tool_error("Pillow is required. Install with: pip install Pillow")

    input_path = Path(params["input_path"])
    if not input_path.exists():
        return tool_error(f"File not found: {input_path}")

    width = params.get("width")
    height = params.get("height")
    maintain_aspect = params.get("maintain_aspect", True)
    quality = min(max(int(params.get("quality", 85)), 1), 100)
    out_format = params.get("format")

    if not width and not height:
        return tool_error("Specify at least width or height")

    try:
        img = Image.open(input_path)
        orig_w, orig_h = img.size

        if maintain_aspect:
            if width and height:
                ratio = min(width / orig_w, height / orig_h)
            elif width:
                ratio = width / orig_w
            else:
                ratio = height / orig_h
            new_w = int(orig_w * ratio)
            new_h = int(orig_h * ratio)
        else:
            new_w = int(width) if width else orig_w
            new_h = int(height) if height else orig_h

        resized = img.resize((new_w, new_h), Image.LANCZOS)

        # Determine output
        suffix = out_format.lower() if out_format else input_path.suffix.lstrip(".")
        if suffix == "jpg":
            suffix = "jpeg"
        out_path_str = params.get("output_path")
        if out_path_str:
            out_path = Path(out_path_str)
        else:
            out_path = input_path.with_name(f"{input_path.stem}_resized.{suffix}")

        save_kwargs = {}
        if suffix in ("jpeg", "jpg", "webp"):
            save_kwargs["quality"] = quality
        if suffix == "jpeg" and resized.mode in ("RGBA", "P"):
            resized = resized.convert("RGB")

        resized.save(str(out_path), **save_kwargs)

        return tool_response(
            output_path=str(out_path),
            original_size={"width": orig_w, "height": orig_h},
            new_size={"width": new_w, "height": new_h},
            format=suffix,
            quality=quality,
        )
    except Exception as e:
        return tool_error(f"Image processing failed: {e}")


__all__ = ["resize_image_tool"]
