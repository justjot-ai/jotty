"""
OpenAI Image Generation Skill

Generate, edit, create variations, and analyze images.
Uses DALL-E for generation and litellm for VLM-powered analysis.
"""

import base64
import json as json_mod
import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional

import requests

from Jotty.core.infrastructure.utils.env_loader import get_env, load_jotty_env
from Jotty.core.infrastructure.utils.skill_status import SkillStatus
from Jotty.core.infrastructure.utils.tool_helpers import tool_error, tool_response, tool_wrapper

load_jotty_env()

logger = logging.getLogger(__name__)

# Status emitter for progress updates
status = SkillStatus("openai-image-gen")

OPENAI_API_BASE = "https://api.openai.com/v1"
DEFAULT_OUTPUT_DIR = os.path.expanduser("~/jotty/images")


class OpenAIImageClient:
    """Client for interacting with OpenAI's image generation APIs."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def generate_image(
        self,
        prompt: str,
        size: str = "1024x1024",
        quality: str = "standard",
        style: str = "vivid",
        model: str = "dall-e-3",
        n: int = 1,
    ) -> Dict[str, Any]:
        """Generate an image using DALL-E 3."""
        url = f"{OPENAI_API_BASE}/images/generations"
        payload = {
            "model": model,
            "prompt": prompt,
            "size": size,
            "quality": quality,
            "style": style,
            "n": n,
            "response_format": "b64_json",
        }

        response = requests.post(url, headers=self.headers, json=payload, timeout=120)
        response.raise_for_status()
        return response.json()

    def edit_image(
        self,
        image_path: str,
        prompt: str,
        mask_path: Optional[str] = None,
        size: str = "1024x1024",
        n: int = 1,
    ) -> Dict[str, Any]:
        """Edit an image using DALL-E 2."""
        url = f"{OPENAI_API_BASE}/images/edits"

        files = {
            "image": ("image.png", open(image_path, "rb"), "image/png"),
            "prompt": (None, prompt),
            "size": (None, size),
            "n": (None, str(n)),
            "response_format": (None, "b64_json"),
        }

        if mask_path:
            files["mask"] = ("mask.png", open(mask_path, "rb"), "image/png")

        headers = {"Authorization": f"Bearer {self.api_key}"}
        response = requests.post(url, headers=headers, files=files, timeout=120)
        response.raise_for_status()
        return response.json()

    def create_variation(
        self, image_path: str, size: str = "1024x1024", n: int = 1
    ) -> Dict[str, Any]:
        """Create variations of an image using DALL-E 2."""
        url = f"{OPENAI_API_BASE}/images/variations"

        files = {
            "image": ("image.png", open(image_path, "rb"), "image/png"),
            "size": (None, size),
            "n": (None, str(n)),
            "response_format": (None, "b64_json"),
        }

        headers = {"Authorization": f"Bearer {self.api_key}"}
        response = requests.post(url, headers=headers, files=files, timeout=120)
        response.raise_for_status()
        return response.json()


def _save_image_from_b64(b64_data: str, output_dir: str, prefix: str = "generated") -> str:
    """Save base64 encoded image data to file."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}.png"
    file_path = os.path.join(output_dir, filename)

    image_data = base64.b64decode(b64_data)
    with open(file_path, "wb") as f:
        f.write(image_data)

    return file_path


@tool_wrapper()
def generate_image_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate an image from a text prompt using DALL-E 3.

    Args:
        params: Dictionary containing:
            - prompt (str, required): Text description of the image to generate
            - size (str, optional): Image size - '1024x1024', '1792x1024', or '1024x1792' (default: '1024x1024')
            - quality (str, optional): Image quality - 'standard' or 'hd' (default: 'standard')
            - style (str, optional): Image style - 'vivid' or 'natural' (default: 'vivid')
            - output_path (str, optional): Output directory (default: ~/jotty/images)

    Returns:
        Dictionary with:
            - success (bool): Whether generation succeeded
            - image_path (str): Path to the generated image
            - revised_prompt (str): The revised prompt used by DALL-E 3
            - error (str, optional): Error message if failed
    """
    status.set_callback(params.pop("_status_callback", None))

    prompt = params.get("prompt")
    if not prompt:
        return {"success": False, "error": "Missing required parameter: prompt"}

    size = params.get("size", "1024x1024")
    valid_sizes = ["1024x1024", "1792x1024", "1024x1792"]
    if size not in valid_sizes:
        return {"success": False, "error": f"Invalid size '{size}'. Valid sizes: {valid_sizes}"}

    quality = params.get("quality", "standard")
    if quality not in ["standard", "hd"]:
        return {
            "success": False,
            "error": f"Invalid quality '{quality}'. Valid options: 'standard', 'hd'",
        }

    style = params.get("style", "vivid")
    if style not in ["vivid", "natural"]:
        return {
            "success": False,
            "error": f"Invalid style '{style}'. Valid options: 'vivid', 'natural'",
        }

    output_dir = params.get("output_path", DEFAULT_OUTPUT_DIR)

    try:
        client = OpenAIImageClient()

        logger.info(
            f"Generating image with DALL-E 3: prompt='{prompt[:100]}...', size={size}, quality={quality}, style={style}"
        )

        response = client.generate_image(
            prompt=prompt, size=size, quality=quality, style=style, model="dall-e-3"
        )

        image_data = response["data"][0]
        b64_image = image_data["b64_json"]
        revised_prompt = image_data.get("revised_prompt", prompt)

        file_path = _save_image_from_b64(b64_image, output_dir, prefix="dalle3")

        logger.info(f"Image saved to: {file_path}")

        return {
            "success": True,
            "image_path": file_path,
            "revised_prompt": revised_prompt,
            "size": size,
            "quality": quality,
            "style": style,
        }

    except requests.exceptions.HTTPError as e:
        error_msg = str(e)
        if e.response is not None:
            try:
                error_detail = e.response.json()
                error_msg = error_detail.get("error", {}).get("message", str(e))
            except Exception:
                pass
        logger.error(f"OpenAI API error: {error_msg}")
        return {"success": False, "error": f"OpenAI API error: {error_msg}"}

    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return {"success": False, "error": str(e)}

    except Exception as e:
        logger.error(f"Image generation failed: {e}", exc_info=True)
        return {"success": False, "error": f"Image generation failed: {str(e)}"}


@tool_wrapper()
def edit_image_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Edit an existing image with a text prompt using DALL-E 2.

    Note: The input image must be a square PNG image less than 4MB.
    If a mask is provided, the transparent areas indicate where the image should be edited.

    Args:
        params: Dictionary containing:
            - image_path (str, required): Path to the image to edit (PNG, square, <4MB)
            - prompt (str, required): Text description of the desired edit
            - mask_path (str, optional): Path to mask image (transparent areas will be edited)
            - size (str, optional): Output size - '256x256', '512x512', or '1024x1024' (default: '1024x1024')
            - n (int, optional): Number of images to generate (default: 1, max: 10)
            - output_path (str, optional): Output directory (default: ~/jotty/images)

    Returns:
        Dictionary with:
            - success (bool): Whether edit succeeded
            - image_paths (list): Paths to the edited images
            - error (str, optional): Error message if failed
    """
    status.set_callback(params.pop("_status_callback", None))

    image_path = params.get("image_path")
    if not image_path:
        return {"success": False, "error": "Missing required parameter: image_path"}

    if not os.path.exists(image_path):
        return {"success": False, "error": f"Image file not found: {image_path}"}

    prompt = params.get("prompt")
    if not prompt:
        return {"success": False, "error": "Missing required parameter: prompt"}

    mask_path = params.get("mask_path")
    if mask_path and not os.path.exists(mask_path):
        return {"success": False, "error": f"Mask file not found: {mask_path}"}

    size = params.get("size", "1024x1024")
    valid_sizes = ["256x256", "512x512", "1024x1024"]
    if size not in valid_sizes:
        return {
            "success": False,
            "error": f"Invalid size '{size}'. Valid sizes for edit: {valid_sizes}",
        }

    n = params.get("n", 1)
    if n < 1 or n > 10:
        return {"success": False, "error": "n must be between 1 and 10"}

    output_dir = params.get("output_path", DEFAULT_OUTPUT_DIR)

    try:
        client = OpenAIImageClient()

        logger.info(f"Editing image with DALL-E 2: image={image_path}, prompt='{prompt[:100]}...'")

        response = client.edit_image(
            image_path=image_path, prompt=prompt, mask_path=mask_path, size=size, n=n
        )

        image_paths = []
        for i, image_data in enumerate(response["data"]):
            b64_image = image_data["b64_json"]
            file_path = _save_image_from_b64(b64_image, output_dir, prefix=f"dalle2_edit_{i}")
            image_paths.append(file_path)

        logger.info(f"Edited {len(image_paths)} image(s) saved")

        return {
            "success": True,
            "image_paths": image_paths,
            "count": len(image_paths),
            "size": size,
        }

    except requests.exceptions.HTTPError as e:
        error_msg = str(e)
        if e.response is not None:
            try:
                error_detail = e.response.json()
                error_msg = error_detail.get("error", {}).get("message", str(e))
            except Exception:
                pass
        logger.error(f"OpenAI API error: {error_msg}")
        return {"success": False, "error": f"OpenAI API error: {error_msg}"}

    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return {"success": False, "error": str(e)}

    except Exception as e:
        logger.error(f"Image edit failed: {e}", exc_info=True)
        return {"success": False, "error": f"Image edit failed: {str(e)}"}


@tool_wrapper()
def create_variation_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create variations of an existing image using DALL-E 2.

    Note: The input image must be a square PNG image less than 4MB.

    Args:
        params: Dictionary containing:
            - image_path (str, required): Path to the source image (PNG, square, <4MB)
            - size (str, optional): Output size - '256x256', '512x512', or '1024x1024' (default: '1024x1024')
            - n (int, optional): Number of variations to generate (default: 1, max: 10)
            - output_path (str, optional): Output directory (default: ~/jotty/images)

    Returns:
        Dictionary with:
            - success (bool): Whether variation creation succeeded
            - image_paths (list): Paths to the variation images
            - error (str, optional): Error message if failed
    """
    status.set_callback(params.pop("_status_callback", None))

    image_path = params.get("image_path")
    if not image_path:
        return {"success": False, "error": "Missing required parameter: image_path"}

    if not os.path.exists(image_path):
        return {"success": False, "error": f"Image file not found: {image_path}"}

    size = params.get("size", "1024x1024")
    valid_sizes = ["256x256", "512x512", "1024x1024"]
    if size not in valid_sizes:
        return {
            "success": False,
            "error": f"Invalid size '{size}'. Valid sizes for variations: {valid_sizes}",
        }

    n = params.get("n", 1)
    if n < 1 or n > 10:
        return {"success": False, "error": "n must be between 1 and 10"}

    output_dir = params.get("output_path", DEFAULT_OUTPUT_DIR)

    try:
        client = OpenAIImageClient()

        logger.info(f"Creating {n} variation(s) of image: {image_path}")

        response = client.create_variation(image_path=image_path, size=size, n=n)

        image_paths = []
        for i, image_data in enumerate(response["data"]):
            b64_image = image_data["b64_json"]
            file_path = _save_image_from_b64(b64_image, output_dir, prefix=f"dalle2_variation_{i}")
            image_paths.append(file_path)

        logger.info(f"Created {len(image_paths)} variation(s)")

        return {
            "success": True,
            "image_paths": image_paths,
            "count": len(image_paths),
            "size": size,
            "source_image": image_path,
        }

    except requests.exceptions.HTTPError as e:
        error_msg = str(e)
        if e.response is not None:
            try:
                error_detail = e.response.json()
                error_msg = error_detail.get("error", {}).get("message", str(e))
            except Exception:
                pass
        logger.error(f"OpenAI API error: {error_msg}")
        return {"success": False, "error": f"OpenAI API error: {error_msg}"}

    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return {"success": False, "error": str(e)}

    except Exception as e:
        logger.error(f"Variation creation failed: {e}", exc_info=True)
        return {"success": False, "error": f"Variation creation failed: {str(e)}"}


# =========================================================================
# Image Analysis (litellm-powered VLM)
# =========================================================================


class ImageAnalyzer:
    """Image analysis using litellm for unified VLM access."""

    _instance = None

    def __init__(self):
        self._model = get_env("VLM_MODEL") or "claude-sonnet-4-5-20250929"
        self._api_key = get_env("LITELLM_API_KEY") or get_env("OPENAI_API_KEY") or ""
        self._api_base = get_env("LITELLM_BASE_URL") or ""

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @staticmethod
    def _encode(path: str) -> str:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    @staticmethod
    def _mime(path: str) -> str:
        ext = os.path.splitext(path)[1].lower()
        return {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }.get(ext, "image/png")

    def analyze(
        self, image_path: str, question: str, detail: str = "high", max_tokens: int = 1000
    ) -> Dict[str, Any]:
        """Analyze image with VLM via litellm."""
        try:
            import litellm
        except ImportError:
            return {"success": False, "error": "litellm not installed. pip install litellm"}

        if not os.path.exists(image_path):
            return {"success": False, "error": f"Image not found: {image_path}"}

        b64 = self._encode(image_path)
        mime = self._mime(image_path)

        kwargs = {
            "model": self._model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{b64}", "detail": detail},
                        },
                    ],
                }
            ],
            "max_tokens": max_tokens,
        }
        if self._api_key:
            kwargs["api_key"] = self._api_key
        if self._api_base:
            kwargs["api_base"] = self._api_base

        response = litellm.completion(**kwargs)
        return {
            "success": True,
            "analysis": response.choices[0].message.content,
            "model": self._model,
        }

    def describe(self, image_path: str) -> Dict[str, Any]:
        """Get detailed image description."""
        return self.analyze(
            image_path,
            "Describe this image in detail: subject, colors, composition, style, mood, "
            "any text visible, and overall impression.",
        )

    def extract_brand_theme(self, image_path: str) -> Dict[str, Any]:
        """Extract brand theme as structured JSON."""
        question = (
            "Analyze this image and extract the brand/design theme. "
            "Return ONLY JSON with keys: primary_color, secondary_color, accent_color, "
            "background_color, text_color, font_style, design_style, corner_style, notes. "
            "All colors as #HEXCODE."
        )
        result = self.analyze(image_path, question, detail="high")
        if not result.get("success"):
            return result

        analysis = result["analysis"]
        if "```json" in analysis:
            analysis = analysis.split("```json")[1].split("```")[0]
        elif "```" in analysis:
            analysis = analysis.split("```")[1].split("```")[0]

        try:
            return {"success": True, "theme": json_mod.loads(analysis.strip())}
        except json_mod.JSONDecodeError:
            return {"success": True, "theme": {"raw_analysis": result["analysis"]}}


@tool_wrapper(required_params=["image_path"])
def analyze_image_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze an image using VLM (Vision Language Model).

    Args:
        params: Dictionary containing:
            - image_path (str, required): Path to image file
            - question (str, optional): What to analyze (default: general description)
            - detail (str, optional): 'low' or 'high' (default: 'high')

    Returns:
        Dictionary with success, analysis, model
    """
    status.set_callback(params.pop("_status_callback", None))
    status.emit("Analyzing", "Analyzing image with VLM")
    return ImageAnalyzer.get_instance().analyze(
        params["image_path"],
        params.get("question", "Describe this image in detail."),
        params.get("detail", "high"),
    )


@tool_wrapper(required_params=["image_path"])
def describe_image_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get a detailed description of an image.

    Args:
        params: Dictionary containing:
            - image_path (str, required): Path to image file

    Returns:
        Dictionary with success, description, model
    """
    status.set_callback(params.pop("_status_callback", None))
    result = ImageAnalyzer.get_instance().describe(params["image_path"])
    if result.get("success") and "analysis" in result:
        result["description"] = result.pop("analysis")
    return result


@tool_wrapper(required_params=["image_path"])
def extract_brand_theme_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract brand theme (colors, fonts, style) from a reference image.

    Args:
        params: Dictionary containing:
            - image_path (str, required): Path to reference image

    Returns:
        Dictionary with success, theme dict
    """
    status.set_callback(params.pop("_status_callback", None))
    status.emit("Extracting", "Extracting brand theme")
    return ImageAnalyzer.get_instance().extract_brand_theme(params["image_path"])


@tool_wrapper(required_params=["prompt"])
def generate_brand_image_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate an image following brand guidelines.

    Args:
        params: Dictionary containing:
            - prompt (str, required): What to generate
            - brand_theme (dict, optional): Brand theme dict with colors/style
            - reference_image_path (str, optional): Extract theme from this image
            - size (str, optional): Image size (default: '1024x1024')
            - quality (str, optional): 'standard' or 'hd'

    Returns:
        Dictionary with success, image_path, theme_used, theme_source
    """
    status.set_callback(params.pop("_status_callback", None))

    brand_theme = params.get("brand_theme")
    ref_image = params.get("reference_image_path")

    default_theme = {
        "primary_color": "#0066CC",
        "secondary_color": "#1E3A5F",
        "background_color": "#F8FAFC",
        "design_style": "minimalist, clean",
        "font_style": "clean modern sans-serif",
        "corner_style": "rounded",
    }

    if brand_theme:
        theme, source = brand_theme, "provided"
    elif ref_image and os.path.exists(ref_image):
        ext_result = ImageAnalyzer.get_instance().extract_brand_theme(ref_image)
        if ext_result.get("success"):
            theme, source = ext_result["theme"], "extracted"
        else:
            theme, source = default_theme, "default"
    else:
        theme, source = default_theme, "default"

    enhanced_prompt = (
        f"{params['prompt']}\n\n"
        f"DESIGN GUIDELINES:\n"
        f"- Primary: {theme.get('primary_color', '#0066CC')}\n"
        f"- Secondary: {theme.get('secondary_color', '#1E3A5F')}\n"
        f"- Background: {theme.get('background_color', '#F8FAFC')}\n"
        f"- Style: {theme.get('design_style', 'minimalist')}\n"
        f"- Professional, high contrast, generous whitespace"
    )

    result = generate_image_tool(
        {
            "prompt": enhanced_prompt,
            "size": params.get("size", "1024x1024"),
            "quality": params.get("quality", "standard"),
        }
    )
    result["theme_used"] = theme
    result["theme_source"] = source
    return result


__all__ = [
    "generate_image_tool",
    "edit_image_tool",
    "create_variation_tool",
    "analyze_image_tool",
    "describe_image_tool",
    "extract_brand_theme_tool",
    "generate_brand_image_tool",
]
