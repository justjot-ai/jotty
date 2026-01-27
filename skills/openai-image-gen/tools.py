"""
OpenAI Image Generation Skill

Generate, edit, and create variations of images using OpenAI's DALL-E API.
"""
import os
import logging
import base64
import requests
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

# Constants
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
            "Content-Type": "application/json"
        }

    def generate_image(
        self,
        prompt: str,
        size: str = "1024x1024",
        quality: str = "standard",
        style: str = "vivid",
        model: str = "dall-e-3",
        n: int = 1
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
            "response_format": "b64_json"
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
        n: int = 1
    ) -> Dict[str, Any]:
        """Edit an image using DALL-E 2."""
        url = f"{OPENAI_API_BASE}/images/edits"

        files = {
            "image": ("image.png", open(image_path, "rb"), "image/png"),
            "prompt": (None, prompt),
            "size": (None, size),
            "n": (None, str(n)),
            "response_format": (None, "b64_json")
        }

        if mask_path:
            files["mask"] = ("mask.png", open(mask_path, "rb"), "image/png")

        headers = {"Authorization": f"Bearer {self.api_key}"}
        response = requests.post(url, headers=headers, files=files, timeout=120)
        response.raise_for_status()
        return response.json()

    def create_variation(
        self,
        image_path: str,
        size: str = "1024x1024",
        n: int = 1
    ) -> Dict[str, Any]:
        """Create variations of an image using DALL-E 2."""
        url = f"{OPENAI_API_BASE}/images/variations"

        files = {
            "image": ("image.png", open(image_path, "rb"), "image/png"),
            "size": (None, size),
            "n": (None, str(n)),
            "response_format": (None, "b64_json")
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
    prompt = params.get("prompt")
    if not prompt:
        return {"success": False, "error": "Missing required parameter: prompt"}

    size = params.get("size", "1024x1024")
    valid_sizes = ["1024x1024", "1792x1024", "1024x1792"]
    if size not in valid_sizes:
        return {"success": False, "error": f"Invalid size '{size}'. Valid sizes: {valid_sizes}"}

    quality = params.get("quality", "standard")
    if quality not in ["standard", "hd"]:
        return {"success": False, "error": f"Invalid quality '{quality}'. Valid options: 'standard', 'hd'"}

    style = params.get("style", "vivid")
    if style not in ["vivid", "natural"]:
        return {"success": False, "error": f"Invalid style '{style}'. Valid options: 'vivid', 'natural'"}

    output_dir = params.get("output_path", DEFAULT_OUTPUT_DIR)

    try:
        client = OpenAIImageClient()

        logger.info(f"Generating image with DALL-E 3: prompt='{prompt[:100]}...', size={size}, quality={quality}, style={style}")

        response = client.generate_image(
            prompt=prompt,
            size=size,
            quality=quality,
            style=style,
            model="dall-e-3"
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
            "style": style
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
        return {"success": False, "error": f"Invalid size '{size}'. Valid sizes for edit: {valid_sizes}"}

    n = params.get("n", 1)
    if n < 1 or n > 10:
        return {"success": False, "error": "n must be between 1 and 10"}

    output_dir = params.get("output_path", DEFAULT_OUTPUT_DIR)

    try:
        client = OpenAIImageClient()

        logger.info(f"Editing image with DALL-E 2: image={image_path}, prompt='{prompt[:100]}...'")

        response = client.edit_image(
            image_path=image_path,
            prompt=prompt,
            mask_path=mask_path,
            size=size,
            n=n
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
            "size": size
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
    image_path = params.get("image_path")
    if not image_path:
        return {"success": False, "error": "Missing required parameter: image_path"}

    if not os.path.exists(image_path):
        return {"success": False, "error": f"Image file not found: {image_path}"}

    size = params.get("size", "1024x1024")
    valid_sizes = ["256x256", "512x512", "1024x1024"]
    if size not in valid_sizes:
        return {"success": False, "error": f"Invalid size '{size}'. Valid sizes for variations: {valid_sizes}"}

    n = params.get("n", 1)
    if n < 1 or n > 10:
        return {"success": False, "error": "n must be between 1 and 10"}

    output_dir = params.get("output_path", DEFAULT_OUTPUT_DIR)

    try:
        client = OpenAIImageClient()

        logger.info(f"Creating {n} variation(s) of image: {image_path}")

        response = client.create_variation(
            image_path=image_path,
            size=size,
            n=n
        )

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
            "source_image": image_path
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
