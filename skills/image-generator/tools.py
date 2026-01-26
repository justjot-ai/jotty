import os
import tempfile
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

def generate_image_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate images using open-source diffusion models via Hugging Face.
    
    Args:
        params: Dictionary containing:
            - prompt (str): Text description of the image to generate
            - model (str, optional): Model to use. Options:
                - "sdxl" (default): stabilityai/stable-diffusion-xl-base-1.0
                - "flux": black-forest-labs/FLUX.1-dev
                - "sd1.5": runwayml/stable-diffusion-v1-5
            - negative_prompt (str, optional): What to avoid in the image
            - num_inference_steps (int, optional): Number of denoising steps (default: 50)
            - guidance_scale (float, optional): How closely to follow prompt (default: 7.5)
            - width (int, optional): Image width in pixels (default: 1024 for SDXL, 512 for SD1.5)
            - height (int, optional): Image height in pixels (default: 1024 for SDXL, 512 for SD1.5)
            - seed (int, optional): Random seed for reproducibility
            - output_path (str, optional): Path to save image (default: temp file)
    
    Returns:
        Dictionary with:
            - success (bool): Whether generation succeeded
            - image_path (str): Path to generated image
            - error (str, optional): Error message if failed
            - model_used (str): Model identifier used
    """
    try:
        import torch
        from diffusers import (
            StableDiffusionPipeline,
            StableDiffusionXLPipeline,
            FluxPipeline,
            DPMSolverMultistepScheduler
        )
        from PIL import Image
    except ImportError as e:
        return {
            "success": False,
            "error": f"Required libraries not installed: {str(e)}. Install with: pip install torch diffusers transformers accelerate pillow"
        }
    
    prompt = params.get("prompt")
    if not prompt:
        return {
            "success": False,
            "error": "Missing required parameter: prompt"
        }
    
    model_key = params.get("model", "sdxl").lower()
    model_map = {
        "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
        "flux": "black-forest-labs/FLUX.1-dev",
        "sd1.5": "runwayml/stable-diffusion-v1-5"
    }
    
    if model_key not in model_map:
        return {
            "success": False,
            "error": f"Invalid model '{model_key}'. Choose from: {list(model_map.keys())}"
        }
    
    model_id = model_map[model_key]
    negative_prompt = params.get("negative_prompt", "")
    num_inference_steps = params.get("num_inference_steps", 50)
    guidance_scale = params.get("guidance_scale", 7.5)
    seed = params.get("seed")
    output_path = params.get("output_path")
    
    default_size = 1024 if model_key == "sdxl" else 512
    width = params.get("width", default_size)
    height = params.get("height", default_size)
    
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if device == "cuda" else torch.float32
        
        logger.info(f"Loading model: {model_id} on {device}")
        
        if model_key == "sdxl":
            pipe = StableDiffusionXLPipeline.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                use_safetensors=True,
                variant="fp16" if device == "cuda" else None
            )
        elif model_key == "flux":
            pipe = FluxPipeline.from_pretrained(
                model_id,
                torch_dtype=torch_dtype
            )
        else:
            pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                safety_checker=None
            )
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        
        pipe = pipe.to(device)
        
        if hasattr(pipe, 'enable_attention_slicing'):
            pipe.enable_attention_slicing()
        
        generator = None
        if seed is not None:
            generator = torch.Generator(device=device).manual_seed(seed)
        
        logger.info(f"Generating image with prompt: {prompt[:100]}...")
        
        generation_params = {
            "prompt": prompt,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "generator": generator,
            "width": width,
            "height": height
        }
        
        if negative_prompt and model_key != "flux":
            generation_params["negative_prompt"] = negative_prompt
        
        result = pipe(**generation_params)
        image = result.images[0]
        
        if output_path is None:
            temp_dir = tempfile.gettempdir()
            output_path = os.path.join(temp_dir, f"generated_image_{os.getpid()}.png")
        
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        image.save(output_path)
        
        logger.info(f"Image saved to: {output_path}")
        
        del pipe
        if device == "cuda":
            torch.cuda.empty_cache()
        
        return {
            "success": True,
            "image_path": output_path,
            "model_used": model_id,
            "width": width,
            "height": height,
            "seed": seed
        }
        
    except Exception as e:
        logger.error(f"Error generating image: {str(e)}", exc_info=True)
        return {
            "success": False,
            "error": f"Image generation failed: {str(e)}",
            "model_used": model_id
        }


def list_available_models_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    List available image generation models and their details.
    
    Args:
        params: Dictionary (can be empty)
    
    Returns:
        Dictionary with:
            - success (bool): Always True
            - models (list): List of available models with details
    """
    models = [
        {
            "key": "sdxl",
            "id": "stabilityai/stable-diffusion-xl-base-1.0",
            "name": "Stable Diffusion XL",
            "default_size": 1024,
            "description": "High quality image generation, best overall quality"
        },
        {
            "key": "flux",
            "id": "black-forest-labs/FLUX.1-dev",
            "name": "FLUX.1 Dev",
            "default_size": 1024,
            "description": "Latest generation model, excellent prompt following"
        },
        {
            "key": "sd1.5",
            "id": "runwayml/stable-diffusion-v1-5",
            "name": "Stable Diffusion 1.5",
            "default_size": 512,
            "description": "Faster generation, lower resource requirements"
        }
    ]
    
    return {
        "success": True,
        "models": models
    }


def validate_image_params_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate image generation parameters before generation.
    
    Args:
        params: Dictionary containing proposed generation parameters
    
    Returns:
        Dictionary with:
            - success (bool): Whether parameters are valid
            - validated_params (dict): Corrected/validated parameters
            - warnings (list): List of warnings or suggestions
            - error (str, optional): Error message if validation failed
    """
    warnings = []
    validated = params.copy()
    
    if "prompt" not in params or not params["prompt"]:
        return {
            "success": False,
            "error": "prompt is required"
        }
    
    if len(params["prompt"]) > 1000:
        warnings.append("Prompt is very long, consider shortening for better results")
    
    model_key = params.get("model", "sdxl").lower()
    if model_key not in ["sdxl", "flux", "sd1.5"]:
        warnings.append(f"Invalid model '{model_key}', defaulting to 'sdxl'")
        validated["model"] = "sdxl"
        model_key = "sdxl"
    
    default_size = 1024 if model_key == "sdxl" else 512
    width = params.get("width", default_size)
    height = params.get("height", default_size)
    
    if width % 8 != 0:
        width = (width // 8) * 8
        validated["width"] = width
        warnings.append(f"Width adjusted to {width} (must be divisible by 8)")
    
    if height % 8 != 0:
        height = (height // 8) * 8
        validated["height"] = height
        warnings.append(f"Height adjusted to {height} (must be divisible by 8)")
    
    if width > 2048 or height > 2048:
        warnings.append("Very large image size may cause memory issues")
    
    if width < 256 or height < 256:
        warnings.append("Very small image size may produce poor results")
    
    num_steps = params.get("num_inference_steps", 50)
    if num_steps < 20:
        warnings.append("Low number of inference steps may reduce quality")
    elif num_steps > 100:
        warnings.append("High number of inference steps increases generation time significantly")
    
    guidance_scale = params.get("guidance_scale", 7.5)
    if guidance_scale < 1.0:
        warnings.append("Very low guidance scale may ignore prompt")
    elif guidance_scale > 20.0:
        warnings.append("Very high guidance scale may cause artifacts")
    
    return {
        "success": True,
        "validated_params": validated,
        "warnings": warnings
    }