import os
from typing import Dict, Any
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, DiffusionPipeline
from PIL import Image
import io
import base64

def generate_image_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate images using open-source diffusion models.
    
    Args:
        params: Dictionary containing:
            - prompt (str, required): Text description of the image to generate
            - model (str, optional): Model to use. Options:
                - "sdxl" or "stabilityai/stable-diffusion-xl-base-1.0" (default)
                - "flux" or "black-forest-labs/FLUX.1-dev"
                - "sd15" or "runwayml/stable-diffusion-v1-5"
            - negative_prompt (str, optional): What to avoid in the image
            - num_inference_steps (int, optional): Number of denoising steps (default: 50)
            - guidance_scale (float, optional): How closely to follow prompt (default: 7.5)
            - width (int, optional): Image width (default: 512 or 1024 for SDXL)
            - height (int, optional): Image height (default: 512 or 1024 for SDXL)
            - seed (int, optional): Random seed for reproducibility
            - output_path (str, optional): Path to save the image
            - return_base64 (bool, optional): Return image as base64 string (default: False)
    
    Returns:
        Dictionary with:
            - success (bool): Whether generation succeeded
            - message (str): Status message
            - output_path (str, optional): Path where image was saved
            - base64_image (str, optional): Base64 encoded image if requested
    """
    try:
        prompt = params.get("prompt")
        if not prompt:
            return {
                "success": False,
                "error": "Missing required parameter: prompt"
            }
        
        model_name = params.get("model", "sdxl")
        model_map = {
            "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
            "flux": "black-forest-labs/FLUX.1-dev",
            "sd15": "runwayml/stable-diffusion-v1-5",
            "stabilityai/stable-diffusion-xl-base-1.0": "stabilityai/stable-diffusion-xl-base-1.0",
            "black-forest-labs/FLUX.1-dev": "black-forest-labs/FLUX.1-dev",
            "runwayml/stable-diffusion-v1-5": "runwayml/stable-diffusion-v1-5"
        }
        
        model_id = model_map.get(model_name, "stabilityai/stable-diffusion-xl-base-1.0")
        
        negative_prompt = params.get("negative_prompt", None)
        num_inference_steps = params.get("num_inference_steps", 50)
        guidance_scale = params.get("guidance_scale", 7.5)
        seed = params.get("seed", None)
        output_path = params.get("output_path", "generated_image.png")
        return_base64 = params.get("return_base64", False)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if device == "cuda" else torch.float32
        
        if "xl" in model_id.lower() or "sdxl" in model_name.lower():
            default_size = 1024
            pipeline = StableDiffusionXLPipeline.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                use_safetensors=True
            )
        elif "flux" in model_id.lower():
            default_size = 1024
            pipeline = DiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch_dtype
            )
        else:
            default_size = 512
            pipeline = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                use_safetensors=True
            )
        
        pipeline = pipeline.to(device)
        
        width = params.get("width", default_size)
        height = params.get("height", default_size)
        
        generator = None
        if seed is not None:
            generator = torch.Generator(device=device).manual_seed(seed)
        
        generation_kwargs = {
            "prompt": prompt,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "width": width,
            "height": height,
            "generator": generator
        }
        
        if negative_prompt:
            generation_kwargs["negative_prompt"] = negative_prompt
        
        result = pipeline(**generation_kwargs)
        image = result.images[0]
        
        response = {
            "success": True,
            "message": f"Image generated successfully using {model_id}"
        }
        
        if output_path:
            image.save(output_path)
            response["output_path"] = output_path
        
        if return_base64:
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            response["base64_image"] = img_str
        
        return response
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

def list_available_models_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    List available image generation models.
    
    Args:
        params: Empty dictionary or any parameters (not used)
    
    Returns:
        Dictionary with:
            - success (bool): Always True
            - models (list): List of available models with descriptions
    """
    try:
        models = [
            {
                "name": "sdxl",
                "full_name": "stabilityai/stable-diffusion-xl-base-1.0",
                "description": "Stable Diffusion XL - High quality 1024x1024 images",
                "default_size": "1024x1024"
            },
            {
                "name": "flux",
                "full_name": "black-forest-labs/FLUX.1-dev",
                "description": "FLUX.1 - Advanced diffusion model with high detail",
                "default_size": "1024x1024"
            },
            {
                "name": "sd15",
                "full_name": "runwayml/stable-diffusion-v1-5",
                "description": "Stable Diffusion 1.5 - Fast and reliable 512x512 images",
                "default_size": "512x512"
            }
        ]
        
        return {
            "success": True,
            "models": models
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }