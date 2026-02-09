import os
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, FluxPipeline
from PIL import Image
import io
import base64
from typing import Dict, Any, Optional
import gc

def _get_device():
    """Determine the best available device for inference."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def _cleanup_memory():
    """Clean up GPU/memory after generation."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def _image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def generate_image_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate images using open-source diffusion models.
    
    Expected params:
        - prompt (str): Text description of the image to generate
        - model (str, optional): Model ID to use. Options:
            - "stabilityai/stable-diffusion-xl-base-1.0" (default)
            - "black-forest-labs/FLUX.1-dev"
            - "runwayml/stable-diffusion-v1-5"
        - negative_prompt (str, optional): Text describing what to avoid
        - num_inference_steps (int, optional): Number of denoising steps (default: 50)
        - guidance_scale (float, optional): Guidance scale for classifier-free guidance (default: 7.5)
        - width (int, optional): Image width (default: 1024 for SDXL, 512 for SD1.5)
        - height (int, optional): Image height (default: 1024 for SDXL, 512 for SD1.5)
        - seed (int, optional): Random seed for reproducibility
        - output_path (str, optional): Path to save the image (default: generated_image.png)
        - return_base64 (bool, optional): Return image as base64 string (default: False)
    
    Returns:
        Dict with success status, message, output_path, and optionally base64_image
    """
    try:
        prompt = params.get("prompt")
        if not prompt:
            return {"success": False, "error": "Missing required parameter: prompt"}
        
        model_id = params.get("model", "stabilityai/stable-diffusion-xl-base-1.0")
        negative_prompt = params.get("negative_prompt", None)
        num_inference_steps = params.get("num_inference_steps", 50)
        guidance_scale = params.get("guidance_scale", 7.5)
        seed = params.get("seed", None)
        output_path = params.get("output_path", "generated_image.png")
        return_base64 = params.get("return_base64", False)
        
        device = _get_device()
        
        # Determine default dimensions based on model
        if "xl" in model_id.lower() or "sdxl" in model_id.lower():
            default_size = 1024
        else:
            default_size = 512
            
        width = params.get("width", default_size)
        height = params.get("height", default_size)
        
        # Load appropriate pipeline based on model
        if "flux" in model_id.lower():
            pipeline = FluxPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            )
        elif "xl" in model_id.lower() or "sdxl" in model_id.lower():
            pipeline = StableDiffusionXLPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if device != "cpu" else torch.float32,
                use_safetensors=True,
            )
        else:
            pipeline = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if device != "cpu" else torch.float32,
                safety_checker=None,
            )
        
        pipeline = pipeline.to(device)
        
        # Set up generator for reproducibility
        generator = None
        if seed is not None:
            generator = torch.Generator(device=device).manual_seed(seed)
        
        # Generate image
        generation_params = {
            "prompt": prompt,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "width": width,
            "height": height,
            "generator": generator,
        }
        
        if negative_prompt:
            generation_params["negative_prompt"] = negative_prompt
        
        result = pipeline(**generation_params)
        image = result.images[0]
        
        # Save image
        image.save(output_path)
        
        response = {
            "success": True,
            "message": f"Image generated successfully using {model_id}",
            "output_path": output_path,
            "model": model_id,
            "prompt": prompt,
            "dimensions": f"{width}x{height}",
        }
        
        if return_base64:
            response["base64_image"] = _image_to_base64(image)
        
        # Clean up
        del pipeline
        _cleanup_memory()
        
        return response
        
    except Exception as e:
        _cleanup_memory()
        return {"success": False, "error": f"Image generation failed: {str(e)}"}

def list_available_models_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    List available image generation models.
    
    Expected params:
        None required
    
    Returns:
        Dict with success status and list of available models
    """
    models = [
        {
            "id": "stabilityai/stable-diffusion-xl-base-1.0",
            "name": "Stable Diffusion XL",
            "description": "High-quality 1024x1024 image generation",
            "default_size": "1024x1024"
        },
        {
            "id": "black-forest-labs/FLUX.1-dev",
            "name": "FLUX.1 Dev",
            "description": "Advanced diffusion model with high quality output",
            "default_size": "1024x1024"
        },
        {
            "id": "runwayml/stable-diffusion-v1-5",
            "name": "Stable Diffusion v1.5",
            "description": "Fast and efficient 512x512 image generation",
            "default_size": "512x512"
        }
    ]
    
    return {
        "success": True,
        "models": models,
        "message": f"Found {len(models)} available models"
    }

def batch_generate_images_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate multiple images from a list of prompts.
    
    Expected params:
        - prompts (list): List of text prompts
        - model (str, optional): Model ID to use
        - output_dir (str, optional): Directory to save images (default: generated_images)
        - base_filename (str, optional): Base filename pattern (default: image)
        - Other parameters same as generate_image_tool
    
    Returns:
        Dict with success status and list of generated image paths
    """
    try:
        prompts = params.get("prompts")
        if not prompts or not isinstance(prompts, list):
            return {"success": False, "error": "Missing or invalid parameter: prompts (must be a list)"}
        
        output_dir = params.get("output_dir", "generated_images")
        base_filename = params.get("base_filename", "image")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        generated_images = []
        errors = []
        
        for idx, prompt in enumerate(prompts):
            output_path = os.path.join(output_dir, f"{base_filename}_{idx + 1}.png")
            
            # Create params for single image generation
            single_params = params.copy()
            single_params["prompt"] = prompt
            single_params["output_path"] = output_path
            single_params.pop("prompts", None)
            single_params.pop("output_dir", None)
            single_params.pop("base_filename", None)
            
            result = generate_image_tool(single_params)
            
            if result.get("success"):
                generated_images.append({
                    "index": idx + 1,
                    "prompt": prompt,
                    "path": output_path
                })
            else:
                errors.append({
                    "index": idx + 1,
                    "prompt": prompt,
                    "error": result.get("error")
                })
        
        return {
            "success": len(generated_images) > 0,
            "message": f"Generated {len(generated_images)} out of {len(prompts)} images",
            "generated_images": generated_images,
            "errors": errors if errors else None,
            "output_dir": output_dir
        }
        
    except Exception as e:
        return {"success": False, "error": f"Batch generation failed: {str(e)}"}