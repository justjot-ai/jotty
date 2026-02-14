import os
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, DiffusionPipeline
from PIL import Image
import io
import base64

def generate_image_tool(params: dict) -> dict:
    """
    Generate images using open-source models like Stable Diffusion, Flux, or SDXL.
    
    Args:
        params: dict with keys:
            - prompt: str (required) - Text description of the image
            - model: str (optional) - Model to use, defaults to "stabilityai/stable-diffusion-xl-base-1.0"
            - negative_prompt: str (optional) - What to avoid in the image
            - num_inference_steps: int (optional) - Number of denoising steps, defaults to 50
            - guidance_scale: float (optional) - How closely to follow the prompt, defaults to 7.5
            - width: int (optional) - Image width, defaults to 1024
            - height: int (optional) - Image height, defaults to 1024
            - output_path: str (optional) - Path to save the image, if not provided returns base64
            - seed: int (optional) - Random seed for reproducibility
    
    Returns:
        dict with success status, message, and image data
    """
    try:
        prompt = params.get("prompt")
        if not prompt:
            return {"success": False, "error": "prompt is required"}
        
        model_id = params.get("model", "stabilityai/stable-diffusion-xl-base-1.0")
        negative_prompt = params.get("negative_prompt", "")
        num_inference_steps = params.get("num_inference_steps", 50)
        guidance_scale = params.get("guidance_scale", 7.5)
        width = params.get("width", 1024)
        height = params.get("height", 1024)
        output_path = params.get("output_path")
        seed = params.get("seed")
        
        device= "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        
        # Load appropriate pipeline based on model
        if "xl" in model_id.lower() or "sdxl" in model_id.lower():
            pipe = StableDiffusionXLPipeline.from_pretrained(
                model_id,
                torch_dtype=dtype,
                use_safetensors=True,
                variant="fp16" if device == "cuda" else None
            )
        elif "flux" in model_id.lower():
            pipe = DiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=dtype
            )
        else:
            pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=dtype,
                use_safetensors=True
            )
        
        pipe = pipe.to(device)
        
        # Set random seed if provided
        if seed is not None:
            generator = torch.Generator(device=device).manual_seed(seed)
        else:
            generator = None
        
        # Generate image
        generation_params = {
            "prompt": prompt,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "generator": generator
        }
        
        if negative_prompt:
            generation_params["negative_prompt"] = negative_prompt
        
        # Add dimensions for non-FLUX models
        if "flux" not in model_id.lower():
            generation_params["width"] = width
            generation_params["height"] = height
        
        image = pipe(**generation_params).images[0]
        
        # Save or encode image
        if output_path:
            image.save(output_path)
            return {
                "success": True,
                "message": f"Image generated and saved to {output_path}",
"path": output_path
            }
        else:
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            return {
                "success": True,
                "message": "Image generated successfully",
                "image_base64": img_str
            }
    
    except Exception as e:
        return {"success": False, "error": str(e)}

def list_available_models_tool(params: dict) -> dict:
    """
    List available open-source image generation models.
    
    Args:
        params: dict (no required parameters)
    
    Returns:
        dict with list of available models
    """
    models = [
        {
            "id": "stabilityai/stable-diffusion-xl-base-1.0",
            "name": "Stable Diffusion XL",
            "description": "High-quality text-to-image model with 1024x1024 resolution"
        },
        {
            "id": "runwayml/stable-diffusion-v1-5",
            "name": "Stable Diffusion v1.5",
            "description": "Classic Stable Diffusion model, fast and reliable"
        },
        {
            "id": "stabilityai/stable-diffusion-2-1",
            "name": "Stable Diffusion 2.1",
            "description": "Improved version with better image quality"
        },
        {
            "id": "black-forest-labs/FLUX.1-dev",
            "name": "FLUX.1 Dev",
            "description": "Advanced diffusion model with high quality outputs"
        },
        {
            "id": "black-forest-labs/FLUX.1-schnell",
            "name": "FLUX.1 Schnell",
            "description": "Fast version of FLUX.1 for quick generation"
        }
    ]
    
    return {
        "success": True,
        "models": models
    }