import os
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, FluxPipeline
from PIL import Image
import io
import base64
from pathlib import Path

def generate_image_tool(params: dict) -> dict:
    """
    Generate images using open-source models like Stable Diffusion, Flux, or SDXL.
    
    Args:
        params: dict with keys:
            - prompt: str (required) - text description of image to generate
            - model: str (optional) - model to use, default: "stabilityai/stable-diffusion-xl-base-1.0"
            - negative_prompt: str (optional) - what to avoid in image
            - num_inference_steps: int (optional) - number of denoising steps, default: 50
            - guidance_scale: float (optional) - how closely to follow prompt, default: 7.5
            - width: int (optional) - image width, default: 1024
            - height: int (optional) - image height, default: 1024
            - seed: int (optional) - random seed for reproducibility
            - output_path: str (optional) - where to save image, default: "generated_image.png"
            - return_base64: bool (optional) - return base64 encoded image, default: False
    
    Returns:
        dict with success status, message, and output_path or base64_image
    """
    try:
        prompt = params.get("prompt")
        if not prompt:
            return {"success": False, "error": "prompt parameter is required"}
        
        model_id = params.get("model", "stabilityai/stable-diffusion-xl-base-1.0")
        negative_prompt = params.get("negative_prompt", "")
        num_inference_steps = params.get("num_inference_steps", 50)
        guidance_scale = params.get("guidance_scale", 7.5)
        width = params.get("width", 1024)
        height = params.get("height", 1024)
        seed = params.get("seed")
        output_path = params.get("output_path", "generated_image.png")
        return_base64 = params.get("return_base64", False)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if device == "cuda" else torch.float32
        
        if "sdxl" in model_id.lower() or "stable-diffusion-xl" in model_id.lower():
            pipeline = StableDiffusionXLPipeline.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                use_safetensors=True
            )
        elif "flux" in model_id.lower():
            pipeline = FluxPipeline.from_pretrained(
                model_id,
                torch_dtype=torch_dtype
            )
        else:
            pipeline = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                safety_checker=None
            )
        
        pipeline = pipeline.to(device)
        
        generator = None
        if seed is not None:
            generator = torch.Generator(device=device).manual_seed(seed)
        
        generation_kwargs = {
            "prompt": prompt,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "generator": generator
        }
        
        if negative_prompt:
            generation_kwargs["negative_prompt"] = negative_prompt
        
        if "flux" not in model_id.lower():
            generation_kwargs["width"] = width
            generation_kwargs["height"] = height
        
        image = pipeline(**generation_kwargs).images[0]
        
        result = {
            "success": True,
            "message": "Image generated successfully",
            "model": model_id,
            "prompt": prompt
        }
        
        if return_base64:
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            result["base64_image"] = img_str
        else:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            image.save(output_path)
            result["output_path"] = output_path
        
        del pipeline
        if device == "cuda":
            torch.cuda.empty_cache()
        
        return result
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to generate image: {str(e)}"
        }

def list_available_models_tool(params: dict) -> dict:
    """
    List available open-source image generation models.
    
    Args:
        params: dict (can be empty)
    
    Returns:
        dict with list of available models
    """
    models = [
        {
            "name": "Stable Diffusion XL Base 1.0",
            "id": "stabilityai/stable-diffusion-xl-base-1.0",
            "type": "SDXL",
            "recommended_size": "1024x1024"
        },
        {
            "name": "Stable Diffusion v1.5",
            "id": "runwayml/stable-diffusion-v1-5",
            "type": "SD",
            "recommended_size": "512x512"
        },
        {
            "name": "Stable Diffusion v2.1",
            "id": "stabilityai/stable-diffusion-2-1",
            "type": "SD",
            "recommended_size": "768x768"
        },
        {
            "name": "FLUX.1 Dev",
            "id": "black-forest-labs/FLUX.1-dev",
            "type": "Flux",
            "recommended_size": "1024x1024"
        },
        {
            "name": "FLUX.1 Schnell",
            "id": "black-forest-labs/FLUX.1-schnell",
            "type": "Flux",
            "recommended_size": "1024x1024"
        }
    ]
    
    return {
        "success": True,
        "models": models,
        "message": f"Found {len(models)} available models"
    }