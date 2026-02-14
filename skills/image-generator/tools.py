import os
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, DiffusionPipeline
from PIL import Image
import io
import base64

def image_generator_tool(params: dict) -> dict:
    """
    Generate images using open-source models like Stable Diffusion, Flux, or SDXL.
    
    Args:
        params: dict with keys:
            - prompt: str (required) - Text description of the image to generate
            - model: str (optional) - Model to use, defaults to "stabilityai/stable-diffusion-xl-base-1.0"
                Options: "stabilityai/stable-diffusion-xl-base-1.0", 
                        "black-forest-labs/FLUX.1-dev",
                        "runwayml/stable-diffusion-v1-5"
            - num_inference_steps: int (optional) - Number of denoising steps, defaults to 50
            - guidance_scale: float (optional) - Guidance scale for generation, defaults to 7.5
            - width: int (optional) - Image width, defaults to 512 or 1024 depending on model
            - height: int (optional) - Image height, defaults to 512 or 1024 depending on model
            - seed: int (optional) - Random seed for reproducibility
            - output_path: str (optional) - Path to save the image, defaults to "generated_image.png"
    
    Returns:
        dict with keys:
            - success: bool
            - message: str
            - output_path: str (if successful)
            - image_base64: str (if successful) - Base64 encoded image
    """
    try:
        prompt = params.get("prompt")
        if not prompt:
            return {"success": False, "message": "Prompt is required"}
        
        model_id = params.get("model", "stabilityai/stable-diffusion-xl-base-1.0")
        num_inference_steps = params.get("num_inference_steps", 50)
        guidance_scale = params.get("guidance_scale", 7.5)
        seed = params.get("seed")
        output_path = params.get("output_path", "generated_image.png")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if device == "cuda" else torch.float32
        
        if "xl" in model_id.lower() or "sdxl" in model_id.lower():
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
        
        image = pipeline(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            generator=generator
        ).images[0]
        
        image.save(output_path)
        
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        return {
            "success": True,
            "message": f"Image generated successfully using {model_id}",
            "output_path": output_path,
            "image_base64": img_base64
        }
        
    except Exception as e:
        return {
            "success": False,
            "message": f"Error generating image: {str(e)}"
        }

def list_available_models_tool(params: dict) -> dict:
    """
    List available open-source image generation models.
    
    Args:
        params: dict (empty or unused)
    
    Returns:
        dict with keys:
            - success: bool
            - models: list of str
    """
    models = [
        "stabilityai/stable-diffusion-xl-base-1.0",
        "black-forest-labs/FLUX.1-dev",
        "runwayml/stable-diffusion-v1-5",
        "stabilityai/stable-diffusion-2-1",
        "stabilityai/stable-diffusion-2-1-base"
    ]
    
    return {
        "success": True,
        "models": models,
        "message": f"Found {len(models)} available models"
    }