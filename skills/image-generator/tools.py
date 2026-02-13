import os
import base64
from io import BytesIO
from typing import Dict, Any

def image_generator_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate images using open-source models like Stable Diffusion, Flux, or SDXL.
    
    Args:
        params: Dictionary containing:
            - prompt (str, required): Text description of the image to generate
            - model (str, optional): Model to use. Options:
                - "sdxl" or "stable-diffusion-xl-base-1.0" (default)
                - "flux" or "FLUX.1-dev"
                - "sd1.5" or "stable-diffusion-v1-5"
            - output_path (str, optional): Path to save the image. If not provided, returns base64
            - num_inference_steps (int, optional): Number of denoising steps (default: 50)
            - guidance_scale (float, optional): Guidance scale for generation (default: 7.5)
            - width (int, optional): Image width (default: 1024 for SDXL, 512 for SD1.5)
            - height (int, optional): Image height (default: 1024 for SDXL, 512 for SD1.5)
            - seed (int, optional): Random seed for reproducibility
    
    Returns:
        Dictionary with:
            - success (bool): Whether generation succeeded
            - message (str): Success or error message
            - output_path (str, optional): Path where image was saved
            - image_base64 (str, optional): Base64 encoded image if no output_path
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
            "message": f"Required library not installed: {str(e)}. Install with: pip install torch diffusers transformers accelerate pillow"
        }
    
    # Validate required parameters
    if "prompt" not in params:
        return {
            "success": False,
            "message": "Missing required parameter: prompt"
        }
    
    prompt = params["prompt"]
    model = params.get("model", "sdxl").lower()
    output_path = params.get("output_path")
    num_inference_steps = params.get("num_inference_steps", 50)
    guidance_scale = params.get("guidance_scale", 7.5)
    seed = params.get("seed")
    
    # Map model names to Hugging Face model IDs
    model_mapping = {
        "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
        "stable-diffusion-xl-base-1.0": "stabilityai/stable-diffusion-xl-base-1.0",
        "flux": "black-forest-labs/FLUX.1-dev",
        "flux.1-dev": "black-forest-labs/FLUX.1-dev",
        "sd1.5": "runwayml/stable-diffusion-v1-5",
        "stable-diffusion-v1-5": "runwayml/stable-diffusion-v1-5"
    }
    
    if model not in model_mapping:
        return {
            "success": False,
            "message": f"Unsupported model: {model}. Choose from: sdxl, flux, sd1.5"
        }
    
    model_id = model_mapping[model]
    
    # Set default dimensions based on model
    if "sdxl" in model or "xl" in model:
        default_width = 1024
        default_height = 1024
    else:
        default_width = 512
        default_height = 512
    
    width = params.get("width", default_width)
    height = params.get("height", default_height)
    
    try:
        # Determine device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load appropriate pipeline
        if "flux" in model:
            pipe = FluxPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            )
        elif "sdxl" in model or "xl" in model:
            pipe = StableDiffusionXLPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                use_safetensors=True
            )
        else:
            pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            )
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        
        pipe = pipe.to(device)
        
        # Enable memory optimizations if on GPU
        if device == "cuda":
            pipe.enable_attention_slicing()
        
        # Set seed for reproducibility
        generator = None
        if seed is not None:
            generator = torch.Generator(device=device).manual_seed(seed)
        
        # Generate image
        result = pipe(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            generator=generator
        )
        
        image = result.images[0]
        
        # Save or encode image
        if output_path:
            # Ensure directory exists
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
            image.save(output_path)
            return {
                "success": True,
                "message": f"Image generated successfully using {model_id}",
                "output_path": output_path
            }
        else:
            # Convert to base64
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            return {
                "success": True,
                "message": f"Image generated successfully using {model_id}",
                "image_base64": img_base64
            }
    
    except Exception as e:
        return {
            "success": False,
            "message": f"Error generating image: {str(e)}"
        }