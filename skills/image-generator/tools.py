import os
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, FluxPipeline
from PIL import Image
import io
import base64
from typing import Dict, Any, Optional

def image_generator_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate images using open-source models like Stable Diffusion, Flux, or SDXL.
    
    Args:
        params: Dictionary with keys:
            - prompt (str, required): Text description of the image to generate
            - model (str, optional): Model to use. Options:
                - "sdxl" (default): stabilityai/stable-diffusion-xl-base-1.0
                - "flux": black-forest-labs/FLUX.1-dev
                - "sd1.5": runwayml/stable-diffusion-v1-5
            - output_path (str, optional): Path to save the generated image
            - num_inference_steps (int, optional): Number of denoising steps (default: 50)
            - guidance_scale (float, optional): Guidance scale for generation (default: 7.5)
            - width (int, optional): Image width (default: 1024 for SDXL/Flux, 512 for SD1.5)
            - height (int, optional): Image height (default: 1024 for SDXL/Flux, 512 for SD1.5)
            - seed (int, optional): Random seed for reproducibility
            - negative_prompt (str, optional): Negative prompt to guide generation away from
            - return_base64 (bool, optional): Return base64 encoded image data (default: False)
    
    Returns:
        Dictionary with:
            - success (bool): Whether generation succeeded
            - message (str): Success/error message
            - output_path (str, optional): Path where image was saved
            - base64_data (str, optional): Base64 encoded image if return_base64=True
    """
    try:
        prompt = params.get("prompt")
        if not prompt:
            return {"success": False, "message": "Error: 'prompt' parameter is required"}
        
        model_choice = params.get("model", "sdxl").lower()
        output_path = params.get("output_path")
        num_inference_steps = params.get("num_inference_steps", 50)
        guidance_scale = params.get("guidance_scale", 7.5)
        seed = params.get("seed")
        negative_prompt = params.get("negative_prompt", "")
        return_base64 = params.get("return_base64", False)
        
        # Determine model and default dimensions
        if model_choice == "sdxl":
            model_id = "stabilityai/stable-diffusion-xl-base-1.0"
            default_size = 1024
            pipeline_class = StableDiffusionXLPipeline
        elif model_choice == "flux":
            model_id = "black-forest-labs/FLUX.1-dev"
            default_size = 1024
            pipeline_class = FluxPipeline
        elif model_choice in ["sd1.5", "sd"]:
            model_id = "runwayml/stable-diffusion-v1-5"
            default_size = 512
            pipeline_class = StableDiffusionPipeline
        else:
            return {"success": False, "message": f"Error: Unknown model '{model_choice}'. Use 'sdxl', 'flux', or 'sd1.5'"}
        
        width = params.get("width", default_size)
        height = params.get("height", default_size)
        
        # Check if CUDA is available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load the pipeline
        if device == "cuda":
            pipe = pipeline_class.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                use_safetensors=True
            )
            pipe = pipe.to(device)
        else:
            pipe = pipeline_class.from_pretrained(
                model_id,
                torch_dtype=torch.float32
            )
        
        # Set random seed if provided
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
            "generator": generator
        }
        
        if negative_prompt:
            generation_params["negative_prompt"] = negative_prompt
        
        result = pipe(**generation_params)
        image = result.images[0]
        
        response = {"success": True, "message": f"Image generated successfully using {model_choice}"}
        
        # Save image if output path provided
        if output_path:
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
            image.save(output_path)
            response["output_path"] = output_path
        
        # Return base64 if requested
        if return_base64:
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            response["base64_data"] = img_str
        
        return response
        
    except Exception as e:
        return {"success": False, "message": f"Error generating image: {str(e)}"}

def generate_image_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Alias for image_generator_tool"""
    return image_generator_tool(params)