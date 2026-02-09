import os
from typing import Dict, Any
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, FluxPipeline
from PIL import Image
import io
import base64

def image_generator_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate images using open-source models like Stable Diffusion, Flux, or SDXL.
    
    Args:
        params: Dictionary containing:
            - prompt (str): Text description of the image to generate
            - model (str, optional): Model ID (default: "runwayml/stable-diffusion-v1-5")
            - output_path (str, optional): Path to save the image (default: "generated_image.png")
            - num_inference_steps (int, optional): Number of denoising steps (default: 50)
            - guidance_scale (float, optional): Guidance scale for generation (default: 7.5)
            - width (int, optional): Image width (default: 512)
            - height (int, optional): Image height (default: 512)
            - seed (int, optional): Random seed for reproducibility
            - return_base64 (bool, optional): Return image as base64 string (default: False)
    
    Returns:
        Dictionary with success status and result information
    """
    try:
        prompt = params.get("prompt")
        if not prompt:
            return {"success": False, "error": "Missing required parameter: prompt"}
        
        model_id = params.get("model", "runwayml/stable-diffusion-v1-5")
        output_path = params.get("output_path", "generated_image.png")
        num_inference_steps = params.get("num_inference_steps", 50)
        guidance_scale = params.get("guidance_scale", 7.5)
        width = params.get("width", 512)
        height = params.get("height", 512)
        seed = params.get("seed")
        return_base64 = params.get("return_base64", False)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if device == "cuda" else torch.float32
        
        if "xl" in model_id.lower() or "sdxl" in model_id.lower():
            pipe = StableDiffusionXLPipeline.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                use_safetensors=True
            )
            width = params.get("width", 1024)
            height = params.get("height", 1024)
        elif "flux" in model_id.lower():
            pipe = FluxPipeline.from_pretrained(
                model_id,
                torch_dtype=torch_dtype
            )
            width = params.get("width", 1024)
            height = params.get("height", 1024)
        else:
            pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                use_safetensors=True
            )
        
        pipe = pipe.to(device)
        
        if hasattr(pipe, 'enable_attention_slicing'):
            pipe.enable_attention_slicing()
        
        generator = None
        if seed is not None:
            generator = torch.Generator(device=device).manual_seed(seed)
        
        generation_kwargs = {
            "prompt": prompt,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "width": width,
            "height": height,
        }
        
        if generator is not None:
            generation_kwargs["generator"] = generator
        
        image = pipe(**generation_kwargs).images[0]
        
        image.save(output_path)
        
        result = {
            "success": True,
            "output_path": output_path,
            "model": model_id,
            "prompt": prompt,
            "width": width,
            "height": height
        }
        
        if return_base64:
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            result["base64_image"] = img_str
        
        return result
        
    except Exception as e:
        return {"success": False, "error": str(e)}

def stable_diffusion_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate images using Stable Diffusion v1.5.
    
    Args:
        params: Dictionary containing generation parameters
    
    Returns:
        Dictionary with success status and result information
    """
    params["model"] = params.get("model", "runwayml/stable-diffusion-v1-5")
    return image_generator_tool(params)

def sdxl_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate images using Stable Diffusion XL.
    
    Args:
        params: Dictionary containing generation parameters
    
    Returns:
        Dictionary with success status and result information
    """
    params["model"] = params.get("model", "stabilityai/stable-diffusion-xl-base-1.0")
    return image_generator_tool(params)

def flux_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate images using Flux model.
    
    Args:
        params: Dictionary containing generation parameters
    
    Returns:
        Dictionary with success status and result information
    """
    params["model"] = params.get("model", "black-forest-labs/FLUX.1-dev")
    return image_generator_tool(params)