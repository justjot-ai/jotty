import os
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, DiffusionPipeline
from PIL import Image
import io
import base64
from typing import Dict, Any, Optional
import gc

def _clear_memory():
    """Clear GPU/CPU memory after generation"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def _save_image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def generate_image_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate images using Stable Diffusion or Flux models
    
    params:
        prompt: str - Text description of the image to generate (required)
        model: str - Model to use (default: "stabilityai/stable-diffusion-xl-base-1.0")
                     Options: "stabilityai/stable-diffusion-xl-base-1.0",
                             "black-forest-labs/FLUX.1-dev",
                             "runwayml/stable-diffusion-v1-5"
        negative_prompt: str - What to avoid in the image (optional)
        num_inference_steps: int - Number of denoising steps (default: 50)
        guidance_scale: float - How closely to follow prompt (default: 7.5)
        width: int - Image width (default: 1024 for SDXL, 512 for SD1.5)
        height: int - Image height (default: 1024 for SDXL, 512 for SD1.5)
        seed: int - Random seed for reproducibility (optional)
        output_path: str - Path to save the image (optional, if not provided returns base64)
    """
    try:
        prompt = params.get("prompt")
        if not prompt:
            return {"success": False, "error": "Prompt is required"}
        
        model_id = params.get("model", "stabilityai/stable-diffusion-xl-base-1.0")
        negative_prompt = params.get("negative_prompt", "")
        num_inference_steps = params.get("num_inference_steps", 50)
        guidance_scale = params.get("guidance_scale", 7.5)
        seed = params.get("seed")
        output_path = params.get("output_path")
        
        # Determine default dimensions based on model
        if "xl" in model_id.lower() or "sdxl" in model_id.lower():
            default_width, default_height = 1024, 1024
        elif "flux" in model_id.lower():
            default_width, default_height = 1024, 1024
        else:
            default_width, default_height = 512, 512
        
        width = params.get("width", default_width)
        height = params.get("height", default_height)
        
        # Set device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        # Load appropriate pipeline
        if "xl" in model_id.lower() or "sdxl" in model_id.lower():
            pipeline = StableDiffusionXLPipeline.from_pretrained(
                model_id,
                torch_dtype=dtype,
                use_safetensors=True,
                variant="fp16" if torch.cuda.is_available() else None
            )
        elif "flux" in model_id.lower():
            pipeline = DiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=dtype
            )
        else:
            pipeline = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=dtype,
                use_safetensors=True
            )
        
        pipeline = pipeline.to(device)
        
        # Enable memory optimizations
        if hasattr(pipeline, "enable_attention_slicing"):
            pipeline.enable_attention_slicing()
        if hasattr(pipeline, "enable_vae_slicing"):
            pipeline.enable_vae_slicing()
        
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
        
        result = pipeline(**generation_params)
        image = result.images[0]
        
        # Save or encode image
        response = {"success": True, "prompt": prompt, "model": model_id}
        
        if output_path:
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
            image.save(output_path)
            response["output_path"] = output_path
        else:
            response["image_base64"] = _save_image_to_base64(image)
        
        # Cleanup
        del pipeline
        _clear_memory()
        
        return response
        
    except Exception as e:
        _clear_memory()
        return {"success": False, "error": str(e)}

def generate_batch_images_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate multiple images from a list of prompts
    
    params:
        prompts: list[str] - List of text descriptions (required)
        model: str - Model to use (default: "stabilityai/stable-diffusion-xl-base-1.0")
        negative_prompt: str - What to avoid in all images (optional)
        num_inference_steps: int - Number of denoising steps (default: 50)
        guidance_scale: float - How closely to follow prompt (default: 7.5)
        width: int - Image width (default based on model)
        height: int - Image height (default based on model)
        seed: int - Starting random seed (optional, increments for each image)
        output_dir: str - Directory to save images (optional)
    """
    try:
        prompts = params.get("prompts")
        if not prompts or not isinstance(prompts, list):
            return {"success": False, "error": "prompts must be a list of strings"}
        
        model_id = params.get("model", "stabilityai/stable-diffusion-xl-base-1.0")
        negative_prompt = params.get("negative_prompt", "")
        num_inference_steps = params.get("num_inference_steps", 50)
        guidance_scale = params.get("guidance_scale", 7.5)
        seed = params.get("seed")
        output_dir = params.get("output_dir")
        
        # Determine default dimensions
        if "xl" in model_id.lower() or "sdxl" in model_id.lower() or "flux" in model_id.lower():
            default_width, default_height = 1024, 1024
        else:
            default_width, default_height = 512, 512
        
        width = params.get("width", default_width)
        height = params.get("height", default_height)
        
        # Load pipeline once
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        if "xl" in model_id.lower() or "sdxl" in model_id.lower():
            pipeline = StableDiffusionXLPipeline.from_pretrained(
                model_id,
                torch_dtype=dtype,
                use_safetensors=True,
                variant="fp16" if torch.cuda.is_available() else None
            )
        elif "flux" in model_id.lower():
            pipeline = DiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=dtype
            )
        else:
            pipeline = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=dtype,
                use_safetensors=True
            )
        
        pipeline = pipeline.to(device)
        
        if hasattr(pipeline, "enable_attention_slicing"):
            pipeline.enable_attention_slicing()
        if hasattr(pipeline, "enable_vae_slicing"):
            pipeline.enable_vae_slicing()
        
        results = []
        
        for idx, prompt in enumerate(prompts):
            generator = None
            if seed is not None:
                generator = torch.Generator(device=device).manual_seed(seed + idx)
            
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
            
            result = pipeline(**generation_params)
            image = result.images[0]
            
            image_result = {"prompt": prompt, "index": idx}
            
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                filename = f"image_{idx:03d}.png"
                filepath = os.path.join(output_dir, filename)
                image.save(filepath)
                image_result["output_path"] = filepath
            else:
                image_result["image_base64"] = _save_image_to_base64(image)
            
            results.append(image_result)
        
        # Cleanup
        del pipeline
        _clear_memory()
        
        return {
            "success": True,
            "model": model_id,
            "count": len(results),
            "images": results
        }
        
    except Exception as e:
        _clear_memory()
        return {"success": False, "error": str(e)}