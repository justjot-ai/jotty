# image-generator

## Description
Generate images using open-source models like Stable Diffusion, Flux, or SDXL. No API key required.

## Tools

### generate_image
Generate an image from a text prompt using open-source diffusion models.

**Parameters:**
- `prompt` (string, required): Text description of the image to generate
- `model` (string, optional): Model to use. Options:
  - `stabilityai/stable-diffusion-xl-base-1.0` (default)
  - `black-forest-labs/FLUX.1-dev`
  - `runwayml/stable-diffusion-v1-5`
- `output_path` (string, optional): Path to save the generated image (default: `generated_image.png`)
- `num_inference_steps` (integer, optional): Number of denoising steps (default: 50)
- `guidance_scale` (float, optional): How closely to follow the prompt (default: 7.5)
- `width` (integer, optional): Image width in pixels (default: 1024)
- `height` (integer, optional): Image height in pixels (default: 1024)
- `seed` (integer, optional): Random seed for reproducibility

**Returns:**
- Path to the generated image file

## Usage