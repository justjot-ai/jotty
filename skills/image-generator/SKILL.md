---
name: image-generation
description: "Generate images using open-source models like Stable Diffusion, Flux, or SDXL. No API key required."
---

# image-generator

## Description
Generate images using open-source models like Stable Diffusion, Flux, or SDXL. No API key required.

## Type
base

## Capabilities
- media

## Triggers
- "generate image"
- "create image"
- "image generation"
- "stable diffusion"
- "flux"
- "sdxl"

## Category
media

## Tools

### image_generator_tool
Generate images using open-source models like Stable Diffusion, Flux, or SDXL. No API key required.

**Parameters:**
- `prompt` (str, required): Description of the image to generate

**Returns:**
- `success` (bool): Whether generation succeeded
- `message` (str): Result message

## Requirements
Use open-source models via Hugging Face transformers or diffusers library. Support models like: stabilityai/stable-diffusion-xl-base-1.0, black-forest-labs/FLUX.1-dev, runwayml/stable-diffusion-v1-5

## Dependencies
None
