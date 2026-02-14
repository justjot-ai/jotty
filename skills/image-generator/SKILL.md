python
# Basic usage with default SDXL model
generate_image(
    prompt="A serene mountain landscape at sunset with a crystal clear lake"
)

# Using Flux model with custom parameters
generate_image(
    prompt="A futuristic city with flying cars and neon lights",
    model="flux",
    negative_prompt="blurry, low quality, distorted",
    num_inference_steps=75,
    guidance_scale=8.0,
    seed=42,
    output_path="./futuristic_city.png"
)

# Using Stable Diffusion 1.5 with specific dimensions
generate_image(
    prompt="Portrait of a wise old wizard with a long white beard",
    model="sd1.5",
    width=512,
    height=768,
    negative_prompt="cartoon, anime, low quality"
)

# High-quality generation with more steps
generate_image(
    prompt="Photorealistic portrait of a red fox in autumn forest",
    model="sdxl",
    width=1024,
    height=1024,
    num_inference_steps=100,
    guidance_scale=9.0
)