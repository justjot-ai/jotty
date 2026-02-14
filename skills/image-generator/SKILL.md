python
# Generate an image with default settings (SDXL)
generate_image(
    prompt="A serene mountain landscape at sunset with purple clouds"
)

# Generate with specific model and custom settings
generate_image(
    prompt="A cyberpunk cityscape with neon lights",
    model="black-forest-labs/FLUX.1-dev",
    output_path="cyberpunk_city.png",
    width=1024,
    height=768,
    num_inference_steps=30,
    guidance_scale=8.0
)

# Quick generation with Stable Diffusion 1.5
generate_image(
    prompt="A cute robot playing guitar",
    model="runwayml/stable-diffusion-v1-5",
    width=512,
    height=512
)