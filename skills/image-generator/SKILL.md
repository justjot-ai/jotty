python
# Generate an image with default SDXL model
generate_image(
    prompt="A serene mountain landscape at sunset with purple clouds"
)

# Generate with specific model and parameters
generate_image(
    prompt="A futuristic city with flying cars and neon lights",
    model="black-forest-labs/FLUX.1-dev",
    width=1024,
    height=768,
    num_inference_steps=30,
    guidance_scale=8.0,
    seed=42,
    output_path="./futuristic_city.png"
)

# Quick generation with Stable Diffusion v1.5
generate_image(
    prompt="A cute cat wearing a wizard hat",
    model="runwayml/stable-diffusion-v1-5",
    width=512,
    height=512
)