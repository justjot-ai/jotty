python
# Generate image with default SDXL model
generate_image(
    prompt="A serene mountain landscape at sunset with snow-capped peaks"
)

# Generate with specific model and custom settings
generate_image(
    prompt="A cyberpunk city with neon lights and flying cars",
    model="black-forest-labs/FLUX.1-dev",
    width=1024,
    height=1024,
    num_inference_steps=30,
    guidance_scale=8.0,
    output_path="./cyberpunk_city.png"
)

# Quick generation with SD1.5
generate_image(
    prompt="A cute cat wearing a wizard hat",
    model="runwayml/stable-diffusion-v1-5",
    num_inference_steps=25
)