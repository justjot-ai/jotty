python
# Generate an image with default SDXL model
generate_image(
    prompt="A serene mountain landscape at sunset with snow-capped peaks"
)

# Use FLUX model for generation
generate_image(
    prompt="A cyberpunk city with neon lights and flying cars",
    model="black-forest-labs/FLUX.1-dev"
)

# Customize generation parameters
generate_image(
    prompt="A cute cat wearing a wizard hat",
    model="runwayml/stable-diffusion-v1-5",
    width=512,
    height=512,
    num_inference_steps=30,
    guidance_scale=8.0,
    output_path="wizard_cat.png"
)

# High-quality generation with SDXL
generate_image(
    prompt="Photorealistic portrait of an astronaut on Mars",
    model="stabilityai/stable-diffusion-xl-base-1.0",
    width=1024,
    height=1024,
    num_inference_steps=75
)