python
# Basic usage with default SDXL model
generate_image(
    prompt="A serene mountain landscape at sunset with golden light"
)

# Using Flux model with custom settings
generate_image(
    prompt="A cyberpunk city street with neon lights and rain",
    model="black-forest-labs/FLUX.1-dev",
    width=1024,
    height=1024,
    num_inference_steps=30,
    output_path="./cyberpunk_city.png"
)

# Using Stable Diffusion v1.5
generate_image(
    prompt="A cute robot reading a book in a cozy library",
    model="runwayml/stable-diffusion-v1-5",
    guidance_scale=9.0
)