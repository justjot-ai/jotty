python
# Generate an image with default SDXL model
generate_image(
    prompt="A serene mountain landscape at sunset, digital art",
    output_path="mountain_sunset.png"
)

# Use Flux model for more detailed generation
generate_image(
    prompt="A futuristic city with flying cars, cyberpunk style, highly detailed",
    model="flux",
    num_inference_steps=30,
    output_path="futuristic_city.png"
)

# Generate with specific dimensions and seed
generate_image(
    prompt="A cute robot playing guitar, cartoon style",
    model="sd15",
    width=512,
    height=512,
    seed=42,
    guidance_scale=8.0,
    output_path="robot_guitarist.png"
)