python
# Generate image with default SDXL model
generate_image(
    prompt="A serene mountain landscape at sunset with purple clouds"
)

# Generate with specific model and parameters
generate_image(
    prompt="A cyberpunk city street with neon lights",
    model="flux",
    negative_prompt="blurry, low quality",
    width=1024,
    height=768,
    num_inference_steps=30,
    guidance_scale=8.0,
    output_path="./cyberpunk_city.png"
)

# Quick generation with SD 1.5 (faster, lower quality)
generate_image(
    prompt="A cute cat wearing a hat",
    model="sd-1.5",
    width=512,
    height=512
)