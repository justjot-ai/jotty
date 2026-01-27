# Algorithmic Art Skill

## Description
Generative art toolkit using Pillow and numpy for creating various types of algorithmic and procedural art including noise-based images, flow fields, geometric patterns, fractals, and landscapes. Also includes artistic filters to transform existing images.

## Installation
```bash
pip install Pillow numpy
```

## Tools

### create_noise_art_tool
Create Perlin/Simplex noise-based generative art.

**Parameters:**
- `output_path` (str, optional): Path to save image (default: temp file)
- `width` (int, optional): Image width in pixels (default: 800)
- `height` (int, optional): Image height in pixels (default: 600)
- `seed` (int, optional): Random seed for reproducibility
- `color_palette` (str, optional): Color palette name (default: "sunset"). Options: sunset, ocean, forest, fire, neon, pastel, monochrome, cyberpunk, earth, cosmic
- `noise_scale` (float, optional): Scale of noise pattern (default: 0.01)

**Returns:**
- `success` (bool): Whether generation succeeded
- `image_path` (str): Path to generated image
- `width` (int): Image width
- `height` (int): Image height
- `seed` (int): Seed used for generation
- `error` (str, optional): Error message if failed

---

### create_flow_field_tool
Create flow field visualization with particles following noise-based vector fields.

**Parameters:**
- `output_path` (str, optional): Path to save image (default: temp file)
- `width` (int, optional): Image width in pixels (default: 800)
- `height` (int, optional): Image height in pixels (default: 600)
- `seed` (int, optional): Random seed for reproducibility
- `num_particles` (int, optional): Number of particles to simulate (default: 5000)
- `color_palette` (str, optional): Color palette name (default: "ocean")

**Returns:**
- `success` (bool): Whether generation succeeded
- `image_path` (str): Path to generated image
- `width` (int): Image width
- `height` (int): Image height
- `seed` (int): Seed used
- `num_particles` (int): Number of particles used
- `error` (str, optional): Error message if failed

---

### create_geometric_pattern_tool
Create geometric pattern art with various shapes.

**Parameters:**
- `output_path` (str, optional): Path to save image (default: temp file)
- `width` (int, optional): Image width in pixels (default: 800)
- `height` (int, optional): Image height in pixels (default: 600)
- `pattern` (str, optional): Pattern type (default: "circles"). Options:
  - `circles` - Random circles in a grid
  - `triangles` - Tessellated triangles
  - `hexagons` - Honeycomb hexagon pattern
  - `voronoi` - Voronoi diagram cells
- `colors` (list or str, optional): List of RGB tuples or palette name (default: "pastel")
- `seed` (int, optional): Random seed for reproducibility

**Returns:**
- `success` (bool): Whether generation succeeded
- `image_path` (str): Path to generated image
- `width` (int): Image width
- `height` (int): Image height
- `pattern` (str): Pattern type used
- `error` (str, optional): Error message if failed

---

### create_fractal_tool
Create fractal images including Mandelbrot and Julia sets.

**Parameters:**
- `output_path` (str, optional): Path to save image (default: temp file)
- `fractal_type` (str, optional): Type of fractal (default: "mandelbrot"). Options:
  - `mandelbrot` - Classic Mandelbrot set
  - `julia` - Julia set (can specify c_real and c_imag parameters)
- `width` (int, optional): Image width in pixels (default: 800)
- `height` (int, optional): Image height in pixels (default: 600)
- `iterations` (int, optional): Max iterations for escape calculation (default: 100)
- `color_map` (str, optional): Color palette name (default: "cosmic")
- `c_real` (float, optional): Real part of c for Julia set (default: -0.7)
- `c_imag` (float, optional): Imaginary part of c for Julia set (default: 0.27015)

**Returns:**
- `success` (bool): Whether generation succeeded
- `image_path` (str): Path to generated image
- `width` (int): Image width
- `height` (int): Image height
- `fractal_type` (str): Fractal type used
- `iterations` (int): Iterations used
- `error` (str, optional): Error message if failed

---

### create_generative_landscape_tool
Create generative landscape art with procedural terrain and scenery.

**Parameters:**
- `output_path` (str, optional): Path to save image (default: temp file)
- `width` (int, optional): Image width in pixels (default: 800)
- `height` (int, optional): Image height in pixels (default: 600)
- `seed` (int, optional): Random seed for reproducibility
- `style` (str, optional): Landscape style (default: "mountains"). Options:
  - `mountains` - Layered mountain silhouettes with sunset sky
  - `waves` - Ocean waves with foam effects
  - `terrain` - Top-down terrain map with water, land, and snow

**Returns:**
- `success` (bool): Whether generation succeeded
- `image_path` (str): Path to generated image
- `width` (int): Image width
- `height` (int): Image height
- `style` (str): Style used
- `seed` (int): Seed used
- `error` (str, optional): Error message if failed

---

### apply_artistic_filter_tool
Apply artistic effects to an existing image.

**Parameters:**
- `image_path` (str, required): Path to input image
- `output_path` (str, optional): Path to save filtered image (default: temp file)
- `filter` (str, optional): Filter type (default: "pointillism"). Options:
  - `pointillism` - Converts image to dots like pointillist painting
  - `mosaic` - Creates mosaic tile effect
  - `sketch` - Pencil sketch effect
  - `watercolor` - Soft watercolor painting effect

**Returns:**
- `success` (bool): Whether filter was applied successfully
- `image_path` (str): Path to filtered image
- `filter` (str): Filter type applied
- `original_path` (str): Path to original image
- `error` (str, optional): Error message if failed

---

### list_color_palettes_tool
List all available color palettes for generative art.

**Parameters:**
- None required (empty dictionary)

**Returns:**
- `success` (bool): Always True
- `palettes` (list): List of available palette objects containing:
  - `name` (str): Palette name
  - `colors` (list): RGB tuples
  - `hex_colors` (list): Hex color strings

## Available Color Palettes
- `sunset` - Warm sunset colors (oranges, purples)
- `ocean` - Deep ocean blues and purples
- `forest` - Natural green tones
- `fire` - Warm fire colors (reds, oranges, yellows)
- `neon` - Bright neon colors
- `pastel` - Soft pastel tones
- `monochrome` - Grayscale values
- `cyberpunk` - Dark with neon accents
- `earth` - Natural earth tones (browns, tans)
- `cosmic` - Space-themed deep purples and blues
