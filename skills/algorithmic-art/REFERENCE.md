# Algorithmic Art Skill - API Reference

## Contents

### Tools

| Function | Description |
|----------|-------------|
| [`create_noise_art_tool`](#create_noise_art_tool) | Create Perlin/Simplex noise-based generative art. |
| [`create_flow_field_tool`](#create_flow_field_tool) | Create flow field visualization art. |
| [`create_geometric_pattern_tool`](#create_geometric_pattern_tool) | Create geometric pattern art. |
| [`create_fractal_tool`](#create_fractal_tool) | Create fractal images (Mandelbrot, Julia sets). |
| [`create_generative_landscape_tool`](#create_generative_landscape_tool) | Create generative landscape art. |
| [`apply_artistic_filter_tool`](#apply_artistic_filter_tool) | Apply artistic effects to an existing image. |
| [`list_color_palettes_tool`](#list_color_palettes_tool) | List all available color palettes for generative art. |

### Helper Functions

| Function | Description |
|----------|-------------|
| [`get_palette`](#get_palette) | Get a color palette by name. |
| [`list_palettes`](#list_palettes) | List available palette names. |
| [`interpolate_color`](#interpolate_color) | Interpolate between two colors. |
| [`noise2d`](#noise2d) | Generate 2D Perlin noise. |
| [`fractal_noise`](#fractal_noise) | Generate fractal noise with multiple octaves. |

---

## `create_noise_art_tool`

Create Perlin/Simplex noise-based generative art.

**Parameters:**

- **output_path** (`str, optional`): Path to save image (default: temp file)
- **width** (`int, optional`): Image width in pixels (default: 800)
- **height** (`int, optional`): Image height in pixels (default: 600)
- **seed** (`int, optional`): Random seed for reproducibility
- **color_palette** (`str, optional`): Color palette name (default: "sunset")
- **noise_scale** (`float, optional`): Scale of noise (default: 0.01)

**Returns:** Dictionary with: - success (bool): Whether generation succeeded - image_path (str): Path to generated image - error (str, optional): Error message if failed

---

## `create_flow_field_tool`

Create flow field visualization art.

**Parameters:**

- **output_path** (`str, optional`): Path to save image (default: temp file)
- **width** (`int, optional`): Image width in pixels (default: 800)
- **height** (`int, optional`): Image height in pixels (default: 600)
- **seed** (`int, optional`): Random seed for reproducibility
- **num_particles** (`int, optional`): Number of particles (default: 5000)
- **color_palette** (`str, optional`): Color palette name (default: "ocean")

**Returns:** Dictionary with: - success (bool): Whether generation succeeded - image_path (str): Path to generated image - error (str, optional): Error message if failed

---

## `create_geometric_pattern_tool`

Create geometric pattern art.

**Parameters:**

- **output_path** (`str, optional`): Path to save image (default: temp file)
- **width** (`int, optional`): Image width in pixels (default: 800)
- **height** (`int, optional`): Image height in pixels (default: 600)
- **pattern** (`str, optional`): Pattern type - circles, triangles, hexagons, voronoi (default: "circles")
- **colors** (`list, optional`): List of RGB tuples or palette name (default: "pastel")

**Returns:** Dictionary with: - success (bool): Whether generation succeeded - image_path (str): Path to generated image - error (str, optional): Error message if failed

---

## `create_fractal_tool`

Create fractal images (Mandelbrot, Julia sets).

**Parameters:**

- **output_path** (`str, optional`): Path to save image (default: temp file)
- **fractal_type** (`str, optional`): Type of fractal - mandelbrot, julia (default: "mandelbrot")
- **width** (`int, optional`): Image width in pixels (default: 800)
- **height** (`int, optional`): Image height in pixels (default: 600)
- **iterations** (`int, optional`): Max iterations for escape calculation (default: 100)
- **color_map** (`str, optional`): Color palette name (default: "cosmic")

**Returns:** Dictionary with: - success (bool): Whether generation succeeded - image_path (str): Path to generated image - error (str, optional): Error message if failed

---

## `create_generative_landscape_tool`

Create generative landscape art.

**Parameters:**

- **output_path** (`str, optional`): Path to save image (default: temp file)
- **width** (`int, optional`): Image width in pixels (default: 800)
- **height** (`int, optional`): Image height in pixels (default: 600)
- **seed** (`int, optional`): Random seed for reproducibility
- **style** (`str, optional`): Landscape style - mountains, waves, terrain (default: "mountains")

**Returns:** Dictionary with: - success (bool): Whether generation succeeded - image_path (str): Path to generated image - error (str, optional): Error message if failed

---

## `apply_artistic_filter_tool`

Apply artistic effects to an existing image.

**Parameters:**

- **image_path** (`str, required`): Path to input image
- **output_path** (`str, optional`): Path to save filtered image (default: temp file)
- **filter** (`str, optional`): Filter type - pointillism, mosaic, sketch, watercolor (default: "pointillism")

**Returns:** Dictionary with: - success (bool): Whether filter was applied successfully - image_path (str): Path to filtered image - error (str, optional): Error message if failed

---

## `list_color_palettes_tool`

List all available color palettes for generative art.

**Parameters:**

- **params** (`Dict[str, Any]`)

**Returns:** Dictionary with: - success (bool): Always True - palettes (list): List of available palette names with colors

---

## `get_palette`

Get a color palette by name.

**Parameters:**

- **name** (`str`)

**Returns:** `List[Tuple[int, int, int]]`

---

## `list_palettes`

List available palette names.

**Returns:** `List[str]`

---

## `interpolate_color`

Interpolate between two colors.

**Parameters:**

- **color1** (`Tuple[int, int, int]`)
- **color2** (`Tuple[int, int, int]`)
- **t** (`float`)

**Returns:** `Tuple[int, int, int]`

---

## `noise2d`

Generate 2D Perlin noise.

**Parameters:**

- **x** (`float`)
- **y** (`float`)

**Returns:** `float`

---

## `fractal_noise`

Generate fractal noise with multiple octaves.

**Parameters:**

- **x** (`float`)
- **y** (`float`)
- **octaves** (`int`)

**Returns:** `float`
