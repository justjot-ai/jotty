import logging
import math
import os
import random
import tempfile
from typing import Any, Dict, List, Optional, Tuple

from Jotty.core.infrastructure.utils.skill_status import SkillStatus
from Jotty.core.infrastructure.utils.tool_helpers import tool_error, tool_response, tool_wrapper

# Status emitter for progress updates
status = SkillStatus("algorithmic-art")


logger = logging.getLogger(__name__)


class ColorPaletteManager:
    """Manages color palettes for generative art."""

    PALETTES = {
        "sunset": [(255, 94, 77), (255, 154, 86), (255, 206, 93), (172, 135, 197), (87, 95, 207)],
        "ocean": [(0, 63, 92), (47, 75, 124), (102, 81, 145), (160, 81, 149), (212, 80, 135)],
        "forest": [(27, 94, 32), (56, 142, 60), (102, 187, 106), (165, 214, 167), (200, 230, 201)],
        "fire": [(255, 87, 34), (255, 138, 101), (255, 183, 77), (255, 213, 79), (255, 241, 118)],
        "neon": [(255, 0, 255), (0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 128)],
        "pastel": [
            (255, 179, 186),
            (255, 223, 186),
            (255, 255, 186),
            (186, 255, 201),
            (186, 225, 255),
        ],
        "monochrome": [(0, 0, 0), (64, 64, 64), (128, 128, 128), (192, 192, 192), (255, 255, 255)],
        "cyberpunk": [(15, 14, 23), (57, 21, 74), (148, 33, 106), (255, 0, 102), (0, 255, 255)],
        "earth": [(139, 90, 43), (160, 120, 60), (180, 160, 100), (200, 190, 140), (220, 220, 180)],
        "cosmic": [(10, 10, 35), (30, 30, 80), (75, 30, 120), (140, 50, 160), (200, 100, 200)],
    }

    @classmethod
    def get_palette(cls, name: str) -> List[Tuple[int, int, int]]:
        """Get a color palette by name."""
        return cls.PALETTES.get(name.lower(), cls.PALETTES["sunset"])

    @classmethod
    def list_palettes(cls) -> List[str]:
        """List available palette names."""
        return list(cls.PALETTES.keys())

    @classmethod
    def interpolate_color(
        cls, color1: Tuple[int, int, int], color2: Tuple[int, int, int], t: float
    ) -> Tuple[int, int, int]:
        """Interpolate between two colors."""
        return (
            int(color1[0] + (color2[0] - color1[0]) * t),
            int(color1[1] + (color2[1] - color1[1]) * t),
            int(color1[2] + (color2[2] - color1[2]) * t),
        )


class NoiseGenerator:
    """Generates Perlin-like noise for generative art."""

    def __init__(self, seed: int = None):
        self.seed = seed if seed is not None else random.randint(0, 2**31)
        random.seed(self.seed)
        self.permutation = list(range(256))
        random.shuffle(self.permutation)
        self.permutation = self.permutation * 2

    def _fade(self, t: float) -> float:
        """Smoothstep fade function."""
        return t * t * t * (t * (t * 6 - 15) + 10)

    def _lerp(self, a: float, b: float, t: float) -> float:
        """Linear interpolation."""
        return a + t * (b - a)

    def _grad(self, hash: int, x: float, y: float) -> float:
        """Gradient function."""
        h = hash & 3
        if h == 0:
            return x + y
        elif h == 1:
            return -x + y
        elif h == 2:
            return x - y
        else:
            return -x - y

    def noise2d(self, x: float, y: float) -> float:
        """Generate 2D Perlin noise."""
        xi = int(x) & 255
        yi = int(y) & 255
        xf = x - int(x)
        yf = y - int(y)

        u = self._fade(xf)
        v = self._fade(yf)

        aa = self.permutation[self.permutation[xi] + yi]
        ab = self.permutation[self.permutation[xi] + yi + 1]
        ba = self.permutation[self.permutation[xi + 1] + yi]
        bb = self.permutation[self.permutation[xi + 1] + yi + 1]

        x1 = self._lerp(self._grad(aa, xf, yf), self._grad(ba, xf - 1, yf), u)
        x2 = self._lerp(self._grad(ab, xf, yf - 1), self._grad(bb, xf - 1, yf - 1), u)

        return (self._lerp(x1, x2, v) + 1) / 2

    def fractal_noise(self, x: float, y: float, octaves: int = 4) -> float:
        """Generate fractal noise with multiple octaves."""
        value = 0
        amplitude = 1
        frequency = 1
        max_value = 0

        for _ in range(octaves):
            value += amplitude * self.noise2d(x * frequency, y * frequency)
            max_value += amplitude
            amplitude *= 0.5
            frequency *= 2

        return value / max_value


@tool_wrapper()
def create_noise_art_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create Perlin/Simplex noise-based generative art.

    Args:
        params: Dictionary containing:
            - output_path (str, optional): Path to save image (default: temp file)
            - width (int, optional): Image width in pixels (default: 800)
            - height (int, optional): Image height in pixels (default: 600)
            - seed (int, optional): Random seed for reproducibility
            - color_palette (str, optional): Color palette name (default: "sunset")
            - noise_scale (float, optional): Scale of noise (default: 0.01)

    Returns:
        Dictionary with:
            - success (bool): Whether generation succeeded
            - image_path (str): Path to generated image
            - error (str, optional): Error message if failed
    """
    status.set_callback(params.pop("_status_callback", None))
    try:
        import numpy as np
        from PIL import Image
    except ImportError as e:
        return {
            "success": False,
            "error": f"Required libraries not installed: {str(e)}. Install with: pip install Pillow numpy",
        }

    width = params.get("width", 800)
    height = params.get("height", 600)
    seed = params.get("seed")
    color_palette = params.get("color_palette", "sunset")
    noise_scale = params.get("noise_scale", 0.01)
    output_path = params.get("output_path")

    try:
        noise_gen = NoiseGenerator(seed)
        palette = ColorPaletteManager.get_palette(color_palette)

        img = Image.new("RGB", (width, height))
        pixels = img.load()

        for y in range(height):
            for x in range(width):
                noise_val = noise_gen.fractal_noise(x * noise_scale, y * noise_scale, octaves=4)
                palette_idx = noise_val * (len(palette) - 1)
                idx1 = int(palette_idx)
                idx2 = min(idx1 + 1, len(palette) - 1)
                t = palette_idx - idx1
                color = ColorPaletteManager.interpolate_color(palette[idx1], palette[idx2], t)
                pixels[x, y] = color

        if output_path is None:
            temp_dir = tempfile.gettempdir()
            output_path = os.path.join(temp_dir, f"noise_art_{os.getpid()}.png")

        os.makedirs(
            os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True
        )
        img.save(output_path)

        logger.info(f"Noise art saved to: {output_path}")

        return {
            "success": True,
            "image_path": output_path,
            "width": width,
            "height": height,
            "seed": noise_gen.seed,
            "color_palette": color_palette,
        }

    except Exception as e:
        logger.error(f"Error creating noise art: {str(e)}", exc_info=True)
        return {"success": False, "error": f"Noise art generation failed: {str(e)}"}


@tool_wrapper()
def create_flow_field_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create flow field visualization art.

    Args:
        params: Dictionary containing:
            - output_path (str, optional): Path to save image (default: temp file)
            - width (int, optional): Image width in pixels (default: 800)
            - height (int, optional): Image height in pixels (default: 600)
            - seed (int, optional): Random seed for reproducibility
            - num_particles (int, optional): Number of particles (default: 5000)
            - color_palette (str, optional): Color palette name (default: "ocean")

    Returns:
        Dictionary with:
            - success (bool): Whether generation succeeded
            - image_path (str): Path to generated image
            - error (str, optional): Error message if failed
    """
    status.set_callback(params.pop("_status_callback", None))
    try:
        import numpy as np
        from PIL import Image, ImageDraw
    except ImportError as e:
        return {
            "success": False,
            "error": f"Required libraries not installed: {str(e)}. Install with: pip install Pillow numpy",
        }

    width = params.get("width", 800)
    height = params.get("height", 600)
    seed = params.get("seed")
    num_particles = params.get("num_particles", 5000)
    color_palette = params.get("color_palette", "ocean")
    output_path = params.get("output_path")

    try:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        noise_gen = NoiseGenerator(seed)
        palette = ColorPaletteManager.get_palette(color_palette)

        img = Image.new("RGB", (width, height), (10, 10, 20))
        draw = ImageDraw.Draw(img, "RGBA")

        noise_scale = 0.005
        step_length = 2
        num_steps = 100

        for _ in range(num_particles):
            x = random.uniform(0, width)
            y = random.uniform(0, height)
            color_idx = random.randint(0, len(palette) - 1)
            base_color = palette[color_idx]

            points = [(x, y)]

            for _ in range(num_steps):
                noise_val = noise_gen.noise2d(x * noise_scale, y * noise_scale)
                angle = noise_val * math.pi * 4

                x += math.cos(angle) * step_length
                y += math.sin(angle) * step_length

                if 0 <= x < width and 0 <= y < height:
                    points.append((x, y))
                else:
                    break

            if len(points) > 1:
                for i in range(len(points) - 1):
                    alpha = int(100 * (1 - i / len(points)))
                    color_with_alpha = base_color + (alpha,)
                    draw.line([points[i], points[i + 1]], fill=color_with_alpha, width=1)

        if output_path is None:
            temp_dir = tempfile.gettempdir()
            output_path = os.path.join(temp_dir, f"flow_field_{os.getpid()}.png")

        os.makedirs(
            os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True
        )
        img.save(output_path)

        logger.info(f"Flow field art saved to: {output_path}")

        return {
            "success": True,
            "image_path": output_path,
            "width": width,
            "height": height,
            "seed": seed,
            "num_particles": num_particles,
            "color_palette": color_palette,
        }

    except Exception as e:
        logger.error(f"Error creating flow field: {str(e)}", exc_info=True)
        return {"success": False, "error": f"Flow field generation failed: {str(e)}"}


@tool_wrapper()
def create_geometric_pattern_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create geometric pattern art.

    Args:
        params: Dictionary containing:
            - output_path (str, optional): Path to save image (default: temp file)
            - width (int, optional): Image width in pixels (default: 800)
            - height (int, optional): Image height in pixels (default: 600)
            - pattern (str, optional): Pattern type - circles, triangles, hexagons, voronoi (default: "circles")
            - colors (list, optional): List of RGB tuples or palette name (default: "pastel")

    Returns:
        Dictionary with:
            - success (bool): Whether generation succeeded
            - image_path (str): Path to generated image
            - error (str, optional): Error message if failed
    """
    try:
        import numpy as np
        from PIL import Image, ImageDraw
    except ImportError as e:
        return {
            "success": False,
            "error": f"Required libraries not installed: {str(e)}. Install with: pip install Pillow numpy",
        }

    width = params.get("width", 800)
    height = params.get("height", 600)
    pattern = params.get("pattern", "circles").lower()
    colors = params.get("colors", "pastel")
    output_path = params.get("output_path")
    seed = params.get("seed")

    try:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        if isinstance(colors, str):
            palette = ColorPaletteManager.get_palette(colors)
        else:
            palette = [tuple(c) for c in colors]

        img = Image.new("RGB", (width, height), (255, 255, 255))
        draw = ImageDraw.Draw(img)

        if pattern == "circles":
            cell_size = 50
            for y in range(0, height + cell_size, cell_size):
                for x in range(0, width + cell_size, cell_size):
                    radius = random.randint(10, cell_size // 2)
                    color = random.choice(palette)
                    cx = x + random.randint(-10, 10)
                    cy = y + random.randint(-10, 10)
                    draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=color)

        elif pattern == "triangles":
            cell_size = 60
            for y in range(0, height + cell_size, cell_size):
                for x in range(0, width + cell_size, cell_size):
                    offset = (cell_size // 2) if (y // cell_size) % 2 == 1 else 0
                    cx = x + offset
                    cy = y
                    color = random.choice(palette)
                    size = cell_size // 2
                    points = [(cx, cy - size), (cx - size, cy + size), (cx + size, cy + size)]
                    if random.random() > 0.5:
                        points = [(cx, cy + size), (cx - size, cy - size), (cx + size, cy - size)]
                    draw.polygon(points, fill=color)

        elif pattern == "hexagons":
            hex_size = 40
            hex_height = hex_size * math.sqrt(3)
            for row in range(int(height / hex_height) + 2):
                for col in range(int(width / (hex_size * 1.5)) + 2):
                    cx = col * hex_size * 1.5
                    cy = row * hex_height + (hex_height / 2 if col % 2 == 1 else 0)
                    color = random.choice(palette)
                    points = []
                    for i in range(6):
                        angle = math.pi / 3 * i
                        px = cx + hex_size * math.cos(angle)
                        py = cy + hex_size * math.sin(angle)
                        points.append((px, py))
                    draw.polygon(points, fill=color, outline=(50, 50, 50))

        elif pattern == "voronoi":
            num_points = 50
            points = [
                (random.randint(0, width), random.randint(0, height)) for _ in range(num_points)
            ]
            point_colors = [random.choice(palette) for _ in range(num_points)]

            pixels = img.load()
            for y in range(height):
                for x in range(width):
                    min_dist = float("inf")
                    closest_idx = 0
                    for i, (px, py) in enumerate(points):
                        dist = (x - px) ** 2 + (y - py) ** 2
                        if dist < min_dist:
                            min_dist = dist
                            closest_idx = i
                    pixels[x, y] = point_colors[closest_idx]

        else:
            return {
                "success": False,
                "error": f"Unknown pattern type: {pattern}. Supported: circles, triangles, hexagons, voronoi",
            }

        if output_path is None:
            temp_dir = tempfile.gettempdir()
            output_path = os.path.join(temp_dir, f"geometric_{pattern}_{os.getpid()}.png")

        os.makedirs(
            os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True
        )
        img.save(output_path)

        logger.info(f"Geometric pattern saved to: {output_path}")

        return {
            "success": True,
            "image_path": output_path,
            "width": width,
            "height": height,
            "pattern": pattern,
        }

    except Exception as e:
        logger.error(f"Error creating geometric pattern: {str(e)}", exc_info=True)
        return {"success": False, "error": f"Geometric pattern generation failed: {str(e)}"}


@tool_wrapper()
def create_fractal_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create fractal images (Mandelbrot, Julia sets).

    Args:
        params: Dictionary containing:
            - output_path (str, optional): Path to save image (default: temp file)
            - fractal_type (str, optional): Type of fractal - mandelbrot, julia (default: "mandelbrot")
            - width (int, optional): Image width in pixels (default: 800)
            - height (int, optional): Image height in pixels (default: 600)
            - iterations (int, optional): Max iterations for escape calculation (default: 100)
            - color_map (str, optional): Color palette name (default: "cosmic")

    Returns:
        Dictionary with:
            - success (bool): Whether generation succeeded
            - image_path (str): Path to generated image
            - error (str, optional): Error message if failed
    """
    try:
        import numpy as np
        from PIL import Image
    except ImportError as e:
        return {
            "success": False,
            "error": f"Required libraries not installed: {str(e)}. Install with: pip install Pillow numpy",
        }

    width = params.get("width", 800)
    height = params.get("height", 600)
    fractal_type = params.get("fractal_type", "mandelbrot").lower()
    iterations = params.get("iterations", 100)
    color_map = params.get("color_map", "cosmic")
    output_path = params.get("output_path")

    try:
        palette = ColorPaletteManager.get_palette(color_map)

        img = Image.new("RGB", (width, height))
        pixels = img.load()

        if fractal_type == "mandelbrot":
            x_min, x_max = -2.5, 1.0
            y_min, y_max = -1.25, 1.25

            for py in range(height):
                for px in range(width):
                    x0 = x_min + (x_max - x_min) * px / width
                    y0 = y_min + (y_max - y_min) * py / height

                    x, y = 0, 0
                    iteration = 0

                    while x * x + y * y <= 4 and iteration < iterations:
                        xtemp = x * x - y * y + x0
                        y = 2 * x * y + y0
                        x = xtemp
                        iteration += 1

                    if iteration == iterations:
                        pixels[px, py] = (0, 0, 0)
                    else:
                        t = iteration / iterations
                        palette_idx = t * (len(palette) - 1)
                        idx1 = int(palette_idx)
                        idx2 = min(idx1 + 1, len(palette) - 1)
                        frac = palette_idx - idx1
                        color = ColorPaletteManager.interpolate_color(
                            palette[idx1], palette[idx2], frac
                        )
                        pixels[px, py] = color

        elif fractal_type == "julia":
            c_real = params.get("c_real", -0.7)
            c_imag = params.get("c_imag", 0.27015)

            x_min, x_max = -1.5, 1.5
            y_min, y_max = -1.5, 1.5

            for py in range(height):
                for px in range(width):
                    x = x_min + (x_max - x_min) * px / width
                    y = y_min + (y_max - y_min) * py / height

                    iteration = 0

                    while x * x + y * y <= 4 and iteration < iterations:
                        xtemp = x * x - y * y + c_real
                        y = 2 * x * y + c_imag
                        x = xtemp
                        iteration += 1

                    if iteration == iterations:
                        pixels[px, py] = (0, 0, 0)
                    else:
                        t = iteration / iterations
                        palette_idx = t * (len(palette) - 1)
                        idx1 = int(palette_idx)
                        idx2 = min(idx1 + 1, len(palette) - 1)
                        frac = palette_idx - idx1
                        color = ColorPaletteManager.interpolate_color(
                            palette[idx1], palette[idx2], frac
                        )
                        pixels[px, py] = color

        else:
            return {
                "success": False,
                "error": f"Unknown fractal type: {fractal_type}. Supported: mandelbrot, julia",
            }

        if output_path is None:
            temp_dir = tempfile.gettempdir()
            output_path = os.path.join(temp_dir, f"fractal_{fractal_type}_{os.getpid()}.png")

        os.makedirs(
            os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True
        )
        img.save(output_path)

        logger.info(f"Fractal saved to: {output_path}")

        return {
            "success": True,
            "image_path": output_path,
            "width": width,
            "height": height,
            "fractal_type": fractal_type,
            "iterations": iterations,
        }

    except Exception as e:
        logger.error(f"Error creating fractal: {str(e)}", exc_info=True)
        return {"success": False, "error": f"Fractal generation failed: {str(e)}"}


@tool_wrapper()
def create_generative_landscape_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create generative landscape art.

    Args:
        params: Dictionary containing:
            - output_path (str, optional): Path to save image (default: temp file)
            - width (int, optional): Image width in pixels (default: 800)
            - height (int, optional): Image height in pixels (default: 600)
            - seed (int, optional): Random seed for reproducibility
            - style (str, optional): Landscape style - mountains, waves, terrain (default: "mountains")

    Returns:
        Dictionary with:
            - success (bool): Whether generation succeeded
            - image_path (str): Path to generated image
            - error (str, optional): Error message if failed
    """
    try:
        import numpy as np
        from PIL import Image, ImageDraw
    except ImportError as e:
        return {
            "success": False,
            "error": f"Required libraries not installed: {str(e)}. Install with: pip install Pillow numpy",
        }

    width = params.get("width", 800)
    height = params.get("height", 600)
    seed = params.get("seed")
    style = params.get("style", "mountains").lower()
    output_path = params.get("output_path")

    try:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        noise_gen = NoiseGenerator(seed)

        img = Image.new("RGB", (width, height))
        draw = ImageDraw.Draw(img)

        if style == "mountains":
            sky_top = (25, 25, 112)
            sky_bottom = (255, 140, 0)
            for y in range(height // 2):
                t = y / (height // 2)
                color = ColorPaletteManager.interpolate_color(sky_top, sky_bottom, t)
                draw.line([(0, y), (width, y)], fill=color)

            sun_x = width * 0.7
            sun_y = height * 0.35
            sun_radius = 40
            draw.ellipse(
                [sun_x - sun_radius, sun_y - sun_radius, sun_x + sun_radius, sun_y + sun_radius],
                fill=(255, 200, 100),
            )

            mountain_colors = [(30, 30, 50), (50, 50, 80), (70, 70, 100), (90, 90, 120)]

            for layer, color in enumerate(mountain_colors):
                base_y = height // 2 + layer * 30
                amplitude = 100 - layer * 15
                frequency = 0.003 + layer * 0.001

                points = [(0, height)]
                for x in range(width + 1):
                    noise_val = noise_gen.fractal_noise(x * frequency, layer * 100, octaves=4)
                    y = base_y - amplitude * noise_val
                    points.append((x, y))
                points.append((width, height))

                draw.polygon(points, fill=color)

        elif style == "waves":
            for y in range(height):
                t = y / height
                color = ColorPaletteManager.interpolate_color((0, 50, 100), (0, 100, 150), t)
                draw.line([(0, y), (width, y)], fill=color)

            wave_colors = [(255, 255, 255, 100), (200, 220, 255, 80), (150, 200, 255, 60)]

            draw_alpha = ImageDraw.Draw(img, "RGBA")

            for layer, color in enumerate(wave_colors):
                base_y = height * 0.4 + layer * height * 0.15
                amplitude = 30 + layer * 10
                frequency = 0.01 - layer * 0.002

                for x in range(width):
                    for offset in range(5):
                        noise_val = noise_gen.noise2d(
                            x * frequency + offset * 0.1, layer + offset * 0.05
                        )
                        y = (
                            base_y
                            + amplitude * math.sin(noise_val * math.pi * 4 + x * 0.02)
                            + offset * 3
                        )
                        alpha = max(0, color[3] - offset * 20)
                        draw_alpha.point((x, int(y)), fill=(color[0], color[1], color[2], alpha))

        elif style == "terrain":
            for y in range(height):
                for x in range(width):
                    noise_val = noise_gen.fractal_noise(x * 0.005, y * 0.005, octaves=6)

                    if noise_val < 0.3:
                        color = ColorPaletteManager.interpolate_color(
                            (0, 50, 150), (0, 100, 200), noise_val / 0.3
                        )
                    elif noise_val < 0.4:
                        color = (238, 214, 175)
                    elif noise_val < 0.6:
                        color = ColorPaletteManager.interpolate_color(
                            (34, 139, 34), (85, 170, 85), (noise_val - 0.4) / 0.2
                        )
                    elif noise_val < 0.8:
                        color = ColorPaletteManager.interpolate_color(
                            (85, 170, 85), (139, 90, 43), (noise_val - 0.6) / 0.2
                        )
                    else:
                        color = ColorPaletteManager.interpolate_color(
                            (139, 90, 43), (255, 255, 255), (noise_val - 0.8) / 0.2
                        )

                    img.putpixel((x, y), color)

        else:
            return {
                "success": False,
                "error": f"Unknown style: {style}. Supported: mountains, waves, terrain",
            }

        if output_path is None:
            temp_dir = tempfile.gettempdir()
            output_path = os.path.join(temp_dir, f"landscape_{style}_{os.getpid()}.png")

        os.makedirs(
            os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True
        )
        img.save(output_path)

        logger.info(f"Landscape saved to: {output_path}")

        return {
            "success": True,
            "image_path": output_path,
            "width": width,
            "height": height,
            "style": style,
            "seed": seed,
        }

    except Exception as e:
        logger.error(f"Error creating landscape: {str(e)}", exc_info=True)
        return {"success": False, "error": f"Landscape generation failed: {str(e)}"}


@tool_wrapper()
def apply_artistic_filter_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply artistic effects to an existing image.

    Args:
        params: Dictionary containing:
            - image_path (str, required): Path to input image
            - output_path (str, optional): Path to save filtered image (default: temp file)
            - filter (str, optional): Filter type - pointillism, mosaic, sketch, watercolor (default: "pointillism")

    Returns:
        Dictionary with:
            - success (bool): Whether filter was applied successfully
            - image_path (str): Path to filtered image
            - error (str, optional): Error message if failed
    """
    try:
        import numpy as np
        from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
    except ImportError as e:
        return {
            "success": False,
            "error": f"Required libraries not installed: {str(e)}. Install with: pip install Pillow numpy",
        }

    image_path = params.get("image_path")
    if not image_path:
        return {"success": False, "error": "Missing required parameter: image_path"}

    if not os.path.exists(image_path):
        return {"success": False, "error": f"Image not found: {image_path}"}

    filter_type = params.get("filter", "pointillism").lower()
    output_path = params.get("output_path")

    try:
        img = Image.open(image_path).convert("RGB")
        width, height = img.size

        if filter_type == "pointillism":
            output = Image.new("RGB", (width, height), (255, 255, 255))
            draw = ImageDraw.Draw(output)

            dot_size = max(3, min(width, height) // 100)

            for y in range(0, height, dot_size):
                for x in range(0, width, dot_size):
                    region = img.crop((x, y, min(x + dot_size, width), min(y + dot_size, height)))
                    avg_color = tuple(int(c) for c in np.array(region).mean(axis=(0, 1)))

                    cx = x + dot_size // 2 + random.randint(-2, 2)
                    cy = y + dot_size // 2 + random.randint(-2, 2)
                    radius = dot_size // 2 + random.randint(-1, 1)

                    draw.ellipse(
                        [cx - radius, cy - radius, cx + radius, cy + radius], fill=avg_color
                    )

            img = output

        elif filter_type == "mosaic":
            tile_size = max(10, min(width, height) // 30)

            output = Image.new("RGB", (width, height))

            for y in range(0, height, tile_size):
                for x in range(0, width, tile_size):
                    region = img.crop((x, y, min(x + tile_size, width), min(y + tile_size, height)))
                    avg_color = tuple(int(c) for c in np.array(region).mean(axis=(0, 1)))

                    for ty in range(y, min(y + tile_size, height)):
                        for tx in range(x, min(x + tile_size, width)):
                            output.putpixel((tx, ty), avg_color)

            img = output

        elif filter_type == "sketch":
            gray = img.convert("L")
            inverted = Image.eval(gray, lambda x: 255 - x)
            blurred = inverted.filter(ImageFilter.GaussianBlur(radius=21))

            output = Image.new("L", (width, height))
            gray_pixels = gray.load()
            blur_pixels = blurred.load()
            output_pixels = output.load()

            for y in range(height):
                for x in range(width):
                    if blur_pixels[x, y] == 0:
                        output_pixels[x, y] = 255
                    else:
                        output_pixels[x, y] = min(
                            255, int(gray_pixels[x, y] * 256 / (256 - blur_pixels[x, y]))
                        )

            img = output.convert("RGB")

        elif filter_type == "watercolor":
            img = img.filter(ImageFilter.MedianFilter(size=5))
            img = img.filter(ImageFilter.SMOOTH_MORE)

            enhancer = ImageEnhance.Color(img)
            img = enhancer.enhance(1.3)

            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(0.9)

            output = img.copy()
            output_pixels = output.load()
            img_pixels = img.load()

            for y in range(1, height - 1):
                for x in range(1, width - 1):
                    if random.random() < 0.1:
                        neighbor_x = x + random.choice([-1, 0, 1])
                        neighbor_y = y + random.choice([-1, 0, 1])
                        if 0 <= neighbor_x < width and 0 <= neighbor_y < height:
                            c1 = img_pixels[x, y]
                            c2 = img_pixels[neighbor_x, neighbor_y]
                            blended = tuple(int((c1[i] + c2[i]) / 2) for i in range(3))
                            output_pixels[x, y] = blended

            img = output

        else:
            return {
                "success": False,
                "error": f"Unknown filter type: {filter_type}. Supported: pointillism, mosaic, sketch, watercolor",
            }

        if output_path is None:
            temp_dir = tempfile.gettempdir()
            output_path = os.path.join(temp_dir, f"filtered_{filter_type}_{os.getpid()}.png")

        os.makedirs(
            os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True
        )
        img.save(output_path)

        logger.info(f"Filtered image saved to: {output_path}")

        return {
            "success": True,
            "image_path": output_path,
            "filter": filter_type,
            "original_path": image_path,
        }

    except Exception as e:
        logger.error(f"Error applying filter: {str(e)}", exc_info=True)
        return {"success": False, "error": f"Filter application failed: {str(e)}"}


@tool_wrapper()
def list_color_palettes_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    List all available color palettes for generative art.

    Args:
        params: Dictionary (can be empty)

    Returns:
        Dictionary with:
            - success (bool): Always True
            - palettes (list): List of available palette names with colors
    """
    status.set_callback(params.pop("_status_callback", None))

    palettes = []
    for name, colors in ColorPaletteManager.PALETTES.items():
        palettes.append(
            {
                "name": name,
                "colors": colors,
                "hex_colors": ["#{:02x}{:02x}{:02x}".format(r, g, b) for r, g, b in colors],
            }
        )

    return {"success": True, "palettes": palettes}
