"""
Chart Renderers

Output renderers for different visualization formats and contexts.
Supports: HTML, SVG, PNG, interactive widgets, and more.
"""
from typing import Dict, Any, Optional, List, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
import base64

logger = logging.getLogger(__name__)


@dataclass
class RenderResult:
    """Result of rendering a chart."""
    success: bool
    output: Any = None  # Rendered content
    format: str = None  # Output format (html, svg, png, etc.)
    error: str = None
    metadata: Dict[str, Any] = None


class ChartRenderer(ABC):
    """
    Abstract base class for chart renderers.

    Renderers convert ChartResult objects into different output formats.
    """

    @abstractmethod
    def render(self, chart: Any, **kwargs: Any) -> RenderResult:
        """
        Render a chart to the target format.

        Args:
            chart: ChartResult object
            **kwargs: Renderer-specific options

        Returns:
            RenderResult with rendered content
        """
        pass

    @abstractmethod
    def render_multiple(self, charts: List, **kwargs: Any) -> RenderResult:
        """
        Render multiple charts (for dashboards).

        Args:
            charts: List of ChartResult objects
            **kwargs: Renderer-specific options

        Returns:
            RenderResult with combined content
        """
        pass


class HTMLRenderer(ChartRenderer):
    """
    Renders charts to HTML for web display.

    Supports:
    - Single chart HTML
    - Dashboard layout with grid
    - Inline base64 images
    - Interactive charts (Altair, Plotly)
    """

    def __init__(self, title: str = 'Visualization', theme: str = 'light', include_code: bool = False) -> None:
        """
        Initialize HTML renderer.

        Args:
            title: Page/section title
            theme: Color theme (light/dark)
            include_code: Include chart code in output
        """
        self.title = title
        self.theme = theme
        self.include_code = include_code

    def render(self, chart: Any, **kwargs: Any) -> RenderResult:
        """Render single chart to HTML."""
        try:
            html = self._chart_to_html(chart)
            return RenderResult(
                success=True,
                output=html,
                format='html'
            )
        except Exception as e:
            logger.error(f"HTML render failed: {e}")
            return RenderResult(success=False, error=str(e))

    def render_multiple(self, charts: List, columns: int = 2, **kwargs: Any) -> RenderResult:
        """
        Render multiple charts as HTML dashboard.

        Args:
            charts: List of ChartResult objects
            columns: Number of columns in grid
            **kwargs: Additional options

        Returns:
            RenderResult with dashboard HTML
        """
        try:
            chart_htmls = [self._chart_to_html(c) for c in charts if c.success]

            # Build grid layout
            grid_html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>{self.title}</title>
                <style>
                    body {{
                        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                        background: {'#1a1a1a' if self.theme == 'dark' else '#ffffff'};
                        color: {'#ffffff' if self.theme == 'dark' else '#333333'};
                        padding: 20px;
                        margin: 0;
                    }}
                    .dashboard {{
                        display: grid;
                        grid-template-columns: repeat({columns}, 1fr);
                        gap: 20px;
                        max-width: 1400px;
                        margin: 0 auto;
                    }}
                    .chart-card {{
                        background: {'#2d2d2d' if self.theme == 'dark' else '#f8f9fa'};
                        border-radius: 8px;
                        padding: 16px;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    }}
                    .chart-title {{
                        font-size: 14px;
                        font-weight: 600;
                        margin-bottom: 12px;
                        color: {'#e0e0e0' if self.theme == 'dark' else '#495057'};
                    }}
                    .chart-image {{
                        width: 100%;
                        height: auto;
                        border-radius: 4px;
                    }}
                    h1 {{
                        text-align: center;
                        margin-bottom: 30px;
                    }}
                    .code-block {{
                        background: {'#1e1e1e' if self.theme == 'dark' else '#f4f4f4'};
                        padding: 10px;
                        border-radius: 4px;
                        font-family: monospace;
                        font-size: 11px;
                        overflow-x: auto;
                        margin-top: 10px;
                        max-height: 200px;
                        overflow-y: auto;
                    }}
                </style>
            </head>
            <body>
                <h1>{self.title}</h1>
                <div class="dashboard">
                    {''.join(f'<div class="chart-card">{html}</div>' for html in chart_htmls)}
                </div>
            </body>
            </html>
            """

            return RenderResult(
                success=True,
                output=grid_html,
                format='html',
                metadata={'chart_count': len(chart_htmls), 'columns': columns}
            )

        except Exception as e:
            logger.error(f"Dashboard render failed: {e}")
            return RenderResult(success=False, error=str(e))

    def _chart_to_html(self, chart: Any) -> str:
        """Convert single chart to HTML snippet."""
        parts = []

        # Title from goal
        if chart.goal and chart.goal.question:
            parts.append(f'<div class="chart-title">{chart.goal.question[:100]}</div>')

        # Image
        if chart.raster:
            # Handle both bytes and already-encoded base64 strings
            if isinstance(chart.raster, bytes):
                b64 = base64.b64encode(chart.raster).decode('utf-8')
            else:
                # Already base64 encoded string
                b64 = chart.raster
            parts.append(f'<img class="chart-image" src="data:image/png;base64,{b64}" />')
        elif chart.svg:
            parts.append(f'<div class="chart-svg">{chart.svg}</div>')
        elif chart.spec:
            # Vega/Altair spec - render as JSON for now
            import json
            parts.append(f'<pre class="code-block">{json.dumps(chart.spec, indent=2)}</pre>')

        # Code
        if self.include_code and chart.code:
            parts.append(f'<pre class="code-block">{chart.code}</pre>')

        return '\n'.join(parts)


class MatplotlibRenderer(ChartRenderer):
    """
    Renders charts using matplotlib for static images.
    """

    def __init__(self, figsize: tuple = (10, 6), dpi: int = 100) -> None:
        """
        Initialize matplotlib renderer.

        Args:
            figsize: Figure size (width, height) in inches
            dpi: Dots per inch for output
        """
        self.figsize = figsize
        self.dpi = dpi

    def render(self, chart: Any, format: str = 'png', **kwargs: Any) -> RenderResult:
        """
        Render chart to image format.

        Args:
            chart: ChartResult object
            format: Output format (png, svg, pdf)
            **kwargs: Additional options

        Returns:
            RenderResult with image bytes
        """
        if chart.raster and format == 'png':
            return RenderResult(success=True, output=chart.raster, format='png')

        if chart.code:
            # Execute code to generate image
            try:
                import matplotlib.pyplot as plt
                import io

                # Execute chart code
                exec(chart.code, {'plt': plt, 'pd': __import__('pandas')})

                # Save to buffer
                buf = io.BytesIO()
                plt.savefig(buf, format=format, dpi=self.dpi, bbox_inches='tight')
                plt.close()
                buf.seek(0)

                return RenderResult(
                    success=True,
                    output=buf.read(),
                    format=format
                )
            except Exception as e:
                logger.error(f"Matplotlib render failed: {e}")
                return RenderResult(success=False, error=str(e))

        return RenderResult(success=False, error="No chart data to render")

    def render_multiple(self, charts: List, format: str = 'png', **kwargs: Any) -> RenderResult:
        """Render multiple charts as subplot grid."""
        try:
            import matplotlib.pyplot as plt
            import io
            import math

            n = len([c for c in charts if c.success])
            cols = min(2, n)
            rows = math.ceil(n / cols)

            fig, axes = plt.subplots(rows, cols, figsize=(self.figsize[0] * cols, self.figsize[1] * rows))

            # Render each chart
            for i, chart in enumerate([c for c in charts if c.success]):
                if chart.raster:
                    from PIL import Image
                    import io as io_module
                    img = Image.open(io_module.BytesIO(chart.raster))
                    ax = axes.flat[i] if n > 1 else axes
                    ax.imshow(img)
                    ax.axis('off')
                    if chart.goal:
                        ax.set_title(chart.goal.question[:50] + '...')

            plt.tight_layout()

            buf = io.BytesIO()
            plt.savefig(buf, format=format, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            buf.seek(0)

            return RenderResult(
                success=True,
                output=buf.read(),
                format=format,
                metadata={'chart_count': n}
            )

        except Exception as e:
            logger.error(f"Multi-chart render failed: {e}")
            return RenderResult(success=False, error=str(e))


class AltairRenderer(ChartRenderer):
    """
    Renders Altair/Vega-Lite charts for interactive web display.
    """

    def render(self, chart: Any, **kwargs: Any) -> RenderResult:
        """Render chart to Vega-Lite spec or HTML."""
        if chart.spec:
            return RenderResult(success=True, output=chart.spec, format='vega-lite')

        if chart.code and 'altair' in chart.code.lower():
            try:
                import altair as alt
                # Execute code to get chart object
                local_vars = {}
                exec(chart.code, {'alt': alt, 'pd': __import__('pandas')}, local_vars)

                # Find the chart object
                for var in local_vars.values():
                    if isinstance(var, alt.Chart):
                        return RenderResult(
                            success=True,
                            output=var.to_dict(),
                            format='vega-lite'
                        )

            except Exception as e:
                logger.error(f"Altair render failed: {e}")
                return RenderResult(success=False, error=str(e))

        return RenderResult(success=False, error="No Altair chart data")

    def render_multiple(self, charts: List, **kwargs: Any) -> RenderResult:
        """Render multiple charts as concatenated Vega-Lite."""
        specs = []
        for chart in charts:
            result = self.render(chart)
            if result.success:
                specs.append(result.output)

        if not specs:
            return RenderResult(success=False, error="No charts to render")

        # Combine as hconcat/vconcat
        combined = {
            '$schema': 'https://vega.github.io/schema/vega-lite/v5.json',
            'vconcat': specs
        }

        return RenderResult(
            success=True,
            output=combined,
            format='vega-lite',
            metadata={'chart_count': len(specs)}
        )


class PlotlyRenderer(ChartRenderer):
    """
    Renders Plotly charts for interactive web display.
    """

    def render(self, chart: Any, **kwargs: Any) -> RenderResult:
        """Render chart to Plotly JSON or HTML."""
        if chart.code and 'plotly' in chart.code.lower():
            try:
                import plotly.graph_objects as go
                import plotly.express as px

                local_vars = {}
                exec(chart.code, {'go': go, 'px': px, 'pd': __import__('pandas')}, local_vars)

                # Find the figure object
                for var in local_vars.values():
                    if isinstance(var, go.Figure):
                        return RenderResult(
                            success=True,
                            output=var.to_json(),
                            format='plotly-json'
                        )

            except Exception as e:
                logger.error(f"Plotly render failed: {e}")
                return RenderResult(success=False, error=str(e))

        return RenderResult(success=False, error="No Plotly chart data")

    def render_multiple(self, charts: List, **kwargs: Any) -> RenderResult:
        """Render multiple Plotly charts."""
        figures = []
        for chart in charts:
            result = self.render(chart)
            if result.success:
                figures.append(result.output)

        return RenderResult(
            success=True,
            output=figures,
            format='plotly-json-array',
            metadata={'chart_count': len(figures)}
        )


class RendererFactory:
    """
    Factory for creating appropriate renderer based on output needs.
    """

    RENDERERS = {
        'html': HTMLRenderer,
        'matplotlib': MatplotlibRenderer,
        'altair': AltairRenderer,
        'plotly': PlotlyRenderer,
    }

    @classmethod
    def create(cls, renderer_type: str = 'html', **kwargs: Any) -> ChartRenderer:
        """
        Create a renderer instance.

        Args:
            renderer_type: Type of renderer (html, matplotlib, altair, plotly)
            **kwargs: Renderer-specific options

        Returns:
            ChartRenderer instance
        """
        renderer_class = cls.RENDERERS.get(renderer_type.lower())
        if not renderer_class:
            raise ValueError(f"Unknown renderer type: {renderer_type}")
        return renderer_class(**kwargs)

    @classmethod
    def get_available(cls) -> List[str]:
        """Get list of available renderer types."""
        return list(cls.RENDERERS.keys())


__all__ = [
    'ChartRenderer',
    'RenderResult',
    'HTMLRenderer',
    'MatplotlibRenderer',
    'AltairRenderer',
    'PlotlyRenderer',
    'RendererFactory',
]
