"""
Diagram Image Generator using Mermaid.ink
==========================================

Uses Mermaid.ink (FREE, no API key, 100% reliable) to create
professional diagram images for presentations.

This replaces shape-based diagrams with clean, professional Mermaid diagrams.
"""

import asyncio
import base64
import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp

logger = logging.getLogger(__name__)

# Mermaid.ink - 100% free, reliable, no API key required
MERMAID_URL = "https://mermaid.ink/img"

# Color palette for diagrams (matches our PPTX theme)
COLORS = {
    "primary": "#1e3a5f",  # Navy
    "secondary": "#2563eb",  # Blue
    "success": "#059669",  # Green
    "warning": "#ea580c",  # Orange
    "purple": "#7c3aed",  # Purple
    "text": "#fff",  # White text
    "light": "#60a5fa",  # Light blue
}


class MermaidDiagramGenerator:
    """Generate diagram images using Mermaid.ink (FREE, reliable)."""

    def __init__(self, cache_dir: str = "/tmp/diagram_cache") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, content: str) -> Path:
        """Get cache file path for mermaid content."""
        content_hash = hashlib.md5(content.encode()).hexdigest()[:12]
        return self.cache_dir / f"mermaid_{content_hash}.jpg"

    async def generate_diagram(
        self, mermaid_code: str, width: int = 1200, use_cache: bool = True
    ) -> Optional[str]:
        """
        Generate a diagram image from Mermaid code.

        Args:
            mermaid_code: Mermaid diagram definition
            width: Image width
            use_cache: Whether to use cached images

        Returns:
            Path to the generated image file, or None if failed
        """
        # Check cache first
        cache_path = self._get_cache_path(mermaid_code)
        if use_cache and cache_path.exists():
            logger.info(f"Using cached diagram: {cache_path}")
            return str(cache_path)

        try:
            # Base64 encode the mermaid code
            encoded = base64.urlsafe_b64encode(mermaid_code.encode()).decode()
            url = f"{MERMAID_URL}/{encoded}?bgColor=white&width={width}"

            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=60)) as response:
                    if response.status != 200:
                        logger.error(f"Mermaid.ink error: {response.status}")
                        return None

                    # Read image bytes
                    image_bytes = await response.read()

                    if len(image_bytes) < 1000:
                        logger.error("Received invalid image (too small)")
                        return None

                    # Save to cache
                    with open(cache_path, "wb") as f:
                        f.write(image_bytes)

                    logger.info(
                        f"Generated Mermaid diagram: {cache_path} ({len(image_bytes)} bytes)"
                    )
                    return str(cache_path)

        except asyncio.TimeoutError:
            logger.error("Mermaid diagram generation timed out")
            return None
        except Exception as e:
            logger.error(f"Mermaid diagram generation failed: {e}")
            return None


class DiagramImageGenerator:
    """
    Generate professional diagram images for presentations.

    Converts LIDA-style specs into Mermaid diagrams, then renders as images.
    100% FREE, reliable, no API key needed.
    """

    def __init__(self, api_key: str = None, cache_dir: str = "/tmp/diagram_cache") -> None:
        self.mermaid = MermaidDiagramGenerator(cache_dir)

    def _create_architecture_mermaid(self, spec: Dict[str, Any], paper_title: str = "") -> str:
        """Create Mermaid code for architecture diagram."""
        nodes = spec.get("nodes", [])

        # Group nodes by row
        rows = {}
        for node in nodes:
            row = node.get("row", 0)
            if row not in rows:
                rows[row] = []
            node_id = node.get("id", f"n{len(rows[row])}")
            label = node.get("label", node_id)
            rows[row].append((node_id, label))

        # Build mermaid flowchart
        lines = ["graph TD"]

        # Add nodes with styles
        color_cycle = ["primary", "secondary", "success", "purple", "warning"]
        node_colors = {}

        for row_idx in sorted(rows.keys()):
            for i, (node_id, label) in enumerate(rows[row_idx]):
                # Clean label for mermaid
                clean_label = label.replace('"', "'").replace("[", "(").replace("]", ")")[:30]
                lines.append(f'    {node_id}["{clean_label}"]')
                node_colors[node_id] = color_cycle[(row_idx + i) % len(color_cycle)]

        # Add connections between rows
        sorted_rows = sorted(rows.keys())
        for i in range(len(sorted_rows) - 1):
            current_row = rows[sorted_rows[i]]
            next_row = rows[sorted_rows[i + 1]]
            # Connect each node to all in next row (or just center)
            for src_id, _ in current_row:
                for dst_id, _ in next_row[:2]:  # Limit connections
                    lines.append(f"    {src_id} --> {dst_id}")

        # Add styles
        lines.append("")
        for node_id, color_name in node_colors.items():
            color = COLORS.get(color_name, COLORS["primary"])
            lines.append(f'    style {node_id} fill:{color},color:{COLORS["text"]}')

        return "\n".join(lines)

    def _create_comparison_mermaid(self, spec: Dict[str, Any]) -> str:
        """Create Mermaid code for comparison diagram."""
        left_title = spec.get("left_title", "Before")
        right_title = spec.get("right_title", "After")
        left_items = spec.get("left_items", [])
        right_items = spec.get("right_items", [])

        lines = ["graph LR"]

        # Left side
        lines.append(f'    subgraph LEFT["{left_title}"]')
        for i, item in enumerate(left_items[:4]):
            text = item.get("point", item) if isinstance(item, dict) else item
            text = str(text).replace('"', "'")[:40]
            lines.append(f' L{i}[" {text}"]')
        lines.append("    end")

        # Right side
        lines.append(f'    subgraph RIGHT["{right_title}"]')
        for i, item in enumerate(right_items[:4]):
            text = item.get("point", item) if isinstance(item, dict) else item
            text = str(text).replace('"', "'")[:40]
            lines.append(f' R{i}[" {text}"]')
        lines.append("    end")

        # VS connection
        lines.append("    LEFT -.->|VS| RIGHT")

        # Styles
        lines.append("")
        lines.append(f'    style LEFT fill:#fff5f5,stroke:{COLORS["warning"]}')
        lines.append(f'    style RIGHT fill:#f0fff4,stroke:{COLORS["success"]}')
        for i in range(len(left_items[:4])):
            lines.append(f'    style L{i} fill:#fef3e2,stroke:{COLORS["warning"]}')
        for i in range(len(right_items[:4])):
            lines.append(f'    style R{i} fill:#e6f7f1,stroke:{COLORS["success"]}')

        return "\n".join(lines)

    def _create_concept_map_mermaid(self, spec: Dict[str, Any]) -> str:
        """Create Mermaid code for concept map."""
        center = spec.get("central_concept", "Main Concept")
        related = spec.get("related_concepts", [])

        center_clean = str(center).replace('"', "'")[:25]
        lines = ["graph TD"]
        lines.append(f'    CENTER(("{center_clean}"))')

        color_cycle = ["success", "purple", "warning", "light"]
        for i, concept in enumerate(related[:6]):
            label = concept.get("label", concept) if isinstance(concept, dict) else concept
            label = str(label).replace('"', "'")[:20]
            lines.append(f'    C{i}["{label}"]')
            lines.append(f"    CENTER --- C{i}")

        # Styles
        lines.append("")
        lines.append(
            f'    style CENTER fill:{COLORS["primary"]},color:{COLORS["text"]},stroke-width:3px'
        )
        for i in range(len(related[:6])):
            color = COLORS[color_cycle[i % len(color_cycle)]]
            lines.append(f'    style C{i} fill:{color},color:{COLORS["text"]}')

        return "\n".join(lines)

    def _create_flow_mermaid(self, spec: Dict[str, Any], paper_title: str = "") -> str:
        """Create Mermaid code for flow diagram."""
        nodes = spec.get("nodes", [])

        lines = ["graph LR"]

        color_cycle = ["primary", "secondary", "success", "purple", "warning"]
        for i, node in enumerate(nodes[:5]):
            label = node.get("label", f"Step {i+1}")
            label = str(label).replace('"', "'")[:25]
            lines.append(f'    S{i}["{i+1}. {label}"]')
            if i > 0:
                lines.append(f"    S{i-1} --> S{i}")

        # Styles
        lines.append("")
        for i in range(len(nodes[:5])):
            color = COLORS[color_cycle[i % len(color_cycle)]]
            lines.append(f'    style S{i} fill:{color},color:{COLORS["text"]}')

        return "\n".join(lines)

    def _create_metrics_mermaid(self, spec: Dict[str, Any]) -> str:
        """Create Mermaid code for metrics display."""
        metrics = spec.get("metrics", [])

        lines = ["graph LR"]

        color_cycle = ["primary", "success", "purple", "warning"]
        for i, metric in enumerate(metrics[:4]):
            value = metric.get("value", "?")
            label = metric.get("label", "Metric")
            lines.append(f'    M{i}["{value}<br/>{label}"]')

        # Styles
        lines.append("")
        for i in range(len(metrics[:4])):
            color = COLORS[color_cycle[i % len(color_cycle)]]
            lines.append(f'    style M{i} fill:{color},color:{COLORS["text"]},stroke-width:2px')

        return "\n".join(lines)

    async def generate_architecture_diagram(
        self, spec: Dict[str, Any], paper_title: str = ""
    ) -> Optional[str]:
        """Generate an architecture diagram image."""
        mermaid_code = self._create_architecture_mermaid(spec, paper_title)
        logger.debug(f"Architecture mermaid:\n{mermaid_code}")
        return await self.mermaid.generate_diagram(mermaid_code)

    async def generate_comparison_diagram(self, spec: Dict[str, Any]) -> Optional[str]:
        """Generate a comparison diagram image."""
        mermaid_code = self._create_comparison_mermaid(spec)
        logger.debug(f"Comparison mermaid:\n{mermaid_code}")
        return await self.mermaid.generate_diagram(mermaid_code)

    async def generate_concept_map(self, spec: Dict[str, Any]) -> Optional[str]:
        """Generate a concept map image."""
        mermaid_code = self._create_concept_map_mermaid(spec)
        logger.debug(f"Concept map mermaid:\n{mermaid_code}")
        return await self.mermaid.generate_diagram(mermaid_code)

    async def generate_flow_diagram(
        self, spec: Dict[str, Any], paper_title: str = ""
    ) -> Optional[str]:
        """Generate a flow diagram image."""
        mermaid_code = self._create_flow_mermaid(spec, paper_title)
        logger.debug(f"Flow mermaid:\n{mermaid_code}")
        return await self.mermaid.generate_diagram(mermaid_code)

    async def generate_metrics_display(self, spec: Dict[str, Any]) -> Optional[str]:
        """Generate a metrics display image."""
        mermaid_code = self._create_metrics_mermaid(spec)
        logger.debug(f"Metrics mermaid:\n{mermaid_code}")
        return await self.mermaid.generate_diagram(mermaid_code)

    def _auto_generate_concept_map_spec(
        self, paper_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Auto-generate concept map spec from paper concepts."""
        concepts = paper_data.get("concepts", [])
        if len(concepts) < 2:
            return None

        # Use first concept as center, rest as related
        return {
            "central_concept": concepts[0].get("name", "Main Concept"),
            "related_concepts": [c.get("name", "") for c in concepts[1:5]],
        }

    def _auto_generate_flow_spec(self, paper_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Auto-generate flow spec from paper sections or concepts."""
        sections = paper_data.get("sections", [])
        concepts = paper_data.get("concepts", [])

        # Try sections first
        if len(sections) >= 3:
            return {
                "nodes": [
                    {"label": s.get("title", f"Step {i+1}")[:20]}
                    for i, s in enumerate(sections[:5])
                ]
            }

        # Fallback to concepts
        if len(concepts) >= 3:
            return {
                "nodes": [
                    {"label": c.get("name", f"Step {i+1}")[:20]} for i, c in enumerate(concepts[:5])
                ]
            }

        return None

    async def generate_all_diagrams(
        self, paper_data: Dict[str, Any], visualization_specs: Dict[str, Any]
    ) -> Dict[str, str]:
        """
        Generate Mermaid diagram images for a presentation.

        NOTE: Only generates FLOW diagrams as Mermaid images.
        Other diagram types (architecture, concept_map, comparison, metrics)
        use shape-based rendering which provides better control over layout.

        Args:
            paper_data: Paper content (title, concepts, etc.)
            visualization_specs: LIDA-style visualization specs

        Returns:
            Dict mapping diagram type to image path (currently only 'flow')
        """
        paper_title = paper_data.get("paper_title", "")
        results = {}

        # Only generate FLOW diagram as Mermaid image
        # Mermaid excels at horizontal left-to-right flowcharts
        # Other diagram types use shape-based rendering for better control
        flow_spec = visualization_specs.get("flow")
        if not flow_spec:
            flow_spec = self._auto_generate_flow_spec(paper_data)

        if flow_spec:
            logger.info(" Generating flow diagram (Mermaid)...")
            path = await self.generate_flow_diagram(flow_spec, paper_title)
            if path:
                results["flow"] = path

        if results:
            logger.info(f" Generated {len(results)} Mermaid diagram image(s)")
        return results


# Convenience functions for testing
async def test_mermaid_diagram() -> Any:
    """Test Mermaid.ink diagram generation."""
    generator = MermaidDiagramGenerator()

    mermaid_code = """graph TD
    A[Input] --> B[Process]
    B --> C[Output]
    style A fill:#1e3a5f,color:#fff
    style B fill:#2563eb,color:#fff
    style C fill:#059669,color:#fff"""

    result = await generator.generate_diagram(mermaid_code, use_cache=False)
    if result:
        logger.info(f"Mermaid diagram generated: {result}")
    else:
        logger.error("Mermaid diagram failed")
    return result


async def test_architecture_diagram() -> Any:
    """Test full architecture diagram generation."""
    generator = DiagramImageGenerator()

    spec = {
        "title": "Transformer Architecture",
        "nodes": [
            {"id": "input", "label": "Input Embedding", "row": 0},
            {"id": "pos", "label": "Positional Encoding", "row": 0},
            {"id": "encoder", "label": "Encoder Stack", "row": 1},
            {"id": "decoder", "label": "Decoder Stack", "row": 1},
            {"id": "attention", "label": "Multi-Head Attention", "row": 2},
            {"id": "ffn", "label": "Feed Forward", "row": 2},
            {"id": "output", "label": "Output Softmax", "row": 3},
        ],
    }

    result = await generator.generate_architecture_diagram(spec, "Transformer")
    if result:
        logger.info(f"Architecture diagram generated: {result}")
    else:
        logger.error("Architecture diagram failed")
    return result


async def test_all_diagrams() -> Any:
    """Test all diagram types."""
    generator = DiagramImageGenerator()

    # Test architecture
    logger.info("1. Testing architecture diagram...")
    arch = await generator.generate_architecture_diagram(
        {
            "nodes": [
                {"id": "a", "label": "Input", "row": 0},
                {"id": "b", "label": "Encoder", "row": 1},
                {"id": "c", "label": "Decoder", "row": 1},
                {"id": "d", "label": "Output", "row": 2},
            ]
        }
    )
    logger.info(f"   Architecture: {'OK' if arch else 'FAILED'} - {arch}")

    # Test comparison
    logger.info("2. Testing comparison diagram...")
    comp = await generator.generate_comparison_diagram(
        {
            "left_title": "RNN",
            "right_title": "Transformer",
            "left_items": ["Sequential", "Slow", "Limited context"],
            "right_items": ["Parallel", "Fast", "Global context"],
        }
    )
    logger.info(f"   Comparison: {'OK' if comp else 'FAILED'} - {comp}")

    # Test concept map
    logger.info("3. Testing concept map...")
    cmap = await generator.generate_concept_map(
        {
            "central_concept": "Attention",
            "related_concepts": ["Query", "Key", "Value", "Softmax"],
        }
    )
    logger.info(f"   Concept map: {'OK' if cmap else 'FAILED'} - {cmap}")

    # Test flow
    logger.info("4. Testing flow diagram...")
    flow = await generator.generate_flow_diagram(
        {
            "nodes": [
                {"label": "Input"},
                {"label": "Embed"},
                {"label": "Attend"},
                {"label": "Output"},
            ]
        }
    )
    logger.info(f"   Flow: {'OK' if flow else 'FAILED'} - {flow}")

    return arch, comp, cmap, flow
