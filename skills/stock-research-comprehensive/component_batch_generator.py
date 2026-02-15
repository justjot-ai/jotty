"""
Batch Component Generator
=========================

Generates 100+ components automatically by combining:
1. Template patterns from existing components
2. Category-specific variations
3. Animation and style variations

This doesn't require LLM - it uses templates and patterns to generate variations.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class ComponentTemplateGenerator:
    """
    Generates component variations using templates.
    No LLM required - pure pattern-based generation.
    """

    # Color schemes for variations
    COLOR_SCHEMES = {
        "default": {"accent": "#f6ad55", "secondary": "#4299e1"},
        "blue": {"accent": "#4299e1", "secondary": "#9f7aea"},
        "purple": {"accent": "#9f7aea", "secondary": "#ed64a6"},
        "teal": {"accent": "#38b2ac", "secondary": "#4299e1"},
        "green": {"accent": "#48bb78", "secondary": "#38b2ac"},
        "pink": {"accent": "#ed64a6", "secondary": "#9f7aea"},
        "red": {"accent": "#f56565", "secondary": "#ed64a6"},
    }

    # Animation styles
    ANIMATIONS = {
        "slide_up": "animate-slide-up",
        "slide_left": "animate-slide-left",
        "slide_right": "animate-slide-right",
        "fade": "animate-fade",
        "scale": "animate-scale",
        "bounce": "animate-bounce-in",
    }

    # Layout patterns
    LAYOUTS = {
        "grid_2": "grid grid-cols-1 md:grid-cols-2 gap-6",
        "grid_3": "grid grid-cols-1 md:grid-cols-3 gap-6",
        "grid_4": "grid grid-cols-2 md:grid-cols-4 gap-6",
        "flex_row": "flex flex-wrap gap-6",
        "flex_col": "flex flex-col gap-4",
        "split": "grid grid-cols-1 md:grid-cols-2 gap-8",
    }

    def __init__(self) -> None:
        self.generated_count = 0

    def generate_stats_variants(self) -> List[Dict[str, Any]]:
        """Generate 20+ stats component variants"""
        variants = []

        # Base stats configurations
        configs = [
            ("stats_single_large", "Single large stat with description", 1),
            ("stats_pair", "Two stats side by side", 2),
            ("stats_trio", "Three stats in a row", 3),
            ("stats_quad", "Four stats grid", 4),
            ("stats_row_5", "Five stats in a row", 5),
            ("stats_row_6", "Six stats compact", 6),
        ]

        for name, desc, count in configs:
            # Generate with different styles
            for style in ["card", "minimal", "bordered", "gradient"]:
                for show_progress in [True, False]:
                    variant_name = f"{name}_{style}{'_progress' if show_progress else ''}"
                    variants.append(
                        {
                            "name": variant_name,
                            "category": "data",
                            "description": f"{desc} with {style} style",
                            "template": self._stats_template(count, style, show_progress),
                            "data_schema": {
                                "stats": f"Array of {count} stat objects",
                                "title": "Optional title",
                            },
                        }
                    )

        return variants

    def generate_card_variants(self) -> List[Dict[str, Any]]:
        """Generate 30+ card component variants"""
        variants = []

        card_types = [
            ("feature_card", "Feature with icon and description"),
            ("info_card", "Information card with details"),
            ("profile_card", "Person or author profile"),
            ("stat_card", "Statistic with trend"),
            ("action_card", "Card with CTA button"),
            ("image_card", "Card with image header"),
            ("quote_card", "Quote with attribution"),
            ("code_card", "Code snippet card"),
            ("link_card", "External link preview"),
            ("metric_card", "KPI metric display"),
        ]

        for card_name, description in card_types:
            for layout in ["vertical", "horizontal", "compact"]:
                for border_style in ["none", "subtle", "gradient"]:
                    variant_name = f"{card_name}_{layout}_{border_style}"
                    variants.append(
                        {
                            "name": variant_name,
                            "category": "content",
                            "description": f"{description} - {layout} layout with {border_style} border",
                            "template": self._card_template(card_name, layout, border_style),
                        }
                    )

        return variants

    def generate_chart_variants(self) -> List[Dict[str, Any]]:
        """Generate 25+ chart component variants"""
        variants = []

        chart_types = [
            ("bar", ["vertical", "horizontal", "stacked", "grouped"]),
            ("line", ["simple", "area", "multi", "sparkline"]),
            ("pie", ["simple", "donut", "half", "nested"]),
            ("scatter", ["simple", "bubble", "connected"]),
            ("radar", ["simple", "filled", "comparison"]),
        ]

        for chart_type, styles in chart_types:
            for style in styles:
                for size in ["small", "medium", "large"]:
                    variant_name = f"chart_{chart_type}_{style}_{size}"
                    variants.append(
                        {
                            "name": variant_name,
                            "category": "data",
                            "description": f"{chart_type.title()} chart - {style} style, {size} size",
                            "template": self._chart_template(chart_type, style, size),
                        }
                    )

        return variants

    def generate_comparison_variants(self) -> List[Dict[str, Any]]:
        """Generate 20+ comparison component variants"""
        variants = []

        comparison_types = [
            ("before_after", ["split", "overlay", "slider", "tabs"]),
            ("versus", ["cards", "table", "stats"]),
            ("pros_cons", ["split", "list", "cards"]),
            ("old_new", ["timeline", "cards", "split"]),
            ("feature_matrix", ["table", "grid", "cards"]),
        ]

        for comp_type, layouts in comparison_types:
            for layout in layouts:
                for animation in ["slide", "fade", "none"]:
                    variant_name = f"comparison_{comp_type}_{layout}_{animation}"
                    variants.append(
                        {
                            "name": variant_name,
                            "category": "comparison",
                            "description": f"{comp_type.replace('_', ' ').title()} comparison - {layout} layout",
                            "template": self._comparison_template(comp_type, layout),
                        }
                    )

        return variants

    def generate_process_variants(self) -> List[Dict[str, Any]]:
        """Generate 20+ process/steps component variants"""
        variants = []

        process_types = [
            ("steps", ["horizontal", "vertical", "circular", "zigzag"]),
            ("timeline", ["vertical", "horizontal", "alternating"]),
            ("pipeline", ["linear", "branching", "parallel"]),
            ("workflow", ["simple", "detailed", "swimlane"]),
            ("roadmap", ["horizontal", "vertical", "milestones"]),
        ]

        for proc_type, layouts in process_types:
            for layout in layouts:
                for connector in ["line", "arrow", "dotted", "none"]:
                    variant_name = f"process_{proc_type}_{layout}_{connector}"
                    variants.append(
                        {
                            "name": variant_name,
                            "category": "process",
                            "description": f"{proc_type.title()} - {layout} layout with {connector} connectors",
                            "template": self._process_template(proc_type, layout, connector),
                        }
                    )

        return variants

    def generate_code_variants(self) -> List[Dict[str, Any]]:
        """Generate 15+ code component variants"""
        variants = []

        code_types = [
            ("code_block", ["simple", "with_filename", "with_copy", "with_line_numbers"]),
            ("code_diff", ["inline", "split", "unified"]),
            ("terminal", ["simple", "animated", "interactive"]),
            ("api_example", ["request", "response", "full"]),
        ]

        for code_type, styles in code_types:
            for style in styles:
                for theme in ["dark", "light"]:
                    variant_name = f"{code_type}_{style}_{theme}"
                    variants.append(
                        {
                            "name": variant_name,
                            "category": "code",
                            "description": f"{code_type.replace('_', ' ').title()} - {style} with {theme} theme",
                            "template": self._code_template(code_type, style, theme),
                        }
                    )

        return variants

    def generate_diagram_variants(self) -> List[Dict[str, Any]]:
        """Generate 30+ diagram component variants"""
        variants = []

        diagram_types = [
            ("architecture", ["horizontal", "vertical", "layered", "circular"]),
            ("flowchart", ["simple", "decision", "process", "swimlane"]),
            ("network", ["simple", "hierarchical", "circular"]),
            ("tree", ["vertical", "horizontal", "radial"]),
            ("venn", ["2_circles", "3_circles", "4_circles"]),
            ("sequence", ["simple", "detailed", "grouped"]),
        ]

        for diag_type, layouts in diagram_types:
            for layout in layouts:
                for interactive in [True, False]:
                    variant_name = (
                        f"diagram_{diag_type}_{layout}{'_interactive' if interactive else ''}"
                    )
                    variants.append(
                        {
                            "name": variant_name,
                            "category": "visual",
                            "description": f"{diag_type.title()} diagram - {layout} layout",
                            "template": self._diagram_template(diag_type, layout, interactive),
                        }
                    )

        return variants

    def generate_specialized_variants(self) -> List[Dict[str, Any]]:
        """Generate 50+ specialized research component variants"""
        variants = []

        specialized_types = [
            # ML/AI specific
            ("attention_heatmap", ["single_head", "multi_head", "cross_attention"]),
            ("confusion_matrix", ["simple", "normalized", "annotated"]),
            ("loss_curve", ["single", "multi", "with_validation"]),
            ("embedding_space", ["2d", "3d", "tsne"]),
            ("model_architecture", ["cnn", "rnn", "transformer", "custom"]),
            # Data specific
            ("dataset_preview", ["table", "grid", "samples"]),
            ("distribution", ["histogram", "density", "box"]),
            ("correlation", ["matrix", "scatter", "heatmap"]),
            # Results specific
            ("benchmark_table", ["simple", "highlighted", "ranked"]),
            ("ablation_study", ["table", "chart", "cards"]),
            ("experiment_results", ["summary", "detailed", "comparison"]),
        ]

        for spec_type, subtypes in specialized_types:
            for subtype in subtypes:
                variant_name = f"specialized_{spec_type}_{subtype}"
                variants.append(
                    {
                        "name": variant_name,
                        "category": "specialized",
                        "description": f"{spec_type.replace('_', ' ').title()} - {subtype} variant",
                        "template": self._specialized_template(spec_type, subtype),
                    }
                )

        return variants

    def generate_all_variants(self) -> List[Dict[str, Any]]:
        """Generate all component variants"""
        all_variants = []

        generators = [
            ("Stats", self.generate_stats_variants),
            ("Cards", self.generate_card_variants),
            ("Charts", self.generate_chart_variants),
            ("Comparison", self.generate_comparison_variants),
            ("Process", self.generate_process_variants),
            ("Code", self.generate_code_variants),
            ("Diagrams", self.generate_diagram_variants),
            ("Specialized", self.generate_specialized_variants),
        ]

        for name, generator in generators:
            variants = generator()
            logger.info(f"Generated {len(variants)} {name} variants")
            all_variants.extend(variants)

        self.generated_count = len(all_variants)
        return all_variants

    # Template methods (simplified - would be full HTML in production)
    def _stats_template(self, count: int, style: str, progress: bool) -> str:
        return f"Stats template: {count} items, {style} style, progress={progress}"

    def _card_template(self, card_type: str, layout: str, border: str) -> str:
        return f"Card template: {card_type}, {layout} layout, {border} border"

    def _chart_template(self, chart_type: str, style: str, size: str) -> str:
        return f"Chart template: {chart_type}, {style} style, {size} size"

    def _comparison_template(self, comp_type: str, layout: str) -> str:
        return f"Comparison template: {comp_type}, {layout} layout"

    def _process_template(self, proc_type: str, layout: str, connector: str) -> str:
        return f"Process template: {proc_type}, {layout} layout, {connector} connectors"

    def _code_template(self, code_type: str, style: str, theme: str) -> str:
        return f"Code template: {code_type}, {style} style, {theme} theme"

    def _diagram_template(self, diag_type: str, layout: str, interactive: bool) -> str:
        return f"Diagram template: {diag_type}, {layout} layout, interactive={interactive}"

    def _specialized_template(self, spec_type: str, subtype: str) -> str:
        return f"Specialized template: {spec_type}, {subtype} variant"


def generate_component_registry() -> Dict[str, int]:
    """Generate a registry of all available components"""
    generator = ComponentTemplateGenerator()
    all_variants = generator.generate_all_variants()

    # Count by category
    registry = {}
    for variant in all_variants:
        category = variant.get("category", "other")
        registry[category] = registry.get(category, 0) + 1

    return {
        "total": len(all_variants),
        "by_category": registry,
        "components": [v["name"] for v in all_variants],
    }


def save_component_manifest(output_path: str = None) -> Any:
    """Save a manifest of all generated components"""
    import json

    if output_path is None:
        output_path = Path(__file__).parent / "component_manifest.json"

    registry = generate_component_registry()

    with open(output_path, "w") as f:
        json.dump(registry, f, indent=2)

    logger.info(f"Saved manifest with {registry['total']} components to {output_path}")
    return registry


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    logger.info("Generating Component Variants...")
    logger.info("=" * 50)

    registry = save_component_manifest()

    logger.info("Component Summary:")
    logger.info(f"   Total components: {registry['total']}")
    logger.info("   By category:")
    for cat, count in sorted(registry["by_category"].items()):
        logger.info(f"   - {cat}: {count}")

    logger.info(f"Component library now has {registry['total']}+ variants!")
