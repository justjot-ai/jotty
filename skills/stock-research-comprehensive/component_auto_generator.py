"""
Component Auto-Discovery & Generation System
=============================================

Uses LLM to automatically:
1. Analyze existing components and identify gaps
2. Suggest new component types based on research paper patterns
3. Generate component code automatically
4. Test and validate new components
5. Continuously expand the library toward 1000+ components

This creates a self-improving component library that learns from each paper processed.
"""

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import dspy

logger = logging.getLogger(__name__)


def configure_dspy_with_claude_cli() -> bool:
    """Configure DSPy to use Claude CLI as the LM"""
    try:
        from core.integration.direct_claude_cli_lm import DirectClaudeCLI

        lm = DirectClaudeCLI(model="sonnet", max_retries=2)
        dspy.configure(lm=lm)
        logger.info(" DSPy configured with Claude CLI (sonnet)")
        return True
    except Exception as e:
        logger.warning(f" Could not configure Claude CLI: {e}")
        return False


class ComponentCategory(Enum):
    """Categories of slide components"""

    TITLE = "title"
    CONTENT = "content"
    DATA = "data"
    VISUAL = "visual"
    COMPARISON = "comparison"
    PROCESS = "process"
    INTERACTIVE = "interactive"
    CODE = "code"
    MATH = "math"
    MEDIA = "media"
    SOCIAL = "social"
    NAVIGATION = "navigation"
    ENDING = "ending"
    SPECIALIZED = "specialized"


@dataclass
class ComponentSpec:
    """Specification for a slide component"""

    name: str
    category: ComponentCategory
    description: str
    use_cases: List[str]
    data_schema: Dict[str, Any]
    html_template: str
    css_classes: List[str]
    animations: List[str]
    variants: List[str] = field(default_factory=list)
    examples: List[Dict] = field(default_factory=list)


class ComponentSuggestionSignature(dspy.Signature):
    """Suggest new slide components based on research paper content patterns.

    Analyze the paper content and existing components to suggest NEW component types
    that would enhance the presentation. Focus on components that:
    1. Visualize specific concepts from this paper better
    2. Fill gaps in the current component library
    3. Would be reusable across many research papers
    4. Provide unique educational value

    Be creative but practical. Each suggestion should be implementable.
    """

    paper_abstract: str = dspy.InputField(desc="The paper's abstract/summary")
    paper_concepts: str = dspy.InputField(desc="Key concepts from the paper as JSON")
    existing_components: str = dspy.InputField(desc="List of existing component names")
    category_focus: str = dspy.InputField(desc="Category to focus suggestions on, or 'all'")

    suggestions_json: str = dspy.OutputField(
        desc="""JSON array of 5-10 component suggestions. Each: {
        "name": "component_name_snake_case",
        "category": "one of: title, content, data, visual, comparison, process, interactive, code, math, media, ending, specialized",
        "description": "What this component does and why it's valuable",
        "use_cases": ["use case 1", "use case 2"],
        "data_fields": {"field_name": "field_description"},
        "visual_elements": ["element 1", "element 2"],
        "animation_style": "fade/slide/scale/bounce/none",
        "priority": 1-10
    }"""
    )


class ComponentCodeGeneratorSignature(dspy.Signature):
    """Generate the actual HTML/CSS code for a slide component.

    Create production-ready code that:
    1. Follows the existing component styling (navy theme, glass effects, gradients)
    2. Uses Tailwind CSS classes
    3. Includes smooth animations
    4. Is responsive and accessible
    5. Handles edge cases (missing data, long text, etc.)

    The code should be a Python method that returns an HTML string.
    Use the existing patterns from the codebase.
    """

    component_spec: str = dspy.InputField(desc="The component specification as JSON")
    existing_component_code: str = dspy.InputField(
        desc="Example of an existing component method for reference"
    )
    style_guide: str = dspy.InputField(desc="CSS color variables and style guide")

    python_method_code: str = dspy.OutputField(
        desc="Complete Python method code that renders the component HTML. Include docstring."
    )
    example_data: str = dspy.OutputField(desc="Example data dict to test this component with")


class PaperComponentAnalyzerSignature(dspy.Signature):
    """Analyze a research paper to identify what specialized components it needs.

    Look for:
    1. Unique data structures that need custom visualization
    2. Specific algorithms or processes that need step-by-step breakdown
    3. Comparison patterns (before/after, old/new, etc.)
    4. Mathematical formulas or equations
    5. Code examples or pseudocode
    6. Experimental results that need charts
    7. Architecture diagrams
    8. Timeline or historical context

    Be specific about what the paper needs, not generic suggestions.
    """

    paper_title: str = dspy.InputField()
    paper_abstract: str = dspy.InputField()
    paper_concepts: str = dspy.InputField(desc="Key concepts as JSON")
    paper_methodology: str = dspy.InputField(desc="Methodology description if available")

    needed_components: str = dspy.OutputField(
        desc="""JSON array of components this specific paper needs: [
        {"component_type": "existing or new", "name": "component_name", "reason": "why needed", "data_hint": "what data to show"}
    ]"""
    )
    visualization_suggestions: str = dspy.OutputField(
        desc="Specific visualization suggestions for key concepts"
    )


class ComponentAutoGenerator:
    """
    Automatically discovers and generates new slide components using LLM.

    This system:
    1. Analyzes papers to find component gaps
    2. Suggests new components based on patterns
    3. Generates component code automatically
    4. Validates and tests new components
    5. Adds them to the library
    """

    # Master list of component ideas to reach 1000+
    COMPONENT_IDEAS = {
        # Data Visualization (100+ variants)
        "data": [
            "bar_chart_horizontal",
            "bar_chart_stacked",
            "bar_chart_grouped",
            "bar_chart_animated",
            "line_chart_multi",
            "line_chart_area",
            "line_chart_sparkline",
            "line_chart_gradient",
            "pie_chart_donut",
            "pie_chart_nested",
            "pie_chart_exploded",
            "pie_chart_progress",
            "scatter_plot_bubble",
            "scatter_plot_connected",
            "scatter_plot_quadrant",
            "heatmap_interactive",
            "heatmap_calendar",
            "heatmap_correlation",
            "radar_chart",
            "radar_chart_filled",
            "radar_chart_comparison",
            "treemap",
            "treemap_nested",
            "sunburst_chart",
            "sankey_diagram",
            "chord_diagram",
            "network_graph",
            "gauge_chart",
            "gauge_multi",
            "speedometer",
            "funnel_chart",
            "pyramid_chart",
            "waterfall_chart",
            "box_plot",
            "violin_plot",
            "histogram",
            "candlestick",
            "ohlc_chart",
            "stock_chart",
            "geo_map",
            "choropleth",
            "bubble_map",
            "stream_graph",
            "ridgeline_plot",
            "density_plot",
        ],
        # Content & Text (80+ variants)
        "content": [
            "paragraph_highlight",
            "paragraph_callout",
            "paragraph_sidebar",
            "bullet_icons",
            "bullet_checkmarks",
            "bullet_numbers_fancy",
            "bullet_timeline_inline",
            "bullet_progress",
            "bullet_expandable",
            "quote_large",
            "quote_sidebar",
            "quote_with_image",
            "definition_card",
            "definition_inline",
            "definition_tooltip",
            "fact_box",
            "fact_banner",
            "fact_counter",
            "tip_box",
            "warning_box",
            "info_box",
            "success_box",
            "callout_important",
            "callout_note",
            "callout_example",
            "blockquote_fancy",
            "blockquote_author",
            "blockquote_source",
            "text_columns_2",
            "text_columns_3",
            "text_with_sidebar",
            "abstract_summary",
            "tldr_box",
            "key_points_box",
            "annotation_text",
            "footnote_inline",
            "reference_card",
        ],
        # Visual & Diagrams (150+ variants)
        "visual": [
            "architecture_horizontal",
            "architecture_vertical",
            "architecture_circular",
            "architecture_layered",
            "architecture_microservices",
            "architecture_pipeline",
            "flowchart_decision",
            "flowchart_process",
            "flowchart_swimlane",
            "flowchart_circular",
            "flowchart_branching",
            "flowchart_parallel",
            "diagram_venn_2",
            "diagram_venn_3",
            "diagram_venn_4",
            "diagram_tree",
            "diagram_tree_horizontal",
            "diagram_org_chart",
            "diagram_mind_map",
            "diagram_concept_map",
            "diagram_spider",
            "diagram_sequence",
            "diagram_state_machine",
            "diagram_entity_relationship",
            "diagram_class",
            "diagram_component",
            "diagram_deployment",
            "diagram_activity",
            "diagram_use_case",
            "diagram_data_flow",
            "neural_network_simple",
            "neural_network_detailed",
            "neural_network_animated",
            "transformer_attention",
            "transformer_encoder",
            "transformer_decoder",
            "cnn_layers",
            "rnn_unrolled",
            "lstm_cell",
            "gru_cell",
            "matrix_visualization",
            "tensor_3d",
            "embedding_space",
            "attention_heatmap",
            "attention_heads",
            "attention_pattern",
            "gradient_flow",
            "backprop_visual",
            "loss_landscape",
            "image_grid",
            "image_comparison",
            "image_before_after",
            "image_gallery",
            "image_carousel",
            "image_lightbox",
            "icon_feature_grid",
            "icon_process",
            "icon_benefits",
            "logo_grid",
            "partner_logos",
            "tech_stack",
            "infographic_vertical",
            "infographic_horizontal",
            "infographic_circular",
        ],
        # Comparison & Analysis (60+ variants)
        "comparison": [
            "before_after_split",
            "before_after_overlay",
            "before_after_slider",
            "comparison_table_simple",
            "comparison_table_feature",
            "comparison_table_pricing",
            "comparison_cards_2",
            "comparison_cards_3",
            "comparison_cards_4",
            "pros_cons_split",
            "pros_cons_cards",
            "pros_cons_table",
            "versus_battle",
            "versus_stats",
            "versus_features",
            "old_new_paradigm",
            "problem_solution",
            "challenge_approach",
            "side_by_side_code",
            "side_by_side_images",
            "side_by_side_stats",
            "benchmark_table",
            "benchmark_chart",
            "benchmark_radar",
            "feature_matrix",
            "compatibility_matrix",
            "support_matrix",
            "rating_comparison",
            "score_comparison",
            "rank_comparison",
            "ablation_study",
            "experiment_results",
            "performance_table",
        ],
        # Process & Steps (50+ variants)
        "process": [
            "steps_horizontal",
            "steps_vertical",
            "steps_circular",
            "steps_numbered",
            "steps_icons",
            "steps_timeline",
            "steps_branching",
            "steps_parallel",
            "steps_nested",
            "workflow_linear",
            "workflow_branching",
            "workflow_loop",
            "pipeline_horizontal",
            "pipeline_vertical",
            "pipeline_stages",
            "roadmap_horizontal",
            "roadmap_vertical",
            "roadmap_milestones",
            "journey_map",
            "user_flow",
            "decision_tree",
            "algorithm_steps",
            "pseudocode_steps",
            "method_breakdown",
            "training_pipeline",
            "inference_pipeline",
            "data_pipeline",
            "lifecycle_stages",
            "phase_diagram",
            "sprint_timeline",
        ],
        # Interactive Elements (40+ variants)
        "interactive": [
            "tabs_horizontal",
            "tabs_vertical",
            "tabs_pills",
            "accordion_simple",
            "accordion_nested",
            "accordion_faq",
            "toggle_switch",
            "toggle_comparison",
            "toggle_view",
            "slider_range",
            "slider_comparison",
            "slider_timeline",
            "modal_popup",
            "modal_lightbox",
            "modal_gallery",
            "tooltip_hover",
            "tooltip_click",
            "tooltip_tour",
            "dropdown_menu",
            "dropdown_filter",
            "dropdown_select",
            "search_box",
            "filter_tags",
            "sort_options",
            "pagination_dots",
            "pagination_numbers",
            "pagination_infinite",
            "progress_linear",
            "progress_circular",
            "progress_steps",
        ],
        # Code & Technical (60+ variants)
        "code": [
            "code_block_python",
            "code_block_javascript",
            "code_block_rust",
            "code_block_multiline",
            "code_block_highlighted",
            "code_block_diff",
            "code_inline",
            "code_variable",
            "code_function",
            "code_comparison",
            "code_before_after",
            "code_evolution",
            "terminal_output",
            "terminal_animated",
            "terminal_interactive",
            "api_endpoint",
            "api_request",
            "api_response",
            "json_viewer",
            "xml_viewer",
            "yaml_viewer",
            "schema_definition",
            "type_definition",
            "interface_definition",
            "algorithm_pseudocode",
            "algorithm_complexity",
            "algorithm_trace",
            "regex_explainer",
            "sql_query",
            "graphql_query",
            "git_diff",
            "git_history",
            "git_branch",
            "config_file",
            "env_variables",
            "package_json",
        ],
        # Math & Formulas (50+ variants)
        "math": [
            "formula_block",
            "formula_inline",
            "formula_numbered",
            "formula_derivation",
            "formula_step_by_step",
            "formula_annotated",
            "equation_system",
            "equation_matrix",
            "equation_integral",
            "matrix_display",
            "matrix_operation",
            "matrix_transformation",
            "vector_notation",
            "vector_operations",
            "vector_space",
            "graph_function",
            "graph_3d",
            "graph_parametric",
            "probability_distribution",
            "probability_tree",
            "probability_table",
            "statistics_summary",
            "statistics_test",
            "statistics_confidence",
            "proof_block",
            "theorem_box",
            "lemma_box",
            "notation_table",
            "symbol_legend",
            "variable_definitions",
        ],
        # Results & Metrics (40+ variants)
        "results": [
            "metrics_grid",
            "metrics_cards",
            "metrics_dashboard",
            "kpi_single",
            "kpi_comparison",
            "kpi_trend",
            "score_card",
            "score_breakdown",
            "score_radar",
            "accuracy_display",
            "precision_recall",
            "f1_breakdown",
            "confusion_matrix",
            "roc_curve",
            "pr_curve",
            "loss_curve",
            "training_progress",
            "validation_metrics",
            "benchmark_results",
            "sota_comparison",
            "ablation_results",
            "experiment_table",
            "hyperparameter_table",
            "result_summary",
            "improvement_delta",
            "percentage_change",
            "trend_indicator",
        ],
        # Timeline & History (30+ variants)
        "timeline": [
            "timeline_vertical",
            "timeline_horizontal",
            "timeline_alternating",
            "timeline_compact",
            "timeline_detailed",
            "timeline_milestones",
            "timeline_branching",
            "timeline_parallel",
            "timeline_connected",
            "history_cards",
            "history_timeline",
            "history_eras",
            "evolution_stages",
            "version_history",
            "changelog",
            "roadmap_future",
            "roadmap_past",
            "roadmap_both",
            "event_sequence",
            "story_timeline",
            "career_path",
        ],
        # Team & Credits (25+ variants)
        "team": [
            "author_card",
            "author_grid",
            "author_list",
            "team_photo_grid",
            "team_cards",
            "team_org_chart",
            "contributor_list",
            "contributor_stats",
            "contributor_avatars",
            "acknowledgments",
            "funding_sources",
            "institution_logos",
            "citation_card",
            "bibtex_display",
            "reference_list",
        ],
        # Ending & CTA (25+ variants)
        "ending": [
            "thank_you_simple",
            "thank_you_animated",
            "thank_you_contact",
            "qa_slide",
            "qa_with_links",
            "qa_timer",
            "summary_bullets",
            "summary_cards",
            "summary_visual",
            "next_steps_list",
            "next_steps_cards",
            "next_steps_timeline",
            "cta_single",
            "cta_multiple",
            "cta_with_qr",
            "contact_info",
            "social_links",
            "follow_up",
            "resources_list",
            "further_reading",
            "related_papers",
        ],
        # Specialized Research (100+ domain-specific)
        "specialized": [
            # NLP
            "tokenization_visual",
            "embedding_comparison",
            "attention_weights",
            "word_cloud",
            "sentiment_analysis",
            "ner_highlight",
            "translation_parallel",
            "summarization_comparison",
            "qa_example",
            # Computer Vision
            "image_classification",
            "object_detection",
            "segmentation_mask",
            "feature_maps",
            "convolution_visual",
            "pooling_visual",
            "augmentation_examples",
            "dataset_samples",
            "class_distribution",
            # Reinforcement Learning
            "environment_visual",
            "reward_curve",
            "policy_diagram",
            "state_action_space",
            "trajectory_visual",
            "q_table",
            # Generative AI
            "generation_samples",
            "interpolation_grid",
            "latent_space",
            "diffusion_steps",
            "vae_diagram",
            "gan_architecture",
            # Graphs & Networks
            "knowledge_graph",
            "social_network",
            "citation_network",
            "molecule_structure",
            "protein_structure",
            "pathway_diagram",
            # Time Series
            "forecast_chart",
            "seasonality_decomposition",
            "anomaly_detection",
            "correlation_matrix",
            "lag_plot",
            "autocorrelation",
            # Optimization
            "loss_surface",
            "gradient_descent",
            "hyperparameter_search",
            "pareto_front",
            "convergence_plot",
            "learning_rate_schedule",
        ],
    }

    def __init__(self) -> None:
        self.suggestion_module = dspy.ChainOfThought(ComponentSuggestionSignature)
        self.code_generator = dspy.ChainOfThought(ComponentCodeGeneratorSignature)
        self.paper_analyzer = dspy.ChainOfThought(PaperComponentAnalyzerSignature)
        self.generated_components: List[ComponentSpec] = []
        self.component_registry: Dict[str, str] = {}  # name -> code

    def get_existing_components(self) -> List[str]:
        """Get list of all existing component names"""
        # Base components from HTMLSlideGenerator
        base_components = [
            "title_hero",
            "title_minimal",
            "title_centered",
            "title_split",
            "stats_grid",
            "stats_inline",
            "feature_cards",
            "feature_icons",
            "comparison_table",
            "process_steps",
            "quote",
            "code_block",
            "timeline",
            "diagram",
            "icon_grid",
            "formula",
            "before_after",
            "bullet_points",
            "definition",
            "pros_cons",
            "checklist",
            "authors",
            "qa",
            "key_takeaways",
            "two_column",
            "architecture",
            "chart_bar",
            "chart_line",
        ]

        # Add generated components
        return base_components + list(self.component_registry.keys())

    def get_total_component_count(self) -> int:
        """Get total count of all component ideas"""
        count = 0
        for category, components in self.COMPONENT_IDEAS.items():
            count += len(components)
        return count + len(self.get_existing_components())

    async def analyze_paper_needs(self, paper_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze what components a specific paper needs"""
        try:
            result = self.paper_analyzer(
                paper_title=paper_data.get("paper_title", ""),
                paper_abstract=paper_data.get("hook", paper_data.get("abstract", "")),
                paper_concepts=json.dumps(paper_data.get("concepts", [])[:5]),
                paper_methodology=paper_data.get("methodology", "")[:500],
            )

            return {
                "needed_components": json.loads(result.needed_components),
                "visualization_suggestions": result.visualization_suggestions,
            }
        except Exception as e:
            logger.error(f"Paper analysis failed: {e}")
            return {"needed_components": [], "visualization_suggestions": ""}

    async def suggest_new_components(
        self, paper_data: Dict[str, Any], category_focus: str = "all", num_suggestions: int = 10
    ) -> List[Dict[str, Any]]:
        """Suggest new components based on paper patterns"""
        try:
            existing = self.get_existing_components()

            result = self.suggestion_module(
                paper_abstract=paper_data.get("hook", paper_data.get("abstract", ""))[:500],
                paper_concepts=json.dumps(paper_data.get("concepts", [])[:5]),
                existing_components=", ".join(existing[:30]),
                category_focus=category_focus,
            )

            suggestions = json.loads(result.suggestions_json)

            # Filter out already existing components
            new_suggestions = [
                s
                for s in suggestions
                if s.get("name", "").lower() not in [c.lower() for c in existing]
            ]

            return new_suggestions[:num_suggestions]

        except Exception as e:
            logger.error(f"Component suggestion failed: {e}")
            return []

    async def generate_component_code(
        self, component_spec: Dict[str, Any], reference_component: str = None
    ) -> Tuple[str, Dict[str, Any]]:
        """Generate Python code for a new component"""
        try:
            # Get reference code from existing component
            if not reference_component:
                reference_component = self._get_reference_component_code()

            style_guide = self._get_style_guide()

            result = self.code_generator(
                component_spec=json.dumps(component_spec),
                existing_component_code=reference_component,
                style_guide=style_guide,
            )

            # Parse example data
            try:
                example_data = json.loads(result.example_data)
            except Exception:
                example_data = {}

            return result.python_method_code, example_data

        except Exception as e:
            logger.error(f"Code generation failed: {e}")
            return "", {}

    def _get_reference_component_code(self) -> str:
        """Get example component code for reference"""
        return '''
    def _render_feature_cards(self, data: Dict, index: int) -> str:
        """Render feature cards slide"""
        label = html.escape(data.get("label", "Features"))
        title = html.escape(data.get("title", "Key Features"))
        features = data.get("features", [])

        features_html = ""
        for i, feat in enumerate(features[:6]):
            icon = feat.get("icon", "")
            feat_title = html.escape(feat.get("title", ""))
            description = html.escape(feat.get("description", ""))

            features_html += f\'\'\'
<div class="gradient-border animate-slide-up opacity-0 delay-{(i+1)*100}">
  <div class="bg-[{self.colors['bg_card']}] rounded-[14px] p-6 h-full">
    <div class="icon-square bg-blue-500/20 text-blue-400 mb-4">{icon}</div>
    <h3 class="text-xl font-semibold text-white">{feat_title}</h3>
    <p class="text-[{self.colors['text_secondary']}] mt-3 text-sm leading-relaxed">{description}</p>
  </div>
</div>\'\'\'

        return f\'\'\'
<div class="relative w-full min-h-screen bg-gradient-to-br from-[{self.colors['bg_secondary']}] via-[{self.colors['bg_tertiary']}] to-[{self.colors['bg_primary']}] p-12">
  <div class="absolute right-8 top-8 text-[120px] font-bold text-white/5">{index:02d}</div>
  <span class="text-[{self.config.accent_color}] text-sm font-medium tracking-widest uppercase">{label}</span>
  <h2 class="text-4xl font-bold text-white mt-4">{title}</h2>
  <div class="grid grid-cols-1 md:grid-cols-3 gap-6 mt-10">{features_html}</div>
</div>\'\'\'
'''

    def _get_style_guide(self) -> str:
        """Get the style guide for component generation"""
        return """
CSS Color Variables:
- bg_primary: #0a1929 (darkest navy)
- bg_secondary: #102a43 (dark navy)
- bg_tertiary: #243b53 (medium navy)
- bg_card: #1a365d (card background)
- text_primary: #ffffff (white)
- text_secondary: #9fb3c8 (light gray-blue)
- text_muted: #627d98 (muted gray)
- border: #334e68 (subtle border)
- accent_color: #f6ad55 (gold)
- accent_blue: #4299e1
- accent_purple: #9f7aea
- accent_teal: #38b2ac
- accent_green: #48bb78
- accent_red: #f56565

CSS Classes Available:
- .glass - glassmorphism effect
- .gradient-border - gradient border wrapper
- .gradient-text - gradient text effect
- .card-hover - hover lift effect
- .animate-slide-up, .animate-slide-left, .animate-fade, .animate-scale, .animate-bounce-in
- .delay-100 through .delay-800
- .icon-circle, .icon-square - icon containers
- .tag - small pill/tag
- .progress-bar, .progress-fill - progress indicators

Animation Pattern:
- Use opacity-0 with animation class
- Stagger delays: delay-{(index+1)*100}
- Common: animate-slide-up opacity-0 delay-200
"""

    async def auto_discover_and_generate(
        self, paper_data: Dict[str, Any], max_new_components: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Main auto-discovery loop: analyze paper, suggest components, generate code.

        Returns list of newly generated components with their code.
        """
        logger.info(" Starting auto-discovery loop...")

        # Step 1: Analyze what the paper needs
        logger.info(" Analyzing paper component needs...")
        paper_needs = await self.analyze_paper_needs(paper_data)

        # Step 2: Suggest new components
        logger.info(" Suggesting new components...")
        suggestions = await self.suggest_new_components(
            paper_data, num_suggestions=max_new_components
        )

        # Step 3: Generate code for top suggestions
        generated = []
        for suggestion in suggestions[:max_new_components]:
            logger.info(f" Generating component: {suggestion.get('name', 'unknown')}")

            code, example = await self.generate_component_code(suggestion)

            if code:
                component_info = {
                    "name": suggestion.get("name", ""),
                    "category": suggestion.get("category", ""),
                    "description": suggestion.get("description", ""),
                    "code": code,
                    "example_data": example,
                    "use_cases": suggestion.get("use_cases", []),
                }
                generated.append(component_info)

                # Register the component
                self.component_registry[suggestion.get("name", "")] = code

        logger.info(f" Generated {len(generated)} new components")
        return generated

    def get_component_roadmap(self) -> Dict[str, Any]:
        """Get roadmap to 1000+ components"""
        total_ideas = self.get_total_component_count()
        existing = len(self.get_existing_components())

        roadmap = {
            "current_count": existing,
            "total_planned": total_ideas,
            "progress_percent": round(existing / 1000 * 100, 1),
            "categories": {},
        }

        for category, components in self.COMPONENT_IDEAS.items():
            roadmap["categories"][category] = {
                "planned": len(components),
                "examples": components[:5],
            }

        return roadmap


class ComponentLibraryExpander:
    """
    Batch generator to expand component library toward 1000+ components.
    """

    def __init__(self, output_dir: str = None) -> None:
        self.auto_generator = ComponentAutoGenerator()
        self.output_dir = (
            Path(output_dir)
            if output_dir
            else Path(__file__).parent.parent / "generated_components"
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def expand_category(
        self, category: str, sample_paper_data: Dict[str, Any], num_components: int = 10
    ) -> List[Dict[str, Any]]:
        """Generate multiple components for a specific category"""
        logger.info(f" Expanding category: {category}")

        # Get component ideas for this category
        ideas = ComponentAutoGenerator.COMPONENT_IDEAS.get(category, [])

        generated = []
        for i, component_name in enumerate(ideas[:num_components]):
            logger.info(f"  [{i+1}/{num_components}] Generating: {component_name}")

            # Create a spec from the idea
            spec = {
                "name": component_name,
                "category": category,
                "description": f"Component for displaying {component_name.replace('_', ' ')}",
                "use_cases": [f"Visualize {component_name.replace('_', ' ')} in research papers"],
                "data_fields": {"title": "Component title", "content": "Main content"},
                "visual_elements": ["Animated entry", "Hover effects"],
                "animation_style": "slide",
            }

            code, example = await self.auto_generator.generate_component_code(spec)

            if code:
                generated.append(
                    {
                        "name": component_name,
                        "category": category,
                        "code": code,
                        "example_data": example,
                    }
                )

        return generated

    async def generate_all_categories(
        self, sample_paper_data: Dict[str, Any], components_per_category: int = 5
    ) -> Dict[str, List[Dict]]:
        """Generate components for all categories"""
        all_generated = {}

        for category in ComponentAutoGenerator.COMPONENT_IDEAS.keys():
            generated = await self.expand_category(
                category, sample_paper_data, components_per_category
            )
            all_generated[category] = generated

            # Save to file
            output_file = self.output_dir / f"{category}_components.py"
            self._save_components_to_file(generated, output_file)

        return all_generated

    def _save_components_to_file(self, components: List[Dict], output_file: Path) -> Any:
        """Save generated components to a Python file"""
        code_parts = [
            '"""Auto-generated slide components"""\n',
            "from typing import Dict, Any\n",
            "import html\n\n",
        ]

        for comp in components:
            code_parts.append(f"\n# Component: {comp['name']}\n")
            code_parts.append(comp.get("code", "# Code generation failed\n"))
            code_parts.append("\n")

        with open(output_file, "w") as f:
            f.write("\n".join(code_parts))

        logger.info(f" Saved {len(components)} components to {output_file}")


# Quick test function
async def test_auto_generation() -> Any:
    """Test the auto-generation system"""
    sample_paper = {
        "paper_title": "Attention Is All You Need",
        "hook": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms.",
        "concepts": [
            {
                "name": "Self-Attention",
                "description": "Mechanism relating different positions of a sequence",
            },
            {"name": "Multi-Head Attention", "description": "Parallel attention layers"},
            {"name": "Positional Encoding", "description": "Inject sequence order"},
        ],
    }

    generator = ComponentAutoGenerator()

    logger.info("Component Library Status:")
    roadmap = generator.get_component_roadmap()
    logger.info(f"   Current: {roadmap['current_count']} components")
    logger.info(f"   Planned: {roadmap['total_planned']} component ideas")
    logger.info(f"   Progress: {roadmap['progress_percent']}% toward 1000")

    logger.info("Categories:")
    for cat, info in roadmap["categories"].items():
        logger.info(f"   {cat}: {info['planned']} planned - {info['examples'][:3]}...")

    logger.info("Running auto-discovery...")
    new_components = await generator.auto_discover_and_generate(sample_paper, max_new_components=3)

    logger.info(f"Generated {len(new_components)} new components:")
    for comp in new_components:
        logger.info(f"   - {comp['name']}: {comp['description'][:50]}...")

    return new_components


if __name__ == "__main__":
    asyncio.run(test_auto_generation())
