"""
LIDA-style Visualization Planner for PPTX Generation
=====================================================

Combines Microsoft LIDA's intelligent visualization recommendation approach
with PptxGenJS's precise implementation capabilities.

Architecture:
    Paper Data → LLM (plans & specs) → JSON Specs → PptxGenJS (renders)

The LLM acts as a visualization expert that:
1. Analyzes content deeply to find visualization opportunities
2. Generates SPECIFIC, DETAILED specs (not just "include architecture diagram")
3. Provides exact data: nodes, labels, colors, connections, positions

PptxGenJS then takes these specs and renders pixel-perfect slides.

Inspired by:
- Microsoft LIDA (https://github.com/microsoft/lida) - goal-based visualization
- DSPy - structured LLM outputs
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from enum import Enum

logger = logging.getLogger(__name__)

# Try to import DSPy for LLM-based planning
try:
    import dspy
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False

# Try to import DirectClaudeCLI
try:
    from Jotty.core.integration.direct_claude_cli_lm import DirectClaudeCLI
    CLAUDE_CLI_AVAILABLE = True
except ImportError:
    try:
        from ...integration.direct_claude_cli_lm import DirectClaudeCLI
        CLAUDE_CLI_AVAILABLE = True
    except ImportError:
        CLAUDE_CLI_AVAILABLE = False
        DirectClaudeCLI = None


class VisualizationType(Enum):
    """Types of visualizations that can be generated."""
    ARCHITECTURE = "architecture"
    FLOW = "flow"
    CONCEPT_MAP = "concept_map"
    COMPARISON = "comparison"
    TIMELINE = "timeline"
    METRICS = "metrics"
    CODE_BLOCK = "code_block"
    QUOTE = "quote"


@dataclass
class VisualizationSpec:
    """
    Detailed specification for a single visualization.

    This is the contract between LLM (planner) and PptxGenJS (renderer).
    Every field must be concrete and actionable.
    """
    viz_type: VisualizationType
    title: str
    subtitle: Optional[str] = None

    # For architecture/flow diagrams
    nodes: List[Dict[str, Any]] = field(default_factory=list)
    # Each node: {"id": "enc", "label": "Encoder", "sublabel": "6 layers", "color": "blue", "position": [x, y]}

    connections: List[Dict[str, Any]] = field(default_factory=list)
    # Each connection: {"from": "enc", "to": "dec", "label": "attention", "style": "arrow"}

    # For comparison diagrams
    left_items: List[Dict[str, str]] = field(default_factory=list)
    right_items: List[Dict[str, str]] = field(default_factory=list)
    left_title: str = "Before"
    right_title: str = "After"

    # For metrics
    metrics: List[Dict[str, Any]] = field(default_factory=list)
    # Each metric: {"value": "10x", "label": "Faster Training", "color": "green"}

    # For concept maps
    central_concept: Optional[str] = None
    related_concepts: List[Dict[str, str]] = field(default_factory=list)

    # For code blocks
    code: Optional[str] = None
    language: str = "python"

    # Styling
    color_scheme: str = "default"  # "default", "warm", "cool", "monochrome"
    layout: str = "horizontal"  # "horizontal", "vertical", "radial", "grid"

    # Reasoning (for debugging/improvement)
    reasoning: str = ""
    confidence: float = 0.0


@dataclass
class SlideSpec:
    """Complete specification for a single slide."""
    slide_type: str  # "title", "content", "diagram", "code", "quote", "metrics"
    title: str
    content: Optional[str] = None
    visualization: Optional[VisualizationSpec] = None
    bullets: List[str] = field(default_factory=list)
    speaker_notes: str = ""
    has_eureka_badge: bool = False


@dataclass
class DeckPlan:
    """Complete plan for the entire presentation."""
    slides: List[SlideSpec] = field(default_factory=list)
    total_slides: int = 0
    estimated_duration: str = ""
    key_visualizations: List[str] = field(default_factory=list)
    planning_reasoning: str = ""


# =============================================================================
# LIDA-STYLE LLM VISUALIZATION PLANNER
# =============================================================================

if DSPY_AVAILABLE:

    class ArchitectureSpecSignature(dspy.Signature):
        """
        Generate a detailed architecture diagram specification.

        Output a JSON object with exact nodes, positions, and connections
        that can be directly rendered by a diagramming library.
        """
        paper_title: str = dspy.InputField(desc="Title of the paper")
        paper_content: str = dspy.InputField(desc="Key content describing the architecture")
        concepts: str = dspy.InputField(desc="JSON list of key concepts")

        architecture_spec: str = dspy.OutputField(desc="""Generate a detailed JSON spec:
{
    "title": "Architecture Overview",
    "nodes": [
        {"id": "input", "label": "Input Layer", "sublabel": "Embeddings", "color": "blue", "row": 0, "col": 0},
        {"id": "encoder", "label": "Encoder", "sublabel": "Self-Attention", "color": "green", "row": 1, "col": 0},
        ...
    ],
    "connections": [
        {"from": "input", "to": "encoder", "style": "arrow"},
        ...
    ],
    "layout": "vertical",
    "reasoning": "Why this architecture visualization helps understanding"
}
Be SPECIFIC - extract actual component names from the paper, not generic placeholders.""")


    class FlowSpecSignature(dspy.Signature):
        """Generate a detailed flow/process diagram specification."""
        paper_title: str = dspy.InputField()
        sections: str = dspy.InputField(desc="JSON list of paper sections")

        flow_spec: str = dspy.OutputField(desc="""Generate a detailed JSON spec:
{
    "title": "How It Works",
    "nodes": [
        {"id": "step1", "label": "Step Name", "sublabel": "Brief description", "color": "blue", "order": 1},
        ...
    ],
    "connections": [
        {"from": "step1", "to": "step2", "label": "then", "style": "arrow"}
    ],
    "layout": "horizontal",
    "reasoning": "Why this flow helps understanding"
}
Extract actual process steps from the paper content.""")


    class ComparisonSpecSignature(dspy.Signature):
        """Generate a detailed comparison diagram specification."""
        paper_title: str = dspy.InputField()
        paper_hook: str = dspy.InputField(desc="The hook explaining what's new/different")
        paper_summary: str = dspy.InputField()

        comparison_spec: str = dspy.OutputField(desc="""Generate a detailed JSON spec:
{
    "title": "Innovation Comparison",
    "left_title": "Traditional Approach",
    "right_title": "New Approach",
    "left_items": [
        {"point": "Sequential processing", "is_negative": true},
        {"point": "O(n) path length", "is_negative": true}
    ],
    "right_items": [
        {"point": "Parallel processing", "is_positive": true},
        {"point": "O(1) path length", "is_positive": true}
    ],
    "reasoning": "What makes this comparison meaningful"
}
Extract ACTUAL comparisons from the paper, not generic pros/cons.""")


    class ConceptMapSpecSignature(dspy.Signature):
        """Generate a detailed concept relationship map specification."""
        paper_title: str = dspy.InputField()
        concepts: str = dspy.InputField(desc="JSON list of concepts with descriptions")

        concept_map_spec: str = dspy.OutputField(desc="""Generate a detailed JSON spec:
{
    "title": "Concept Relationships",
    "central_concept": {"id": "main", "label": "Core Concept Name"},
    "related_concepts": [
        {"id": "c1", "label": "Concept 1", "relation_to_center": "enables", "color": "blue"},
        {"id": "c2", "label": "Concept 2", "relation_to_center": "uses", "color": "green"}
    ],
    "connections": [
        {"from": "c1", "to": "c2", "label": "feeds into"}
    ],
    "layout": "radial",
    "reasoning": "How these concepts interconnect"
}
Map ACTUAL relationships between the paper's concepts.""")


    class MetricsSpecSignature(dspy.Signature):
        """Generate a detailed metrics/statistics visualization specification."""
        paper_title: str = dspy.InputField()
        paper_content: str = dspy.InputField(desc="Content containing quantitative claims")

        metrics_spec: str = dspy.OutputField(desc="""Generate a detailed JSON spec:
{
    "title": "Key Results",
    "metrics": [
        {"value": "10x", "label": "Faster Training", "comparison": "vs RNNs", "color": "green"},
        {"value": "28.4", "label": "BLEU Score", "comparison": "state-of-the-art", "color": "blue"}
    ],
    "summary": "Brief summary of what these metrics mean",
    "reasoning": "Why these metrics matter"
}
Extract ACTUAL numbers and results from the paper.""")


    class FullDeckPlanSignature(dspy.Signature):
        """
        Generate a complete presentation plan with all visualizations.

        This is the master planner that decides the overall deck structure
        and which visualizations to include.
        """
        paper_title: str = dspy.InputField()
        paper_hook: str = dspy.InputField()
        paper_summary: str = dspy.InputField()
        concepts: str = dspy.InputField()
        sections: str = dspy.InputField()
        key_insights: str = dspy.InputField()

        deck_plan: str = dspy.OutputField(desc="""Generate a complete deck plan as JSON:
{
    "title_slide": {"title": "Paper Title", "subtitle": "Key hook"},
    "visualizations_needed": [
        {
            "type": "architecture",
            "reasoning": "Paper describes encoder-decoder with specific layers",
            "priority": 1,
            "slide_position": 4
        },
        {
            "type": "flow",
            "reasoning": "Clear 4-step process described",
            "priority": 2,
            "slide_position": 8
        }
    ],
    "visualizations_skipped": [
        {
            "type": "timeline",
            "reasoning": "No historical dates in content"
        }
    ],
    "slide_outline": [
        {"position": 1, "type": "title", "content": "..."},
        {"position": 2, "type": "agenda", "content": "..."},
        ...
    ],
    "estimated_slides": 18,
    "planning_reasoning": "Overall strategy for this presentation"
}
Be strategic - only include visualizations that genuinely enhance understanding.""")


class LIDAStylePlanner:
    """
    LIDA-inspired visualization planner that uses LLM to generate
    detailed, specific visualization specifications.
    """

    def __init__(self, lm=None):
        """Initialize with optional language model."""
        self.lm = lm
        if self.lm is None and CLAUDE_CLI_AVAILABLE and DirectClaudeCLI:
            self.lm = DirectClaudeCLI(model="sonnet", max_retries=2)  # Use sonnet for better reasoning
            if DSPY_AVAILABLE:
                dspy.configure(lm=self.lm)

        if DSPY_AVAILABLE:
            self.deck_planner = dspy.Predict(FullDeckPlanSignature)
            self.arch_planner = dspy.Predict(ArchitectureSpecSignature)
            self.flow_planner = dspy.Predict(FlowSpecSignature)
            self.comparison_planner = dspy.Predict(ComparisonSpecSignature)
            self.concept_map_planner = dspy.Predict(ConceptMapSpecSignature)
            self.metrics_planner = dspy.Predict(MetricsSpecSignature)

    def _extract_json(self, text: str) -> Optional[dict]:
        """Extract JSON from LLM response."""
        import re
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to find JSON in text
        patterns = [r'\{[\s\S]*\}', r'```json\s*([\s\S]*?)\s*```', r'```\s*([\s\S]*?)\s*```']
        for pattern in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                try:
                    return json.loads(match)
                except json.JSONDecodeError:
                    continue
        return None

    def plan_deck(self, paper_data: Dict[str, Any]) -> Optional[DeckPlan]:
        """
        Generate a complete deck plan using LLM.

        Returns structured plan with all visualization specs.
        """
        if not DSPY_AVAILABLE:
            logger.warning("DSPy not available, cannot plan deck")
            return None

        try:
            # Generate overall deck plan
            result = self.deck_planner(
                paper_title=paper_data.get('paper_title', ''),
                paper_hook=paper_data.get('hook', '')[:500],
                paper_summary=paper_data.get('summary', '')[:500],
                concepts=json.dumps([c.get('name', '') for c in paper_data.get('concepts', [])[:6]]),
                sections=json.dumps([s.get('title', '') for s in paper_data.get('sections', [])[:5]]),
                key_insights=json.dumps(paper_data.get('key_insights', [])[:5])
            )

            plan_data = self._extract_json(result.deck_plan)
            if not plan_data:
                logger.warning("Could not parse deck plan")
                return None

            logger.info(f"📋 Deck plan generated: {plan_data.get('estimated_slides', '?')} slides")

            # Build DeckPlan object
            deck_plan = DeckPlan(
                total_slides=plan_data.get('estimated_slides', 15),
                planning_reasoning=plan_data.get('planning_reasoning', ''),
                key_visualizations=[v['type'] for v in plan_data.get('visualizations_needed', [])]
            )

            return deck_plan, plan_data

        except Exception as e:
            logger.error(f"Deck planning failed: {e}")
            return None

    def generate_architecture_spec(self, paper_data: Dict[str, Any]) -> Optional[VisualizationSpec]:
        """Generate detailed architecture diagram specification."""
        if not DSPY_AVAILABLE:
            return None

        try:
            result = self.arch_planner(
                paper_title=paper_data.get('paper_title', ''),
                paper_content=paper_data.get('hook', '') + '\n' + paper_data.get('summary', ''),
                concepts=json.dumps([
                    {"name": c.get('name', ''), "description": c.get('description', '')[:100]}
                    for c in paper_data.get('concepts', [])[:6]
                ])
            )

            spec_data = self._extract_json(result.architecture_spec)
            if not spec_data:
                return None

            return VisualizationSpec(
                viz_type=VisualizationType.ARCHITECTURE,
                title=spec_data.get('title', 'Architecture Overview'),
                nodes=spec_data.get('nodes', []),
                connections=spec_data.get('connections', []),
                layout=spec_data.get('layout', 'vertical'),
                reasoning=spec_data.get('reasoning', ''),
                confidence=0.9
            )

        except Exception as e:
            logger.error(f"Architecture spec generation failed: {e}")
            return None

    def generate_flow_spec(self, paper_data: Dict[str, Any]) -> Optional[VisualizationSpec]:
        """Generate detailed flow diagram specification."""
        if not DSPY_AVAILABLE:
            return None

        try:
            result = self.flow_planner(
                paper_title=paper_data.get('paper_title', ''),
                sections=json.dumps([
                    {"title": s.get('title', ''), "content": s.get('content', '')[:200]}
                    for s in paper_data.get('sections', [])[:5]
                ])
            )

            spec_data = self._extract_json(result.flow_spec)
            if not spec_data:
                return None

            return VisualizationSpec(
                viz_type=VisualizationType.FLOW,
                title=spec_data.get('title', 'How It Works'),
                nodes=spec_data.get('nodes', []),
                connections=spec_data.get('connections', []),
                layout=spec_data.get('layout', 'horizontal'),
                reasoning=spec_data.get('reasoning', ''),
                confidence=0.85
            )

        except Exception as e:
            logger.error(f"Flow spec generation failed: {e}")
            return None

    def generate_comparison_spec(self, paper_data: Dict[str, Any]) -> Optional[VisualizationSpec]:
        """Generate detailed comparison diagram specification."""
        if not DSPY_AVAILABLE:
            return None

        try:
            result = self.comparison_planner(
                paper_title=paper_data.get('paper_title', ''),
                paper_hook=paper_data.get('hook', '')[:500],
                paper_summary=paper_data.get('summary', '')[:500]
            )

            spec_data = self._extract_json(result.comparison_spec)
            if not spec_data:
                return None

            return VisualizationSpec(
                viz_type=VisualizationType.COMPARISON,
                title=spec_data.get('title', 'Innovation Comparison'),
                left_title=spec_data.get('left_title', 'Traditional'),
                right_title=spec_data.get('right_title', 'New Approach'),
                left_items=spec_data.get('left_items', []),
                right_items=spec_data.get('right_items', []),
                reasoning=spec_data.get('reasoning', ''),
                confidence=0.85
            )

        except Exception as e:
            logger.error(f"Comparison spec generation failed: {e}")
            return None

    def generate_concept_map_spec(self, paper_data: Dict[str, Any]) -> Optional[VisualizationSpec]:
        """Generate detailed concept map specification."""
        if not DSPY_AVAILABLE:
            return None

        try:
            result = self.concept_map_planner(
                paper_title=paper_data.get('paper_title', ''),
                concepts=json.dumps([
                    {"name": c.get('name', ''), "description": c.get('description', '')[:150]}
                    for c in paper_data.get('concepts', [])[:6]
                ])
            )

            spec_data = self._extract_json(result.concept_map_spec)
            if not spec_data:
                return None

            return VisualizationSpec(
                viz_type=VisualizationType.CONCEPT_MAP,
                title=spec_data.get('title', 'Concept Relationships'),
                central_concept=spec_data.get('central_concept', {}).get('label'),
                related_concepts=spec_data.get('related_concepts', []),
                connections=spec_data.get('connections', []),
                layout=spec_data.get('layout', 'radial'),
                reasoning=spec_data.get('reasoning', ''),
                confidence=0.8
            )

        except Exception as e:
            logger.error(f"Concept map spec generation failed: {e}")
            return None

    def generate_metrics_spec(self, paper_data: Dict[str, Any]) -> Optional[VisualizationSpec]:
        """Generate detailed metrics visualization specification."""
        if not DSPY_AVAILABLE:
            return None

        try:
            result = self.metrics_planner(
                paper_title=paper_data.get('paper_title', ''),
                paper_content=paper_data.get('hook', '') + '\n' + paper_data.get('summary', '')
            )

            spec_data = self._extract_json(result.metrics_spec)
            if not spec_data:
                return None

            return VisualizationSpec(
                viz_type=VisualizationType.METRICS,
                title=spec_data.get('title', 'Key Results'),
                metrics=spec_data.get('metrics', []),
                reasoning=spec_data.get('reasoning', ''),
                confidence=0.85
            )

        except Exception as e:
            logger.error(f"Metrics spec generation failed: {e}")
            return None

    def generate_all_specs(self, paper_data: Dict[str, Any]) -> Dict[str, VisualizationSpec]:
        """
        Generate all visualization specifications for a paper.

        Returns dict mapping viz type to its detailed spec.
        """
        specs = {}

        logger.info("🎨 Generating LIDA-style visualization specs...")

        # Generate each spec type
        arch_spec = self.generate_architecture_spec(paper_data)
        if arch_spec:
            specs['architecture'] = arch_spec
            logger.info(f"  ✅ Architecture: {len(arch_spec.nodes)} nodes")

        flow_spec = self.generate_flow_spec(paper_data)
        if flow_spec:
            specs['flow'] = flow_spec
            logger.info(f"  ✅ Flow: {len(flow_spec.nodes)} steps")

        comparison_spec = self.generate_comparison_spec(paper_data)
        if comparison_spec:
            specs['comparison'] = comparison_spec
            logger.info(f"  ✅ Comparison: {len(comparison_spec.left_items)} vs {len(comparison_spec.right_items)} points")

        concept_map_spec = self.generate_concept_map_spec(paper_data)
        if concept_map_spec:
            specs['concept_map'] = concept_map_spec
            logger.info(f"  ✅ Concept Map: {len(concept_map_spec.related_concepts)} concepts")

        metrics_spec = self.generate_metrics_spec(paper_data)
        if metrics_spec:
            specs['metrics'] = metrics_spec
            logger.info(f"  ✅ Metrics: {len(metrics_spec.metrics)} stats")

        logger.info(f"📊 Generated {len(specs)} visualization specs")

        return specs


def convert_specs_to_pptx_data(specs: Dict[str, VisualizationSpec]) -> Dict[str, Any]:
    """
    Convert VisualizationSpecs to the JSON format expected by PptxGenJS.

    This is the bridge between LLM planning and PPTX rendering.
    """
    pptx_data = {
        'visualization_specs': {}
    }

    for viz_type, spec in specs.items():
        pptx_data['visualization_specs'][viz_type] = {
            'title': spec.title,
            'subtitle': spec.subtitle,
            'nodes': spec.nodes,
            'connections': spec.connections,
            'left_items': spec.left_items,
            'right_items': spec.right_items,
            'left_title': spec.left_title,
            'right_title': spec.right_title,
            'metrics': spec.metrics,
            'central_concept': spec.central_concept,
            'related_concepts': spec.related_concepts,
            'layout': spec.layout,
            'color_scheme': spec.color_scheme,
            'reasoning': spec.reasoning,
            'confidence': spec.confidence
        }

    return pptx_data


__all__ = [
    'VisualizationType',
    'VisualizationSpec',
    'SlideSpec',
    'DeckPlan',
    'LIDAStylePlanner',
    'convert_specs_to_pptx_data',
    'DSPY_AVAILABLE',
    'CLAUDE_CLI_AVAILABLE',
]
