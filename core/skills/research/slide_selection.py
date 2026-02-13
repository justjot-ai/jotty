"""
Slide Selection - Intelligent slide type selection using heuristics and LLM.

Extracted from html_slide_generator.py.
"""

import re
import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

try:
    import dspy
    DSPY_AVAILABLE = True
except ImportError:
    dspy = None
    DSPY_AVAILABLE = False

from .content_analysis import ContentPattern, ContentAnalysis, ContentAnalyzer

class SlideTypeSelector:
    """
    Intelligent slide type selector that combines content analysis
    with presentation rhythm and variety optimization.
    Forces maximum variety to avoid repetitive presentations.
    """

    # Slide type categories for variety balancing
    VISUAL_TYPES = {"DIAGRAM", "ARCHITECTURE", "CHART_BAR", "CHART_LINE", "FLOWCHART", "TIMELINE"}
    DATA_TYPES = {"STATS_GRID", "COMPARISON_TABLE", "CHART_BAR", "CHART_LINE", "PROGRESS_BARS"}
    TEXT_TYPES = {"BULLET_POINTS", "TWO_COLUMN", "DEFINITION", "QUOTE", "NARRATIVE"}
    SPECIAL_TYPES = {"FORMULA", "CODE_BLOCK", "BEFORE_AFTER", "PROS_CONS", "CHECKLIST"}

    # Extended rhythm sequence for better variety
    RHYTHM_SEQUENCE = ["TEXT", "VISUAL", "DATA", "SPECIAL", "TEXT", "VISUAL", "SPECIAL", "DATA"]

    # Type rotation pools - rotate through these for variety
    TEXT_ROTATION = ["TWO_COLUMN", "BULLET_POINTS", "DEFINITION", "QUOTE"]
    VISUAL_ROTATION = ["DIAGRAM", "TIMELINE", "PROCESS_STEPS", "ARCHITECTURE"]
    DATA_ROTATION = ["STATS_GRID", "COMPARISON_TABLE", "CHART_BAR"]
    SPECIAL_ROTATION = ["BEFORE_AFTER", "PROS_CONS", "FORMULA", "CODE_BLOCK", "CHECKLIST"]

    def __init__(self):
        self.analyzer = ContentAnalyzer()
        self.used_types: List[str] = []
        self.rhythm_index = 0
        self.rotation_indices = {"TEXT": 0, "VISUAL": 0, "DATA": 0, "SPECIAL": 0}

    def reset(self):
        """Reset state for a new presentation."""
        self.used_types = []
        self.rhythm_index = 0
        self.rotation_indices = {"TEXT": 0, "VISUAL": 0, "DATA": 0, "SPECIAL": 0}

    def select_slide_type(
        self,
        content: str,
        title: str = "",
        level: int = 1,
        force_variety: bool = True,
        preferred_types: List[str] = None
    ) -> Tuple[str, ContentAnalysis]:
        """
        Select the best slide type for given content with maximum variety.
        """
        # Analyze content
        analysis = self.analyzer.analyze(content, title)

        # Get suggested types from analysis
        candidates = list(analysis.suggested_types)

        # Add preferred types if specified
        if preferred_types:
            candidates = preferred_types + candidates

        # Add level-based suggestions
        level_suggestions = self._get_level_suggestions(level)
        candidates.extend(level_suggestions)

        # Get rhythm-based suggestion with rotation
        rhythm_type = self._get_rhythm_suggestion()
        if rhythm_type:
            candidates.insert(0, rhythm_type)

        # Aggressively filter out recently used types - look at last 5 types
        if force_variety and len(self.used_types) >= 1:
            recent = set(self.used_types[-5:])
            filtered = [t for t in candidates if t not in recent]
            if filtered:
                candidates = filtered

        # Also avoid same category as last 2 slides
        if force_variety and len(self.used_types) >= 2:
            last_categories = [self._get_category(t) for t in self.used_types[-2:]]
            def is_different_category(t):
                return self._get_category(t) not in last_categories
            filtered = [t for t in candidates if is_different_category(t)]
            if filtered:
                candidates = filtered

        # Select best candidate
        selected = candidates[0] if candidates else "BULLET_POINTS"

        # Validate selection exists
        valid_types = {e.name for e in SlideType}
        if selected not in valid_types:
            selected = "BULLET_POINTS"

        # Track usage
        self.used_types.append(selected)
        self._advance_rhythm()

        return selected, analysis

    def _get_category(self, slide_type: str) -> str:
        """Get the category of a slide type."""
        if slide_type in self.VISUAL_TYPES:
            return "VISUAL"
        elif slide_type in self.DATA_TYPES:
            return "DATA"
        elif slide_type in self.SPECIAL_TYPES:
            return "SPECIAL"
        return "TEXT"

    def _get_level_suggestions(self, level: int) -> List[str]:
        """Get slide type suggestions based on content level."""
        suggestions = {
            1: ["DEFINITION", "ICON_GRID", "TWO_COLUMN"],           # Basics
            2: ["QUOTE", "DIAGRAM", "BEFORE_AFTER"],                 # Intuition
            3: ["FORMULA", "CODE_BLOCK", "PROCESS_STEPS"],           # Math/Technical
            4: ["STATS_GRID", "COMPARISON_TABLE", "PROS_CONS"],      # Applications
            5: ["ARCHITECTURE", "TIMELINE", "FLOWCHART"],            # Advanced
        }
        return suggestions.get(level, ["TWO_COLUMN"])

    def _get_rhythm_suggestion(self) -> Optional[str]:
        """Get a suggestion based on visual rhythm with rotation."""
        category = self.RHYTHM_SEQUENCE[self.rhythm_index % len(self.RHYTHM_SEQUENCE)]

        # Get from rotation pool
        rotation_map = {
            "TEXT": self.TEXT_ROTATION,
            "VISUAL": self.VISUAL_ROTATION,
            "DATA": self.DATA_ROTATION,
            "SPECIAL": self.SPECIAL_ROTATION,
        }

        pool = rotation_map.get(category, self.TEXT_ROTATION)
        idx = self.rotation_indices.get(category, 0)
        suggested = pool[idx % len(pool)]

        # Advance rotation for this category
        self.rotation_indices[category] = idx + 1

        return suggested

    def _advance_rhythm(self):
        """Advance the rhythm counter."""
        self.rhythm_index += 1

    def get_variety_score(self) -> float:
        """Calculate how varied the slide types are (0-1, higher = more varied)."""
        if len(self.used_types) < 2:
            return 1.0
        unique = len(set(self.used_types))
        return unique / len(self.used_types)

    def suggest_missing_types(self) -> List[str]:
        """Suggest slide types that haven't been used for variety."""
        all_types = {"DIAGRAM", "STATS_GRID", "QUOTE", "TIMELINE", "BEFORE_AFTER", "CODE_BLOCK", "PROS_CONS"}
        used = set(self.used_types)
        missing = all_types - used
        return list(missing)[:3]


# =============================================================================
# LLM-BASED INTELLIGENT SLIDE SELECTOR (Agent-powered)
# =============================================================================
# Uses DSPy to understand content semantically and select optimal slide types.
# Ensures consistency: similar content types get similar slide formats.

if DSPY_AVAILABLE:
    class SlideSelectionSignature(dspy.Signature):
        """Select the optimal slide type for content.

        You are an expert presentation designer. Analyze the content and select
        the BEST slide type that will communicate it effectively.

        Rules:
        1. CONSISTENCY: Similar content types MUST use similar slide formats
           - All "concept explanations" should use the SAME slide type
           - All "math derivations" should use the SAME slide type
           - All "code examples" should use the SAME slide type
        2. MATCH CONTENT: The slide type must match what the content actually contains
           - Only use FORMULA if there are actual LaTeX/math equations to display
           - Only use CODE_BLOCK if there's actual code
           - Only use CHART_* if there's numerical data to visualize
        3. READABILITY: Choose formats that make content easy to understand
        """
        content: str = dspy.InputField(desc="The content to display on this slide")
        content_type: str = dspy.InputField(desc="Type: concept|section|math|code|comparison|summary")
        title: str = dspy.InputField(desc="Slide title")
        available_types: str = dspy.InputField(desc="Comma-separated list of available slide types")
        previously_used: str = dspy.InputField(desc="Recently used types (for variety)")

        slide_type: str = dspy.OutputField(desc="Selected slide type (must be from available_types)")
        reasoning: str = dspy.OutputField(desc="Brief explanation of why this type is best")
        has_displayable_math: bool = dspy.OutputField(desc="True if content has actual LaTeX/equations to render")
        has_displayable_code: bool = dspy.OutputField(desc="True if content has actual code to display")


class LLMSlideSelector:
    """
    LLM-powered slide type selector using DSPy.

    Uses semantic understanding instead of regex patterns.
    Ensures consistency across similar content types.
    """

    # Slide types grouped by purpose
    CONCEPT_TYPES = ["TWO_COLUMN", "DEFINITION", "FEATURE_CARDS"]
    MATH_TYPES = ["FORMULA", "PROCESS_STEPS"]
    CODE_TYPES = ["CODE_BLOCK"]
    COMPARISON_TYPES = ["BEFORE_AFTER", "PROS_CONS", "COMPARISON_TABLE"]
    DATA_TYPES = ["STATS_GRID", "CHART_BAR", "CHART_LINE"]
    VISUAL_TYPES = ["DIAGRAM", "TIMELINE", "ARCHITECTURE", "PROCESS_STEPS"]
    SUMMARY_TYPES = ["KEY_TAKEAWAYS", "BULLET_POINTS"]

    def __init__(self, use_llm: bool = True):
        """Initialize selector.

        Args:
            use_llm: If True, use DSPy LLM selection. If False, fall back to heuristics.
        """
        self.use_llm = use_llm and DSPY_AVAILABLE
        self.used_types: List[str] = []
        self.content_type_mapping: Dict[str, str] = {}  # Track what type each content_type uses
        self._selector = None

        if self.use_llm:
            try:
                self._selector = dspy.Predict(SlideSelectionSignature)
                logger.info(" LLM slide selector initialized")
            except Exception as e:
                logger.warning(f"Could not init LLM selector: {e}, using heuristics")
                self.use_llm = False

        # Fallback heuristic selector
        self._heuristic_selector = SlideTypeSelector()

    def reset(self):
        """Reset state for a new presentation."""
        self.used_types = []
        self.content_type_mapping = {}
        self._heuristic_selector.reset()

    def select(
        self,
        content: str,
        title: str,
        content_type: str = "section",
        force_consistency: bool = True
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Select the best slide type for content.

        Args:
            content: The content to display
            title: Slide title
            content_type: One of: concept, section, math, code, comparison, summary
            force_consistency: If True, same content_types always get same slide type

        Returns:
            Tuple of (slide_type, metadata_dict)
        """
        # CONSISTENCY: If we've already chosen a type for this content_type, reuse it
        if force_consistency and content_type in self.content_type_mapping:
            chosen = self.content_type_mapping[content_type]
            self.used_types.append(chosen)
            return chosen, {"source": "consistency", "reasoning": f"Consistent with other {content_type} slides"}

        if self.use_llm and self._selector:
            return self._select_with_llm(content, title, content_type)
        else:
            return self._select_with_heuristics(content, title, content_type)

    def _select_with_llm(
        self,
        content: str,
        title: str,
        content_type: str
    ) -> Tuple[str, Dict[str, Any]]:
        """Use LLM for intelligent selection."""
        try:
            # Determine available types based on content_type
            available = self._get_available_types(content_type)
            recently_used = ", ".join(self.used_types[-5:]) if self.used_types else "none"

            result = self._selector(
                content=content[:1500],  # Limit content size
                content_type=content_type,
                title=title,
                available_types=", ".join(available),
                previously_used=recently_used
            )

            slide_type = str(result.slide_type).upper().strip()

            # Validate the selection
            if slide_type not in available:
                logger.warning(f"LLM selected invalid type '{slide_type}', using fallback")
                slide_type = available[0]

            # Don't use FORMULA if no actual math
            if slide_type == "FORMULA" and not result.has_displayable_math:
                slide_type = "TWO_COLUMN"

            # Don't use CODE_BLOCK if no actual code
            if slide_type == "CODE_BLOCK" and not result.has_displayable_code:
                slide_type = "TWO_COLUMN"

            # Track for consistency
            self.used_types.append(slide_type)
            self.content_type_mapping[content_type] = slide_type

            return slide_type, {
                "source": "llm",
                "reasoning": str(result.reasoning),
                "has_math": result.has_displayable_math,
                "has_code": result.has_displayable_code
            }

        except Exception as e:
            logger.warning(f"LLM selection failed: {e}, using heuristics")
            return self._select_with_heuristics(content, title, content_type)

    def _select_with_heuristics(
        self,
        content: str,
        title: str,
        content_type: str
    ) -> Tuple[str, Dict[str, Any]]:
        """Fallback to heuristic selection."""
        # Simple mapping based on content_type
        type_map = {
            "concept": "TWO_COLUMN",
            "math": "FORMULA" if self._has_latex(content) else "PROCESS_STEPS",
            "code": "CODE_BLOCK" if "```" in content or "def " in content else "TWO_COLUMN",
            "comparison": "BEFORE_AFTER",
            "summary": "KEY_TAKEAWAYS",
            "section": "TWO_COLUMN",
        }

        slide_type = type_map.get(content_type, "TWO_COLUMN")

        # Track for consistency
        self.used_types.append(slide_type)
        self.content_type_mapping[content_type] = slide_type

        return slide_type, {"source": "heuristic", "reasoning": f"Default for {content_type}"}

    def _get_available_types(self, content_type: str) -> List[str]:
        """Get available slide types for a content type."""
        type_pools = {
            "concept": self.CONCEPT_TYPES,
            "math": self.MATH_TYPES + ["TWO_COLUMN"],
            "code": self.CODE_TYPES + ["TWO_COLUMN"],
            "comparison": self.COMPARISON_TYPES,
            "summary": self.SUMMARY_TYPES,
            "section": self.CONCEPT_TYPES + self.VISUAL_TYPES,
        }
        return type_pools.get(content_type, self.CONCEPT_TYPES)

    def _has_latex(self, content: str) -> bool:
        """Check if content has actual LaTeX math."""
        latex_patterns = [
            r'\$[^$]+\$',  # Inline math
            r'\$\$[^$]+\$\$',  # Display math
            r'\\frac\{',  # Fractions
            r'\\sum',  # Sum
            r'\\int',  # Integral
            r'\\alpha|\\beta|\\gamma',  # Greek letters
        ]
        return any(re.search(p, content) for p in latex_patterns)


# =============================================================================
# LIDA-INSPIRED VISUALIZATION GOAL GENERATOR
# =============================================================================
# Inspired by Microsoft LIDA: https://github.com/microsoft/lida
# Generates visualization goals and specs for optimal slide content

@dataclass
class VisualizationGoal:
    """A visualization goal inspired by LIDA's goal generation."""
    question: str           # What question does this viz answer?
    viz_type: str           # Recommended visualization type
    rationale: str          # Why this visualization?
    data_fields: List[str]  # What data fields are needed
    priority: int           # 1-5, higher = more important
    aesthetic: str          # Visual style recommendation


class VisualizationGoalGenerator:
    """
    LIDA-inspired goal generator for slide visualizations.

    Instead of just picking slide types, this generates visualization GOALS
    that describe what the audience should learn/understand from each slide.

    Based on Microsoft LIDA principles:
    1. Generate goals from content semantics
    2. Match goals to visualization types
    3. Consider audience and learning objectives
    4. Optimize for insight and comprehension
    """

    # Visualization goal templates by content type
    GOAL_TEMPLATES = {
        "comparison": [
            VisualizationGoal(
                question="How do these approaches differ?",
                viz_type="BEFORE_AFTER",
                rationale="Side-by-side comparison maximizes contrast perception",
                data_fields=["before_state", "after_state", "change_metric"],
                priority=5,
                aesthetic="split-screen with highlight colors"
            ),
            VisualizationGoal(
                question="Which option is better and why?",
                viz_type="COMPARISON_TABLE",
                rationale="Tables enable systematic multi-attribute comparison",
                data_fields=["items", "attributes", "values"],
                priority=4,
                aesthetic="alternating rows with accent highlights"
            ),
        ],
        "process": [
            VisualizationGoal(
                question="What are the steps to achieve this?",
                viz_type="PROCESS_STEPS",
                rationale="Sequential layout matches mental model of processes",
                data_fields=["steps", "descriptions", "icons"],
                priority=5,
                aesthetic="numbered cards with connecting arrows"
            ),
            VisualizationGoal(
                question="How does this flow work?",
                viz_type="FLOWCHART",
                rationale="Flowcharts show decision points and branches",
                data_fields=["nodes", "edges", "decisions"],
                priority=4,
                aesthetic="clean boxes with gradient connections"
            ),
        ],
        "quantitative": [
            VisualizationGoal(
                question="What are the key metrics?",
                viz_type="STATS_GRID",
                rationale="Large numbers with context create instant impact",
                data_fields=["metrics", "values", "trends"],
                priority=5,
                aesthetic="big bold numbers with subtle animations"
            ),
            VisualizationGoal(
                question="How do values compare?",
                viz_type="CHART_BAR",
                rationale="Bar charts are universally understood for comparison",
                data_fields=["categories", "values", "labels"],
                priority=4,
                aesthetic="gradient bars with value labels"
            ),
        ],
        "temporal": [
            VisualizationGoal(
                question="How did this evolve over time?",
                viz_type="TIMELINE",
                rationale="Timelines show progression and milestones",
                data_fields=["dates", "events", "significance"],
                priority=5,
                aesthetic="horizontal flow with milestone markers"
            ),
        ],
        "structural": [
            VisualizationGoal(
                question="How is this system organized?",
                viz_type="ARCHITECTURE",
                rationale="Architecture diagrams reveal component relationships",
                data_fields=["components", "connections", "layers"],
                priority=5,
                aesthetic="layered boxes with data flow arrows"
            ),
            VisualizationGoal(
                question="What are the main components?",
                viz_type="DIAGRAM",
                rationale="Diagrams abstract complexity into visual chunks",
                data_fields=["elements", "relationships", "annotations"],
                priority=4,
                aesthetic="clean shapes with labeled connections"
            ),
        ],
        "conceptual": [
            VisualizationGoal(
                question="What does this term mean?",
                viz_type="DEFINITION",
                rationale="Clear definition with context builds foundation",
                data_fields=["term", "definition", "examples"],
                priority=5,
                aesthetic="centered term with supporting text"
            ),
            VisualizationGoal(
                question="What are the key points?",
                viz_type="BULLET_POINTS",
                rationale="Bullet points chunk information for scanning",
                data_fields=["points", "icons", "emphasis"],
                priority=3,
                aesthetic="icon-prefixed points with hierarchy"
            ),
        ],
        "code": [
            VisualizationGoal(
                question="How is this implemented?",
                viz_type="CODE_BLOCK",
                rationale="Code with syntax highlighting aids comprehension",
                data_fields=["code", "language", "annotations"],
                priority=5,
                aesthetic="dark theme with syntax colors"
            ),
        ],
        "mathematical": [
            VisualizationGoal(
                question="What is the mathematical formulation?",
                viz_type="FORMULA",
                rationale="Centered formula with explanation builds understanding",
                data_fields=["formula", "variables", "derivation"],
                priority=5,
                aesthetic="large centered equation with variable legend"
            ),
        ],
        "evaluative": [
            VisualizationGoal(
                question="What are the pros and cons?",
                viz_type="PROS_CONS",
                rationale="Two-column layout enables balanced evaluation",
                data_fields=["pros", "cons", "neutral"],
                priority=5,
                aesthetic="green/red color coding with icons"
            ),
        ],
        "inspirational": [
            VisualizationGoal(
                question="What is the key takeaway?",
                viz_type="QUOTE",
                rationale="Featured quotes create memorable moments",
                data_fields=["quote", "author", "context"],
                priority=4,
                aesthetic="large italic text with attribution"
            ),
        ],
    }

    def __init__(self):
        self.analyzer = ContentAnalyzer()

    def generate_goals(self, content: str, title: str = "", max_goals: int = 3) -> List[VisualizationGoal]:
        """
        Generate visualization goals for content.

        Args:
            content: The text content to visualize
            title: Section title for context
            max_goals: Maximum number of goals to return

        Returns:
            List of VisualizationGoal objects, prioritized
        """
        analysis = self.analyzer.analyze(content, title)
        goals = []

        # Map detected patterns to goal templates
        pattern_to_template = {
            ContentPattern.COMPARISON: "comparison",
            ContentPattern.STEP_BY_STEP: "process",
            ContentPattern.STATISTICS: "quantitative",
            ContentPattern.TIMELINE: "temporal",
            ContentPattern.ARCHITECTURE: "structural",
            ContentPattern.DEFINITION: "conceptual",
            ContentPattern.CODE_SNIPPET: "code",
            ContentPattern.MATH_FORMULA: "mathematical",
            ContentPattern.PROS_CONS: "evaluative",
            ContentPattern.QUOTE: "inspirational",
            ContentPattern.LIST_ITEMS: "conceptual",
        }

        # Collect goals from matched patterns
        for pattern in analysis.patterns:
            template_key = pattern_to_template.get(pattern)
            if template_key and template_key in self.GOAL_TEMPLATES:
                goals.extend(self.GOAL_TEMPLATES[template_key])

        # Add default conceptual goals if none found
        if not goals:
            goals = self.GOAL_TEMPLATES["conceptual"].copy()

        # Sort by priority and deduplicate by viz_type
        seen_types = set()
        unique_goals = []
        for goal in sorted(goals, key=lambda g: -g.priority):
            if goal.viz_type not in seen_types:
                seen_types.add(goal.viz_type)
                unique_goals.append(goal)

        return unique_goals[:max_goals]

    def get_best_goal(self, content: str, title: str = "") -> VisualizationGoal:
        """Get the single best visualization goal for content."""
        goals = self.generate_goals(content, title, max_goals=1)
        return goals[0] if goals else self.GOAL_TEMPLATES["conceptual"][0]


