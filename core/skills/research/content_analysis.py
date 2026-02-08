"""
Content Analysis - Pattern learning, content type detection, and component mapping.

Extracted from html_slide_generator.py.
"""

"""
Jotty HTML Slide Generator
Generates world-class interactive HTML presentations from research paper data.
40+ slide types with intelligent content-aware selection, animations, and responsive design.

Features:
- Intelligent slide type selection based on content analysis
- DSPy-powered recommendations for optimal visualization
- Visual rhythm balancing (text → visual → data → text)
- Pattern detection for formulas, comparisons, timelines, etc.
- Schema-driven LLM output with minimal context (6 meta-schemas → 400+ components)
- Auto-learning pattern crystallization (inspired by OpenClaw Foundry)
- Beautiful.ai-style research presentation templates
"""

import json
import re
import logging
import random
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import html

# DSPy for LLM-based slide selection
try:
    import dspy
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False
    dspy = None

logger = logging.getLogger(__name__)


# =============================================================================
# AUTO-LEARNING PATTERN SYSTEM (Inspired by OpenClaw Foundry)
# =============================================================================
# Tracks content→component mappings, learns what works, crystallizes patterns.
# The system that improves itself over time.


@dataclass
class ComponentPattern:
    """A learned pattern for content→component mapping."""
    content_hash: str           # Hash of content patterns detected
    content_patterns: List[str] # ContentPattern names that triggered this
    component: str              # Component that was selected
    meta_schema: str            # Which meta-schema was used
    success_score: float = 0.0  # 0-1, based on presentation feedback
    usage_count: int = 1        # How many times this pattern was used
    last_used: str = ""         # ISO timestamp
    crystallized: bool = False  # True if pattern is proven (5+ uses, 70%+ success)


class PatternCrystallizer:
    """
    Auto-learning system that tracks and crystallizes successful patterns.

    Inspired by OpenClaw Foundry's pattern crystallization:
    - Observe: Track content patterns → component selections
    - Learn: Calculate success rates from feedback
    - Crystallize: Promote high-value patterns (5+ uses, 70%+ success)
    - Apply: Use crystallized patterns for future selections

    Storage: JSON file in ~/.jotty/learned_patterns.json
    """

    CRYSTALLIZATION_THRESHOLD = 5    # Min uses to crystallize
    SUCCESS_THRESHOLD = 0.7          # Min success rate to crystallize
    STALE_DAYS = 30                  # Prune patterns unused for this long

    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = Path(storage_path or Path.home() / ".jotty" / "learned_patterns.json")
        self.patterns: Dict[str, ComponentPattern] = {}
        self._load()

    def _load(self):
        """Load patterns from storage."""
        if self.storage_path.exists():
            try:
                data = json.loads(self.storage_path.read_text())
                for key, p in data.get("patterns", {}).items():
                    self.patterns[key] = ComponentPattern(**p)
                logger.info(f"📚 Loaded {len(self.patterns)} learned patterns")
            except Exception as e:
                logger.warning(f"Could not load patterns: {e}")

    def _save(self):
        """Save patterns to storage."""
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "patterns": {k: {
                    "content_hash": p.content_hash,
                    "content_patterns": p.content_patterns,
                    "component": p.component,
                    "meta_schema": p.meta_schema,
                    "success_score": p.success_score,
                    "usage_count": p.usage_count,
                    "last_used": p.last_used,
                    "crystallized": p.crystallized,
                } for k, p in self.patterns.items()},
                "updated": datetime.now().isoformat(),
            }
            self.storage_path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.warning(f"Could not save patterns: {e}")

    def _hash_patterns(self, content_patterns: List[str]) -> str:
        """Create stable hash for a set of content patterns."""
        sorted_patterns = sorted(content_patterns)
        return hashlib.md5("|".join(sorted_patterns).encode()).hexdigest()[:12]

    def record(self, content_patterns: List[str], component: str, meta_schema: str):
        """Record a content→component mapping."""
        pattern_names = [p.name if hasattr(p, 'name') else str(p) for p in content_patterns]
        key = f"{self._hash_patterns(pattern_names)}_{component}"

        if key in self.patterns:
            self.patterns[key].usage_count += 1
            self.patterns[key].last_used = datetime.now().isoformat()
        else:
            self.patterns[key] = ComponentPattern(
                content_hash=self._hash_patterns(pattern_names),
                content_patterns=pattern_names,
                component=component,
                meta_schema=meta_schema,
                last_used=datetime.now().isoformat(),
            )

        # Check for crystallization
        pattern = self.patterns[key]
        if (pattern.usage_count >= self.CRYSTALLIZATION_THRESHOLD and
            pattern.success_score >= self.SUCCESS_THRESHOLD and
            not pattern.crystallized):
            pattern.crystallized = True
            logger.info(f"✨ Crystallized pattern: {pattern_names} → {component}")

        self._save()

    def feedback(self, content_patterns: List[str], component: str, success: bool):
        """Provide feedback on a component selection."""
        pattern_names = [p.name if hasattr(p, 'name') else str(p) for p in content_patterns]
        key = f"{self._hash_patterns(pattern_names)}_{component}"

        if key in self.patterns:
            pattern = self.patterns[key]
            # Exponential moving average for success score
            alpha = 0.3
            new_score = 1.0 if success else 0.0
            pattern.success_score = alpha * new_score + (1 - alpha) * pattern.success_score
            self._save()

    def suggest(self, content_patterns: List[str]) -> Optional[str]:
        """Suggest a component based on crystallized patterns."""
        pattern_names = [p.name if hasattr(p, 'name') else str(p) for p in content_patterns]
        content_hash = self._hash_patterns(pattern_names)

        # Find crystallized patterns matching this content
        candidates = [
            p for p in self.patterns.values()
            if p.content_hash == content_hash and p.crystallized
        ]

        if candidates:
            # Return highest success score
            best = max(candidates, key=lambda p: p.success_score)
            logger.debug(f"💡 Suggesting crystallized: {best.component} (score: {best.success_score:.0%})")
            return best.component

        return None

    def get_stats(self) -> Dict:
        """Get learning statistics."""
        crystallized = sum(1 for p in self.patterns.values() if p.crystallized)
        return {
            "total_patterns": len(self.patterns),
            "crystallized": crystallized,
            "avg_success": sum(p.success_score for p in self.patterns.values()) / max(len(self.patterns), 1),
            "total_uses": sum(p.usage_count for p in self.patterns.values()),
        }


# =============================================================================
# BEAUTIFUL.AI-STYLE RESEARCH TEMPLATES
# =============================================================================
# Professional research presentation structure with assertion-evidence approach.

class ResearchTemplateStructure:
    """
    Beautiful.ai-inspired research presentation structure.

    Follows the assertion-evidence approach:
    - Each slide title = assertion (key statement)
    - Slide content = evidence (supporting data/visuals)

    Standard research narrative arc:
    1. Title → 2. Team → 3. Agenda → 4. Methodology →
    5. Findings (multiple) → 6. Analysis → 7. Conclusion → 8. Q&A
    """

    STANDARD_FLOW = [
        {"type": "TITLE_HERO", "purpose": "Hook and context", "required": True},
        {"type": "STATS_GRID", "purpose": "Overview metrics", "required": False},
        {"type": "FEATURE_CARDS", "purpose": "Key concepts preview", "required": False},
        # Dynamic concept/section slides inserted here
        {"type": "COMPARISON_TABLE", "purpose": "Results comparison", "required": False},
        {"type": "KEY_TAKEAWAYS", "purpose": "Summary", "required": True},
        {"type": "ICON_GRID", "purpose": "Next steps", "required": False},
        {"type": "AUTHORS", "purpose": "Credits", "required": False},
        {"type": "QA", "purpose": "Discussion", "required": True},
    ]

    # Component recommendations by content type (assertion-evidence pairings)
    ASSERTION_EVIDENCE_MAP = {
        # Assertion type → best evidence component
        "introduces_concept": ["DEFINITION", "TWO_COLUMN", "FEATURE_CARDS"],
        "compares_approaches": ["BEFORE_AFTER", "COMPARISON_TABLE", "PROS_CONS"],
        "shows_process": ["PROCESS_STEPS", "TIMELINE", "FLOWCHART"],
        "presents_data": ["STATS_GRID", "CHART_BAR", "PROGRESS_BARS"],
        "explains_math": ["FORMULA", "TWO_COLUMN"],
        "demonstrates_code": ["CODE_BLOCK"],
        "highlights_insight": ["QUOTE", "DEFINITION"],
        "lists_benefits": ["BULLET_POINTS", "ICON_GRID", "CHECKLIST"],
    }

    @classmethod
    def classify_assertion(cls, title: str, content: str) -> str:
        """Classify what type of assertion a slide is making."""
        title_lower = title.lower()
        content_lower = content.lower()

        if any(w in title_lower for w in ["what is", "definition", "introduction", "overview"]):
            return "introduces_concept"
        if any(w in title_lower for w in ["vs", "comparison", "difference", "before", "after"]):
            return "compares_approaches"
        if any(w in title_lower for w in ["how", "process", "step", "workflow", "pipeline"]):
            return "shows_process"
        if any(w in title_lower for w in ["result", "performance", "metric", "accuracy", "%"]):
            return "presents_data"
        if any(w in title_lower for w in ["equation", "formula", "math", "theorem"]):
            return "explains_math"
        if any(w in title_lower for w in ["code", "implementation", "algorithm"]):
            return "demonstrates_code"
        if any(w in title_lower for w in ["insight", "key", "important", "takeaway"]):
            return "highlights_insight"
        if any(w in title_lower for w in ["benefit", "advantage", "feature", "why"]):
            return "lists_benefits"

        # Content-based fallback
        if "$" in content or "\\(" in content:
            return "explains_math"
        if "```" in content or "def " in content:
            return "demonstrates_code"

        return "introduces_concept"  # Default

    @classmethod
    def get_evidence_components(cls, assertion_type: str) -> List[str]:
        """Get recommended components for an assertion type."""
        return cls.ASSERTION_EVIDENCE_MAP.get(assertion_type, ["TWO_COLUMN"])


# =============================================================================
# MINIMAL META-SCHEMA SYSTEM (6 schemas → 400+ components)
# =============================================================================
# These 6 simple formats are all the LLM needs to output.
# ComponentMapper handles intelligent mapping to specific slide types.

class MetaSchemaRegistry:
    """
    Minimal schema registry for LLM output.
    Only 6 meta-schemas to minimize context, but maps to 40+ slide types.

    Design principle: LLM outputs simple structure, we add creativity in mapping.
    """

    SCHEMAS = {
        "points": {
            "description": "List of key points with optional descriptions",
            "format": {"title": "str", "points": [{"title": "str", "description": "str?"}]},
            "maps_to": ["BULLET_POINTS", "KEY_TAKEAWAYS", "CHECKLIST", "ICON_GRID", "FEATURE_CARDS", "NUMBERED_LIST"],
        },
        "comparison": {
            "description": "Two-sided comparison (before/after, pros/cons, left/right)",
            "format": {"title": "str", "side_a": {"title": "str", "items": ["str"]}, "side_b": {"title": "str", "items": ["str"]}},
            "maps_to": ["BEFORE_AFTER", "PROS_CONS", "TWO_COLUMN", "SIDE_BY_SIDE", "COMPARISON_TABLE"],
        },
        "sequence": {
            "description": "Ordered steps or timeline events",
            "format": {"title": "str", "items": [{"label": "str", "title": "str", "description": "str?"}]},
            "maps_to": ["PROCESS_STEPS", "TIMELINE", "FLOWCHART", "NUMBERED_LIST"],
        },
        "metrics": {
            "description": "Numerical data with labels",
            "format": {"title": "str", "stats": [{"value": "str", "label": "str", "trend": "str?"}]},
            "maps_to": ["STATS_GRID", "STATS_INLINE", "PROGRESS_BARS", "CHART_BAR"],
        },
        "highlight": {
            "description": "Single important concept (quote, definition, formula)",
            "format": {"title": "str", "content": "str", "attribution": "str?", "context": "str?"},
            "maps_to": ["QUOTE", "DEFINITION", "FORMULA", "CODE_BLOCK"],
        },
        "cards": {
            "description": "Grid of feature cards with icons",
            "format": {"title": "str", "cards": [{"icon": "str", "title": "str", "description": "str"}]},
            "maps_to": ["FEATURE_CARDS", "ICON_GRID", "ADVANTAGES", "TEAM_GRID"],
        },
    }

    @classmethod
    def get_prompt_schemas(cls) -> str:
        """Get minimal schema descriptions for LLM prompt (low token count)."""
        lines = ["Output JSON matching ONE of these formats:"]
        for name, schema in cls.SCHEMAS.items():
            lines.append(f"- {name}: {schema['description']}")
        return "\n".join(lines)

    @classmethod
    def get_schema_for_content(cls, content: str, patterns: List['ContentPattern']) -> str:
        """Suggest best meta-schema based on content patterns."""
        pattern_map = {
            ContentPattern.STEP_BY_STEP: "sequence",
            ContentPattern.TIMELINE: "sequence",
            ContentPattern.COMPARISON: "comparison",
            ContentPattern.PROS_CONS: "comparison",
            ContentPattern.STATISTICS: "metrics",
            ContentPattern.QUOTE: "highlight",
            ContentPattern.DEFINITION: "highlight",
            ContentPattern.MATH_FORMULA: "highlight",
            ContentPattern.CODE_SNIPPET: "highlight",
            ContentPattern.LIST_ITEMS: "points",
            ContentPattern.ARCHITECTURE: "cards",
        }
        for pattern in patterns:
            if pattern in pattern_map:
                return pattern_map[pattern]
        return "points"  # Default


class ComponentMapper:
    """
    Maps meta-schema output to specific slide components with variety.
    Now with auto-learning from PatternCrystallizer and assertion-evidence approach.

    Features:
    - Crystallized pattern suggestions (learned from successful presentations)
    - Assertion-evidence component matching
    - Variety balancing
    - Beautiful.ai-style research flow
    """

    def __init__(self, enable_learning: bool = True):
        self.used_components: List[str] = []
        self.crystallizer = PatternCrystallizer() if enable_learning else None
        self.component_pools = {
            "points": ["BULLET_POINTS", "KEY_TAKEAWAYS", "CHECKLIST", "ICON_GRID", "FEATURE_CARDS"],
            "comparison": ["BEFORE_AFTER", "PROS_CONS", "TWO_COLUMN", "COMPARISON_TABLE"],
            "sequence": ["PROCESS_STEPS", "TIMELINE", "FLOWCHART"],
            "metrics": ["STATS_GRID", "PROGRESS_BARS", "CHART_BAR"],
            "highlight": ["QUOTE", "DEFINITION", "FORMULA"],
            "cards": ["FEATURE_CARDS", "ICON_GRID", "ADVANTAGES"],
        }

    def reset(self):
        self.used_components = []

    def map_to_component(
        self,
        meta_schema: str,
        data: Dict,
        content_patterns: List['ContentPattern'],
        title: str = "",
        content: str = "",
        force_variety: bool = True
    ) -> Tuple[str, Dict]:
        """
        Map meta-schema output to a specific component type with variety.
        Uses auto-learning and assertion-evidence approach.

        Returns:
            Tuple of (component_type, transformed_data)
        """
        pool = self.component_pools.get(meta_schema, ["BULLET_POINTS"])

        # 1. Check for crystallized pattern (learned from past success)
        if self.crystallizer:
            crystallized = self.crystallizer.suggest(content_patterns)
            if crystallized and crystallized in pool:
                logger.debug(f"🔮 Using crystallized pattern: {crystallized}")
                self.used_components.append(crystallized)
                self.crystallizer.record(content_patterns, crystallized, meta_schema)
                return crystallized, self._transform_data(meta_schema, crystallized, data)

        # 2. Use assertion-evidence approach for research presentations
        if title or content:
            assertion_type = ResearchTemplateStructure.classify_assertion(title, content)
            evidence_components = ResearchTemplateStructure.get_evidence_components(assertion_type)
            # Prefer evidence components that are in our pool
            evidence_in_pool = [c for c in evidence_components if c in pool]
            if evidence_in_pool:
                pool = evidence_in_pool + [c for c in pool if c not in evidence_in_pool]

        # 3. Content-based preferences
        preferred = self._get_pattern_preferences(content_patterns, meta_schema)

        # 4. Filter out recently used for variety
        if force_variety and len(self.used_components) >= 2:
            recent = set(self.used_components[-4:])
            available = [c for c in pool if c not in recent]
            if not available:
                available = pool
        else:
            available = pool

        # 5. Select: prefer content-matched, then first available
        candidates = [c for c in preferred if c in available]
        if candidates:
            selected = candidates[0]
        else:
            selected = available[0] if available else random.choice(pool)

        self.used_components.append(selected)

        # 6. Record pattern for learning
        if self.crystallizer:
            self.crystallizer.record(content_patterns, selected, meta_schema)

        # Transform data to match component schema
        transformed = self._transform_data(meta_schema, selected, data)

        return selected, transformed

    def provide_feedback(self, content_patterns: List['ContentPattern'], component: str, success: bool):
        """Provide feedback on a component selection to improve learning."""
        if self.crystallizer:
            self.crystallizer.feedback(content_patterns, component, success)

    def get_learning_stats(self) -> Dict:
        """Get auto-learning statistics."""
        if self.crystallizer:
            return self.crystallizer.get_stats()
        return {"learning_enabled": False}

    def _get_pattern_preferences(self, patterns: List['ContentPattern'], meta_schema: str) -> List[str]:
        """Get preferred components based on content patterns."""
        prefs = []
        for pattern in patterns:
            if pattern == ContentPattern.MATH_FORMULA:
                prefs.append("FORMULA")
            elif pattern == ContentPattern.CODE_SNIPPET:
                prefs.append("CODE_BLOCK")
            elif pattern == ContentPattern.QUOTE:
                prefs.append("QUOTE")
            elif pattern == ContentPattern.TIMELINE:
                prefs.append("TIMELINE")
            elif pattern == ContentPattern.STATISTICS:
                prefs.extend(["STATS_GRID", "CHART_BAR"])
            elif pattern == ContentPattern.COMPARISON:
                prefs.append("BEFORE_AFTER")
            elif pattern == ContentPattern.PROS_CONS:
                prefs.append("PROS_CONS")
            elif pattern == ContentPattern.STEP_BY_STEP:
                prefs.append("PROCESS_STEPS")
        return prefs

    def _transform_data(self, meta_schema: str, component: str, data: Dict) -> Dict:
        """Transform meta-schema data to component-specific format."""
        # Most transformations are straightforward field mappings
        if meta_schema == "points" and component == "ICON_GRID":
            # Add icons to points
            icons = ["🎯", "💡", "⚡", "🔧", "📊", "🚀", "✨", "🔬"]
            return {
                "label": "Key Points",
                "title": data.get("title", ""),
                "items": [
                    {"icon": icons[i % len(icons)], "title": p.get("title", ""), "description": p.get("description", "")}
                    for i, p in enumerate(data.get("points", []))
                ]
            }
        elif meta_schema == "comparison" and component == "PROS_CONS":
            return {
                "label": "Analysis",
                "title": data.get("title", ""),
                "pros": [{"title": item, "description": ""} for item in data.get("side_a", {}).get("items", [])],
                "cons": [{"title": item, "description": ""} for item in data.get("side_b", {}).get("items", [])],
            }
        elif meta_schema == "highlight" and component == "QUOTE":
            return {
                "quote": data.get("content", ""),
                "author": data.get("attribution", ""),
                "source": data.get("context", ""),
            }
        elif meta_schema == "highlight" and component == "DEFINITION":
            return {
                "term": data.get("title", ""),
                "definition": data.get("content", ""),
                "also_known_as": [],
            }
        elif meta_schema == "sequence" and component == "TIMELINE":
            return {
                "label": "Timeline",
                "title": data.get("title", ""),
                "events": [
                    {"year": item.get("label", ""), "title": item.get("title", ""), "description": item.get("description", "")}
                    for item in data.get("items", [])
                ]
            }
        # Default: return as-is with label added
        result = {"label": meta_schema.title(), **data}
        return result

    def get_variety_score(self) -> float:
        """Calculate variety score (0-1, higher = more varied)."""
        if len(self.used_components) < 2:
            return 1.0
        unique = len(set(self.used_components))
        return unique / len(self.used_components)


# =============================================================================
# INTELLIGENT SLIDE TYPE SELECTION SYSTEM
# =============================================================================

class ContentPattern(Enum):
    """Detected content patterns that influence slide type selection"""
    MATH_FORMULA = "math_formula"           # Contains equations, formulas
    STEP_BY_STEP = "step_by_step"           # Sequential steps or process
    COMPARISON = "comparison"                # Before/after, A vs B
    PROS_CONS = "pros_cons"                  # Advantages/disadvantages
    TIMELINE = "timeline"                    # Historical progression
    CODE_SNIPPET = "code_snippet"            # Programming code
    STATISTICS = "statistics"                # Numbers, percentages, metrics
    DEFINITION = "definition"                # Term definition
    QUOTE = "quote"                          # Notable quote
    LIST_ITEMS = "list_items"                # Bullet points
    ARCHITECTURE = "architecture"            # System/model architecture
    FLOWCHART = "flowchart"                  # Process flow
    TABLE_DATA = "table_data"                # Tabular comparisons
    VISUAL_CONCEPT = "visual_concept"        # Needs diagram/visualization
    NARRATIVE = "narrative"                  # Story/explanation text


@dataclass
class ContentAnalysis:
    """Result of analyzing content for slide type selection"""
    patterns: List[ContentPattern]
    has_math: bool = False
    has_code: bool = False
    has_numbers: bool = False
    has_comparison: bool = False
    has_steps: bool = False
    has_list: bool = False
    word_count: int = 0
    sentence_count: int = 0
    complexity_score: float = 0.0  # 0-1, higher = more complex
    suggested_types: List[str] = field(default_factory=list)
    confidence: float = 0.0


class ContentAnalyzer:
    """
    Analyzes content to detect patterns and recommend slide types.
    Uses regex patterns and heuristics for fast, accurate detection.
    """

    # Pattern definitions for content detection
    MATH_PATTERNS = [
        r'\$[^$]+\$',                    # LaTeX inline: $...$
        r'\$\$[^$]+\$\$',                # LaTeX display: $$...$$
        r'\\[\[\(].+?\\[\]\)]',          # LaTeX \[...\] or \(...\)
        r'[α-ωΑ-Ω∑∏∫∂∇×÷±≤≥≠≈∞]',       # Greek/math symbols
        r'\b[a-z]\s*=\s*[^,\n]{3,}',     # Equations like x = ...
        r'\b(?:sin|cos|tan|log|exp|sqrt)\b',  # Math functions
        r'(?:\d+\s*[+\-*/^]\s*)+\d+',    # Arithmetic expressions
    ]

    STEP_PATTERNS = [
        r'(?:step\s*\d+|first|second|third|finally|next|then)\s*[:,]',
        r'^\s*\d+\.\s+\w',               # Numbered list
        r'(?:begin|start)\s+(?:by|with)',
    ]

    COMPARISON_PATTERNS = [
        r'\b(?:before|after)\b.*\b(?:before|after)\b',
        r'\b(?:vs\.?|versus|compared to|in contrast)\b',
        r'\b(?:traditional|conventional|previous)\b.*\b(?:new|proposed|our)\b',
        r'\b(?:better|worse|faster|slower|more|less)\s+than\b',
    ]

    PROS_CONS_PATTERNS = [
        r'\b(?:advantage|disadvantage|pro|con|benefit|drawback)\b',
        r'\b(?:strength|weakness|positive|negative)\b',
        r'(?:✓|✗|✔|✘|👍|👎)',
    ]

    CODE_PATTERNS = [
        r'```[\w]*\n',                   # Code block
        r'\bdef\s+\w+\s*\(',             # Python function
        r'\bclass\s+\w+',                # Class definition
        r'\bimport\s+\w+',               # Import statement
        r'\breturn\s+\w+',               # Return statement
        r'(?:->|=>)\s*\{',               # Arrow functions
    ]

    QUOTE_PATTERNS = [
        r'^["\'].*["\']$',               # Quoted text
        r'—\s*\w+',                       # Attribution dash
        r'\b(?:said|stated|wrote|argued)\b',
    ]

    STATS_PATTERNS = [
        r'\d+(?:\.\d+)?%',               # Percentages
        r'\d+(?:,\d{3})+',               # Large numbers with commas
        r'\b\d+x\b',                      # Multipliers (10x, 100x)
        r'\b(?:billion|million|thousand)\b',
        r'(?:accuracy|precision|recall|F1)\s*[:=]\s*\d+',
    ]

    TIMELINE_PATTERNS = [
        r'\b(?:19|20)\d{2}\b',           # Years
        r'\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\b',
        r'\b(?:evolution|history|timeline|progression)\b',
        r'(?:first|then|later|finally|eventually)',
    ]

    ARCHITECTURE_PATTERNS = [
        r'\b(?:layer|module|component|block|encoder|decoder)\b',
        r'\b(?:input|output|hidden)\s+(?:layer|size|dimension)\b',
        r'\b(?:architecture|structure|design|framework)\b',
        r'→|➜|->',                        # Arrows indicating flow
    ]

    def analyze(self, content: str, title: str = "") -> ContentAnalysis:
        """
        Analyze content and return detected patterns with slide type suggestions.

        Args:
            content: The text content to analyze
            title: Optional title for additional context

        Returns:
            ContentAnalysis with detected patterns and recommendations
        """
        if not content:
            return ContentAnalysis(patterns=[], suggested_types=["BULLET_POINTS"])

        full_text = f"{title} {content}".lower()
        patterns = []
        suggested_types = []

        # Detect patterns
        has_math = self._matches_any(content, self.MATH_PATTERNS)
        has_steps = self._matches_any(full_text, self.STEP_PATTERNS)
        has_comparison = self._matches_any(full_text, self.COMPARISON_PATTERNS)
        has_pros_cons = self._matches_any(full_text, self.PROS_CONS_PATTERNS)
        has_code = self._matches_any(content, self.CODE_PATTERNS)
        has_quote = self._matches_any(content, self.QUOTE_PATTERNS)
        has_stats = self._matches_any(content, self.STATS_PATTERNS)
        has_timeline = self._matches_any(full_text, self.TIMELINE_PATTERNS)
        has_architecture = self._matches_any(full_text, self.ARCHITECTURE_PATTERNS)

        # Build pattern list and suggestions
        if has_math:
            patterns.append(ContentPattern.MATH_FORMULA)
            suggested_types.append("FORMULA")

        if has_code:
            patterns.append(ContentPattern.CODE_SNIPPET)
            suggested_types.append("CODE_BLOCK")

        if has_steps:
            patterns.append(ContentPattern.STEP_BY_STEP)
            suggested_types.extend(["PROCESS_STEPS", "TIMELINE"])

        if has_comparison:
            patterns.append(ContentPattern.COMPARISON)
            suggested_types.extend(["BEFORE_AFTER", "COMPARISON_TABLE"])

        if has_pros_cons:
            patterns.append(ContentPattern.PROS_CONS)
            suggested_types.append("PROS_CONS")

        if has_quote:
            patterns.append(ContentPattern.QUOTE)
            suggested_types.append("QUOTE")

        if has_stats:
            patterns.append(ContentPattern.STATISTICS)
            suggested_types.extend(["STATS_GRID", "CHART_BAR"])

        if has_timeline:
            patterns.append(ContentPattern.TIMELINE)
            suggested_types.append("TIMELINE")

        if has_architecture:
            patterns.append(ContentPattern.ARCHITECTURE)
            suggested_types.extend(["ARCHITECTURE", "DIAGRAM"])

        # Count metrics
        word_count = len(content.split())
        sentence_count = len(re.split(r'[.!?]+', content))

        # Check for list items
        has_list = bool(re.search(r'(?:^|\n)\s*[-•*]\s+\w', content)) or \
                   bool(re.search(r'(?:^|\n)\s*\d+\.\s+\w', content))
        if has_list:
            patterns.append(ContentPattern.LIST_ITEMS)
            suggested_types.append("BULLET_POINTS")

        # Calculate complexity
        complexity = min(1.0, (word_count / 500) * 0.5 + (len(patterns) / 5) * 0.5)

        # Default suggestions if none found
        if not suggested_types:
            if word_count > 300:
                suggested_types = ["TWO_COLUMN", "SPLIT_CONTENT"]
            else:
                suggested_types = ["BULLET_POINTS", "DEFINITION"]

        # Calculate confidence based on pattern matches
        confidence = min(1.0, len(patterns) * 0.2 + 0.3)

        return ContentAnalysis(
            patterns=patterns,
            has_math=has_math,
            has_code=has_code,
            has_numbers=has_stats,
            has_comparison=has_comparison,
            has_steps=has_steps,
            has_list=has_list,
            word_count=word_count,
            sentence_count=sentence_count,
            complexity_score=complexity,
            suggested_types=suggested_types,
            confidence=confidence
        )

    def _matches_any(self, text: str, patterns: List[str]) -> bool:
        """Check if text matches any of the given regex patterns."""
        for pattern in patterns:
            try:
                if re.search(pattern, text, re.IGNORECASE | re.MULTILINE):
                    return True
            except re.error:
                continue
        return False


