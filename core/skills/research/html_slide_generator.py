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
                logger.info("🧠 LLM slide selector initialized")
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


class SlideType(Enum):
    """Available slide component types"""
    # Title & Hero Slides
    TITLE_HERO = "title_hero"
    TITLE_MINIMAL = "title_minimal"
    TITLE_CENTERED = "title_centered"
    TITLE_SPLIT = "title_split"

    # Content Slides
    BULLET_POINTS = "bullet_points"
    NUMBERED_LIST = "numbered_list"
    TWO_COLUMN = "two_column"
    THREE_COLUMN = "three_column"
    SPLIT_CONTENT = "split_content"

    # Visual Slides
    ARCHITECTURE = "architecture"
    DIAGRAM = "diagram"
    FLOWCHART = "flowchart"
    PROCESS_STEPS = "process_steps"
    TIMELINE = "timeline"

    # Data Slides
    STATS_GRID = "stats_grid"
    STATS_INLINE = "stats_inline"
    COMPARISON_TABLE = "comparison_table"
    CHART_BAR = "chart_bar"
    CHART_LINE = "chart_line"
    CHART_RADAR = "chart_radar"
    PROGRESS_BARS = "progress_bars"

    # Feature Slides
    FEATURE_CARDS = "feature_cards"
    FEATURE_ICONS = "feature_icons"
    ICON_GRID = "icon_grid"
    ADVANTAGES = "advantages"

    # Comparison Slides
    BEFORE_AFTER = "before_after"
    PROS_CONS = "pros_cons"
    SIDE_BY_SIDE = "side_by_side"

    # Special Slides
    QUOTE = "quote"
    DEFINITION = "definition"
    FORMULA = "formula"
    CODE_BLOCK = "code_block"
    CHECKLIST = "checklist"

    # Team & Credits
    AUTHORS = "authors"
    TEAM_GRID = "team_grid"

    # Ending Slides
    KEY_TAKEAWAYS = "key_takeaways"
    SUMMARY = "summary"
    QA = "qa"
    THANK_YOU = "thank_you"
    REFERENCES = "references"


@dataclass
class SlideConfig:
    """Configuration for a single slide"""
    slide_type: SlideType
    data: Dict[str, Any]
    animation: str = "slide_up"  # slide_up, slide_left, fade, scale, bounce
    delay_start: float = 0.1


@dataclass
class PresentationConfig:
    """Configuration for the entire presentation"""
    title: str
    arxiv_id: str = ""
    authors: List[str] = field(default_factory=list)
    theme: str = "navy"  # navy, dark, light
    accent_color: str = "#f6ad55"  # gold
    branding: str = "Jotty Learning"


class HTMLSlideGenerator:
    """Generates complete HTML slide presentations"""

    THEME_COLORS = {
        "navy": {
            "bg_primary": "#0a1929",
            "bg_secondary": "#102a43",
            "bg_tertiary": "#243b53",
            "bg_card": "#1a365d",
            "text_primary": "#ffffff",
            "text_secondary": "#9fb3c8",
            "text_muted": "#627d98",
            "border": "#334e68",
            "accent_blue": "#4299e1",
            "accent_purple": "#9f7aea",
            "accent_teal": "#38b2ac",
            "accent_pink": "#ed64a6",
            "accent_green": "#48bb78",
            "accent_red": "#f56565",
        }
    }

    def __init__(self, config: PresentationConfig):
        self.config = config
        self.colors = self.THEME_COLORS.get(config.theme, self.THEME_COLORS["navy"])
        self.slides: List[SlideConfig] = []

    def add_slide(self, slide_type: SlideType, data: Dict[str, Any],
                  animation: str = "slide_up", delay_start: float = 0.1):
        """Add a slide to the presentation"""
        self.slides.append(SlideConfig(
            slide_type=slide_type,
            data=data,
            animation=animation,
            delay_start=delay_start
        ))

    def _format_text(self, text: str) -> str:
        """Format short text with basic markdown (bold, italic).
        Use for titles, bullet points, short descriptions.
        """
        if not text:
            return ""

        # Clean up malformed markdown before escaping
        # Remove stray ** used as bullets (** at start of text/line followed by space)
        text = re.sub(r'^(\s*)\*\*\s+', r'\1', text, flags=re.MULTILINE)
        # Remove trailing ** without opening
        text = re.sub(r'\*\*$', '', text)

        # First escape HTML
        text = html.escape(text)

        # Handle bold **text** (allow content to span lines with [\s\S]+?)
        text = re.sub(r'\*\*(.+?)\*\*', r'<strong class="text-white font-semibold">\1</strong>', text, flags=re.DOTALL)

        # Handle italic *text* (but not if it looks like math)
        text = re.sub(r'(?<![*\w])\*([^\s*][^*\n]{0,48}[^\s*])\*(?![*\w])', r'<em class="text-blue-200">\1</em>', text)

        return text

    def _format_content(self, text: str) -> str:
        """Format content with markdown, LaTeX, and step handling.

        Handles:
        - Step formatting (Step 1:, Step 2:, etc.) - ensures each step is on new line
        - Math expressions ($...$, $$...$$) - wraps in styled spans
        - Bold (**text**) and italic (*text*)
        - Code (`code`)
        - Headers (###, ####)
        - Numbered lists (1., 2., etc.)
        - Bullet lists (- item)
        """
        import re

        if not text:
            return ""

        # =================================================================
        # STEP 0: PROTECT CODE BLOCKS FIRST (before any processing)
        # =================================================================
        # Store code blocks and replace with placeholders
        code_blocks = []
        inline_codes = []

        def store_code_block(match):
            code_blocks.append(match.group(0))
            return f'__CODE_BLOCK_{len(code_blocks) - 1}__'

        def store_inline_code(match):
            inline_codes.append(match.group(0))
            return f'__INLINE_CODE_{len(inline_codes) - 1}__'

        # Extract code blocks first (before HTML escaping)
        text = re.sub(r'```[\w]*\n?(.*?)```', store_code_block, text, flags=re.DOTALL)
        text = re.sub(r'`([^`]+)`', store_inline_code, text)

        # Clean up malformed markdown before escaping
        # Remove stray ** used as bullets (** at start of text/line followed by space)
        text = re.sub(r'^(\s*)\*\*\s+', r'\1', text, flags=re.MULTILINE)
        # Remove trailing ** without opening
        text = re.sub(r'\*\*$', '', text)

        # Now escape HTML
        text = html.escape(text)

        # =================================================================
        # STEP 1: Handle Step formatting COMPREHENSIVELY
        # =================================================================
        # First, handle already bold steps (from **Step N:**)
        # Pattern: **Step N:** or **Step N.** - already handled by markdown

        # Handle "Step N:" pattern - make bold and add line break before
        # This handles: "Step 1:", "Step 2:", etc. anywhere in text
        def format_step(match):
            prefix = match.group(1) or ''
            step_num = match.group(2)
            # Check if already wrapped in strong
            if '<strong' in prefix:
                return match.group(0)
            return f'{prefix}<br/><br/><strong class="text-white font-semibold">Step {step_num}:</strong>'

        # Match Step N: with optional preceding text (period, newline, or start)
        text = re.sub(r'(^|[.!?]\s*|\n\s*)Step\s*(\d+)\s*:', format_step, text)

        # =================================================================
        # STEP 2: Handle markdown bold **text** (allow multiline with DOTALL)
        # =================================================================
        text = re.sub(r'\*\*(.+?)\*\*', r'<strong class="text-white font-semibold">\1</strong>', text, flags=re.DOTALL)

        # =================================================================
        # STEP 3: Handle LaTeX/Math expressions (MathJax-compatible)
        # =================================================================
        # Protect existing LaTeX delimiters for MathJax processing
        math_placeholders = []

        def protect_math(match):
            math_placeholders.append(match.group(0))
            return f'__MATH_{len(math_placeholders) - 1}__'

        # Protect display math $$...$$ and \[...\]
        text = re.sub(r'\$\$[^$]+\$\$', protect_math, text)
        text = re.sub(r'\\\[[^\]]+\\\]', protect_math, text)
        # Protect inline math $...$ and \(...\) (skip pure numbers like $50)
        text = re.sub(r'\$(?![0-9]+\$)[^$\n]+?\$', protect_math, text)
        text = re.sub(r'\\\([^)]+\\\)', protect_math, text)

        # Detect ASCII math formulas and wrap for MathJax
        ascii_formula_patterns = [
            # Full equations: L_G = ..., p_data = ...
            r'[A-Z]_[A-Za-z]+\s*=\s*[^\.\n]+',
            r'[A-Z]\*?\([^)]+\)\s*=\s*[^\.\n]+',
            # Expectations: E_z[...], E[...]
            r'E_?[a-z]?\[[^\]]+\]',
            # Nested functions: D(G(z)), G(x)
            r'[A-Z]\([A-Z]\([^)]+\)\)',
            # Subscripted terms: p_data(x), L_D
            r'[a-z]_[a-z]+\([^)]+\)',
        ]

        def convert_ascii_to_latex(formula: str) -> str:
            """Convert ASCII math notation to proper LaTeX."""
            latex = formula
            # Tilde subscript: E_x~p_data → E_{x \sim p_{data}}
            latex = re.sub(r'~([a-z])_([a-z]+)', r' \\sim \1_{\2}', latex)
            latex = re.sub(r'~([a-z_]+)', r' \\sim \1', latex)
            # Subscripts: L_G → L_{G}, p_data → p_{data}
            latex = re.sub(r'([A-Za-z])_([A-Za-z0-9]+)', r'\1_{\2}', latex)
            # Superscript asterisk: D* → D^{*}
            latex = re.sub(r'([A-Za-z])\*', r'\1^{*}', latex)
            # Infinity: infty → \infty
            latex = re.sub(r'\binfty\b', r'\\infty', latex)
            # Sum, integral: sum → \sum, int → \int
            latex = re.sub(r'\bsum\b', r'\\sum', latex)
            latex = re.sub(r'\bint\b', r'\\int', latex)
            return latex

        for pattern in ascii_formula_patterns:
            def wrap_ascii_formula(match):
                formula = match.group(0).strip()
                # Don't double-wrap
                if formula.startswith('$'):
                    return formula
                # Convert ASCII to LaTeX
                latex = convert_ascii_to_latex(formula)
                # Wrap and protect
                wrapped = f'${latex}$'
                math_placeholders.append(wrapped)
                return f'__MATH_{len(math_placeholders) - 1}__'

            text = re.sub(pattern, wrap_ascii_formula, text)

        # =================================================================
        # STEP 4: Handle italic *text* (but NOT single * in other contexts)
        # =================================================================
        # Only match *short phrase* patterns - max 50 chars, no newlines, no special chars
        # Skip if preceded by word char (like θ*) or followed by word char
        def replace_italic(match):
            content = match.group(1)
            # Skip if content is too long or contains line breaks
            if len(content) > 50 or '\n' in content or '<br' in content:
                return match.group(0)
            return f'<em class="text-blue-200">{content}</em>'

        # Pattern: *text* where text is short and doesn't contain problematic chars
        text = re.sub(r'(?<![*\w])\*([^\s*][^*\n]{0,48}[^\s*])\*(?![*\w])', replace_italic, text)

        # =================================================================
        # STEP 5: Handle headers and lists (cleaner without circles)
        # =================================================================
        text = re.sub(r'^####\s+(.+)$', r'<h5 class="text-lg font-semibold text-white mt-4 mb-2">\1</h5>', text, flags=re.MULTILINE)
        text = re.sub(r'^###\s+(.+)$', r'<h4 class="text-xl font-semibold text-white mt-4 mb-2">\1</h4>', text, flags=re.MULTILINE)
        # Lists use left border accent instead of circles (cleaner in glass cards)
        text = re.sub(r'^-\s+(.+)$', r'<div class="pl-3 border-l-2 border-orange-400/50 my-2">\1</div>', text, flags=re.MULTILINE)
        text = re.sub(r'^(\d+)\.\s+(?!<strong)(.+)$', r'<div class="pl-3 border-l-2 border-blue-400/50 my-2"><span class="text-blue-400 font-semibold mr-2">\1.</span>\2</div>', text, flags=re.MULTILINE)

        # =================================================================
        # STEP 6: Handle line breaks
        # =================================================================
        text = re.sub(r'\n\n+', '<br/><br/>', text)
        text = text.replace('\n', '<br/>')
        text = re.sub(r'(<br/>\s*){3,}', '<br/><br/>', text)

        # Clean up leading line breaks after step formatting
        text = re.sub(r'^(<br/>)+', '', text)

        # =================================================================
        # STEP 7: RESTORE CODE BLOCKS (after all processing)
        # =================================================================
        # Restore inline codes first
        for i, code in enumerate(inline_codes):
            # Extract the code content and format it
            code_match = re.match(r'`([^`]+)`', code)
            if code_match:
                code_content = html.escape(code_match.group(1))
                formatted = f'<code class="px-1.5 py-0.5 bg-blue-500/20 text-blue-300 rounded text-sm font-mono">{code_content}</code>'
                text = text.replace(f'__INLINE_CODE_{i}__', formatted)

        # Restore code blocks
        for i, block in enumerate(code_blocks):
            # Extract and format code block
            block_match = re.match(r'```[\w]*\n?(.*?)```', block, re.DOTALL)
            if block_match:
                code_content = html.escape(block_match.group(1).strip())
                # Convert newlines in code to <br/>
                code_content = code_content.replace('\n', '<br/>')
                formatted = f'<pre class="my-3 p-3 bg-black/30 rounded-lg overflow-x-auto"><code class="text-green-300 text-sm font-mono">{code_content}</code></pre>'
                text = text.replace(f'__CODE_BLOCK_{i}__', formatted)

        # =================================================================
        # STEP 8: RESTORE MATH EXPRESSIONS (for MathJax)
        # =================================================================
        # Restore math expressions intact for MathJax to process
        for i, math in enumerate(math_placeholders):
            text = text.replace(f'__MATH_{i}__', math)

        return text

    def generate(self) -> str:
        """Generate the complete HTML presentation"""
        slides_html = []
        for idx, slide in enumerate(self.slides, 1):
            slide_html = self._render_slide(slide, idx)
            slides_html.append(slide_html)

        return self._wrap_html(slides_html)

    def _wrap_html(self, slides_html: List[str]) -> str:
        """Wrap slides in full HTML document"""
        total_slides = len(slides_html)

        return f'''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{html.escape(self.config.title)} - {self.config.branding}</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <!-- MathJax for proper LaTeX rendering -->
  <script>
    MathJax = {{
      tex: {{
        inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
        displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']],
        processEscapes: true
      }},
      svg: {{ fontCache: 'global' }},
      startup: {{ typeset: true }}
    }};
  </script>
  <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js" async></script>
  {self._get_styles()}
</head>
<body class="bg-[{self.colors['bg_primary']}] text-white overflow-x-hidden">

{"".join(slides_html)}

{self._get_navigation(total_slides)}
{self._get_branding()}
{self._get_scripts(total_slides)}
</body>
</html>'''

    def _get_styles(self) -> str:
        """Get CSS styles with enhanced visual polish and interactivity"""
        return f'''<style>
    body {{ font-family: 'Inter', sans-serif; }}
    .mono {{ font-family: 'JetBrains Mono', monospace; }}

    .slide {{ display: none; min-height: 100vh; }}
    .slide.active {{ display: block; }}

    /* Enhanced Animations */
    @keyframes float {{ 0%, 100% {{ transform: translateY(0) rotate(0deg); }} 50% {{ transform: translateY(-20px) rotate(1deg); }} }}
    @keyframes pulse {{ 0%, 100% {{ opacity: 1; transform: scale(1); }} 50% {{ opacity: 0.7; transform: scale(0.98); }} }}
    @keyframes slideInLeft {{ from {{ opacity: 0; transform: translateX(-50px); }} to {{ opacity: 1; transform: translateX(0); }} }}
    @keyframes slideInRight {{ from {{ opacity: 0; transform: translateX(50px); }} to {{ opacity: 1; transform: translateX(0); }} }}
    @keyframes slideInUp {{ from {{ opacity: 0; transform: translateY(30px); }} to {{ opacity: 1; transform: translateY(0); }} }}
    @keyframes fadeIn {{ from {{ opacity: 0; }} to {{ opacity: 1; }} }}
    @keyframes scaleIn {{ from {{ opacity: 0; transform: scale(0.8); }} to {{ opacity: 1; transform: scale(1); }} }}
    @keyframes bounceIn {{ 0% {{ opacity: 0; transform: scale(0.3); }} 50% {{ transform: scale(1.05); }} 70% {{ transform: scale(0.9); }} 100% {{ opacity: 1; transform: scale(1); }} }}
    @keyframes shimmer {{ 0% {{ background-position: -200% 0; }} 100% {{ background-position: 200% 0; }} }}
    @keyframes glow {{ 0%, 100% {{ box-shadow: 0 0 20px rgba(246, 173, 85, 0.3); }} 50% {{ box-shadow: 0 0 40px rgba(246, 173, 85, 0.6); }} }}
    @keyframes typewriter {{ from {{ width: 0; }} to {{ width: 100%; }} }}
    @keyframes blink {{ 50% {{ border-color: transparent; }} }}
    @keyframes gradient {{ 0% {{ background-position: 0% 50%; }} 50% {{ background-position: 100% 50%; }} 100% {{ background-position: 0% 50%; }} }}
    @keyframes countUp {{ from {{ opacity: 0; transform: translateY(20px) scale(0.5); }} to {{ opacity: 1; transform: translateY(0) scale(1); }} }}
    @keyframes ripple {{ 0% {{ transform: scale(0.8); opacity: 1; }} 100% {{ transform: scale(2); opacity: 0; }} }}
    @keyframes shake {{ 0%, 100% {{ transform: translateX(0); }} 10%, 30%, 50%, 70%, 90% {{ transform: translateX(-2px); }} 20%, 40%, 60%, 80% {{ transform: translateX(2px); }} }}

    .animate-float {{ animation: float 6s ease-in-out infinite; }}
    .animate-pulse-slow {{ animation: pulse 3s ease-in-out infinite; }}
    .animate-slide-left {{ animation: slideInLeft 0.6s ease-out forwards; }}
    .animate-slide-right {{ animation: slideInRight 0.6s ease-out forwards; }}
    .animate-slide-up {{ animation: slideInUp 0.6s ease-out forwards; }}
    .animate-fade {{ animation: fadeIn 0.8s ease-out forwards; }}
    .animate-scale {{ animation: scaleIn 0.5s ease-out forwards; }}
    .animate-bounce-in {{ animation: bounceIn 0.8s ease-out forwards; }}
    .animate-glow {{ animation: glow 2s ease-in-out infinite; }}
    .animate-gradient {{ background-size: 200% 200%; animation: gradient 3s ease infinite; }}
    .animate-count {{ animation: countUp 0.8s ease-out forwards; }}

    .delay-100 {{ animation-delay: 0.1s; }} .delay-200 {{ animation-delay: 0.2s; }}
    .delay-300 {{ animation-delay: 0.3s; }} .delay-400 {{ animation-delay: 0.4s; }}
    .delay-500 {{ animation-delay: 0.5s; }} .delay-600 {{ animation-delay: 0.6s; }}
    .delay-700 {{ animation-delay: 0.7s; }} .delay-800 {{ animation-delay: 0.8s; }}
    .delay-900 {{ animation-delay: 0.9s; }} .delay-1000 {{ animation-delay: 1.0s; }}

    /* Glass & Gradients */
    .glass {{ background: rgba(255,255,255,0.05); backdrop-filter: blur(12px); border: 1px solid rgba(255,255,255,0.1); }}
    .glass-dark {{ background: rgba(0,0,0,0.4); backdrop-filter: blur(12px); border: 1px solid rgba(255,255,255,0.08); }}
    .glass-glow {{ background: rgba(255,255,255,0.05); backdrop-filter: blur(12px); border: 1px solid rgba(255,255,255,0.15); box-shadow: 0 8px 32px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.1); }}
    .gradient-text {{ background: linear-gradient(135deg, #f6ad55, #ed64a6, #9f7aea); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }}
    .gradient-border {{ background: linear-gradient(135deg, {self.colors['accent_blue']}, {self.colors['accent_purple']}); padding: 2px; border-radius: 16px; }}
    .gradient-border-gold {{ background: linear-gradient(135deg, {self.config.accent_color}, #ed8936); padding: 2px; border-radius: 16px; }}
    .gradient-border-glow {{ background: linear-gradient(135deg, {self.colors['accent_blue']}, {self.colors['accent_purple']}); padding: 2px; border-radius: 16px; box-shadow: 0 0 20px rgba(66, 153, 225, 0.3); }}

    /* Shimmer effect for loading/emphasis */
    .shimmer {{ background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent); background-size: 200% 100%; animation: shimmer 2s infinite; }}

    /* Text styles */
    .highlight {{ color: {self.config.accent_color}; font-weight: 600; }}
    .highlight-glow {{ color: {self.config.accent_color}; font-weight: 600; text-shadow: 0 0 20px rgba(246, 173, 85, 0.5); }}
    .code-inline {{ background: rgba(66, 153, 225, 0.2); color: #90cdf4; padding: 2px 8px; border-radius: 4px; font-family: 'JetBrains Mono', monospace; font-size: 0.9em; }}
    .text-glow {{ text-shadow: 0 0 30px currentColor; }}

    /* Interactive cards */
    .card-hover {{ transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1); }}
    .card-hover:hover {{ transform: translateY(-8px) scale(1.02); box-shadow: 0 25px 50px rgba(0,0,0,0.4); }}
    .card-3d {{ transition: all 0.3s ease; transform-style: preserve-3d; }}
    .card-3d:hover {{ transform: perspective(1000px) rotateX(2deg) rotateY(-2deg) translateY(-5px); }}

    /* Progress & meters */
    .progress-bar {{ height: 8px; border-radius: 4px; background: {self.colors['border']}; overflow: hidden; }}
    .progress-fill {{ height: 100%; border-radius: 4px; transition: width 1.5s cubic-bezier(0.4, 0, 0.2, 1); }}
    .progress-animated {{ background: linear-gradient(90deg, {self.colors['accent_blue']}, {self.colors['accent_purple']}, {self.colors['accent_blue']}); background-size: 200% 100%; animation: shimmer 2s infinite; }}

    /* Icons */
    .icon-circle {{ width: 56px; height: 56px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 1.5rem; transition: all 0.3s ease; }}
    .icon-circle:hover {{ transform: scale(1.1) rotate(5deg); }}
    .icon-square {{ width: 48px; height: 48px; border-radius: 12px; display: flex; align-items: center; justify-content: center; font-size: 1.25rem; transition: all 0.3s ease; }}
    .icon-square:hover {{ transform: scale(1.1); }}
    .icon-glow {{ box-shadow: 0 0 20px currentColor; }}

    /* Tags & badges */
    .tag {{ display: inline-flex; align-items: center; padding: 4px 12px; border-radius: 9999px; font-size: 0.75rem; font-weight: 500; transition: all 0.2s ease; }}
    .tag:hover {{ transform: scale(1.05); }}
    .tag-glow {{ box-shadow: 0 0 15px currentColor; }}

    /* Step numbers */
    .step-number {{ width: 40px; height: 40px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 1.125rem; transition: all 0.3s ease; }}
    .step-number:hover {{ transform: scale(1.15); }}

    /* Quote */
    .quote-mark {{ font-size: 4rem; line-height: 1; color: {self.config.accent_color}; opacity: 0.5; font-family: Georgia, serif; }}

    /* Tooltips */
    .tooltip {{ position: relative; cursor: help; }}
    .tooltip::after {{ content: attr(data-tooltip); position: absolute; bottom: 100%; left: 50%; transform: translateX(-50%) translateY(-8px); padding: 8px 12px; background: {self.colors['bg_card']}; color: white; border-radius: 8px; font-size: 0.75rem; white-space: nowrap; opacity: 0; pointer-events: none; transition: all 0.2s ease; z-index: 100; }}
    .tooltip:hover::after {{ opacity: 1; transform: translateX(-50%) translateY(0); }}

    /* Focus/active states for accessibility */
    button:focus-visible, a:focus-visible {{ outline: 2px solid {self.config.accent_color}; outline-offset: 2px; }}
    .focus-ring:focus {{ ring: 2px; ring-color: {self.config.accent_color}; ring-offset: 2px; }}

    /* Scroll-triggered animations (when JS adds .in-view) */
    .reveal {{ opacity: 0; transform: translateY(30px); transition: all 0.8s cubic-bezier(0.4, 0, 0.2, 1); }}
    .reveal.in-view {{ opacity: 1; transform: translateY(0); }}

    /* Interactive diagram nodes */
    .node {{ transition: all 0.3s ease; cursor: pointer; }}
    .node:hover {{ filter: brightness(1.2); transform: scale(1.05); }}
    .node.active {{ filter: brightness(1.3); box-shadow: 0 0 20px currentColor; }}

    /* Code block enhancements */
    .code-block {{ position: relative; }}
    .code-block .copy-btn {{ position: absolute; top: 8px; right: 8px; opacity: 0; transition: opacity 0.2s; }}
    .code-block:hover .copy-btn {{ opacity: 1; }}

    /* Number counter animation */
    .counter {{ font-variant-numeric: tabular-nums; }}

    /* Responsive adjustments - full width on mobile */
    @media (max-width: 640px) {{
      .text-6xl {{ font-size: 2rem; }}
      .text-7xl {{ font-size: 2.5rem; }}
      .text-5xl {{ font-size: 1.75rem; }}
      .text-4xl {{ font-size: 1.5rem; }}
      .text-3xl {{ font-size: 1.25rem; }}
      .grid-cols-4, .grid-cols-3, .grid-cols-2 {{ grid-template-columns: 1fr; }}
      .gap-8 {{ gap: 1rem; }}
      .gap-6 {{ gap: 0.75rem; }}
      .p-8 {{ padding: 1rem; }}
      .p-6 {{ padding: 0.75rem; }}
      .max-w-4xl, .max-w-5xl {{ max-width: 100%; }}
      .glass, .glass-glow {{ padding: 1rem; }}
    }}
    @media (max-width: 768px) {{
      .text-6xl {{ font-size: 2.5rem; }}
      .text-7xl {{ font-size: 3rem; }}
      .grid-cols-4 {{ grid-template-columns: repeat(2, 1fr); }}
      .grid-cols-3 {{ grid-template-columns: repeat(2, 1fr); }}
    }}

    /* Navigation button effects */
    .nav-btn {{
      position: relative;
      overflow: hidden;
    }}
    .nav-btn::after {{
      content: '';
      position: absolute;
      inset: 0;
      background: radial-gradient(circle at center, rgba(255,255,255,0.3), transparent 70%);
      opacity: 0;
      transition: opacity 0.3s;
    }}
    .nav-btn:hover::after {{
      opacity: 1;
    }}

    /* Floating particles effect */
    .particle {{
      position: absolute;
      width: 4px;
      height: 4px;
      background: rgba(255,255,255,0.1);
      border-radius: 50%;
      pointer-events: none;
      animation: float-particle 20s infinite linear;
    }}
    @keyframes float-particle {{
      0% {{ transform: translateY(100vh) rotate(0deg); opacity: 0; }}
      10% {{ opacity: 0.5; }}
      90% {{ opacity: 0.5; }}
      100% {{ transform: translateY(-100vh) rotate(720deg); opacity: 0; }}
    }}

    /* Slide transition effects */
    .slide {{
      transition: opacity 0.4s ease-out;
    }}
    .slide.active {{
      animation: slideEnter 0.5s ease-out;
    }}
    @keyframes slideEnter {{
      from {{ opacity: 0; transform: scale(0.98); }}
      to {{ opacity: 1; transform: scale(1); }}
    }}

    /* Interactive stat cards */
    .stat-card {{
      transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
      cursor: default;
    }}
    .stat-card:hover {{
      transform: translateY(-4px);
      box-shadow: 0 20px 40px rgba(0,0,0,0.3);
    }}

    /* Glowing accent line */
    .accent-line {{
      background: linear-gradient(90deg, transparent, {self.config.accent_color}, transparent);
      animation: glow-line 3s infinite;
    }}
    @keyframes glow-line {{
      0%, 100% {{ opacity: 0.5; filter: blur(0px); }}
      50% {{ opacity: 1; filter: blur(2px); }}
    }}

    /* Enhanced scrollbar */
    ::-webkit-scrollbar {{ width: 8px; height: 8px; }}
    ::-webkit-scrollbar-track {{ background: {self.colors['bg_secondary']}; }}
    ::-webkit-scrollbar-thumb {{ background: {self.colors['border']}; border-radius: 4px; }}
    ::-webkit-scrollbar-thumb:hover {{ background: {self.colors['text_muted']}; }}

    /* Keyboard shortcut styles */
    kbd {{
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.75rem;
      background: {self.colors['bg_tertiary']};
      border: 1px solid {self.colors['border']};
      border-radius: 4px;
      padding: 2px 6px;
      color: {self.colors['text_secondary']};
      box-shadow: 0 2px 0 {self.colors['border']};
    }}

    /* Selection style */
    ::selection {{
      background: {self.config.accent_color};
      color: {self.colors['bg_primary']};
    }}

    /* Print styles */
    @media print {{
      .slide {{ display: block !important; page-break-after: always; }}
      .fixed {{ display: none !important; }}
      body {{ background: white !important; color: black !important; }}
    }}
  </style>'''

    def _get_navigation(self, total_slides: int) -> str:
        """Get enhanced navigation HTML with progress bar, shortcuts, and grid mode"""
        return f'''
<!-- Top Progress Bar -->
<div id="progressBar" class="fixed top-0 left-0 w-full h-1 bg-[{self.colors['border']}]/30 z-50">
  <div id="progressFill" class="h-full bg-gradient-to-r from-[{self.config.accent_color}] via-[{self.colors['accent_purple']}] to-[{self.colors['accent_blue']}] transition-all duration-500 ease-out" style="width: 0%"></div>
</div>

<!-- Main Navigation -->
<div class="fixed bottom-8 left-1/2 -translate-x-1/2 flex items-center gap-4 z-50">
  <button id="prevBtn" onclick="prevSlide()" class="nav-btn p-3 rounded-full bg-[{self.colors['bg_tertiary']}]/80 border border-[{self.colors['border']}] text-white hover:bg-[{self.colors['border']}] transition-all backdrop-blur-sm hover:scale-110">
    <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7"/></svg>
  </button>
  <div id="indicators" class="flex items-center gap-1 px-4 py-2 bg-[{self.colors['bg_tertiary']}]/80 rounded-full border border-[{self.colors['border']}] backdrop-blur-sm max-w-md overflow-x-auto"></div>
  <button id="nextBtn" onclick="nextSlide()" class="nav-btn p-3 rounded-full bg-[{self.colors['bg_tertiary']}]/80 border border-[{self.colors['border']}] text-white hover:bg-[{self.colors['border']}] transition-all backdrop-blur-sm hover:scale-110">
    <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7"/></svg>
  </button>
</div>

<!-- Right Side Controls - moved to TOP right to avoid overlap with navigation -->
<div class="fixed right-6 top-6 flex items-center gap-2 z-50">
  <div id="slideCounter" class="text-[{self.colors['text_muted']}] text-sm font-mono bg-[{self.colors['bg_tertiary']}]/60 px-3 py-1.5 rounded-lg border border-[{self.colors['border']}] backdrop-blur-sm">1 / {total_slides}</div>
  <button id="slideshowBtn" onclick="toggleSlideshow()" class="nav-btn p-2.5 rounded-lg bg-[{self.colors['bg_tertiary']}]/60 border border-[{self.colors['border']}] text-[{self.colors['text_muted']}] hover:text-white hover:bg-[{self.colors['border']}] transition-all backdrop-blur-sm" title="Slideshow (P)">
    <svg id="playIcon" class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z"/><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>
    <svg id="pauseIcon" class="w-4 h-4 hidden" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 9v6m4-6v6m7-3a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>
  </button>
  <button onclick="toggleSlideshowSettings()" class="nav-btn p-2.5 rounded-lg bg-[{self.colors['bg_tertiary']}]/60 border border-[{self.colors['border']}] text-[{self.colors['text_muted']}] hover:text-white hover:bg-[{self.colors['border']}] transition-all backdrop-blur-sm" title="Slideshow Settings">
    <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>
  </button>
  <button onclick="toggleGrid()" class="nav-btn p-2.5 rounded-lg bg-[{self.colors['bg_tertiary']}]/60 border border-[{self.colors['border']}] text-[{self.colors['text_muted']}] hover:text-white hover:bg-[{self.colors['border']}] transition-all backdrop-blur-sm" title="Slide Overview (G)">
    <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2V6zM14 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V6zM4 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2v-2zM14 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2v-2z"/></svg>
  </button>
  <button onclick="toggleFullscreen()" class="nav-btn p-2.5 rounded-lg bg-[{self.colors['bg_tertiary']}]/60 border border-[{self.colors['border']}] text-[{self.colors['text_muted']}] hover:text-white hover:bg-[{self.colors['border']}] transition-all backdrop-blur-sm" title="Fullscreen (F)">
    <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4"/></svg>
  </button>
  <button onclick="toggleShortcuts()" class="nav-btn p-2.5 rounded-lg bg-[{self.colors['bg_tertiary']}]/60 border border-[{self.colors['border']}] text-[{self.colors['text_muted']}] hover:text-white hover:bg-[{self.colors['border']}] transition-all backdrop-blur-sm" title="Shortcuts (?)">
    <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8.228 9c.549-1.165 2.03-2 3.772-2 2.21 0 4 1.343 4 3 0 1.4-1.278 2.575-3.006 2.907-.542.104-.994.54-.994 1.093m0 3h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>
  </button>
</div>

<!-- Bottom Left Hint -->
<div class="fixed bottom-8 left-8 text-[{self.colors['text_muted']}] text-xs hidden md:block backdrop-blur-sm bg-[{self.colors['bg_tertiary']}]/30 px-3 py-1.5 rounded-lg">Press <kbd class="px-1.5 py-0.5 bg-[{self.colors['bg_tertiary']}] rounded text-xs">?</kbd> for shortcuts</div>

<!-- Grid View Overlay -->
<div id="gridOverlay" class="fixed inset-0 bg-[{self.colors['bg_primary']}]/95 backdrop-blur-lg z-[60] hidden overflow-auto p-8">
  <div class="flex justify-between items-center mb-6">
    <h2 class="text-2xl font-bold text-white">Slide Overview</h2>
    <button onclick="toggleGrid()" class="p-2 rounded-lg bg-[{self.colors['bg_tertiary']}] text-white hover:bg-[{self.colors['border']}] transition-colors">
      <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"/></svg>
    </button>
  </div>
  <div id="gridContainer" class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4"></div>
</div>

<!-- Keyboard Shortcuts Modal -->
<div id="shortcutsModal" class="fixed inset-0 bg-black/80 backdrop-blur-sm z-[70] hidden flex items-center justify-center p-4">
  <div class="glass-glow rounded-2xl p-8 max-w-md w-full animate-scale">
    <div class="flex justify-between items-center mb-6">
      <h3 class="text-xl font-bold text-white">Keyboard Shortcuts</h3>
      <button onclick="toggleShortcuts()" class="text-[{self.colors['text_muted']}] hover:text-white">
        <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"/></svg>
      </button>
    </div>
    <div class="space-y-3 text-sm">
      <div class="flex justify-between"><span class="text-[{self.colors['text_secondary']}]">Next slide</span><kbd class="px-2 py-1 bg-[{self.colors['bg_tertiary']}] rounded text-white">→</kbd></div>
      <div class="flex justify-between"><span class="text-[{self.colors['text_secondary']}]">Previous slide</span><kbd class="px-2 py-1 bg-[{self.colors['bg_tertiary']}] rounded text-white">←</kbd></div>
      <div class="flex justify-between"><span class="text-[{self.colors['text_secondary']}]">First slide</span><kbd class="px-2 py-1 bg-[{self.colors['bg_tertiary']}] rounded text-white">Home</kbd></div>
      <div class="flex justify-between"><span class="text-[{self.colors['text_secondary']}]">Last slide</span><kbd class="px-2 py-1 bg-[{self.colors['bg_tertiary']}] rounded text-white">End</kbd></div>
      <div class="flex justify-between"><span class="text-[{self.colors['text_secondary']}]">Fullscreen</span><kbd class="px-2 py-1 bg-[{self.colors['bg_tertiary']}] rounded text-white">F</kbd></div>
      <div class="flex justify-between"><span class="text-[{self.colors['text_secondary']}]">Grid view</span><kbd class="px-2 py-1 bg-[{self.colors['bg_tertiary']}] rounded text-white">G</kbd></div>
      <div class="flex justify-between"><span class="text-[{self.colors['text_secondary']}]">Toggle shortcuts</span><kbd class="px-2 py-1 bg-[{self.colors['bg_tertiary']}] rounded text-white">?</kbd></div>
      <div class="flex justify-between"><span class="text-[{self.colors['text_secondary']}]">Close overlay</span><kbd class="px-2 py-1 bg-[{self.colors['bg_tertiary']}] rounded text-white">Esc</kbd></div>
      <div class="flex justify-between"><span class="text-[{self.colors['text_secondary']}]">Play/Pause slideshow</span><kbd class="px-2 py-1 bg-[{self.colors['bg_tertiary']}] rounded text-white">P</kbd></div>
    </div>
    <p class="text-[{self.colors['text_muted']}] text-xs mt-6 text-center">Swipe left/right on touch devices</p>
  </div>
</div>

<!-- Slideshow Settings Modal -->
<div id="slideshowModal" class="fixed inset-0 bg-black/80 backdrop-blur-sm z-[70] hidden flex items-center justify-center p-4">
  <div class="glass-glow rounded-2xl p-8 max-w-sm w-full animate-scale">
    <div class="flex justify-between items-center mb-6">
      <h3 class="text-xl font-bold text-white">Slideshow Settings</h3>
      <button onclick="toggleSlideshowSettings()" class="text-[{self.colors['text_muted']}] hover:text-white">
        <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"/></svg>
      </button>
    </div>
    <div class="space-y-4">
      <div>
        <label class="text-[{self.colors['text_secondary']}] text-sm mb-2 block">Slide Duration</label>
        <div class="grid grid-cols-4 gap-2">
          <button onclick="setSlideshowInterval(3000)" class="slideshow-interval-btn px-3 py-2 rounded-lg bg-[{self.colors['bg_tertiary']}] text-[{self.colors['text_muted']}] hover:bg-[{self.colors['border']}] hover:text-white transition-all text-sm" data-interval="3000">3s</button>
          <button onclick="setSlideshowInterval(5000)" class="slideshow-interval-btn px-3 py-2 rounded-lg bg-[{self.config.accent_color}] text-black font-medium transition-all text-sm" data-interval="5000">5s</button>
          <button onclick="setSlideshowInterval(10000)" class="slideshow-interval-btn px-3 py-2 rounded-lg bg-[{self.colors['bg_tertiary']}] text-[{self.colors['text_muted']}] hover:bg-[{self.colors['border']}] hover:text-white transition-all text-sm" data-interval="10000">10s</button>
          <button onclick="setSlideshowInterval(15000)" class="slideshow-interval-btn px-3 py-2 rounded-lg bg-[{self.colors['bg_tertiary']}] text-[{self.colors['text_muted']}] hover:bg-[{self.colors['border']}] hover:text-white transition-all text-sm" data-interval="15000">15s</button>
        </div>
      </div>
      <div>
        <label class="text-[{self.colors['text_secondary']}] text-sm mb-2 block">Or set custom (seconds)</label>
        <div class="flex gap-2">
          <input type="number" id="customInterval" min="1" max="120" value="5" class="flex-1 px-3 py-2 rounded-lg bg-[{self.colors['bg_tertiary']}] border border-[{self.colors['border']}] text-white text-sm focus:outline-none focus:border-[{self.config.accent_color}]">
          <button onclick="setCustomInterval()" class="px-4 py-2 rounded-lg bg-[{self.config.accent_color}] text-black font-medium hover:opacity-90 transition-all text-sm">Set</button>
        </div>
      </div>
      <div class="flex items-center justify-between pt-2">
        <span class="text-[{self.colors['text_secondary']}] text-sm">Loop slideshow</span>
        <button id="loopToggle" onclick="toggleLoop()" class="w-12 h-6 rounded-full bg-[{self.colors['bg_tertiary']}] relative transition-colors">
          <span id="loopIndicator" class="absolute left-1 top-1 w-4 h-4 rounded-full bg-[{self.colors['text_muted']}] transition-all"></span>
        </button>
      </div>
    </div>
    <div class="mt-6 pt-4 border-t border-[{self.colors['border']}]">
      <div class="flex items-center justify-between text-sm">
        <span class="text-[{self.colors['text_muted']}]">Current: <span id="currentIntervalDisplay" class="text-white">5s</span></span>
        <button onclick="toggleSlideshow(); toggleSlideshowSettings();" class="px-4 py-2 rounded-lg bg-[{self.config.accent_color}] text-black font-medium hover:opacity-90 transition-all">Start Slideshow</button>
      </div>
    </div>
  </div>
</div>'''

    def _get_branding(self) -> str:
        """Get branding HTML - positioned at top-left for visibility"""
        return f'''
<div class="fixed top-6 left-6 flex items-center gap-2 z-50">
  <div class="w-10 h-10 rounded-xl bg-gradient-to-br from-orange-500 to-orange-600 flex items-center justify-center text-white font-bold text-lg shadow-lg shadow-orange-500/30">J</div>
  <span class="text-white/80 text-sm font-medium hidden md:inline">{html.escape(self.config.branding)}</span>
</div>'''

    def _get_scripts(self, total_slides: int) -> str:
        """Get enhanced JavaScript with grid view, progress bar, and shortcuts"""
        return f'''
<script>
  let currentSlide = 1;
  const totalSlides = {total_slides};
  let gridOpen = false;
  let shortcutsOpen = false;
  let slideshowSettingsOpen = false;
  let slideshowActive = false;
  let slideshowInterval = 5000;
  let slideshowTimer = null;
  let loopEnabled = true;

  function updateSlide() {{
    document.querySelectorAll('.slide').forEach(slide => slide.classList.remove('active'));
    document.getElementById(`slide-${{currentSlide}}`).classList.add('active');
    document.getElementById('slideCounter').textContent = `${{currentSlide}} / ${{totalSlides}}`;

    // Update progress bar
    const progress = ((currentSlide - 1) / (totalSlides - 1)) * 100;
    document.getElementById('progressFill').style.width = `${{progress}}%`;

    // Update indicators
    const indicators = document.getElementById('indicators');
    indicators.innerHTML = '';
    for (let i = 1; i <= totalSlides; i++) {{
      const btn = document.createElement('button');
      btn.className = `w-2 h-2 rounded-full transition-all shrink-0 ${{i === currentSlide ? 'bg-[{self.config.accent_color}] w-4 shadow-lg shadow-[{self.config.accent_color}]/50' : 'bg-[{self.colors["text_muted"]}] hover:bg-[{self.colors["text_secondary"]}]'}}`;
      btn.onclick = () => goToSlide(i);
      indicators.appendChild(btn);
    }}

    document.getElementById('prevBtn').style.opacity = currentSlide === 1 ? '0.3' : '1';
    document.getElementById('prevBtn').style.pointerEvents = currentSlide === 1 ? 'none' : 'auto';
    document.getElementById('nextBtn').style.opacity = currentSlide === totalSlides ? '0.3' : '1';
    document.getElementById('nextBtn').style.pointerEvents = currentSlide === totalSlides ? 'none' : 'auto';

    // Trigger chart initialization if chart slide
    if (window.initCharts) window.initCharts();

    // Update URL hash without triggering navigation
    history.replaceState(null, null, `#slide-${{currentSlide}}`);
  }}

  function nextSlide() {{
    if (currentSlide < totalSlides) {{
      currentSlide++;
      updateSlide();
      addRipple(document.getElementById('nextBtn'));
    }}
  }}

  function prevSlide() {{
    if (currentSlide > 1) {{
      currentSlide--;
      updateSlide();
      addRipple(document.getElementById('prevBtn'));
    }}
  }}

  function goToSlide(n) {{
    currentSlide = n;
    updateSlide();
    if (gridOpen) toggleGrid();
  }}

  // Ripple effect on buttons
  function addRipple(button) {{
    const ripple = document.createElement('span');
    ripple.className = 'absolute inset-0 rounded-full bg-white/20 animate-ping';
    button.style.position = 'relative';
    button.style.overflow = 'hidden';
    button.appendChild(ripple);
    setTimeout(() => ripple.remove(), 500);
  }}

  // Grid view toggle - now shows actual slide content previews
  function toggleGrid() {{
    const overlay = document.getElementById('gridOverlay');
    const container = document.getElementById('gridContainer');
    gridOpen = !gridOpen;

    if (gridOpen) {{
      overlay.classList.remove('hidden');
      container.innerHTML = '';

      for (let i = 1; i <= totalSlides; i++) {{
        const slide = document.getElementById(`slide-${{i}}`);
        const thumbnail = document.createElement('div');
        thumbnail.className = `cursor-pointer rounded-xl overflow-hidden border-2 transition-all hover:scale-105 ${{i === currentSlide ? 'border-[{self.config.accent_color}] ring-2 ring-[{self.config.accent_color}]/50 shadow-lg shadow-[{self.config.accent_color}]/20' : 'border-[{self.colors["border"]}] hover:border-[{self.colors["text_muted"]}]'}}`;

        // Extract title from slide (h1 or h2)
        let slideTitle = 'Slide ' + i;
        const h1 = slide ? slide.querySelector('h1') : null;
        const h2 = slide ? slide.querySelector('h2') : null;
        const label = slide ? slide.querySelector('.tracking-widest') : null;
        if (h1) slideTitle = h1.textContent.substring(0, 40) + (h1.textContent.length > 40 ? '...' : '');
        else if (h2) slideTitle = h2.textContent.substring(0, 40) + (h2.textContent.length > 40 ? '...' : '');

        // Get label/category if exists
        let labelText = '';
        if (label) labelText = label.textContent.trim();

        thumbnail.innerHTML = `
          <div class="bg-gradient-to-br from-[{self.colors['bg_secondary']}] to-[{self.colors['bg_tertiary']}] p-4 aspect-video flex flex-col justify-between relative overflow-hidden">
            <div class="flex justify-between items-start">
              <div class="w-8 h-8 rounded-lg bg-[{self.config.accent_color}]/20 flex items-center justify-center text-[{self.config.accent_color}] font-bold text-sm">${{String(i).padStart(2, '0')}}</div>
              ${{i === currentSlide ? '<span class="text-xs bg-[{self.config.accent_color}] text-black px-2 py-0.5 rounded-full font-medium">Current</span>' : ''}}
            </div>
            <div class="mt-auto">
              ${{labelText ? '<div class="text-[{self.config.accent_color}] text-[10px] uppercase tracking-wider mb-1 truncate">' + labelText + '</div>' : ''}}
              <div class="text-white font-semibold text-sm leading-tight line-clamp-2">${{slideTitle}}</div>
            </div>
          </div>
        `;
        thumbnail.onclick = () => goToSlide(i);
        container.appendChild(thumbnail);
      }}
      document.body.style.overflow = 'hidden';
    }} else {{
      overlay.classList.add('hidden');
      document.body.style.overflow = '';
    }}
  }}

  // Shortcuts modal toggle
  function toggleShortcuts() {{
    const modal = document.getElementById('shortcutsModal');
    shortcutsOpen = !shortcutsOpen;
    modal.classList.toggle('hidden', !shortcutsOpen);
    document.body.style.overflow = shortcutsOpen ? 'hidden' : '';
  }}

  // Fullscreen toggle
  function toggleFullscreen() {{
    if (!document.fullscreenElement) {{
      document.documentElement.requestFullscreen().catch(err => console.log(err));
    }} else {{
      document.exitFullscreen();
    }}
  }}

  // Slideshow settings modal toggle
  function toggleSlideshowSettings() {{
    const modal = document.getElementById('slideshowModal');
    slideshowSettingsOpen = !slideshowSettingsOpen;
    modal.classList.toggle('hidden', !slideshowSettingsOpen);
    document.body.style.overflow = slideshowSettingsOpen ? 'hidden' : '';
  }}

  // Slideshow toggle (play/pause)
  function toggleSlideshow() {{
    slideshowActive = !slideshowActive;
    const playIcon = document.getElementById('playIcon');
    const pauseIcon = document.getElementById('pauseIcon');
    const slideshowBtn = document.getElementById('slideshowBtn');

    if (slideshowActive) {{
      playIcon.classList.add('hidden');
      pauseIcon.classList.remove('hidden');
      slideshowBtn.classList.add('bg-[{self.config.accent_color}]', 'text-black');
      slideshowBtn.classList.remove('text-[{self.colors["text_muted"]}]');
      startSlideshow();
    }} else {{
      playIcon.classList.remove('hidden');
      pauseIcon.classList.add('hidden');
      slideshowBtn.classList.remove('bg-[{self.config.accent_color}]', 'text-black');
      slideshowBtn.classList.add('text-[{self.colors["text_muted"]}]');
      stopSlideshow();
    }}
  }}

  // Start slideshow timer
  function startSlideshow() {{
    stopSlideshow();
    slideshowTimer = setInterval(() => {{
      if (currentSlide < totalSlides) {{
        nextSlide();
      }} else if (loopEnabled) {{
        goToSlide(1);
      }} else {{
        toggleSlideshow();
      }}
    }}, slideshowInterval);
  }}

  // Stop slideshow timer
  function stopSlideshow() {{
    if (slideshowTimer) {{
      clearInterval(slideshowTimer);
      slideshowTimer = null;
    }}
  }}

  // Set slideshow interval from preset buttons
  function setSlideshowInterval(ms) {{
    slideshowInterval = ms;
    updateIntervalButtons();
    document.getElementById('currentIntervalDisplay').textContent = (ms / 1000) + 's';
    document.getElementById('customInterval').value = ms / 1000;
    if (slideshowActive) startSlideshow();
  }}

  // Set custom interval
  function setCustomInterval() {{
    const input = document.getElementById('customInterval');
    const seconds = parseInt(input.value);
    if (seconds >= 1 && seconds <= 120) {{
      setSlideshowInterval(seconds * 1000);
    }}
  }}

  // Update interval button styles
  function updateIntervalButtons() {{
    document.querySelectorAll('.slideshow-interval-btn').forEach(btn => {{
      const interval = parseInt(btn.getAttribute('data-interval'));
      if (interval === slideshowInterval) {{
        btn.className = 'slideshow-interval-btn px-3 py-2 rounded-lg bg-[{self.config.accent_color}] text-black font-medium transition-all text-sm';
      }} else {{
        btn.className = 'slideshow-interval-btn px-3 py-2 rounded-lg bg-[{self.colors["bg_tertiary"]}] text-[{self.colors["text_muted"]}] hover:bg-[{self.colors["border"]}] hover:text-white transition-all text-sm';
      }}
    }});
  }}

  // Toggle loop setting
  function toggleLoop() {{
    loopEnabled = !loopEnabled;
    const toggle = document.getElementById('loopToggle');
    const indicator = document.getElementById('loopIndicator');
    if (loopEnabled) {{
      toggle.classList.add('bg-[{self.config.accent_color}]');
      toggle.classList.remove('bg-[{self.colors["bg_tertiary"]}]');
      indicator.classList.add('translate-x-6', 'bg-black');
      indicator.classList.remove('bg-[{self.colors["text_muted"]}]');
    }} else {{
      toggle.classList.remove('bg-[{self.config.accent_color}]');
      toggle.classList.add('bg-[{self.colors["bg_tertiary"]}]');
      indicator.classList.remove('translate-x-6', 'bg-black');
      indicator.classList.add('bg-[{self.colors["text_muted"]}]');
    }}
  }}

  // Initialize loop toggle visual state
  function initLoopToggle() {{
    if (loopEnabled) {{
      const toggle = document.getElementById('loopToggle');
      const indicator = document.getElementById('loopIndicator');
      toggle.classList.add('bg-[{self.config.accent_color}]');
      toggle.classList.remove('bg-[{self.colors["bg_tertiary"]}]');
      indicator.classList.add('translate-x-6', 'bg-black');
      indicator.classList.remove('bg-[{self.colors["text_muted"]}]');
    }}
  }}

  // Keyboard navigation
  document.addEventListener('keydown', (e) => {{
    // Close overlays with Escape
    if (e.key === 'Escape') {{
      if (shortcutsOpen) toggleShortcuts();
      if (gridOpen) toggleGrid();
      if (slideshowSettingsOpen) toggleSlideshowSettings();
      if (slideshowActive) toggleSlideshow();
      return;
    }}

    // Don't navigate if overlays are open
    if (gridOpen || shortcutsOpen || slideshowSettingsOpen) return;

    if (e.key === 'ArrowRight' || e.key === ' ') {{ e.preventDefault(); nextSlide(); }}
    else if (e.key === 'ArrowLeft') {{ e.preventDefault(); prevSlide(); }}
    else if (e.key === 'Home') {{ e.preventDefault(); goToSlide(1); }}
    else if (e.key === 'End') {{ e.preventDefault(); goToSlide(totalSlides); }}
    else if (e.key === 'f' || e.key === 'F') toggleFullscreen();
    else if (e.key === 'g' || e.key === 'G') toggleGrid();
    else if (e.key === 'p' || e.key === 'P') toggleSlideshow();
    else if (e.key === '?') toggleShortcuts();
  }});

  // Touch/swipe navigation with velocity detection
  let touchStartX = 0;
  let touchStartY = 0;
  let touchStartTime = 0;

  document.addEventListener('touchstart', e => {{
    if (gridOpen || shortcutsOpen || slideshowSettingsOpen) return;
    touchStartX = e.changedTouches[0].screenX;
    touchStartY = e.changedTouches[0].screenY;
    touchStartTime = Date.now();
  }}, {{ passive: true }});

  document.addEventListener('touchend', e => {{
    if (gridOpen || shortcutsOpen || slideshowSettingsOpen) return;
    // Pause slideshow on manual navigation
    if (slideshowActive) toggleSlideshow();
    const touchEndX = e.changedTouches[0].screenX;
    const touchEndY = e.changedTouches[0].screenY;
    const deltaX = touchEndX - touchStartX;
    const deltaY = touchEndY - touchStartY;
    const deltaTime = Date.now() - touchStartTime;

    // Only swipe if horizontal movement is greater than vertical
    if (Math.abs(deltaX) > Math.abs(deltaY) && Math.abs(deltaX) > 50) {{
      const velocity = Math.abs(deltaX) / deltaTime;
      if (velocity > 0.3 || Math.abs(deltaX) > 100) {{
        if (deltaX < 0) nextSlide();
        else prevSlide();
      }}
    }}
  }}, {{ passive: true }});

  // Animate numbers on slide
  function animateCounters() {{
    const currentSlideEl = document.getElementById(`slide-${{currentSlide}}`);
    if (!currentSlideEl) return;

    currentSlideEl.querySelectorAll('.counter[data-target]').forEach(counter => {{
      if (counter.dataset.animated) return;
      counter.dataset.animated = 'true';

      const target = parseFloat(counter.getAttribute('data-target'));
      const duration = 1500;
      const start = 0;
      const startTime = performance.now();
      const suffix = counter.getAttribute('data-suffix') || '';
      const isFloat = target % 1 !== 0;

      function updateCounter(currentTime) {{
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        const eased = 1 - Math.pow(1 - progress, 3);
        const value = start + (target - start) * eased;
        counter.textContent = (isFloat ? value.toFixed(1) : Math.round(value)) + suffix;
        if (progress < 1) requestAnimationFrame(updateCounter);
      }}
      requestAnimationFrame(updateCounter);
    }});
  }}

  // Progress bar animations
  function animateProgressBars() {{
    const currentSlideEl = document.getElementById(`slide-${{currentSlide}}`);
    if (!currentSlideEl) return;

    currentSlideEl.querySelectorAll('.progress-fill[data-width]').forEach(bar => {{
      if (bar.dataset.animated) return;
      bar.dataset.animated = 'true';
      bar.style.width = '0%';
      setTimeout(() => {{
        bar.style.width = bar.getAttribute('data-width') + '%';
      }}, 300);
    }});
  }}

  // Preload adjacent slides for smoother transitions
  function preloadSlides() {{
    [currentSlide - 1, currentSlide + 1].forEach(i => {{
      if (i >= 1 && i <= totalSlides) {{
        const slide = document.getElementById(`slide-${{i}}`);
        if (slide) slide.querySelectorAll('img').forEach(img => img.loading = 'eager');
      }}
    }});
  }}

  // Initialize on slide change
  const originalUpdateSlide = updateSlide;
  updateSlide = function() {{
    originalUpdateSlide();
    setTimeout(() => {{
      animateCounters();
      animateProgressBars();
      preloadSlides();
    }}, 100);
  }};

  // Handle URL hash on load
  function handleHash() {{
    const hash = window.location.hash;
    if (hash && hash.startsWith('#slide-')) {{
      const slideNum = parseInt(hash.replace('#slide-', ''));
      if (slideNum >= 1 && slideNum <= totalSlides) {{
        currentSlide = slideNum;
      }}
    }}
  }}

  // Initialize
  handleHash();
  updateSlide();
  initLoopToggle();

  // Handle browser back/forward
  window.addEventListener('popstate', () => {{
    handleHash();
    updateSlide();
  }});

  // Stop slideshow when visibility changes (tab switch)
  document.addEventListener('visibilitychange', () => {{
    if (document.hidden && slideshowActive) {{
      toggleSlideshow();
    }}
  }});
</script>'''

    def _render_slide(self, slide: SlideConfig, index: int) -> str:
        """Render a single slide based on its type"""
        render_methods = {
            SlideType.TITLE_HERO: self._render_title_hero,
            SlideType.TITLE_MINIMAL: self._render_title_minimal,
            SlideType.TITLE_CENTERED: self._render_title_centered,
            SlideType.STATS_GRID: self._render_stats_grid,
            SlideType.FEATURE_CARDS: self._render_feature_cards,
            SlideType.COMPARISON_TABLE: self._render_comparison_table,
            SlideType.PROCESS_STEPS: self._render_process_steps,
            SlideType.QUOTE: self._render_quote,
            SlideType.CODE_BLOCK: self._render_code_block,
            SlideType.TIMELINE: self._render_timeline,
            SlideType.DIAGRAM: self._render_diagram,
            SlideType.ICON_GRID: self._render_icon_grid,
            SlideType.FORMULA: self._render_formula,
            SlideType.BEFORE_AFTER: self._render_before_after,
            SlideType.BULLET_POINTS: self._render_bullet_points,
            SlideType.DEFINITION: self._render_definition,
            SlideType.PROS_CONS: self._render_pros_cons,
            SlideType.CHECKLIST: self._render_checklist,
            SlideType.AUTHORS: self._render_authors,
            SlideType.QA: self._render_qa,
            SlideType.KEY_TAKEAWAYS: self._render_key_takeaways,
            SlideType.TWO_COLUMN: self._render_two_column,
            SlideType.ARCHITECTURE: self._render_architecture,
            SlideType.CHART_BAR: self._render_chart_bar,
        }

        render_fn = render_methods.get(slide.slide_type, self._render_generic)
        content = render_fn(slide.data, index)

        active = "active" if index == 1 else ""
        return f'<div id="slide-{index}" class="slide {active}">{content}</div>'

    def _render_title_hero(self, data: Dict, index: int) -> str:
        """Render enhanced title hero slide with particles and glow effects"""
        title = html.escape(data.get("title", "Untitled"))
        subtitle = html.escape(data.get("subtitle", ""))
        hook = html.escape(data.get("hook", ""))
        tags = data.get("tags", [])
        arxiv_id = data.get("arxiv_id", self.config.arxiv_id)
        authors = data.get("authors", self.config.authors)

        tags_html = "".join([
            f'<span class="tag bg-{["blue", "purple", "teal", "orange"][i % 4]}-500/20 text-{["blue", "purple", "teal", "orange"][i % 4]}-300 hover:scale-105 transition-transform">{html.escape(tag)}</span>'
            for i, tag in enumerate(tags[:4])
        ])

        authors_str = " · ".join(authors[:3])
        if len(authors) > 3:
            authors_str += " · et al."

        return f'''
<div class="relative w-full min-h-screen overflow-hidden bg-gradient-to-br from-[{self.colors['bg_secondary']}] via-[{self.colors['bg_tertiary']}] to-[{self.colors['bg_primary']}]">
  <!-- Animated background orbs -->
  <div class="absolute -top-20 -right-20 w-80 h-80 rounded-full bg-blue-500/10 blur-3xl animate-float"></div>
  <div class="absolute -bottom-20 -left-20 w-96 h-96 rounded-full bg-orange-500/10 blur-3xl animate-float" style="animation-delay:-3s"></div>
  <div class="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] rounded-full bg-purple-500/5 blur-3xl animate-pulse-slow"></div>

  <!-- Floating particles -->
  <div class="absolute inset-0 pointer-events-none">
    <div class="particle" style="left: 5%; animation-delay: 0s; width: 6px; height: 6px;"></div>
    <div class="particle" style="left: 15%; animation-delay: 2s;"></div>
    <div class="particle" style="left: 25%; animation-delay: 5s; width: 8px; height: 8px;"></div>
    <div class="particle" style="left: 45%; animation-delay: 8s;"></div>
    <div class="particle" style="left: 65%; animation-delay: 11s; width: 5px; height: 5px;"></div>
    <div class="particle" style="left: 75%; animation-delay: 14s;"></div>
    <div class="particle" style="left: 85%; animation-delay: 17s; width: 7px; height: 7px;"></div>
    <div class="particle" style="left: 95%; animation-delay: 3s;"></div>
  </div>

  <div class="relative z-10 min-h-screen flex flex-col justify-center px-4 sm:px-8 md:px-12 py-8 sm:py-12 md:py-16">
    <div class="absolute right-12 top-20 text-[80px] font-bold text-white/[0.03] pointer-events-none select-none animate-scale opacity-0">{index:02d}</div>

    <span class="inline-flex items-center gap-2 text-[{self.config.accent_color}] text-sm font-medium tracking-widest uppercase animate-fade opacity-0 delay-100">
      <span class="w-2 h-2 rounded-full bg-[{self.config.accent_color}] animate-pulse-slow shadow-lg shadow-[{self.config.accent_color}]/50"></span>
      Research Learning
    </span>

    <h1 class="text-3xl md:text-4xl font-bold text-white mt-6 max-w-5xl leading-tight animate-slide-up opacity-0 delay-200">
      {title}
    </h1>

    {f'<p class="text-xl text-[{self.colors["text_secondary"]}] mt-6 max-w-5xl leading-relaxed animate-slide-up opacity-0 delay-300">{hook}</p>' if hook else ''}

    {self._render_arxiv_badge(arxiv_id) if arxiv_id else ''}

    {f'<p class="text-[{self.colors["text_muted"]}] mt-4 text-sm animate-fade opacity-0 delay-500">{authors_str}</p>' if authors_str else ''}

    {f'<div class="flex flex-wrap gap-4 mt-6 animate-slide-up opacity-0 delay-600">{tags_html}</div>' if tags else ''}

    <!-- Bottom gradient bar with glow -->
    <div class="absolute bottom-0 left-0 right-0">
      <div class="h-1 bg-gradient-to-r from-orange-500 via-pink-500 to-purple-500 shadow-lg shadow-pink-500/30"></div>
      <div class="h-0.5 bg-gradient-to-r from-orange-500 via-pink-500 to-purple-500 blur-sm"></div>
    </div>
  </div>
</div>'''

    def _render_arxiv_badge(self, arxiv_id: str) -> str:
        """Render ArXiv ID badge"""
        return f'''<div class="mt-8 animate-scale opacity-0 delay-400">
      <span class="inline-flex items-center px-4 py-2 bg-gradient-to-r from-[{self.config.accent_color}] to-[#ed8936] text-[{self.colors["bg_primary"]}] rounded-full text-sm font-bold shadow-lg shadow-[{self.config.accent_color}]/30 hover:scale-105 transition-transform cursor-default">
        <svg class="w-4 h-4 mr-2" fill="currentColor" viewBox="0 0 24 24"><path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zm-7 3c1.93 0 3.5 1.57 3.5 3.5S13.93 13 12 13s-3.5-1.57-3.5-3.5S10.07 6 12 6zm7 13H5v-.23c0-.62.28-1.2.76-1.58C7.47 15.82 9.64 15 12 15s4.53.82 6.24 2.19c.48.38.76.97.76 1.58V19z"/></svg>
        {html.escape(arxiv_id)}
      </span>
    </div>'''

    def _render_title_minimal(self, data: Dict, index: int) -> str:
        """Render minimal title slide"""
        title = html.escape(data.get("title", ""))
        subtitle = html.escape(data.get("subtitle", ""))

        return f'''
<div class="relative w-full min-h-screen bg-gradient-to-br from-[{self.colors['bg_secondary']}] via-[{self.colors['bg_tertiary']}] to-[{self.colors['bg_primary']}] flex items-center justify-center px-4 sm:px-8 md:px-12 pt-16 sm:pt-20 pb-8 sm:pb-12">
  <div class="text-center animate-scale opacity-0 delay-200">
    <h1 class="text-6xl md:text-7xl font-bold text-white">{title}</h1>
    {f'<p class="text-2xl text-[{self.colors["text_secondary"]}] mt-6">{subtitle}</p>' if subtitle else ''}
  </div>
</div>'''

    def _render_title_centered(self, data: Dict, index: int) -> str:
        """Render centered title slide with label"""
        label = html.escape(data.get("label", ""))
        title = html.escape(data.get("title", ""))
        subtitle = html.escape(data.get("subtitle", ""))

        return f'''
<div class="relative w-full min-h-screen bg-gradient-to-br from-[{self.colors['bg_secondary']}] via-[{self.colors['bg_tertiary']}] to-[{self.colors['bg_primary']}] px-4 sm:px-8 md:px-12 pt-16 sm:pt-20 pb-8 sm:pb-12">
  <div class="absolute right-8 top-24 text-[64px] font-bold text-white/[0.03] pointer-events-none select-none">{index:02d}</div>
  <span class="text-[{self.config.accent_color}] text-sm font-medium tracking-widest uppercase animate-fade opacity-0 delay-100">{label}</span>
  <h2 class="text-4xl md:text-5xl font-bold text-white mt-4 animate-slide-up opacity-0 delay-200">{title}</h2>
  {f'<p class="text-[{self.colors["text_secondary"]}] mt-3 max-w-5xl animate-fade opacity-0 delay-300">{subtitle}</p>' if subtitle else ''}
</div>'''

    def _render_stats_grid(self, data: Dict, index: int) -> str:
        """Render stats grid slide with animated counters"""
        label = html.escape(data.get("label", "Key Metrics"))
        title = html.escape(data.get("title", "Results"))
        stats = data.get("stats", [])

        stats_html = ""
        for i, stat in enumerate(stats[:8]):
            value = stat.get("value", "")
            label_text = html.escape(stat.get("label", ""))
            color = stat.get("color", "blue")
            progress = stat.get("progress", 0)
            accent = self.colors.get(f'accent_{color}', self.config.accent_color)

            # Check if value is numeric for counter animation
            is_numeric = str(value).replace('.', '').replace('%', '').replace('+', '').replace('K', '').replace('M', '').isdigit()
            suffix = ''
            numeric_val = value
            if is_numeric:
                if 'K' in str(value):
                    numeric_val = str(value).replace('K', '').replace('+', '')
                    suffix = 'K+'
                elif 'M' in str(value):
                    numeric_val = str(value).replace('M', '').replace('+', '')
                    suffix = 'M+'
                elif '%' in str(value):
                    numeric_val = str(value).replace('%', '')
                    suffix = '%'
                value_html = f'<span class="counter" data-target="{numeric_val}" data-suffix="{suffix}">0</span>'
            else:
                value_html = html.escape(str(value))

            stats_html += f'''
<div class="stat-card glass-glow rounded-2xl p-6 text-center animate-slide-up opacity-0 delay-{(i+1)*100}">
  <div class="text-5xl font-bold text-[{accent}] text-glow">{value_html}</div>
  <div class="text-[{self.colors['text_secondary']}] mt-2 font-medium">{label_text}</div>
  {f'<div class="mt-4 progress-bar"><div class="progress-fill progress-animated" data-width="{progress}" style="width:0%"></div></div>' if progress else ''}
</div>'''

        return f'''
<div class="relative w-full min-h-screen bg-gradient-to-br from-[{self.colors['bg_secondary']}] via-[{self.colors['bg_tertiary']}] to-[{self.colors['bg_primary']}] px-4 sm:px-8 md:px-12 pt-16 sm:pt-20 pb-8 sm:pb-12 overflow-hidden">
  <div class="absolute right-8 top-24 text-[64px] font-bold text-white/[0.03] pointer-events-none select-none select-none">{index:02d}</div>
  <div class="absolute inset-0 pointer-events-none">
    <div class="particle" style="left: 10%; animation-delay: 0s;"></div>
    <div class="particle" style="left: 30%; animation-delay: 4s;"></div>
    <div class="particle" style="left: 50%; animation-delay: 8s;"></div>
    <div class="particle" style="left: 70%; animation-delay: 12s;"></div>
    <div class="particle" style="left: 90%; animation-delay: 16s;"></div>
  </div>
  <div class="relative z-10">
    <span class="text-[{self.config.accent_color}] text-sm font-medium tracking-widest uppercase">{label}</span>
    <h2 class="text-4xl font-bold text-white mt-4">{title}</h2>
    <div class="h-0.5 w-24 accent-line mt-4"></div>
    <div class="grid grid-cols-2 md:grid-cols-4 gap-6 mt-10">{stats_html}</div>
  </div>
</div>'''

    def _render_feature_cards(self, data: Dict, index: int) -> str:
        """Render feature cards slide with enhanced visual effects"""
        label = html.escape(data.get("label", "Features"))
        title = html.escape(data.get("title", "Key Features"))
        features = data.get("features", [])

        colors = ["blue", "purple", "teal", "orange", "pink", "green"]
        features_html = ""

        for i, feat in enumerate(features[:6]):
            icon = feat.get("icon", "⭐")
            feat_title = html.escape(feat.get("title", ""))
            description = self._format_text(feat.get("description", ""))
            code = feat.get("code", "")
            color = colors[i % len(colors)]

            features_html += f'''
<div class="gradient-border-glow card-3d animate-slide-up opacity-0 delay-{(i+1)*100}">
  <div class="bg-[{self.colors['bg_card']}] rounded-[14px] p-6 h-full">
    <div class="icon-square bg-{color}-500/20 text-{color}-400 mb-4 icon-glow" style="--tw-shadow-color: var(--tw-{color}-400);">{icon}</div>
    <h3 class="text-xl font-semibold text-white">{feat_title}</h3>
    <p class="text-[{self.colors['text_secondary']}] mt-3 text-sm leading-relaxed">{description}</p>
    {f'<div class="mt-4 pt-4 border-t border-[{self.colors["border"]}]"><code class="code-inline text-xs shimmer">{html.escape(code)}</code></div>' if code else ''}
  </div>
</div>'''

        return f'''
<div class="relative w-full min-h-screen bg-gradient-to-br from-[{self.colors['bg_secondary']}] via-[{self.colors['bg_tertiary']}] to-[{self.colors['bg_primary']}] px-4 sm:px-8 md:px-12 pt-16 sm:pt-20 pb-8 sm:pb-12 overflow-hidden">
  <div class="absolute right-8 top-24 text-[64px] font-bold text-white/[0.03] pointer-events-none select-none select-none">{index:02d}</div>
  <div class="absolute -top-20 -left-20 w-72 h-72 rounded-full bg-purple-500/5 blur-3xl"></div>
  <div class="absolute -bottom-20 -right-20 w-96 h-96 rounded-full bg-blue-500/5 blur-3xl"></div>
  <div class="relative z-10">
    <span class="text-[{self.config.accent_color}] text-sm font-medium tracking-widest uppercase">{label}</span>
    <h2 class="text-4xl font-bold text-white mt-4">{title}</h2>
    <div class="h-0.5 w-24 accent-line mt-4"></div>
    <div class="grid grid-cols-1 md:grid-cols-3 gap-6 mt-10">{features_html}</div>
  </div>
</div>'''

    def _render_comparison_table(self, data: Dict, index: int) -> str:
        """Render comparison table slide"""
        label = html.escape(data.get("label", "Comparison"))
        title = html.escape(data.get("title", "Model Comparison"))
        headers = data.get("headers", [])
        rows = data.get("rows", [])
        highlight_row = data.get("highlight_row", -1)

        header_html = "".join([f'<th class="text-center p-4 text-[{self.colors["text_secondary"]}] font-medium">{html.escape(h)}</th>' for h in headers])

        rows_html = ""
        for i, row in enumerate(rows):
            is_highlight = i == highlight_row
            row_class = f'bg-gradient-to-r from-[{self.colors["accent_green"]}]/10 to-transparent border-l-4 border-[{self.colors["accent_green"]}]' if is_highlight else f'bg-[{self.colors["bg_tertiary"]}]/30 hover:bg-[{self.colors["bg_tertiary"]}]/50 transition-colors'

            cells_html = ""
            for j, cell in enumerate(row):
                if j == 0:
                    color = self.colors["accent_green"] if is_highlight else self.colors["text_secondary"]
                    cells_html += f'<td class="p-4"><div class="flex items-center gap-3"><span class="w-3 h-3 rounded-full bg-[{color}]"></span><span class="font-{"bold" if is_highlight else "medium"} {"text-[" + self.colors["accent_green"] + "]" if is_highlight else ""}">{html.escape(str(cell))}</span></div></td>'
                else:
                    cells_html += f'<td class="text-center p-4 {"font-bold text-[" + self.colors["accent_green"] + "]" if is_highlight else ""}">{html.escape(str(cell))}</td>'

            rows_html += f'<tr class="{row_class}">{cells_html}</tr>'

        return f'''
<div class="relative w-full min-h-screen bg-gradient-to-br from-[{self.colors['bg_secondary']}] via-[{self.colors['bg_tertiary']}] to-[{self.colors['bg_primary']}] px-4 sm:px-8 md:px-12 pt-16 sm:pt-20 pb-8 sm:pb-12">
  <div class="absolute right-8 top-24 text-[64px] font-bold text-white/[0.03] pointer-events-none select-none">{index:02d}</div>
  <span class="text-[{self.config.accent_color}] text-sm font-medium tracking-widest uppercase">{label}</span>
  <h2 class="text-4xl font-bold text-white mt-4">{title}</h2>
  <div class="mt-10 overflow-hidden rounded-2xl border border-[{self.colors['border']}] animate-scale opacity-0 delay-200">
    <table class="w-full">
      <thead class="bg-[{self.colors['bg_card']}]"><tr><th class="text-left p-4"></th>{header_html}</tr></thead>
      <tbody class="divide-y divide-[{self.colors['border']}]">{rows_html}</tbody>
    </table>
  </div>
</div>'''

    def _render_process_steps(self, data: Dict, index: int) -> str:
        """Render process steps slide"""
        label = html.escape(data.get("label", "Process"))
        title = html.escape(data.get("title", "How It Works"))
        steps = data.get("steps", [])

        colors = ["blue", "purple", "pink", "orange", "teal", "green"]
        steps_html = ""

        for i, step in enumerate(steps[:6]):
            step_title = html.escape(step.get("title", ""))
            description = self._format_text(step.get("description", ""))
            color = colors[i % len(colors)]

            steps_html += f'''
<div class="relative animate-slide-up opacity-0 delay-{(i+1)*100}">
  <div class="step-number bg-{color}-500 text-white mb-4 relative z-10">{i+1}</div>
  <div class="glass rounded-xl p-5">
    <h3 class="font-semibold text-white">{step_title}</h3>
    <p class="text-[{self.colors['text_secondary']}] text-sm mt-2">{description}</p>
  </div>
</div>'''

        return f'''
<div class="relative w-full min-h-screen bg-gradient-to-br from-[{self.colors['bg_secondary']}] via-[{self.colors['bg_tertiary']}] to-[{self.colors['bg_primary']}] px-4 sm:px-8 md:px-12 pt-16 sm:pt-20 pb-8 sm:pb-12">
  <div class="absolute right-8 top-24 text-[64px] font-bold text-white/[0.03] pointer-events-none select-none">{index:02d}</div>
  <span class="text-[{self.config.accent_color}] text-sm font-medium tracking-widest uppercase">{label}</span>
  <h2 class="text-4xl font-bold text-white mt-4">{title}</h2>
  <div class="relative mt-12 max-w-5xl mx-auto">
    <div class="absolute top-8 left-8 right-8 h-0.5 bg-gradient-to-r from-blue-500 via-purple-500 to-pink-500 hidden md:block"></div>
    <div class="grid grid-cols-1 md:grid-cols-{min(len(steps), 4)} gap-8">{steps_html}</div>
  </div>
</div>'''

    def _render_quote(self, data: Dict, index: int) -> str:
        """Render quote slide"""
        quote = html.escape(data.get("quote", ""))
        author = html.escape(data.get("author", ""))
        source = html.escape(data.get("source", ""))
        initials = "".join([n[0].upper() for n in author.split()[:2]]) if author else "?"

        return f'''
<div class="relative w-full min-h-screen bg-gradient-to-br from-[{self.colors['bg_secondary']}] via-[{self.colors['bg_tertiary']}] to-[{self.colors['bg_primary']}] flex items-center justify-center px-4 sm:px-8 md:px-12 pt-16 sm:pt-20 pb-8 sm:pb-12">
  <div class="absolute right-8 top-24 text-[64px] font-bold text-white/[0.03] pointer-events-none select-none">{index:02d}</div>
  <div class="max-w-4xl text-center animate-scale opacity-0 delay-200">
    <span class="text-[{self.config.accent_color}] text-sm font-medium tracking-widest uppercase">Quote</span>
    <div class="mt-8">
      <span class="quote-mark">"</span>
      <blockquote class="text-2xl md:text-3xl font-light text-white leading-relaxed -mt-8">{quote}</blockquote>
    </div>
    <div class="mt-8 flex items-center justify-center gap-4">
      <div class="w-12 h-12 rounded-full bg-gradient-to-br from-blue-500 to-purple-500 flex items-center justify-center text-white font-bold">{initials}</div>
      <div class="text-left">
        <div class="font-semibold text-white">{author}</div>
        {f'<div class="text-[{self.colors["text_muted"]}] text-sm">{source}</div>' if source else ''}
      </div>
    </div>
  </div>
</div>'''

    def _render_code_block(self, data: Dict, index: int) -> str:
        """Render code block slide"""
        label = html.escape(data.get("label", "Implementation"))
        title = html.escape(data.get("title", "Code"))
        raw_code = data.get("code", "")
        filename = html.escape(data.get("filename", "code.py"))
        language = data.get("language", "python")
        tags = data.get("tags", [])

        # Separate code from explanation text (LLM sometimes includes text after ```)
        code_part = raw_code
        explanation_part = ""
        if '```' in raw_code:
            # Find last ``` and split - anything after is explanation
            parts = raw_code.rsplit('```', 1)
            if len(parts) == 2:
                code_part = parts[0] + '```'
                explanation_part = parts[1].strip()
                # Extract just the code without markdown code block markers
                code_match = re.search(r'```[\w]*\n?(.*?)```', code_part, re.DOTALL)
                if code_match:
                    code_part = code_match.group(1)

        code = html.escape(code_part)
        explanation_html = self._format_content(explanation_part) if explanation_part else ""

        tags_html = "".join([f'<span class="tag bg-{["blue", "purple", "green"][i % 3]}-500/20 text-{["blue", "purple", "green"][i % 3]}-300">{html.escape(t)}</span>' for i, t in enumerate(tags[:4])])

        return f'''
<div class="relative w-full min-h-screen bg-gradient-to-br from-[{self.colors['bg_secondary']}] via-[{self.colors['bg_tertiary']}] to-[{self.colors['bg_primary']}] px-4 sm:px-8 md:px-12 pt-16 sm:pt-20 pb-8 sm:pb-12">
  <div class="absolute right-8 top-24 text-[64px] font-bold text-white/[0.03] pointer-events-none select-none">{index:02d}</div>
  <span class="text-[{self.config.accent_color}] text-sm font-medium tracking-widest uppercase">{label}</span>
  <h2 class="text-4xl font-bold text-white mt-4">{title}</h2>
  <div class="mt-8 max-w-4xl animate-slide-up opacity-0 delay-200">
    <div class="rounded-2xl overflow-hidden border border-[{self.colors['border']}]">
      <div class="bg-[{self.colors['bg_card']}] px-4 py-3 flex items-center justify-between">
        <div class="flex items-center gap-2">
          <span class="w-3 h-3 rounded-full bg-[{self.colors['accent_red']}]"></span>
          <span class="w-3 h-3 rounded-full bg-[{self.config.accent_color}]"></span>
          <span class="w-3 h-3 rounded-full bg-[{self.colors['accent_green']}]"></span>
        </div>
        <span class="text-[{self.colors['text_muted']}] text-sm mono">{filename}</span>
        <button class="text-[{self.colors['text_muted']}] hover:text-white text-sm">Copy</button>
      </div>
      <pre class="bg-[#0d2137] p-6 overflow-x-auto max-h-[60vh] overflow-y-auto" style="white-space: pre-wrap; word-wrap: break-word;"><code class="mono text-sm leading-relaxed">{code}</code></pre>
    </div>
    {f'<div class="mt-6 glass rounded-xl p-4 text-[{self.colors["text_secondary"]}]">{explanation_html}</div>' if explanation_html else ''}
    {f'<div class="flex gap-4 mt-6">{tags_html}</div>' if tags else ''}
  </div>
</div>'''

    def _render_timeline(self, data: Dict, index: int) -> str:
        """Render timeline slide"""
        label = html.escape(data.get("label", "Timeline"))
        title = html.escape(data.get("title", "Evolution"))
        events = data.get("events", [])

        colors = [self.config.accent_color, self.colors["accent_blue"], self.colors["accent_teal"], self.colors["accent_purple"]]
        events_html = ""

        for i, event in enumerate(events[:6]):
            year = html.escape(str(event.get("year", "")))
            event_title = html.escape(event.get("title", ""))
            description = self._format_text(event.get("description", ""))
            color = colors[i % len(colors)]
            is_highlight = event.get("highlight", False)

            events_html += f'''
<div class="flex items-start gap-6 animate-slide-left opacity-0 delay-{(i+1)*100}">
  <div class="w-[120px] text-right shrink-0"><span class="text-[{color}] font-bold text-lg">{year}</span></div>
  <div class="w-4 h-4 rounded-full bg-[{color}] ring-4 ring-[{color}]/30 shrink-0 mt-1"></div>
  <div class="glass rounded-xl p-5 flex-1 {"border-l-4 border-[" + color + "]" if is_highlight else ""}">
    <h3 class="font-semibold text-white">{event_title}</h3>
    <p class="text-[{self.colors['text_secondary']}] text-sm mt-1">{description}</p>
  </div>
</div>'''

        return f'''
<div class="relative w-full min-h-screen bg-gradient-to-br from-[{self.colors['bg_secondary']}] via-[{self.colors['bg_tertiary']}] to-[{self.colors['bg_primary']}] px-4 sm:px-8 md:px-12 pt-16 sm:pt-20 pb-8 sm:pb-12">
  <div class="absolute right-8 top-24 text-[64px] font-bold text-white/[0.03] pointer-events-none select-none">{index:02d}</div>
  <span class="text-[{self.config.accent_color}] text-sm font-medium tracking-widest uppercase">{label}</span>
  <h2 class="text-4xl font-bold text-white mt-4">{title}</h2>
  <div class="mt-10 max-w-4xl mx-auto relative">
    <div class="absolute left-[60px] top-0 bottom-0 w-0.5 bg-gradient-to-b from-[{self.config.accent_color}] via-[{self.colors['accent_blue']}] to-[{self.colors['accent_purple']}]"></div>
    <div class="space-y-6">{events_html}</div>
  </div>
</div>'''

    def _render_diagram(self, data: Dict, index: int) -> str:
        """Render diagram slide with SVG"""
        label = html.escape(data.get("label", "Diagram"))
        title = html.escape(data.get("title", "Architecture"))
        svg_content = data.get("svg", "")  # Pre-generated SVG or use default

        if not svg_content:
            svg_content = self._generate_default_diagram(data)

        return f'''
<div class="relative w-full min-h-screen bg-gradient-to-br from-[{self.colors['bg_secondary']}] via-[{self.colors['bg_tertiary']}] to-[{self.colors['bg_primary']}] px-4 sm:px-8 md:px-12 pt-16 sm:pt-20 pb-8 sm:pb-12">
  <div class="absolute right-8 top-24 text-[64px] font-bold text-white/[0.03] pointer-events-none select-none">{index:02d}</div>
  <span class="text-[{self.config.accent_color}] text-sm font-medium tracking-widest uppercase">{label}</span>
  <h2 class="text-4xl font-bold text-white mt-4">{title}</h2>
  <div class="mt-8 flex justify-center animate-scale opacity-0 delay-200">
    <div class="glass rounded-2xl p-8 max-w-4xl w-full">{svg_content}</div>
  </div>
</div>'''

    def _generate_default_diagram(self, data: Dict) -> str:
        """Generate a default diagram SVG"""
        nodes = data.get("nodes", [])
        # Basic placeholder - in production this would be more sophisticated
        return '<svg viewBox="0 0 400 300" class="w-full"><text x="200" y="150" text-anchor="middle" fill="white" font-size="16">Diagram placeholder</text></svg>'

    def _render_icon_grid(self, data: Dict, index: int) -> str:
        """Render icon grid slide with interactive cards"""
        label = html.escape(data.get("label", "Applications"))
        title = html.escape(data.get("title", "Use Cases"))
        items = data.get("items", [])

        items_html = ""
        for i, item in enumerate(items[:8]):
            icon = item.get("icon", "⭐")
            item_title = html.escape(item.get("title", ""))
            description = self._format_text(item.get("description", ""))
            url = item.get("url", "")

            # Generate search URL if no URL provided
            if not url and item_title:
                search_query = item_title.replace(' ', '+')
                url = f"https://www.google.com/search?q={search_query}"

            card_content = f'''
  <div class="text-5xl mb-4 group-hover:scale-110 transition-transform">{icon}</div>
  <h3 class="font-semibold text-white group-hover:text-[{self.config.accent_color}] transition-colors">{item_title}</h3>
  <p class="text-[{self.colors['text_muted']}] text-sm mt-2">{description}</p>
  <div class="mt-3 opacity-0 group-hover:opacity-100 transition-opacity">
    <span class="text-[{self.config.accent_color}] text-xs flex items-center justify-center gap-1">
      Learn more <svg class="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17 8l4 4m0 0l-4 4m4-4H3"/></svg>
    </span>
  </div>'''

            if url:
                items_html += f'''
<a href="{html.escape(url)}" target="_blank" rel="noopener" class="group glass rounded-2xl p-6 text-center card-hover animate-bounce-in opacity-0 delay-{(i+1)*100} cursor-pointer block hover:border-[{self.config.accent_color}]/50 border border-transparent transition-all">
{card_content}
</a>'''
            else:
                items_html += f'''
<div class="group glass rounded-2xl p-6 text-center card-hover animate-bounce-in opacity-0 delay-{(i+1)*100}">
{card_content}
</div>'''

        return f'''
<div class="relative w-full min-h-screen bg-gradient-to-br from-[{self.colors['bg_secondary']}] via-[{self.colors['bg_tertiary']}] to-[{self.colors['bg_primary']}] px-4 sm:px-8 md:px-12 pt-16 sm:pt-20 pb-8 sm:pb-12">
  <div class="absolute right-8 top-24 text-[64px] font-bold text-white/[0.03] pointer-events-none select-none">{index:02d}</div>
  <span class="text-[{self.config.accent_color}] text-sm font-medium tracking-widest uppercase">{label}</span>
  <h2 class="text-4xl font-bold text-white mt-4">{title}</h2>
  <div class="grid grid-cols-2 md:grid-cols-4 gap-6 mt-10">{items_html}</div>
</div>'''

    def _render_formula(self, data: Dict, index: int) -> str:
        """Render formula slide with proper MathJax LaTeX rendering"""
        label = html.escape(data.get("label", "Formula"))
        title = html.escape(data.get("title", "Core Equation"))
        formula = data.get("formula", "")
        explanation = data.get("explanation", "")
        intuition = data.get("intuition", "")
        explanations = data.get("explanations", [])

        # Check if formula is actual math (LaTeX or ASCII math notation)
        def is_math_formula(text: str) -> bool:
            """Detect if text contains mathematical notation (LaTeX or ASCII math)."""
            if not text:
                return False

            # LaTeX commands
            latex_indicators = [
                r'\\frac', r'\\sum', r'\\int', r'\\prod',
                r'\\alpha', r'\\beta', r'\\gamma', r'\\theta',
                r'\\partial', r'\\nabla', r'\\infty',
                r'\\left', r'\\right', r'\\mathbb', r'\\mathcal',
            ]

            # Unicode math symbols
            unicode_math = ['∑', '∫', '∂', '∇', '∞', 'α', 'β', 'γ', 'θ', 'λ', 'σ', 'π', 'μ']

            # ASCII math patterns (common in ML/stats papers)
            ascii_math_patterns = [
                r'[A-Z]_[a-z]',           # L_G, D_x, E_z (subscripted variables)
                r'[a-z]_[a-z]+',          # p_data, p_real (word subscripts)
                r'E_?\[',                 # E[...] or E_x[...] (expectations)
                r'[A-Z]\*?\(',            # D(x), G(z), D*(x) (function calls)
                r'p\([^)]+\|[^)]+\)',     # p(x|y) conditional probability
                r'log\s*[(\[]',           # log(...) or log[...]
                r'exp\s*[(\[]',           # exp(...)
                r'max_|min_|argmax|argmin', # Optimization
                r'\s=\s.*[+\-*/]',        # Equations with operators
                r'[a-zA-Z]\s*\^\s*[0-9*]', # Superscripts: x^2, D^*
                r'\([^)]+\)\s*/\s*\(',    # Fractions: (a) / (b)
            ]

            # Check LaTeX indicators
            for indicator in latex_indicators:
                if indicator in text:
                    return True

            # Check Unicode math
            for sym in unicode_math:
                if sym in text:
                    return True

            # Check ASCII math patterns
            for pattern in ascii_math_patterns:
                if re.search(pattern, text):
                    return True

            return False

        def convert_ascii_to_latex(text: str) -> str:
            """Convert ASCII math notation to proper LaTeX for MathJax."""
            if not text:
                return text

            # Subscripts: L_G → L_{G}, p_data → p_{data}
            text = re.sub(r'([A-Za-z])_([A-Za-z]+)', r'\1_{\2}', text)

            # Superscript asterisk: D* → D^{*}
            text = re.sub(r'([A-Za-z])\*', r'\1^{*}', text)

            # Expectations: E_z[...] → \mathbb{E}_{z}[...]
            text = re.sub(r'\bE_?\{?([a-z])?\}?\[', r'\\mathbb{E}_{\1}[', text)

            # Common functions
            text = re.sub(r'\blog\b', r'\\log', text)
            text = re.sub(r'\bexp\b', r'\\exp', text)
            text = re.sub(r'\bmax\b', r'\\max', text)
            text = re.sub(r'\bmin\b', r'\\min', text)

            return text

        # Prepare formula for display
        formula_html = ""
        if formula:
            formula = formula.strip()
            # Check if already has LaTeX delimiters
            if formula.startswith('$') or formula.startswith('\\['):
                formula_html = formula  # Already formatted
            elif is_math_formula(formula):
                # Convert ASCII math to LaTeX and wrap in display math
                latex_formula = convert_ascii_to_latex(formula)
                formula_html = f'$${latex_formula}$$'
            else:
                # Plain text - render as styled text, not math
                formula_html = f'<span class="text-blue-200">{html.escape(formula)}</span>'
        else:
            formula_html = '<span class="text-gray-400 italic">Formula not provided</span>'

        # Build explanation items with inline math support
        expl_html = ""
        colors = [self.config.accent_color, self.colors["accent_blue"], self.colors["accent_green"]]
        for i, expl in enumerate(explanations[:4]):
            symbol = expl.get("symbol", "")
            meaning = html.escape(expl.get("meaning", ""))
            color = colors[i % len(colors)]
            # Keep symbol as-is for MathJax (wrap in inline math if needed)
            if symbol and not symbol.startswith('$'):
                symbol = f'${symbol}$'
            expl_html += f'''
<div class="flex items-center gap-3 animate-slide-up opacity-0 delay-{300 + i*100}">
  <span class="w-4 h-4 rounded bg-[{color}]"></span>
  <span class="text-[{self.colors['text_secondary']}]">{symbol} — {meaning}</span>
</div>'''

        # Add explanation and intuition with math support
        extra_content = ""
        if explanation:
            extra_content += f'<p class="text-[{self.colors["text_secondary"]}] mt-6 max-w-3xl mx-auto text-left leading-relaxed">{self._format_math_content(explanation)}</p>'
        if intuition:
            extra_content += f'<p class="text-[{self.colors["text_muted"]}] mt-4 max-w-3xl mx-auto italic text-left">{self._format_math_content(intuition)}</p>'

        return f'''
<div class="relative w-full min-h-screen bg-gradient-to-br from-[{self.colors['bg_secondary']}] via-[{self.colors['bg_tertiary']}] to-[{self.colors['bg_primary']}] px-4 sm:px-8 md:px-12 pt-16 sm:pt-20 pb-8 sm:pb-12 flex flex-col justify-center">
  <div class="absolute right-8 top-24 text-[64px] font-bold text-white/[0.03] pointer-events-none select-none">{index:02d}</div>
  <span class="text-[{self.config.accent_color}] text-sm font-medium tracking-widest uppercase text-center animate-fade opacity-0 delay-100">{label}</span>
  <h2 class="text-4xl font-bold text-white mt-4 text-center animate-slide-up opacity-0 delay-200">{title}</h2>
  <div class="mt-12 animate-scale opacity-0 delay-300">
    <div class="glass-glow rounded-2xl p-8 max-w-4xl mx-auto">
      <div class="text-xl md:text-2xl text-blue-200 leading-relaxed math-display">{formula_html}</div>
    </div>
    {extra_content}
    {f'<div class="flex flex-wrap justify-center gap-6 mt-10">{expl_html}</div>' if explanations else ''}
  </div>
</div>'''

    def _format_math_content(self, text: str) -> str:
        """Format content preserving and detecting math for MathJax rendering.

        - Keeps existing $...$ and $$...$$ intact
        - Detects ASCII math formulas and wraps them in $...$
        - Escapes HTML in non-math parts
        - Formats step numbers inline with content
        """
        if not text:
            return ""

        import re

        # Placeholders for protected content
        math_placeholders = []

        def protect_math(match):
            math_placeholders.append(match.group(0))
            return f'__MATH_{len(math_placeholders) - 1}__'

        # Step 1: Protect existing LaTeX delimiters
        text = re.sub(r'\$\$[^$]+\$\$', protect_math, text)
        text = re.sub(r'\$[^$]+\$', protect_math, text)

        # Step 2: Detect and wrap ASCII math formulas
        # Pattern: Variable_subscript = expression (e.g., L_G = -E_z[log D(G(z))])
        ascii_formula_patterns = [
            # Full equations: L_G = ..., D*(x) = ...
            r'[A-Z]_[A-Za-z]+\s*=\s*[^.]+(?=\.|\s*$|\s*\n)',
            r'[A-Z]\*?\([^)]+\)\s*=\s*[^.]+(?=\.|\s*$|\s*\n)',
            # Standalone expressions: p_data(x), E_z[...], D(G(z))
            r'[A-Z]_[a-z]+\([^)]+\)',
            r'E_?[a-z]?\[[^\]]+\]',
            r'[A-Z]\([A-Z]\([^)]+\)\)',
        ]

        for pattern in ascii_formula_patterns:
            def wrap_formula(match):
                formula = match.group(0).strip()
                # Don't double-wrap
                if formula.startswith('$'):
                    return formula
                # Convert ASCII to LaTeX
                latex = formula
                # Subscripts: L_G → L_{G}, p_data → p_{data}
                latex = re.sub(r'([A-Za-z])_([A-Za-z]+)', r'\1_{\2}', latex)
                # Superscript asterisk: D* → D^{*}
                latex = re.sub(r'([A-Za-z])\*', r'\1^{*}', latex)
                # Wrap and protect
                wrapped = f'${latex}$'
                math_placeholders.append(wrapped)
                return f'__MATH_{len(math_placeholders) - 1}__'

            text = re.sub(pattern, wrap_formula, text)

        # Step 3: Escape HTML in non-math parts
        text = html.escape(text)

        # Step 4: Format step patterns inline with content
        text = re.sub(
            r'^(\d+)\.\s*\*\*([^*]+)\*\*\s*',
            r'<span class="inline-flex items-baseline gap-2"><span class="flex-shrink-0 w-7 h-7 rounded-full bg-blue-500/30 text-blue-300 text-sm font-bold flex items-center justify-center">\1</span><strong class="text-white">\2</strong></span> ',
            text,
            flags=re.MULTILINE
        )

        # Handle "N:" pattern at start of line
        text = re.sub(
            r'(?:^|\n)(\d+)\s*[.:]\s*',
            r'<br/><span class="inline-flex items-baseline gap-2 mt-3"><span class="flex-shrink-0 w-7 h-7 rounded-full bg-blue-500/30 text-blue-300 text-sm font-bold flex items-center justify-center">\1</span></span> ',
            text
        )

        # Handle bold **text**
        text = re.sub(r'\*\*([^*]+)\*\*', r'<strong class="text-white font-semibold">\1</strong>', text)

        # Step 5: Restore math expressions
        for i, math in enumerate(math_placeholders):
            text = text.replace(f'__MATH_{i}__', math)

        # Convert line breaks
        text = text.replace('\n\n', '<br/><br/>')
        text = text.replace('\n', ' ')

        return text.strip()

    def _render_before_after(self, data: Dict, index: int) -> str:
        """Render before/after comparison slide"""
        label = html.escape(data.get("label", "Comparison"))
        title = html.escape(data.get("title", "The Paradigm Shift"))
        before = data.get("before", {})
        after = data.get("after", {})

        before_items = "".join([f'<li class="pl-3 border-l-2 border-red-400/50 text-[{self.colors["text_secondary"]}] my-2">{self._format_text(item)}</li>' for item in before.get("items", [])])
        after_items = "".join([f'<li class="pl-3 border-l-2 border-green-400/50 text-[{self.colors["text_secondary"]}] my-2">{self._format_text(item)}</li>' for item in after.get("items", [])])

        return f'''
<div class="relative w-full min-h-screen bg-gradient-to-br from-[{self.colors['bg_secondary']}] via-[{self.colors['bg_tertiary']}] to-[{self.colors['bg_primary']}] px-4 sm:px-8 md:px-12 pt-16 sm:pt-20 pb-8 sm:pb-12">
  <div class="absolute right-8 top-24 text-[64px] font-bold text-white/[0.03] pointer-events-none select-none">{index:02d}</div>
  <span class="text-[{self.config.accent_color}] text-sm font-medium tracking-widest uppercase">{label}</span>
  <h2 class="text-4xl font-bold text-white mt-4">{title}</h2>
  <div class="grid grid-cols-1 md:grid-cols-2 gap-8 mt-10 max-w-5xl mx-auto">
    <div class="animate-slide-left opacity-0 delay-100">
      <div class="text-center mb-4"><span class="tag bg-red-500/20 text-red-300 text-lg px-4 py-1">Before</span></div>
      <div class="glass rounded-2xl p-6 border-2 border-red-500/30">
        <h3 class="text-xl font-semibold text-white flex items-center gap-2"><span class="text-[{self.colors['accent_red']}]">✗</span> {html.escape(before.get("title", "Old Approach"))}</h3>
        <ul class="mt-4 space-y-3">{before_items}</ul>
      </div>
    </div>
    <div class="animate-slide-right opacity-0 delay-200">
      <div class="text-center mb-4"><span class="tag bg-green-500/20 text-green-300 text-lg px-4 py-1">After</span></div>
      <div class="glass rounded-2xl p-6 border-2 border-green-500/30">
        <h3 class="text-xl font-semibold text-white flex items-center gap-2"><span class="text-[{self.colors['accent_green']}]">✓</span> {html.escape(after.get("title", "New Approach"))}</h3>
        <ul class="mt-4 space-y-3">{after_items}</ul>
      </div>
    </div>
  </div>
</div>'''

    def _render_bullet_points(self, data: Dict, index: int) -> str:
        """Render bullet points slide"""
        label = html.escape(data.get("label", "Key Points"))
        title = html.escape(data.get("title", "Takeaways"))
        points = data.get("points", [])

        colors = ["blue", "purple", "teal", "orange", "pink", "green"]
        points_html = ""

        for i, point in enumerate(points[:6]):
            raw_title = point.get("title", "")
            description = point.get("description", "")
            color = colors[i % len(colors)]

            # Smart title/description splitting for long titles without descriptions
            if not description and len(raw_title) > 60:
                # Try to split at sentence boundary or after first clause
                split_patterns = ['. ', ': ', ' - ', ', which ', ', that ']
                for pattern in split_patterns:
                    if pattern in raw_title:
                        parts = raw_title.split(pattern, 1)
                        if len(parts[0]) >= 20:
                            raw_title = parts[0] + (pattern.rstrip() if pattern.endswith(' ') else '')
                            description = parts[1]
                            break

            # Use _format_text to handle **bold** and *italic* markdown
            point_title = self._format_text(raw_title)
            description = self._format_text(description) if description else ""

            points_html += f'''
<div class="flex items-start gap-4 animate-slide-left opacity-0 delay-{(i+1)*100}">
  <div class="w-10 h-10 rounded-xl bg-gradient-to-br from-{color}-500 to-{color}-600 flex items-center justify-center text-white font-bold shrink-0">{i+1}</div>
  <div>
    <h3 class="text-lg text-white">{point_title}</h3>
    {f'<p class="text-[{self.colors["text_secondary"]}] mt-1">{description}</p>' if description else ''}
  </div>
</div>'''

        return f'''
<div class="relative w-full min-h-screen bg-gradient-to-br from-[{self.colors['bg_secondary']}] via-[{self.colors['bg_tertiary']}] to-[{self.colors['bg_primary']}] px-4 sm:px-8 md:px-12 pt-16 sm:pt-20 pb-8 sm:pb-12">
  <div class="absolute right-8 top-24 text-[64px] font-bold text-white/[0.03] pointer-events-none select-none">{index:02d}</div>
  <span class="text-[{self.config.accent_color}] text-sm font-medium tracking-widest uppercase">{label}</span>
  <h2 class="text-4xl font-bold text-white mt-4">{title}</h2>
  <div class="mt-10 max-w-5xl space-y-6">{points_html}</div>
</div>'''

    def _render_definition(self, data: Dict, index: int) -> str:
        """Render definition slide"""
        term = html.escape(data.get("term", ""))
        definition = self._format_text(data.get("definition", ""))
        also_known_as = data.get("also_known_as", [])

        aka_str = ", ".join(also_known_as) if also_known_as else ""

        return f'''
<div class="relative w-full min-h-screen bg-gradient-to-br from-[{self.colors['bg_secondary']}] via-[{self.colors['bg_tertiary']}] to-[{self.colors['bg_primary']}] px-4 sm:px-8 md:px-12 pt-16 sm:pt-20 pb-8 sm:pb-12 flex flex-col justify-center">
  <div class="absolute right-8 top-24 text-[64px] font-bold text-white/[0.03] pointer-events-none select-none">{index:02d}</div>
  <div class="max-w-5xl mx-auto animate-scale opacity-0 delay-200">
    <div class="glass rounded-2xl p-8 border-l-4 border-[{self.config.accent_color}]">
      <div class="flex items-center gap-3 mb-4">
        <span class="text-3xl">📖</span>
        <span class="tag bg-[{self.config.accent_color}]/20 text-[{self.config.accent_color}]">Definition</span>
      </div>
      <h3 class="text-3xl font-bold text-white">{term}</h3>
      <p class="text-xl text-[{self.colors['text_secondary']}] mt-4 leading-relaxed">{definition}</p>
      {f'<div class="mt-6 pt-6 border-t border-[{self.colors["border"]}]"><p class="text-sm text-[{self.colors["text_muted"]}]">Also known as: {aka_str}</p></div>' if aka_str else ''}
    </div>
  </div>
</div>'''

    def _render_pros_cons(self, data: Dict, index: int) -> str:
        """Render pros/cons slide"""
        label = html.escape(data.get("label", "Trade-offs"))
        title = html.escape(data.get("title", "Advantages & Limitations"))
        pros = data.get("pros", [])
        cons = data.get("cons", [])

        pros_html = "".join([f'''
<div class="glass rounded-xl p-4 border-l-4 border-green-500">
  <p class="text-white font-medium">{self._format_text(p.get("title", ""))}</p>
  {f'<p class="text-[{self.colors["text_secondary"]}] text-sm mt-1">{self._format_text(p.get("description", ""))}</p>' if p.get("description") else ''}
</div>''' for p in pros[:4]])

        cons_html = "".join([f'''
<div class="glass rounded-xl p-4 border-l-4 border-red-500">
  <p class="text-white font-medium">{self._format_text(c.get("title", ""))}</p>
  {f'<p class="text-[{self.colors["text_secondary"]}] text-sm mt-1">{self._format_text(c.get("description", ""))}</p>' if c.get("description") else ''}
</div>''' for c in cons[:4]])

        return f'''
<div class="relative w-full min-h-screen bg-gradient-to-br from-[{self.colors['bg_secondary']}] via-[{self.colors['bg_tertiary']}] to-[{self.colors['bg_primary']}] px-4 sm:px-8 md:px-12 pt-16 sm:pt-20 pb-8 sm:pb-12">
  <div class="absolute right-8 top-24 text-[64px] font-bold text-white/[0.03] pointer-events-none select-none">{index:02d}</div>
  <span class="text-[{self.config.accent_color}] text-sm font-medium tracking-widest uppercase">{label}</span>
  <h2 class="text-4xl font-bold text-white mt-4">{title}</h2>
  <div class="grid grid-cols-1 md:grid-cols-2 gap-8 mt-10 max-w-5xl mx-auto">
    <div class="animate-slide-left opacity-0 delay-100">
      <h3 class="text-xl font-semibold text-green-400 mb-4 flex items-center gap-2"><span class="text-2xl">✓</span> Advantages</h3>
      <div class="space-y-3">{pros_html}</div>
    </div>
    <div class="animate-slide-right opacity-0 delay-200">
      <h3 class="text-xl font-semibold text-red-400 mb-4 flex items-center gap-2"><span class="text-2xl">✗</span> Limitations</h3>
      <div class="space-y-3">{cons_html}</div>
    </div>
  </div>
</div>'''

    def _render_checklist(self, data: Dict, index: int) -> str:
        """Render checklist slide"""
        label = html.escape(data.get("label", "Checklist"))
        title = html.escape(data.get("title", "Implementation Checklist"))
        items = data.get("items", [])

        items_html = ""
        completed = 0
        for i, item in enumerate(items[:8]):
            item_title = html.escape(item.get("title", ""))
            status = item.get("status", "pending")  # done, in_progress, pending

            if status == "done":
                completed += 1
                icon_class = f"bg-green-500/20 text-green-400"
                icon = "✓"
                tag = f'<span class="tag bg-green-500/20 text-green-300">Done</span>'
            elif status == "in_progress":
                icon_class = f"bg-blue-500/20 text-blue-400"
                icon = "○"
                tag = f'<span class="tag bg-blue-500/20 text-blue-300">In Progress</span>'
            else:
                icon_class = f"bg-[{self.colors['border']}] text-[{self.colors['text_muted']}]"
                icon = "○"
                tag = f'<span class="tag bg-[{self.colors["border"]}] text-[{self.colors["text_muted"]}]">Pending</span>'

            items_html += f'''
<div class="glass rounded-xl p-4 flex items-center gap-4 animate-slide-left opacity-0 delay-{(i+1)*100}">
  <div class="w-8 h-8 rounded-lg {icon_class} flex items-center justify-center text-xl">{icon}</div>
  <div class="flex-1"><p class="text-white font-medium">{item_title}</p></div>
  {tag}
</div>'''

        progress = int((completed / len(items)) * 100) if items else 0

        return f'''
<div class="relative w-full min-h-screen bg-gradient-to-br from-[{self.colors['bg_secondary']}] via-[{self.colors['bg_tertiary']}] to-[{self.colors['bg_primary']}] px-4 sm:px-8 md:px-12 pt-16 sm:pt-20 pb-8 sm:pb-12">
  <div class="absolute right-8 top-24 text-[64px] font-bold text-white/[0.03] pointer-events-none select-none">{index:02d}</div>
  <span class="text-[{self.config.accent_color}] text-sm font-medium tracking-widest uppercase">{label}</span>
  <h2 class="text-4xl font-bold text-white mt-4">{title}</h2>
  <div class="mt-10 max-w-5xl space-y-4">{items_html}</div>
  <div class="mt-8 animate-fade opacity-0 delay-600 max-w-5xl">
    <div class="progress-bar w-full"><div class="progress-fill bg-gradient-to-r from-green-500 to-blue-500" style="width: {progress}%"></div></div>
    <p class="text-[{self.colors['text_muted']}] text-sm mt-2">{completed} of {len(items)} completed ({progress}%)</p>
  </div>
</div>'''

    def _render_authors(self, data: Dict, index: int) -> str:
        """Render authors/team slide"""
        label = html.escape(data.get("label", "Authors"))
        title = html.escape(data.get("title", "Paper Authors"))
        authors = data.get("authors", self.config.authors)
        affiliations = data.get("affiliations", {})

        colors = ["blue-purple", "teal-blue", "orange-pink", "purple-pink", "green-teal", "pink-orange"]
        authors_html = ""

        for i, author in enumerate(authors[:8]):
            initials = "".join([n[0].upper() for n in author.split()[:2]])
            color_pair = colors[i % len(colors)]
            affiliation = affiliations.get(author, "")

            authors_html += f'''
<div class="text-center animate-bounce-in opacity-0 delay-{(i+1)*100}">
  <div class="w-20 h-20 mx-auto rounded-full bg-gradient-to-br from-{color_pair.split("-")[0]}-500 to-{color_pair.split("-")[1]}-500 flex items-center justify-center text-white text-2xl font-bold">{initials}</div>
  <h3 class="text-white font-semibold mt-4">{html.escape(author)}</h3>
  {f'<p class="text-[{self.colors["text_muted"]}] text-sm">{html.escape(affiliation)}</p>' if affiliation else ''}
</div>'''

        remaining = len(self.config.authors) - 8 if len(self.config.authors) > 8 else 0

        return f'''
<div class="relative w-full min-h-screen bg-gradient-to-br from-[{self.colors['bg_secondary']}] via-[{self.colors['bg_tertiary']}] to-[{self.colors['bg_primary']}] px-4 sm:px-8 md:px-12 pt-16 sm:pt-20 pb-8 sm:pb-12">
  <div class="absolute right-8 top-24 text-[64px] font-bold text-white/[0.03] pointer-events-none select-none">{index:02d}</div>
  <span class="text-[{self.config.accent_color}] text-sm font-medium tracking-widest uppercase">{label}</span>
  <h2 class="text-4xl font-bold text-white mt-4">{title}</h2>
  <div class="grid grid-cols-2 md:grid-cols-4 gap-6 mt-10">{authors_html}</div>
  {f'<div class="text-center mt-8 animate-fade opacity-0 delay-500"><p class="text-[{self.colors["text_muted"]}]">+ {remaining} more authors</p></div>' if remaining else ''}
</div>'''

    def _render_qa(self, data: Dict, index: int) -> str:
        """Render enhanced Q&A slide with pulsing effects"""
        title = html.escape(data.get("title", "Questions?"))
        subtitle = html.escape(data.get("subtitle", ""))
        cta_primary = data.get("cta_primary", {})
        cta_secondary = data.get("cta_secondary", {})

        cta_html = ""
        if cta_secondary:
            cta_html += f'<a href="{html.escape(cta_secondary.get("url", "#"))}" class="glass-glow px-6 py-3 rounded-full text-white hover:bg-white/10 transition-all hover:scale-105 flex items-center gap-2">{cta_secondary.get("icon", "📄")} {html.escape(cta_secondary.get("label", "Learn More"))}</a>'
        if cta_primary:
            cta_html += f'<a href="{html.escape(cta_primary.get("url", "#"))}" class="bg-gradient-to-r from-[{self.config.accent_color}] to-[#ed8936] px-6 py-3 rounded-full text-[{self.colors["bg_primary"]}] font-semibold hover:opacity-90 transition-all hover:scale-105 shadow-lg shadow-[{self.config.accent_color}]/30 flex items-center gap-2">{cta_primary.get("icon", "💬")} {html.escape(cta_primary.get("label", "Start Discussion"))}</a>'

        return f'''
<div class="relative w-full min-h-screen bg-gradient-to-br from-[{self.colors['bg_secondary']}] via-[{self.colors['bg_tertiary']}] to-[{self.colors['bg_primary']}] flex items-center justify-center px-4 sm:px-8 md:px-12 pt-16 sm:pt-20 pb-8 sm:pb-12 overflow-hidden">
  <!-- Animated background orbs -->
  <div class="absolute -top-20 -right-20 w-96 h-96 rounded-full bg-orange-500/10 blur-3xl animate-float"></div>
  <div class="absolute -bottom-20 -left-20 w-80 h-80 rounded-full bg-purple-500/10 blur-3xl animate-float" style="animation-delay:-3s"></div>
  <div class="absolute top-1/3 left-1/4 w-64 h-64 rounded-full bg-blue-500/5 blur-3xl animate-pulse-slow"></div>

  <!-- Floating particles -->
  <div class="absolute inset-0 pointer-events-none">
    <div class="particle" style="left: 10%; animation-delay: 1s;"></div>
    <div class="particle" style="left: 30%; animation-delay: 4s;"></div>
    <div class="particle" style="left: 50%; animation-delay: 7s;"></div>
    <div class="particle" style="left: 70%; animation-delay: 10s;"></div>
    <div class="particle" style="left: 90%; animation-delay: 13s;"></div>
  </div>

  <div class="text-center animate-scale opacity-0 delay-200 relative z-10">
    <!-- Pulsing question mark -->
    <div class="relative inline-block mb-6">
      <div class="text-8xl animate-bounce-in" style="animation-delay: 0.3s">❓</div>
      <div class="absolute inset-0 text-8xl opacity-30 animate-ping">❓</div>
    </div>

    <h2 class="text-5xl md:text-6xl font-bold text-white gradient-text animate-slide-up opacity-0 delay-300">{title}</h2>

    {f'<p class="text-xl text-[{self.colors["text_secondary"]}] mt-6 max-w-xl mx-auto animate-fade opacity-0 delay-400">{subtitle}</p>' if subtitle else ''}

    {f'<div class="flex justify-center gap-4 mt-10 animate-slide-up opacity-0 delay-500">{cta_html}</div>' if cta_html else ''}

    <!-- Bottom hint -->
    <p class="text-[{self.colors['text_muted']}] text-sm mt-12 animate-fade opacity-0 delay-600">
      Press <kbd class="px-2 py-1 bg-[{self.colors['bg_tertiary']}] rounded text-xs">←</kbd> to navigate back
    </p>
  </div>
</div>'''

    def _render_key_takeaways(self, data: Dict, index: int) -> str:
        """Render key takeaways slide"""
        return self._render_bullet_points({
            "label": data.get("label", "Summary"),
            "title": data.get("title", "Key Takeaways"),
            "points": data.get("takeaways", [])
        }, index)

    def _render_two_column(self, data: Dict, index: int) -> str:
        """Render two column slide with markdown/LaTeX support"""
        label = html.escape(data.get("label", ""))
        title = html.escape(data.get("title", ""))
        left = data.get("left", {})
        right = data.get("right", {})

        # Use _format_content for proper markdown/LaTeX/step rendering
        left_content = self._format_content(left.get("content", ""))
        right_content = self._format_content(right.get("content", ""))

        return f'''
<div class="relative w-full min-h-screen bg-gradient-to-br from-[{self.colors['bg_secondary']}] via-[{self.colors['bg_tertiary']}] to-[{self.colors['bg_primary']}] px-4 sm:px-8 md:px-12 pt-16 sm:pt-20 pb-8 sm:pb-12">
  <div class="absolute right-8 top-24 text-[64px] font-bold text-white/[0.03] pointer-events-none select-none">{index:02d}</div>
  <span class="text-[{self.config.accent_color}] text-sm font-medium tracking-widest uppercase">{label}</span>
  <h2 class="text-4xl font-bold text-white mt-4">{title}</h2>
  <div class="grid grid-cols-1 md:grid-cols-2 gap-8 mt-10">
    <div class="animate-slide-left opacity-0 delay-100 glass rounded-xl p-6">
      <h3 class="text-xl font-semibold text-white mb-4">{html.escape(left.get("title", ""))}</h3>
      <div class="text-[{self.colors['text_secondary']}] leading-relaxed">{left_content}</div>
    </div>
    <div class="animate-slide-right opacity-0 delay-200 glass rounded-xl p-6">
      <h3 class="text-xl font-semibold text-white mb-4">{html.escape(right.get("title", ""))}</h3>
      <div class="text-[{self.colors['text_secondary']}] leading-relaxed">{right_content}</div>
    </div>
  </div>
</div>'''

    def _render_architecture(self, data: Dict, index: int) -> str:
        """Render architecture slide with diagram"""
        # This reuses diagram but with specific architecture styling
        return self._render_diagram({
            "label": data.get("label", "Architecture"),
            "title": data.get("title", "System Architecture"),
            "svg": data.get("svg", self._generate_architecture_svg(data))
        }, index)

    def _generate_architecture_svg(self, data: Dict) -> str:
        """Generate architecture diagram SVG"""
        # Placeholder - would be more sophisticated in production
        return '''<svg viewBox="0 0 600 400" class="w-full">
  <rect x="50" y="50" width="150" height="80" rx="8" fill="#243b53" stroke="#4299e1" stroke-width="2"/>
  <text x="125" y="95" text-anchor="middle" fill="white" font-size="14">Encoder</text>
  <rect x="400" y="50" width="150" height="80" rx="8" fill="#243b53" stroke="#9f7aea" stroke-width="2"/>
  <text x="475" y="95" text-anchor="middle" fill="white" font-size="14">Decoder</text>
  <path d="M200 90 L400 90" stroke="#627d98" stroke-width="2" marker-end="url(#arrow)"/>
  <defs><marker id="arrow" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto"><polygon points="0 0, 10 3.5, 0 7" fill="#627d98"/></marker></defs>
</svg>'''

    def _render_chart_bar(self, data: Dict, index: int) -> str:
        """Render bar chart slide"""
        label = html.escape(data.get("label", "Data"))
        title = html.escape(data.get("title", "Performance"))
        chart_id = f"chart_{index}"
        chart_data = json.dumps(data.get("chart_data", {}))

        return f'''
<div class="relative w-full min-h-screen bg-gradient-to-br from-[{self.colors['bg_secondary']}] via-[{self.colors['bg_tertiary']}] to-[{self.colors['bg_primary']}] px-4 sm:px-8 md:px-12 pt-16 sm:pt-20 pb-8 sm:pb-12">
  <div class="absolute right-8 top-24 text-[64px] font-bold text-white/[0.03] pointer-events-none select-none">{index:02d}</div>
  <span class="text-[{self.config.accent_color}] text-sm font-medium tracking-widest uppercase">{label}</span>
  <h2 class="text-4xl font-bold text-white mt-4">{title}</h2>
  <div class="mt-8 max-w-4xl mx-auto">
    <div class="glass rounded-2xl p-6 animate-scale opacity-0 delay-200">
      <canvas id="{chart_id}" height="300"></canvas>
    </div>
  </div>
</div>
<script>
  (function() {{
    const chartData = {chart_data};
    window.initCharts = window.initCharts || function() {{}};
    const originalInit = window.initCharts;
    window.initCharts = function() {{
      originalInit();
      const ctx = document.getElementById('{chart_id}');
      if (!ctx || ctx.chart) return;
      ctx.chart = new Chart(ctx, {{
        type: 'bar',
        data: chartData,
        options: {{
          responsive: true,
          plugins: {{ legend: {{ labels: {{ color: '{self.colors["text_secondary"]}' }} }} }},
          scales: {{
            x: {{ ticks: {{ color: '{self.colors["text_secondary"]}' }}, grid: {{ color: '{self.colors["border"]}' }} }},
            y: {{ ticks: {{ color: '{self.colors["text_secondary"]}' }}, grid: {{ color: '{self.colors["border"]}' }} }}
          }}
        }}
      }});
    }};
  }})();
</script>'''

    def _render_generic(self, data: Dict, index: int) -> str:
        """Generic fallback slide renderer"""
        title = html.escape(data.get("title", f"Slide {index}"))
        content = html.escape(data.get("content", ""))

        return f'''
<div class="relative w-full min-h-screen bg-gradient-to-br from-[{self.colors['bg_secondary']}] via-[{self.colors['bg_tertiary']}] to-[{self.colors['bg_primary']}] px-4 sm:px-8 md:px-12 pt-16 sm:pt-20 pb-8 sm:pb-12">
  <div class="absolute right-8 top-24 text-[64px] font-bold text-white/[0.03] pointer-events-none select-none">{index:02d}</div>
  <h2 class="text-4xl font-bold text-white">{title}</h2>
  <p class="text-[{self.colors['text_secondary']}] mt-6 max-w-5xl">{content}</p>
</div>'''


class LearningSlideBuilder:
    """
    World-class intelligent slide builder using LIDA-inspired content analysis.

    Features:
    - Intelligent slide type selection based on content patterns
    - LIDA-inspired visualization goal generation
    - Visual rhythm optimization (text → visual → data → text)
    - Automatic variety balancing across slide types
    - Pattern detection for math, code, comparisons, timelines, etc.

    Creates presentations that are:
    - Visually varied (no repetitive slide types)
    - Content-appropriate (right viz for each concept)
    - Professionally paced (good rhythm and flow)
    """

    def __init__(self, use_llm_selector: bool = True):
        """Initialize the slide builder.

        Args:
            use_llm_selector: If True, use LLM for intelligent slide selection.
                             If False, use heuristic-based selection.
        """
        self.generator: Optional[HTMLSlideGenerator] = None
        self.type_selector = SlideTypeSelector()  # Fallback heuristic selector
        self.llm_selector = LLMSlideSelector(use_llm=use_llm_selector)  # LLM-powered selector
        self.goal_generator = VisualizationGoalGenerator()
        self.analyzer = ContentAnalyzer()
        self.use_llm_selector = use_llm_selector

    def _get_concept_icon(self, concept: Dict, index: int) -> str:
        """Get an appropriate icon for a concept based on its characteristics."""
        if concept.get("icon"):
            return concept.get("icon")

        # Smart icon selection based on content
        name = concept.get("name", "").lower()
        desc = concept.get("description", "").lower()

        if concept.get("math_required") or "formula" in name or "equation" in desc:
            return "📐"
        elif "attention" in name or "focus" in desc:
            return "🎯"
        elif "neural" in name or "network" in desc or "layer" in desc:
            return "🧠"
        elif "train" in name or "learn" in desc:
            return "📈"
        elif "transform" in name:
            return "⚡"
        elif "embed" in name or "vector" in desc:
            return "🔢"
        elif "encode" in name or "decode" in name:
            return "🔄"
        elif "loss" in name or "optim" in desc:
            return "🎚️"
        elif "data" in name or "dataset" in desc:
            return "📊"
        elif "model" in name or "architect" in desc:
            return "🏗️"

        icons = ["💡", "⚡", "🧠", "🎯", "🔬", "📐", "🔧", "🌟", "🚀", "✨"]
        return icons[index % len(icons)]

    def _select_slide_type_for_content(
        self,
        content: str,
        title: str,
        level: int = 1,
        has_code: bool = False,
        has_formula: bool = False
    ) -> SlideType:
        """
        Intelligently select the best slide type for given content.

        Uses content analysis + visualization goals + variety balancing.
        """
        # Get LIDA-inspired visualization goal
        goal = self.goal_generator.get_best_goal(content, title)

        # Get intelligent type selection with variety
        preferred = [goal.viz_type] if goal else None
        selected_type, analysis = self.type_selector.select_slide_type(
            content=content,
            title=title,
            level=level,
            force_variety=True,
            preferred_types=preferred
        )

        # Override for specific content types
        if has_code and not has_formula:
            selected_type = "CODE_BLOCK"
        elif has_formula and analysis.has_math:
            selected_type = "FORMULA"

        # Convert string to enum
        try:
            return SlideType[selected_type]
        except KeyError:
            return SlideType.TWO_COLUMN

    def _extract_stats_from_content(self, content: str) -> List[Dict]:
        """Extract statistics/metrics from content for STATS_GRID slides."""
        stats = []
        # Look for percentage patterns
        percentages = re.findall(r'(\d+(?:\.\d+)?)\s*%', content)
        for i, pct in enumerate(percentages[:4]):
            stats.append({"value": f"{pct}%", "label": f"Metric {i+1}", "color": ["blue", "purple", "orange", "green"][i]})

        # Look for multiplier patterns (10x, 100x)
        multipliers = re.findall(r'(\d+)x\b', content)
        for mult in multipliers[:2]:
            stats.append({"value": f"{mult}x", "label": "Improvement", "color": "teal"})

        return stats[:4]

    def _extract_steps_from_content(self, content: str) -> List[Dict]:
        """Extract step-by-step items from content for PROCESS_STEPS slides."""
        steps = []
        # Look for "Step N:" patterns
        step_matches = re.findall(r'Step\s*(\d+)[:.]\s*([^.]+\.)', content, re.IGNORECASE)
        for num, desc in step_matches[:6]:
            steps.append({"step": f"Step {num}", "description": desc.strip()})

        # Look for numbered list patterns if no steps found
        if not steps:
            numbered = re.findall(r'^(\d+)\.\s+(.+)$', content, re.MULTILINE)
            for num, desc in numbered[:6]:
                steps.append({"step": num, "description": desc.strip()})

        return steps

    def _extract_comparison_from_content(self, content: str) -> Tuple[Dict, Dict]:
        """Extract before/after or comparison data from content."""
        before = {"title": "Traditional Approach", "items": []}
        after = {"title": "New Approach", "items": []}

        # Simple heuristic: split on comparison words
        parts = re.split(r'\b(?:however|but|in contrast|whereas|while)\b', content, flags=re.IGNORECASE)
        if len(parts) >= 2:
            before_sentences = [s.strip() for s in parts[0].split('.') if s.strip()][:3]
            after_sentences = [s.strip() for s in parts[1].split('.') if s.strip()][:3]
            before["items"] = before_sentences
            after["items"] = after_sentences

        return before, after

    def _extract_code_from_content(self, content: str) -> str:
        """Extract code snippet from content if present."""
        # Look for code blocks
        code_match = re.search(r'```(?:\w+)?\n?(.*?)```', content, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()

        # Look for indented code-like patterns
        code_lines = []
        for line in content.split('\n'):
            if line.startswith('    ') or line.startswith('\t'):
                code_lines.append(line.strip())
            elif re.match(r'^(def |class |import |from |return |if |for |while )', line.strip()):
                code_lines.append(line.strip())

        if code_lines:
            return '\n'.join(code_lines[:20])

        return "# Code example\npass"

    def _create_timeline_from_content(self, content: str) -> List[Dict]:
        """Extract timeline events from content."""
        events = []
        # Look for year patterns
        year_matches = re.findall(r'(\b(?:19|20)\d{2}\b)[:\s]+([^.]+\.)', content)
        for year, desc in year_matches[:6]:
            events.append({"date": year, "title": desc.strip()[:50], "description": desc.strip()})

        # Look for phase/stage patterns
        if not events:
            phase_matches = re.findall(r'(?:phase|stage|step)\s*(\d+)[:\s]+([^.]+\.)', content, re.IGNORECASE)
            for num, desc in phase_matches[:6]:
                events.append({"date": f"Phase {num}", "title": desc.strip()[:50], "description": desc.strip()})

        return events

    def _suggest_visualization_for_section(self, content: str, title: str, level: int) -> str:
        """
        LIDA-inspired: Suggest the best visualization type for a section.

        Considers:
        - Content patterns (math, code, comparison, etc.)
        - Section level (fundamentals vs advanced)
        - Presentation rhythm (variety)
        """
        goal = self.goal_generator.get_best_goal(content, title)
        analysis = self.analyzer.analyze(content, title)

        # Priority order based on content
        if analysis.has_code:
            return "CODE_BLOCK"
        elif analysis.has_math and level >= 3:
            return "FORMULA"
        elif analysis.has_steps:
            return "PROCESS_STEPS"
        elif analysis.has_comparison:
            return "BEFORE_AFTER"
        elif analysis.has_numbers:
            return "STATS_GRID"
        elif goal:
            return goal.viz_type
        else:
            return "TWO_COLUMN" if len(content) > 400 else "BULLET_POINTS"

    def build_from_paper_data(self, paper_data: Dict) -> str:
        """
        Build a world-class HTML presentation using intelligent slide selection.

        Uses LIDA-inspired visualization goals and content-aware type selection
        to create varied, engaging, and content-appropriate slides.

        Args:
            paper_data: Dictionary containing paper analysis from the research swarm

        Returns:
            Complete HTML string for the presentation
        """
        # Reset selector for new presentation
        self.type_selector.reset()

        config = PresentationConfig(
            title=paper_data.get("title", "Research Paper"),
            arxiv_id=paper_data.get("arxiv_id", ""),
            authors=paper_data.get("authors", []),
        )

        self.generator = HTMLSlideGenerator(config)

        concepts = paper_data.get("concepts", [])
        sections = paper_data.get("sections", [])
        key_insights = paper_data.get("key_insights", [])

        logger.info(f"🎨 Building intelligent presentation: {len(concepts)} concepts, {len(sections)} sections")

        # ==========================================
        # SLIDE 1: TITLE HERO (always first)
        # ==========================================
        hook_text = paper_data.get("hook", paper_data.get("abstract", ""))
        self.generator.add_slide(SlideType.TITLE_HERO, {
            "title": paper_data.get("title", ""),
            "hook": hook_text[:400] + "..." if len(hook_text) > 400 else hook_text,
            "arxiv_id": paper_data.get("arxiv_id", ""),
            "authors": paper_data.get("authors", []),
            "tags": paper_data.get("tags", ["Research", "AI", "Machine Learning"]),
        })

        # ==========================================
        # SLIDE 2: OVERVIEW STATS (impactful opening)
        # ==========================================
        self.generator.add_slide(SlideType.STATS_GRID, {
            "label": "Paper Overview",
            "title": "What You'll Learn",
            "stats": [
                {"value": str(len(concepts)) if concepts else "3+", "label": "Key Concepts", "color": "blue"},
                {"value": str(len(sections)) if sections else "5+", "label": "Deep Dives", "color": "purple"},
                {"value": paper_data.get("year", "2025"), "label": "Year", "color": "orange"},
                {"value": paper_data.get("learning_time", "20-30 min"), "label": "Learning Time", "color": "green"},
            ]
        })

        # ==========================================
        # SLIDE 3: KEY CONCEPTS OVERVIEW
        # ==========================================
        if concepts:
            self.generator.add_slide(SlideType.FEATURE_CARDS, {
                "label": "Core Contributions",
                "title": "Key Concepts",
                "features": [
                    {
                        "icon": self._get_concept_icon(c, i),
                        "title": c.get("name", ""),
                        "description": c.get("description", "")[:150] + "..." if len(c.get("description", "")) > 150 else c.get("description", ""),
                        "code": c.get("formula", ""),
                    }
                    for i, c in enumerate(concepts[:6])
                ]
            })

        # ==========================================
        # CONCEPT SLIDES - LLM-POWERED CONSISTENT FORMAT
        # All concepts use the same slide type for consistency.
        # The LLM selector ensures: same content_type → same slide format.
        # ==========================================
        # Reset LLM selector for fresh consistency tracking
        self.llm_selector.reset()

        for i, concept in enumerate(concepts):
            name = concept.get("name", f"Concept {i+1}")
            description = concept.get("description", "")
            why_matters = concept.get("why_it_matters", "")
            formula = concept.get("formula", concept.get("code_example", ""))
            icon = self._get_concept_icon(concept, i)

            # Use LLM selector for intelligent, consistent slide type selection
            full_content = f"{description}\n\n{why_matters}\n\n{formula}" if formula else f"{description}\n\n{why_matters}"

            slide_type_str, metadata = self.llm_selector.select(
                content=full_content,
                title=name,
                content_type="concept",  # All concepts get consistent treatment
                force_consistency=True
            )

            try:
                slide_type = SlideType[slide_type_str]
            except KeyError:
                slide_type = SlideType.TWO_COLUMN

            # Build slide data based on selected type
            if slide_type == SlideType.FORMULA and metadata.get("has_math", False):
                self.generator.add_slide(SlideType.FORMULA, {
                    "label": f"Concept {i+1}",
                    "title": name,
                    "formula": formula,
                    "explanation": description,
                    "intuition": why_matters,
                })
            else:
                # TWO_COLUMN is the consistent default for concepts
                self.generator.add_slide(SlideType.TWO_COLUMN, {
                    "label": f"Concept {i+1}",
                    "title": name,
                    "left": {
                        "title": "What It Is",
                        "content": description,
                        "icon": icon
                    },
                    "right": {
                        "title": "Why It Matters",
                        "content": why_matters if why_matters else "Understanding this concept is key to grasping the paper's contribution."
                    }
                })

        # ==========================================
        # INTELLIGENT SECTION SLIDES
        # Each section gets the BEST slide type based on content analysis
        # ==========================================
        level_labels = {
            1: "Fundamentals",
            2: "Intuition",
            3: "Mathematics",
            4: "Applications",
            5: "Advanced"
        }

        level_icons = {
            1: "📚",
            2: "💡",
            3: "📐",
            4: "🚀",
            5: "🔬"
        }

        for i, section in enumerate(sections):
            title = section.get("title", f"Section {i+1}")
            content = section.get("content", "")
            level = section.get("level", 1)
            code = section.get("code_example", "")

            if not content:
                continue

            # Determine content type for LLM selector based on level and content
            content_type = "section"
            if level == 3 or "math" in title.lower() or self.llm_selector._has_latex(content):
                content_type = "math"
            elif code or "```" in content or "def " in content:
                content_type = "code"
            elif any(word in content.lower() for word in ["versus", "vs", "compared to", "unlike", "whereas"]):
                content_type = "comparison"

            # Use LLM selector for intelligent selection
            # Sections can have variety (force_consistency=False for sections)
            selected_type_str, metadata = self.llm_selector.select(
                content=content,
                title=title,
                content_type=content_type,
                force_consistency=False  # Allow variety in sections
            )

            logger.debug(f"📊 Section '{title[:30]}...': type={content_type}, selected={selected_type_str}")

            # ===== CODE SECTIONS =====
            if content_type == "code" or metadata.get("has_code"):
                self.generator.add_slide(SlideType.CODE_BLOCK, {
                    "label": level_labels.get(level, "Implementation"),
                    "title": title,
                    "code": code if code else self._extract_code_from_content(content),
                    "language": "python",
                    "description": content[:200] if not code else content,
                })

            # ===== MATH SECTIONS =====
            elif content_type == "math" and metadata.get("has_math", self.llm_selector._has_latex(content)):
                # Extract actual LaTeX formulas - only use FORMULA slide if we find real math
                formulas = re.findall(r'\$\$([^$]+)\$\$|\$([^$]+)\$', content)
                main_formula = ""
                if formulas:
                    # Get the first non-empty formula
                    for f in formulas:
                        main_formula = f[0] if f[0] else f[1]
                        if main_formula:
                            break

                if main_formula:
                    # Found actual LaTeX - use FORMULA slide
                    self.generator.add_slide(SlideType.FORMULA, {
                        "label": level_labels.get(level, "Mathematics"),
                        "title": title,
                        "formula": f"$${main_formula}$$",
                        "explanation": content,
                        "intuition": "",
                    })
                else:
                    # No actual LaTeX found - use TWO_COLUMN for math explanation
                    sentences = content.split('. ')
                    mid = len(sentences) // 2
                    left_content = '. '.join(sentences[:mid]) + '.' if mid > 0 else content[:len(content)//2]
                    right_content = '. '.join(sentences[mid:]) if mid > 0 else content[len(content)//2:]
                    self.generator.add_slide(SlideType.TWO_COLUMN, {
                        "label": level_labels.get(level, "Mathematics"),
                        "title": title,
                        "left": {"title": "📐 The Math", "content": left_content},
                        "right": {"title": "💡 Intuition", "content": right_content}
                    })

            # ===== COMPARISON SECTIONS =====
            elif content_type == "comparison":
                before, after = self._extract_comparison_from_content(content)
                self.generator.add_slide(SlideType.BEFORE_AFTER, {
                    "label": level_labels.get(level, "Comparison"),
                    "title": title,
                    "before": before if before["items"] else {"title": "Challenge", "items": [content[:100]]},
                    "after": after if after["items"] else {"title": "Solution", "items": ["Addressed by this approach"]},
                })

            # ===== GENERAL SECTIONS - Use TWO_COLUMN for consistency =====
            else:
                # Split content intelligently for two-column layout
                sentences = content.split('. ')
                mid = len(sentences) // 2
                left_content = '. '.join(sentences[:mid]) + '.' if mid > 0 else content[:len(content)//2]
                right_content = '. '.join(sentences[mid:]) if mid > 0 else content[len(content)//2:]
                self.generator.add_slide(SlideType.TWO_COLUMN, {
                    "label": level_labels.get(level, "Content"),
                    "title": title,
                    "left": {"title": level_icons.get(level, "📖") + " Understanding", "content": left_content},
                    "right": {"title": "🎯 Application", "content": right_content}
                })

        # Log variety score
        variety = self.type_selector.get_variety_score()
        logger.info(f"🎨 Slide variety score: {variety:.1%} ({len(set(self.type_selector.used_types))} unique types)")

        # ==========================================
        # METHODOLOGY (if not covered in sections)
        # ==========================================
        methodology = paper_data.get("methodology_steps", [])
        if methodology:
            self.generator.add_slide(SlideType.PROCESS_STEPS, {
                "label": "Methodology",
                "title": "How It Works",
                "steps": methodology[:6]
            })

        # ==========================================
        # COMPARISON: Before/After
        # ==========================================
        if paper_data.get("comparison"):
            self.generator.add_slide(SlideType.BEFORE_AFTER, {
                "label": "Innovation",
                "title": paper_data.get("comparison", {}).get("title", "The Paradigm Shift"),
                "before": paper_data.get("comparison", {}).get("before", {}),
                "after": paper_data.get("comparison", {}).get("after", {}),
            })

        # ==========================================
        # RESULTS TABLE
        # ==========================================
        if paper_data.get("results"):
            self.generator.add_slide(SlideType.COMPARISON_TABLE, {
                "label": "Results",
                "title": "Performance",
                "headers": paper_data.get("results", {}).get("headers", []),
                "rows": paper_data.get("results", {}).get("rows", []),
                "highlight_row": paper_data.get("results", {}).get("highlight_row", -1),
            })

        # ==========================================
        # TIMELINE
        # ==========================================
        if paper_data.get("timeline"):
            self.generator.add_slide(SlideType.TIMELINE, {
                "label": "Context",
                "title": "Historical Evolution",
                "events": paper_data.get("timeline", [])
            })

        # ==========================================
        # KEY INSIGHTS (combine all insights in one rich slide)
        # ==========================================
        if key_insights:
            self.generator.add_slide(SlideType.BULLET_POINTS, {
                "label": "Key Insights",
                "title": "What You Should Remember",
                "points": [
                    {"title": ins if isinstance(ins, str) else ins.get("title", ""), "description": "" if isinstance(ins, str) else ins.get("description", "")}
                    for ins in key_insights[:6]
                ]
            })

        # ==========================================
        # SUMMARY + NEXT STEPS (combined)
        # ==========================================
        summary = paper_data.get("summary", "")
        next_steps = paper_data.get("next_steps", [])
        takeaways = paper_data.get("takeaways", [])

        if summary or takeaways:
            # Combine summary points and takeaways
            all_points = []
            if summary:
                all_points.extend([s.strip() + '.' for s in summary.split('. ') if s.strip()][:3])
            for t in takeaways[:3]:
                if isinstance(t, str):
                    all_points.append(t)
                elif isinstance(t, dict):
                    all_points.append(t.get("title", ""))

            self.generator.add_slide(SlideType.KEY_TAKEAWAYS, {
                "label": "Summary",
                "title": "Key Takeaways",
                "takeaways": [{"title": p, "description": ""} for p in all_points[:6]]
            })

        if next_steps:
            self.generator.add_slide(SlideType.ICON_GRID, {
                "label": "What's Next",
                "title": "Continue Learning",
                "items": [
                    {
                        "icon": ["📚", "🔬", "💻", "🧪", "📊", "🎯", "🚀", "🔗"][i % 8],
                        "title": s if isinstance(s, str) else s.get("title", ""),
                        "description": "" if isinstance(s, str) else s.get("description", "")
                    }
                    for i, s in enumerate(next_steps[:8])
                ]
            })

        # ==========================================
        # QUOTE (if available)
        # ==========================================
        if paper_data.get("key_quote"):
            self.generator.add_slide(SlideType.QUOTE, {
                "quote": paper_data.get("key_quote", ""),
                "author": paper_data.get("authors", [""])[0] + " et al." if paper_data.get("authors") else "",
                "source": paper_data.get("title", ""),
            })

        # ==========================================
        # AUTHORS
        # ==========================================
        self.generator.add_slide(SlideType.AUTHORS, {
            "label": "Credits",
            "title": "Paper Authors",
            "authors": paper_data.get("authors", []),
            "affiliations": paper_data.get("affiliations", {}),
        })

        # ==========================================
        # Q&A
        # ==========================================
        self.generator.add_slide(SlideType.QA, {
            "title": "Questions?",
            "subtitle": f"Let's discuss {paper_data.get('title', 'this research')}",
            "cta_primary": {"label": "Discuss", "icon": "💬", "url": "#"},
            "cta_secondary": {"label": "Read Paper", "icon": "📄", "url": f"https://arxiv.org/abs/{paper_data.get('arxiv_id', '')}"},
        })

        return self.generator.generate()

    def save(self, html_content: str, output_path: str):
        """Save the generated HTML to a file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)


# Example usage and test
if __name__ == "__main__":
    # Test with sample paper data
    sample_paper = {
        "title": "Attention Is All You Need",
        "arxiv_id": "1706.03762",
        "authors": ["Vaswani", "Shazeer", "Parmar", "Uszkoreit", "Jones", "Gomez", "Kaiser", "Polosukhin"],
        "abstract": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms.",
        "tags": ["Transformers", "NLP", "Deep Learning"],
        "year": "2017",
        "citations": "100K+",
        "concepts": [
            {"name": "Self-Attention", "description": "Mechanism relating different positions of a sequence", "icon": "🧠", "formula": "Attention(Q,K,V)"},
            {"name": "Multi-Head Attention", "description": "Parallel attention layers for diverse representations", "icon": "⚡", "formula": "h=8"},
            {"name": "Positional Encoding", "description": "Inject sequence order with sinusoidal functions", "icon": "📍", "formula": "sin(pos/10000^(2i/d))"},
        ],
        "methodology_steps": [
            {"title": "Input Embedding", "description": "Convert tokens to vectors"},
            {"title": "Position Encoding", "description": "Add positional information"},
            {"title": "Attention Layers", "description": "Process through N=6 layers"},
            {"title": "Output Softmax", "description": "Generate probability distribution"},
        ],
        "comparison": {
            "title": "The Paradigm Shift",
            "before": {"title": "RNN/LSTM", "items": ["Sequential processing", "Limited parallelization", "Vanishing gradients"]},
            "after": {"title": "Transformer", "items": ["Parallel processing", "Full parallelization", "Stable gradients"]},
        },
        "timeline": [
            {"year": "2017", "title": "Transformer", "description": "Original paper introduces attention-only architecture", "highlight": True},
            {"year": "2018", "title": "BERT & GPT", "description": "Pre-training revolution begins"},
            {"year": "2020", "title": "GPT-3", "description": "175B parameters, few-shot learning"},
            {"year": "2023+", "title": "GPT-4, Claude", "description": "Multimodal, reasoning, agents"},
        ],
        "key_quote": "We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.",
        "takeaways": [
            {"title": "Attention replaces recurrence", "description": "No more sequential processing bottleneck"},
            {"title": "Multi-head provides diversity", "description": "8 parallel attention heads"},
            {"title": "Positional encoding preserves order", "description": "Sinusoidal functions inject position"},
            {"title": "Residual connections enable depth", "description": "Skip connections and layer norm"},
        ],
    }

    builder = LearningSlideBuilder()
    html = builder.build_from_paper_data(sample_paper)

    # Save test output
    output_path = "/var/www/sites/personal/stock_market/Jotty/learning-slides/public/generated-test.html"
    builder.save(html, output_path)
    print(f"Generated presentation saved to: {output_path}")
