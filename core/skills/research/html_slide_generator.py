"""
HTML Slide Generator
====================

Generates world-class interactive HTML presentations from research paper data.

Classes are split across modules:
- content_analysis.py: Pattern learning, content type detection
- slide_selection.py: Intelligent slide type selection
- _slide_assets_mixin.py: CSS, navigation, JavaScript
- _slide_renderers_mixin.py: Per-slide-type HTML renderers
"""

import os
import re
import json
import html
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

try:
    import dspy
except ImportError:
    dspy = None

from .content_analysis import (
    ComponentPattern, PatternCrystallizer, ResearchTemplateStructure,
    MetaSchemaRegistry, ComponentMapper, ContentPattern, ContentAnalysis, ContentAnalyzer,
)
from .slide_selection import (
    SlideTypeSelector, LLMSlideSelector, VisualizationGoal, VisualizationGoalGenerator,
)
from ._slide_assets_mixin import SlideAssetsMixin
from ._slide_renderers_mixin import SlideRenderersMixin

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



class HTMLSlideGenerator(SlideAssetsMixin, SlideRenderersMixin):
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

    def add_slide(self, slide_type: SlideType, data -> None: Dict[str, Any],
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

    def save(self, html_content: str, output_path: str) -> None:
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
    logger.info(f"Generated presentation saved to: {output_path}")
