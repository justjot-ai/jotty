"""
Slide Renderers Mixin - HTML rendering methods for each slide type.

Extracted from HTMLSlideGenerator in html_slide_generator.py.
"""

from __future__ import annotations

import html
import logging
import re
from typing import TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:
    from .html_slide_generator import SlideConfig, SlideType

logger = logging.getLogger(__name__)


class SlideRenderersMixin:
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

        tags_html = "".join(
            [
                f'<span class="tag bg-{["blue", "purple", "teal", "orange"][i % 4]}-500/20 text-{["blue", "purple", "teal", "orange"][i % 4]}-300 hover:scale-105 transition-transform">{html.escape(tag)}</span>'
                for i, tag in enumerate(tags[:4])
            ]
        )

        authors_str = " · ".join(authors[:3])
        if len(authors) > 3:
            authors_str += " · et al."

        return f"""
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
</div>"""

    def _render_arxiv_badge(self, arxiv_id: str) -> str:
        """Render ArXiv ID badge"""
        return f"""<div class="mt-8 animate-scale opacity-0 delay-400">
      <span class="inline-flex items-center px-4 py-2 bg-gradient-to-r from-[{self.config.accent_color}] to-[#ed8936] text-[{self.colors["bg_primary"]}] rounded-full text-sm font-bold shadow-lg shadow-[{self.config.accent_color}]/30 hover:scale-105 transition-transform cursor-default">
        <svg class="w-4 h-4 mr-2" fill="currentColor" viewBox="0 0 24 24"><path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zm-7 3c1.93 0 3.5 1.57 3.5 3.5S13.93 13 12 13s-3.5-1.57-3.5-3.5S10.07 6 12 6zm7 13H5v-.23c0-.62.28-1.2.76-1.58C7.47 15.82 9.64 15 12 15s4.53.82 6.24 2.19c.48.38.76.97.76 1.58V19z"/></svg>
        {html.escape(arxiv_id)}
      </span>
    </div>"""

    def _render_title_minimal(self, data: Dict, index: int) -> str:
        """Render minimal title slide"""
        title = html.escape(data.get("title", ""))
        subtitle = html.escape(data.get("subtitle", ""))

        return f"""
<div class="relative w-full min-h-screen bg-gradient-to-br from-[{self.colors['bg_secondary']}] via-[{self.colors['bg_tertiary']}] to-[{self.colors['bg_primary']}] flex items-center justify-center px-4 sm:px-8 md:px-12 pt-16 sm:pt-20 pb-8 sm:pb-12">
  <div class="text-center animate-scale opacity-0 delay-200">
    <h1 class="text-6xl md:text-7xl font-bold text-white">{title}</h1>
    {f'<p class="text-2xl text-[{self.colors["text_secondary"]}] mt-6">{subtitle}</p>' if subtitle else ''}
  </div>
</div>"""

    def _render_title_centered(self, data: Dict, index: int) -> str:
        """Render centered title slide with label"""
        label = html.escape(data.get("label", ""))
        title = html.escape(data.get("title", ""))
        subtitle = html.escape(data.get("subtitle", ""))

        return f"""
<div class="relative w-full min-h-screen bg-gradient-to-br from-[{self.colors['bg_secondary']}] via-[{self.colors['bg_tertiary']}] to-[{self.colors['bg_primary']}] px-4 sm:px-8 md:px-12 pt-16 sm:pt-20 pb-8 sm:pb-12">
  <div class="absolute right-8 top-24 text-[64px] font-bold text-white/[0.03] pointer-events-none select-none">{index:02d}</div>
  <span class="text-[{self.config.accent_color}] text-sm font-medium tracking-widest uppercase animate-fade opacity-0 delay-100">{label}</span>
  <h2 class="text-4xl md:text-5xl font-bold text-white mt-4 animate-slide-up opacity-0 delay-200">{title}</h2>
  {f'<p class="text-[{self.colors["text_secondary"]}] mt-3 max-w-5xl animate-fade opacity-0 delay-300">{subtitle}</p>' if subtitle else ''}
</div>"""

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
            accent = self.colors.get(f"accent_{color}", self.config.accent_color)

            # Check if value is numeric for counter animation
            is_numeric = (
                str(value)
                .replace(".", "")
                .replace("%", "")
                .replace("+", "")
                .replace("K", "")
                .replace("M", "")
                .isdigit()
            )
            suffix = ""
            numeric_val = value
            if is_numeric:
                if "K" in str(value):
                    numeric_val = str(value).replace("K", "").replace("+", "")
                    suffix = "K+"
                elif "M" in str(value):
                    numeric_val = str(value).replace("M", "").replace("+", "")
                    suffix = "M+"
                elif "%" in str(value):
                    numeric_val = str(value).replace("%", "")
                    suffix = "%"
                value_html = f'<span class="counter" data-target="{numeric_val}" data-suffix="{suffix}">0</span>'
            else:
                value_html = html.escape(str(value))

            stats_html += f"""
<div class="stat-card glass-glow rounded-2xl p-6 text-center animate-slide-up opacity-0 delay-{(i+1)*100}">
  <div class="text-5xl font-bold text-[{accent}] text-glow">{value_html}</div>
  <div class="text-[{self.colors['text_secondary']}] mt-2 font-medium">{label_text}</div>
  {f'<div class="mt-4 progress-bar"><div class="progress-fill progress-animated" data-width="{progress}" style="width:0%"></div></div>' if progress else ''}
</div>"""

        return f"""
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
</div>"""

    def _render_feature_cards(self, data: Dict, index: int) -> str:
        """Render feature cards slide with enhanced visual effects"""
        label = html.escape(data.get("label", "Features"))
        title = html.escape(data.get("title", "Key Features"))
        features = data.get("features", [])

        colors = ["blue", "purple", "teal", "orange", "pink", "green"]
        features_html = ""

        for i, feat in enumerate(features[:6]):
            icon = feat.get("icon", "")
            feat_title = html.escape(feat.get("title", ""))
            description = self._format_text(feat.get("description", ""))
            code = feat.get("code", "")
            color = colors[i % len(colors)]

            features_html += f"""
<div class="gradient-border-glow card-3d animate-slide-up opacity-0 delay-{(i+1)*100}">
  <div class="bg-[{self.colors['bg_card']}] rounded-[14px] p-6 h-full">
    <div class="icon-square bg-{color}-500/20 text-{color}-400 mb-4 icon-glow" style="--tw-shadow-color: var(--tw-{color}-400);">{icon}</div>
    <h3 class="text-xl font-semibold text-white">{feat_title}</h3>
    <p class="text-[{self.colors['text_secondary']}] mt-3 text-sm leading-relaxed">{description}</p>
    {f'<div class="mt-4 pt-4 border-t border-[{self.colors["border"]}]"><code class="code-inline text-xs shimmer">{html.escape(code)}</code></div>' if code else ''}
  </div>
</div>"""

        return f"""
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
</div>"""

    def _render_comparison_table(self, data: Dict, index: int) -> str:
        """Render comparison table slide"""
        label = html.escape(data.get("label", "Comparison"))
        title = html.escape(data.get("title", "Model Comparison"))
        headers = data.get("headers", [])
        rows = data.get("rows", [])
        highlight_row = data.get("highlight_row", -1)

        header_html = "".join(
            [
                f'<th class="text-center p-4 text-[{self.colors["text_secondary"]}] font-medium">{html.escape(h)}</th>'
                for h in headers
            ]
        )

        rows_html = ""
        for i, row in enumerate(rows):
            is_highlight = i == highlight_row
            row_class = (
                f'bg-gradient-to-r from-[{self.colors["accent_green"]}]/10 to-transparent border-l-4 border-[{self.colors["accent_green"]}]'
                if is_highlight
                else f'bg-[{self.colors["bg_tertiary"]}]/30 hover:bg-[{self.colors["bg_tertiary"]}]/50 transition-colors'
            )

            cells_html = ""
            for j, cell in enumerate(row):
                if j == 0:
                    color = (
                        self.colors["accent_green"]
                        if is_highlight
                        else self.colors["text_secondary"]
                    )
                    cells_html += f'<td class="p-4"><div class="flex items-center gap-3"><span class="w-3 h-3 rounded-full bg-[{color}]"></span><span class="font-{"bold" if is_highlight else "medium"} {"text-[" + self.colors["accent_green"] + "]" if is_highlight else ""}">{html.escape(str(cell))}</span></div></td>'
                else:
                    cells_html += f'<td class="text-center p-4 {"font-bold text-[" + self.colors["accent_green"] + "]" if is_highlight else ""}">{html.escape(str(cell))}</td>'

            rows_html += f'<tr class="{row_class}">{cells_html}</tr>'

        return f"""
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
</div>"""

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

            steps_html += f"""
<div class="relative animate-slide-up opacity-0 delay-{(i+1)*100}">
  <div class="step-number bg-{color}-500 text-white mb-4 relative z-10">{i+1}</div>
  <div class="glass rounded-xl p-5">
    <h3 class="font-semibold text-white">{step_title}</h3>
    <p class="text-[{self.colors['text_secondary']}] text-sm mt-2">{description}</p>
  </div>
</div>"""

        return f"""
<div class="relative w-full min-h-screen bg-gradient-to-br from-[{self.colors['bg_secondary']}] via-[{self.colors['bg_tertiary']}] to-[{self.colors['bg_primary']}] px-4 sm:px-8 md:px-12 pt-16 sm:pt-20 pb-8 sm:pb-12">
  <div class="absolute right-8 top-24 text-[64px] font-bold text-white/[0.03] pointer-events-none select-none">{index:02d}</div>
  <span class="text-[{self.config.accent_color}] text-sm font-medium tracking-widest uppercase">{label}</span>
  <h2 class="text-4xl font-bold text-white mt-4">{title}</h2>
  <div class="relative mt-12 max-w-5xl mx-auto">
    <div class="absolute top-8 left-8 right-8 h-0.5 bg-gradient-to-r from-blue-500 via-purple-500 to-pink-500 hidden md:block"></div>
    <div class="grid grid-cols-1 md:grid-cols-{min(len(steps), 4)} gap-8">{steps_html}</div>
  </div>
</div>"""

    def _render_quote(self, data: Dict, index: int) -> str:
        """Render quote slide"""
        quote = html.escape(data.get("quote", ""))
        author = html.escape(data.get("author", ""))
        source = html.escape(data.get("source", ""))
        initials = "".join([n[0].upper() for n in author.split()[:2]]) if author else "?"

        return f"""
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
</div>"""

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
        if "```" in raw_code:
            # Find last ``` and split - anything after is explanation
            parts = raw_code.rsplit("```", 1)
            if len(parts) == 2:
                code_part = parts[0] + "```"
                explanation_part = parts[1].strip()
                # Extract just the code without markdown code block markers
                code_match = re.search(r"```[\w]*\n?(.*?)```", code_part, re.DOTALL)
                if code_match:
                    code_part = code_match.group(1)

        code = html.escape(code_part)
        explanation_html = self._format_content(explanation_part) if explanation_part else ""

        tags_html = "".join(
            [
                f'<span class="tag bg-{["blue", "purple", "green"][i % 3]}-500/20 text-{["blue", "purple", "green"][i % 3]}-300">{html.escape(t)}</span>'
                for i, t in enumerate(tags[:4])
            ]
        )

        return f"""
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
</div>"""

    def _render_timeline(self, data: Dict, index: int) -> str:
        """Render timeline slide"""
        label = html.escape(data.get("label", "Timeline"))
        title = html.escape(data.get("title", "Evolution"))
        events = data.get("events", [])

        colors = [
            self.config.accent_color,
            self.colors["accent_blue"],
            self.colors["accent_teal"],
            self.colors["accent_purple"],
        ]
        events_html = ""

        for i, event in enumerate(events[:6]):
            year = html.escape(str(event.get("year", "")))
            event_title = html.escape(event.get("title", ""))
            description = self._format_text(event.get("description", ""))
            color = colors[i % len(colors)]
            is_highlight = event.get("highlight", False)

            events_html += f"""
<div class="flex items-start gap-6 animate-slide-left opacity-0 delay-{(i+1)*100}">
  <div class="w-[120px] text-right shrink-0"><span class="text-[{color}] font-bold text-lg">{year}</span></div>
  <div class="w-4 h-4 rounded-full bg-[{color}] ring-4 ring-[{color}]/30 shrink-0 mt-1"></div>
  <div class="glass rounded-xl p-5 flex-1 {"border-l-4 border-[" + color + "]" if is_highlight else ""}">
    <h3 class="font-semibold text-white">{event_title}</h3>
    <p class="text-[{self.colors['text_secondary']}] text-sm mt-1">{description}</p>
  </div>
</div>"""

        return f"""
<div class="relative w-full min-h-screen bg-gradient-to-br from-[{self.colors['bg_secondary']}] via-[{self.colors['bg_tertiary']}] to-[{self.colors['bg_primary']}] px-4 sm:px-8 md:px-12 pt-16 sm:pt-20 pb-8 sm:pb-12">
  <div class="absolute right-8 top-24 text-[64px] font-bold text-white/[0.03] pointer-events-none select-none">{index:02d}</div>
  <span class="text-[{self.config.accent_color}] text-sm font-medium tracking-widest uppercase">{label}</span>
  <h2 class="text-4xl font-bold text-white mt-4">{title}</h2>
  <div class="mt-10 max-w-4xl mx-auto relative">
    <div class="absolute left-[60px] top-0 bottom-0 w-0.5 bg-gradient-to-b from-[{self.config.accent_color}] via-[{self.colors['accent_blue']}] to-[{self.colors['accent_purple']}]"></div>
    <div class="space-y-6">{events_html}</div>
  </div>
</div>"""

    def _render_diagram(self, data: Dict, index: int) -> str:
        """Render diagram slide with SVG"""
        label = html.escape(data.get("label", "Diagram"))
        title = html.escape(data.get("title", "Architecture"))
        svg_content = data.get("svg", "")  # Pre-generated SVG or use default

        if not svg_content:
            svg_content = self._generate_default_diagram(data)

        return f"""
<div class="relative w-full min-h-screen bg-gradient-to-br from-[{self.colors['bg_secondary']}] via-[{self.colors['bg_tertiary']}] to-[{self.colors['bg_primary']}] px-4 sm:px-8 md:px-12 pt-16 sm:pt-20 pb-8 sm:pb-12">
  <div class="absolute right-8 top-24 text-[64px] font-bold text-white/[0.03] pointer-events-none select-none">{index:02d}</div>
  <span class="text-[{self.config.accent_color}] text-sm font-medium tracking-widest uppercase">{label}</span>
  <h2 class="text-4xl font-bold text-white mt-4">{title}</h2>
  <div class="mt-8 flex justify-center animate-scale opacity-0 delay-200">
    <div class="glass rounded-2xl p-8 max-w-4xl w-full">{svg_content}</div>
  </div>
</div>"""

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
            icon = item.get("icon", "")
            item_title = html.escape(item.get("title", ""))
            description = self._format_text(item.get("description", ""))
            url = item.get("url", "")

            # Generate search URL if no URL provided
            if not url and item_title:
                search_query = item_title.replace(" ", "+")
                url = f"https://www.google.com/search?q={search_query}"

            card_content = f"""
  <div class="text-5xl mb-4 group-hover:scale-110 transition-transform">{icon}</div>
  <h3 class="font-semibold text-white group-hover:text-[{self.config.accent_color}] transition-colors">{item_title}</h3>
  <p class="text-[{self.colors['text_muted']}] text-sm mt-2">{description}</p>
  <div class="mt-3 opacity-0 group-hover:opacity-100 transition-opacity">
    <span class="text-[{self.config.accent_color}] text-xs flex items-center justify-center gap-1">
      Learn more <svg class="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17 8l4 4m0 0l-4 4m4-4H3"/></svg>
    </span>
  </div>"""

            if url:
                items_html += f"""
<a href="{html.escape(url)}" target="_blank" rel="noopener" class="group glass rounded-2xl p-6 text-center card-hover animate-bounce-in opacity-0 delay-{(i+1)*100} cursor-pointer block hover:border-[{self.config.accent_color}]/50 border border-transparent transition-all">
{card_content}
</a>"""
            else:
                items_html += f"""
<div class="group glass rounded-2xl p-6 text-center card-hover animate-bounce-in opacity-0 delay-{(i+1)*100}">
{card_content}
</div>"""

        return f"""
<div class="relative w-full min-h-screen bg-gradient-to-br from-[{self.colors['bg_secondary']}] via-[{self.colors['bg_tertiary']}] to-[{self.colors['bg_primary']}] px-4 sm:px-8 md:px-12 pt-16 sm:pt-20 pb-8 sm:pb-12">
  <div class="absolute right-8 top-24 text-[64px] font-bold text-white/[0.03] pointer-events-none select-none">{index:02d}</div>
  <span class="text-[{self.config.accent_color}] text-sm font-medium tracking-widest uppercase">{label}</span>
  <h2 class="text-4xl font-bold text-white mt-4">{title}</h2>
  <div class="grid grid-cols-2 md:grid-cols-4 gap-6 mt-10">{items_html}</div>
</div>"""

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
                r"\\frac",
                r"\\sum",
                r"\\int",
                r"\\prod",
                r"\\alpha",
                r"\\beta",
                r"\\gamma",
                r"\\theta",
                r"\\partial",
                r"\\nabla",
                r"\\infty",
                r"\\left",
                r"\\right",
                r"\\mathbb",
                r"\\mathcal",
            ]

            # Unicode math symbols
            unicode_math = ["∑", "∫", "∂", "∇", "∞", "α", "β", "γ", "θ", "λ", "σ", "π", "μ"]

            # ASCII math patterns (common in ML/stats papers)
            ascii_math_patterns = [
                r"[A-Z]_[a-z]",  # L_G, D_x, E_z (subscripted variables)
                r"[a-z]_[a-z]+",  # p_data, p_real (word subscripts)
                r"E_?\[",  # E[...] or E_x[...] (expectations)
                r"[A-Z]\*?\(",  # D(x), G(z), D*(x) (function calls)
                r"p\([^)]+\|[^)]+\)",  # p(x|y) conditional probability
                r"log\s*[(\[]",  # log(...) or log[...]
                r"exp\s*[(\[]",  # exp(...)
                r"max_|min_|argmax|argmin",  # Optimization
                r"\s=\s.*[+\-*/]",  # Equations with operators
                r"[a-zA-Z]\s*\^\s*[0-9*]",  # Superscripts: x^2, D^*
                r"\([^)]+\)\s*/\s*\(",  # Fractions: (a) / (b)
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
            text = re.sub(r"([A-Za-z])_([A-Za-z]+)", r"\1_{\2}", text)

            # Superscript asterisk: D* → D^{*}
            text = re.sub(r"([A-Za-z])\*", r"\1^{*}", text)

            # Expectations: E_z[...] → \mathbb{E}_{z}[...]
            text = re.sub(r"\bE_?\{?([a-z])?\}?\[", r"\\mathbb{E}_{\1}[", text)

            # Common functions
            text = re.sub(r"\blog\b", r"\\log", text)
            text = re.sub(r"\bexp\b", r"\\exp", text)
            text = re.sub(r"\bmax\b", r"\\max", text)
            text = re.sub(r"\bmin\b", r"\\min", text)

            return text

        # Prepare formula for display
        formula_html = ""
        if formula:
            formula = formula.strip()
            # Check if already has LaTeX delimiters
            if formula.startswith("$") or formula.startswith("\\["):
                formula_html = formula  # Already formatted
            elif is_math_formula(formula):
                # Convert ASCII math to LaTeX and wrap in display math
                latex_formula = convert_ascii_to_latex(formula)
                formula_html = f"$${latex_formula}$$"
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
            if symbol and not symbol.startswith("$"):
                symbol = f"${symbol}$"
            expl_html += f"""
<div class="flex items-center gap-3 animate-slide-up opacity-0 delay-{300 + i*100}">
  <span class="w-4 h-4 rounded bg-[{color}]"></span>
  <span class="text-[{self.colors['text_secondary']}]">{symbol} — {meaning}</span>
</div>"""

        # Add explanation and intuition with math support
        extra_content = ""
        if explanation:
            extra_content += f'<p class="text-[{self.colors["text_secondary"]}] mt-6 max-w-3xl mx-auto text-left leading-relaxed">{self._format_math_content(explanation)}</p>'
        if intuition:
            extra_content += f'<p class="text-[{self.colors["text_muted"]}] mt-4 max-w-3xl mx-auto italic text-left">{self._format_math_content(intuition)}</p>'

        return f"""
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
</div>"""

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

        def protect_math(match: Any) -> Any:
            math_placeholders.append(match.group(0))
            return f"__MATH_{len(math_placeholders) - 1}__"

        # Step 1: Protect existing LaTeX delimiters
        text = re.sub(r"\$\$[^$]+\$\$", protect_math, text)
        text = re.sub(r"\$[^$]+\$", protect_math, text)

        # Step 2: Detect and wrap ASCII math formulas
        # Pattern: Variable_subscript = expression (e.g., L_G = -E_z[log D(G(z))])
        ascii_formula_patterns = [
            # Full equations: L_G = ..., D*(x) = ...
            r"[A-Z]_[A-Za-z]+\s*=\s*[^.]+(?=\.|\s*$|\s*\n)",
            r"[A-Z]\*?\([^)]+\)\s*=\s*[^.]+(?=\.|\s*$|\s*\n)",
            # Standalone expressions: p_data(x), E_z[...], D(G(z))
            r"[A-Z]_[a-z]+\([^)]+\)",
            r"E_?[a-z]?\[[^\]]+\]",
            r"[A-Z]\([A-Z]\([^)]+\)\)",
        ]

        for pattern in ascii_formula_patterns:

            def wrap_formula(match: Any) -> Any:
                formula = match.group(0).strip()
                # Don't double-wrap
                if formula.startswith("$"):
                    return formula
                # Convert ASCII to LaTeX
                latex = formula
                # Subscripts: L_G → L_{G}, p_data → p_{data}
                latex = re.sub(r"([A-Za-z])_([A-Za-z]+)", r"\1_{\2}", latex)
                # Superscript asterisk: D* → D^{*}
                latex = re.sub(r"([A-Za-z])\*", r"\1^{*}", latex)
                # Wrap and protect
                wrapped = f"${latex}$"
                math_placeholders.append(wrapped)
                return f"__MATH_{len(math_placeholders) - 1}__"

            text = re.sub(pattern, wrap_formula, text)

        # Step 3: Escape HTML in non-math parts
        text = html.escape(text)

        # Step 4: Format step patterns inline with content
        text = re.sub(
            r"^(\d+)\.\s*\*\*([^*]+)\*\*\s*",
            r'<span class="inline-flex items-baseline gap-2"><span class="flex-shrink-0 w-7 h-7 rounded-full bg-blue-500/30 text-blue-300 text-sm font-bold flex items-center justify-center">\1</span><strong class="text-white">\2</strong></span> ',
            text,
            flags=re.MULTILINE,
        )

        # Handle "N:" pattern at start of line
        text = re.sub(
            r"(?:^|\n)(\d+)\s*[.:]\s*",
            r'<br/><span class="inline-flex items-baseline gap-2 mt-3"><span class="flex-shrink-0 w-7 h-7 rounded-full bg-blue-500/30 text-blue-300 text-sm font-bold flex items-center justify-center">\1</span></span> ',
            text,
        )

        # Handle bold **text**
        text = re.sub(
            r"\*\*([^*]+)\*\*", r'<strong class="text-white font-semibold">\1</strong>', text
        )

        # Step 5: Restore math expressions
        for i, math in enumerate(math_placeholders):
            text = text.replace(f"__MATH_{i}__", math)

        # Convert line breaks
        text = text.replace("\n\n", "<br/><br/>")
        text = text.replace("\n", " ")

        return text.strip()

    def _render_before_after(self, data: Dict, index: int) -> str:
        """Render before/after comparison slide"""
        label = html.escape(data.get("label", "Comparison"))
        title = html.escape(data.get("title", "The Paradigm Shift"))
        before = data.get("before", {})
        after = data.get("after", {})

        before_items = "".join(
            [
                f'<li class="pl-3 border-l-2 border-red-400/50 text-[{self.colors["text_secondary"]}] my-2">{self._format_text(item)}</li>'
                for item in before.get("items", [])
            ]
        )
        after_items = "".join(
            [
                f'<li class="pl-3 border-l-2 border-green-400/50 text-[{self.colors["text_secondary"]}] my-2">{self._format_text(item)}</li>'
                for item in after.get("items", [])
            ]
        )

        return f"""
<div class="relative w-full min-h-screen bg-gradient-to-br from-[{self.colors['bg_secondary']}] via-[{self.colors['bg_tertiary']}] to-[{self.colors['bg_primary']}] px-4 sm:px-8 md:px-12 pt-16 sm:pt-20 pb-8 sm:pb-12">
  <div class="absolute right-8 top-24 text-[64px] font-bold text-white/[0.03] pointer-events-none select-none">{index:02d}</div>
  <span class="text-[{self.config.accent_color}] text-sm font-medium tracking-widest uppercase">{label}</span>
  <h2 class="text-4xl font-bold text-white mt-4">{title}</h2>
  <div class="grid grid-cols-1 md:grid-cols-2 gap-8 mt-10 max-w-5xl mx-auto">
    <div class="animate-slide-left opacity-0 delay-100">
      <div class="text-center mb-4"><span class="tag bg-red-500/20 text-red-300 text-lg px-4 py-1">Before</span></div>
      <div class="glass rounded-2xl p-6 border-2 border-red-500/30">
        <h3 class="text-xl font-semibold text-white flex items-center gap-2"><span class="text-[{self.colors['accent_red']}]"></span> {html.escape(before.get("title", "Old Approach"))}</h3>
        <ul class="mt-4 space-y-3">{before_items}</ul>
      </div>
    </div>
    <div class="animate-slide-right opacity-0 delay-200">
      <div class="text-center mb-4"><span class="tag bg-green-500/20 text-green-300 text-lg px-4 py-1">After</span></div>
      <div class="glass rounded-2xl p-6 border-2 border-green-500/30">
        <h3 class="text-xl font-semibold text-white flex items-center gap-2"><span class="text-[{self.colors['accent_green']}]"></span> {html.escape(after.get("title", "New Approach"))}</h3>
        <ul class="mt-4 space-y-3">{after_items}</ul>
      </div>
    </div>
  </div>
</div>"""

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
                split_patterns = [". ", ": ", " - ", ", which ", ", that "]
                for pattern in split_patterns:
                    if pattern in raw_title:
                        parts = raw_title.split(pattern, 1)
                        if len(parts[0]) >= 20:
                            raw_title = parts[0] + (
                                pattern.rstrip() if pattern.endswith(" ") else ""
                            )
                            description = parts[1]
                            break

            # Use _format_text to handle **bold** and *italic* markdown
            point_title = self._format_text(raw_title)
            description = self._format_text(description) if description else ""

            points_html += f"""
<div class="flex items-start gap-4 animate-slide-left opacity-0 delay-{(i+1)*100}">
  <div class="w-10 h-10 rounded-xl bg-gradient-to-br from-{color}-500 to-{color}-600 flex items-center justify-center text-white font-bold shrink-0">{i+1}</div>
  <div>
    <h3 class="text-lg text-white">{point_title}</h3>
    {f'<p class="text-[{self.colors["text_secondary"]}] mt-1">{description}</p>' if description else ''}
  </div>
</div>"""

        return f"""
<div class="relative w-full min-h-screen bg-gradient-to-br from-[{self.colors['bg_secondary']}] via-[{self.colors['bg_tertiary']}] to-[{self.colors['bg_primary']}] px-4 sm:px-8 md:px-12 pt-16 sm:pt-20 pb-8 sm:pb-12">
  <div class="absolute right-8 top-24 text-[64px] font-bold text-white/[0.03] pointer-events-none select-none">{index:02d}</div>
  <span class="text-[{self.config.accent_color}] text-sm font-medium tracking-widest uppercase">{label}</span>
  <h2 class="text-4xl font-bold text-white mt-4">{title}</h2>
  <div class="mt-10 max-w-5xl space-y-6">{points_html}</div>
</div>"""

    def _render_definition(self, data: Dict, index: int) -> str:
        """Render definition slide"""
        term = html.escape(data.get("term", ""))
        definition = self._format_text(data.get("definition", ""))
        also_known_as = data.get("also_known_as", [])

        aka_str = ", ".join(also_known_as) if also_known_as else ""

        return f"""
<div class="relative w-full min-h-screen bg-gradient-to-br from-[{self.colors['bg_secondary']}] via-[{self.colors['bg_tertiary']}] to-[{self.colors['bg_primary']}] px-4 sm:px-8 md:px-12 pt-16 sm:pt-20 pb-8 sm:pb-12 flex flex-col justify-center">
  <div class="absolute right-8 top-24 text-[64px] font-bold text-white/[0.03] pointer-events-none select-none">{index:02d}</div>
  <div class="max-w-5xl mx-auto animate-scale opacity-0 delay-200">
    <div class="glass rounded-2xl p-8 border-l-4 border-[{self.config.accent_color}]">
      <div class="flex items-center gap-3 mb-4">
        <span class="text-3xl"></span>
        <span class="tag bg-[{self.config.accent_color}]/20 text-[{self.config.accent_color}]">Definition</span>
      </div>
      <h3 class="text-3xl font-bold text-white">{term}</h3>
      <p class="text-xl text-[{self.colors['text_secondary']}] mt-4 leading-relaxed">{definition}</p>
      {f'<div class="mt-6 pt-6 border-t border-[{self.colors["border"]}]"><p class="text-sm text-[{self.colors["text_muted"]}]">Also known as: {aka_str}</p></div>' if aka_str else ''}
    </div>
  </div>
</div>"""

    def _render_pros_cons(self, data: Dict, index: int) -> str:
        """Render pros/cons slide"""
        label = html.escape(data.get("label", "Trade-offs"))
        title = html.escape(data.get("title", "Advantages & Limitations"))
        pros = data.get("pros", [])
        cons = data.get("cons", [])

        pros_html = "".join(
            [
                f"""
<div class="glass rounded-xl p-4 border-l-4 border-green-500">
  <p class="text-white font-medium">{self._format_text(p.get("title", ""))}</p>
  {f'<p class="text-[{self.colors["text_secondary"]}] text-sm mt-1">{self._format_text(p.get("description", ""))}</p>' if p.get("description") else ''}
</div>"""
                for p in pros[:4]
            ]
        )

        cons_html = "".join(
            [
                f"""
<div class="glass rounded-xl p-4 border-l-4 border-red-500">
  <p class="text-white font-medium">{self._format_text(c.get("title", ""))}</p>
  {f'<p class="text-[{self.colors["text_secondary"]}] text-sm mt-1">{self._format_text(c.get("description", ""))}</p>' if c.get("description") else ''}
</div>"""
                for c in cons[:4]
            ]
        )

        return f"""
<div class="relative w-full min-h-screen bg-gradient-to-br from-[{self.colors['bg_secondary']}] via-[{self.colors['bg_tertiary']}] to-[{self.colors['bg_primary']}] px-4 sm:px-8 md:px-12 pt-16 sm:pt-20 pb-8 sm:pb-12">
  <div class="absolute right-8 top-24 text-[64px] font-bold text-white/[0.03] pointer-events-none select-none">{index:02d}</div>
  <span class="text-[{self.config.accent_color}] text-sm font-medium tracking-widest uppercase">{label}</span>
  <h2 class="text-4xl font-bold text-white mt-4">{title}</h2>
  <div class="grid grid-cols-1 md:grid-cols-2 gap-8 mt-10 max-w-5xl mx-auto">
    <div class="animate-slide-left opacity-0 delay-100">
      <h3 class="text-xl font-semibold text-green-400 mb-4 flex items-center gap-2"><span class="text-2xl"></span> Advantages</h3>
      <div class="space-y-3">{pros_html}</div>
    </div>
    <div class="animate-slide-right opacity-0 delay-200">
      <h3 class="text-xl font-semibold text-red-400 mb-4 flex items-center gap-2"><span class="text-2xl"></span> Limitations</h3>
      <div class="space-y-3">{cons_html}</div>
    </div>
  </div>
</div>"""

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
                icon = ""
                tag = f'<span class="tag bg-green-500/20 text-green-300">Done</span>'
            elif status == "in_progress":
                icon_class = f"bg-blue-500/20 text-blue-400"
                icon = "○"
                tag = f'<span class="tag bg-blue-500/20 text-blue-300">In Progress</span>'
            else:
                icon_class = f"bg-[{self.colors['border']}] text-[{self.colors['text_muted']}]"
                icon = "○"
                tag = f'<span class="tag bg-[{self.colors["border"]}] text-[{self.colors["text_muted"]}]">Pending</span>'

            items_html += f"""
<div class="glass rounded-xl p-4 flex items-center gap-4 animate-slide-left opacity-0 delay-{(i+1)*100}">
  <div class="w-8 h-8 rounded-lg {icon_class} flex items-center justify-center text-xl">{icon}</div>
  <div class="flex-1"><p class="text-white font-medium">{item_title}</p></div>
  {tag}
</div>"""

        progress = int((completed / len(items)) * 100) if items else 0

        return f"""
<div class="relative w-full min-h-screen bg-gradient-to-br from-[{self.colors['bg_secondary']}] via-[{self.colors['bg_tertiary']}] to-[{self.colors['bg_primary']}] px-4 sm:px-8 md:px-12 pt-16 sm:pt-20 pb-8 sm:pb-12">
  <div class="absolute right-8 top-24 text-[64px] font-bold text-white/[0.03] pointer-events-none select-none">{index:02d}</div>
  <span class="text-[{self.config.accent_color}] text-sm font-medium tracking-widest uppercase">{label}</span>
  <h2 class="text-4xl font-bold text-white mt-4">{title}</h2>
  <div class="mt-10 max-w-5xl space-y-4">{items_html}</div>
  <div class="mt-8 animate-fade opacity-0 delay-600 max-w-5xl">
    <div class="progress-bar w-full"><div class="progress-fill bg-gradient-to-r from-green-500 to-blue-500" style="width: {progress}%"></div></div>
    <p class="text-[{self.colors['text_muted']}] text-sm mt-2">{completed} of {len(items)} completed ({progress}%)</p>
  </div>
</div>"""

    def _render_authors(self, data: Dict, index: int) -> str:
        """Render authors/team slide"""
        label = html.escape(data.get("label", "Authors"))
        title = html.escape(data.get("title", "Paper Authors"))
        authors = data.get("authors", self.config.authors)
        affiliations = data.get("affiliations", {})

        colors = [
            "blue-purple",
            "teal-blue",
            "orange-pink",
            "purple-pink",
            "green-teal",
            "pink-orange",
        ]
        authors_html = ""

        for i, author in enumerate(authors[:8]):
            initials = "".join([n[0].upper() for n in author.split()[:2]])
            color_pair = colors[i % len(colors)]
            affiliation = affiliations.get(author, "")

            authors_html += f"""
<div class="text-center animate-bounce-in opacity-0 delay-{(i+1)*100}">
  <div class="w-20 h-20 mx-auto rounded-full bg-gradient-to-br from-{color_pair.split("-")[0]}-500 to-{color_pair.split("-")[1]}-500 flex items-center justify-center text-white text-2xl font-bold">{initials}</div>
  <h3 class="text-white font-semibold mt-4">{html.escape(author)}</h3>
  {f'<p class="text-[{self.colors["text_muted"]}] text-sm">{html.escape(affiliation)}</p>' if affiliation else ''}
</div>"""

        remaining = len(self.config.authors) - 8 if len(self.config.authors) > 8 else 0

        return f"""
<div class="relative w-full min-h-screen bg-gradient-to-br from-[{self.colors['bg_secondary']}] via-[{self.colors['bg_tertiary']}] to-[{self.colors['bg_primary']}] px-4 sm:px-8 md:px-12 pt-16 sm:pt-20 pb-8 sm:pb-12">
  <div class="absolute right-8 top-24 text-[64px] font-bold text-white/[0.03] pointer-events-none select-none">{index:02d}</div>
  <span class="text-[{self.config.accent_color}] text-sm font-medium tracking-widest uppercase">{label}</span>
  <h2 class="text-4xl font-bold text-white mt-4">{title}</h2>
  <div class="grid grid-cols-2 md:grid-cols-4 gap-6 mt-10">{authors_html}</div>
  {f'<div class="text-center mt-8 animate-fade opacity-0 delay-500"><p class="text-[{self.colors["text_muted"]}]">+ {remaining} more authors</p></div>' if remaining else ''}
</div>"""

    def _render_qa(self, data: Dict, index: int) -> str:
        """Render enhanced Q&A slide with pulsing effects"""
        title = html.escape(data.get("title", "Questions?"))
        subtitle = html.escape(data.get("subtitle", ""))
        cta_primary = data.get("cta_primary", {})
        cta_secondary = data.get("cta_secondary", {})

        cta_html = ""
        if cta_secondary:
            cta_html += f'<a href="{html.escape(cta_secondary.get("url", "#"))}" class="glass-glow px-6 py-3 rounded-full text-white hover:bg-white/10 transition-all hover:scale-105 flex items-center gap-2">{cta_secondary.get("icon", "")} {html.escape(cta_secondary.get("label", "Learn More"))}</a>'
        if cta_primary:
            cta_html += f'<a href="{html.escape(cta_primary.get("url", "#"))}" class="bg-gradient-to-r from-[{self.config.accent_color}] to-[#ed8936] px-6 py-3 rounded-full text-[{self.colors["bg_primary"]}] font-semibold hover:opacity-90 transition-all hover:scale-105 shadow-lg shadow-[{self.config.accent_color}]/30 flex items-center gap-2">{cta_primary.get("icon", "")} {html.escape(cta_primary.get("label", "Start Discussion"))}</a>'

        return f"""
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
      <div class="text-8xl animate-bounce-in" style="animation-delay: 0.3s"></div>
      <div class="absolute inset-0 text-8xl opacity-30 animate-ping"></div>
    </div>

    <h2 class="text-5xl md:text-6xl font-bold text-white gradient-text animate-slide-up opacity-0 delay-300">{title}</h2>

    {f'<p class="text-xl text-[{self.colors["text_secondary"]}] mt-6 max-w-xl mx-auto animate-fade opacity-0 delay-400">{subtitle}</p>' if subtitle else ''}

    {f'<div class="flex justify-center gap-4 mt-10 animate-slide-up opacity-0 delay-500">{cta_html}</div>' if cta_html else ''}

    <!-- Bottom hint -->
    <p class="text-[{self.colors['text_muted']}] text-sm mt-12 animate-fade opacity-0 delay-600">
      Press <kbd class="px-2 py-1 bg-[{self.colors['bg_tertiary']}] rounded text-xs">←</kbd> to navigate back
    </p>
  </div>
</div>"""

    def _render_key_takeaways(self, data: Dict, index: int) -> str:
        """Render key takeaways slide"""
        return self._render_bullet_points(
            {
                "label": data.get("label", "Summary"),
                "title": data.get("title", "Key Takeaways"),
                "points": data.get("takeaways", []),
            },
            index,
        )

    def _render_two_column(self, data: Dict, index: int) -> str:
        """Render two column slide with markdown/LaTeX support"""
        label = html.escape(data.get("label", ""))
        title = html.escape(data.get("title", ""))
        left = data.get("left", {})
        right = data.get("right", {})

        # Use _format_content for proper markdown/LaTeX/step rendering
        left_content = self._format_content(left.get("content", ""))
        right_content = self._format_content(right.get("content", ""))

        return f"""
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
</div>"""

    def _render_architecture(self, data: Dict, index: int) -> str:
        """Render architecture slide with diagram"""
        # This reuses diagram but with specific architecture styling
        return self._render_diagram(
            {
                "label": data.get("label", "Architecture"),
                "title": data.get("title", "System Architecture"),
                "svg": data.get("svg", self._generate_architecture_svg(data)),
            },
            index,
        )

    def _generate_architecture_svg(self, data: Dict) -> str:
        """Generate architecture diagram SVG"""
        # Placeholder - would be more sophisticated in production
        return """<svg viewBox="0 0 600 400" class="w-full">
  <rect x="50" y="50" width="150" height="80" rx="8" fill="#243b53" stroke="#4299e1" stroke-width="2"/>
  <text x="125" y="95" text-anchor="middle" fill="white" font-size="14">Encoder</text>
  <rect x="400" y="50" width="150" height="80" rx="8" fill="#243b53" stroke="#9f7aea" stroke-width="2"/>
  <text x="475" y="95" text-anchor="middle" fill="white" font-size="14">Decoder</text>
  <path d="M200 90 L400 90" stroke="#627d98" stroke-width="2" marker-end="url(#arrow)"/>
  <defs><marker id="arrow" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto"><polygon points="0 0, 10 3.5, 0 7" fill="#627d98"/></marker></defs>
</svg>"""

    def _render_chart_bar(self, data: Dict, index: int) -> str:
        """Render bar chart slide"""
        label = html.escape(data.get("label", "Data"))
        title = html.escape(data.get("title", "Performance"))
        chart_id = f"chart_{index}"
        chart_data = json.dumps(data.get("chart_data", {}))

        return f"""
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
</script>"""

    def _render_generic(self, data: Dict, index: int) -> str:
        """Generic fallback slide renderer"""
        title = html.escape(data.get("title", f"Slide {index}"))
        content = html.escape(data.get("content", ""))

        return f"""
<div class="relative w-full min-h-screen bg-gradient-to-br from-[{self.colors['bg_secondary']}] via-[{self.colors['bg_tertiary']}] to-[{self.colors['bg_primary']}] px-4 sm:px-8 md:px-12 pt-16 sm:pt-20 pb-8 sm:pb-12">
  <div class="absolute right-8 top-24 text-[64px] font-bold text-white/[0.03] pointer-events-none select-none">{index:02d}</div>
  <h2 class="text-4xl font-bold text-white">{title}</h2>
  <p class="text-[{self.colors['text_secondary']}] mt-6 max-w-5xl">{content}</p>
</div>"""
