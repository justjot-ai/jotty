"""
Slide Assets Mixin - CSS styles, navigation, branding, and JavaScript for presentations.

Extracted from HTMLSlideGenerator in html_slide_generator.py.
"""

import logging
from typing import Dict

logger = logging.getLogger(__name__)


class SlideAssetsMixin:
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

