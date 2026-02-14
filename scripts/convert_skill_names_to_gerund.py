#!/usr/bin/env python3
"""
Convert all SKILL.md `name:` fields to gerund form (verb-ing).

Rules:
- Only modifies the `name:` field in YAML frontmatter
- Does NOT rename directories
- Lowercase, hyphens only, max 64 chars
"""

import os
import re
from pathlib import Path

SKILLS_DIR = Path("/var/www/sites/personal/stock_market/Jotty/skills")

# ============================================================
# Explicit mapping for every skill (138 skills)
# Format: "original-name" -> "gerund-name"
# ============================================================
EXPLICIT_MAP = {
    # --- Core tools ---
    "web-search": "searching-web",
    "calculator": "calculating",
    "file-operations": "managing-files",
    "shell-exec": "executing-shell",
    "process-manager": "managing-processes",
    "http-client": "requesting-http",
    "text-utils": "processing-text",
    "text-chunker": "chunking-text",
    "terminal-session": "managing-terminal-sessions",
    "database-tools": "managing-databases",
    "voice": "processing-voice",

    # --- Web/scraping ---
    "web-scraper": "scraping-web",
    "browser-automation": "automating-browser",
    "webapp-testing": "testing-webapps",

    # --- Document/file tools ---
    "document-converter": "converting-documents",
    "pdf-tools": "processing-pdfs",
    "docx-tools": "processing-docx",
    "xlsx-tools": "processing-xlsx",
    "pptx-editor": "editing-pptx",
    "simple-pdf-generator": "generating-simple-pdfs",
    "latex-generator": "generating-latex",
    "notebooklm-pdf": "generating-notebooklm-pdfs",

    # --- Image/media ---
    "image-generator": "generating-images",
    "image-enhancer": "enhancing-images",
    "openai-image-gen": "generating-openai-images",
    "gif-creator": "creating-gifs",
    "video-downloader": "downloading-videos",
    "youtube-downloader": "downloading-youtube",
    "visual-inspector": "inspecting-visuals",
    "media-production-pipeline": "producing-media-pipeline",
    "canvas-design": "designing-canvas",
    "algorithmic-art": "creating-algorithmic-art",

    # --- Communication platforms ---
    "telegram-sender": "sending-telegram",
    "discord": "messaging-discord",
    "slack": "messaging-slack",
    "slack-gif-creator": "creating-slack-gifs",
    "whatsapp": "messaging-whatsapp",
    "whatsapp-reader": "reading-whatsapp",
    "internal-comms": "managing-internal-comms",
    "remarkable-sender": "sending-remarkable",
    "remarkable-upload": "uploading-remarkable",

    # --- Productivity platforms ---
    "github": "managing-github",
    "notion": "managing-notion",
    "obsidian": "managing-obsidian",
    "trello": "managing-trello",
    "spotify": "managing-spotify",

    # --- LLM providers ---
    "claude-api-llm": "calling-claude-api",
    "claude-cli-llm": "calling-claude-cli",
    "gemini": "calling-gemini",
    "openai-whisper-api": "transcribing-openai-whisper",

    # --- Research/download ---
    "arxiv-downloader": "downloading-arxiv",
    "summarize": "summarizing",
    "lead-research-assistant": "researching-leads",

    # --- Data science / ML ---
    "automl": "running-automl",
    "auto-sklearn": "running-auto-sklearn",
    "pycaret": "running-pycaret",
    "hyperopt": "optimizing-hyperparameters",
    "clustering": "clustering-data",
    "dimensionality-reduction": "reducing-dimensionality",
    "feature-engineer": "engineering-features",
    "feature-tools": "processing-features",
    "ensemble-builder": "building-ensembles",
    "model-metrics": "evaluating-model-metrics",
    "statistical-tests": "running-statistical-tests",
    "shap-explainer": "explaining-shap",
    "time-series": "analyzing-time-series",
    "data-profiler": "profiling-data",
    "data-validator": "validating-data",
    "technical-analysis": "analyzing-technicals",
    "lida-to-justjot": "sending-lida-to-justjot",

    # --- Financial ---
    "financial-analysis": "analyzing-financials",
    "financial-visualization": "visualizing-financials",
    "screener-financials": "screening-financials",
    "stock-research-comprehensive": "researching-stocks-comprehensive",
    "stock-research-deep": "researching-stocks-deep",
    "investing-commodities": "analyzing-commodities",
    "investing-commodities-to-telegram": "sending-commodities-to-telegram",

    # --- PMI (PlanMyInvesting) ---
    "pmi-alerts": "alerting-pmi",
    "pmi-broker": "brokering-pmi",
    "pmi-market-data": "fetching-pmi-market-data",
    "pmi-portfolio": "managing-pmi-portfolio",
    "pmi-strategies": "running-pmi-strategies",
    "pmi-trading": "trading-pmi",
    "pmi-watchlist": "managing-pmi-watchlist",

    # --- Content/branding ---
    "content-repurposer": "repurposing-content",
    "content-pipeline": "running-content-pipeline",
    "content-branding-pipeline": "running-content-branding-pipeline",
    "content-research-writer": "writing-content-research",
    "brand-guidelines": "managing-brand-guidelines",
    "competitive-ads-extractor": "extracting-competitive-ads",
    "product-launch-pipeline": "launching-products-pipeline",
    "theme-factory": "creating-themes",
    "composite-templates": "composing-templates",

    # --- Converters/utilities ---
    "time-converter": "converting-time",
    "weather-checker": "checking-weather",
    "file-organizer": "organizing-files",
    "invoice-organizer": "organizing-invoices",
    "domain-name-brainstormer": "brainstorming-domain-names",
    "raffle-winner-picker": "picking-raffle-winners",
    "justjot-converters": "converting-justjot",
    "oauth-automation": "automating-oauth",

    # --- Generators ---
    "changelog-generator": "generating-changelogs",
    "mindmap-generator": "generating-mindmaps",
    "slide-generator": "generating-slides",
    "presenton": "presenting-presenton",

    # --- Pipelines (multi-step) ---
    "search-summarize-pdf-telegram": "searching-summarizing-pdf-telegram",
    "search-summarize-pdf-telegram-v2": "searching-summarizing-pdf-telegram-v2",
    "search-to-justjot-idea": "searching-to-justjot-idea",
    "screener-to-pdf-telegram": "screening-to-pdf-telegram",
    "last30days-claude-cli": "researching-last30days-claude-cli",
    "last30days-to-epub-telegram": "sending-last30days-epub-telegram",
    "last30days-to-pdf-remarkable": "sending-last30days-pdf-remarkable",
    "last30days-to-pdf-telegram": "sending-last30days-pdf-telegram",
    "research-to-notion": "researching-to-notion",
    "research-to-pdf": "researching-to-pdf",
    "reddit-trending-to-justjot": "fetching-reddit-trending-to-justjot",
    "trending-topics-to-ideas": "discovering-trending-topics-to-ideas",
    "transformer-paper-pipeline": "processing-transformer-papers",
    "v2v-to-pdf-telegram-remarkable": "converting-v2v-to-pdf-telegram",
    "v2v-trending-search": "searching-v2v-trending",

    # --- Notion pipelines ---
    "notion-knowledge-capture": "capturing-notion-knowledge",
    "notion-knowledge-pipeline": "running-notion-knowledge-pipeline",
    "notion-meeting-intelligence": "analyzing-notion-meetings",
    "notion-research-documentation": "documenting-notion-research",
    "notion-spec-to-implementation": "implementing-notion-specs",

    # --- Meeting/insights ---
    "meeting-insights-analyzer": "analyzing-meeting-insights",
    "meeting-intelligence-pipeline": "running-meeting-intelligence",

    # --- Dev/skills ---
    "dev-workflow": "managing-dev-workflow",
    "skill-composer": "composing-skills",
    "skill-creator": "creating-skills",
    "skill-discovery": "discovering-skills",
    "mcp-builder": "building-mcp",
    "mcp-justjot": "connecting-mcp-justjot",
    "n8n-workflows": "managing-n8n-workflows",
    "multi-source-aggregator": "aggregating-multi-source",

    # --- Testing ---
    "test-skill-validation": "validating-test-skills",

    # --- Artifacts ---
    "artifacts-builder": "building-artifacts",

    # --- Android ---
    "android-automation": "automating-android",

    # --- Outputs (if present) ---
    "outputs": "managing-outputs",
}


def convert_skill_name(original):
    """Convert a skill name to gerund form using explicit map or fallback rules."""
    # Check explicit map first
    if original in EXPLICIT_MAP:
        result = EXPLICIT_MAP[original]
    else:
        # Fallback: should not happen if map is complete, but just in case
        result = "using-" + original
        print(f"  WARNING: No explicit mapping for '{original}', using fallback: '{result}'")

    # Enforce constraints
    result = result.lower()
    result = re.sub(r'[^a-z0-9-]', '-', result)
    result = re.sub(r'-+', '-', result).strip('-')
    if len(result) > 64:
        result = result[:64].rstrip('-')

    return result


def process_skill_md(filepath):
    """Process a single SKILL.md file and update the name field.
    Returns (old_name, new_name) or None if no change needed.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Check for YAML frontmatter
    if not content.startswith('---'):
        return None

    # Find the end of frontmatter
    end_idx = content.find('---', 3)
    if end_idx == -1:
        return None

    frontmatter = content[3:end_idx]
    rest = content[end_idx:]

    # Find and replace the name field
    name_match = re.search(r'^name:\s*(.+)$', frontmatter, re.MULTILINE)
    if not name_match:
        return None

    old_name = name_match.group(1).strip().strip('"').strip("'")
    new_name = convert_skill_name(old_name)

    if old_name == new_name:
        return None

    # Replace the name line in frontmatter
    new_frontmatter = re.sub(
        r'^name:\s*.+$',
        'name: ' + new_name,
        frontmatter,
        count=1,
        flags=re.MULTILINE
    )

    new_content = '---' + new_frontmatter + rest
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(new_content)

    return (old_name, new_name)


def main():
    results = []
    errors = []

    # Find all SKILL.md files
    skill_files = sorted(SKILLS_DIR.glob("*/SKILL.md"))
    print(f"Found {len(skill_files)} SKILL.md files\n")

    for skill_file in skill_files:
        try:
            result = process_skill_md(skill_file)
            if result:
                results.append(result)
        except Exception as e:
            errors.append((skill_file, str(e)))

    # Print summary
    print(f"Total SKILL.md files processed: {len(skill_files)}")
    print(f"Names converted: {len(results)}")
    if errors:
        print(f"Errors: {len(errors)}")
        for path, err in errors:
            print(f"  ERROR: {path}: {err}")

    # Print 10 example conversions (diverse sample)
    print(f"\n{'='*60}")
    print("10 Example Conversions:")
    print(f"{'='*60}")
    if len(results) >= 10:
        step = len(results) // 10
        sample = [results[i * step] for i in range(10)]
    else:
        sample = results[:10]
    for old, new in sample:
        print(f"  {old:45s} -> {new}")

    # Print ALL conversions
    print(f"\n{'='*60}")
    print(f"All {len(results)} Conversions:")
    print(f"{'='*60}")
    for old, new in results:
        print(f"  {old:45s} -> {new}")

    # Validate: check for any names > 64 chars
    long_names = [(old, new) for old, new in results if len(new) > 64]
    if long_names:
        print(f"\nWARNING: {len(long_names)} names exceed 64 chars:")
        for old, new in long_names:
            print(f"  {new} ({len(new)} chars)")


if __name__ == "__main__":
    main()
