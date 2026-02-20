#!/usr/bin/env python3
"""
Generate a super-detailed PDF overview of Jotty using Jotty's own agents.

Uses:
  - Jotty DIRECT tier to write narrative sections (LLM analysis)
  - Codebase scanning for real stats
  - WeasyPrint PDF generation (same engine as olympiad_learning_swarm)
  - Telegram delivery via messaging_tools
"""
import asyncio
import glob
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

# Setup
JOTTY_ROOT = Path("/var/www/sites/personal/stock_market/Jotty")
sys.path.insert(0, str(JOTTY_ROOT))

from dotenv import load_dotenv
load_dotenv(JOTTY_ROOT / ".env")


# ─────────────────────────────────────────────────────
#  PHASE 1: Codebase Analysis (no LLM needed)
# ─────────────────────────────────────────────────────

def count_files(pattern: str, root: str = str(JOTTY_ROOT)) -> int:
    return len(glob.glob(os.path.join(root, pattern), recursive=True))


def count_lines(pattern: str, root: str = str(JOTTY_ROOT)) -> int:
    total = 0
    for f in glob.glob(os.path.join(root, pattern), recursive=True):
        try:
            with open(f, "r", errors="ignore") as fh:
                total += sum(1 for _ in fh)
        except Exception:
            pass
    return total


def list_dirs(path: str) -> list:
    p = Path(path)
    if not p.exists():
        return []
    return sorted([d.name for d in p.iterdir() if d.is_dir() and not d.name.startswith("_") and not d.name.startswith(".")])


def list_all_dirs(path: str) -> list:
    p = Path(path)
    if not p.exists():
        return []
    return sorted([d.name for d in p.iterdir() if d.is_dir() and not d.name.startswith(".")])


def scan_classes_in_file(filepath: str) -> list:
    classes = []
    try:
        with open(filepath, "r", errors="ignore") as f:
            for line in f:
                m = re.match(r'^class\s+(\w+)', line)
                if m:
                    classes.append(m.group(1))
    except Exception:
        pass
    return classes


def scan_functions_in_file(filepath: str) -> list:
    funcs = []
    try:
        with open(filepath, "r", errors="ignore") as f:
            for line in f:
                m = re.match(r'^(?:async\s+)?def\s+(\w+)', line)
                if m and not m.group(1).startswith("_"):
                    funcs.append(m.group(1))
    except Exception:
        pass
    return funcs


def gather_codebase_stats() -> dict:
    """Gather comprehensive stats about the Jotty codebase."""
    print("  Scanning codebase...")
    stats = {}

    # File counts
    stats["py_files_total"] = count_files("**/*.py")
    stats["py_files_core"] = count_files("core/**/*.py")
    stats["py_files_skills"] = count_files("skills/**/*.py")
    stats["py_files_tests"] = count_files("tests/**/*.py")
    stats["py_files_apps"] = count_files("apps/**/*.py")

    # Line counts
    stats["lines_total"] = count_lines("**/*.py")
    stats["lines_core"] = count_lines("core/**/*.py")
    stats["lines_skills"] = count_lines("skills/**/*.py")
    stats["lines_tests"] = count_lines("tests/**/*.py")

    # Skills
    skills_dir = JOTTY_ROOT / "skills"
    all_skill_dirs = list_all_dirs(str(skills_dir))
    stats["skill_dirs"] = [s for s in all_skill_dirs if not s.startswith("_") and s != "__pycache__"]
    stats["skill_count"] = len(stats["skill_dirs"])
    stats["skill_infra"] = [s for s in all_skill_dirs if s.startswith("_")]

    # Swarms
    swarms_dir = JOTTY_ROOT / "core/intelligence/orchestration/swarms"
    stats["swarm_dirs"] = list_dirs(str(swarms_dir))
    stats["swarm_count"] = len([s for s in stats["swarm_dirs"] if s != "base" and s != "templates"])

    # Agents
    agents_dir = JOTTY_ROOT / "core/intelligence/reasoning/agents"
    agent_files = sorted(glob.glob(str(agents_dir / "*.py")))
    stats["agent_files"] = [Path(f).stem for f in agent_files if Path(f).stem != "__init__"]
    stats["agent_classes"] = []
    for f in agent_files:
        stats["agent_classes"].extend(scan_classes_in_file(f))
    stats["agent_count"] = len(stats["agent_classes"])

    # Test counts
    test_files = glob.glob(str(JOTTY_ROOT / "tests/**/*.py"), recursive=True)
    test_func_count = 0
    for tf in test_files:
        try:
            with open(tf, "r", errors="ignore") as fh:
                for line in fh:
                    if re.match(r'\s*(async\s+)?def\s+test_', line):
                        test_func_count += 1
        except Exception:
            pass
    stats["test_files"] = len(test_files)
    stats["test_functions"] = test_func_count

    # Docs
    doc_files = glob.glob(str(JOTTY_ROOT / "docs/**/*.md"), recursive=True)
    stats["doc_files"] = len(doc_files)
    stats["doc_lines"] = count_lines("docs/**/*.md")

    # Facades
    facade_files = glob.glob(str(JOTTY_ROOT / "core/**/facade.py"), recursive=True)
    stats["facade_files"] = [str(Path(f).relative_to(JOTTY_ROOT)) for f in facade_files]
    stats["facade_count"] = len(facade_files)

    # Dataclasses
    dc_count = 0
    for f in glob.glob(str(JOTTY_ROOT / "core/**/*.py"), recursive=True):
        try:
            with open(f, "r", errors="ignore") as fh:
                content = fh.read()
                dc_count += content.count("@dataclass")
        except Exception:
            pass
    stats["dataclass_count"] = dc_count

    # Config files
    config_dir = JOTTY_ROOT / "core/infrastructure/foundation/configs"
    stats["config_files"] = list(glob.glob(str(config_dir / "*.py")))
    stats["config_count"] = len(stats["config_files"])

    # Memory levels
    memory_dir = JOTTY_ROOT / "core/intelligence/memory"
    stats["memory_files"] = list_dirs(str(memory_dir))

    # Learning subsystem
    learning_dir = JOTTY_ROOT / "core/intelligence/learning"
    stats["learning_files"] = [Path(f).stem for f in glob.glob(str(learning_dir / "*.py")) if Path(f).stem != "__init__"]

    # Execution tiers
    exec_dir = JOTTY_ROOT / "core/intelligence/orchestration/execution"
    stats["execution_files"] = [Path(f).stem for f in glob.glob(str(exec_dir / "*.py")) if Path(f).stem != "__init__"]

    # Interface / Modalities
    modalities_dir = JOTTY_ROOT / "core/interface/modalities"
    stats["modality_files"] = [Path(f).stem for f in glob.glob(str(modalities_dir / "*.py")) if Path(f).stem != "__init__"]

    # Apps
    apps_dir = JOTTY_ROOT / "apps"
    stats["app_dirs"] = list_dirs(str(apps_dir))

    print(f"  Done. {stats['py_files_total']} Python files, {stats['lines_total']:,} lines")
    return stats


def gather_swarm_details() -> dict:
    """Get detailed info about each swarm."""
    swarms_root = JOTTY_ROOT / "core/intelligence/orchestration/swarms"
    details = {}

    for swarm_dir in sorted(swarms_root.iterdir()):
        if not swarm_dir.is_dir() or swarm_dir.name.startswith("_") or swarm_dir.name.startswith("."):
            continue
        name = swarm_dir.name
        info = {"name": name, "files": [], "classes": [], "lines": 0, "agents": []}

        for py_file in sorted(swarm_dir.rglob("*.py")):
            rel = str(py_file.relative_to(swarm_dir))
            info["files"].append(rel)
            info["classes"].extend(scan_classes_in_file(str(py_file)))
            try:
                with open(py_file, "r", errors="ignore") as f:
                    info["lines"] += sum(1 for _ in f)
            except Exception:
                pass

            # Look for AGENT_TEAM or agent definitions
            try:
                with open(py_file, "r", errors="ignore") as f:
                    content = f.read()
                    # Pattern: "role": "..." or AgentConfig(name=...)
                    for m in re.finditer(r'"role"\s*:\s*"([^"]+)"', content):
                        if m.group(1) not in info["agents"]:
                            info["agents"].append(m.group(1))
                    for m in re.finditer(r'AGENT_TEAM\s*=\s*\[([^\]]+)\]', content, re.DOTALL):
                        for am in re.finditer(r'"(\w+)"', m.group(1)):
                            if am.group(1) not in info["agents"]:
                                info["agents"].append(am.group(1))
            except Exception:
                pass

        details[name] = info

    return details


def gather_skill_categories() -> dict:
    """Categorize skills into groups."""
    skills_dir = JOTTY_ROOT / "skills"
    categories = {
        "AI & LLM": [],
        "Data & Analytics": [],
        "Document Generation": [],
        "Web & Browser": [],
        "Messaging & Notifications": [],
        "Research & Knowledge": [],
        "Development Tools": [],
        "Financial & Trading": [],
        "Media & Design": [],
        "Productivity": [],
        "Infrastructure": [],
        "Other": [],
    }

    ai_keywords = ["claude", "gemini", "openai", "llm", "automl", "pycaret", "auto-sklearn", "ensemble", "clustering", "dimensionality", "shap", "feature-engineer", "hyperopt"]
    data_keywords = ["data-", "csv", "sql", "database", "pivot", "statistical", "semantic-layer", "profiler", "schema", "validator", "transformer", "text-chunk"]
    doc_keywords = ["document", "pdf", "epub", "docx", "pptx", "latex", "presenton", "slide", "simple-pdf"]
    web_keywords = ["browser", "web-", "http", "webhook", "oauth", "rss"]
    msg_keywords = ["telegram", "whatsapp", "slack", "discord", "email", "messaging", "notion"]
    research_keywords = ["research", "arxiv", "summarize", "knowledge", "search-summarize"]
    dev_keywords = ["code-", "dev-", "diff", "dependency", "changelog", "skill-", "mcp-", "shell", "terminal", "git", "semver", "test-skill", "webapp-testing", "process-manager"]
    finance_keywords = ["pmi-", "stock", "investing", "financial", "crypto", "expense", "tax", "screener"]
    media_keywords = ["image-", "video", "gif", "youtube", "spotify", "canvas", "visual", "chart", "algorithmic-art", "media-production"]
    prod_keywords = ["cron", "time-", "timezone", "habit", "qr-code", "random-data", "weather", "date-calc", "raffle", "invoice"]

    all_skills = list_dirs(str(skills_dir))
    for skill in all_skills:
        s = skill.lower()
        placed = False
        for keywords, cat in [
            (ai_keywords, "AI & LLM"), (finance_keywords, "Financial & Trading"),
            (doc_keywords, "Document Generation"), (msg_keywords, "Messaging & Notifications"),
            (web_keywords, "Web & Browser"), (research_keywords, "Research & Knowledge"),
            (data_keywords, "Data & Analytics"), (dev_keywords, "Development Tools"),
            (media_keywords, "Media & Design"), (prod_keywords, "Productivity"),
        ]:
            for kw in keywords:
                if kw in s:
                    categories[cat].append(skill)
                    placed = True
                    break
            if placed:
                break
        if not placed:
            categories["Other"].append(skill)

    # Remove empty
    return {k: sorted(v) for k, v in categories.items() if v}


# ─────────────────────────────────────────────────────
#  PHASE 2: LLM-Enhanced Narrative Sections
# ─────────────────────────────────────────────────────

async def generate_narrative_section(jotty, section_name: str, context: str) -> str:
    """Use Jotty DIRECT tier to write a narrative section."""
    from Jotty.core.intelligence.orchestration.execution import ExecutionTier

    prompt = f"""You are writing a section for a comprehensive technical overview document about Jotty,
a 280K-line AI agent framework. Write the "{section_name}" section.

Context data:
{context}

Requirements:
- Write in professional technical documentation style
- Be detailed but concise — every sentence should add value
- Use specific numbers and facts from the context
- Do NOT use markdown headers (they'll be added by the template)
- Write 3-8 paragraphs depending on section complexity
- Highlight what makes Jotty unique or impressive
- Be honest about limitations where relevant"""

    try:
        result = await jotty.run(prompt, tier=ExecutionTier.DIRECT)
        if result.success and result.output:
            return result.output.strip()
    except Exception as e:
        print(f"  LLM section '{section_name}' failed: {e}")

    return f"[Section generation failed for: {section_name}]"


# ─────────────────────────────────────────────────────
#  PHASE 3: Assemble Markdown Document
# ─────────────────────────────────────────────────────

def build_markdown(stats: dict, swarm_details: dict, skill_categories: dict, narratives: dict) -> str:
    """Assemble the complete markdown document."""
    now = datetime.now().strftime("%B %d, %Y")

    # Skill list by category
    skill_table = ""
    for cat, skills in sorted(skill_categories.items()):
        skill_table += f"\n**{cat}** ({len(skills)} skills)\n\n"
        for s in skills:
            skill_table += f"- `{s}`\n"

    # Swarm details table
    swarm_section = ""
    for name, info in sorted(swarm_details.items()):
        if name in ("base", "templates"):
            swarm_section += f"\n### {name.replace('_', ' ').title()}\n\n"
            swarm_section += f"Infrastructure module with {info['lines']:,} lines across {len(info['files'])} files. "
            swarm_section += f"Classes: {', '.join(info['classes'][:10])}{'...' if len(info['classes']) > 10 else ''}\n"
            continue
        swarm_section += f"\n### {name.replace('_', ' ').title()}\n\n"
        swarm_section += f"- **Lines of code:** {info['lines']:,}\n"
        swarm_section += f"- **Files:** {len(info['files'])}\n"
        swarm_section += f"- **Classes:** {', '.join(info['classes'][:15])}\n"
        if info["agents"]:
            swarm_section += f"- **Agent roles:** {', '.join(info['agents'][:12])}\n"

    # Agent classes
    agent_section = ""
    for cls in sorted(set(stats["agent_classes"])):
        agent_section += f"- `{cls}`\n"

    # Facades
    facade_section = ""
    for f in sorted(stats["facade_files"]):
        facade_section += f"- `{f}`\n"

    md = f"""# Jotty AI Agent Framework — Complete Technical Overview

**Generated:** {now}
**Version:** Production (main branch)
**Generated by:** Jotty's own DIRECT-tier agents + codebase analysis

---

## Executive Summary

{narratives.get('executive_summary', '')}

---

## 1. Codebase at a Glance

| Metric | Count |
|--------|-------|
| **Total Python files** | {stats['py_files_total']:,} |
| **Total lines of Python** | {stats['lines_total']:,} |
| **Core framework files** | {stats['py_files_core']:,} |
| **Core framework lines** | {stats['lines_core']:,} |
| **Skill package files** | {stats['py_files_skills']:,} |
| **Skill package lines** | {stats['lines_skills']:,} |
| **Test files** | {stats['test_files']:,} |
| **Test functions** | {stats['test_functions']:,} |
| **Test lines** | {stats['lines_tests']:,} |
| **Documentation files** | {stats['doc_files']} |
| **Typed dataclasses** | {stats['dataclass_count']} |
| **Subsystem facades** | {stats['facade_count']} |
| **Configuration modules** | {stats['config_count']} |

---

## 2. Architecture

{narratives.get('architecture', '')}

### Clean Architecture Layers

```
┌──────────────────────────────────────────────────────────┐
│  LAYER 5: APPLICATIONS (apps/)                           │
│  {', '.join(stats['app_dirs'])}                          │
│  Apps use SDK ONLY — never import from core directly     │
├──────────────────────────────────────────────────────────┤
│  LAYER 4: SDK (sdk/)                                     │
│  from jotty import Jotty  — stable public API            │
├──────────────────────────────────────────────────────────┤
│  LAYER 3: CORE API (core/interface/)                     │
│  JottyAPI, ChatAPI, WorkflowAPI, Modalities              │
├──────────────────────────────────────────────────────────┤
│  LAYER 2: CORE FRAMEWORK (core/)                         │
│  Intelligence, Capabilities, Infrastructure              │
└──────────────────────────────────────────────────────────┘
```

### Subsystem Facades

{facade_section}

Each facade provides a zero-configuration entry point with thread-safe singleton management using double-checked locking.

---

## 3. Execution Engine

{narratives.get('execution_engine', '')}

### Execution Tiers

| Tier | Name | Description |
|------|------|-------------|
| 1 | **DIRECT** | Single LLM call — fastest, cheapest |
| 2 | **AGENTIC** | Planning + multi-step tool use |
| 3 | **WORKFLOW** | Predefined multi-stage pipelines |
| 4 | **RESEARCH** | Full domain swarm with multi-agent coordination |

### Execution Components

{chr(10).join(f'- `{f}`' for f in stats['execution_files'])}

---

## 4. Multi-Agent Swarms ({stats['swarm_count']} Active Swarms)

{narratives.get('swarms', '')}

{swarm_section}

---

## 5. Agent Types ({stats['agent_count']} Agent Classes)

{narratives.get('agents', '')}

### Complete Agent Registry

{agent_section}

---

## 6. Skill Ecosystem ({stats['skill_count']} Skills)

{narratives.get('skills', '')}

### Skills by Category

{skill_table}

---

## 7. Memory System

{narratives.get('memory', '')}

### Memory Subsystem Components

{chr(10).join(f'- `{f}`' for f in stats['memory_files'])}

The memory system provides 5 levels of storage: episodic (event-based), semantic (factual knowledge), procedural (how-to), meta (self-reflective), and causal (cause-effect chains).

---

## 8. Learning & Reinforcement Learning

{narratives.get('learning', '')}

### Learning Components

{chr(10).join(f'- `{f}`' for f in stats['learning_files'])}

---

## 9. Context Management

{narratives.get('context', '')}

---

## 10. Interface & Modalities

{narratives.get('interface', '')}

### Supported Modalities

{chr(10).join(f'- `{f}`' for f in stats['modality_files'])}

### Application Entry Points

| App | Purpose |
|-----|---------|
{chr(10).join(f'| `{a}` | Application |' for a in stats['app_dirs'])}

---

## 11. Testing & Quality

{narratives.get('testing', '')}

| Metric | Value |
|--------|-------|
| Test files | {stats['test_files']:,} |
| Test functions | {stats['test_functions']:,} |
| Test lines | {stats['lines_tests']:,} |
| Test-to-code ratio | {stats['lines_tests'] / max(stats['lines_core'], 1):.2f} |

---

## 12. Infrastructure & Production Readiness

{narratives.get('infrastructure', '')}

### Key Infrastructure Components

- **Budget Tracking:** Per-agent, per-swarm, per-session cost tracking with 4 scope levels
- **Circuit Breakers:** Fault tolerance with configurable thresholds and recovery
- **LLM Call Cache:** Response caching with TTL and cache statistics
- **Distributed Tracing:** OpenTelemetry-compatible spans across agent chains
- **Safety Gates:** ValidatorAgent and SafetyConstraint enforcement
- **Import Boundary Linter:** 25+ module rules enforced in CI

---

## 13. Unique Differentiators

{narratives.get('differentiators', '')}

---

## 14. Configuration System

The framework uses {stats['dataclass_count']} typed dataclasses with `__post_init__()` validation. Configuration is spread across {stats['config_count']} focused config modules in `core/infrastructure/foundation/configs/`.

Key config classes:
- **SwarmConfig** — 100+ fields for swarm behavior tuning
- **AgentConfig** — Agent runtime parameters
- **ExecutionConfig** — Tier selection and execution parameters
- **CompressionConfig** — Context compression strategies
- **ChunkingConfig** — Content chunking parameters

---

## Appendix A: File Distribution

| Directory | Python Files | Lines |
|-----------|-------------|-------|
| `core/` | {stats['py_files_core']:,} | {stats['lines_core']:,} |
| `skills/` | {stats['py_files_skills']:,} | {stats['lines_skills']:,} |
| `tests/` | {stats['py_files_tests']:,} | {stats['lines_tests']:,} |
| `apps/` | {stats['py_files_apps']:,} | — |
| **Total** | **{stats['py_files_total']:,}** | **{stats['lines_total']:,}** |

---

*This document was generated by Jotty analyzing its own codebase — using its DIRECT execution tier for narrative generation, codebase scanning for statistics, and WeasyPrint for PDF rendering.*
"""
    return md


# ─────────────────────────────────────────────────────
#  PHASE 4: PDF Generation (WeasyPrint)
# ─────────────────────────────────────────────────────

def markdown_to_pdf(markdown_content: str, output_path: str) -> str:
    """Convert markdown to a professional PDF using WeasyPrint."""
    import markdown
    from weasyprint import HTML

    # Convert markdown to HTML
    html_body = markdown.markdown(
        markdown_content,
        extensions=["tables", "fenced_code", "codehilite", "toc", "nl2br"],
    )

    # Professional CSS
    css = """
    @page {
        size: A4;
        margin: 2cm 2.2cm;
        @bottom-center {
            content: "Jotty AI Framework — Page " counter(page) " of " counter(pages);
            font-size: 9px;
            color: #666;
            font-family: 'Segoe UI', Arial, sans-serif;
        }
        @top-center {
            content: "CONFIDENTIAL — Technical Overview";
            font-size: 8px;
            color: #999;
            font-family: 'Segoe UI', Arial, sans-serif;
        }
    }
    body {
        font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
        font-size: 11pt;
        line-height: 1.6;
        color: #1a1a1a;
        max-width: 100%;
    }
    h1 {
        font-size: 26pt;
        color: #1a365d;
        border-bottom: 3px solid #2b6cb0;
        padding-bottom: 12px;
        margin-top: 0;
        page-break-before: avoid;
    }
    h2 {
        font-size: 18pt;
        color: #2b6cb0;
        border-bottom: 1.5px solid #bee3f8;
        padding-bottom: 6px;
        margin-top: 30px;
        page-break-after: avoid;
    }
    h3 {
        font-size: 14pt;
        color: #2c5282;
        margin-top: 20px;
        page-break-after: avoid;
    }
    p {
        margin: 8px 0;
        text-align: justify;
    }
    table {
        border-collapse: collapse;
        width: 100%;
        margin: 15px 0;
        font-size: 10pt;
        page-break-inside: avoid;
    }
    th {
        background-color: #2b6cb0;
        color: white;
        padding: 8px 12px;
        text-align: left;
        font-weight: 600;
    }
    td {
        padding: 6px 12px;
        border-bottom: 1px solid #e2e8f0;
    }
    tr:nth-child(even) td {
        background-color: #f7fafc;
    }
    code {
        background-color: #edf2f7;
        padding: 2px 6px;
        border-radius: 3px;
        font-size: 9.5pt;
        font-family: 'Consolas', 'Monaco', monospace;
        color: #c53030;
    }
    pre {
        background-color: #1a202c;
        color: #e2e8f0;
        padding: 16px;
        border-radius: 6px;
        font-size: 9pt;
        line-height: 1.5;
        overflow-x: auto;
        page-break-inside: avoid;
        margin: 12px 0;
    }
    pre code {
        background: none;
        color: #e2e8f0;
        padding: 0;
    }
    ul, ol {
        padding-left: 24px;
        margin: 8px 0;
    }
    li {
        margin: 3px 0;
    }
    strong {
        color: #1a365d;
    }
    hr {
        border: none;
        border-top: 1px solid #cbd5e0;
        margin: 25px 0;
    }
    blockquote {
        border-left: 4px solid #2b6cb0;
        margin: 12px 0;
        padding: 8px 16px;
        background-color: #ebf8ff;
        font-style: italic;
    }
    """

    full_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>{css}</style>
</head>
<body>
{html_body}
</body>
</html>"""

    HTML(string=full_html).write_pdf(output_path)
    return output_path


# ─────────────────────────────────────────────────────
#  PHASE 5: Telegram Delivery
# ─────────────────────────────────────────────────────

async def send_telegram(file_path: str, caption: str):
    """Send the PDF to Telegram using Jotty's messaging tools."""
    try:
        import importlib
        spec = importlib.util.spec_from_file_location(
            "messaging_tools",
            str(JOTTY_ROOT / "skills" / "messaging_tools" / "manager.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        manager = mod.OutputChannelManager()
        result = await manager.send_to_telegram(
            file_path=file_path,
            caption=caption,
        )
        return result
    except Exception as e:
        print(f"  Telegram send failed: {e}")
        # Fallback: use telegram-sender skill directly
        try:
            from skills.telegram_sender.tools import send_document_tool
            result = await send_document_tool({
                "file_path": file_path,
                "caption": caption,
            })
            return result
        except Exception as e2:
            print(f"  Telegram fallback also failed: {e2}")
            # Last resort: raw API call
            try:
                import aiohttp
                token = os.environ.get("TELEGRAM_TOKEN") or os.environ.get("TELEGRAM_BOT_TOKEN")
                chat_id = os.environ.get("TELEGRAM_CHAT_ID")
                if token and chat_id:
                    url = f"https://api.telegram.org/bot{token}/sendDocument"
                    async with aiohttp.ClientSession() as session:
                        with open(file_path, "rb") as f:
                            data = aiohttp.FormData()
                            data.add_field("chat_id", chat_id)
                            data.add_field("caption", caption[:1024])
                            data.add_field("document", f, filename=os.path.basename(file_path))
                            async with session.post(url, data=data) as resp:
                                r = await resp.json()
                                print(f"  Telegram raw API: {'OK' if r.get('ok') else r}")
                                return r
                else:
                    print("  No TELEGRAM_TOKEN/TELEGRAM_CHAT_ID in env")
            except Exception as e3:
                print(f"  Telegram raw API failed: {e3}")
    return None


# ─────────────────────────────────────────────────────
#  MAIN ORCHESTRATOR
# ─────────────────────────────────────────────────────

async def main():
    start = time.time()
    print("=" * 70)
    print("  JOTTY SELF-ANALYSIS: Generating Comprehensive PDF Overview")
    print("=" * 70)

    # Phase 1: Gather stats
    print("\n[Phase 1/5] Scanning codebase...")
    stats = gather_codebase_stats()
    swarm_details = gather_swarm_details()
    skill_categories = gather_skill_categories()

    print(f"  Skills: {stats['skill_count']} | Swarms: {stats['swarm_count']} | Agents: {stats['agent_count']}")
    print(f"  Tests: {stats['test_functions']} functions in {stats['test_files']} files")

    # Phase 2: Generate narrative sections with Jotty
    print("\n[Phase 2/5] Using Jotty DIRECT tier for narrative generation...")
    from Jotty import Jotty
    jotty = Jotty(log_level="WARNING")

    # Context summaries for each section
    sections_to_generate = {
        "executive_summary": f"""Jotty is a {stats['lines_total']:,}-line Python AI agent framework with {stats['skill_count']} skills,
{stats['swarm_count']} domain swarms, {stats['agent_count']} agent classes, and {stats['test_functions']} tests.
It features 5-layer clean architecture, 5-level brain-inspired memory, TD-Lambda reinforcement learning,
Thompson sampling for strategy selection, multi-agent coordination with 8 paradigms
(pipeline, parallel, consensus, debate, iterative, hierarchical, blackboard, custom),
and supports text+voice modalities across CLI, Web, Telegram, WhatsApp, and API platforms.
Write a compelling 3-paragraph executive summary.""",

        "architecture": f"""5-layer clean architecture following Google/Amazon/Stripe patterns:
Layer 5: Apps ({', '.join(stats['app_dirs'])}) — use SDK only
Layer 4: SDK — stable public API (from jotty import Jotty)
Layer 3: Core API — JottyAPI, ChatAPI, WorkflowAPI, modalities ({', '.join(stats['modality_files'])})
Layer 2: Core Framework — {stats['py_files_core']} files, {stats['lines_core']:,} lines
- intelligence/: memory (5 levels), learning (TD-lambda, Q-learning), orchestration (swarms, execution), reasoning (agents)
- capabilities/: {stats['skill_count']} skills with unified registry
- infrastructure/: {stats['dataclass_count']} dataclasses, {stats['facade_count']} facades, context management
Import boundaries enforced by 607-line AST linter with 25+ module rules in CI.
{stats['facade_count']} thread-safe facades with double-checked locking singletons.""",

        "execution_engine": f"""4 execution tiers: DIRECT (single LLM call), AGENTIC (planning + tools),
WORKFLOW (predefined pipelines), RESEARCH (full swarm).
TierExecutor routes tasks based on complexity classification.
Components: {', '.join(stats['execution_files'])}.
Intent classification determines which tier handles each request.
Thompson sampling selects optimal strategy/paradigm based on learned episode history.
Budget tracking across all tiers with per-agent cost attribution.""",

        "swarms": f"""Multi-agent swarms with true asyncio.gather parallelism.
{stats['swarm_count']} active swarms: {', '.join(stats['swarm_dirs'])}.
Inheritance: Concrete swarm -> SwarmTemplate -> SwarmLearning -> SwarmLearningMixin.
PhaseExecutor wraps asyncio.gather for parallel agent execution.
TeamCoordinator supports 8 coordination patterns.
CodingSwarm: 1634 lines, 8 agents across 10 phases (design, implement, test, optimize, document).
ResearchSwarm: 1604 lines, 10 agents with parallel data collection + analysis.
OlympiadLearningSwarm: 12 agents for educational content from foundation to olympiad level.
Shared state: agents share memory, context manager, and message bus within swarms.""",

        "agents": f"""{stats['agent_count']} agent classes across {len(stats['agent_files'])} files.
Base: BaseAgent with _UNSET sentinel pattern for lazy property initialization.
Key types: AutoAgent (general-purpose), ChatAssistant (conversational), DomainAgent (specialized),
ExpertAgent (deep expertise), SwarmAgent (coordination), PipelineAgent (workflows),
MetaAgent (self-reflection), CompositeAgent (multi-capability).
Specialized: LaTeX, Mermaid, PlantUML agents for diagram generation.
Domain experts: Backend, Frontend, Designer, QA, Product Manager, UX Researcher.
All agents support memory injection, skill registry access, and budget tracking.""",

        "skills": f"""{stats['skill_count']} skill packages organized by domain:
{json.dumps({k: len(v) for k, v in skill_categories.items()}, indent=2)}
Categories: AI/LLM ({len(skill_categories.get('AI & LLM', []))}),
Data ({len(skill_categories.get('Data & Analytics', []))}),
Documents ({len(skill_categories.get('Document Generation', []))}),
Web ({len(skill_categories.get('Web & Browser', []))}),
Messaging ({len(skill_categories.get('Messaging & Notifications', []))}),
Research ({len(skill_categories.get('Research & Knowledge', []))}),
Dev Tools ({len(skill_categories.get('Development Tools', []))}),
Finance ({len(skill_categories.get('Financial & Trading', []))}),
Media ({len(skill_categories.get('Media & Design', []))}),
Productivity ({len(skill_categories.get('Productivity', []))}),
Skills are lazily loaded via plugin discovery and managed by UnifiedRegistry.""",

        "memory": f"""5-level brain-inspired memory system:
1. Episodic — event/experience storage
2. Semantic — factual knowledge
3. Procedural — how-to/skills
4. Meta — self-reflective learnings
5. Causal — cause-effect chains
Components: {', '.join(stats['memory_files'])}
Features: LLM-powered RAG retrieval, relevance scoring, memory consolidation,
BrainInspiredMemoryManager with automatic level classification.
Memory facade provides zero-config entry: get_memory_system(), get_brain_manager(), get_rag_retriever().""",

        "learning": f"""Reinforcement learning with TD-Lambda (gamma=0.99, lambda=0.95) and Q-learning.
Components: {', '.join(stats['learning_files'])}.
Thompson sampling for strategy/paradigm selection with exploration-exploitation balance.
ReasoningCreditAssigner for multi-agent credit attribution.
Episode-based learning: each task execution creates an episode with state, action, reward, next_state.
Reward signals from quality scores, check pass rates, execution time.
Transfer learning across domains via shared episode memory.""",

        "context": f"""Smart context management with priority-based token budgeting.
ContextPriority: CRITICAL (never compressed) > HIGH > MEDIUM > LOW.
SmartContextManager handles: build_context, compress_parts, overflow detection.
Compression strategies: simple_truncate, prefix_suffix_compress, structured_extract, intelligent_compress (LLM-based with Shapley credits).
Chunking: configurable max_chunk_tokens, overlap, sentence preservation.
GlobalContextGuard for hard overflow detection. ContentGate for relevance filtering.
Canonical home: core/infrastructure/context/ with models, utils, context_manager, facade.""",

        "interface": f"""Multi-platform, multi-modal interface layer.
Platforms: {', '.join(stats['app_dirs'])}
Modalities: {', '.join(stats['modality_files'])}
Text: parse_input, format_output with platform-specific formatting.
Voice: SpeechToText (Whisper), TextToSpeech (OpenAI), AudioProcessor.
APIs: JottyAPI (unified), ChatAPI, WorkflowAPI.
Gateway: UnifiedGateway with ChannelRouter for webhook-based integrations.
A2UI: Agent-to-UI response formatting for rich output rendering.""",

        "testing": f"""{stats['test_functions']} test functions across {stats['test_files']} files ({stats['lines_tests']:,} lines).
5 test categories: unit/, integration/, architecture/, domain/, e2e/.
Professional fixtures: 623-line conftest.py with pre-wired mocks.
All tests use mocks — never call real LLM providers.
Architecture tests enforce import boundaries and structural rules.
Test-to-code ratio: {stats['lines_tests'] / max(stats['lines_core'], 1):.2f}.""",

        "infrastructure": f"""Production-ready infrastructure:
- Budget tracking: per-agent, per-swarm, per-session, per-model cost attribution (4 scopes)
- Circuit breakers: configurable thresholds with automatic recovery
- LLM call cache: response caching with TTL and hit/miss statistics
- Distributed tracing: OpenTelemetry-compatible spans
- Safety gates: ValidatorAgent + SafetyConstraint enforcement
- Import boundary linter: 607-line AST scanner with 25+ rules in CI
- Thread safety: all facades use double-checked locking singletons
- Exception hierarchy: 16-class hierarchy rooted at JottyError
- Config validation: {stats['dataclass_count']} dataclasses with __post_init__()
- Type coverage: py.typed marker (PEP 561), targeting 100% coverage""",

        "differentiators": f"""What makes Jotty unique:
1. Self-improving: TD-Lambda RL + Thompson sampling lets the system learn from every execution
2. True multi-agent: asyncio.gather parallelism with 8 coordination patterns, not just sequential chaining
3. 5-level memory: brain-inspired episodic/semantic/procedural/meta/causal memory
4. {stats['skill_count']} skills: massive capability surface from browser automation to financial analysis
5. Clean architecture: 5-layer separation enforced by CI linter (not just convention)
6. Multi-modal: text + voice across 5 platforms (CLI, Web, Telegram, WhatsApp, API)
7. Production-grade: budget tracking, circuit breakers, distributed tracing, safety gates
8. Self-documenting: can analyze its own codebase and generate this PDF
9. {stats['lines_total']:,} lines of Python with {stats['test_functions']} tests and {stats['dataclass_count']} typed dataclasses"""
    }

    narratives = {}
    # Generate sections in batches to avoid overwhelming the API
    section_names = list(sections_to_generate.keys())
    for i in range(0, len(section_names), 3):
        batch = section_names[i:i+3]
        print(f"  Generating: {', '.join(batch)}...")
        tasks = [
            generate_narrative_section(jotty, name, sections_to_generate[name])
            for name in batch
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for name, result in zip(batch, results):
            if isinstance(result, Exception):
                print(f"    {name}: FAILED ({result})")
                narratives[name] = f"[Generation failed: {result}]"
            else:
                narratives[name] = result
                print(f"    {name}: OK ({len(result)} chars)")

    # Phase 3: Assemble document
    print("\n[Phase 3/5] Assembling document...")
    markdown_content = build_markdown(stats, swarm_details, skill_categories, narratives)

    # Save markdown
    md_path = str(JOTTY_ROOT / "outputs" / "Jotty_Complete_Overview.md")
    os.makedirs(os.path.dirname(md_path), exist_ok=True)
    with open(md_path, "w") as f:
        f.write(markdown_content)
    print(f"  Markdown saved: {md_path} ({len(markdown_content):,} chars)")

    # Phase 4: Generate PDF
    print("\n[Phase 4/5] Generating PDF with WeasyPrint...")
    pdf_path = str(JOTTY_ROOT / "outputs" / "Jotty_Complete_Overview.pdf")
    try:
        markdown_to_pdf(markdown_content, pdf_path)
        pdf_size = os.path.getsize(pdf_path)
        print(f"  PDF saved: {pdf_path} ({pdf_size / 1024:.0f} KB)")
    except Exception as e:
        print(f"  PDF generation failed: {e}")
        print("  Trying fallback with simple-pdf-generator...")
        try:
            import importlib
            spec = importlib.util.spec_from_file_location(
                "simple_pdf",
                str(JOTTY_ROOT / "skills" / "simple-pdf-generator" / "tools.py"),
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            result = await mod.create_pdf_from_markdown_tool({
                "markdown_content": markdown_content,
                "output_path": pdf_path,
                "title": "Jotty AI Framework Overview",
            })
            print(f"  Fallback PDF: {result}")
        except Exception as e2:
            print(f"  Fallback also failed: {e2}")
            pdf_path = None

    # Phase 5: Send to Telegram
    print("\n[Phase 5/5] Sending to Telegram...")
    if pdf_path and os.path.exists(pdf_path):
        telegram_result = await send_telegram(
            pdf_path,
            f"Jotty AI Framework — Complete Technical Overview\n"
            f"{stats['lines_total']:,} lines | {stats['skill_count']} skills | "
            f"{stats['swarm_count']} swarms | {stats['agent_count']} agents | "
            f"{stats['test_functions']} tests\n"
            f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        )
        if telegram_result:
            print("  Telegram: Sent!")
        else:
            print("  Telegram: Could not send (check TELEGRAM_TOKEN)")
    else:
        print("  Skipping Telegram (no PDF)")

    elapsed = time.time() - start
    print(f"\n{'=' * 70}")
    print(f"  DONE in {elapsed:.1f}s")
    print(f"  Markdown: {md_path}")
    if pdf_path:
        print(f"  PDF: {pdf_path}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    asyncio.run(main())
