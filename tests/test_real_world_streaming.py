#!/usr/bin/env python3
"""
JOTTY V2 — REAL-WORLD SWARM MANAGER TEST (STREAMING LOG)
=========================================================

Tests SwarmManager with a genuinely complex multi-step task:

  "Find the top 5 trending AI papers this week, summarize each,
   compare their approaches, and create a PDF report with citations."

This SHOULD exercise:
  - Intent parsing & autonomous setup
  - Skill discovery (arxiv-downloader, web-search, research-to-pdf, etc.)
  - Multi-step planning via AgenticPlanner
  - Tool/skill execution (not just a single LLM call)
  - PDF artifact generation
"""

import asyncio
import os
import sys
import time
from pathlib import Path

# ── Load API keys ──
for env_file in [
    Path(__file__).parents[2] / '.env.anthropic',
    Path(__file__).parents[1] / '.env.anthropic',
    Path(__file__).parents[1] / '.env',
]:
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if '=' in line and not line.startswith('#'):
                    k, v = line.split('=', 1)
                    k, v = k.strip(), v.strip()
                    if v and k not in os.environ:
                        os.environ[k] = v

import logging
logging.basicConfig(
    level=logging.WARNING,
    format='%(levelname)-8s %(name)s: %(message)s',
)

# Colors
B = '\033[1m'
G = '\033[92m'
R = '\033[91m'
Y = '\033[93m'
C = '\033[96m'
D = '\033[2m'
E = '\033[0m'


async def main():
    from Jotty.core.orchestration.v2.swarm_manager import SwarmManager
    from Jotty.core.foundation.config_defaults import DEFAULTS

    print(f'\n{B}{"═" * 70}{E}')
    print(f'{B}  JOTTY V2 — REAL-WORLD SWARM TEST: AI PAPERS REPORT{E}')
    print(f'{D}  Centralized defaults: '
          f'model={DEFAULTS.DEFAULT_MODEL_ALIAS}, '
          f'max_tokens={DEFAULTS.LLM_MAX_OUTPUT_TOKENS}, '
          f'temp={DEFAULTS.LLM_TEMPERATURE}, '
          f'timeout={DEFAULTS.LLM_TIMEOUT_SECONDS}s{E}')
    print(f'{D}  Time: {time.strftime("%Y-%m-%d %H:%M:%S")}{E}')
    print(f'{B}{"═" * 70}{E}')

    sm = SwarmManager(enable_lotus=False, enable_zero_config=False)
    t0 = time.time()
    events = []

    def on_status(stage, detail=''):
        elapsed = time.time() - t0
        events.append((elapsed, stage, detail))
        # Color-code by stage type
        if 'error' in stage.lower() or 'fail' in stage.lower():
            color = R
        elif any(w in stage.lower() for w in ['complete', 'done', 'success', 'result']):
            color = G
        elif any(w in stage.lower() for w in ['agent', 'running', 'batch', 'executing']):
            color = C
        elif any(w in stage.lower() for w in ['plan', 'analyz', 'deciding', 'intent']):
            color = Y
        elif any(w in stage.lower() for w in ['skill', 'tool', 'search', 'arxiv', 'pdf']):
            color = Y
        else:
            color = D
        trunc = (detail[:120] + '...') if len(detail) > 120 else detail
        print(f'  {color}[{elapsed:6.1f}s] {stage}: {trunc}{E}', flush=True)

    goal = (
        'Find the top 5 trending AI papers this week, summarize each, '
        'compare their approaches, and create a PDF report with citations.'
    )

    print(f'\n{B}  GOAL:{E}')
    print(f'  {goal}\n')
    print(f'{B}  ── STREAMING LOG ──{E}')

    result = await sm.run(
        goal,
        ensemble=False,
        status_callback=on_status,
    )

    elapsed = time.time() - t0
    output = str(getattr(result, 'output', result))
    success = getattr(result, 'success', None)

    # ── Results ──
    print(f'\n{B}{"═" * 70}{E}')
    print(f'{B}  RESULTS{E}')
    print(f'{B}{"═" * 70}{E}')
    print(f'  Success:  {G if success else R}{success}{E}')
    print(f'  Time:     {elapsed:.1f}s')
    print(f'  Output:   {len(output):,} chars')

    # Quality checks — what we expect from a real multi-step agent task
    checks = {
        # Content quality
        'has_papers':      any(w in output.lower() for w in [
            'arxiv', 'paper', 'transformer', 'llm', 'diffusion', 'attention',
            'neural', 'language model', 'foundation model',
        ]),
        'has_summaries':   any(w in output.lower() for w in [
            'summary', 'summarize', 'abstract', 'contribution', 'propose',
            'introduce', 'present', 'demonstrate',
        ]),
        'has_comparison':  any(w in output.lower() for w in [
            'compar', 'contrast', 'versus', 'differ', 'similar', 'approach',
            'advantage', 'limitation',
        ]),
        'has_citations':   any(w in output.lower() for w in [
            'citation', 'reference', 'arxiv.org', 'et al', '[1]', '(20',
        ]),
        'structured':      any(w in output for w in ['##', '###', '**', '---', '|']),
        'substantial':     len(output) > 2000,

        # Pipeline checks — did the system actually USE tools?
        'used_skills':     any(
            'skill' in str(getattr(e, '__dict__', e)).lower()
            for e in getattr(result, 'trajectory', [])
        ) or any('skill' in d.lower() for _, s, d in events if d),
        'multi_step':      len([e for e in events if 'execut' in e[1].lower()]) >= 1,
    }
    passed = sum(checks.values())
    total = len(checks)
    quality = passed / total * 100

    print(f'  Quality:  {quality:.0f}% ({passed}/{total} checks)')

    print(f'\n  {B}Quality checks:{E}')
    for k, v in checks.items():
        icon = f'{G}✓{E}' if v else f'{R}✗{E}'
        print(f'    {icon} {k}')

    # Execution path analysis — was this a real agent pipeline or just a direct LLM call?
    fast_path = any('fast path' in d.lower() or 'direct llm' in d.lower() for _, s, d in events)
    used_architect = any('architect' in s.lower() for _, s, _ in events)
    used_auditor = any('auditor' in s.lower() for _, s, _ in events)
    used_tools = any(
        w in d.lower()
        for _, s, d in events
        for w in ['skill', 'tool', 'arxiv', 'search', 'pdf', 'download']
    )

    print(f'\n  {B}Execution path analysis:{E}')
    print(f'    {"🔴" if fast_path else "🟢"} Fast path (direct LLM): {"YES — bypassed pipeline" if fast_path else "NO — used agent pipeline"}')
    print(f'    {"🟢" if used_architect else "🔴"} Architect (planning):  {"YES" if used_architect else "NO — skipped"}')
    print(f'    {"🟢" if used_auditor else "🔴"} Auditor (validation):  {"YES" if used_auditor else "NO — skipped"}')
    print(f'    {"🟢" if used_tools else "🔴"} Tool/skill use:        {"YES" if used_tools else "NO — no tools invoked"}')

    # Timeline
    print(f'\n  {B}Execution timeline ({len(events)} events):{E}')
    for t, stage, detail in events:
        print(f'    {D}[{t:6.1f}s]{E} {stage}: {detail[:100]}')

    # Check for PDF artifact
    pdf_files = list(Path('.').glob('**/*.pdf'))
    recent_pdfs = [p for p in pdf_files if p.stat().st_mtime > t0]
    if recent_pdfs:
        print(f'\n  {G}{B}PDF artifacts generated:{E}')
        for p in recent_pdfs:
            sz = p.stat().st_size / 1024
            print(f'    {G}✓ {p} ({sz:.1f} KB){E}')
    else:
        print(f'\n  {R}No PDF artifacts generated{E}')

    # Output preview
    print(f'\n{D}  Output preview (first 1200 chars):{E}')
    preview = output[:1200].replace('\n', '\n  ')
    print(f'  {preview}')
    if len(output) > 1200:
        print(f'  {D}... ({len(output) - 1200:,} more chars){E}')

    # Verdict
    print(f'\n{B}{"═" * 70}{E}')
    if quality >= 50:
        print(f'  {G}PASSED — quality {quality:.0f}%{E}')
    else:
        print(f'  {R}FAILED — quality {quality:.0f}% below 50% threshold{E}')
    if fast_path:
        print(f'  {Y}WARNING: Hit fast path — agent pipeline was NOT exercised{E}')
    print(f'{B}{"═" * 70}{E}\n')

    # Don't fail on quality alone — the diagnostic info is the real value
    if quality < 25:
        sys.exit(1)


if __name__ == "__main__":
    # Suppress noisy LiteLLM shutdown errors (background worker CancelledError)
    logging.getLogger("LiteLLM").setLevel(logging.CRITICAL)
    logging.getLogger("litellm").setLevel(logging.CRITICAL)
    asyncio.run(main())
