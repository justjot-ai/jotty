#!/usr/bin/env python3
"""
Comprehensive Guide Generator with Parallel Section Writing
============================================================

Uses Jotty's parallel execution pattern (asyncio.gather):
1. Planner Agent: Determines what sections to generate
2. Research Agent: Uses DuckDuckGo to gather information
3. Content Writer Agents: Creates sections in PARALLEL (3-5x speedup!)

This version uses the same asyncio.gather pattern that MultiAgentsOrchestrator
uses internally, without the full Conductor overhead.
"""

import sys
import asyncio
from pathlib import Path
import dspy
from typing import List
from duckduckgo_search import DDGS
import time

# Add Jotty to path
sys.path.insert(0, str(Path(__file__).parent))

from core.tools.content_generation import Document, Section, SectionType, ContentGenerators
from core.integration.direct_claude_cli_lm import DirectClaudeCLI


# =============================================================================
# MULTI-AGENT SIGNATURES (Same as simple version)
# =============================================================================

class PlannerSignature(dspy.Signature):
    """Planner Agent - Determines what sections to generate for a topic"""
    topic: str = dspy.InputField(desc="Topic to create guide about")
    goal: str = dspy.InputField(desc="Specific goal for the guide")

    section_titles: str = dspy.OutputField(
        desc="List of 10-15 section titles that should be covered. "
        "Format: one title per line. "
        "Each title should be specific to the topic. "
        "Example for Poodles: History, Breed Types, Grooming, Health. "
        "Example for Python: Syntax Basics, Data Types, Control Flow, Functions."
    )
    research_queries: str = dspy.OutputField(
        desc="Search queries to research this topic. "
        "Format: one query per line. "
        "5-10 specific queries that will help gather comprehensive information."
    )


class ResearcherSignature(dspy.Signature):
    """Researcher Agent - Conducts web research"""
    topic: str = dspy.InputField(desc="Topic to research")
    queries: str = dspy.InputField(desc="Search queries (one per line)")

    research_summary: str = dspy.OutputField(
        desc="Summary of research findings. "
        "Include key facts, statistics, and important information discovered. "
        "Write in flowing paragraphs."
    )


class ContentWriterSignature(dspy.Signature):
    """Content Writer Agent - Creates guide sections"""
    topic: str = dspy.InputField(desc="Topic of the guide")
    section_title: str = dspy.InputField(desc="Title of this section")
    research_context: str = dspy.InputField(desc="Research findings to use")

    section_content: str = dspy.OutputField(
        desc="Well-written section content for beginners. "
        "CRITICAL: Write in flowing paragraphs. "
        "DO NOT use hard line breaks within paragraphs. "
        "Use only natural paragraph breaks. "
        "Include specific details from research. "
        "Write 2-4 comprehensive paragraphs."
    )


# =============================================================================
# WEB SEARCH TOOL
# =============================================================================

def search_web(query: str, max_results: int = 5) -> List[dict]:
    """
    Search the web using DuckDuckGo (free, no API key needed)

    Args:
        query: Search query
        max_results: Maximum number of results to return

    Returns:
        List of search results with title, body, href
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
            return results
    except Exception as e:
        print(f"   ⚠️  Search failed for '{query}': {e}")
        return []


# =============================================================================
# ASYNC WRITER WRAPPER (for parallel execution)
# =============================================================================

async def write_section_async(
    writer: dspy.Module,
    topic: str,
    section_title: str,
    research_context: str,
    section_num: int,
    total_sections: int
) -> tuple[str, str]:
    """
    Async wrapper for section writing to enable parallel execution.

    This uses asyncio.run_in_executor to run the synchronous DSPy module
    in a thread pool, allowing multiple writers to run concurrently.

    Returns:
        tuple: (section_title, section_content)
    """
    loop = asyncio.get_event_loop()

    # Run DSPy module in thread pool (DSPy modules are synchronous)
    result = await loop.run_in_executor(
        None,  # Use default executor
        lambda: writer(
            topic=topic,
            section_title=section_title,
            research_context=research_context
        )
    )

    print(f"   ✅ Completed {section_num}/{total_sections}: {section_title} ({len(result.section_content)} chars)")

    return (section_title, result.section_content)


# =============================================================================
# PARALLEL EXECUTION WORKFLOW
# =============================================================================

async def run_parallel_guide(topic: str, goal: str):
    """
    Run multi-agent workflow with PARALLEL section writing

    Uses asyncio.gather (same as MultiAgentsOrchestrator does internally)
    for 3-5x speedup on section generation.

    Phase 1: Planning - Determine sections
    Phase 2: Research - Gather information
    Phase 3: Content Generation - Write sections IN PARALLEL
    """

    print("\n" + "=" * 100)
    print(f"  JOTTY PARALLEL GUIDE GENERATOR: '{topic} for Dummies'")
    print("  Using asyncio.gather for Parallel Execution!")
    print("=" * 100 + "\n")

    # Configure LLM
    print("🔧 Configuring Claude CLI...")
    try:
        lm = DirectClaudeCLI(model="haiku")
        dspy.configure(lm=lm)
        print("✅ Configured with Claude CLI (Haiku)\n")
    except Exception as e:
        print(f"❌ Failed to configure LLM: {e}")
        print("   Install: npm install -g @anthropic-ai/claude-code\n")
        return None

    # =================================================================
    # PHASE 1: PLANNING
    # =================================================================

    print("=" * 100)
    print("  PHASE 1: PLANNING")
    print("=" * 100 + "\n")

    print("🤖 Agent 1: Planner")
    print(f"   Task: Determine sections for '{topic}'\n")

    planner = dspy.ChainOfThought(PlannerSignature)
    plan_result = planner(topic=topic, goal=goal)

    section_titles = [
        line.strip()
        for line in plan_result.section_titles.strip().split('\n')
        if line.strip()
    ]

    research_queries = [
        line.strip()
        for line in plan_result.research_queries.strip().split('\n')
        if line.strip()
    ]

    print(f"✅ Planned {len(section_titles)} sections:")
    for i, title in enumerate(section_titles, 1):
        print(f"   {i}. {title}")
    print(f"\n✅ Generated {len(research_queries)} research queries\n")

    # =================================================================
    # PHASE 2: RESEARCH
    # =================================================================

    print("=" * 100)
    print("  PHASE 2: RESEARCH")
    print("=" * 100 + "\n")

    print("🤖 Agent 2: Researcher")
    print(f"   Task: Conduct web research on '{topic}'\n")

    # Conduct web searches
    print(f"🔍 Searching for {len(research_queries)} queries...")
    all_results = []
    for i, query in enumerate(research_queries, 1):
        print(f"   {i}/{len(research_queries)}: {query}")
        results = search_web(query, max_results=3)
        all_results.extend(results)

    print(f"✅ Found {len(all_results)} search results\n")

    # Create researcher and run
    researcher = dspy.ChainOfThought(ResearcherSignature)
    research_result = researcher(
        topic=topic,
        queries="\n".join(research_queries)
    )

    print(f"✅ Research complete ({len(research_result.research_summary)} chars)\n")

    # =================================================================
    # PHASE 3: CONTENT GENERATION - PARALLEL! 🚀
    # =================================================================

    print("=" * 100)
    print("  PHASE 3: CONTENT GENERATION (PARALLEL)")
    print("=" * 100 + "\n")

    print(f"🤖 Agent 3: Content Writers (x{len(section_titles)})")
    print(f"   Task: Write {len(section_titles)} sections IN PARALLEL\n")

    # Create writer module (shared by all tasks)
    writer = dspy.ChainOfThought(ContentWriterSignature)

    # Create parallel tasks for all sections
    print(f"🚀 Spawning {len(section_titles)} parallel writers...")
    start_time = time.time()

    tasks = [
        write_section_async(
            writer=writer,
            topic=topic,
            section_title=section_title,
            research_context=research_result.research_summary,
            section_num=i+1,
            total_sections=len(section_titles)
        )
        for i, section_title in enumerate(section_titles)
    ]

    # Execute ALL writers in parallel using asyncio.gather
    # This is the SAME pattern MultiAgentsOrchestrator uses internally!
    sections = await asyncio.gather(*tasks)

    elapsed = time.time() - start_time

    print(f"\n✅ Parallel execution complete in {elapsed:.2f}s")
    estimated_sequential = elapsed * len(section_titles)
    print(f"   (Sequential would take ~{estimated_sequential:.2f}s)")
    print(f"   Speedup: {len(section_titles):.1f}x\n")

    # =================================================================
    # GENERATE OUTPUT FILES
    # =================================================================

    print("=" * 100)
    print("  GENERATING OUTPUT FILES")
    print("=" * 100 + "\n")

    # Create document
    doc = Document(
        title=f"{topic} for Dummies: A Comprehensive Guide",
        sections=[
            Section(
                type=SectionType.TEXT,
                content=content,
                title=title
            )
            for title, content in sections
        ]
    )

    # Generate outputs
    topic_slug = topic.lower().replace(" ", "_").replace("/", "_")
    output_dir = Path("outputs") / f"{topic_slug}_guide"
    output_dir.mkdir(parents=True, exist_ok=True)

    generators = ContentGenerators()

    print("📄 Generating PDF...")
    pdf_path = generators.generate_pdf(doc, output_path=output_dir, format='a4')
    print(f"   ✅ Saved: {pdf_path}")

    print("📄 Generating Markdown...")
    md_path = generators.export_markdown(doc, output_path=output_dir)
    print(f"   ✅ Saved: {md_path}")

    print("📄 Generating HTML...")
    html_path = generators.generate_html(doc, output_path=output_dir)
    print(f"   ✅ Saved: {html_path}")

    print("\n" + "=" * 100)
    print("  ✨ PARALLEL GUIDE GENERATION COMPLETE!")
    print("=" * 100)
    print(f"\nOutputs saved to: {output_dir}/")
    print(f"  - Markdown: {md_path.name}")
    print(f"  - HTML: {html_path.name}")
    print(f"  - PDF: {pdf_path.name}\n")

    return doc


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate comprehensive guide with parallel execution")
    parser.add_argument("topic", help="Topic to create guide about")
    parser.add_argument("--goal", default="A comprehensive beginner's guide", help="Specific goal for the guide")

    args = parser.parse_args()

    # Run async workflow
    asyncio.run(run_parallel_guide(args.topic, args.goal))
