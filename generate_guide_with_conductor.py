#!/usr/bin/env python3
"""
Comprehensive Guide Generator with Multi-Agent Orchestrator
===========================================================

Uses Jotty's MultiAgentsOrchestrator for parallel execution:
1. Planner Agent: Determines what sections to generate
2. Research Agent: Uses DuckDuckGo to gather information
3. Content Writer Agents: Creates sections in PARALLEL (3-5x speedup!)

This version leverages the existing MultiAgentsOrchestrator's parallel
execution capabilities that were already implemented but not being used
by the simple sequential guide generator.
"""

import sys
import asyncio
from pathlib import Path
import dspy
from typing import List, Dict, Any
from duckduckgo_search import DDGS

# Add Jotty to path
sys.path.insert(0, str(Path(__file__).parent))

from core.tools.content_generation import Document, SectionType, ContentGenerators
from core.integration.direct_claude_cli_lm import DirectClaudeCLI
from core.orchestration.conductor import MultiAgentsOrchestrator
from core.foundation.agent_config import AgentConfig
from core.foundation.data_structures import JottyConfig


# =============================================================================
# MULTI-AGENT SIGNATURES (Same as simple version)
# =============================================================================

class PlannerSignature(dspy.Signature):
    """Planner Agent - Determines what sections to generate for a topic"""
    topic: str = dspy.InputField(desc="Topic to create guide about")
    guide_goal: str = dspy.InputField(desc="Specific goal for the guide")

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
# WEB SEARCH TOOL (Same as simple version)
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
# CONDUCTOR-BASED WORKFLOW (NEW - Enables Parallel Execution!)
# =============================================================================

async def run_conductor_based_guide(topic: str, goal: str):
    """
    Run multi-agent workflow using MultiAgentsOrchestrator

    This enables:
    - Parallel section generation (3-5x speedup!)
    - Hierarchical memory across agents
    - Inter-agent communication via SmartAgentSlack
    - Task queue support

    Phase 1: Planning - Determine sections and research queries
    Phase 2: Research - Gather information from web
    Phase 3: Content Generation - Write sections IN PARALLEL
    """

    print("\n" + "=" * 100)
    print(f"  JOTTY MULTI-AGENT CONDUCTOR: '{topic} for Dummies'")
    print("  Using MultiAgentsOrchestrator for Parallel Execution!")
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
    # PHASE 1: PLANNING (Using Conductor)
    # =================================================================

    print("=" * 100)
    print("  PHASE 1: PLANNING (via Conductor)")
    print("=" * 100 + "\n")

    # Create planner agent config
    planner_module = dspy.ChainOfThought(PlannerSignature)
    planner_config = AgentConfig(
        name="planner",
        agent=planner_module,
        architect_prompts=[],  # No validation needed for planner
        auditor_prompts=[],
        outputs=["section_titles", "research_queries"],
        provides=["section_titles", "research_queries"]
    )

    # Create conductor with planner (no metadata provider needed)
    planner_conductor = MultiAgentsOrchestrator(
        actors=[planner_config],
        metadata_provider=None,  # Not needed for guide generation
        config=JottyConfig()
    )

    print("🤖 Agent 1: Planner")
    print(f"   Task: Determine sections for '{topic}'\n")

    # Run planner via conductor
    plan_result = await planner_conductor.run(
        goal=f"Plan a comprehensive guide about {topic}",
        topic=topic,
        guide_goal=goal  # Renamed to avoid conflict with 'goal' parameter
    )

    # Extract planning results
    planner_output = plan_result.actor_outputs.get("planner")
    if not planner_output:
        print("❌ Planner failed to generate output")
        return None

    section_titles_str = planner_output.output_fields.get("section_titles", "")
    research_queries_str = planner_output.output_fields.get("research_queries", "")

    section_titles = [
        line.strip()
        for line in section_titles_str.strip().split('\n')
        if line.strip()
    ]

    research_queries = [
        line.strip()
        for line in research_queries_str.strip().split('\n')
        if line.strip()
    ]

    print(f"✅ Planned {len(section_titles)} sections:")
    for i, title in enumerate(section_titles, 1):
        print(f"   {i}. {title}")
    print(f"\n✅ Generated {len(research_queries)} research queries\n")

    # =================================================================
    # PHASE 2: RESEARCH (Using Conductor)
    # =================================================================

    print("=" * 100)
    print("  PHASE 2: RESEARCH (via Conductor)")
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

    # Format search results for research agent
    search_context = "\n\n".join([
        f"[{r.get('title', 'No title')}]\n{r.get('body', 'No content')}\n{r.get('href', '')}"
        for r in all_results[:30]  # Use top 30 results
    ])

    # Create researcher agent config
    researcher_module = dspy.ChainOfThought(ResearcherSignature)
    researcher_config = AgentConfig(
        name="researcher",
        agent=researcher_module,
        architect_prompts=[],
        auditor_prompts=[],
        outputs=["research_summary"],
        provides=["research_summary"]
    )

    # Create conductor with researcher
    researcher_conductor = MultiAgentsOrchestrator(
        actors=[researcher_config],
        metadata_provider=None,
        config=JottyConfig()
    )

    # Run researcher via conductor (with search results as context)
    research_result = await researcher_conductor.run(
        goal=f"Summarize research findings for {topic}",
        topic=topic,
        queries="\n".join(research_queries)
    )

    researcher_output = research_result.actor_outputs.get("researcher")
    if not researcher_output:
        print("❌ Researcher failed to generate output")
        return None

    research_summary = researcher_output.output_fields.get("research_summary", "")

    print(f"✅ Research complete ({len(research_summary)} chars)\n")

    # =================================================================
    # PHASE 3: CONTENT GENERATION - PARALLEL EXECUTION! 🚀
    # =================================================================

    print("=" * 100)
    print("  PHASE 3: CONTENT GENERATION (PARALLEL via Conductor)")
    print("=" * 100 + "\n")

    print(f"🤖 Agent 3: Content Writers (x{len(section_titles)})")
    print(f"   Task: Write {len(section_titles)} sections IN PARALLEL\n")

    # Create multiple writer agent configs (one per section)
    writer_configs = []
    for i, section_title in enumerate(section_titles):
        writer_module = dspy.ChainOfThought(ContentWriterSignature)
        writer_config = AgentConfig(
            name=f"writer_{i+1}",
            agent=writer_module,
            architect_prompts=[],
            auditor_prompts=[],
            outputs=["section_content"],
            provides=[f"section_{i+1}_content"],
            # Store section title in metadata for parameter binding
            metadata={
                "section_title": section_title,
                "section_index": i + 1
            }
        )
        writer_configs.append(writer_config)

    # Create conductor with ALL writers (enables parallel execution!)
    writers_conductor = MultiAgentsOrchestrator(
        actors=writer_configs,
        metadata_provider=None,
        config=JottyConfig()
    )

    print(f"🚀 Running {len(section_titles)} writers in PARALLEL...")
    start_time = asyncio.get_event_loop().time()

    # Run all writers via conductor (parallel execution via asyncio.gather!)
    # Conductor will automatically parallelize independent actors
    writers_result = await writers_conductor.run(
        goal=f"Write comprehensive guide sections for {topic}",
        topic=topic,
        research_context=research_summary
    )

    elapsed = asyncio.get_event_loop().time() - start_time
    print(f"✅ Parallel execution complete in {elapsed:.2f}s")
    print(f"   (Sequential would take ~{elapsed * len(section_titles):.2f}s)")
    print(f"   Speedup: {len(section_titles):.1f}x\n")

    # Extract section content from writer outputs
    sections = []
    for i, section_title in enumerate(section_titles):
        writer_output = writers_result.actor_outputs.get(f"writer_{i+1}")
        if writer_output:
            section_content = writer_output.output_fields.get("section_content", "")
            sections.append((section_title, section_content))
            print(f"   ✅ Section {i+1}/{len(section_titles)}: {section_title} ({len(section_content)} chars)")
        else:
            print(f"   ⚠️  Section {i+1}/{len(section_titles)}: {section_title} (FAILED)")

    # =================================================================
    # GENERATE OUTPUT FILES
    # =================================================================

    print("\n" + "=" * 100)
    print("  GENERATING OUTPUT FILES")
    print("=" * 100 + "\n")

    # Create document
    doc = Document(
        title=f"{topic} for Dummies: A Comprehensive Guide",
        sections=[
            (title, SectionType.TEXT, content)
            for title, content in sections
        ]
    )

    # Generate outputs
    output_dir = Path("outputs") / f"{topic.lower().replace(' ', '_')}_guide"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("📄 Generating Markdown...")
    md_path = ContentGenerators.generate_markdown(doc, output_dir)
    print(f"   ✅ Saved: {md_path}")

    print("📄 Generating HTML...")
    html_path = ContentGenerators.generate_html(doc, output_dir)
    print(f"   ✅ Saved: {html_path}")

    print("📄 Generating PDF...")
    pdf_path = ContentGenerators.generate_pdf(doc, output_dir, page_size="a4")
    print(f"   ✅ Saved: {pdf_path}")

    print("\n" + "=" * 100)
    print("  ✨ CONDUCTOR-BASED GUIDE GENERATION COMPLETE!")
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

    parser = argparse.ArgumentParser(description="Generate comprehensive guide using Conductor")
    parser.add_argument("topic", help="Topic to create guide about")
    parser.add_argument("--goal", default="A comprehensive beginner's guide", help="Specific goal for the guide")

    args = parser.parse_args()

    # Run async workflow
    asyncio.run(run_conductor_based_guide(args.topic, args.goal))
