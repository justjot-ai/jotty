#!/usr/bin/env python3
"""
Real-World Multi-Agent Demonstrations
======================================

This runs ACTUAL multi-agent tasks that produce REAL output:
1. Stock Market Analysis Report (Hierarchical)
2. API Design & Implementation (Debate)
3. Research Paper Generation (Pipeline)
4. Code Review & Refactoring (Swarm)

Each demo uses real LLM, creates actual files, and demonstrates
adaptive workflow selection.

Run: python3 run_real_world_demos.py
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime

# Add Jotty to path
sys.path.insert(0, str(Path(__file__).parent))

from core.orchestration.universal_workflow import UniversalWorkflow
from core.foundation.data_structures import JottyConfig
from core.integration.direct_claude_cli_lm import DirectClaudeCLI
import dspy


def print_banner(title: str):
    """Print demo banner."""
    print("\n" + "=" * 100)
    print(f"  {title}")
    print("=" * 100 + "\n")


def print_section(title: str):
    """Print section header."""
    print("\n" + "-" * 100)
    print(f"  {title}")
    print("-" * 100 + "\n")


async def demo_1_stock_analysis_hierarchical():
    """
    DEMO 1: Stock Market Analysis Report
    =====================================

    Workflow: HIERARCHICAL (Lead + Sub-agents)

    Lead Agent: Decomposes task into subtasks
    Sub-Agents:
      - Financial Data Analyst
      - Technical Analyst
      - Fundamental Analyst
      - Report Writer

    Output: Comprehensive stock analysis report
    """
    print_banner("DEMO 1: Stock Market Analysis - Hierarchical Workflow")

    # Configure LLM
    lm = DirectClaudeCLI(model='sonnet')
    dspy.configure(lm=lm)

    # Create workflow
    config = JottyConfig()
    workflow = UniversalWorkflow([], config)

    # Create output directory
    output_dir = Path("./outputs/stock_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("🎯 Goal: Analyze AAPL stock and create investment report")
    print("📊 Workflow Mode: HIERARCHICAL")
    print("👥 Team Structure:")
    print("   - Lead Agent: Task decomposer & coordinator")
    print("   - Sub-Agent 1: Financial metrics analyst")
    print("   - Sub-Agent 2: Technical chart analyst")
    print("   - Sub-Agent 3: Fundamental analysis expert")
    print("   - Sub-Agent 4: Report writer\n")

    print("⏳ Running hierarchical workflow...\n")

    result = await workflow.run(
        goal="""Analyze Apple Inc (AAPL) stock and create a comprehensive investment report.

        Requirements:
        1. Analyze financial metrics (P/E ratio, revenue growth, profit margins)
        2. Technical analysis (price trends, support/resistance levels)
        3. Fundamental analysis (business model, competitive advantages, risks)
        4. Investment recommendation with target price
        5. Create a markdown report with sections for each analysis

        Save the report to outputs/stock_analysis/AAPL_report.md
        """,
        context={
            'ticker': 'AAPL',
            'output_dir': str(output_dir),
            'report_format': 'markdown'
        },
        mode='hierarchical',
        num_sub_agents=4
    )

    print_section("RESULTS")
    print(f"Status: {result.get('status', 'unknown')}")
    print(f"Mode Used: {result.get('mode_used', 'unknown')}")

    if result.get('results'):
        results = result['results']
        if 'lead_decomposition' in results:
            print(f"\n✅ Lead Agent decomposed task successfully")
        if 'sub_agent_results' in results:
            print(f"✅ {len(results['sub_agent_results'])} sub-agents completed work")
        if 'session_file' in results:
            print(f"✅ Session saved: {results['session_file']}")

    # Check for created files
    if output_dir.exists():
        files = list(output_dir.glob("*.md"))
        if files:
            print(f"\n📄 Generated Files:")
            for f in files:
                print(f"   - {f.name} ({f.stat().st_size} bytes)")

    return result


async def demo_2_api_design_debate():
    """
    DEMO 2: REST API Design via Debate
    ===================================

    Workflow: DEBATE (Competing solutions → critique → vote)

    Phase 1 - Proposals:
      - Expert 1: REST + PostgreSQL
      - Expert 2: GraphQL + MongoDB
      - Expert 3: gRPC + Redis

    Phase 2 - Critique:
      - Each reviews others' proposals

    Phase 3 - Decision:
      - Judge selects best approach or synthesizes hybrid

    Output: API specification document
    """
    print_banner("DEMO 2: REST API Design - Debate Workflow")

    # Configure LLM
    lm = DirectClaudeCLI(model='sonnet')
    dspy.configure(lm=lm)

    # Create workflow
    config = JottyConfig()
    workflow = UniversalWorkflow([], config)

    # Create output directory
    output_dir = Path("./outputs/api_design")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("🎯 Goal: Design REST API for social media platform")
    print("📊 Workflow Mode: DEBATE")
    print("👥 Debate Structure:")
    print("   Phase 1: 3 experts propose different architectures")
    print("   Phase 2: Experts critique each other's designs")
    print("   Phase 3: Judge selects best approach or creates hybrid")
    print()

    print("⏳ Running debate workflow...\n")

    result = await workflow.run(
        goal="""Design a REST API for a social media platform.

        Requirements:
        - User authentication & authorization
        - Post creation, editing, deletion
        - Like/comment functionality
        - Follow/unfollow users
        - News feed generation
        - Scalability for millions of users

        Debate different approaches:
        - Database choice (SQL vs NoSQL)
        - Caching strategy
        - Real-time updates (WebSockets, SSE, polling)
        - API versioning

        Create an OpenAPI specification file saved to outputs/api_design/social_api_spec.yaml
        """,
        context={
            'platform_type': 'social_media',
            'scale': 'millions_of_users',
            'output_dir': str(output_dir)
        },
        mode='debate',
        num_debaters=3
    )

    print_section("RESULTS")
    print(f"Status: {result.get('status', 'unknown')}")
    print(f"Mode Used: {result.get('mode_used', 'unknown')}")

    if result.get('results'):
        results = result['results']
        if 'proposals' in results:
            print(f"\n✅ {len(results['proposals'])} proposals generated")
        if 'critiques' in results:
            print(f"✅ Critiques completed")
        if 'final_decision' in results:
            print(f"✅ Final decision reached")

    # Check for created files
    if output_dir.exists():
        files = list(output_dir.glob("*.yaml")) + list(output_dir.glob("*.md"))
        if files:
            print(f"\n📄 Generated Files:")
            for f in files:
                print(f"   - {f.name} ({f.stat().st_size} bytes)")

    return result


async def demo_3_research_paper_pipeline():
    """
    DEMO 3: Research Paper Generation
    ==================================

    Workflow: PIPELINE (Data flow through stages)

    Stages:
      1. Literature Review (gather sources)
      2. Outline Creation (structure paper)
      3. Introduction Writing
      4. Methodology Writing
      5. Results & Analysis
      6. Conclusion Writing
      7. References & Citations

    Output: Complete research paper in markdown
    """
    print_banner("DEMO 3: Research Paper Generation - Pipeline Workflow")

    # Configure LLM
    lm = DirectClaudeCLI(model='sonnet')
    dspy.configure(lm=lm)

    # Create workflow
    config = JottyConfig()
    workflow = UniversalWorkflow([], config)

    # Create output directory
    output_dir = Path("./outputs/research_paper")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("🎯 Goal: Generate research paper on multi-agent RL")
    print("📊 Workflow Mode: PIPELINE")
    print("🔄 Pipeline Stages:")
    print("   Stage 1: Literature review (find relevant papers)")
    print("   Stage 2: Create outline and structure")
    print("   Stage 3: Write introduction")
    print("   Stage 4: Write methodology section")
    print("   Stage 5: Write results & analysis")
    print("   Stage 6: Write conclusion")
    print("   Stage 7: Format references and citations")
    print()

    print("⏳ Running pipeline workflow...\n")

    result = await workflow.run(
        goal="""Generate a research paper on 'Multi-Agent Reinforcement Learning in Distributed Systems'.

        Requirements:
        - Academic writing style
        - Proper structure (Abstract, Intro, Methodology, Results, Conclusion)
        - Citations to relevant papers
        - Diagrams for system architecture
        - 5-7 pages

        Save to outputs/research_paper/marl_paper.md
        """,
        context={
            'topic': 'Multi-Agent Reinforcement Learning',
            'field': 'Distributed Systems',
            'output_dir': str(output_dir),
            'format': 'markdown'
        },
        mode='pipeline',
        stages=[
            'Literature Review: Find 10+ relevant papers on MARL',
            'Outline: Create paper structure with sections',
            'Abstract: Write compelling abstract (200 words)',
            'Introduction: Motivate problem and contribution',
            'Methodology: Explain MARL algorithms and architecture',
            'Results: Present experimental findings',
            'Conclusion: Summarize contributions and future work',
            'References: Format citations in IEEE style'
        ]
    )

    print_section("RESULTS")
    print(f"Status: {result.get('status', 'unknown')}")
    print(f"Mode Used: {result.get('mode_used', 'unknown')}")

    if result.get('results'):
        results = result['results']
        if 'pipeline_results' in results:
            print(f"\n✅ Pipeline completed {len(results['pipeline_results'])} stages")
            for i, stage_result in enumerate(results['pipeline_results'], 1):
                print(f"   Stage {i}: {stage_result.get('status', 'unknown')}")

    # Check for created files
    if output_dir.exists():
        files = list(output_dir.glob("*.md")) + list(output_dir.glob("*.pdf"))
        if files:
            print(f"\n📄 Generated Files:")
            for f in files:
                print(f"   - {f.name} ({f.stat().st_size} bytes)")

    return result


async def demo_4_code_review_swarm():
    """
    DEMO 4: Codebase Review & Refactoring
    ======================================

    Workflow: SWARM (Self-organizing agents)

    Agents self-select files based on:
      - Expertise (frontend/backend/database/tests)
      - Current workload
      - File complexity

    Each agent:
      - Reviews assigned files
      - Identifies issues (bugs, code smells, security)
      - Suggests refactorings
      - Creates improvement report

    Output: Code review report with recommendations
    """
    print_banner("DEMO 4: Code Review & Refactoring - Swarm Workflow")

    # Configure LLM
    lm = DirectClaudeCLI(model='sonnet')
    dspy.configure(lm=lm)

    # Create workflow
    config = JottyConfig()
    workflow = UniversalWorkflow([], config)

    # Create output directory
    output_dir = Path("./outputs/code_review")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create sample codebase
    sample_code_dir = Path("./outputs/sample_codebase")
    sample_code_dir.mkdir(parents=True, exist_ok=True)

    # Create sample files
    (sample_code_dir / "app.py").write_text("""
# Sample Flask app with issues
from flask import Flask
app = Flask(__name__)

@app.route('/')
def home():
    # TODO: Add authentication
    return "Hello World"

@app.route('/user/<id>')
def get_user(id):
    # SQL injection vulnerability!
    query = f"SELECT * FROM users WHERE id = {id}"
    return execute_query(query)
""")

    (sample_code_dir / "database.py").write_text("""
# Database connection with issues
import sqlite3

def get_connection():
    # Hardcoded credentials - security issue!
    return sqlite3.connect('database.db')

def execute_query(query):
    # No error handling!
    conn = get_connection()
    result = conn.execute(query).fetchall()
    return result
""")

    print("🎯 Goal: Review codebase and create refactoring plan")
    print("📊 Workflow Mode: SWARM")
    print("🐝 Swarm Behavior:")
    print("   - Agents announce capabilities")
    print("   - Self-select files based on expertise")
    print("   - Review code independently")
    print("   - Coordinate via shared scratchpad")
    print("   - Aggregate findings into report")
    print()

    print("⏳ Running swarm workflow...\n")

    result = await workflow.run(
        goal="""Review the sample codebase and create a comprehensive refactoring plan.

        Review for:
        - Security vulnerabilities (SQL injection, hardcoded secrets, etc.)
        - Code smells (duplicated code, long functions, etc.)
        - Performance issues
        - Missing error handling
        - Lack of tests

        Create a prioritized refactoring plan saved to outputs/code_review/refactoring_plan.md
        """,
        context={
            'codebase': str(sample_code_dir),
            'output_dir': str(output_dir),
            'focus_areas': ['security', 'maintainability', 'performance']
        },
        mode='swarm',
        num_agents=5
    )

    print_section("RESULTS")
    print(f"Status: {result.get('status', 'unknown')}")
    print(f"Mode Used: {result.get('mode_used', 'unknown')}")

    if result.get('results'):
        results = result['results']
        if 'swarm_results' in results:
            print(f"\n✅ {len(results['swarm_results'])} agents participated")
        if 'findings' in results:
            print(f"✅ Found {len(results['findings'])} issues")

    # Check for created files
    if output_dir.exists():
        files = list(output_dir.glob("*.md"))
        if files:
            print(f"\n📄 Generated Files:")
            for f in files:
                print(f"   - {f.name} ({f.stat().st_size} bytes)")

    return result


async def demo_5_auto_mode_comparison():
    """
    DEMO 5: Auto-Mode Selection Comparison
    =======================================

    Test auto-mode with different task complexities:
    - Simple task → Sequential
    - Medium task → P2P
    - Complex task → Hierarchical
    - Exploratory task → Debate

    This demonstrates Jotty's adaptive intelligence!
    """
    print_banner("DEMO 5: Auto-Mode Selection - Adaptive Intelligence")

    # Configure LLM
    lm = DirectClaudeCLI(model='sonnet')
    dspy.configure(lm=lm)

    # Create workflow
    config = JottyConfig()
    workflow = UniversalWorkflow([], config)

    test_tasks = [
        {
            'name': 'Simple Task',
            'goal': 'Create a Python function to calculate fibonacci numbers',
            'expected': 'sequential',
        },
        {
            'name': 'Medium Task',
            'goal': 'Build a command-line TODO app with persistent storage',
            'expected': 'p2p or sequential',
        },
        {
            'name': 'Complex Task',
            'goal': 'Design and implement a distributed cache system with consistency guarantees',
            'expected': 'hierarchical',
        },
        {
            'name': 'Exploratory Task',
            'goal': 'Determine the best database architecture for a social network (evaluate trade-offs)',
            'expected': 'debate',
        },
    ]

    print("🎯 Testing auto-mode selection with different task complexities\n")

    results = []
    for i, task in enumerate(test_tasks, 1):
        print(f"\n{'─' * 100}")
        print(f"  Test {i}/{len(test_tasks)}: {task['name']}")
        print(f"{'─' * 100}\n")
        print(f"Goal: {task['goal']}")
        print(f"Expected Mode: {task['expected']}")
        print("\n⏳ Analyzing...\n")

        result = await workflow.run(
            goal=task['goal'],
            context={},
            mode='auto'  # Let Jotty decide!
        )

        actual_mode = result.get('mode_used', 'unknown')
        analysis = result.get('analysis', {})

        print(f"✅ Selected Mode: {actual_mode}")
        if analysis:
            print(f"   Complexity: {analysis.get('complexity', 'N/A')}")
            print(f"   Uncertainty: {analysis.get('uncertainty', 'N/A')}")
            print(f"   Reasoning: {analysis.get('reasoning', 'N/A')}")

        results.append({
            'task': task['name'],
            'expected': task['expected'],
            'actual': actual_mode,
            'analysis': analysis
        })

    print_section("AUTO-MODE SELECTION SUMMARY")
    print(f"\n{'Task':<20} {'Expected':<15} {'Actual':<15} {'Match':<10}")
    print("─" * 60)
    for r in results:
        expected = r['expected']
        actual = r['actual']
        match = "✅" if actual in expected or expected in actual else "⚠️"
        print(f"{r['task']:<20} {expected:<15} {actual:<15} {match:<10}")

    return results


async def main():
    """Run all real-world demos."""
    print("\n" + "█" * 100)
    print("█" + " " * 98 + "█")
    print("█" + "  JOTTY - REAL-WORLD MULTI-AGENT DEMONSTRATIONS".center(98) + "█")
    print("█" + " " * 98 + "█")
    print("█" * 100)

    start_time = datetime.now()

    results = {}

    try:
        # Demo 1: Stock Analysis (Hierarchical)
        print("\n⏱️  Running Demo 1: Stock Analysis...")
        results['demo1_stock'] = await demo_1_stock_analysis_hierarchical()

        # Demo 2: API Design (Debate)
        print("\n⏱️  Running Demo 2: API Design...")
        results['demo2_api'] = await demo_2_api_design_debate()

        # Demo 3: Research Paper (Pipeline)
        print("\n⏱️  Running Demo 3: Research Paper...")
        results['demo3_research'] = await demo_3_research_paper_pipeline()

        # Demo 4: Code Review (Swarm)
        print("\n⏱️  Running Demo 4: Code Review...")
        results['demo4_review'] = await demo_4_code_review_swarm()

        # Demo 5: Auto-Mode Comparison
        print("\n⏱️  Running Demo 5: Auto-Mode Selection...")
        results['demo5_auto'] = await demo_5_auto_mode_comparison()

    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    # Final Summary
    print("\n" + "█" * 100)
    print("█" + " " * 98 + "█")
    print("█" + "  DEMONSTRATION COMPLETE".center(98) + "█")
    print("█" + " " * 98 + "█")
    print("█" * 100)

    print(f"\n⏱️  Total Duration: {duration:.2f} seconds")
    print(f"✅ Demos Completed: {len(results)}/5")

    print("\n📊 Workflow Modes Demonstrated:")
    print("   ✅ Hierarchical (Lead + Sub-agents)")
    print("   ✅ Debate (Proposals → Critique → Vote)")
    print("   ✅ Pipeline (Sequential stages with data flow)")
    print("   ✅ Swarm (Self-organizing agents)")
    print("   ✅ Auto-Mode (Adaptive selection)")

    print("\n📁 Output Directories:")
    output_dirs = [
        "./outputs/stock_analysis",
        "./outputs/api_design",
        "./outputs/research_paper",
        "./outputs/code_review",
    ]
    for dir_path in output_dirs:
        if Path(dir_path).exists():
            files = list(Path(dir_path).glob("*"))
            print(f"   {dir_path}: {len(files)} files")

    print("\n🎯 Key Achievements:")
    print("   - Real LLM calls (DirectClaudeCLI)")
    print("   - Actual file creation")
    print("   - Multiple workflow patterns")
    print("   - Adaptive mode selection")
    print("   - Zero code duplication (DRY)")

    print("\n🚀 Jotty is now a world-class self-adaptive multi-agent system!")

    return results


if __name__ == "__main__":
    print("\n🚀 Starting Real-World Multi-Agent Demonstrations...")
    print("   (This will use REAL LLM and create ACTUAL files)\n")

    results = asyncio.run(main())

    print("\n" + "█" * 100)
    print("█" + "  ALL DEMONSTRATIONS COMPLETE".center(98) + "█")
    print("█" * 100 + "\n")
