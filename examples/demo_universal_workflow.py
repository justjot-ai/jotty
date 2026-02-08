#!/usr/bin/env python3
"""
Universal Workflow Demo
=======================

Demonstrates adaptive workflow with multiple modes.

Run: python3 demo_universal_workflow.py
"""

import asyncio
import logging
from pathlib import Path

# Import Universal Workflow
from core.orchestration.universal_workflow import UniversalWorkflow
from core.foundation.jotty_config import JottyConfig
from core.integration.direct_claude_cli_lm import DirectClaudeCLI
import dspy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def demo_auto_mode():
    """Demo 1: Auto mode - Jotty picks best workflow."""
    print("\n" + "=" * 90)
    print("DEMO 1: AUTO MODE - Jotty Analyzes and Picks Best Workflow")
    print("=" * 90 + "\n")

    # Configure DSPy
    lm = DirectClaudeCLI(model='sonnet')
    dspy.configure(lm=lm)

    # Create workflow
    config = JottyConfig()
    workflow = UniversalWorkflow([], config)

    # Run with auto mode
    result = await workflow.run(
        goal="Build a simple calculator with basic operations",
        context={},
        mode='auto'  # Jotty decides!
    )

    print("\n✅ DEMO 1 COMPLETE")
    print(f"   Mode used: {result['mode_used']}")
    if result.get('analysis'):
        print(f"   Analysis:")
        print(f"      Complexity: {result['analysis']['complexity']}")
        print(f"      Uncertainty: {result['analysis']['uncertainty']}")
        print(f"      Reasoning: {result['analysis']['reasoning']}")


async def demo_stock_screener():
    """Demo 2: Stock screener with flexible context."""
    print("\n" + "=" * 90)
    print("DEMO 2: STOCK SCREENER - Flexible Context")
    print("=" * 90 + "\n")

    # Configure DSPy
    lm = DirectClaudeCLI(model='sonnet')
    dspy.configure(lm=lm)

    # Create workflow
    config = JottyConfig()
    workflow = UniversalWorkflow([], config)

    # Run with flexible context
    result = await workflow.run(
        goal="Build a stock market screening system to find undervalued growth stocks",
        context={
            # Flexible context - use what's relevant!
            'data_folder': '/var/www/sites/personal/stock_market/common/Data/FUNDAMENTALS',
            'quality_threshold': 0.85,
            'coding_style': 'Google Python Style Guide'
        },
        mode='auto'  # Let Jotty pick best mode
    )

    print("\n✅ DEMO 2 COMPLETE")
    print(f"   Mode used: {result['mode_used']}")
    print(f"   Status: {result['status']}")


async def demo_all_modes():
    """Demo 3: Show all available modes."""
    print("\n" + "=" * 90)
    print("DEMO 3: ALL MODES AVAILABLE")
    print("=" * 90 + "\n")

    modes = [
        ('sequential', 'Waterfall: A → B → C'),
        ('parallel', 'Independent: A, B, C (all at once)'),
        ('p2p', 'P2P Discovery + Sequential Delivery'),
        ('hierarchical', 'Lead + Sub-Agents'),
        ('debate', 'Propose → Critique → Vote'),
        ('round-robin', 'Iterative Refinement'),
        ('pipeline', 'Data Flow: A → B → C'),
        ('swarm', 'Self-Organizing Agents'),
    ]

    print("Available Workflow Modes:\n")
    for mode, description in modes:
        print(f"  {mode:15} → {description}")

    print("\nUsage:")
    print("  result = await workflow.run(goal='...', mode='hierarchical')")


async def demo_context_types():
    """Demo 4: Show flexible context handling."""
    print("\n" + "=" * 90)
    print("DEMO 4: FLEXIBLE CONTEXT TYPES")
    print("=" * 90 + "\n")

    examples = [
        ("Data Analysis", {
            'data_folder': '/path/to/data',
            'database': 'postgres://user:pass@host/db',
            'time_limit': '1 hour'
        }),
        ("Code Refactoring", {
            'codebase': '/path/to/repo',
            'requirements_doc': 'docs/REQUIREMENTS.md',
            'coding_style': 'PEP 8',
            'frameworks': ['FastAPI', 'SQLAlchemy']
        }),
        ("API Integration", {
            'codebase': '/path/to/repo',
            'api_docs': 'https://api.example.com/docs',
            'api_key': 'sk_test_...',
            'github_repo': 'https://github.com/user/repo'
        }),
        ("Resume Session", {
            'session_id': 'sess_123',
            'previous_output': 'output.json'
        })
    ]

    print("Context is FLEXIBLE - use what's relevant:\n")
    for task_type, context in examples:
        print(f"{task_type}:")
        for key, value in context.items():
            print(f"  {key}: {value}")
        print()


async def main():
    """Run all demos."""
    print("\n" + "=" * 90)
    print("UNIVERSAL WORKFLOW DEMO")
    print("=" * 90)

    # Demo 3: Show modes (no execution)
    await demo_all_modes()

    # Demo 4: Show context types (no execution)
    await demo_context_types()

    # Demo 1: Auto mode (requires Claude CLI)
    # Uncomment to run:
    # await demo_auto_mode()

    # Demo 2: Stock screener (requires Claude CLI + data)
    # Uncomment to run:
    # await demo_stock_screener()

    print("\n" + "=" * 90)
    print("DEMO COMPLETE")
    print("=" * 90)
    print("\nTo run with actual execution:")
    print("  1. Uncomment demo_auto_mode() or demo_stock_screener()")
    print("  2. Ensure Claude CLI is configured")
    print("  3. Run: python3 demo_universal_workflow.py")


if __name__ == "__main__":
    asyncio.run(main())
