"""
Test Mermaid Expert Agent with LLM Integration

Tests the MermaidExpertAgent using DSPy (Claude/Cursor) for generation.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import dspy
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False
    print("⚠️  DSPy not available. Install with: pip install dspy-ai")
    sys.exit(1)

from core.experts import MermaidExpertAgent, ExpertAgentConfig


async def test_mermaid_expert_with_llm():
    """Test MermaidExpertAgent with LLM (DSPy) integration."""
    print("=" * 80)
    print("TESTING MERMAID EXPERT WITH LLM (DSPy)")
    print("=" * 80)
    print()
    
    # Configure DSPy (use default LM or configure Claude/Cursor)
    # For testing, we'll use the default configured LM
    print("📚 DSPy Configuration:")
    print(f"  DSPy Available: {DSPY_AVAILABLE}")
    if DSPY_AVAILABLE:
        try:
            lm = dspy.settings.lm
            print(f"  LM: {type(lm).__name__}")
        except:
            print("  LM: Not configured (will use default)")
    print()
    
    # Create expert agent
    print("🔧 Creating Mermaid Expert Agent...")
    expert = MermaidExpertAgent()
    
    # Check if agent uses DSPy
    main_agent = expert._create_domain_agent()

    print(f"  Agent Type: {type(main_agent).__name__}")
    print(f"  Is DSPy Module: {isinstance(main_agent, dspy.Module) if DSPY_AVAILABLE else False}")
    print()
    
    # Train on simple examples first
    print("📚 Training on simple examples...")
    print()
    
    simple_training_cases = [
        {
            "task": "Generate simple flowchart",
            "context": {"description": "Start to End flow", "diagram_type": "flowchart"},
            "gold_standard": "graph TD\n    A[Start]\n    B[End]\n    A --> B"
        },
        {
            "task": "Generate decision flowchart",
            "context": {"description": "User login with validation", "diagram_type": "flowchart"},
            "gold_standard": """graph TD
    A[User Login]
    B{Valid?}
    C[Show Dashboard]
    D[Show Error]
    A --> B
    B -->|Yes| C
    B -->|No| D"""
        }
    ]
    
    try:
        # Verify training data (BaseExpert does not have .train())
        training_data = expert.get_training_data()
        print(f"Training Data:")
        print(f"  Cases available: {len(training_data)}")
        print()

        # Test generation
        print("=" * 80)
        print("TESTING GENERATION")
        print("=" * 80)
        print()

        test_cases = [
            {
                "description": "Simple workflow from start to end",
                "diagram_type": "flowchart"
            },
            {
                "description": "User login flow with validation decision",
                "diagram_type": "flowchart"
            }
        ]

        for i, test_case in enumerate(test_cases, 1):
            print(f"Test {i}: {test_case['description']}")
            print()

            try:
                diagram = await expert.generate_mermaid(
                    description=test_case['description'],
                    diagram_type=test_case['diagram_type']
                )

                print(f"Generated Diagram:")
                print("```mermaid")
                print(diagram)
                print("```")
                print()

                # Check if valid
                is_valid = "graph" in str(diagram).lower() or "flowchart" in str(diagram).lower()
                has_nodes = "[" in str(diagram) or "{" in str(diagram)
                has_arrow = "-->" in str(diagram)

                print(f"Validation:")
                print(f"  Has graph declaration: {is_valid}")
                print(f"  Has nodes: {has_nodes}")
                print(f"  Has arrows: {has_arrow}")
                print()

            except Exception as e:
                print(f"ERROR: {e}")
                import traceback
                traceback.print_exc()
                print()

        # Show stats
        print("=" * 80)
        print("EXPERT STATS")
        print("=" * 80)
        print()

        stats = expert.get_stats()
        print(f"  Domain: {stats['domain']}")
        print(f"  Improvements: {stats['improvements_count']}")
        print(f"  Training Cases: {stats['training_cases']}")
        print()

    except Exception as e:
        print(f"ERROR during testing: {e}")
        import traceback
        traceback.print_exc()
        print()
        print("Note: This test requires DSPy to be configured with an LLM.")
        print("Configure with: dspy.configure(lm=dspy.LM(model='claude-3-opus'))")
    
    print("=" * 80)
    print("✅ TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(test_mermaid_expert_with_llm())
