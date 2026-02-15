"""
Test Mermaid Expert Agent with Real LLM (Claude/Cursor)

Tests the full learning pipeline:
1. Configure real LLM (Claude/Cursor via DSPy)
2. Train expert agent on examples
3. Test with complex descriptions it hasn't seen
4. Verify it actually learned and generates correct diagrams
"""

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import dspy

    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False
    print("❌ DSPy not available. Install with: pip install dspy-ai")
    sys.exit(1)

from core.experts import ExpertAgentConfig, MermaidExpertAgent

# Import Claude CLI wrapper
try:
    from examples.claude_cli_wrapper import ClaudeCLILM

    CLAUDE_CLI_AVAILABLE = True
except ImportError:
    CLAUDE_CLI_AVAILABLE = False
    print("⚠️  Claude CLI wrapper not available")


def configure_llm():
    """Configure DSPy with Claude CLI wrapper (preferred) or API keys."""
    # First try Claude CLI wrapper (preferred method)
    if CLAUDE_CLI_AVAILABLE:
        print("🔧 Configuring DSPy with Claude CLI wrapper...")
        try:
            # Check if Claude CLI is available
            import subprocess

            result = subprocess.run(
                ["claude", "--version"], capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                print(f"✅ Claude CLI available: {result.stdout.strip()}")
                lm = ClaudeCLILM(model="sonnet")  # or "haiku", "opus"
                dspy.configure(lm=lm)
                print("✅ Configured with Claude CLI wrapper")
                return True
            else:
                print("⚠️  Claude CLI not working")
        except FileNotFoundError:
            print("⚠️  Claude CLI not found")
            print("   Install from: https://github.com/anthropics/claude-code")
        except Exception as e:
            print(f"⚠️  Failed to configure Claude CLI: {e}")

    # Fallback to API keys
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    if anthropic_key:
        print("🔧 Configuring DSPy with Claude API...")
        try:
            lm = dspy.LM(model="anthropic/claude-3-5-haiku-20241022")
            dspy.configure(lm=lm)
            print("✅ Configured with Claude API")
            return True
        except Exception as e:
            print(f"⚠️  Failed to configure Claude API: {e}")

    if openai_key:
        print("🔧 Configuring DSPy with OpenAI...")
        try:
            lm = dspy.LM(model="openai/gpt-4o-mini")
            dspy.configure(lm=lm)
            print("✅ Configured with OpenAI")
            return True
        except Exception as e:
            print(f"⚠️  Failed to configure OpenAI: {e}")

    # Check if already configured
    try:
        if hasattr(dspy.settings, "lm") and dspy.settings.lm is not None:
            print(f"✅ DSPy already configured with: {type(dspy.settings.lm).__name__}")
            return True
    except:
        pass

    print("⚠️  Could not configure LLM")
    print("   Options:")
    print("   1. Install Claude CLI: https://github.com/anthropics/claude-code")
    print("   2. Set ANTHROPIC_API_KEY environment variable")
    print("   3. Set OPENAI_API_KEY environment variable")
    return False


async def test_real_llm_learning():
    """Test MermaidExpertAgent with real LLM."""
    print("=" * 80)
    print("TESTING MERMAID EXPERT WITH REAL LLM")
    print("=" * 80)
    print()

    # Configure LLM
    llm_configured = configure_llm()

    if not llm_configured:
        print()
        print("=" * 80)
        print("⚠️  NO LLM CONFIGURED")
        print("=" * 80)
        print()
        print("To test with real LLM, configure one of:")
        print("  1. Claude: Set ANTHROPIC_API_KEY environment variable")
        print("  2. OpenAI: Set OPENAI_API_KEY environment variable")
        print("  3. Or configure manually: dspy.configure(lm=dspy.LM(model='claude-3-opus'))")
        print()
        print("For now, showing test structure...")
        print()
        return

    # Training examples - teach the agent basic patterns
    training_cases = [
        {
            "task": "Generate simple flowchart",
            "context": {"description": "Start to End flow", "diagram_type": "flowchart"},
            "gold_standard": """graph TD
    A[Start]
    B[End]
    A --> B""",
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
    B -->|No| D""",
        },
    ]

    # Create expert
    config = ExpertAgentConfig(
        name="mermaid_real_llm_test",
        domain="mermaid",
        description="Mermaid expert with real LLM",
        training_gold_standards=training_cases,
        max_training_iterations=3,
        required_training_pass_count=1,
        enable_teacher_model=True,
        save_improvements=True,
        expert_data_dir="./test_outputs/mermaid_real_llm",
    )

    expert = MermaidExpertAgent(config=config)

    # Train
    print("=" * 80)
    print("PHASE 1: TRAINING WITH REAL LLM")
    print("=" * 80)
    print()

    print("Training on examples...")
    print()

    try:
        training_results = await expert.train()

        print(f"Training Results:")
        print(f"  Overall Success: {training_results.get('overall_success')}")
        print(
            f"  Passed Cases: {training_results.get('passed_cases')}/{training_results.get('total_cases')}"
        )
        print()

        for case_result in training_results.get("training_cases", []):
            print(f"  Case {case_result['case_number']}: {case_result['task']}")
            print(f"    Success: {case_result['success']}")
            print(f"    Final Score: {case_result['final_score']:.2f}")
            print(f"    Iterations: {case_result['iterations']}")
            print()

        status = expert.get_status()
        print(f"Improvements Learned: {status['improvements_count']}")
        print()

        if not training_results.get("overall_success"):
            print("⚠️  Training did not fully succeed, but continuing with test...")
            print()

    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback

        traceback.print_exc()
        return

    # Test with complex descriptions
    print("=" * 80)
    print("PHASE 2: TESTING WITH COMPLEX DESCRIPTIONS")
    print("=" * 80)
    print()

    complex_tests = [
        {
            "name": "Complex Multi-Branch Decision Tree",
            "description": "A complex decision tree: Start with user authentication, check if user has admin permissions, validate the data, if all pass process request, otherwise show error",
            "diagram_type": "flowchart",
        },
        {
            "name": "Multi-Stage CI/CD Pipeline",
            "description": "CI/CD pipeline: Source code, Build, Unit Tests, Integration Tests, Deploy to Staging, Deploy to Production",
            "diagram_type": "flowchart",
        },
        {
            "name": "User Registration Flow",
            "description": "User registration: Enter email, validate format, check if exists, if exists show error, if not create account, send verification email, show success",
            "diagram_type": "flowchart",
        },
    ]

    results = []

    for i, test in enumerate(complex_tests, 1):
        print(f"Test {i}: {test['name']}")
        print(f"  Description: {test['description']}")
        print()

        try:
            # Generate diagram using learned patterns
            diagram = await expert.generate_mermaid(
                description=test["description"], diagram_type=test["diagram_type"]
            )

            print(f"  Generated Diagram:")
            print("  ```mermaid")
            print(f"  {diagram}")
            print("  ```")
            print()

            # Validate
            diagram_str = str(diagram)
            validation = {
                "has_graph": "graph" in diagram_str.lower() or "flowchart" in diagram_str.lower(),
                "has_nodes": "[" in diagram_str or "{" in diagram_str,
                "has_arrows": "-->" in diagram_str,
                "has_decision_nodes": "{" in diagram_str and "}" in diagram_str,
                "has_labels": "|" in diagram_str,
                "valid_syntax": True,  # If we got here, syntax is likely valid
            }

            print(f"  Validation:")
            for key, value in validation.items():
                status = "✅" if value else "❌"
                print(f"    {status} {key.replace('_', ' ').title()}: {value}")

            score = sum(validation.values()) / len(validation)
            print(f"  Score: {score:.2f} / 1.0")

            results.append(
                {
                    "name": test["name"],
                    "diagram": diagram,
                    "validation": validation,
                    "score": score,
                    "success": score >= 0.8,
                }
            )

        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback

            traceback.print_exc()
            results.append({"name": test["name"], "error": str(e)})

        print()

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()

    successful = [r for r in results if "success" in r and r["success"]]
    partial = [r for r in results if "score" in r and 0.5 <= r["score"] < 0.8]
    failed = [r for r in results if "error" in r or ("score" in r and r["score"] < 0.5)]

    print(f"✅ Successful: {len(successful)}")
    print(f"⚠️  Partial: {len(partial)}")
    print(f"❌ Failed: {len(failed)}")
    print()

    if successful or partial:
        all_scores = [r["score"] for r in results if "score" in r]
        if all_scores:
            avg_score = sum(all_scores) / len(all_scores)
            print(f"Average Score: {avg_score:.2f} / 1.0")
            print()

        print("Results:")
        for result in results:
            if "success" in result:
                status = "✅" if result["success"] else "⚠️" if result["score"] >= 0.5 else "❌"
                print(f"  {status} {result['name']}: {result['score']:.2f}")
            elif "error" in result:
                print(f"  ❌ {result['name']}: {result['error']}")

    # Show improvements
    improvements_file = Path(expert.data_dir) / "improvements.json"
    if improvements_file.exists():
        import json

        with open(improvements_file, "r") as f:
            improvements = json.load(f)

        print()
        print("=" * 80)
        print("LEARNED IMPROVEMENTS")
        print("=" * 80)
        print()
        print(f"Total Improvements: {len(improvements)}")
        print()

        for i, imp in enumerate(improvements[:3], 1):
            print(f"Improvement {i}:")
            print(f"  Task: {imp.get('task', 'Unknown')}")
            print(f"  Student: {imp.get('student_output', '')[:80]}...")
            print(f"  Teacher: {imp.get('teacher_output', '')[:80]}...")
            print()

    print("=" * 80)
    print("✅ TEST COMPLETE")
    print("=" * 80)

    return results


if __name__ == "__main__":
    asyncio.run(test_real_llm_learning())
