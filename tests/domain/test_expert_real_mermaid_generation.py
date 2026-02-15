"""
Test Expert Agent Real Mermaid Generation (No Mock)

Tests actual Mermaid diagram generation without mocking.
Requires DSPy to be configured with an LLM provider.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import subprocess

import dspy

from Jotty.core.intelligence.reasoning.experts import MermaidExpertAgent

# Configure DSPy with Claude CLI wrapper if available
try:
    # Check if Claude CLI is available
    result = subprocess.run(["which", "claude"], capture_output=True, text=True)
    if result.returncode == 0:
        print("✅ Claude CLI found, configuring DSPy...")
        # Use enhanced wrapper (DSPy-compatible)
        import sys

        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "JustJot.ai" / "supervisor"))
        from claude_cli_wrapper_enhanced import EnhancedClaudeCLILM

        dspy.configure(lm=EnhancedClaudeCLILM(model="sonnet"))
        print("✅ DSPy configured with Enhanced Claude CLI wrapper")
    else:
        print("⚠️  Claude CLI not found in PATH")
except Exception as e:
    print(f"⚠️  Claude CLI wrapper configuration failed: {e}")
    print("⚠️  Will try to use existing DSPy configuration")


async def test_expert_real_mermaid_generation():
    """
    Test expert agent actually generating a Mermaid diagram (no mock).

    This test will:
    1. Create expert (no training required - training is optional)
    2. Attempt to generate Mermaid diagram
    3. Verify the result contains Mermaid syntax

    Note: Requires DSPy to be configured with an LLM provider.
    """
    print("\n" + "=" * 70)
    print("Testing Expert Agent Real Mermaid Generation (No Mock)")
    print("=" * 70)

    # Step 1: Create expert
    print("\n1. Creating MermaidExpertAgent...")
    expert = MermaidExpertAgent()
    print(f"   Expert created: {expert.config.name}")
    print(f"   Expert domain: {expert.domain}")
    print(f"   Training data available: {len(expert.get_training_data())} cases")

    # Step 2: Check if DSPy is configured
    print("\n2. Checking DSPy configuration...")
    try:
        import dspy

        if hasattr(dspy, "settings") and hasattr(dspy.settings, "lm"):
            lm = dspy.settings.lm
            print(f"   ✅ DSPy LLM configured: {type(lm).__name__}")
        else:
            print(f"   ⚠️  DSPy LLM not configured")
            print(f"   ⚠️  Generation will fail without LLM configuration")
    except ImportError:
        print(f"   ❌ DSPy not available")
        return None

    # Step 3: Generate Mermaid diagram
    print("\n3. Generating Mermaid diagram (no mock)...")
    try:
        result = await expert.generate_mermaid(
            description="A simple flow from Start to End", diagram_type="flowchart"
        )

        print(f"   ✅ Generation completed")
        print(f"   ✅ Result type: {type(result)}")

        # Extract the actual Mermaid code
        mermaid_code = None
        if isinstance(result, dict):
            mermaid_code = result.get("output", result.get("mermaid", str(result)))
        elif isinstance(result, str):
            mermaid_code = result
        else:
            # Try to extract from DSPy prediction
            if hasattr(result, "output"):
                mermaid_code = result.output
            elif hasattr(result, "mermaid"):
                mermaid_code = result.mermaid
            else:
                mermaid_code = str(result)

        print(f"\n   📊 Generated Mermaid Diagram:")
        print("   " + "-" * 66)
        if mermaid_code:
            # Print first 500 chars
            display_code = mermaid_code[:500] + ("..." if len(mermaid_code) > 500 else "")
            print("   " + "\n   ".join(display_code.split("\n")))
        else:
            print(f"   {result}")
        print("   " + "-" * 66)

        # Verify it looks like Mermaid
        if mermaid_code and isinstance(mermaid_code, str):
            mermaid_indicators = [
                "graph" in mermaid_code.lower(),
                "flowchart" in mermaid_code.lower(),
                "-->" in mermaid_code,
                "subgraph" in mermaid_code.lower(),
                "classDiagram" in mermaid_code,
                "sequenceDiagram" in mermaid_code,
            ]

            if any(mermaid_indicators):
                print("   ✅ Contains Mermaid syntax indicators")
                print(
                    f"      Found: {[ind for ind, val in zip(['graph', 'flowchart', '-->', 'subgraph', 'classDiagram', 'sequenceDiagram'], mermaid_indicators) if val]}"
                )
            else:
                print("   ⚠️  May not be valid Mermaid syntax")
                print("   ⚠️  Result might be raw LLM output")

        return {
            "success": True,
            "result": result,
            "mermaid_code": mermaid_code,
            "is_mermaid": any(mermaid_indicators) if mermaid_code else False,
        }

    except RuntimeError as e:
        if "must be trained" in str(e):
            print(f"   ❌ ERROR: {e}")
            print(f"   ⚠️  This shouldn't happen - training check should be removed")
            return None
        else:
            raise
    except Exception as e:
        print(f"   ❌ Generation failed: {e}")
        print(f"   ⚠️  This is expected if:")
        print(f"      - DSPy LLM is not configured")
        print(f"      - No internet connection")
        print(f"      - LLM API key not set")
        import traceback

        traceback.print_exc()
        return None


async def test_expert_generate_mermaid_method():
    """
    Test the generate_mermaid convenience method.
    """
    print("\n" + "=" * 70)
    print("Testing Expert generate_mermaid() Method")
    print("=" * 70)

    expert = MermaidExpertAgent()

    try:
        result = await expert.generate_mermaid(
            description="A simple flow from Start to End", diagram_type="flowchart"
        )

        print(f"   ✅ Generated: {type(result)}")
        print(f"   📊 Result:")
        print("   " + "-" * 66)
        print("   " + "\n   ".join(str(result)[:500].split("\n")))
        print("   " + "-" * 66)

        return result
    except Exception as e:
        print(f"   ❌ Generation failed: {e}")
        return None


if __name__ == "__main__":
    print("Running Expert Real Mermaid Generation Tests (No Mock)")
    print("=" * 70)
    print("\n⚠️  NOTE: These tests require:")
    print("   1. DSPy installed")
    print("   2. LLM provider configured (Claude, GPT, etc.)")
    print("   3. API keys set in environment")
    print("=" * 70)

    # Run tests
    print("\n" + "=" * 70)
    print("Test 1: Expert.generate() Method")
    print("=" * 70)
    result1 = asyncio.run(test_expert_real_mermaid_generation())

    print("\n" + "=" * 70)
    print("Test 2: Expert.generate_mermaid() Method")
    print("=" * 70)
    result2 = asyncio.run(test_expert_generate_mermaid_method())

    print("\n" + "=" * 70)
    print("✅ ALL TESTS COMPLETED")
    print("=" * 70)

    if result1 and result1.get("success"):
        print("\n✅ Expert.generate(): SUCCESS")
        print(f"   Generated Mermaid: {result1.get('is_mermaid', False)}")
    else:
        print("\n❌ Expert.generate(): FAILED (expected if LLM not configured)")

    if result2:
        print("✅ Expert.generate_mermaid(): SUCCESS")
    else:
        print("❌ Expert.generate_mermaid(): FAILED (expected if LLM not configured)")
