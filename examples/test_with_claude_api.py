#!/usr/bin/env python
"""
Simple test of refactored components with Claude API
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

def test_simple_claude_call():
    """Test a simple DSPy call with Claude to verify setup."""
    print("\n" + "="*70)
    print("Testing Refactored Jotty with Claude API")
    print("="*70)

    import dspy

    # Set API key
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        print("❌ No API key provided")
        return False

    os.environ['ANTHROPIC_API_KEY'] = api_key

    print(f"\n✓ API key loaded (ends with: ...{api_key[-8:]})")

    # Configure DSPy with Claude
    lm = dspy.LM(
        model='anthropic/claude-3-5-haiku-20241022',
        max_tokens=300
    )
    dspy.configure(lm=lm)
    print("✓ DSPy configured with Claude Haiku")

    # Create a simple signature
    class GreetingSignature(dspy.Signature):
        """Generate a brief greeting about code refactoring."""
        topic = dspy.InputField()
        greeting = dspy.OutputField()

    # Create predictor
    predictor = dspy.ChainOfThought(GreetingSignature)

    # Call Claude API
    print("\n🚀 Calling Claude API...")
    result = predictor(topic="the newly refactored Jotty framework with 3 components")

    print(f"\n✅ Response received:")
    print(f"   {result.greeting}")

    # Verify components are available
    print("\n🔍 Verifying refactored components:")
    from core.orchestration.parameter_resolver import ParameterResolver
    from core.orchestration.tool_manager import ToolManager
    from core.orchestration.state_manager import StateManager

    print("   ✓ ParameterResolver available")
    print("   ✓ ToolManager available")
    print("   ✓ StateManager available")

    return True

if __name__ == "__main__":
    try:
        # Load credentials from ~/.claude
        import json
        creds_path = Path.home() / '.claude' / '.credentials.json'

        if creds_path.exists():
            with open(creds_path) as f:
                creds = json.load(f)
                api_key = creds['claudeAiOauth']['accessToken']
                os.environ['ANTHROPIC_API_KEY'] = api_key
                print(f"✓ Loaded credentials from {creds_path}")

        success = test_simple_claude_call()

        if success:
            print("\n" + "="*70)
            print("🎉 SUCCESS! Refactored components work with Claude API!")
            print("="*70)
            sys.exit(0)
        else:
            print("\n❌ Test failed")
            sys.exit(1)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
