"""
Functional Tests for Module-Based Configuration
================================================

Tests ACTUAL multi-agent execution with different module combinations using
Jotty's existing UnifiedLMProvider (opencode, claude-cli, openrouter).

Tests:
1. Minimal (jotty_minimal.py) + OpenCode LLM
2. Full MAS + Simple Memory + OpenCode
3. Full MAS + Dynamic Spawning + OpenCode
4. Conductor WITHOUT learning/memory + OpenCode
5. Conductor WITH learning + OpenCode
6. Conductor WITH memory + OpenCode
7. Conductor WITH ALL features + OpenCode
"""

import os
import sys
import asyncio
from pathlib import Path

# Add Jotty to path
sys.path.insert(0, str(Path(__file__).parent))

import dspy

# Use existing Claude CLI wrapper (enhanced version with JSON schema support)
print("Configuring DSPy with Claude CLI...")
try:
    # Copy the enhanced wrapper to Jotty examples if not exists
    justjot_wrapper = Path("/var/www/sites/personal/stock_market/JustJot.ai/supervisor/claude_cli_wrapper_enhanced.py")
    jotty_wrapper = Path(__file__).parent / "examples" / "claude_cli_wrapper_enhanced.py"

    if justjot_wrapper.exists() and not jotty_wrapper.exists():
        import shutil
        shutil.copy(justjot_wrapper, jotty_wrapper)
        print(f"  Copied enhanced wrapper from JustJot.ai")

    # Import the enhanced Claude CLI wrapper
    sys.path.insert(0, str(Path(__file__).parent / "examples"))
    from claude_cli_wrapper_enhanced import EnhancedClaudeCLILM

    # Configure DSPy with Claude CLI
    lm = EnhancedClaudeCLILM(model="sonnet")
    dspy.configure(lm=lm)
    print("✅ DSPy configured with Claude CLI (sonnet)")

except Exception as e:
    print(f"❌ Failed to configure Claude CLI: {e}")
    print("   Make sure: npm install -g @anthropic-ai/claude-code")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# Test 1: Minimal Config
# ============================================================================

async def test_minimal():
    """Test 1: jotty_minimal.py with Claude Sonnet"""
    print("\n" + "="*80)
    print("TEST 1: Minimal Config (jotty_minimal.py + Claude Sonnet)")
    print("="*80)

    try:
        from jotty_minimal import Orchestrator, setup_logging

        setup_logging("INFO")  # Changed to INFO to see what's happening

        orchestrator = Orchestrator()

        print("  Running: Plan Python hello world steps")
        result = await orchestrator.run(
            goal="Plan the steps to create a Python hello world program",
            max_steps=3
        )

        print(f"  Success: {result['success']}")
        print(f"  Steps: {len(result.get('results', []))}")

        # Print error if failed
        if not result['success'] and 'error' in result:
            print(f"  Error: {result['error']}")

        if result['success']:
            print("✅ Minimal config works")
            return True
        else:
            print("❌ Failed")
            return False

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

# ============================================================================
# Test 2: Full MAS + Memory
# ============================================================================

async def test_full_mas_memory():
    """Test 2: Full MAS with memory"""
    print("\n" + "="*80)
    print("TEST 2: Full MAS + Simple Memory + Claude Sonnet")
    print("="*80)

    try:
        from jotty_minimal import Orchestrator

        orchestrator = Orchestrator(
            max_spawned_per_agent=5,
            max_memory_entries=1000
        )

        print("  Running: Explain async/await")
        result = await orchestrator.run(
            goal="Explain Python async/await in 2 sentences",
            max_steps=5
        )

        print(f"  Success: {result['success']}")
        print(f"  Memory entries: {len(orchestrator.memory.entries)}")

        if result['success'] and len(orchestrator.memory.entries) > 0:
            print("✅ Full MAS + Memory works")
            return True
        else:
            print("❌ Failed")
            return False

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

# ============================================================================
# Test 3: Dynamic Spawning
# ============================================================================

def test_spawning_assessment():
    """Test 3: Complexity assessment"""
    print("\n" + "="*80)
    print("TEST 3: Complexity Assessment")
    print("="*80)

    try:
        from jotty_minimal import DynamicSpawner

        spawner = DynamicSpawner()

        simple = spawner.assess_complexity("Write hello world")
        complex_task = spawner.assess_complexity(
            "Research and analyze multiple algorithms in detail"
        )

        print(f"  Simple: {simple['complexity_score']}/5, spawn: {simple['should_spawn']}")
        print(f"  Complex: {complex_task['complexity_score']}/5, spawn: {complex_task['should_spawn']}")

        if not simple['should_spawn'] and complex_task['should_spawn']:
            print("✅ Complexity assessment works")
            return True
        else:
            print("❌ Assessment incorrect")
            return False

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

# ============================================================================
# Test 4: Conductor WITHOUT features
# ============================================================================

async def test_conductor_minimal():
    """Test 4: Conductor without learning/memory"""
    print("\n" + "="*80)
    print("TEST 4: Conductor WITHOUT learning/memory")
    print("="*80)

    try:
        from core.orchestration.conductor import MultiAgentsOrchestrator
        from core.foundation.data_structures import AgentConfig

        class SimpleSignature(dspy.Signature):
            task = dspy.InputField()
            result = dspy.OutputField()

        agents = [
            AgentConfig(
                name="worker",
                agent=dspy.ChainOfThought(SimpleSignature),
                architect_prompts=[],
                auditor_prompts=[]
            )
        ]

        conductor = MultiAgentsOrchestrator(
            actors=agents,
            enable_learning=False,
            enable_validation=False,
            enable_memory=False,
            max_steps=3
        )

        print("  Running: Count 1 to 3")
        result = await conductor.run(goal="Count from 1 to 3", context={})

        print(f"  Success: {result is not None}")

        if result:
            print("✅ Conductor (minimal) works")
            return True
        else:
            print("❌ Failed")
            return False

    except ImportError:
        print("  ⚠️ Skipped: MultiAgentsOrchestrator not available")
        return None
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

# ============================================================================
# Test 5: Conductor WITH Learning
# ============================================================================

async def test_conductor_learning():
    """Test 5: Conductor with learning"""
    print("\n" + "="*80)
    print("TEST 5: Conductor WITH learning (Q-Learning)")
    print("="*80)

    try:
        from core.orchestration.conductor import MultiAgentsOrchestrator
        from core.foundation.data_structures import AgentConfig

        class SimpleSignature(dspy.Signature):
            task = dspy.InputField()
            result = dspy.OutputField()

        agents = [
            AgentConfig(
                name="worker",
                agent=dspy.ChainOfThought(SimpleSignature),
                architect_prompts=[],
                auditor_prompts=[]
            )
        ]

        conductor = MultiAgentsOrchestrator(
            actors=agents,
            enable_learning=True,  # Enable learning
            enable_validation=False,
            enable_memory=False,
            max_steps=3
        )

        print("  Running: Simple math")
        result = await conductor.run(goal="What is 2 + 2?", context={})

        print(f"  Success: {result is not None}")

        if result:
            print("✅ Conductor + Learning works")
            return True
        else:
            print("❌ Failed")
            return False

    except ImportError:
        print("  ⚠️ Skipped: MultiAgentsOrchestrator not available")
        return None
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

# ============================================================================
# Test 6: Conductor WITH Memory
# ============================================================================

async def test_conductor_memory():
    """Test 6: Conductor with memory"""
    print("\n" + "="*80)
    print("TEST 6: Conductor WITH memory")
    print("="*80)

    try:
        from core.orchestration.conductor import MultiAgentsOrchestrator
        from core.foundation.data_structures import AgentConfig

        class SimpleSignature(dspy.Signature):
            task = dspy.InputField()
            result = dspy.OutputField()

        agents = [
            AgentConfig(
                name="worker",
                agent=dspy.ChainOfThought(SimpleSignature),
                architect_prompts=[],
                auditor_prompts=[]
            )
        ]

        conductor = MultiAgentsOrchestrator(
            actors=agents,
            enable_learning=False,
            enable_validation=False,
            enable_memory=True,  # Enable memory
            max_steps=3
        )

        print("  Running: Memory test")
        result = await conductor.run(
            goal="Remember: my name is Jotty. What is my name?",
            context={}
        )

        print(f"  Success: {result is not None}")

        if result:
            print("✅ Conductor + Memory works")
            return True
        else:
            print("❌ Failed")
            return False

    except ImportError:
        print("  ⚠️ Skipped: MultiAgentsOrchestrator not available")
        return None
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

# ============================================================================
# Test 7: Conductor WITH ALL Features
# ============================================================================

async def test_conductor_all_features():
    """Test 7: Conductor with ALL features"""
    print("\n" + "="*80)
    print("TEST 7: Conductor WITH ALL features")
    print("="*80)

    try:
        from core.orchestration.conductor import MultiAgentsOrchestrator
        from core.foundation.data_structures import AgentConfig

        class SimpleSignature(dspy.Signature):
            task = dspy.InputField()
            result = dspy.OutputField()

        agents = [
            AgentConfig(
                name="worker",
                agent=dspy.ChainOfThought(SimpleSignature),
                architect_prompts=["Check feasibility"],
                auditor_prompts=["Verify correctness"]
            )
        ]

        conductor = MultiAgentsOrchestrator(
            actors=agents,
            enable_learning=True,  # Learning
            enable_validation=True,  # Validation
            enable_memory=True,  # Memory
            max_steps=3
        )

        print("  Running: All features test")
        result = await conductor.run(
            goal="What is 1 + 1?",
            context={}
        )

        print(f"  Success: {result is not None}")

        if result:
            print("✅ Conductor + ALL features works")
            return True
        else:
            print("❌ Failed")
            return False

    except ImportError:
        print("  ⚠️ Skipped: MultiAgentsOrchestrator not available")
        return None
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

# ============================================================================
# Main Runner
# ============================================================================

async def run_all_tests():
    """Run all functional tests"""
    print("\n" + "#"*80)
    print("# FUNCTIONAL TESTS - Module Configs with LLM")
    print("#"*80)

    tests = [
        ("Minimal Config", test_minimal),
        ("Full MAS + Memory", test_full_mas_memory),
        ("Complexity Assessment", lambda: test_spawning_assessment()),
        ("Conductor (Minimal)", test_conductor_minimal),
        ("Conductor + Learning", test_conductor_learning),
        ("Conductor + Memory", test_conductor_memory),
        ("Conductor + ALL Features", test_conductor_all_features),
    ]

    results = {}
    for name, test_func in tests:
        if asyncio.iscoroutinefunction(test_func):
            result = await test_func()
        else:
            result = test_func()
        results[name] = result

    # Summary
    print("\n" + "#"*80)
    print("# SUMMARY")
    print("#"*80)

    passed = sum(1 for r in results.values() if r is True)
    failed = sum(1 for r in results.values() if r is False)
    skipped = sum(1 for r in results.values() if r is None)

    for name, result in results.items():
        if result is True:
            print(f"  ✅ {name}")
        elif result is False:
            print(f"  ❌ {name}")
        else:
            print(f"  ⚠️ {name} (skipped)")

    print(f"\nResults: {passed} passed, {failed} failed, {skipped} skipped")

    if failed > 0:
        print("\n❌ Some tests failed")
        return False
    else:
        print("\n✅ All functional tests passed!")
        return True

if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
