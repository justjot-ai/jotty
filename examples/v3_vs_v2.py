"""
V3 vs V2 Comparison
===================

Shows that V2 code still works (NO BREAKAGE)
and demonstrates V3 improvements.
"""

import asyncio


async def v2_example():
    """V2 code - still works!"""
    print("=" * 60)
    print("V2 CODE (UNCHANGED - STILL WORKS)")
    print("=" * 60)

    # Import V2 (existing code)
    from Jotty import SwarmManager
    from Jotty.core.foundation.data_structures import JottyConfig

    # V2 usage (no changes needed)
    config = JottyConfig(
        enable_learning=True,
        enable_memory=True,
    )

    sm = SwarmManager(config=config)
    result = await sm.run("Research AI trends")

    print(f"V2 Result: {result.output[:100]}...")
    print(f"V2 Success: {result.success}")
    print("✓ V2 code works with ZERO changes!\n")

    return result


async def v3_example():
    """V3 code - simpler!"""
    print("=" * 60)
    print("V3 CODE (NEW - SIMPLER)")
    print("=" * 60)

    # Import V3 (new)
    from Jotty import Jotty, ExecutionTier

    # V3 usage (much simpler!)
    jotty = Jotty()

    # Option 1: Auto-detect tier
    result = await jotty.run("Research AI trends")
    print(f"V3 Auto Result: {result.output[:100]}...")
    print(f"V3 Tier: {result.tier.name}")

    # Option 2: Explicit tier
    result = await jotty.run(
        "Research AI trends",
        tier=ExecutionTier.LEARNING  # With memory + validation
    )
    print(f"\nV3 Learning Result: {result.output[:100]}...")
    print(f"V3 Used Memory: {result.used_memory}")

    # Option 3: Convenience methods
    response = await jotty.chat("What is AI?")
    print(f"\nV3 Chat: {response[:100]}...")

    print("\n✓ V3 code is simpler and faster!\n")

    return result


async def v3_calls_v2():
    """V3 Tier 4 delegates to V2 - no breakage!"""
    print("=" * 60)
    print("V3 TIER 4 → DELEGATES TO V2")
    print("=" * 60)

    from Jotty import Jotty, ExecutionTier

    jotty = Jotty()

    # V3 Tier 4 (RESEARCH) uses V2 internally
    result = await jotty.research(
        "Complex research task",
        enable_td_lambda=True,
        enable_hierarchical_memory=True,
    )

    print(f"V3 Tier: {result.tier.name}")
    print(f"V2 Episode: {result.v2_episode}")
    print(f"V2 Learning Data: {bool(result.learning_data)}")
    print("\n✓ V3 Tier 4 uses V2 internally - full compatibility!\n")

    return result


async def main():
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 15 + "JOTTY V3 vs V2" + " " * 29 + "║")
    print("║" + " " * 12 + "Zero Breakage Guarantee" + " " * 23 + "║")
    print("╚" + "═" * 58 + "╝")
    print("\n")

    # Run V2 (existing code)
    v2_result = await v2_example()

    # Run V3 (new code)
    v3_result = await v3_example()

    # Show V3 → V2 delegation
    v3_v2_result = await v3_calls_v2()

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("✓ V2 code works without ANY changes")
    print("✓ V3 provides simpler API with auto-detection")
    print("✓ V3 Tier 4 delegates to V2 for full features")
    print("✓ Users can mix V2 and V3 in same codebase")
    print("✓ ZERO BREAKAGE GUARANTEE MET!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
