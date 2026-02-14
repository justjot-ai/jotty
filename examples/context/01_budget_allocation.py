"""
Example 1: Token Budget Allocation

Demonstrates:
- Creating a context manager with budget limits
- Registering critical info (never truncated)
- Adding context chunks with priorities
- Building context that fits within token limits
"""
import asyncio
from Jotty.core.context import get_context_manager
from Jotty.core.context.context_manager import ContextPriority


async def main():
    # Create context manager (max 10,000 tokens, 85% safety margin)
    ctx = get_context_manager(max_tokens=10000, safety_margin=0.85)

    print("=== Token Budget Allocation ===\n")
    print(f"Max tokens: {ctx.max_tokens}")
    print(f"Effective limit: {ctx.effective_limit} (with safety margin)")
    print()

    # Register critical info (NEVER truncated)
    ctx.register_todo("Complete Q1 report by March 1")
    ctx.register_goal("Create comprehensive analysis of sales data")
    ctx.register_critical_memory("Budget constraint: $50,000 max")

    print("✅ Registered critical info (always preserved)")
    print()

    # Add context chunks with different priorities
    ctx.add_chunk(
        content="Recent sales data shows 15% growth in Q4 2025",
        category="recent_data",
        priority=ContextPriority.HIGH
    )

    ctx.add_chunk(
        content="Historical trend: Sales typically peak in Q4, drop in Q1",
        category="historical",
        priority=ContextPriority.MEDIUM
    )

    ctx.add_chunk(
        content="Verbose logs from data processing...\n" + "x" * 5000,
        category="logs",
        priority=ContextPriority.LOW
    )

    print("✅ Added context chunks:")
    print("  - HIGH priority: Recent sales data")
    print("  - MEDIUM priority: Historical trends")
    print("  - LOW priority: Verbose logs (5000+ chars)")
    print()

    # Build final context
    result = ctx.build_context(
        system_prompt="You are a data analyst",
        user_input="Analyze sales trends and predict Q1 2026"
    )

    print("=== Budget Allocation Result ===\n")
    print(f"Truncated: {result['truncated']}")
    print(f"\nPreserved:")
    print(f"  - Task List: {result['preserved']['todo']}")
    print(f"  - Goal: {result['preserved']['goal']}")
    print(f"  - Critical memories: {result['preserved']['critical_memories']}")
    print(f"  - Chunks included: {result['preserved']['chunks_included']}/{result['preserved']['chunks_total']}")
    print(f"\nToken usage:")
    print(f"  - Total: {result['stats']['total_tokens']}")
    print(f"  - Remaining: {result['stats']['budget_remaining']}")
    print(f"  - Utilization: {(result['stats']['total_tokens'] / ctx.effective_limit * 100):.1f}%")

    print("\n=== Priority-Based Allocation ===\n")
    print("Order of inclusion:")
    print("1. CRITICAL (Task List, Goal, Memories) → Always included")
    print("2. HIGH priority chunks → Included if space")
    print("3. MEDIUM priority chunks → Included if space")
    print("4. LOW priority chunks → Included only if abundant space")
    print()
    print("If budget tight:")
    print("- CRITICAL chunks: Compressed if needed (never dropped)")
    print("- HIGH chunks: Compressed to fit")
    print("- MEDIUM/LOW chunks: Dropped")

    print("\n✅ Example complete!")


if __name__ == "__main__":
    asyncio.run(main())
