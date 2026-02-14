#!/usr/bin/env python3
"""
Distributed Tracing Example
============================

Shows how to use distributed tracing for multi-swarm execution.
"""

import asyncio

async def main():
    from Jotty.core.observability import get_distributed_tracer
    from Jotty.core.orchestration import SwarmAdapter, get_multi_swarm_coordinator, MergeStrategy

    print("\n" + "="*70)
    print("EXAMPLE: Distributed Tracing")
    print("="*70 + "\n")

    # Create tracer
    tracer = get_distributed_tracer("example-service")

    print("Distributed Tracer initialized")
    print(f"  Service: {tracer.service_name}")
    print()

    # Create coordinator
    coordinator = get_multi_swarm_coordinator()

    # Example: Trace multi-swarm execution
    print("Executing multi-swarm with distributed tracing...\n")

    with tracer.trace("multi_swarm_research") as trace_id:
        print(f"🔍 Trace ID: {trace_id}")

        # Inject headers for downstream propagation
        headers = tracer.inject_headers(trace_id)
        print(f"📨 Injected headers: {list(headers.keys())}")
        print(f"   traceparent: {headers['traceparent'][:50]}...")
        print(f"   x-jotty-service: {headers['x-jotty-service']}")
        print()

        # Create swarms
        swarms = SwarmAdapter.quick_swarms([
            ("Researcher", "Research AI trends. 1 sentence."),
            ("Analyst", "Analyze AI trends. 1 sentence."),
        ])

        # Execute with tracing
        result = await coordinator.execute_parallel(
            swarms=swarms,
            task="What are AI trends in 2026?",
            merge_strategy=MergeStrategy.VOTING
        )

        print(f"✅ Execution complete")
        print(f"   Result: {result.output[:80]}...")
        print()

        # Get trace context
        context = tracer.get_context(trace_id)
        print("📊 Trace Context:")
        print(f"   Service: {context.get('service')}")
        print(f"   Operation: {context.get('operation')}")
        print(f"   Duration: {context.get('duration_ms', 0):.0f}ms")
        print()

    print("="*70)
    print("✅ Example complete!")
    print("="*70 + "\n")

    print("Key Takeaways:")
    print("  1. Trace ID propagates through all operations")
    print("  2. W3C traceparent header for downstream services")
    print("  3. Full context available for debugging")
    print("  4. Works with existing monitoring tools")
    print()


if __name__ == '__main__':
    asyncio.run(main())
