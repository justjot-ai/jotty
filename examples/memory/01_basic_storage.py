"""
Example 1: Basic Memory Storage and Retrieval

Demonstrates:
- Storing memories at different levels
- Retrieving memories by relevance
- Checking memory system status
"""

import asyncio

from Jotty.core.intelligence.memory import get_memory_system


async def main():
    # Get the memory system (singleton)
    memory = get_memory_system()

    print("=== Storing Memories ===\n")

    # Store an episodic memory (specific event, fast decay)
    mem_id_1 = memory.store(
        content="Meeting with Product team on Feb 15, 2026 - discussed Q1 roadmap",
        level="episodic",
        goal="planning",
        metadata={"date": "2026-02-15", "team": "product"},
    )
    print(f"✅ Stored episodic memory: {mem_id_1}")

    # Store a semantic memory (general knowledge, medium decay)
    mem_id_2 = memory.store(
        content="Python best practice: Use type hints for better code clarity",
        level="semantic",
        goal="coding",
        metadata={"language": "python", "topic": "best-practices"},
    )
    print(f"✅ Stored semantic memory: {mem_id_2}")

    # Store a procedural memory (how-to knowledge)
    mem_id_3 = memory.store(
        content="To deploy: 1) Run tests, 2) Build Docker image, 3) Push to registry, 4) Update k8s",
        level="procedural",
        goal="devops",
        metadata={"process": "deployment"},
    )
    print(f"✅ Stored procedural memory: {mem_id_3}")

    print("\n=== Retrieving Memories ===\n")

    # Retrieve relevant memories for a query
    results = memory.retrieve(
        query="What was discussed in the last team meeting?", goal="planning", top_k=5
    )

    print(f"Found {len(results)} relevant memories:\n")
    for i, result in enumerate(results, 1):
        print(f"{i}. [{result.level}] {result.content}")
        print(f"   Relevance: {result.relevance:.2f}")
        print(f"   Memory ID: {result.memory_id}\n")

    print("=== Memory System Status ===\n")

    # Check status
    status = memory.status()
    print(f"Backend: {status['backend']}")
    print(f"Total memories: {status.get('total_memories', 'N/A')}")
    print(f"Operations: {status.get('operations', {})}")

    print("\n✅ Example complete!")


if __name__ == "__main__":
    asyncio.run(main())
