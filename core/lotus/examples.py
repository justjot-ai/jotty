"""
LOTUS Optimization Examples

Demonstrates how to use the LOTUS optimization layer for cost-efficient LLM operations.
"""

import asyncio
from typing import List


# =============================================================================
# Example 1: Basic LOTUS Optimizer Usage
# =============================================================================

async def example_basic_optimizer():
    """
    Basic usage of LotusOptimizer for cost-efficient LLM operations.
    """
    from Jotty.core.lotus import LotusOptimizer, LotusConfig

    # Create optimizer with default config
    config = LotusConfig()
    optimizer = LotusOptimizer(config)

    # Sample data
    documents = [
        "I love this product! Great quality.",
        "Terrible experience, would not recommend.",
        "Average product, nothing special.",
        "Best purchase I ever made!",
        "Complete waste of money.",
    ]

    # Execute filter operation with all optimizations
    positive_docs = await optimizer.execute_filter(
        items=documents,
        condition="Keep only positive reviews",
        agent="review_filter",
    )

    print(f"Positive reviews: {positive_docs}")

    # Get optimization stats
    stats = optimizer.get_stats()
    print(f"Cache hits: {stats['cache']['hits']}")
    print(f"Proxy resolution rate: {stats['cascade']['proxy_rate']:.1%}")

    # Get cost savings
    savings = optimizer.get_savings_estimate()
    print(f"Estimated savings: ${savings['total_savings']:.4f}")


# =============================================================================
# Example 2: Using Semantic Operators (LOTUS-style API)
# =============================================================================

async def example_semantic_operators():
    """
    Using LOTUS-style semantic operators for declarative data processing.
    """
    from Jotty.core.lotus import SemanticDataFrame, LotusConfig

    # Sample data
    articles = [
        {"title": "AI Advances in 2024", "content": "Machine learning continues to..."},
        {"title": "Climate Change Report", "content": "Global temperatures have..."},
        {"title": "New AI Model Released", "content": "OpenAI announced..."},
        {"title": "Sports Update", "content": "The championship game..."},
        {"title": "Tech Industry Layoffs", "content": "Several companies..."},
    ]

    # Create SemanticDataFrame
    config = LotusConfig()
    sdf = SemanticDataFrame(articles, config)

    # Chain operations (lazy evaluation)
    result = await (
        sdf
        .sem_filter("about artificial intelligence or machine learning")
        .sem_map("summarize in one sentence")
        .sem_topk(3, "most informative about AI trends")
        .execute()
    )

    print(f"Top AI articles: {result.data}")
    print(f"Operations: {len(result.stats['operations'])}")
    print(f"Cache hits: {result.cache_hits}")


# =============================================================================
# Example 3: Model Cascade for Cost Optimization
# =============================================================================

async def example_model_cascade():
    """
    Demonstrates model cascade for 10x cost reduction.
    """
    from Jotty.core.lotus import ModelCascade, LotusConfig

    config = LotusConfig()
    cascade = ModelCascade(config)

    # Items to classify
    items = [
        "This is clearly spam",           # Easy - proxy can handle
        "Buy now! Limited offer!",        # Easy
        "Meeting tomorrow at 3pm",        # Easy - clearly not spam
        "Congratulations, you won!!!",    # Ambiguous - may need oracle
        "Please review the attached doc", # Moderate
    ]

    def prompt_fn(item):
        return f"Is this spam? Answer YES or NO with confidence.\n\nText: {item}"

    def parse_fn(response):
        is_spam = "yes" in response.lower()
        confidence = 0.9 if "confident" in response.lower() else 0.7
        return is_spam, confidence

    # Execute with cascade
    results = await cascade.execute("classify", items, prompt_fn, parse_fn)

    # Analyze tier usage
    stats = cascade.get_stats()
    print(f"Total items: {stats['total_items']}")
    print(f"Proxy resolved: {stats['proxy_resolved']} ({stats['proxy_rate']:.1%})")
    print(f"Oracle resolved: {stats['oracle_resolved']}")
    print(f"Cost savings: ${stats['cost_savings']:.4f}")


# =============================================================================
# Example 4: Semantic Cache for Repeat Queries
# =============================================================================

async def example_semantic_cache():
    """
    Demonstrates semantic caching for zero-cost repeat queries.
    """
    from Jotty.core.lotus import SemanticCache, LotusConfig

    config = LotusConfig()
    cache = SemanticCache(config)

    # Simulate repeated queries
    instruction = "Summarize this text"
    content = "The quick brown fox jumps over the lazy dog."

    # First query - cache miss, computes result
    async def compute():
        return "A fox jumps over a dog."  # Simulated LLM result

    result1 = await cache.get_or_compute_async(instruction, content, compute)
    print(f"First query result: {result1}")

    # Second query (same instruction + content) - cache hit, free!
    result2 = await cache.get_or_compute_async(instruction, content, compute)
    print(f"Second query result: {result2}")

    # Check stats
    stats = cache.get_stats()
    print(f"Cache hits: {stats['hits']}")
    print(f"Cache misses: {stats['misses']}")
    print(f"Hit rate: {stats['hit_rate']:.1%}")


# =============================================================================
# Example 5: Adaptive Validation
# =============================================================================

async def example_adaptive_validation():
    """
    Demonstrates adaptive validation to skip unnecessary checks.
    """
    from Jotty.core.lotus import AdaptiveValidator

    validator = AdaptiveValidator(
        skip_threshold=0.95,
        sample_rate=0.10,
        min_samples=5,
    )

    agent = "my_agent"
    operation = "filter"

    # Simulate building up trust
    for i in range(10):
        # Most validations succeed
        validator.record_result(agent, operation, success=(i % 10 != 0))

    # Now check if we can skip
    decision = validator.should_validate(agent, operation)
    print(f"Should validate: {decision.should_validate}")
    print(f"Reason: {decision.reason}")
    print(f"Confidence: {decision.confidence:.1%}")

    # Get stats
    stats = validator.get_stats()
    print(f"Skip rate: {stats['skip_rate']:.1%}")
    print(f"Trusted agents: {validator.get_trusted_agents()}")


# =============================================================================
# Example 6: Enhancing Existing SwarmManager
# =============================================================================

def example_enhance_swarm():
    """
    Shows how to add LOTUS optimization to existing SwarmManager.
    """
    # Option 1: Enable during initialization
    code1 = '''
from Jotty.core.orchestration.swarm_manager import SwarmManager

# LOTUS is enabled by default
swarm = SwarmManager(
    agents="analyze sales data",  # Zero-config
    enable_lotus=True,            # Enable LOTUS optimization (default)
)

# Run with optimizations applied automatically
result = await swarm.run(goal="Analyze Q4 sales trends")

# Check optimization stats
print(swarm.get_lotus_stats())
print(swarm.get_lotus_savings())
'''

    # Option 2: Enhance existing swarm
    code2 = '''
from Jotty.core.lotus.integration import enhance_swarm_manager

# Existing swarm without LOTUS
swarm = SwarmManager(agents=[...], enable_lotus=False)

# Add LOTUS optimization later
enhanced_swarm = enhance_swarm_manager(swarm)

# Now has optimization
print(enhanced_swarm.lotus_stats())
'''

    print("Option 1 - Enable during init:")
    print(code1)
    print("\nOption 2 - Enhance existing:")
    print(code2)


# =============================================================================
# Run Examples
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("LOTUS Optimization Examples")
    print("=" * 60)

    # Note: These examples require DSPy configured with an LM
    # For demonstration, we show the patterns

    print("\n1. Basic Optimizer Usage")
    print("-" * 40)
    # asyncio.run(example_basic_optimizer())

    print("\n2. Semantic Operators")
    print("-" * 40)
    # asyncio.run(example_semantic_operators())

    print("\n3. Model Cascade")
    print("-" * 40)
    # asyncio.run(example_model_cascade())

    print("\n4. Semantic Cache")
    print("-" * 40)
    asyncio.run(example_semantic_cache())

    print("\n5. Adaptive Validation")
    print("-" * 40)
    asyncio.run(example_adaptive_validation())

    print("\n6. Enhancing SwarmManager")
    print("-" * 40)
    example_enhance_swarm()

    print("\n" + "=" * 60)
    print("Examples complete!")
    print("=" * 60)
