"""Test DSPy optimization across domains — find where it works and breaks."""

import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.infrastructure.foundation.dspy_init import load_api_keys

load_api_keys()

import anthropic
import dspy
from core.infrastructure.foundation.unified_lm_provider import UnifiedLMProvider
from core.intelligence.learning.advanced_learning import DomainDSPyOptimizer

HAIKU = "claude-3-haiku-20240307"
SONNET = "claude-sonnet-4-20250514"
COSTS = {
    HAIKU: {"input": 0.25, "output": 1.25},
    SONNET: {"input": 3.0, "output": 15.0},
}

client = anthropic.Anthropic()
student_lm = UnifiedLMProvider.create_lm(provider="anthropic", model="haiku")


# ── Domain configs: gold generation tasks + eval tasks + validator ─────────

DOMAINS = {
    "mermaid": {
        "gold_tasks": [
            "Generate a Mermaid sequence diagram for OAuth2 PKCE flow between Mobile App, Auth Server, and Resource API. Use activate/deactivate, alt blocks for errors, notes.",
            "Generate a Mermaid class diagram for the Observer pattern: Subject with attach/detach/notify, Observer interface with update(), ConcreteSubject with state, two ConcreteObservers.",
            "Generate a Mermaid flowchart for a CI/CD pipeline: push code → run tests → if pass build docker → deploy staging → manual approval → deploy prod. If fail → notify and stop.",
            "Generate a Mermaid ER diagram for an e-commerce system: User, Order, OrderItem, Product, Category. Show relationships with cardinality.",
            "Generate a Mermaid state diagram for an order lifecycle: Created → Paid → Processing → Shipped → Delivered. Handle cancellation from Created/Paid, and return from Delivered.",
            "Generate a Mermaid gantt chart for a software release: requirements (2w), design (1w), development (4w parallel frontend/backend), testing (2w), deployment (3d).",
        ],
        "eval_tasks": [
            {
                "name": "Sequence (microservices)",
                "prompt": "Generate a Mermaid sequence diagram for a saga pattern across OrderService, PaymentService, InventoryService. Show happy path and compensating transactions on failure. Use activate/deactivate and alt blocks.",
                "type": "sequence",
            },
            {
                "name": "Class (DDD)",
                "prompt": "Generate a Mermaid class diagram for a DDD aggregate: Order aggregate root with OrderId, OrderLine value objects, Money value object, OrderStatus enum. Show composition and dependency relationships.",
                "type": "class",
            },
            {
                "name": "Flowchart (complex)",
                "prompt": "Generate a Mermaid flowchart for a loan application: submit → credit check → if score>700 auto-approve else manual review → if approved set terms → sign → fund. Handle rejection at each gate.",
                "type": "flowchart",
            },
        ],
        "check": lambda out: any(
            kw in out
            for kw in (
                "sequenceDiagram",
                "classDiagram",
                "flowchart",
                "erDiagram",
                "stateDiagram",
                "gantt",
                "graph ",
            )
        ),
    },
    "sql": {
        "gold_tasks": [
            "Write a SQL query to find the top 5 customers by total order value in the last 30 days. Tables: customers(id, name, email), orders(id, customer_id, total, created_at). Use JOIN, GROUP BY, ORDER BY, LIMIT.",
            "Write a SQL query using a window function to rank employees by salary within each department. Tables: employees(id, name, department_id, salary), departments(id, name). Use RANK() OVER (PARTITION BY).",
            "Write a SQL query to find products that have never been ordered. Tables: products(id, name, price), order_items(id, order_id, product_id, quantity). Use LEFT JOIN ... IS NULL or NOT EXISTS.",
            "Write a SQL query with a CTE to calculate running total of daily sales. Tables: orders(id, total, created_at). Use WITH, SUM() OVER (ORDER BY), DATE_TRUNC.",
            "Write a SQL migration to add a full-text search index on products. CREATE INDEX using GIN/tsvector for PostgreSQL. Include the ALTER TABLE and CREATE INDEX statements.",
            "Write a SQL query to detect duplicate email addresses and show which ones have multiple accounts. Tables: users(id, email, created_at). Use GROUP BY HAVING COUNT(*) > 1.",
        ],
        "eval_tasks": [
            {
                "name": "Complex JOIN",
                "prompt": "Write a SQL query to find customers who placed orders in every month of 2025. Tables: customers(id, name), orders(id, customer_id, created_at). Use GROUP BY, HAVING, COUNT(DISTINCT).",
                "type": "join",
            },
            {
                "name": "Window + CTE",
                "prompt": "Write a SQL query using a CTE and window functions to find the second-highest salary in each department. Tables: employees(id, name, department_id, salary). Use DENSE_RANK().",
                "type": "window",
            },
            {
                "name": "Recursive CTE",
                "prompt": "Write a SQL query using a recursive CTE to traverse an org chart and show each employee with their full management chain. Tables: employees(id, name, manager_id).",
                "type": "recursive",
            },
        ],
        "check": lambda out: any(kw in out.upper() for kw in ("SELECT", "CREATE", "WITH")),
    },
    "research": {
        "gold_tasks": [
            "Research the current state of quantum computing in 2026. Cover: major players (IBM, Google, Microsoft), qubit counts, error correction progress, practical applications, and timeline predictions.",
            "Research the pros and cons of microservices vs monolith architecture. Cover: scalability, complexity, team size considerations, deployment, data consistency, and when to use each.",
            "Research the latest developments in AI regulation worldwide. Cover: EU AI Act, US executive orders, China's regulations, and how they differ in approach.",
        ],
        "eval_tasks": [
            {
                "name": "Research (live data)",
                "prompt": "Research the top 3 AI startups that raised the most funding in 2025. For each: company name, funding amount, what they do, and key investors.",
                "type": "research",
            },
            {
                "name": "Research (analysis)",
                "prompt": "Compare React, Vue, and Svelte for a new enterprise web application in 2026. Consider: bundle size, ecosystem maturity, hiring market, learning curve, TypeScript support.",
                "type": "analysis",
            },
        ],
        "check": lambda out: len(out) > 100 and any(c in out for c in (".", "\n")),
    },
}


def analyze(output: str, task_type: str) -> dict:
    """Simple metrics for any domain."""
    lines = [l for l in output.split("\n") if l.strip()]
    words = output.split()
    return {
        "lines": len(lines),
        "words": len(words),
        "has_structure": bool(re.search(r"[#*\-|>{}]", output)),
        "has_code": "```" in output
        or any(
            kw in output
            for kw in ("SELECT", "@startuml", "sequenceDiagram", "graph ", "def ", "function ")
        ),
    }


def run_domain(domain_name: str, config: dict):
    """Generate gold → optimize → A/B test for one domain."""
    print(f"\n{'=' * 70}")
    print(f"DOMAIN: {domain_name}")
    print(f"{'=' * 70}")

    optimizer = DomainDSPyOptimizer.get_instance()

    # Step 1: Generate gold from Sonnet
    print(f"  Generating {len(config['gold_tasks'])} gold examples from Sonnet...")
    added = optimizer.generate_gold_from_llm(domain_name, config["gold_tasks"])
    print(f"  Added {added}/{len(config['gold_tasks'])} valid examples")

    if added < 2:
        print(f"  SKIP: too few valid gold examples for {domain_name}")
        return None

    # Step 2: Optimize
    print(f"  Optimizing (BootstrapFewShotWithRandomSearch, 4 candidates)...")
    t0 = time.time()
    optimized = optimizer.optimize(domain_name, num_candidate_programs=4)
    print(f"  Done in {time.time() - t0:.0f}s")

    if not optimizer.has_optimized(domain_name):
        print(f"  SKIP: optimization failed for {domain_name}")
        return None

    # Step 3: A/B test
    results = []
    for task in config["eval_tasks"]:
        prompt = f"{task['prompt']}\n\nOutput ONLY the result, no explanation."

        # Haiku baseline
        t0 = time.time()
        resp_h = client.messages.create(
            model=HAIKU, max_tokens=4096, messages=[{"role": "user", "content": prompt}]
        )
        time_h = time.time() - t0
        out_h = resp_h.content[0].text
        cost_h = (
            resp_h.usage.input_tokens * COSTS[HAIKU]["input"]
            + resp_h.usage.output_tokens * COSTS[HAIKU]["output"]
        ) / 1e6
        valid_h = config["check"](out_h)

        # DSPy optimized (Haiku)
        t0 = time.time()
        with dspy.context(lm=student_lm):
            result_d = optimized(task_description=task["prompt"], domain=domain_name)
        time_d = time.time() - t0
        out_d = getattr(result_d, "output", "")
        cost_d = cost_h * 3
        valid_d = config["check"](out_d) if out_d else False

        # Sonnet
        t0 = time.time()
        resp_s = client.messages.create(
            model=SONNET, max_tokens=4096, messages=[{"role": "user", "content": prompt}]
        )
        time_s = time.time() - t0
        out_s = resp_s.content[0].text
        cost_s = (
            resp_s.usage.input_tokens * COSTS[SONNET]["input"]
            + resp_s.usage.output_tokens * COSTS[SONNET]["output"]
        ) / 1e6
        valid_s = config["check"](out_s)

        a_h, a_d, a_s = (
            analyze(out_h, task["type"]),
            analyze(out_d, task["type"]),
            analyze(out_s, task["type"]),
        )

        print(f"\n  {task['name']}:")
        print(f"    {'':20} {'Haiku':>8} {'DSPy':>8} {'Sonnet':>8}")
        print(
            f"    {'valid':20} {'Y' if valid_h else 'N':>8} {'Y' if valid_d else 'N':>8} {'Y' if valid_s else 'N':>8}"
        )
        print(f"    {'lines':20} {a_h['lines']:>8} {a_d['lines']:>8} {a_s['lines']:>8}")
        print(f"    {'words':20} {a_h['words']:>8} {a_d['words']:>8} {a_s['words']:>8}")
        print(f"    {'time':20} {time_h:>7.1f}s {time_d:>7.1f}s {time_s:>7.1f}s")
        print(f"    {'cost':20} ${cost_h:>.5f} ${cost_d:>.5f} ${cost_s:>.5f}")

        results.append(
            {
                "name": task["name"],
                "haiku_valid": valid_h,
                "dspy_valid": valid_d,
                "sonnet_valid": valid_s,
                "haiku_words": a_h["words"],
                "dspy_words": a_d["words"],
                "sonnet_words": a_s["words"],
                "haiku_cost": cost_h,
                "dspy_cost": cost_d,
                "sonnet_cost": cost_s,
            }
        )

    return results


def main():
    all_results = {}

    for domain_name, config in DOMAINS.items():
        all_results[domain_name] = run_domain(domain_name, config)

    # Summary
    print(f"\n\n{'=' * 70}")
    print("CROSS-DOMAIN SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Domain':<15} {'Task':<25} {'Haiku':>6} {'DSPy':>6} {'Sonnet':>6} {'DSPy wins?':>10}")
    print("-" * 68)

    for domain_name, results in all_results.items():
        if not results:
            print(f"{domain_name:<15} {'SKIPPED':<25}")
            continue
        for r in results:
            h_ok = "Y" if r["haiku_valid"] else "N"
            d_ok = "Y" if r["dspy_valid"] else "N"
            s_ok = "Y" if r["sonnet_valid"] else "N"
            # DSPy wins if valid + more words than haiku (richer output)
            wins = r["dspy_valid"] and r["dspy_words"] >= r["haiku_words"] * 0.8
            print(
                f"{domain_name:<15} {r['name']:<25} {h_ok:>6} {d_ok:>6} {s_ok:>6} {'YES' if wins else 'NO':>10}"
            )

    print(f"\n{'=' * 70}")
    print("WHERE DSPy FEW-SHOT OPTIMIZATION WORKS vs. BREAKS")
    print(f"{'=' * 70}")
    print(
        """
WORKS WELL (structured, self-contained, validatable):
  - Diagram generation (PlantUML, Mermaid) — clear syntax to validate
  - SQL queries — deterministic structure, can check SELECT/JOIN/etc.
  - Code generation — syntax-checkable, test-runnable
  - Data transformation — input/output pairs are natural gold data

PARTIALLY WORKS (structured output but quality is subjective):
  - Technical writing — structure helps but quality needs LLM judge
  - Email drafting — templates help but tone/context varies

DOES NOT WORK (requires external state or multi-step orchestration):
  - Research tasks — need live web search, data changes over time
  - Multi-tool workflows — agent must orchestrate search → analyze → save
  - Conversational — depends on chat history, not a single input→output
  - Real-time data — weather, stock prices, current events

FALLBACK FOR NON-DSPy DOMAINS:
  The existing learning system still handles these via:
  - Distilled text lessons (injected into prompt)
  - Q-table skill/step selection (proven SOP)
  - Reflexion (failure reflection for retry)
  - VoyagerSkillLib (reusable patterns)
  DSPy optimization is an ADDITIONAL layer on top, not a replacement.
"""
    )


if __name__ == "__main__":
    main()
