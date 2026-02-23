"""
A/B/C Test: PlantUML Knowledge Distillation
Haiku baseline vs Haiku+lessons (from web research + teacher) vs Sonnet
"""

import asyncio
import os
import re
import sys
import time

sys.path.insert(0, str(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.infrastructure.foundation.dspy_init import load_api_keys

load_api_keys()

import anthropic

HAIKU = "claude-3-haiku-20240307"
SONNET = "claude-sonnet-4-20250514"

TASKS = [
    {
        "name": "Deployment Diagram (K8s)",
        "prompt": (
            "Generate a PlantUML deployment diagram for a Kubernetes-based microservices architecture. "
            "Show: cloud 'AWS' containing node 'EKS Cluster' containing "
            "node 'Node Pool' with components: API Gateway, Auth Service, Order Service, Payment Service. "
            "Outside cluster: database 'RDS PostgreSQL', database 'ElastiCache Redis', "
            "cloud 'Stripe API' (external). Show connections with protocols (HTTPS, gRPC, TCP). "
            "Use proper PlantUML syntax: @startuml/@enduml, node, component, database, cloud keywords. "
            "Output ONLY the PlantUML code."
        ),
    },
    {
        "name": "Sequence Diagram (Saga)",
        "prompt": (
            "Generate a PlantUML sequence diagram for a distributed transaction (Saga pattern): "
            "OrderService -> PaymentService -> InventoryService -> ShippingService. "
            "Show: happy path with all services succeeding, "
            "then an alt block where InventoryService fails, triggering compensating transactions "
            "(ShippingService.cancelShipment, PaymentService.refund, OrderService.cancelOrder). "
            "Use activate/deactivate, return arrows with HTTP codes (200, 500), "
            "group block for 'Compensation', note right with business rules, colored notes. "
            "Use @startuml/@enduml. Output ONLY the PlantUML code."
        ),
    },
    {
        "name": "Class Diagram (Patterns)",
        "prompt": (
            "Generate a PlantUML class diagram for an e-commerce system combining "
            "Strategy + Factory patterns. Include: "
            "interface PaymentStrategy <<interface>> with +pay(amount: BigDecimal): boolean, "
            "CreditCardPayment, PayPalPayment, CryptoPayment implementing it (..|>), "
            "PaymentFactory with +createPayment(type: String): PaymentStrategy, "
            "Order class with -items: List<OrderItem>, -total: BigDecimal, -strategy: PaymentStrategy, "
            "OrderItem with -product: Product, -quantity: int, -price: BigDecimal, "
            "Product with -id: long, -name: String, -price: BigDecimal, -category: String. "
            "Show multiplicities (1..*, 0..1), proper visibility (+/-/#/~). "
            "Use @startuml/@enduml. Output ONLY the PlantUML code."
        ),
    },
    {
        "name": "Activity Diagram (Swimlanes)",
        "prompt": (
            "Generate a PlantUML activity diagram (new/beta syntax) for an employee onboarding process. "
            "Use swimlanes: |HR|, |IT|, |Manager|, |Employee|. "
            "Flow: HR creates account -> IT provisions laptop AND email (fork/parallel) -> "
            "Manager assigns buddy -> Employee completes orientation -> "
            "if (background check passed?) then (yes) -> confirmed, else (no) -> terminated. "
            "Use :action; syntax, start/stop, if/then/else, fork/fork again/end fork. "
            "Use @startuml/@enduml. Output ONLY the PlantUML code."
        ),
    },
]


def get_lessons_for_task(task_prompt: str) -> str:
    """Retrieve sub-domain-specific lessons via hierarchical retrieval."""
    from core.intelligence.learning.learning_service import LearningService

    ls = LearningService()
    lessons = ls.retrieve_distilled_lessons(
        domain="plantuml",
        goal=task_prompt,
        top_k=5,
    )
    if not lessons:
        return ""
    lines = [f"{i+1}. {l['lesson']}" for i, l in enumerate(lessons)]
    domain_tags = set(l.get("domain", "") for l in lessons)
    return (
        f"DOMAIN LESSONS (learned from web research + expert evaluation, sub-domains: {', '.join(domain_tags)}):\n"
        + "\n".join(lines)
    )


def extract_plantuml(text: str) -> str:
    """Extract PlantUML code from response."""
    match = re.search(r"@startuml.*?@enduml", text, re.DOTALL)
    if match:
        return match.group(0).strip()
    match = re.search(r"```(?:plantuml|puml)?\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def analyze_plantuml(code: str, task_name: str) -> dict:
    """Analyze PlantUML code quality."""
    lines = [l for l in code.split("\n") if l.strip()]
    result = {
        "total_lines": len(lines),
        "total_chars": len(code),
        "has_startuml": "@startuml" in code,
        "has_enduml": "@enduml" in code,
    }

    if "deployment" in task_name.lower():
        result["nodes"] = len(re.findall(r"\bnode\b", code))
        result["components"] = len(re.findall(r"\bcomponent\b", code))
        result["databases"] = len(re.findall(r"\bdatabase\b", code))
        result["clouds"] = len(re.findall(r"\bcloud\b", code))
        result["connections"] = len(re.findall(r"--|->|<-|\.\.>|<\.\.|\.\.", code))
        result["protocols"] = len(re.findall(r"HTTPS|gRPC|TCP|HTTP", code))
        expected = [
            "API Gateway",
            "Auth Service",
            "Order Service",
            "Payment Service",
            "PostgreSQL",
            "Redis",
            "Stripe",
        ]
        result["expected_components"] = sum(1 for e in expected if e.lower() in code.lower())
        result["expected_total"] = len(expected)

    elif "sequence" in task_name.lower():
        result["participants"] = len(re.findall(r"\bparticipant\b|\bactor\b", code))
        result["activates"] = len(re.findall(r"\bactivate\b", code))
        result["deactivates"] = len(re.findall(r"\bdeactivate\b", code))
        result["alt_blocks"] = len(re.findall(r"\balt\b", code))
        result["group_blocks"] = len(re.findall(r"\bgroup\b", code))
        result["notes"] = len(re.findall(r"\bnote\b", code, re.IGNORECASE))
        result["return_arrows"] = len(re.findall(r"<--|<-|return", code))
        result["http_codes"] = len(re.findall(r"\b200\b|\b500\b|\b302\b|\b401\b", code))
        expected = ["OrderService", "PaymentService", "InventoryService", "ShippingService"]
        result["services_found"] = sum(1 for e in expected if e in code)

    elif "class" in task_name.lower():
        result["classes"] = len(re.findall(r"\bclass\b", code))
        result["interfaces"] = len(re.findall(r"\binterface\b|<<interface>>", code))
        result["implements"] = len(re.findall(r"\.\.\|>", code))
        result["extends"] = len(re.findall(r"--\|>", code))
        result["associations"] = len(re.findall(r"--|->|o--|\\*--", code))
        result["visibility_markers"] = len(re.findall(r"[+\-#~]\w", code))
        result["multiplicities"] = len(re.findall(r'"[01\*].*?"|\b1\.\.\*\b|\b0\.\.1\b', code))
        result["methods"] = len(re.findall(r"[+\-#~]\w+\(", code))
        expected = [
            "PaymentStrategy",
            "CreditCardPayment",
            "PayPalPayment",
            "CryptoPayment",
            "PaymentFactory",
            "Order",
            "OrderItem",
            "Product",
        ]
        result["classes_found"] = sum(1 for e in expected if e in code)
        result["classes_expected"] = len(expected)

    elif "activity" in task_name.lower():
        result["swimlanes"] = len(set(re.findall(r"\|(\w+)\|", code)))
        result["actions"] = len(re.findall(r":.*?;", code))
        result["if_blocks"] = len(re.findall(r"\bif\b", code))
        result["forks"] = len(re.findall(r"\bfork\b", code))
        result["has_start"] = "start" in code.lower()
        result["has_stop"] = "stop" in code.lower() or "end" in code.lower()
        expected_lanes = ["HR", "IT", "Manager", "Employee"]
        result["lanes_found"] = sum(1 for e in expected_lanes if f"|{e}|" in code)
        result["lanes_expected"] = len(expected_lanes)

    return result


async def call_claude(model: str, prompt: str, system: str = "") -> tuple:
    client = anthropic.Anthropic()
    messages = [{"role": "user", "content": prompt}]
    kwargs = {"model": model, "max_tokens": 4096, "messages": messages}
    if system:
        kwargs["system"] = system
    t0 = time.time()
    response = client.messages.create(**kwargs)
    elapsed = time.time() - t0
    text = response.content[0].text
    return text, elapsed, response.usage.input_tokens, response.usage.output_tokens


async def main():
    print("=" * 70)
    print("PLANTUML KNOWLEDGE DISTILLATION TEST")
    print("Can Haiku + web-research lessons match Sonnet?")
    print("=" * 70)

    COSTS = {
        HAIKU: {"input": 0.25, "output": 1.25},
        SONNET: {"input": 3.0, "output": 15.0},
    }

    all_results = {}

    for task in TASKS:
        print(f"\n{'─' * 70}")
        print(f"TASK: {task['name']}")
        print(f"{'─' * 70}")

        task_results = {}

        # A: Haiku baseline
        print("\n  [A] Haiku baseline (no lessons)...")
        raw_a, time_a, in_a, out_a = await call_claude(HAIKU, task["prompt"])
        code_a = extract_plantuml(raw_a)
        analysis_a = analyze_plantuml(code_a, task["name"])
        cost_a = (in_a * COSTS[HAIKU]["input"] + out_a * COSTS[HAIKU]["output"]) / 1_000_000
        print(f"      Time: {time_a:.1f}s | Cost: ${cost_a:.5f}")
        task_results["haiku_baseline"] = {"code": code_a, "analysis": analysis_a, "cost": cost_a}

        # B: Haiku + lessons
        task_lessons = get_lessons_for_task(task["prompt"])
        print(f"  [B] Haiku + web-research lessons...")
        if task_lessons:
            for line in task_lessons.split("\n")[:4]:
                print(f"      {line}")
        raw_b, time_b, in_b, out_b = await call_claude(HAIKU, task["prompt"], system=task_lessons)
        code_b = extract_plantuml(raw_b)
        analysis_b = analyze_plantuml(code_b, task["name"])
        cost_b = (in_b * COSTS[HAIKU]["input"] + out_b * COSTS[HAIKU]["output"]) / 1_000_000
        print(f"      Time: {time_b:.1f}s | Cost: ${cost_b:.5f}")
        task_results["haiku_lessons"] = {"code": code_b, "analysis": analysis_b, "cost": cost_b}

        # C: Sonnet
        print(f"  [C] Sonnet (gold standard)...")
        raw_c, time_c, in_c, out_c = await call_claude(SONNET, task["prompt"])
        code_c = extract_plantuml(raw_c)
        analysis_c = analyze_plantuml(code_c, task["name"])
        cost_c = (in_c * COSTS[SONNET]["input"] + out_c * COSTS[SONNET]["output"]) / 1_000_000
        print(f"      Time: {time_c:.1f}s | Cost: ${cost_c:.5f}")
        task_results["sonnet"] = {"code": code_c, "analysis": analysis_c, "cost": cost_c}

        all_results[task["name"]] = task_results

        # Comparison table
        print(f"\n  {'Metric':<25} {'Haiku':>10} {'Haiku+Learn':>12} {'Sonnet':>10}")
        print(f"  {'─' * 57}")
        for key in analysis_a:
            va = analysis_a[key]
            vb = analysis_b.get(key, "—")
            vc = analysis_c.get(key, "—")
            if isinstance(va, bool):
                va, vb, vc = "✓" if va else "✗", "✓" if vb else "✗", "✓" if vc else "✗"
            print(f"  {key:<25} {str(va):>10} {str(vb):>12} {str(vc):>10}")

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    total_cost = {"haiku_baseline": 0, "haiku_lessons": 0, "sonnet": 0}
    for results in all_results.values():
        for variant in total_cost:
            total_cost[variant] += results[variant]["cost"]

    print(f"\n  {'':25} {'Haiku':>10} {'Haiku+Learn':>12} {'Sonnet':>10}")
    print(f"  {'─' * 57}")
    print(
        f"  {'Total cost':25} ${total_cost['haiku_baseline']:.5f}  ${total_cost['haiku_lessons']:.5f}  ${total_cost['sonnet']:.5f}"
    )
    print(
        f"  {'Cost vs Sonnet':25} {total_cost['haiku_baseline']/total_cost['sonnet']:.1%}       {total_cost['haiku_lessons']/total_cost['sonnet']:.1%}       100.0%"
    )

    # Quality score per task
    print(f"\n  QUALITY SCORECARD:")
    for task_name, results in all_results.items():
        scores = {}
        for variant in ["haiku_baseline", "haiku_lessons", "sonnet"]:
            a = results[variant]["analysis"]
            total_metrics = len([v for v in a.values() if isinstance(v, (int, float, bool))])
            positive = sum(
                1
                for v in a.values()
                if (isinstance(v, bool) and v) or (isinstance(v, (int, float)) and v > 0)
            )
            scores[variant] = positive / max(total_metrics, 1)
        print(
            f"  {task_name:<30} Haiku={scores['haiku_baseline']:.0%}  Haiku+Learn={scores['haiku_lessons']:.0%}  Sonnet={scores['sonnet']:.0%}"
        )

    # Save outputs
    for task_name, results in all_results.items():
        slug = task_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        for variant in ["haiku_baseline", "haiku_lessons", "sonnet"]:
            path = f"/tmp/plantuml_ab_{slug}_{variant}.puml"
            with open(path, "w") as f:
                f.write(results[variant]["code"])

    print(f"\n  Output files: /tmp/plantuml_ab_*.puml")


if __name__ == "__main__":
    asyncio.run(main())
