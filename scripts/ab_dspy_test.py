"""A/B/C test: Haiku baseline vs Haiku+DSPy-optimized vs Sonnet."""

import os
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
from scripts.dspy_plantuml_optimization import (
    PlantUMLGenerator,
    analyze_plantuml,
)

HAIKU = "claude-3-haiku-20240307"
SONNET = "claude-sonnet-4-20250514"

COSTS = {
    HAIKU: {"input": 0.25, "output": 1.25},
    SONNET: {"input": 3.0, "output": 15.0},
}

TASKS = [
    {
        "name": "Deployment (K8s)",
        "description": (
            "Generate a PlantUML deployment diagram for a Kubernetes "
            "microservices architecture with API Gateway, Auth, Order, "
            "Payment services inside an EKS cluster, plus RDS PostgreSQL, "
            "ElastiCache Redis, and external Stripe API. Show connections "
            "with protocols."
        ),
        "diagram_type": "deployment",
    },
    {
        "name": "Sequence (Saga)",
        "description": (
            "Generate a PlantUML sequence diagram for a Saga pattern "
            "distributed transaction: OrderService -> PaymentService -> "
            "InventoryService -> ShippingService. Show happy path, then "
            "alt block where InventoryService fails triggering compensating "
            "transactions. Use activate/deactivate, group blocks, return "
            "arrows with HTTP codes, and notes."
        ),
        "diagram_type": "sequence",
    },
    {
        "name": "Class (Strategy+Factory)",
        "description": (
            "Generate a PlantUML class diagram for an e-commerce system "
            "with Strategy+Factory patterns: PaymentStrategy interface with "
            "pay() method, CreditCardPayment/PayPalPayment/CryptoPayment "
            "implementing it, PaymentFactory, Order with items and strategy, "
            "OrderItem with product/quantity/price, Product with id/name/price. "
            "Show visibility modifiers, multiplicities, and implements (..|>) "
            "relationships."
        ),
        "diagram_type": "class",
    },
    {
        "name": "Activity (Swimlanes)",
        "description": (
            "Generate a PlantUML activity diagram with swimlanes |HR|, |IT|, "
            "|Manager|, |Employee| for an employee onboarding process: "
            "HR creates account, IT provisions laptop AND email in parallel "
            "(fork), Manager assigns buddy, Employee completes orientation, "
            "then if/else for background check pass/fail."
        ),
        "diagram_type": "activity",
    },
]


def score_analysis(a: dict, dtype: str) -> float:
    s = 0.0
    if a.get("valid"):
        s += 20
    if a.get("lines", 0) > 10:
        s += 20
    if dtype == "deployment":
        s += min(a.get("nodes", 0), 4) * 5
        s += min(a.get("components", 0), 6) * 3.3
        s += min(a.get("protocols", 0), 3) * 6.7
    elif dtype == "sequence":
        s += min(a.get("participants", 0), 4) * 5
        s += min(a.get("activates", 0), 3) * 6.7
        s += min(a.get("alt_blocks", 0), 2) * 10
        s += min(a.get("notes", 0), 2) * 5
    elif dtype == "class":
        s += min(a.get("classes", 0), 5) * 4
        s += min(a.get("interfaces", 0), 2) * 10
        s += min(a.get("implements", 0), 2) * 10
        s += min(a.get("visibility", 0), 10) * 2
    elif dtype == "activity":
        s += min(a.get("swimlanes", 0), 4) * 5
        s += min(a.get("forks", 0), 2) * 10
        s += min(a.get("ifs", 0), 2) * 10
        s += min(a.get("actions", 0), 8) * 2.5
    return min(s, 100)


def main():
    # Load optimized module
    save_path = os.path.expanduser("~/jotty/learning/dspy_optimized/plantuml_generator.json")
    optimized = PlantUMLGenerator()
    optimized.load(save_path)
    student_lm = UnifiedLMProvider.create_lm(provider="anthropic", model="haiku")

    client = anthropic.Anthropic()

    print("=" * 70)
    print("A/B/C TEST: Haiku vs Haiku+DSPy-optimized vs Sonnet")
    print("=" * 70)

    scorecard = {"haiku": [], "dspy": [], "sonnet": []}
    total_cost = {"haiku": 0.0, "dspy": 0.0, "sonnet": 0.0}

    for task in TASKS:
        print(f"\n{'=' * 70}")
        print(f"TASK: {task['name']}")
        print(f"{'=' * 70}")

        prompt = (
            f"{task['description']} "
            f"Use proper PlantUML syntax with @startuml/@enduml. "
            f"Output ONLY the PlantUML code."
        )

        # A: Haiku baseline
        print("  [A] Haiku baseline...")
        t0 = time.time()
        resp_a = client.messages.create(
            model=HAIKU,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        time_a = time.time() - t0
        raw_a = resp_a.content[0].text
        cost_a = (
            resp_a.usage.input_tokens * COSTS[HAIKU]["input"]
            + resp_a.usage.output_tokens * COSTS[HAIKU]["output"]
        ) / 1e6
        total_cost["haiku"] += cost_a
        match_a = re.search(r"@startuml.*?@enduml", raw_a, re.DOTALL)
        code_a = match_a.group(0) if match_a else raw_a
        analysis_a = analyze_plantuml(code_a, task["diagram_type"])
        print(f"      {time_a:.1f}s | ${cost_a:.5f} | {analysis_a['lines']} lines")

        # B: Haiku + DSPy optimized module
        print("  [B] Haiku + DSPy optimized...")
        t0 = time.time()
        with dspy.context(lm=student_lm):
            result_b = optimized(
                description=task["description"],
                diagram_type=task["diagram_type"],
            )
        time_b = time.time() - t0
        code_b = result_b.plantuml_code if hasattr(result_b, "plantuml_code") else ""
        if "```" in code_b:
            code_b = re.sub(r"```(?:plantuml|puml)?\s*", "", code_b).strip().rstrip("`")
        match_b = re.search(r"@startuml.*?@enduml", code_b, re.DOTALL)
        if match_b:
            code_b = match_b.group(0)
        analysis_b = analyze_plantuml(code_b, task["diagram_type"])
        cost_b = cost_a * 3
        total_cost["dspy"] += cost_b
        print(f"      {time_b:.1f}s | ~${cost_b:.5f} | {analysis_b['lines']} lines")

        # C: Sonnet
        print("  [C] Sonnet (gold standard)...")
        t0 = time.time()
        resp_c = client.messages.create(
            model=SONNET,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        time_c = time.time() - t0
        raw_c = resp_c.content[0].text
        cost_c = (
            resp_c.usage.input_tokens * COSTS[SONNET]["input"]
            + resp_c.usage.output_tokens * COSTS[SONNET]["output"]
        ) / 1e6
        total_cost["sonnet"] += cost_c
        match_c = re.search(r"@startuml.*?@enduml", raw_c, re.DOTALL)
        code_c = match_c.group(0) if match_c else raw_c
        analysis_c = analyze_plantuml(code_c, task["diagram_type"])
        print(f"      {time_c:.1f}s | ${cost_c:.5f} | {analysis_c['lines']} lines")

        # Comparison
        print(f"\n  {'Metric':<20} {'Haiku':>8} {'DSPy':>11} {'Sonnet':>8}")
        print(f"  {'=' * 49}")
        for key in analysis_a:
            va = analysis_a[key]
            vb = analysis_b.get(key, "-")
            vc = analysis_c.get(key, "-")
            if isinstance(va, bool):
                va = "Y" if va else "N"
                vb = "Y" if vb else "N"
                vc = "Y" if vc else "N"
            print(f"  {key:<20} {str(va):>8} {str(vb):>11} {str(vc):>8}")

        sa = score_analysis(analysis_a, task["diagram_type"])
        sb = score_analysis(analysis_b, task["diagram_type"])
        sc = score_analysis(analysis_c, task["diagram_type"])
        scorecard["haiku"].append(sa)
        scorecard["dspy"].append(sb)
        scorecard["sonnet"].append(sc)
        print(f"\n  QUALITY: Haiku={sa:.0f}/100  DSPy={sb:.0f}/100  Sonnet={sc:.0f}/100")

        # Save
        slug = task["name"].lower().replace(" ", "_").replace("(", "").replace(")", "")
        for variant, code in [("haiku", code_a), ("dspy", code_b), ("sonnet", code_c)]:
            with open(f"/tmp/plantuml_dspy_{slug}_{variant}.puml", "w") as f:
                f.write(code)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    hdr = f"{'Task':<25} {'Haiku':>8} {'DSPy':>8} {'Sonnet':>8}"
    print(hdr)
    print("-" * len(hdr))
    for i, task in enumerate(TASKS):
        sh = scorecard["haiku"][i]
        sd = scorecard["dspy"][i]
        ss = scorecard["sonnet"][i]
        print(f"{task['name']:<25} {sh:>7.0f}% {sd:>7.0f}% {ss:>7.0f}%")
    print("-" * len(hdr))
    avg_h = sum(scorecard["haiku"]) / len(TASKS)
    avg_d = sum(scorecard["dspy"]) / len(TASKS)
    avg_s = sum(scorecard["sonnet"]) / len(TASKS)
    print(f"{'AVERAGE':<25} {avg_h:>7.1f}% {avg_d:>7.1f}% {avg_s:>7.1f}%")

    print(
        f"\n{'Cost':<25} ${total_cost['haiku']:.5f} ${total_cost['dspy']:.5f} ${total_cost['sonnet']:.5f}"
    )

    if total_cost["sonnet"] > 0:
        pct = total_cost["dspy"] / total_cost["sonnet"] * 100
        print(f"DSPy cost as % of Sonnet: {pct:.1f}%")
    if avg_s > 0:
        print(f"DSPy quality as % of Sonnet: {avg_d / avg_s * 100:.1f}%")

    # Value assessment
    print("\n" + "=" * 70)
    print("VALUE ASSESSMENT")
    print("=" * 70)
    if avg_d >= avg_s * 0.9:
        print("DSPy achieves >=90% of Sonnet quality at a fraction of the cost!")
    elif avg_d >= avg_s * 0.75:
        print("DSPy achieves >=75% of Sonnet quality — good value for cost savings.")
    else:
        print("DSPy still has quality gap vs Sonnet — needs more training data.")


if __name__ == "__main__":
    main()
