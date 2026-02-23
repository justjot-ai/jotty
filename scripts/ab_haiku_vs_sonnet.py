"""
A/B/C Test: Knowledge Distillation — Haiku baseline vs Haiku+lessons vs Sonnet

Tests whether distilled lessons from Sonnet probation can boost Haiku
to produce Sonnet-quality Mermaid diagrams (knowledge distillation).
"""

import asyncio
import json
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

# Complex task designed to stress-test a weaker model
TASKS = [
    {
        "name": "Complex ER Diagram",
        "prompt": (
            "Generate a Mermaid erDiagram for a hospital management system. "
            "Tables: Patient (id PK, name, dob, blood_type, insurance_id FK), "
            "Doctor (id PK, name, specialty, department_id FK, license_number), "
            "Department (id PK, name, floor, head_doctor_id FK), "
            "Appointment (id PK, patient_id FK, doctor_id FK, datetime, status, notes), "
            "Prescription (id PK, appointment_id FK, medication, dosage, duration, refills), "
            "MedicalRecord (id PK, patient_id FK, doctor_id FK, diagnosis, treatment, date, followup_date), "
            "Insurance (id PK, provider_name, plan_type, coverage_pct). "
            "Show ALL relationships with correct cardinality (one-to-many, many-to-one). "
            "Output ONLY the Mermaid code, no explanations."
        ),
    },
    {
        "name": "Complex Sequence Diagram",
        "prompt": (
            "Generate a Mermaid sequence diagram for a microservices payment flow: "
            "User → API Gateway → Auth Service (JWT validation) → Order Service → "
            "Payment Service → Stripe API → Payment Service → Order Service → "
            "Notification Service (email + push) → User. "
            "Include: alt blocks for payment success/failure, opt block for retry on timeout, "
            "par block for parallel email+push notifications, "
            "activate/deactivate for each service call, "
            "note blocks for important business rules. "
            "Output ONLY the Mermaid code, no explanations."
        ),
    },
    {
        "name": "Complex State Diagram",
        "prompt": (
            "Generate a Mermaid stateDiagram-v2 for an e-commerce order lifecycle. "
            "States: Draft, Submitted, PaymentPending, PaymentFailed, Confirmed, "
            "Processing, QualityCheck, Shipped, InTransit, OutForDelivery, Delivered, "
            "ReturnRequested, ReturnApproved, ReturnShipped, Refunded, Cancelled, Archived. "
            "Include: composite state for 'Fulfillment' containing Processing+QualityCheck+Shipped, "
            "choice pseudostate for payment result, "
            "fork/join for parallel QC and packaging, "
            "transitions with guard conditions [paymentOk], [paymentFail], [qcPass], [qcFail], "
            "and entry/exit actions where appropriate. "
            "Output ONLY the Mermaid code, no explanations."
        ),
    },
]


def get_lessons_for_task(task_prompt: str) -> str:
    """Retrieve sub-domain-specific lessons via hierarchical retrieval."""
    from core.intelligence.learning.learning_service import LearningService

    ls = LearningService()
    lessons = ls.retrieve_distilled_lessons(
        domain="mermaid",
        goal=task_prompt,
        top_k=5,
    )
    if not lessons:
        return ""
    lines = [f"{i+1}. {l['lesson']}" for i, l in enumerate(lessons)]
    domain_tags = set(l.get("domain", "") for l in lessons)
    return f"DOMAIN LESSONS (sub-domains: {', '.join(domain_tags)}):\n" + "\n".join(lines)


def extract_mermaid(text: str) -> str:
    """Extract mermaid code from response, stripping markdown fences."""
    match = re.search(r"```(?:mermaid)?\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # If no fences, assume entire response is mermaid
    return text.strip()


def analyze_mermaid(code: str, task_name: str) -> dict:
    """Analyze mermaid code quality."""
    lines = code.strip().split("\n")
    non_empty = [l for l in lines if l.strip()]

    result = {
        "total_lines": len(non_empty),
        "total_chars": len(code),
    }

    if "erDiagram" in task_name.lower() or "erDiagram" in code:
        tables = set()
        relationships = []
        fields = 0
        for line in lines:
            stripped = line.strip()
            if (
                stripped
                and not stripped.startswith("%")
                and "{" not in stripped
                and "}" not in stripped
            ):
                if "||" in stripped or "}|" in stripped or "}o" in stripped or "o{" in stripped:
                    relationships.append(stripped)
            if re.match(r"^\s+\w+\s+\w+", stripped) and "{" not in stripped and "}" not in stripped:
                fields += 1
            # Table names: lines like "    TableName {"
            m = re.match(r"^\s*(\w+)\s*\{", stripped)
            if m:
                tables.add(m.group(1))
        expected = [
            "Patient",
            "Doctor",
            "Department",
            "Appointment",
            "Prescription",
            "MedicalRecord",
            "Insurance",
        ]
        result["tables_found"] = len(tables)
        result["tables_expected"] = len(expected)
        result["tables_missing"] = [t for t in expected if t not in tables]
        result["relationships"] = len(relationships)
        result["fields"] = fields

    elif "sequence" in task_name.lower() or "sequenceDiagram" in code:
        participants = re.findall(r"participant\s+(\w+)", code)
        activates = len(re.findall(r"activate\s+", code))
        deactivates = len(re.findall(r"deactivate\s+", code))
        alt_blocks = len(re.findall(r"\balt\b", code))
        par_blocks = len(re.findall(r"\bpar\b", code))
        opt_blocks = len(re.findall(r"\bopt\b", code))
        notes = len(re.findall(r"\bNote\b", code, re.IGNORECASE))
        arrows = len(re.findall(r"->>|-->>|-\)", code))
        result["participants"] = len(set(participants))
        result["activate_pairs"] = min(activates, deactivates)
        result["alt_blocks"] = alt_blocks
        result["par_blocks"] = par_blocks
        result["opt_blocks"] = opt_blocks
        result["notes"] = notes
        result["message_arrows"] = arrows

    elif "state" in task_name.lower() or "stateDiagram" in code:
        states = set(re.findall(r"state\s+\"?(\w+)\"?", code))
        # Also count plain transitions: StateA --> StateB
        transition_states = re.findall(r"(\w+)\s*-->\s*(\w+)", code)
        for s1, s2 in transition_states:
            if s1 != "[*]":
                states.add(s1)
            if s2 != "[*]":
                states.add(s2)
        transitions = len(re.findall(r"-->", code))
        guards = len(re.findall(r"\[.*?\]", code))
        composites = len(re.findall(r"state\s+\w+\s*\{", code))
        choice = len(re.findall(r"<<choice>>|<<fork>>|<<join>>", code))
        expected_states = [
            "Draft",
            "Submitted",
            "PaymentPending",
            "PaymentFailed",
            "Confirmed",
            "Processing",
            "QualityCheck",
            "Shipped",
            "InTransit",
            "OutForDelivery",
            "Delivered",
            "ReturnRequested",
            "ReturnApproved",
            "ReturnShipped",
            "Refunded",
            "Cancelled",
            "Archived",
        ]
        result["states_found"] = len(states)
        result["states_expected"] = len(expected_states)
        result["states_missing"] = [s for s in expected_states if s not in states]
        result["transitions"] = transitions
        result["guard_conditions"] = guards
        result["composite_states"] = composites
        result["choice_fork_join"] = choice

    return result


async def call_claude(model: str, prompt: str, system: str = "") -> tuple[str, float, int, int]:
    """Call Claude API and return (response, time, input_tokens, output_tokens)."""
    client = anthropic.Anthropic()
    messages = [{"role": "user", "content": prompt}]
    kwargs = {"model": model, "max_tokens": 4096, "messages": messages}
    if system:
        kwargs["system"] = system

    t0 = time.time()
    response = client.messages.create(**kwargs)
    elapsed = time.time() - t0

    text = response.content[0].text
    usage = response.usage
    return text, elapsed, usage.input_tokens, usage.output_tokens


async def main():
    print("=" * 70)
    print("KNOWLEDGE DISTILLATION TEST")
    print("Can Haiku + lessons match Sonnet quality?")
    print("=" * 70)

    # Cost rates (per 1M tokens)
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

        # A: Haiku baseline (no lessons)
        print("\n  [A] Haiku baseline (no lessons)...")
        raw_a, time_a, in_a, out_a = await call_claude(HAIKU, task["prompt"])
        code_a = extract_mermaid(raw_a)
        analysis_a = analyze_mermaid(code_a, task["name"])
        cost_a = (in_a * COSTS[HAIKU]["input"] + out_a * COSTS[HAIKU]["output"]) / 1_000_000
        print(f"      Time: {time_a:.1f}s | Tokens: {in_a}+{out_a} | Cost: ${cost_a:.5f}")
        task_results["haiku_baseline"] = {
            "code": code_a,
            "analysis": analysis_a,
            "time": time_a,
            "cost": cost_a,
            "in": in_a,
            "out": out_a,
        }

        # B: Haiku + sub-domain-specific lessons (hierarchical retrieval)
        task_lessons = get_lessons_for_task(task["prompt"])
        print(f"  [B] Haiku + sub-domain lessons...")
        if task_lessons:
            print(f"      Lessons injected:\n{task_lessons}")
        raw_b, time_b, in_b, out_b = await call_claude(HAIKU, task["prompt"], system=task_lessons)
        code_b = extract_mermaid(raw_b)
        analysis_b = analyze_mermaid(code_b, task["name"])
        cost_b = (in_b * COSTS[HAIKU]["input"] + out_b * COSTS[HAIKU]["output"]) / 1_000_000
        print(f"      Time: {time_b:.1f}s | Tokens: {in_b}+{out_b} | Cost: ${cost_b:.5f}")
        task_results["haiku_lessons"] = {
            "code": code_b,
            "analysis": analysis_b,
            "time": time_b,
            "cost": cost_b,
            "in": in_b,
            "out": out_b,
        }

        # C: Sonnet gold standard
        print("  [C] Sonnet (gold standard)...")
        raw_c, time_c, in_c, out_c = await call_claude(SONNET, task["prompt"])
        code_c = extract_mermaid(raw_c)
        analysis_c = analyze_mermaid(code_c, task["name"])
        cost_c = (in_c * COSTS[SONNET]["input"] + out_c * COSTS[SONNET]["output"]) / 1_000_000
        print(f"      Time: {time_c:.1f}s | Tokens: {in_c}+{out_c} | Cost: ${cost_c:.5f}")
        task_results["sonnet"] = {
            "code": code_c,
            "analysis": analysis_c,
            "time": time_c,
            "cost": cost_c,
            "in": in_c,
            "out": out_c,
        }

        all_results[task["name"]] = task_results

        # Print comparison table for this task
        print(f"\n  {'Metric':<25} {'Haiku':>12} {'Haiku+Lessons':>14} {'Sonnet':>12}")
        print(f"  {'─' * 63}")
        for key in analysis_a:
            va = analysis_a[key]
            vb = analysis_b.get(key, "—")
            vc = analysis_c.get(key, "—")
            if isinstance(va, list):
                va = len(va)
                vb = len(vb) if isinstance(vb, list) else vb
                vc = len(vc) if isinstance(vc, list) else vc
            print(f"  {key:<25} {str(va):>12} {str(vb):>14} {str(vc):>12}")

    # Final summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")

    total_cost = {"haiku_baseline": 0, "haiku_lessons": 0, "sonnet": 0}
    for task_name, results in all_results.items():
        for variant in total_cost:
            total_cost[variant] += results[variant]["cost"]

    print(f"\n  {'':25} {'Haiku':>12} {'Haiku+Lessons':>14} {'Sonnet':>12}")
    print(f"  {'─' * 63}")
    print(
        f"  {'Total cost':25} ${total_cost['haiku_baseline']:.5f}   ${total_cost['haiku_lessons']:.5f}   ${total_cost['sonnet']:.5f}"
    )
    print(
        f"  {'Cost vs Sonnet':25} {total_cost['haiku_baseline']/total_cost['sonnet']:.1%}          {total_cost['haiku_lessons']/total_cost['sonnet']:.1%}          100.0%"
    )

    # Save outputs for inspection
    for task_name, results in all_results.items():
        slug = task_name.lower().replace(" ", "_")
        for variant in ["haiku_baseline", "haiku_lessons", "sonnet"]:
            path = f"/tmp/mermaid_ab_{slug}_{variant}.mmd"
            with open(path, "w") as f:
                f.write(results[variant]["code"])

    print(f"\n  Output files saved to /tmp/mermaid_ab_*.mmd")
    print(
        f"  Inspect with: cat /tmp/mermaid_ab_complex_er_diagram_{{haiku_baseline,haiku_lessons,sonnet}}.mmd"
    )


if __name__ == "__main__":
    asyncio.run(main())
