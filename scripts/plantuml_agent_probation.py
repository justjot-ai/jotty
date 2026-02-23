"""
PlantUML Agent Probation — knowledge distillation test.

Phase 1: RESEARCH — agent searches the web for PlantUML syntax, patterns,
          and real-world examples. Lessons are distilled from findings.
Phase 2: TEACHER — expert-crafted generation tasks using PlantUML syntax
          that Haiku typically struggles with (deployment, component, activity).
Phase 3: COMPLEX — harder tasks combining multiple PlantUML features.

Goal: can distilled lessons from Sonnet teach Haiku to produce
      Sonnet-quality PlantUML at 15x lower cost?
"""

import asyncio
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.infrastructure.foundation.dspy_init import load_api_keys

load_api_keys()

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("core.intelligence.learning.crystallization").setLevel(logging.INFO)
logging.getLogger("Jotty.core.intelligence.reasoning.agents.autonomous_agent").setLevel(
    logging.INFO
)
logging.getLogger("Jotty.core.intelligence.learning.learning_service").setLevel(logging.INFO)

# Phase 1: RESEARCH — force the agent to search the web for PlantUML knowledge
RESEARCH_CURRICULUM = [
    # Task 1: Search for PlantUML sequence diagram syntax and examples
    "Research PlantUML sequence diagram syntax by searching the web. "
    "Find at least 3 real examples showing: participants, arrows (-> vs -->), "
    "alt/else blocks, loop blocks, activate/deactivate, notes, and grouping. "
    "Compile a comprehensive syntax reference with examples. "
    "Save to /tmp/plantuml_sequence_reference.md",
    # Task 2: Search for PlantUML class diagram syntax
    "Research PlantUML class diagram syntax by searching the web. "
    "Find examples showing: class definitions with fields and methods, "
    "visibility modifiers (+, -, #, ~), inheritance (--|>), composition (*--), "
    "aggregation (o--), association (--), interfaces, abstract classes, "
    "and stereotypes. Compile a syntax reference. "
    "Save to /tmp/plantuml_class_reference.md",
    # Task 3: Search for PlantUML component/deployment diagram syntax
    "Research PlantUML deployment and component diagram syntax by searching the web. "
    "Find examples showing: node, component, database, cloud, folder, frame, "
    "package, artifact, interface lollipop notation, and port connections. "
    "Include examples of nested components and realistic architectures. "
    "Save to /tmp/plantuml_deployment_reference.md",
    # Task 4: Search for PlantUML activity diagram syntax (new beta syntax)
    "Research PlantUML activity diagram (new/beta) syntax by searching the web. "
    "Find examples showing: start/stop, if/elseif/else, while loops, "
    "fork/fork again/end fork for parallel, swimlanes with |Swimlane|, "
    "notes, colors, and detach. Compile a reference. "
    "Save to /tmp/plantuml_activity_reference.md",
]

# Phase 2: TEACHER — expert-crafted generation tasks
TEACHER_CURRICULUM = [
    # Sequence diagram (builds on research)
    "Generate a PlantUML sequence diagram for an OAuth2 authorization code flow: "
    "User → Browser → App Server → Authorization Server → App Server → Resource Server. "
    "Include: activate/deactivate blocks, alt block for token refresh vs new auth, "
    "note right of Authorization Server with 'Validates credentials', "
    "return arrows with response codes (200, 302, 401). "
    "Wrap in @startuml/@enduml. Save to /tmp/plantuml_oauth2_sequence.puml",
    # Class diagram
    "Generate a PlantUML class diagram for a design pattern: Observer pattern. "
    "Classes: Subject (abstract, with attach/detach/notify methods), "
    "ConcreteSubject (with state field), Observer (interface, with update method), "
    "ConcreteObserverA and ConcreteObserverB (implementing Observer). "
    "Use proper visibility modifiers (+/-/#), show inheritance with --|>, "
    "show interface implementation with ..|>, add stereotypes <<interface>> and <<abstract>>. "
    "Wrap in @startuml/@enduml. Save to /tmp/plantuml_observer_class.puml",
    # Activity diagram with swimlanes
    "Generate a PlantUML activity diagram (new/beta syntax) for an employee onboarding process. "
    "Use swimlanes: |HR|, |IT|, |Manager|, |Employee|. "
    "Flow: HR creates account → IT provisions laptop + email (fork/parallel) → "
    "Manager assigns buddy → Employee completes orientation → "
    "if (background check passed?) then (yes) → confirmed, else (no) → terminated. "
    "Wrap in @startuml/@enduml. Save to /tmp/plantuml_onboarding_activity.puml",
    # Deployment diagram
    "Generate a PlantUML deployment diagram for a Kubernetes-based microservices architecture. "
    "Show: cloud 'AWS' containing node 'EKS Cluster' containing "
    "node 'Node Pool' with components: API Gateway, Auth Service, Order Service, Payment Service. "
    "Outside cluster: database 'RDS PostgreSQL', database 'ElastiCache Redis', "
    "cloud 'Stripe API' (external). Show connections between components with protocols "
    "(HTTPS, gRPC, TCP). Use proper PlantUML deployment syntax with node, component, database. "
    "Wrap in @startuml/@enduml. Save to /tmp/plantuml_k8s_deployment.puml",
]

# Phase 3: COMPLEX — harder synthesis tasks
COMPLEX_CURRICULUM = [
    # Complex sequence with multiple interaction fragments
    "Generate a PlantUML sequence diagram for a distributed transaction (Saga pattern): "
    "OrderService → PaymentService → InventoryService → ShippingService. "
    "Show: the happy path with all services succeeding, "
    "then an alt block where InventoryService fails, triggering compensating transactions "
    "(ShippingService.cancelShipment, PaymentService.refund, OrderService.cancelOrder). "
    "Use activate/deactivate, group blocks, return arrows, and colored notes. "
    "Wrap in @startuml/@enduml. Save to /tmp/plantuml_saga_sequence.puml",
    # Complex class diagram with patterns
    "Generate a PlantUML class diagram for an e-commerce system combining "
    "Strategy + Factory patterns. Classes: PaymentStrategy <<interface>> with pay() method, "
    "CreditCardPayment, PayPalPayment, CryptoPayment implementing it. "
    "PaymentFactory with createPayment(type): PaymentStrategy. "
    "Order class with items: List<OrderItem>, total: BigDecimal, strategy: PaymentStrategy. "
    "OrderItem with product: Product, quantity: int, price: BigDecimal. "
    "Product with id, name, price, category. "
    "Show all relationships, multiplicities (1..*, 0..1), and method signatures. "
    "Wrap in @startuml/@enduml. Save to /tmp/plantuml_ecommerce_class.puml",
]

PLANTUML_CURRICULUM = RESEARCH_CURRICULUM + TEACHER_CURRICULUM + COMPLEX_CURRICULUM

PLANTUML_THRESHOLDS = {
    "min_episodes": 15,
    "min_success_rate": 0.80,
    "min_plan_consistency": 0.40,
    "min_role_q": 0.50,
    "min_plans": 6,
}


async def main():
    from core.intelligence.learning.crystallization import (
        list_crystallized,
        run_probation,
        should_crystallize,
    )
    from core.intelligence.learning.facade import get_td_lambda

    t0 = time.time()
    print("=" * 60)
    print("PLANTUML AGENT PROBATION")
    print("=" * 60)
    print(f"Curriculum: {len(PLANTUML_CURRICULUM)} tasks")
    print(f"  Phase 1 (Research):  {len(RESEARCH_CURRICULUM)} tasks")
    print(f"  Phase 2 (Teacher):   {len(TEACHER_CURRICULUM)} tasks")
    print(f"  Phase 3 (Complex):   {len(COMPLEX_CURRICULUM)} tasks")
    print(f"Domain: creation:plantuml")
    print(f"Thresholds: {PLANTUML_THRESHOLDS}")
    print()

    result = await run_probation(
        task_type="creation",
        domain="plantuml",
        max_tasks=len(PLANTUML_CURRICULUM),
        goals=PLANTUML_CURRICULUM,
        thresholds=PLANTUML_THRESHOLDS,
    )

    elapsed = time.time() - t0
    print()
    print("=" * 60)
    print(f"PROBATION COMPLETE ({elapsed:.0f}s)")
    print("=" * 60)
    print(f"  Tasks run:    {result['tasks_run']}")
    print(f"  Succeeded:    {result['succeeded']}")
    print(f"  Success rate: {result['success_rate']:.0%}")
    print(f"  Graduated:    {result['graduated']}")
    print(f"  Reason:       {result['reason']}")

    if result.get("config"):
        c = result["config"]
        print()
        print("CRYSTALLIZED CONFIG:")
        print(f"  Domain key:  {c.domain_key}")
        print(f"  SOP:         {' → '.join(c.sop_roles)}")
        print(f"  Skills:      {c.skills}")
        print(f"  Bindings:    {c.role_skill_map}")
        prompt = c.prompt_guidance
        print(f"  Prompt guidance ({len(prompt)} chars):")
        if prompt:
            for line in prompt.split("\n")[:8]:
                print(f"    {line}")

    # Show Q-table state
    td = get_td_lambda()
    print()
    print("Q-TABLE STATE:")
    for key in ["creation", "creation:plantuml", "research:plantuml"]:
        plans = td.step_q._plan_history.get(key, [])
        if plans:
            print(f"\n  [{key}] {len(plans)} plans:")
            from collections import Counter

            template_counts = Counter(roles for roles, _ in plans)
            for template, count in template_counts.most_common(3):
                avg_r = sum(r for ro, r in plans if ro == template) / count
                print(f"    {' → '.join(template)} x{count} (avg reward={avg_r:.2f})")

    # Check distilled lessons
    from core.intelligence.learning.learning_service import LearningService

    ls = LearningService()
    import sqlite3

    db_path = Path.home() / "jotty" / "learning" / "learning.db"
    conn = sqlite3.connect(str(db_path))
    rows = conn.execute(
        """SELECT domain, lesson, context_type FROM distilled_lessons
           WHERE domain LIKE 'plantuml%' ORDER BY domain, timestamp"""
    ).fetchall()
    print(f"\nDISTILLED LESSONS ({len(rows)}):")
    for domain, lesson, ctype in rows:
        print(f"  [{domain:25s}] [{ctype:10s}] {lesson[:80]}")
    conn.close()

    # Check output files
    print()
    output_files = [
        "/tmp/plantuml_sequence_reference.md",
        "/tmp/plantuml_class_reference.md",
        "/tmp/plantuml_deployment_reference.md",
        "/tmp/plantuml_activity_reference.md",
        "/tmp/plantuml_oauth2_sequence.puml",
        "/tmp/plantuml_observer_class.puml",
        "/tmp/plantuml_onboarding_activity.puml",
        "/tmp/plantuml_k8s_deployment.puml",
        "/tmp/plantuml_saga_sequence.puml",
        "/tmp/plantuml_ecommerce_class.puml",
    ]
    for f in output_files:
        if os.path.exists(f):
            size = os.path.getsize(f)
            print(f"  ✓ {f} ({size:,} bytes)")
        else:
            print(f"  ✗ {f} (not created)")

    print(f"\nAll crystallized: {[c.domain_key for c in list_crystallized()]}")


if __name__ == "__main__":
    asyncio.run(main())
