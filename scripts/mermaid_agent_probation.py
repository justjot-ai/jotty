"""
Mermaid Agent Probation — crystallization test for diagram generation.

Trains an agent to become a domain expert at generating Mermaid diagrams.
The curriculum covers: flowcharts, sequence diagrams, class diagrams,
state diagrams, ER diagrams, and Gantt charts.

Thresholds set to 100% success — mermaid is deterministic, no web
search variability, so perfection is achievable.
"""

import asyncio
import logging
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
logging.getLogger("Jotty.core.intelligence.learning.crystallization").setLevel(logging.INFO)
logging.getLogger("Jotty.core.intelligence.reasoning.agents.autonomous_agent").setLevel(
    logging.INFO
)
logging.getLogger("Jotty.core.intelligence.learning.learning_service").setLevel(logging.DEBUG)

MERMAID_CURRICULUM = [
    # Round 1: Simple flowcharts
    "Generate a Mermaid flowchart diagram showing a user login flow: "
    "start → enter credentials → validate → [valid: dashboard, invalid: error → retry]. "
    "Save the .mmd file to /tmp/mermaid_login_flow.mmd",
    "Generate a Mermaid flowchart for an e-commerce checkout process: "
    "cart → shipping address → payment → [success: confirmation, fail: retry payment]. "
    "Save to /tmp/mermaid_checkout.mmd",
    # Round 2: Sequence diagrams
    "Generate a Mermaid sequence diagram showing a REST API request flow: "
    "Client → API Gateway → Auth Service → Backend → Database, with responses back. "
    "Save to /tmp/mermaid_api_sequence.mmd",
    "Generate a Mermaid sequence diagram for a websocket chat system: "
    "User A → Server → User B, including connection handshake, message send, "
    "and disconnect. Save to /tmp/mermaid_chat_sequence.mmd",
    # Round 3: Class diagrams
    "Generate a Mermaid class diagram for a simple blog system with classes: "
    "User (id, name, email), Post (id, title, content, author), "
    "Comment (id, text, author, post). Show relationships. "
    "Save to /tmp/mermaid_blog_classes.mmd",
    # Round 4: State and ER diagrams
    "Generate a Mermaid state diagram for an order lifecycle: "
    "Created → Paid → Shipped → Delivered, with cancel transitions from "
    "Created and Paid states. Save to /tmp/mermaid_order_state.mmd",
    "Generate a Mermaid ER diagram for a school database: "
    "Student, Course, Teacher, Enrollment tables with relationships. "
    "Save to /tmp/mermaid_school_er.mmd",
    # Round 5: Gantt charts and complex
    "Generate a Mermaid Gantt chart for a 4-week software project: "
    "Week 1: Design + Requirements, Week 2-3: Development (frontend parallel with backend), "
    "Week 4: Testing + Deployment. Save to /tmp/mermaid_project_gantt.mmd",
]

MERMAID_THRESHOLDS = {
    "min_episodes": 10,
    "min_success_rate": 1.0,
    "min_plan_consistency": 0.50,
    "min_role_q": 0.60,
    "min_plans": 5,
}


async def main():
    from core.intelligence.learning.crystallization import (
        run_probation,
        should_crystallize,
        list_crystallized,
    )
    from core.intelligence.learning.facade import get_td_lambda

    t0 = time.time()
    print("=" * 60)
    print("MERMAID AGENT PROBATION")
    print("=" * 60)
    print(f"Curriculum: {len(MERMAID_CURRICULUM)} tasks")
    print(f"Domain: creation:mermaid")
    print(f"Thresholds: {MERMAID_THRESHOLDS}")
    print()

    result = await run_probation(
        task_type="creation",
        domain="mermaid",
        max_tasks=len(MERMAID_CURRICULUM),
        goals=MERMAID_CURRICULUM,
        thresholds=MERMAID_THRESHOLDS,
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

    # Show Q-table state
    td = get_td_lambda()
    print()
    print("Q-TABLE STATE:")
    for key in ["creation", "creation:mermaid"]:
        plans = td.step_q._plan_history.get(key, [])
        if plans:
            print(f"\n  [{key}] {len(plans)} plans:")
            from collections import Counter

            template_counts = Counter(roles for roles, _ in plans)
            for template, count in template_counts.most_common(3):
                avg_r = sum(r for ro, r in plans if ro == template) / count
                print(f"    {' → '.join(template)} x{count} (avg reward={avg_r:.2f})")

        guidance = td.step_q.get_role_guidance(*td.step_q._split_key(key))
        if guidance:
            print(f"  Roles:")
            for g in guidance:
                print(
                    f"    {g['role']}: {g['best_skill']} "
                    f"Q={g['best_q']:.2f} ({g['total_visits']} visits)"
                )

    # Convergence check with custom thresholds
    ok, stats = should_crystallize("creation", domain="mermaid", thresholds=MERMAID_THRESHOLDS)
    print(f"\n  Convergence: {ok}")
    for k in ["success_rate", "plan_consistency", "total_obs"]:
        if k in stats:
            v = stats[k]
            print(f"    {k}: {v:.0%}" if isinstance(v, float) else f"    {k}: {v}")
    print(f"    reasons: {stats.get('reasons', [])}")

    # Check output files
    print()
    import os

    for f in [
        "/tmp/mermaid_login_flow.mmd",
        "/tmp/mermaid_checkout.mmd",
        "/tmp/mermaid_api_sequence.mmd",
        "/tmp/mermaid_chat_sequence.mmd",
        "/tmp/mermaid_blog_classes.mmd",
        "/tmp/mermaid_order_state.mmd",
        "/tmp/mermaid_school_er.mmd",
        "/tmp/mermaid_project_gantt.mmd",
    ]:
        if os.path.exists(f):
            size = os.path.getsize(f)
            print(f"  ✓ {f} ({size:,} bytes)")
        else:
            print(f"  ✗ {f} (not created)")

    print(f"\nAll crystallized: {[c.domain_key for c in list_crystallized()]}")


if __name__ == "__main__":
    asyncio.run(main())
