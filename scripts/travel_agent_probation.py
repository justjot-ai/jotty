"""
Travel Agent Probation — real-world crystallization test.

Trains an agent to become a domain expert at creating international
travel itineraries. The curriculum covers: destination research,
itinerary planning, cost estimation, and report generation.

After enough successful runs with a consistent plan template,
the agent should auto-crystallize into a domain expert.
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
# Show crystallization + key progress
logging.getLogger("core.intelligence.learning.crystallization").setLevel(logging.INFO)
logging.getLogger("Jotty.core.intelligence.learning.crystallization").setLevel(logging.INFO)
logging.getLogger("Jotty.core.intelligence.reasoning.agents.autonomous_agent").setLevel(
    logging.INFO
)


TRAVEL_CURRICULUM = [
    # Round 1: Simple destination research
    "Research the top 5 must-visit places in Tokyo Japan for a first-time visitor, "
    "include estimated costs in USD, and save the travel guide to /tmp/tokyo_guide.md",
    "Research the best time to visit Bali Indonesia, top attractions, "
    "local food recommendations, and save the research to /tmp/bali_guide.md",
    # Round 2: Multi-destination itinerary research
    "Research and compile a 7-day travel itinerary for Portugal (Lisbon + Porto). "
    "Research daily activities, restaurant recommendations, transportation between cities, "
    "and estimated budget. Save the research to /tmp/portugal_itinerary.md",
    "Research and compile a 5-day Thailand trip covering Bangkok and Chiang Mai. "
    "Research flights between cities, hotel recommendations, must-see temples, "
    "street food spots, and total budget estimate. Save to /tmp/thailand_itinerary.md",
    # Round 3: Complex multi-country itinerary research
    "Research and compile a detailed 10-day Europe trip covering Paris, Amsterdam, and Berlin. "
    "Research inter-city train options, daily activities, "
    "hotel budget per city, top restaurants, visa requirements for Indian passport holders, "
    "and total trip cost breakdown. Save the research to /tmp/europe_itinerary.md",
    "Research and compile a 7-day Japan trip covering Tokyo, Kyoto, and Osaka. "
    "Research JR Pass options, ryokan stays, day trips to Mount Fuji, "
    "food tours, cultural etiquette tips, and a packing list. "
    "Save the complete research to /tmp/japan_itinerary.md",
]


async def main():
    from core.intelligence.learning.crystallization import (
        run_probation,
        should_crystallize,
        list_crystallized,
        load,
    )
    from core.intelligence.learning.facade import get_td_lambda

    t0 = time.time()
    print("=" * 60)
    print("TRAVEL AGENT PROBATION")
    print("=" * 60)
    print(f"Curriculum: {len(TRAVEL_CURRICULUM)} tasks")
    print(f"Domain: research:travel")
    print()

    result = await run_probation(
        task_type="research",
        domain="travel",
        max_tasks=len(TRAVEL_CURRICULUM),
        goals=TRAVEL_CURRICULUM,
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

    # Show Q-table state regardless
    td = get_td_lambda()
    print()
    print("Q-TABLE STATE:")
    for key in ["research", "research:travel"]:
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

    # Convergence check
    ok, stats = should_crystallize("research", domain="travel")
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
        "/tmp/tokyo_guide.md",
        "/tmp/bali_guide.md",
        "/tmp/portugal_itinerary.md",
        "/tmp/thailand_itinerary.md",
        "/tmp/europe_itinerary.md",
        "/tmp/japan_itinerary.md",
    ]:
        if os.path.exists(f):
            size = os.path.getsize(f)
            print(f"  ✓ {f} ({size:,} bytes)")
        else:
            print(f"  ✗ {f} (not created)")

    print(f"\nAll crystallized: {[c.domain_key for c in list_crystallized()]}")


if __name__ == "__main__":
    asyncio.run(main())
