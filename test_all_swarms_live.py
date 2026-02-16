"""
Live test: Run all 13 swarms with complex, real-world tasks.
Each swarm's full output is saved to /tmp/swarm_outputs/<swarm_name>.json
Streaming log goes to /tmp/swarm_test_output.log
"""

import asyncio
import json
import time
import traceback
from dataclasses import asdict
from pathlib import Path

OUTPUT_DIR = Path("/tmp/swarm_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

RESULTS = []


def save_output(name: str, result: object, elapsed: float) -> None:
    """Save full swarm output to a JSON file."""
    safe_name = name.split(".")[0].strip().replace(" ", "_").replace("'", "").replace("/", "_")
    # Remove leading number + dot
    for i, c in enumerate(safe_name):
        if c.isalpha():
            safe_name = safe_name[i:]
            break

    out = {
        "swarm": name,
        "success": getattr(result, "success", None),
        "elapsed_seconds": round(elapsed, 1),
        "output": None,
        "error": getattr(result, "error", None),
    }

    output = getattr(result, "output", None)
    if isinstance(output, dict):
        out["output"] = output
    elif output is not None:
        out["output"] = str(output)

    # Include extra dataclass fields
    try:
        full = asdict(result)
        for k in (
            "success",
            "output",
            "error",
            "swarm_name",
            "domain",
            "execution_time",
            "agent_traces",
        ):
            full.pop(k, None)
        if full:
            out["extra_fields"] = full
    except Exception:
        pass

    filepath = OUTPUT_DIR / f"{safe_name}.json"
    with open(filepath, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"         -> saved to {filepath}")


def report(name, result=None, error=None, elapsed=0.0):
    """Record and print a test result."""
    if error:
        RESULTS.append(
            {"swarm": name, "success": False, "error": str(error)[:200], "time": elapsed}
        )
        print(f"  FAIL  {name} ({elapsed:.1f}s)")
        print(f"         {str(error)[:200]}")
        # Save error output too
        safe_name = name.split(".")[0].strip().replace(" ", "_").replace("'", "")
        for i, c in enumerate(safe_name):
            if c.isalpha():
                safe_name = safe_name[i:]
                break
        filepath = OUTPUT_DIR / f"{safe_name}.json"
        with open(filepath, "w") as f:
            json.dump(
                {"swarm": name, "success": False, "error": str(error), "elapsed": elapsed},
                f,
                indent=2,
            )
        print(f"         -> saved to {filepath}")
    else:
        success = getattr(result, "success", False)
        output = getattr(result, "output", {})
        keys = list(output.keys()) if isinstance(output, dict) else []
        RESULTS.append({"swarm": name, "success": success, "keys": keys, "time": elapsed})
        status = "  OK  " if success else "  WARN "
        print(f"{status} {name} ({elapsed:.1f}s) keys={keys}")
        save_output(name, result, elapsed)


async def run_test(name, coro):
    """Run a single test with error handling."""
    t = time.time()
    try:
        r = await coro
        report(name, r, elapsed=time.time() - t)
    except Exception as e:
        report(name, error=e, elapsed=time.time() - t)
        traceback.print_exc()


async def main():
    print("=" * 80)
    print("LIVE SWARM TEST — 13 swarms with complex tasks")
    print(f"Output dir: {OUTPUT_DIR}")
    print("=" * 80)
    total_start = time.time()

    # ---- Batch 1: Core swarms ----
    print("\n--- Batch 1: Core swarms (Fundamental, IdeaWriter, Pilot) ---")

    from Jotty.core.execution.swarms.templates.fundamental_swarm import FundamentalSwarm

    await run_test(
        "1. FundamentalSwarm.analyze('RELIANCE.NS')",
        FundamentalSwarm().analyze("RELIANCE.NS"),
    )

    from Jotty.core.execution.swarms.templates.idea_writer_swarm import IdeaWriterSwarm

    await run_test(
        "2. IdeaWriterSwarm.write('AGI labor markets')",
        IdeaWriterSwarm().write(
            topic="The economic implications of artificial general intelligence on global labor markets",
            sections=["introduction", "body", "body", "body", "conclusion"],
        ),
    )

    from Jotty.core.execution.swarms.pilot_swarm.swarm import PilotSwarm

    await run_test(
        "3. PilotSwarm.run_goal('trading platform arch')",
        PilotSwarm().run_goal(
            goal=(
                "Design a microservices architecture for a real-time stock trading platform "
                "that handles 100K concurrent users, supports order matching, portfolio management, "
                "and risk assessment. Include technology choices and failure recovery strategies."
            ),
        ),
    )

    # ---- Batch 2: Learning swarms ----
    print("\n--- Batch 2: Learning swarms (Perspective, Olympiad) ---")

    from Jotty.core.execution.swarms.perspective_learning_swarm.swarm import (
        PerspectiveLearningSwarm,
    )

    await run_test(
        "4. PerspectiveLearningSwarm.teach('quantum entanglement')",
        PerspectiveLearningSwarm().teach(
            topic="Quantum entanglement and its implications for computing and cryptography",
            student_name="Alex",
            age_group="advanced_high_school",
            depth="deep",
        ),
    )

    from Jotty.core.execution.swarms.olympiad_learning_swarm.swarm import OlympiadLearningSwarm

    await run_test(
        "5. OlympiadLearningSwarm.teach('combinatorics')",
        OlympiadLearningSwarm().teach(
            topic="Combinatorics: Pigeonhole principle, inclusion-exclusion, and generating functions",
            subject="mathematics",
            student_name="Priya",
            depth="deep",
            target_level="olympiad",
        ),
    )

    # ---- Batch 3: Code swarms ----
    print("\n--- Batch 3: Code swarms (Coding, Review, Testing) ---")

    from Jotty.core.execution.swarms.coding_swarm.swarm import CodingSwarm

    await run_test(
        "6. CodingSwarm.generate('LRU cache')",
        CodingSwarm().generate(
            requirements=(
                "Implement a thread-safe LRU cache in Python with: configurable max size, "
                "TTL-based expiration, hit/miss statistics tracking, a decorator for memoization, "
                "and an eviction callback. Include type hints and comprehensive docstrings."
            ),
            language="python",
            include_tests=True,
        ),
    )

    from Jotty.core.execution.swarms.templates.review_swarm import ReviewSwarm

    buggy_code = """
import threading
import sqlite3

class UserDB:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.conn = sqlite3.connect("users.db")
        return cls._instance

    def get_user(self, user_id):
        cursor = self.conn.execute(
            f"SELECT * FROM users WHERE id = {user_id}"
        )
        return cursor.fetchone()

    def update_balance(self, user_id, amount):
        self.conn.execute(
            f"UPDATE users SET balance = balance + {amount} WHERE id = {user_id}"
        )
        self.conn.commit()

    def transfer(self, from_id, to_id, amount):
        sender = self.get_user(from_id)
        if sender[2] >= amount:
            self.update_balance(from_id, -amount)
            self.update_balance(to_id, amount)
"""
    await run_test(
        "7. ReviewSwarm.review('buggy UserDB')",
        ReviewSwarm().review(
            code=buggy_code,
            language="python",
            context="Production banking code handling money transfers. Review for security, concurrency, and correctness.",
        ),
    )

    from Jotty.core.execution.swarms.templates.testing_swarm import TestingSwarm

    scheduler_code = """
from datetime import datetime, timedelta
from typing import List, Optional

class Scheduler:
    def __init__(self):
        self.events = []
        self.recurring = []

    def add_event(self, name: str, start: datetime, duration: timedelta,
                  priority: int = 0) -> dict:
        event = {"name": name, "start": start, "end": start + duration,
                 "priority": priority}
        for existing in self.events:
            if self._overlaps(event, existing):
                if event["priority"] <= existing["priority"]:
                    raise ValueError(f"Conflicts with {existing['name']}")
                self.events.remove(existing)
        self.events.append(event)
        return event

    def add_recurring(self, name: str, start: datetime, duration: timedelta,
                      interval: timedelta, count: int = 10) -> List[dict]:
        events = []
        current = start
        for _ in range(count):
            try:
                events.append(self.add_event(name, current, duration))
            except ValueError:
                pass
            current += interval
        self.recurring.append({"name": name, "events": events})
        return events

    def get_free_slots(self, date: datetime, slot_duration: timedelta) -> List[tuple]:
        day_start = date.replace(hour=9, minute=0)
        day_end = date.replace(hour=17, minute=0)
        day_events = sorted(
            [e for e in self.events if e["start"].date() == date.date()],
            key=lambda e: e["start"]
        )
        slots = []
        current = day_start
        for event in day_events:
            if event["start"] - current >= slot_duration:
                slots.append((current, event["start"]))
            current = max(current, event["end"])
        if day_end - current >= slot_duration:
            slots.append((current, day_end))
        return slots

    def _overlaps(self, a, b):
        return a["start"] < b["end"] and b["start"] < a["end"]
"""
    await run_test(
        "8. TestingSwarm.generate_tests('Scheduler')",
        TestingSwarm().generate_tests(
            code=scheduler_code,
            language="python",
        ),
    )

    # ---- Batch 4: Infrastructure & analysis ----
    print("\n--- Batch 4: Infrastructure & analysis (DataAnalysis, DevOps, Learning) ---")

    from Jotty.core.execution.swarms.templates.data_analysis_swarm import DataAnalysisSwarm

    await run_test(
        "9. DataAnalysisSwarm.analyze('e-commerce decline')",
        DataAnalysisSwarm().analyze(
            data={
                "description": "Monthly sales data for an e-commerce platform (2023-2025)",
                "columns": [
                    "date",
                    "revenue",
                    "orders",
                    "avg_order_value",
                    "return_rate",
                    "customer_acquisition_cost",
                    "lifetime_value",
                    "channel",
                ],
                "sample": [
                    {
                        "date": "2024-01",
                        "revenue": 450000,
                        "orders": 3200,
                        "avg_order_value": 140.6,
                        "return_rate": 0.08,
                        "customer_acquisition_cost": 45,
                        "lifetime_value": 380,
                        "channel": "organic",
                    },
                    {
                        "date": "2024-06",
                        "revenue": 520000,
                        "orders": 3800,
                        "avg_order_value": 136.8,
                        "return_rate": 0.12,
                        "customer_acquisition_cost": 52,
                        "lifetime_value": 350,
                        "channel": "paid",
                    },
                    {
                        "date": "2025-01",
                        "revenue": 380000,
                        "orders": 2900,
                        "avg_order_value": 131.0,
                        "return_rate": 0.15,
                        "customer_acquisition_cost": 68,
                        "lifetime_value": 310,
                        "channel": "organic",
                    },
                ],
                "shape": [24, 8],
            },
            question=(
                "Identify the root cause of declining revenue and rising return rates. "
                "Recommend specific actions to improve unit economics and customer retention."
            ),
        ),
    )

    from Jotty.core.execution.swarms.templates.devops_swarm import DevOpsSwarm

    await run_test(
        "10. DevOpsSwarm.design_infrastructure('K8s SaaS')",
        DevOpsSwarm().design_infrastructure(
            app_name="tradeflow",
            app_type="microservices",
            language="python",
            requirements=(
                "Multi-region SaaS platform: 3 regions (us-east, eu-west, ap-south), "
                "auto-scaling 10-500 pods, PostgreSQL with read replicas, Redis cluster, "
                "zero-downtime deployments, DR with RPO<5min RTO<15min, "
                "Prometheus/Grafana monitoring, log aggregation, cost optimization."
            ),
            scale="large",
            compliance=["SOC2", "GDPR"],
        ),
    )

    from Jotty.core.execution.swarms.templates.learning_swarm import LearningSwarm

    await run_test(
        "11. LearningSwarm.evaluate_and_improve('coding_swarm')",
        LearningSwarm().evaluate_and_improve(
            swarm_name="coding_swarm",
        ),
    )

    # ---- Batch 5: Research swarms ----
    print("\n--- Batch 5: Research swarms (Research, ArXiv) ---")

    from Jotty.core.execution.swarms.research_swarm.swarm import ResearchSwarm

    await run_test(
        "12. ResearchSwarm.research('transformers vs SSMs')",
        ResearchSwarm().research(
            query=(
                "Compare transformer architectures vs state space models (Mamba, S4) for long-context NLP. "
                "Analyze computational complexity, memory efficiency, and benchmark performance "
                "on tasks requiring 100K+ token context windows."
            ),
        ),
    )

    from Jotty.core.execution.swarms.arxiv_learning_swarm.swarm import ArxivLearningSwarm

    await run_test(
        "13. ArxivLearningSwarm.learn('attention mechanisms')",
        ArxivLearningSwarm().learn(
            topic="attention mechanisms in transformer architectures",
            depth="deep",
        ),
    )

    # ---- Summary ----
    total = time.time() - total_start
    passed = sum(1 for r in RESULTS if r["success"])
    warned = sum(1 for r in RESULTS if r["success"] is False and "error" not in r)
    failed = sum(1 for r in RESULTS if not r["success"] and "error" in r)

    print("\n" + "=" * 80)
    print(f"RESULTS: {passed} passed / {len(RESULTS)} total ({total:.1f}s)")
    print(f"Output files: {OUTPUT_DIR}/")
    print("=" * 80)

    for r in RESULTS:
        status = "PASS" if r["success"] else "FAIL"
        print(f"  [{status}] {r['swarm']} ({r['time']:.1f}s)")
        if not r["success"] and "error" in r:
            print(f"         Error: {r['error'][:150]}")

    if passed == len(RESULTS):
        print("\nALL 13 SWARMS PASSED")
    else:
        print(f"\n{passed}/{len(RESULTS)} passed")


if __name__ == "__main__":
    asyncio.run(main())
