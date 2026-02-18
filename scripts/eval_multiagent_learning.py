#!/usr/bin/env python3
"""
Jotty Multi-Agent Learning Evaluation
======================================

Tests whether multi-agent coordination, credit assignment, and learning
actually work together as a system:

1. PIPELINE EXECUTION: Multi-stage pipeline with agent dependencies
2. CREDIT ASSIGNMENT: Verify agents get differential credit
3. LEARNING LOOP: Verify episodes are recorded and patterns extracted
4. COORDINATION: Verify stages pass context to downstream stages
5. A/B: Compare with-learning vs without-learning quality

This is a REAL eval that makes actual LLM calls. Costs ~$0.50-1.00.
"""

import asyncio
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("eval_multiagent")
logger.setLevel(logging.INFO)


def separator(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}\n")


def subsep(title: str) -> None:
    print(f"\n  --- {title} ---\n")


# =============================================================================
# MODEL CONFIG — use a weaker model so learning has room to help
# =============================================================================

EVAL_MODEL = "claude-3-haiku-20240307"

# =============================================================================
# TEST SCENARIOS — genuinely hard cross-domain tasks that stress a weak model
# =============================================================================

PIPELINE_TASKS = [
    {
        "id": "P1_consensus_raft",
        "goal": (
            "Build a Raft consensus implementation with formal correctness reasoning:\n"
            "Stage 1 (Formal Model): Define Raft's safety properties using TLA+-style "
            "invariants — election safety, log matching, leader completeness, state machine safety. "
            "Prove why split-brain is impossible under the leader election protocol.\n"
            "Stage 2 (Implementation): Implement Raft in Python with leader election, "
            "log replication, and membership changes. Handle edge cases: simultaneous "
            "elections, log conflicts, partitioned leaders.\n"
            "Stage 3 (Adversarial Tests): Write tests that simulate network partitions, "
            "message delays, duplicate messages, and Byzantine-like failures. "
            "Prove each safety invariant holds under the test scenarios."
        ),
        "stages": [
            {
                "name": "formal_model",
                "task": (
                    "Define the formal safety properties of the Raft consensus protocol:\n"
                    "1. ELECTION SAFETY: At most one leader per term — prove this from the "
                    "majority vote requirement. Show the pigeonhole argument.\n"
                    "2. LOG MATCHING: If two logs contain an entry with the same index and term, "
                    "then the logs are identical through that index. Prove by induction.\n"
                    "3. LEADER COMPLETENESS: If an entry is committed in term T, it appears "
                    "in all leaders' logs for terms > T. Prove using the voting mechanism.\n"
                    "4. STATE MACHINE SAFETY: If any server has applied an entry at index i, "
                    "no other server will apply a different entry at index i.\n"
                    "For each property, provide the formal invariant (as a predicate over "
                    "the system state), the proof sketch, and a concrete counterexample "
                    "that would violate it if the protocol were broken.\n"
                    "Also explain: why can't Raft have split-brain? What happens during "
                    "a network partition? Walk through the exact message sequence."
                ),
                "checks": [
                    "election safety",
                    "log matching",
                    "leader completeness",
                    "state machine safety",
                    "majority",
                    "partition",
                ],
            },
            {
                "name": "implementation",
                "task": (
                    "Implement Raft consensus in Python with these requirements:\n"
                    "- RaftNode class with states: Follower, Candidate, Leader\n"
                    "- Leader election: RequestVote RPC with term comparison\n"
                    "- Log replication: AppendEntries RPC with consistency check\n"
                    "- Commit logic: entry committed when replicated to majority\n"
                    "- Handle: simultaneous elections (randomized timeout), "
                    "log conflicts (force overwrite), stale leaders (term comparison)\n"
                    "- Membership changes: joint consensus for adding/removing nodes\n"
                    "Use asyncio. Each RaftNode must be independently runnable. "
                    "Simulate the network layer (message passing with configurable delays/drops).\n"
                    "Include type hints, dataclasses for messages, and proper logging."
                ),
                "depends_on": ["formal_model"],
                "checks": [
                    "class RaftNode",
                    "RequestVote",
                    "AppendEntries",
                    "async def",
                    "Follower",
                    "Leader",
                    "term",
                ],
            },
            {
                "name": "adversarial_testing",
                "task": (
                    "Write adversarial tests that verify each safety property from the "
                    "formal model holds under failure conditions:\n"
                    "1. TEST ELECTION SAFETY: Create 5 nodes, partition into [2,3], verify "
                    "only the majority partition elects a leader. Heal partition, verify "
                    "single leader emerges.\n"
                    "2. TEST LOG MATCHING: Simulate leader crash mid-replication. New leader "
                    "must handle conflicting log entries correctly.\n"
                    "3. TEST LEADER COMPLETENESS: Commit entry, crash leader, verify new "
                    "leader has the committed entry.\n"
                    "4. TEST PARTITION TOLERANCE: Partition leader from majority, verify "
                    "it steps down. Verify new leader is elected. Verify client requests "
                    "to old leader fail.\n"
                    "5. TEST CONCURRENT ELECTIONS: Force 3 simultaneous candidates. Verify "
                    "at most one wins per term.\n"
                    "Use pytest-asyncio. Include a NetworkSimulator that can inject: "
                    "partitions, message delays, message drops, message duplication."
                ),
                "depends_on": ["implementation"],
                "checks": [
                    "def test_",
                    "assert",
                    "partition",
                    "async",
                    "election",
                    "NetworkSimulator",
                ],
            },
        ],
        "domain": "system_design",
    },
]

LEARNING_CURVE_TASKS = [
    {
        "id": "R1_raft",
        "goal": (
            "Build a Raft consensus implementation with formal correctness reasoning:\n"
            "Stage 1 (Formal Model): Define Raft safety properties with invariants.\n"
            "Stage 2 (Implementation): Implement Raft in Python.\n"
            "Stage 3 (Testing): Adversarial tests for partitions and failures."
        ),
        "stages": PIPELINE_TASKS[0]["stages"],
        "domain": "system_design",
    },
    {
        "id": "R2_crdt",
        "goal": (
            "Build a CRDT (Conflict-free Replicated Data Type) library:\n"
            "Stage 1: Mathematical foundation — semilattices, convergence proofs.\n"
            "Stage 2: Implement GCounter, PNCounter, ORSet, LWWRegister.\n"
            "Stage 3: Property-based tests with Hypothesis."
        ),
        "stages": [
            {
                "name": "theory",
                "task": (
                    "Explain the mathematical foundation of CRDTs:\n"
                    "1. Define join-semilattice. Prove commutativity, associativity, idempotency.\n"
                    "2. Prove monotonic join guarantees convergence without coordination.\n"
                    "3. Compare CmRDT vs CvRDT — network requirements, trade-offs.\n"
                    "4. Walk through G-Counter merge and OR-Set conflict resolution.\n"
                    "5. What CAN'T be a CRDT? Give impossibility example."
                ),
                "checks": ["semilattice", "commutativ", "associativ", "idempoten", "convergence"],
            },
            {
                "name": "implementation",
                "task": (
                    "Implement CRDTs in Python:\n"
                    "- BaseCRDT: abstract with merge(), value(), clone()\n"
                    "- GCounter: grow-only counter {node_id: count}\n"
                    "- PNCounter: positive-negative (pair of GCounters)\n"
                    "- ORSet: observed-remove set with unique add tags\n"
                    "- LWWRegister: last-writer-wins with HLC timestamps\n"
                    "Use dataclasses, type hints, ABC."
                ),
                "depends_on": ["theory"],
                "checks": [
                    "class GCounter",
                    "class PNCounter",
                    "class ORSet",
                    "class LWWRegister",
                    "def merge",
                ],
            },
            {
                "name": "verification",
                "task": (
                    "Property-based tests with Hypothesis:\n"
                    "1. Commutativity: merge(a,b) == merge(b,a) for all CRDTs\n"
                    "2. Associativity: merge(merge(a,b),c) == merge(a,merge(b,c))\n"
                    "3. Idempotency: merge(a,a) == a\n"
                    "4. Convergence: all replicas converge regardless of delivery order\n"
                    "Test all 4 CRDT types. Use @given with strategies."
                ),
                "depends_on": ["implementation"],
                "checks": ["def test_", "@given", "hypothesis", "assert", "commutativ"],
            },
        ],
        "domain": "system_design",
    },
    {
        "id": "R3_dist_lock",
        "goal": (
            "Build a distributed lock service (like Chubby/ZooKeeper):\n"
            "Stage 1: Design lock protocol with fencing tokens and lease-based expiry.\n"
            "Stage 2: Implement lock server and client in Python.\n"
            "Stage 3: Test deadlock detection, lock contention, and split-brain."
        ),
        "stages": [
            {
                "name": "protocol_design",
                "task": (
                    "Design a distributed lock protocol:\n"
                    "1. Lock acquisition: compare-and-swap with fencing tokens\n"
                    "2. Lease-based expiry: prevent dead client locks\n"
                    "3. Fairness: FIFO ordering of lock requests via sequence numbers\n"
                    "4. Reentrant locks: same client can re-acquire, tracked via owner+count\n"
                    "5. Read-write locks: shared readers, exclusive writers\n"
                    "6. Deadlock prevention: timeout + lock ordering strategy\n"
                    "Explain why simple mutex is insufficient in distributed systems. "
                    "Walk through a fencing token preventing stale-lock writes."
                ),
                "checks": ["fencing token", "lease", "deadlock", "reentrant", "fairness"],
            },
            {
                "name": "implementation",
                "task": (
                    "Implement the distributed lock service in Python:\n"
                    "- LockServer: manages lock state, issues fencing tokens\n"
                    "- LockClient: acquire/release with automatic renewal\n"
                    "- FencingToken: monotonically increasing token per lock\n"
                    "- LeaseManager: tracks expiry, auto-releases stale locks\n"
                    "- WaitQueue: FIFO ordering for contended locks\n"
                    "Use asyncio. Include proper timeout handling and error recovery."
                ),
                "depends_on": ["protocol_design"],
                "checks": [
                    "class LockServer",
                    "class LockClient",
                    "FencingToken",
                    "async def acquire",
                    "async def release",
                ],
            },
            {
                "name": "testing",
                "task": (
                    "Test the lock service under adversarial conditions:\n"
                    "1. Deadlock detection: 2 clients acquire locks in opposite order\n"
                    "2. Stale lock: client dies holding lock, verify auto-expiry\n"
                    "3. Fencing: old fencing token rejected after lock re-acquisition\n"
                    "4. Contention: 10 concurrent clients competing for same lock\n"
                    "5. Split-brain: network partition during lock hold\n"
                    "Use pytest-asyncio. Include timing assertions for lease expiry."
                ),
                "depends_on": ["implementation"],
                "checks": ["def test_", "assert", "deadlock", "fencing", "async"],
            },
        ],
        "domain": "system_design",
    },
    {
        "id": "R4_event_source",
        "goal": (
            "Build an event sourcing system with CQRS:\n"
            "Stage 1: Design event store, projections, snapshots.\n"
            "Stage 2: Implement EventStore, EventBus, ProjectionEngine.\n"
            "Stage 3: Test event ordering, idempotency, snapshot recovery."
        ),
        "stages": [
            {
                "name": "architecture",
                "task": (
                    "Design an event sourcing system with CQRS:\n"
                    "1. Event store: append-only log, partitioned by aggregate ID\n"
                    "2. CQRS: separate read models (projections) from write model\n"
                    "3. Snapshots: periodic materialization to avoid full replay\n"
                    "4. Event bus: async pub/sub with at-least-once delivery\n"
                    "5. Consistency: causal ordering within aggregates, eventual across\n"
                    "6. Schema evolution: how to handle event versioning over time\n"
                    "Explain the trade-offs vs traditional CRUD. When NOT to use event sourcing."
                ),
                "checks": ["event store", "CQRS", "snapshot", "projection", "schema"],
            },
            {
                "name": "implementation",
                "task": (
                    "Implement event sourcing in Python:\n"
                    "- Event: dataclass with aggregate_id, type, data, version, timestamp\n"
                    "- EventStore: append, read_stream, get_snapshot, with versioning\n"
                    "- EventBus: publish/subscribe with async handlers\n"
                    "- ProjectionEngine: rebuild read models from event stream\n"
                    "- SnapshotManager: create/restore snapshots at intervals\n"
                    "- Aggregate: base class for event-sourced entities\n"
                    "Use asyncio, type hints, dataclasses."
                ),
                "depends_on": ["architecture"],
                "checks": [
                    "class Event",
                    "class EventStore",
                    "class Aggregate",
                    "async def",
                    "class Projection",
                ],
            },
            {
                "name": "testing",
                "task": (
                    "Test the event sourcing system:\n"
                    "1. Ordering: events maintain causal order within aggregate\n"
                    "2. Idempotency: replaying same event doesn't corrupt state\n"
                    "3. Projection rebuild: rebuild read model from scratch matches live\n"
                    "4. Snapshot recovery: restore snapshot + replay new events = correct\n"
                    "5. Schema evolution: v1 events readable by v2 projection\n"
                    "Use pytest-asyncio."
                ),
                "depends_on": ["implementation"],
                "checks": ["def test_", "assert", "async", "snapshot", "idempoten"],
            },
        ],
        "domain": "system_design",
    },
]

AB_TEST_TASK = {
    "id": "AB_gossip",
    "goal": (
        "Build a gossip protocol for cluster membership:\n"
        "Stage 1: Design SWIM-style failure detection with protocol analysis.\n"
        "Stage 2: Implement gossip protocol with protocol buffers.\n"
        "Stage 3: Test under network failures and churn."
    ),
    "stages": [
        {
            "name": "design",
            "task": (
                "Design a SWIM-style gossip protocol for cluster membership:\n"
                "1. Failure detection: ping, ping-req, suspect, dead states\n"
                "2. Dissemination: piggybacked protocol messages on ping/ack\n"
                "3. Protocol period: T_protocol, T_suspect, T_dead timers\n"
                "4. Consistency: eventually consistent membership list\n"
                "5. Prove: expected detection time is O(log N) protocol periods\n"
                "Compare with heartbeat-based detection. Why is SWIM better?"
            ),
            "checks": ["SWIM", "failure detection", "ping", "suspect", "protocol period"],
        },
        {
            "name": "implementation",
            "task": (
                "Implement SWIM gossip protocol in Python:\n"
                "- GossipNode: maintains membership list and node states\n"
                "- FailureDetector: ping/ping-req/suspect/dead state machine\n"
                "- Disseminator: piggyback membership updates on protocol messages\n"
                "- MembershipList: versioned list with lamport timestamps\n"
                "Use asyncio/UDP. Include configurable protocol timers."
            ),
            "depends_on": ["design"],
            "checks": ["class GossipNode", "class FailureDetector", "async def", "ping", "suspect"],
        },
        {
            "name": "testing",
            "task": (
                "Test gossip protocol under adversarial conditions:\n"
                "1. Node join: new node discovers full cluster within O(log N) rounds\n"
                "2. Node failure: dead node detected within expected time bound\n"
                "3. Network partition: nodes in minority partition mark majority as suspect\n"
                "4. Message loss: 30%% packet loss, verify protocol still converges\n"
                "5. Rapid churn: 10 nodes join/leave simultaneously\n"
                "Use pytest-asyncio with a simulated network layer."
            ),
            "depends_on": ["implementation"],
            "checks": ["def test_", "assert", "async", "partition", "failure"],
        },
    ],
    "domain": "system_design",
}


class MultiAgentEval:
    def __init__(self) -> None:
        self.results: List[Dict[str, Any]] = []
        self._orch = None
        self._learning = None

    def _get_orchestrator(self) -> Any:
        if self._orch is None:
            from Jotty.core.intelligence.orchestration.core.swarm_manager import Orchestrator

            self._orch = Orchestrator()
        return self._orch

    def _get_learning(self) -> Any:
        if self._learning is None:
            from Jotty.core.intelligence.learning.learning_service import LearningService

            self._learning = LearningService.get_instance()
        return self._learning

    def _get_learning_pipeline(self) -> Any:
        orch = self._get_orchestrator()
        # The learning pipeline lives on the Orchestrator (which IS the SwarmManager)
        # It's a LazyComponent that initializes on first access
        try:
            if "_lazy_learning" in orch.__dict__:
                return orch.learning
            # Try to access — may trigger lazy init
            lp = orch.learning
            return lp
        except Exception as e:
            print(f"  [debug] learning pipeline access failed: {e}")
            return None

    def _snapshot(self, label: str, domain: str = "") -> Dict[str, Any]:
        ls = self._get_learning()
        guidance = ls.query(domain or "general", "")
        ctx = ls.build_context_string(domain or "general", "")
        return {
            "label": label,
            "episode_count": guidance.get("total_episodes", 0),
            "success_rate": guidance.get("success_rate", 0.0),
            "context_length": len(ctx),
            "has_learning": guidance.get("has_learning", False),
        }

    async def run_pipeline_task(self, task: Dict[str, Any], learn: bool = True) -> Dict[str, Any]:
        """Run a multi-stage pipeline task through the orchestrator."""
        subsep(f"Pipeline: {task['id']}{'' if learn else ' [NO LEARNING]'}")

        orch = self._get_orchestrator()
        ls = self._get_learning()

        from Jotty.core.intelligence.learning.learning_service import classify_domain

        det_domain, det_task = classify_domain(task["goal"])
        before_snap = self._snapshot(f"before_{task['id']}", det_domain)

        print(f"  Domain: {det_domain}/{det_task}")
        print(f"  Stages: {len(task['stages'])}")
        print(f"  Learning state: {before_snap['episode_count']} episodes")

        # Build stages for the orchestrator pipeline
        stages = []
        for stage_def in task["stages"]:
            stage = {
                "name": stage_def["name"],
                "task": stage_def["task"],
                "task_type": stage_def["name"],
            }
            if "depends_on" in stage_def:
                stage["depends_on"] = stage_def["depends_on"]

            # Use a callable (async function) for each stage that calls chat()
            async def make_stage_fn(stage_task: str, stage_name: str, deps: List[str]):
                async def stage_fn(context: Dict[str, Any]) -> str:
                    prev = context.get("previous_outputs", {})
                    full_task = stage_task
                    if prev:
                        prev_context = "\n".join(
                            f"[{k} output summary]: {str(v)[:2000]}" for k, v in prev.items()
                        )
                        full_task = f"{prev_context}\n\nBased on the above, {full_task}"

                    result = await orch.chat(
                        message=full_task,
                        provider="anthropic",
                        model=EVAL_MODEL,
                        learn=learn,
                        enabled_tools=[],
                        max_steps=1,
                    )
                    return getattr(result, "content", str(result))

                return stage_fn

            stage["callable"] = await make_stage_fn(
                stage_def["task"],
                stage_def["name"],
                stage_def.get("depends_on", []),
            )
            stages.append(stage)

        start = time.time()
        error_msg = None
        result = None
        stage_details: List[Dict[str, Any]] = []

        try:
            result = await orch.run(
                goal=task["goal"],
                stages=stages,
                learn=learn,
                provider="anthropic",
            )

            # Extract stage results
            if hasattr(result, "metadata") and result.metadata:
                raw_stages = result.metadata.get("stage_results", {})
                for sname, sdata in raw_stages.items():
                    output_text = str(sdata.get("output", ""))
                    stage_details.append(
                        {
                            "name": sname,
                            "success": sdata.get("success", False),
                            "time": sdata.get("time", 0),
                            "output_length": len(output_text),
                        }
                    )
        except Exception as e:
            error_msg = str(e)
            print(f"  ERROR: {error_msg[:200]}")

        elapsed = time.time() - start
        after_snap = self._snapshot(f"after_{task['id']}", det_domain)

        # Analyze overall result
        content = ""
        if result:
            # Pipeline stores per-stage metadata
            meta = getattr(result, "metadata", None) or {}
            if not meta and hasattr(result, "override_metadata"):
                meta = result.override_metadata or {}
            raw_stages = meta.get("stages", {})
            for sname, sdata in raw_stages.items():
                stage_details.append(
                    {
                        "name": sname,
                        "success": sdata.get("success", False),
                        "time": sdata.get("time", 0),
                        "output_length": sdata.get("output_length", 0),
                    }
                )

            # Combined output from all stages
            if hasattr(result, "output") and result.output:
                content = result.output
            elif hasattr(result, "content") and result.content:
                content = result.content
            else:
                content = str(result)

        # Check passes across all stages
        all_checks = {}
        for stage_def in task["stages"]:
            for check in stage_def.get("checks", []):
                found = check.lower() in content.lower()
                all_checks[f"{stage_def['name']}:{check}"] = found

        checks_passed = sum(1 for v in all_checks.values() if v)
        checks_total = len(all_checks)

        # Quality assessment
        from Jotty.core.intelligence.learning.learning_service import analyze_response

        ra = analyze_response(content, task["goal"]) if content else {}
        heuristic_q = ra.get("quality_score", 0.5)
        check_ratio = checks_passed / max(checks_total, 1)
        quality_score = (
            check_ratio * 0.4 + heuristic_q * 0.4 + (0.2 if len(content) > 5000 else 0.1)
        )

        success = checks_passed >= checks_total * 0.5 and len(content) > 2000

        record = {
            "task_id": task["id"],
            "domain": det_domain,
            "learn": learn,
            "success": success,
            "quality_score": quality_score,
            "content_length": len(content),
            "checks_passed": checks_passed,
            "checks_total": checks_total,
            "elapsed_seconds": elapsed,
            "episodes_before": before_snap["episode_count"],
            "episodes_after": after_snap["episode_count"],
            "stages": stage_details,
            "error": error_msg,
        }
        self.results.append(record)

        status = "PASS" if success else "FAIL"
        print(
            f"  [{status}] Quality={quality_score:.2f} | "
            f"Checks={checks_passed}/{checks_total} | "
            f"Length={len(content)} | Time={elapsed:.1f}s"
        )
        print(
            f"  Episodes: {before_snap['episode_count']} → {after_snap['episode_count']} "
            f"(+{after_snap['episode_count'] - before_snap['episode_count']})"
        )
        if stage_details:
            for sd in stage_details:
                s = "OK" if sd["success"] else "FAIL"
                print(
                    f"    Stage {sd['name']:15s}: [{s}] {sd['output_length']} chars, {sd['time']:.1f}s"
                )

        for check_name, passed in all_checks.items():
            marker = "+" if passed else "-"
            print(f"    [{marker}] {check_name}")

        return record

    async def run_single_task(self, task: Dict[str, Any], learn: bool = True) -> Dict[str, Any]:
        """Run a single-shot task (no pipeline) for comparison."""
        subsep(f"Single-shot: {task['id']}{'' if learn else ' [NO LEARNING]'}")

        orch = self._get_orchestrator()
        ls = self._get_learning()
        from Jotty.core.intelligence.learning.learning_service import classify_domain

        det_domain, _ = classify_domain(task["goal"])
        before_snap = self._snapshot(f"before_{task['id']}", det_domain)

        start = time.time()
        content = ""
        error_msg = None

        try:
            result = await orch.chat(
                message=task["goal"],
                provider="anthropic",
                model=EVAL_MODEL,
                learn=learn,
                enabled_tools=[],
                max_steps=1,
            )
            content = getattr(result, "content", str(result))
        except Exception as e:
            error_msg = str(e)
            print(f"  ERROR: {error_msg[:200]}")

        elapsed = time.time() - start
        after_snap = self._snapshot(f"after_{task['id']}", det_domain)

        checks_passed = sum(1 for c in task["checks"] if c.lower() in content.lower())
        checks_total = len(task["checks"])

        from Jotty.core.intelligence.learning.learning_service import analyze_response

        ra = analyze_response(content, task["goal"]) if content else {}
        heuristic_q = ra.get("quality_score", 0.5)
        check_ratio = checks_passed / max(checks_total, 1)
        quality_score = (
            check_ratio * 0.4 + heuristic_q * 0.4 + (0.2 if len(content) > 5000 else 0.1)
        )

        success = checks_passed >= checks_total * 0.5 and len(content) > 2000

        record = {
            "task_id": task["id"],
            "domain": det_domain,
            "learn": learn,
            "success": success,
            "quality_score": quality_score,
            "content_length": len(content),
            "checks_passed": checks_passed,
            "checks_total": checks_total,
            "elapsed_seconds": elapsed,
            "episodes_before": before_snap["episode_count"],
            "episodes_after": after_snap["episode_count"],
            "error": error_msg,
        }
        self.results.append(record)

        status = "PASS" if success else "FAIL"
        print(
            f"  [{status}] Quality={quality_score:.2f} | "
            f"Checks={checks_passed}/{checks_total} | "
            f"Length={len(content)} | Time={elapsed:.1f}s"
        )

        return record

    def _print_credit_analysis(self) -> None:
        """Print credit assignment state from the learning pipeline."""
        subsep("CREDIT ASSIGNMENT ANALYSIS")

        lp = self._get_learning_pipeline()
        if not lp:
            print("  Learning pipeline not available (orchestrator not initialized).")
            return

        # Credit assignment state
        if hasattr(lp, "credit_assigner"):
            ca = lp.credit_assigner
            print(f"  Credit Assigner: {type(ca).__name__}")

            if ca.improvement_credits:
                print(f"  Credits recorded: {len(ca.improvement_credits)} patterns")
                for cid, credit in list(ca.improvement_credits.items())[:8]:
                    imp = credit.improvement or {}
                    stage = imp.get("stage", imp.get("agent", "?"))
                    ctype = imp.get("type", "unknown")
                    ctxs = credit.contexts[-1] if credit.contexts else ""
                    marginal = ""
                    if ctxs and "marginal" in str(ctxs):
                        marginal = f" (marginal in context)"
                    print(
                        f"    [{ctype:14s}] {stage:20s} "
                        f"direct={credit.direct_credit:.3f} "
                        f"apps={credit.application_count} "
                        f"success={credit.success_count}"
                    )
            else:
                print("  No credit records yet.")

            try:
                stats = ca.get_credit_statistics()
                if stats:
                    # Compact display
                    print(
                        f"\n  Credit summary: "
                        f"{stats.get('total_improvements', 0)} patterns, "
                        f"{stats.get('total_applications', 0)} applications, "
                        f"avg_direct={stats.get('avg_direct_credit', 0):.3f}"
                    )
            except Exception as e:
                print(f"  Stats error: {e}")
        else:
            print("  No credit_assigner found on learning pipeline.")

        # Credit weights
        if hasattr(lp, "credit_weights"):
            cw = lp.credit_weights
            print(f"\n  Adaptive credit weights: {cw}")

        # Effectiveness tracker
        if hasattr(lp, "effectiveness"):
            eff = lp.effectiveness
            report = eff.improvement_report()
            if report:
                print(f"\n  Effectiveness report:")
                for task_type, stats in report.items():
                    if task_type == "__global__":
                        continue
                    print(
                        f"    {task_type:20s}: recent={stats.get('recent_rate', 0):.2f} "
                        f"historical={stats.get('historical_rate', 0):.2f} "
                        f"trend={stats.get('trend', 0):+.2f}"
                    )

        # Stigmergy trails
        if hasattr(lp, "stigmergy"):
            stig = lp.stigmergy
            trails = stig.get_all_trails() if hasattr(stig, "get_all_trails") else {}
            if trails:
                print(f"\n  Stigmergy trails: {len(trails)} active")
                for trail_key, trail_data in list(trails.items())[:5]:
                    print(f"    - {trail_key}: strength={trail_data.get('strength', 0):.3f}")

    def _print_learning_analysis(self) -> None:
        """Print learning state after multi-agent runs."""
        subsep("LEARNING STATE ANALYSIS")

        ls = self._get_learning()

        # Per-domain episode counts
        for domain in ["system_design", "data_science", "coding", "general"]:
            g = ls.query(domain, "")
            if g.get("has_learning"):
                ctx = ls.build_context_string(domain, "")
                retrieval = ls.build_retrieval_context(domain, "", "")
                print(
                    f"  {domain:15s}: {g['total_episodes']} episodes, "
                    f"rate={g['success_rate']:.0%}, "
                    f"context={len(ctx)} chars, "
                    f"retrieval={len(retrieval)} chars"
                )

                # Show what context actually says
                if ctx:
                    print(f"    Context preview: {ctx[:200]}...")

        # Pattern analysis
        try:
            patterns = ls._store.get_patterns(limit=100)
            by_type = {}
            for p in patterns:
                t = p.pattern_type
                by_type[t] = by_type.get(t, 0) + 1

            print(f"\n  Total patterns: {len(patterns)}")
            for ptype, count in sorted(by_type.items(), key=lambda x: -x[1]):
                print(f"    {ptype:25s}: {count}")

            # Show causal patterns
            causal = [p for p in patterns if p.pattern_type == "causal"]
            if causal:
                print(f"\n  Causal patterns ({len(causal)}):")
                for c in causal[:5]:
                    print(f"    - [{', '.join(c.applicable_domains)}] {c.recommendation}")
        except Exception as e:
            print(f"  Pattern analysis error: {e}")

    def _print_results(self) -> None:
        """Print final results — learning curve and A/B comparison."""
        subsep("FINAL RESULTS")

        # All results table
        for r in self.results:
            status = "PASS" if r["success"] else "FAIL"
            learn_tag = "" if r.get("learn", True) else " [NO-LEARN]"
            print(
                f"  [{status}] {r['task_id']:25s} Q={r['quality_score']:.2f} "
                f"Checks={r['checks_passed']}/{r['checks_total']} "
                f"Len={r['content_length']:6d} Time={r['elapsed_seconds']:.0f}s "
                f"Ep={r['episodes_before']}→{r['episodes_after']}{learn_tag}"
            )

        # Learning curve
        curve_results = [
            r for r in self.results if r["task_id"].startswith("R") and r.get("learn", True)
        ]
        if len(curve_results) >= 2:
            subsep("LEARNING CURVE")
            qualities = [r["quality_score"] for r in curve_results]
            checks = [r["checks_passed"] / max(r["checks_total"], 1) for r in curve_results]
            episodes = [r["episodes_before"] for r in curve_results]

            for i, r in enumerate(curve_results):
                bar_len = int(r["quality_score"] * 40)
                bar = "█" * bar_len + "░" * (40 - bar_len)
                print(
                    f"  R{i+1} {bar} Q={r['quality_score']:.2f} "
                    f"Chk={r['checks_passed']}/{r['checks_total']} "
                    f"Ep={r['episodes_before']}"
                )

            first_q = qualities[0]
            last_q = qualities[-1]
            trend = last_q - first_q
            best_q = max(qualities)
            worst_q = min(qualities)
            print(f"\n  Trend: R1={first_q:.2f} → R{len(qualities)}={last_q:.2f} ({trend:+.2f})")
            print(f"  Range: {worst_q:.2f} — {best_q:.2f}")
            print(f"  Check trend: R1={checks[0]:.0%} → R{len(checks)}={checks[-1]:.0%}")

        # A/B comparison
        ab_no = [r for r in self.results if r["task_id"] == "AB_no_learn"]
        ab_yes = [r for r in self.results if r["task_id"] == "AB_with_learn"]
        if ab_no and ab_yes:
            subsep("A/B TEST — SAME TASK")
            no_q = ab_no[0]["quality_score"]
            yes_q = ab_yes[0]["quality_score"]
            no_c = ab_no[0]["checks_passed"] / max(ab_no[0]["checks_total"], 1)
            yes_c = ab_yes[0]["checks_passed"] / max(ab_yes[0]["checks_total"], 1)
            delta = yes_q - no_q
            winner = (
                "LEARNING WINS"
                if delta > 0.02
                else ("NO DIFFERENCE" if abs(delta) <= 0.02 else "BASELINE WINS")
            )
            print(f"  Without learning: Q={no_q:.2f} Checks={no_c:.0%}")
            print(f"  With learning:    Q={yes_q:.2f} Checks={yes_c:.0%}")
            print(f"  Delta: {delta:+.2f} → {winner}")

        # Scores
        subsep("SCORES")
        scores = {}

        # 1. Success rate
        all_results = self.results
        scores["Success Rate"] = sum(1 for r in all_results if r["success"]) / max(
            len(all_results), 1
        )

        # 2. Average quality
        learned = [r for r in all_results if r.get("learn", True)]
        if learned:
            scores["Avg Quality"] = sum(r["quality_score"] for r in learned) / len(learned)

        # 3. Learning active (episodes recorded)
        episodes_added = sum(
            max(0, r["episodes_after"] - r["episodes_before"])
            for r in all_results
            if r.get("learn", True)
        )
        scores["Episodes Recorded"] = min(1.0, episodes_added / 10)

        # 4. Learning curve trend (positive = learning helps over time)
        if len(curve_results) >= 2:
            trend_val = qualities[-1] - qualities[0]
            if trend_val > 0.03:
                scores["Learning Curve"] = min(1.0, 0.5 + trend_val * 5)
            elif trend_val >= -0.02:
                scores["Learning Curve"] = 0.5
            else:
                scores["Learning Curve"] = max(0.0, 0.5 + trend_val * 5)

        # 5. A/B test (learning helps on same task)
        if ab_no and ab_yes:
            ab_delta = ab_yes[0]["quality_score"] - ab_no[0]["quality_score"]
            if ab_delta > 0.02:
                scores["A/B Learning Δ"] = min(1.0, 0.5 + ab_delta * 5)
            elif ab_delta >= -0.02:
                scores["A/B Learning Δ"] = 0.5
            else:
                scores["A/B Learning Δ"] = max(0.0, 0.5 + ab_delta * 5)

        # 6. Credit assignment
        lp = self._get_learning_pipeline()
        if lp and hasattr(lp, "credit_assigner"):
            credits = lp.credit_assigner.improvement_credits
            stage_credits = [
                c for c in credits.values() if (c.improvement or {}).get("type") == "pipeline_stage"
            ]
            ca_score = min(0.5, len(stage_credits) / 6)
            if stage_credits and any(c.success_count > 0 for c in stage_credits):
                ca_score += 0.3
            if hasattr(lp, "episode_count") and lp.episode_count > 0:
                ca_score += 0.2
            scores["Credit Assignment"] = min(1.0, ca_score)

        for name, score in scores.items():
            bar = "#" * int(score * 20) + "-" * (20 - int(score * 20))
            print(f"  {name:25s} [{bar}] {score * 100:5.1f}%")

        overall = sum(scores.values()) / max(len(scores), 1)
        stars = int(overall * 5 + 0.5)
        star_str = "*" * stars + "." * (5 - stars)
        print(f"\n  OVERALL: [{star_str}] {overall * 100:.1f}%")

    async def run_all(self) -> None:
        separator("JOTTY LEARNING CURVE EVALUATION")
        print(f"  Date: {datetime.now().isoformat()}")
        print(f"  Model: {EVAL_MODEL}")
        print(f"  Learning curve: {len(LEARNING_CURVE_TASKS)} tasks (same domain, sequential)")
        print(f"  A/B test: same task with vs without learning")
        print(f"  Goal: does quality improve as learning accumulates?")

        initial_snap = self._snapshot("initial", "system_design")
        print(f"  Initial episodes: {initial_snap['episode_count']}")

        # ── LEARNING CURVE: Run N tasks sequentially, all system_design ──
        for i, task in enumerate(LEARNING_CURVE_TASKS):
            sep_label = f"ROUND {i+1}/{len(LEARNING_CURVE_TASKS)} — {task['id']}"
            if i == 0:
                sep_label += " (cold start, 0 prior episodes)"
            else:
                snap = self._snapshot(f"before_R{i+1}", "system_design")
                sep_label += f" ({snap['episode_count']} episodes of learning)"
            separator(sep_label)
            await self.run_pipeline_task(task, learn=True)
            print("  Waiting for background learning...")
            await asyncio.sleep(3)

        # ── A/B TEST: Same task, learn=False vs learn=True ──
        separator("A/B TEST — Same task: without learning vs with learning")
        snap = self._snapshot("before_ab", "system_design")
        print(f"  Episodes available: {snap['episode_count']}")

        print("\n  --- A: NO LEARNING (learn=False) ---")
        await self.run_pipeline_task({**AB_TEST_TASK, "id": "AB_no_learn"}, learn=False)

        print("\n  --- B: WITH LEARNING (learn=True) ---")
        await self.run_pipeline_task({**AB_TEST_TASK, "id": "AB_with_learn"}, learn=True)
        await asyncio.sleep(3)

        # ── ANALYSIS ──
        separator("ANALYSIS")
        self._print_credit_analysis()
        self._print_learning_analysis()
        self._print_results()

        results_dir = Path(__file__).parent / "eval_results"
        results_dir.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = results_dir / f"{ts}_learning_curve.json"
        with open(results_file, "w") as f:
            json.dump(
                {
                    "timestamp": datetime.now().isoformat(),
                    "model": EVAL_MODEL,
                    "results": self.results,
                },
                f,
                indent=2,
                default=str,
            )
        print(f"\n  Report saved: {results_file}")


async def main() -> None:
    evaluator = MultiAgentEval()
    await evaluator.run_all()


if __name__ == "__main__":
    asyncio.run(main())
