#!/usr/bin/env python3
"""
Jotty HARD Learning Evaluation
===============================

Genuinely brutal tasks that weak models FAIL at:
- Multi-Paxos (not Raft — much harder)
- Lock-free data structures with ABA prevention
- Byzantine fault tolerance with crypto proofs
- Distributed transactions with saga compensation

Checks are STRUCTURAL, not keyword matches:
- Regex patterns verifying actual algorithm logic
- Code block counting (must have substantial code)
- Section completeness (all N parts with substance)
- Specific reasoning chains verified
- 70% threshold to pass (not 50%)
"""

import asyncio
import json
import logging
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("eval_hard")
logger.setLevel(logging.INFO)

EVAL_MODEL = "claude-3-haiku-20240307"


def separator(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}\n")


def subsep(title: str) -> None:
    print(f"\n  --- {title} ---\n")


# =============================================================================
# SMART CHECKS — regex patterns, structural verification, not keyword matching
# =============================================================================

def check_regex(content: str, pattern: str, flags: int = re.IGNORECASE) -> bool:
    return bool(re.search(pattern, content, flags))


def check_code_blocks(content: str, min_blocks: int = 3, min_lines_per_block: int = 8) -> bool:
    blocks = re.findall(r'```[\w]*\n(.*?)```', content, re.DOTALL)
    substantial = [b for b in blocks if len(b.strip().split('\n')) >= min_lines_per_block]
    return len(substantial) >= min_blocks


def check_sections(content: str, required_sections: List[str]) -> Tuple[int, int]:
    """Check that content has substantial sections (>100 chars each)."""
    found = 0
    for section in required_sections:
        pattern = rf'(?:#{1,3}\s*.*?{re.escape(section)}|(?:^|\n)\**\s*{re.escape(section)})'
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            after = content[match.end():match.end() + 500]
            if len(after.strip()) > 100:
                found += 1
    return found, len(required_sections)


def check_test_assertions(content: str, min_assertions: int = 5) -> bool:
    assertions = re.findall(r'assert\s+\w+', content)
    return len(assertions) >= min_assertions


def check_math_reasoning(content: str, min_formulas: int = 2) -> bool:
    formulas = re.findall(r'O\([^)]+\)|Θ\([^)]+\)|n\s*(?:log|lg)\s*n|2\^n|n\^[23k]|≤|≥|∀|∃|⟹|→', content)
    return len(formulas) >= min_formulas


# =============================================================================
# HARD TASKS — genuinely difficult, specific, verifiable
# =============================================================================

HARD_TASKS = [
    {
        "id": "H1_multi_paxos",
        "goal": (
            "Implement Multi-Paxos consensus protocol (NOT Raft — Paxos is fundamentally different "
            "and harder).\n\n"
            "PART A — PROTOCOL SPECIFICATION:\n"
            "1. Define the EXACT message flow for Single-Decree Paxos: Prepare(n) → Promise(n, v_accepted, n_accepted) "
            "→ Accept(n, v) → Accepted(n, v). Explain why proposal numbers must be globally unique and monotonically increasing.\n"
            "2. Extend to Multi-Paxos: explain how a stable leader eliminates Phase 1 for subsequent slots. "
            "Define the slot-based log structure. Explain what happens when the leader changes mid-sequence — "
            "which slots need re-proposing?\n"
            "3. Prove SAFETY: no two different values can be chosen for the same slot. The proof must reference "
            "the quorum intersection property (any two majorities share at least one member) and explain "
            "how Promise messages prevent conflicting Accept messages.\n"
            "4. Prove LIVENESS requires leader election: show a concrete scenario where two proposers "
            "with alternating proposal numbers create a livelock (proposer dueling).\n\n"
            "PART B — IMPLEMENTATION:\n"
            "Implement Multi-Paxos in Python with these EXACT classes and methods:\n"
            "- PaxosNode with propose(value, slot), handle_prepare(msg), handle_promise(msg), "
            "handle_accept(msg), handle_accepted(msg)\n"
            "- ProposalNumber: (counter, node_id) with comparison operators\n"
            "- PaxosLog: slot-indexed log with gaps allowed, committed/uncommitted tracking\n"
            "- LeaderElection: leader lease with term numbers, handle_leader_change()\n"
            "Handle: proposal number conflicts, promise with already-accepted value (must use highest-numbered "
            "accepted value), log gaps, out-of-order messages.\n\n"
            "PART C — CORRECTNESS TESTS:\n"
            "Write tests proving each safety property:\n"
            "- test_no_split_decision: 5 nodes, concurrent proposals for same slot, verify exactly one wins\n"
            "- test_accepted_value_preserved: proposer 2 must use value from proposer 1's accepted-but-not-committed value\n"
            "- test_leader_change_recovery: leader dies at slot 5, new leader must re-propose uncommitted slots\n"
            "- test_proposer_dueling: demonstrate livelock with 2 proposers, then show leader election fixes it\n"
            "- test_log_gaps: commit slot 3 before slot 2, verify both eventually committed"
        ),
        "checks": [
            # PART A: Protocol reasoning (not just keywords — actual logic)
            ("Prepare-Promise flow", lambda c: check_regex(c, r'[Pp]repare\s*\(\s*n\s*\).*?[Pp]romise', re.DOTALL)),
            ("Accept-Accepted flow", lambda c: check_regex(c, r'[Aa]ccept\s*\(\s*n.*?\).*?[Aa]ccepted', re.DOTALL)),
            ("Proposal numbers globally unique", lambda c: check_regex(c, r'propos\w+\s+number\w*\s+.*(?:unique|global|monoton)', re.DOTALL)),
            ("Quorum intersection proof", lambda c: check_regex(c, r'(?:quorum|majorit)\w*\s+.*(?:intersect|share|overlap|common)', re.DOTALL)),
            ("Proposer dueling/livelock scenario", lambda c: check_regex(c, r'(?:duel|livelock|alternat\w+\s+propos)', re.DOTALL)),
            ("Multi-Paxos leader eliminates Phase 1", lambda c: check_regex(c, r'(?:leader|stable).*(?:eliminat|skip|bypass).*(?:phase\s*1|prepare)', re.DOTALL)),
            ("Slot-based log structure", lambda c: check_regex(c, r'slot.*(?:log|index|sequence)', re.DOTALL)),

            # PART B: Implementation (actual code patterns, not just class names)
            ("PaxosNode.propose method", lambda c: check_regex(c, r'def\s+propose\s*\(self.*(?:value|slot)', re.DOTALL)),
            ("handle_prepare with proposal comparison", lambda c: check_regex(c, r'def\s+handle_prepare.*(?:proposal|number).*(?:>|<|>=|<=|compar)', re.DOTALL)),
            ("handle_promise collects responses", lambda c: check_regex(c, r'def\s+handle_promise.*(?:promise|response|collect|count)', re.DOTALL)),
            ("ProposalNumber comparison", lambda c: check_regex(c, r'(?:__lt__|__gt__|__le__|__ge__|<|>).*(?:counter|node_id|proposal)', re.DOTALL)),
            ("Log with gap handling", lambda c: check_regex(c, r'(?:gap|slot.*None|uncommitted.*slot|slot.*missing)', re.DOTALL)),
            ("Highest accepted value used", lambda c: check_regex(c, r'(?:highest|max|largest).*(?:accepted|promised).*(?:value|proposal)', re.DOTALL)),
            ("Substantial code (3+ blocks, 8+ lines each)", lambda c: check_code_blocks(c, 3, 8)),

            # PART C: Tests (actual test logic, not just test_ prefix)
            ("test_no_split_decision with assertion", lambda c: check_regex(c, r'def\s+test_\w*split\w*.*assert', re.DOTALL)),
            ("test_accepted_value_preserved", lambda c: check_regex(c, r'def\s+test_\w*(?:accepted|preserved|value)\w*', re.DOTALL)),
            ("test_leader_change_recovery", lambda c: check_regex(c, r'def\s+test_\w*(?:leader|recovery|change)\w*', re.DOTALL)),
            ("test_proposer_dueling", lambda c: check_regex(c, r'def\s+test_\w*(?:duel|livelock)\w*', re.DOTALL)),
            ("5+ test assertions total", lambda c: check_test_assertions(c, 5)),
            ("Complexity analysis", lambda c: check_math_reasoning(c, 2)),
        ],
        "domain": "system_design",
    },
    {
        "id": "H2_lock_free_queue",
        "goal": (
            "Implement a lock-free concurrent queue with ABA problem prevention.\n\n"
            "PART A — THEORY (must be precise, not hand-wavy):\n"
            "1. Define linearizability formally: every concurrent execution has an equivalent sequential "
            "execution where each operation takes effect at a single linearization point between its "
            "invocation and response. Give a concrete example: 2 threads doing enqueue(1) and dequeue() — "
            "show all possible linearizations.\n"
            "2. Explain the ABA problem with a SPECIFIC scenario: Thread T1 reads head=A, gets preempted. "
            "Thread T2 pops A, pushes B, pushes A back. T1 resumes, CAS succeeds (head still A), but the "
            "queue state is wrong. Draw the exact pointer states at each step.\n"
            "3. Compare THREE solutions to ABA: (a) tagged pointers with version counter — show bit layout "
            "(48-bit pointer + 16-bit counter on 64-bit), (b) hazard pointers — explain the scan/retire/reclaim "
            "cycle, (c) epoch-based reclamation — explain grace periods. Analyze memory overhead of each.\n"
            "4. Prove that Michael-Scott queue is linearizable: identify the exact linearization points for "
            "enqueue (successful CAS on tail.next) and dequeue (successful CAS on head).\n\n"
            "PART B — IMPLEMENTATION:\n"
            "Implement Michael-Scott lock-free queue in Python with:\n"
            "- Node class with value and AtomicRef(next)\n"
            "- LockFreeQueue with enqueue(value) and dequeue() → Optional[T]\n"
            "- AtomicRef: simulated CAS with compare_and_swap(expected, new) → bool\n"
            "- Tagged pointer: AtomicStamped with (reference, stamp) to prevent ABA\n"
            "- enqueue must: create node, loop until CAS(tail.next, null, new_node) succeeds, "
            "then advance tail\n"
            "- dequeue must: loop until CAS(head, old_head, old_head.next) succeeds, handle empty queue\n"
            "- Both must handle helping: if tail is behind, advance it before retrying\n\n"
            "PART C — VERIFICATION:\n"
            "1. test_concurrent_enqueue_dequeue: 4 threads, 1000 ops each, verify no lost items "
            "(sum of all dequeued == sum of all enqueued)\n"
            "2. test_aba_prevention: deliberately create ABA scenario, verify tagged pointer catches it\n"
            "3. test_linearizability: record invocation/response timestamps, verify exists valid linearization\n"
            "4. test_empty_dequeue: concurrent dequeues on empty queue, verify all return None safely\n"
            "5. test_memory_ordering: verify enqueue-before-dequeue ordering using happens-before"
        ),
        "checks": [
            # PART A: Theory (actual reasoning, not keywords)
            ("Linearizability formal definition", lambda c: check_regex(c, r'lineariz\w+\s+point.*(?:invocation|response|effect)', re.DOTALL)),
            ("ABA scenario with specific steps", lambda c: check_regex(c, r'(?:T1|thread\s*1).*(?:preempt|suspend|pause).*(?:T2|thread\s*2).*(?:pop|push|remove)', re.DOTALL)),
            ("Tagged pointer bit layout", lambda c: check_regex(c, r'(?:48|16|64)\s*-?\s*bit.*(?:pointer|counter|tag)', re.DOTALL)),
            ("Hazard pointer explanation", lambda c: check_regex(c, r'hazard\s+pointer.*(?:scan|retire|reclaim|protect)', re.DOTALL)),
            ("Epoch-based reclamation", lambda c: check_regex(c, r'epoch.*(?:grace|period|reclaim|retire)', re.DOTALL)),
            ("Michael-Scott linearization points", lambda c: check_regex(c, r'(?:lineariz\w+\s+point|CAS.*tail\.next|CAS.*head)', re.DOTALL)),
            ("Memory overhead comparison", lambda c: check_regex(c, r'(?:memory|space|overhead).*(?:O\(|per\s+thread|per\s+node)', re.DOTALL)),

            # PART B: Implementation (actual algorithm logic)
            ("CAS operation implementation", lambda c: check_regex(c, r'def\s+compare_and_swap\s*\(self.*expected.*new', re.DOTALL)),
            ("enqueue with CAS loop", lambda c: check_regex(c, r'def\s+enqueue.*while.*(?:compare_and_swap|cas|CAS)', re.DOTALL)),
            ("dequeue with CAS loop", lambda c: check_regex(c, r'def\s+dequeue.*while.*(?:compare_and_swap|cas|CAS)', re.DOTALL)),
            ("Tail advancement helping", lambda c: check_regex(c, r'(?:tail|advance|help).*(?:behind|lagging|CAS.*tail)', re.DOTALL)),
            ("Tagged/stamped pointer for ABA", lambda c: check_regex(c, r'(?:stamp|tag|version|counter).*(?:CAS|compare_and_swap|atomic)', re.DOTALL)),
            ("Sentinel/dummy head node", lambda c: check_regex(c, r'(?:sentinel|dummy|head)\s*=\s*(?:Node|new)', re.DOTALL)),
            ("Substantial code", lambda c: check_code_blocks(c, 3, 8)),

            # PART C: Verification
            ("test_concurrent with 4 threads", lambda c: check_regex(c, r'def\s+test_\w*concurrent.*(?:thread|Thread|executor)', re.DOTALL)),
            ("test_aba_prevention", lambda c: check_regex(c, r'def\s+test_\w*aba', re.DOTALL)),
            ("Sum verification (enqueued == dequeued)", lambda c: check_regex(c, r'(?:sum|total|count).*(?:enqueue|dequeue).*(?:==|assert|equal)', re.DOTALL)),
            ("5+ assertions", lambda c: check_test_assertions(c, 5)),
            ("Complexity analysis", lambda c: check_math_reasoning(c, 2)),
        ],
        "domain": "system_design",
    },
    {
        "id": "H3_byzantine_bft",
        "goal": (
            "Implement Practical Byzantine Fault Tolerance (PBFT) with cryptographic verification.\n\n"
            "PART A — PROTOCOL (must be exact, not approximate):\n"
            "1. Define the Byzantine generals problem precisely: up to f faulty nodes out of 3f+1 total. "
            "Prove why 3f+1 is the minimum: show that with only 3f nodes, a Byzantine node can send "
            "conflicting messages to create an unresolvable split (give the EXACT 3-node scenario where "
            "1 Byzantine node prevents consensus).\n"
            "2. Describe PBFT's THREE phases with exact message counts:\n"
            "   - Pre-prepare: primary sends ⟨PRE-PREPARE, v, n, d⟩ — what does each field mean?\n"
            "   - Prepare: each replica sends ⟨PREPARE, v, n, d, i⟩ — when does replica accept? (need 2f matching)\n"
            "   - Commit: each replica sends ⟨COMMIT, v, n, d, i⟩ — when is request committed? (need 2f+1 matching)\n"
            "3. Explain view changes: when does a replica trigger view change? What's in the VIEW-CHANGE "
            "message (proof of prepared requests)? How does the new primary construct NEW-VIEW?\n"
            "4. Calculate total messages per request: pre-prepare(1) + prepare(n-1) + commit(n) = O(n²). "
            "Why is this a scalability bottleneck? How does HotStuff reduce to O(n)?\n\n"
            "PART B — IMPLEMENTATION:\n"
            "- PBFTNode with state machine: request → pre_prepare → prepare → commit → reply\n"
            "- MessageLog: tracks pre-prepare/prepare/commit per (view, sequence)\n"
            "- CryptoVerifier: sign_message(msg, private_key) and verify_signature(msg, sig, public_key) "
            "using HMAC-SHA256\n"
            "- ViewChangeManager: detect timeout, broadcast VIEW-CHANGE, construct NEW-VIEW\n"
            "- CheckpointManager: periodic stable checkpoints with 2f+1 proofs, garbage collect old logs\n"
            "Handle: duplicate messages, out-of-order delivery, primary failure detection.\n\n"
            "PART C — TESTS:\n"
            "- test_normal_case: 4 nodes (f=1), client sends request, verify all honest nodes reply with same result\n"
            "- test_byzantine_primary: primary sends conflicting pre-prepares, verify honest nodes detect and trigger view change\n"
            "- test_byzantine_replica: 1 of 4 replicas sends wrong prepare messages, verify protocol still commits correctly\n"
            "- test_view_change: kill primary, verify new primary elected and pending requests re-processed\n"
            "- test_message_counts: verify exactly (n-1) prepare messages and n commit messages per request\n"
            "- test_checkpoint: after 100 requests, verify checkpoint created and old logs pruned"
        ),
        "checks": [
            # PART A: Protocol reasoning
            ("3f+1 minimum with proof", lambda c: check_regex(c, r'3\s*f\s*\+\s*1.*(?:minimum|require|need|at\s+least)', re.DOTALL)),
            ("3-node impossibility scenario", lambda c: check_regex(c, r'(?:3\s*node|three\s*node|1\s*byzantine).*(?:conflicting|different|split|impossible)', re.DOTALL)),
            ("PRE-PREPARE message format", lambda c: check_regex(c, r'PRE.?PREPARE.*(?:view|sequence|digest|primary)', re.DOTALL)),
            ("2f prepare threshold", lambda c: check_regex(c, r'2\s*f\s*(?:\+?\s*1)?\s*(?:matching|prepare|message|quorum)', re.DOTALL)),
            ("2f+1 commit threshold", lambda c: check_regex(c, r'2\s*f\s*\+\s*1\s*(?:commit|matching|message|quorum)', re.DOTALL)),
            ("View change trigger mechanism", lambda c: check_regex(c, r'(?:timeout|view.?change).*(?:trigger|detect|suspect|primary\s+fail)', re.DOTALL)),
            ("O(n²) message complexity", lambda c: check_regex(c, r'O\(\s*n\s*[²2\^]\s*\)', re.DOTALL)),
            ("HotStuff O(n) comparison", lambda c: check_regex(c, r'HotStuff.*O\(\s*n\s*\)', re.DOTALL)),

            # PART B: Implementation
            ("PBFT state machine implementation", lambda c: check_regex(c, r'class\s+PBFTNode.*def\s+(?:handle|process|on)_(?:pre_prepare|prepare|commit)', re.DOTALL)),
            ("MessageLog per (view, sequence)", lambda c: check_regex(c, r'(?:view|sequence).*(?:log|dict|map|track)', re.DOTALL)),
            ("HMAC or crypto signing", lambda c: check_regex(c, r'(?:hmac|HMAC|sign|signature|SHA|sha256|digest|hashlib)', re.DOTALL)),
            ("View change manager", lambda c: check_regex(c, r'(?:view.?change|ViewChange).*(?:class|def|broadcast|new.?view)', re.DOTALL)),
            ("Checkpoint with 2f+1 proofs", lambda c: check_regex(c, r'checkpoint.*(?:2f|proof|stable|garbage|prune)', re.DOTALL)),
            ("Duplicate message handling", lambda c: check_regex(c, r'(?:duplicate|already|seen|processed).*(?:message|request|ignore|skip)', re.DOTALL)),
            ("Substantial code", lambda c: check_code_blocks(c, 3, 8)),

            # PART C: Tests
            ("test_normal_case with 4 nodes", lambda c: check_regex(c, r'def\s+test_\w*normal.*(?:4|four|3f\+1)', re.DOTALL)),
            ("test_byzantine with conflicting messages", lambda c: check_regex(c, r'def\s+test_\w*byzantine.*(?:conflict|wrong|malicious)', re.DOTALL)),
            ("test_view_change", lambda c: check_regex(c, r'def\s+test_\w*view.*(?:change|new.*primary|elect)', re.DOTALL)),
            ("5+ assertions", lambda c: check_test_assertions(c, 5)),
            ("Math reasoning", lambda c: check_math_reasoning(c, 3)),
        ],
        "domain": "system_design",
    },
    {
        "id": "H4_saga_2pc",
        "goal": (
            "Implement a distributed transaction system supporting BOTH Two-Phase Commit (2PC) AND "
            "the Saga pattern, with formal analysis of when each is appropriate.\n\n"
            "PART A — FORMAL ANALYSIS:\n"
            "1. Define ACID precisely. Then define BASE (Basically Available, Soft state, Eventually consistent). "
            "Prove that 2PC provides ACID but blocks during coordinator failure — show the exact blocking "
            "scenario: participant voted YES, coordinator dies. The participant CANNOT safely abort (another "
            "participant may have committed) NOR commit (coordinator may have decided abort). This is the "
            "fundamental impossibility.\n"
            "2. Define Saga formally: a sequence of local transactions T1...Tn with compensating transactions "
            "C1...Cn-1. If Ti fails, execute Ci-1...C1 in reverse order. Prove Saga provides ACD but NOT "
            "Isolation — give a concrete dirty read scenario where a concurrent transaction sees intermediate "
            "saga state.\n"
            "3. Compare: 2PC = strong consistency + blocking risk. Saga = eventual consistency + no blocking "
            "but no isolation. Give SPECIFIC use cases for each: 2PC for bank transfers (need atomicity), "
            "Saga for e-commerce order (can compensate: cancel shipment, refund payment).\n"
            "4. Define the Saga Execution Coordinator (SEC) pattern: orchestration (central coordinator calls "
            "each step) vs choreography (each service publishes events, next service reacts). "
            "Analyze failure modes of each.\n\n"
            "PART B — IMPLEMENTATION:\n"
            "Implement BOTH protocols:\n\n"
            "2PC:\n"
            "- Coordinator: prepare_all() → collect_votes() → commit_all() or abort_all()\n"
            "- Participant: vote(tx_id) → {YES, NO}, commit(tx_id), abort(tx_id)\n"
            "- TransactionLog: write-ahead log with PREPARE, COMMIT, ABORT records for crash recovery\n"
            "- Handle: coordinator crash after prepare (participants timeout and query), "
            "participant crash after vote (coordinator retries), network partition\n\n"
            "Saga:\n"
            "- SagaOrchestrator: execute_saga(steps) with forward execution and backward compensation\n"
            "- SagaStep: action + compensating_action + idempotency_key\n"
            "- CompensationLog: track which steps completed for crash recovery\n"
            "- Handle: compensation failure (retry with exponential backoff), partial compensation, "
            "idempotency (same request processed twice = same result)\n\n"
            "PART C — TESTS:\n"
            "- test_2pc_happy_path: all participants vote YES, verify all committed\n"
            "- test_2pc_one_no_vote: one participant votes NO, verify all aborted\n"
            "- test_2pc_coordinator_crash: coordinator dies after prepare, verify participants eventually resolve\n"
            "- test_saga_happy_path: all steps succeed in order\n"
            "- test_saga_step3_fails: steps 1,2 succeed, step 3 fails — verify C2, C1 executed in reverse\n"
            "- test_saga_compensation_failure: compensation C2 fails, verify retry with backoff\n"
            "- test_saga_idempotency: execute same saga twice with same key, verify single execution\n"
            "- test_saga_dirty_read: concurrent saga sees intermediate state (demonstrate isolation gap)"
        ),
        "checks": [
            # PART A: Formal analysis
            ("ACID definition", lambda c: check_regex(c, r'[Aa]tomic\w*.*[Cc]onsisten\w*.*[Ii]solat\w*.*[Dd]urabil', re.DOTALL)),
            ("2PC blocking impossibility proof", lambda c: check_regex(c, r'(?:participant|node).*(?:voted|YES).*(?:coordinator|crash|die|fail).*(?:cannot|impossible|block)', re.DOTALL)),
            ("Saga = ACD not Isolation", lambda c: check_regex(c, r'[Ss]aga.*(?:no\w*\s+isolation|without\s+isolation|ACD|lack\w*\s+isolation)', re.DOTALL)),
            ("Dirty read scenario for Saga", lambda c: check_regex(c, r'(?:dirty|intermediate|partial).*(?:read|see|observ|visible).*(?:saga|concurrent|transaction)', re.DOTALL)),
            ("Orchestration vs choreography", lambda c: check_regex(c, r'orchestrat\w+.*(?:central|coordinator).*choreograph\w+.*(?:event|publish|react)', re.DOTALL)),
            ("Specific use case mapping", lambda c: check_regex(c, r'(?:bank|transfer|payment).*(?:2PC|two.phase|atomic)', re.DOTALL)),

            # PART B: Implementation
            ("Coordinator prepare/commit/abort", lambda c: check_regex(c, r'class\s+Coordinator.*def\s+(?:prepare|commit|abort)', re.DOTALL)),
            ("Write-ahead log", lambda c: check_regex(c, r'(?:write.ahead|WAL|TransactionLog).*(?:PREPARE|COMMIT|ABORT|log|record)', re.DOTALL)),
            ("Coordinator crash recovery", lambda c: check_regex(c, r'(?:crash|recovery|timeout).*(?:coordinator|query|retry|resolve)', re.DOTALL)),
            ("SagaOrchestrator with compensation", lambda c: check_regex(c, r'(?:Saga|saga).*(?:compensat|rollback|reverse|backward)', re.DOTALL)),
            ("Idempotency key implementation", lambda c: check_regex(c, r'idempoten\w+.*(?:key|token|check|duplicate|seen)', re.DOTALL)),
            ("Exponential backoff for retry", lambda c: check_regex(c, r'(?:exponential|backoff|retry).*(?:delay|sleep|wait|2\s*\*\*)', re.DOTALL)),
            ("Substantial code", lambda c: check_code_blocks(c, 3, 8)),

            # PART C: Tests
            ("test_2pc_happy_path", lambda c: check_regex(c, r'def\s+test_\w*2pc\w*happy|def\s+test_\w*commit\w*all', re.DOTALL)),
            ("test_saga_compensation reverse order", lambda c: check_regex(c, r'def\s+test_\w*(?:saga|compensat|fail).*(?:reverse|C2.*C1|backward)', re.DOTALL)),
            ("test_saga_idempotency", lambda c: check_regex(c, r'def\s+test_\w*idempoten', re.DOTALL)),
            ("test_saga_dirty_read", lambda c: check_regex(c, r'def\s+test_\w*(?:dirty|isolation|intermediate)', re.DOTALL)),
            ("5+ assertions", lambda c: check_test_assertions(c, 5)),
            ("Math reasoning", lambda c: check_math_reasoning(c, 2)),
        ],
        "domain": "system_design",
    },
]

AB_HARD_TASK = {
    "id": "AB_vector_clock",
    "goal": (
        "Implement vector clocks with causal broadcast protocol and formal verification.\n\n"
        "PART A — THEORY:\n"
        "1. Define the happens-before relation (→) formally: (a) if a and b are events in the same "
        "process and a comes before b, then a→b; (b) if a is send(m) and b is receive(m), then a→b; "
        "(c) transitivity. Events are CONCURRENT (a||b) iff neither a→b nor b→a.\n"
        "2. Define vector clocks: each process i maintains V[i]. On local event: V[i][i]++. "
        "On send: attach V[i]. On receive from j with timestamp T: V[i] = max(V[i], T); V[i][i]++. "
        "Prove: a→b IFF V(a) < V(b) where < means componentwise ≤ with at least one strict <.\n"
        "3. Give a CONCRETE example with 3 processes (P0, P1, P2) and 6+ events showing: "
        "(a) a causal chain, (b) two concurrent events, (c) the exact vector clock values at each event.\n"
        "4. Define causal broadcast: if send(m1)→send(m2), then every process delivers m1 before m2. "
        "Explain the delivery condition: process j delivers message m from i when: "
        "T(m)[i] == V[j][i] + 1 AND T(m)[k] ≤ V[j][k] for all k ≠ i.\n\n"
        "PART B — IMPLEMENTATION:\n"
        "- VectorClock: increment(process_id), merge(other_clock), happens_before(other) → bool, "
        "concurrent(other) → bool, __lt__, __le__ operators\n"
        "- CausalProcess: maintains vector clock, pending message buffer\n"
        "- CausalBroadcast: send(msg), deliver() with delivery condition check, "
        "buffer out-of-order messages until causally ready\n"
        "- Message: (sender, data, timestamp=VectorClock)\n"
        "Handle: buffered messages delivered in correct causal order, "
        "network reordering, process join/leave.\n\n"
        "PART C — TESTS:\n"
        "- test_happens_before_transitive: a→b and b→c implies a→c via vector clock comparison\n"
        "- test_concurrent_detection: two events with incomparable vector clocks detected as concurrent\n"
        "- test_causal_delivery_order: send m1→m2, deliver in reverse network order, "
        "verify m2 buffered until m1 delivered\n"
        "- test_three_process_scenario: reproduce the 3-process example from PART A, verify all clocks\n"
        "- test_network_reorder: 10 messages with random network delays, verify causal order preserved"
    ),
    "checks": [
        ("Happens-before formal definition", lambda c: check_regex(c, r'(?:a\s*→\s*b|happens.before).*(?:send|receive|transit)', re.DOTALL)),
        ("Vector clock update rules", lambda c: check_regex(c, r'V\[i\]\[i\]\s*\+\+|V\[i\]\s*=\s*max', re.DOTALL)),
        ("Proof: a→b IFF V(a) < V(b)", lambda c: check_regex(c, r'(?:V\(a\)|vector.*a).*(?:<|less).*(?:V\(b\)|vector.*b).*(?:componentwise|≤|<=)', re.DOTALL)),
        ("3-process concrete example", lambda c: check_regex(c, r'P[012].*(?:\[[\d,\s]+\]|\(\d+,\s*\d+,\s*\d+\)).*P[012]', re.DOTALL)),
        ("Causal delivery condition", lambda c: check_regex(c, r'T\(m\)\[i\]\s*==\s*V\[j\]\[i\]\s*\+\s*1|deliver.*condition.*V\[j\]', re.DOTALL)),
        ("VectorClock class with methods", lambda c: check_regex(c, r'class\s+VectorClock.*def\s+(?:increment|merge|happens_before)', re.DOTALL)),
        ("happens_before comparison", lambda c: check_regex(c, r'def\s+(?:happens_before|__lt__).*(?:all|every|component|<=)', re.DOTALL)),
        ("concurrent detection", lambda c: check_regex(c, r'def\s+concurrent.*(?:not.*happens_before|incomparable|neither)', re.DOTALL)),
        ("CausalBroadcast with buffer", lambda c: check_regex(c, r'(?:buffer|pending|queue).*(?:deliver|causal|ready|wait)', re.DOTALL)),
        ("Delivery condition implementation", lambda c: check_regex(c, r'(?:T\[.*sender.*\]|timestamp\[.*\]).*(?:==|<=).*(?:V\[|clock\[)', re.DOTALL)),
        ("Substantial code", lambda c: check_code_blocks(c, 3, 8)),
        ("test_causal_delivery_order", lambda c: check_regex(c, r'def\s+test_\w*causal.*(?:deliver|order|buffer)', re.DOTALL)),
        ("test_concurrent_detection", lambda c: check_regex(c, r'def\s+test_\w*concurrent.*assert', re.DOTALL)),
        ("5+ assertions", lambda c: check_test_assertions(c, 5)),
        ("Math reasoning", lambda c: check_math_reasoning(c, 2)),
    ],
    "domain": "system_design",
}

# =============================================================================
# EVALUATOR
# =============================================================================

class HardEval:
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

    def _snapshot(self, domain: str = "system_design") -> Dict[str, Any]:
        ls = self._get_learning()
        guidance = ls.query(domain, "")
        ctx = ls.build_context_string(domain, "")
        ret = ls.build_retrieval_context(domain, "")
        return {
            "episodes": guidance.get("total_episodes", 0),
            "success_rate": guidance.get("success_rate", 0.0),
            "context_len": len(ctx),
            "retrieval_len": len(ret),
        }

    def _run_checks(self, content: str, checks: List) -> Dict[str, bool]:
        results = {}
        for check_name, check_fn in checks:
            try:
                results[check_name] = check_fn(content)
            except Exception:
                results[check_name] = False
        return results

    async def run_task(self, task: Dict, learn: bool = True) -> Dict[str, Any]:
        tag = "" if learn else " [NO LEARNING]"
        subsep(f"{task['id']}{tag}")

        orch = self._get_orchestrator()
        snap_before = self._snapshot()

        print(f"  Model: {EVAL_MODEL}")
        print(f"  Episodes: {snap_before['episodes']} | Rate: {snap_before['success_rate']:.0%}")
        print(f"  Context: {snap_before['context_len']} chars | Retrieval: {snap_before['retrieval_len']} chars")

        start = time.time()
        content = ""
        error = None

        try:
            result = await orch.chat(
                message=task["goal"],
                provider="anthropic",
                model=EVAL_MODEL,
                learn=learn,
                enabled_tools=[],
                max_steps=1,
                temperature=0,
            )
            content = getattr(result, "content", str(result))
        except Exception as e:
            error = str(e)
            print(f"  ERROR: {error[:200]}")

        elapsed = time.time() - start
        snap_after = self._snapshot()

        check_results = self._run_checks(content, task["checks"])
        passed = sum(1 for v in check_results.values() if v)
        total = len(check_results)
        check_ratio = passed / max(total, 1)

        # Strict quality: 40% check ratio + 30% content depth + 30% structure
        code_blocks = len(re.findall(r'```[\w]*\n.+?```', content, re.DOTALL))
        depth_score = min(1.0, len(content) / 8000)
        structure_score = min(1.0, code_blocks / 4)
        quality = check_ratio * 0.4 + depth_score * 0.3 + structure_score * 0.3

        # STRICT: 70% checks AND 3000 chars to pass
        success = check_ratio >= 0.70 and len(content) >= 3000

        record = {
            "task_id": task["id"],
            "learn": learn,
            "success": success,
            "quality": round(quality, 3),
            "checks_passed": passed,
            "checks_total": total,
            "check_ratio": round(check_ratio, 3),
            "content_length": len(content),
            "code_blocks": code_blocks,
            "elapsed": round(elapsed, 1),
            "ep_before": snap_before["episodes"],
            "ep_after": snap_after["episodes"],
            "context_injected": snap_before["context_len"] + snap_before["retrieval_len"],
            "error": error,
        }
        self.results.append(record)

        status = "PASS" if success else "FAIL"
        print(
            f"\n  [{status}] Q={quality:.2f} | Checks={passed}/{total} ({check_ratio:.0%}) | "
            f"Len={len(content)} | Code={code_blocks} blocks | {elapsed:.0f}s"
        )
        print(f"  Episodes: {snap_before['episodes']}→{snap_after['episodes']}")

        for name, ok in check_results.items():
            mark = "+" if ok else "-"
            print(f"    [{mark}] {name}")

        return record

    def _print_results(self) -> None:
        separator("FINAL RESULTS")

        for r in self.results:
            s = "PASS" if r["success"] else "FAIL"
            tag = "" if r["learn"] else " [BASELINE]"
            ctx_tag = f" ctx={r['context_injected']}" if r["context_injected"] > 0 else ""
            print(
                f"  [{s}] {r['task_id']:20s} Q={r['quality']:.2f} "
                f"Chk={r['checks_passed']}/{r['checks_total']} ({r['check_ratio']:.0%}) "
                f"Len={r['content_length']:5d} Ep={r['ep_before']}→{r['ep_after']}"
                f"{tag}{ctx_tag}"
            )

        # Learning curve
        curve = [r for r in self.results if r["task_id"].startswith("H") and r["learn"]]
        if len(curve) >= 2:
            subsep("LEARNING CURVE")
            for i, r in enumerate(curve):
                bar_len = int(r["check_ratio"] * 40)
                bar = "█" * bar_len + "░" * (40 - bar_len)
                print(
                    f"  {r['task_id']:15s} {bar} {r['check_ratio']:.0%} "
                    f"({r['checks_passed']}/{r['checks_total']}) Ep={r['ep_before']}"
                )
            first = curve[0]["check_ratio"]
            last = curve[-1]["check_ratio"]
            print(f"\n  Check trend: {first:.0%} → {last:.0%} ({last - first:+.0%})")

        # A/B
        ab_no = [r for r in self.results if r["task_id"] == "AB_no_learn"]
        ab_yes = [r for r in self.results if r["task_id"] == "AB_with_learn"]
        if ab_no and ab_yes:
            subsep("A/B TEST")
            n, y = ab_no[0], ab_yes[0]
            delta = y["check_ratio"] - n["check_ratio"]
            winner = "LEARNING WINS" if delta > 0.05 else ("NEUTRAL" if abs(delta) <= 0.05 else "BASELINE WINS")
            print(f"  Baseline:  {n['check_ratio']:.0%} ({n['checks_passed']}/{n['checks_total']}) Len={n['content_length']}")
            print(f"  Learning:  {y['check_ratio']:.0%} ({y['checks_passed']}/{y['checks_total']}) Len={y['content_length']}")
            print(f"  Delta: {delta:+.0%} → {winner}")

        # Overall stats
        subsep("SUMMARY")
        all_learned = [r for r in self.results if r["learn"]]
        avg_checks = sum(r["check_ratio"] for r in all_learned) / max(len(all_learned), 1)
        pass_rate = sum(1 for r in all_learned if r["success"]) / max(len(all_learned), 1)
        print(f"  Avg check pass rate: {avg_checks:.0%}")
        print(f"  Task pass rate (≥70% checks + ≥3000 chars): {pass_rate:.0%}")
        print(f"  Total episodes recorded: {self.results[-1]['ep_after'] if self.results else 0}")

    async def run_all(self) -> None:
        separator("JOTTY HARD LEARNING EVALUATION")
        print(f"  Date: {datetime.now().isoformat()}")
        print(f"  Model: {EVAL_MODEL} (weakest Anthropic, 4096 max output)")
        print(f"  Tasks: {len(HARD_TASKS)} genuinely hard distributed systems problems")
        print(f"  Checks: structural regex (not keyword matching)")
        print(f"  Pass threshold: 70% checks AND 3000+ chars")
        print(f"  Initial episodes: {self._snapshot()['episodes']}")

        # Learning curve
        for i, task in enumerate(HARD_TASKS):
            snap = self._snapshot()
            sep_label = f"ROUND {i+1}/{len(HARD_TASKS)} — {task['id']}"
            if i == 0:
                sep_label += " (cold start)"
            else:
                sep_label += f" ({snap['episodes']} episodes)"
            separator(sep_label)
            await self.run_task(task, learn=True)
            print("  Waiting for background learning...")
            await asyncio.sleep(20)

        # A/B test
        separator("A/B TEST — Vector Clocks")
        snap = self._snapshot()
        print(f"  Episodes: {snap['episodes']}")

        await self.run_task({**AB_HARD_TASK, "id": "AB_no_learn"}, learn=False)
        await self.run_task({**AB_HARD_TASK, "id": "AB_with_learn"}, learn=True)
        await asyncio.sleep(15)

        # Results
        self._print_results()

        results_dir = Path(__file__).parent / "eval_results"
        results_dir.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = results_dir / f"{ts}_hard_learning.json"
        with open(out, "w") as f:
            json.dump({"timestamp": datetime.now().isoformat(), "model": EVAL_MODEL,
                        "results": self.results}, f, indent=2, default=str)
        print(f"\n  Saved: {out}")


async def main() -> None:
    await HardEval().run_all()

if __name__ == "__main__":
    asyncio.run(main())
