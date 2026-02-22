#!/usr/bin/env python3
"""
Jotty Super-Complex Real-World Evaluation
==========================================

Tests Jotty across 8 dimensions using REAL API calls (Anthropic).
Each dimension tests a fundamentally different capability.

Dimensions:
  1. PURE REASONING   — Deep technical generation (no tools)
  2. TOOL CALLING     — Multi-step tool use (web search, calculator, etc.)
  3. MULTI-TURN       — Conversational coherence across turns
  4. SDK LOCAL MODE   — SDK client in local mode (chat + workflow)
  5. LEARNING SYSTEM  — Does repeated execution actually learn?
  6. ERROR RECOVERY   — Graceful handling of bad inputs / tool failures
  7. WORKFLOW MODE    — AutoAgent autonomous planning + execution
  8. MEMORY SYSTEM    — Store, retrieve, use memories
  9. MAS SWARM        — Multi-Agent Swarm (CodingSwarm + auto-detect)

Uses Sonnet for quality ceiling, Haiku for speed/cost baseline.
"""

import asyncio
import json
import logging
import os
import re
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(line_buffering=True)

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("eval_supercomplex")
logger.setLevel(logging.INFO)

SONNET = "claude-sonnet-4-20250514"
HAIKU = "claude-3-haiku-20240307"

# Which model to use for tests (Sonnet for quality ceiling)
EVAL_MODEL = SONNET
TIMEOUT = 120


# =============================================================================
# CHECK HELPERS
# =============================================================================


def rx(content: str, pattern: str) -> bool:
    return bool(re.search(pattern, content, re.IGNORECASE | re.DOTALL))


def code_blocks(content: str, n: int = 1, min_lines: int = 3) -> bool:
    blocks = re.findall(r"```[\w]*\n(.*?)```", content, re.DOTALL)
    return len([b for b in blocks if len(b.strip().split("\n")) >= min_lines]) >= n


def word_count(content: str, n: int) -> bool:
    return len(content.split()) >= n


def has_structure(content: str) -> bool:
    """Content has sections/headers/organization."""
    return bool(re.findall(r"(?:^|\n)#{1,3}\s+\w+|(?:^|\n)\d+\.\s+\w+|(?:^|\n)[*-]\s+\w+", content))


@dataclass
class TestResult:
    dimension: str
    test_name: str
    success: bool
    quality: float  # 0-1
    content_length: int = 0
    elapsed: float = 0.0
    checks: Dict[str, bool] = field(default_factory=dict)
    error: Optional[str] = None
    notes: str = ""

    @property
    def check_ratio(self) -> float:
        if not self.checks:
            return 0.0
        return sum(1 for v in self.checks.values() if v) / len(self.checks)


# =============================================================================
# DIMENSION 1: PURE REASONING (no tools)
# =============================================================================


async def test_pure_reasoning(orch) -> List[TestResult]:
    """Test deep technical generation without any tool use."""
    results = []

    # Test 1A: Cross-domain synthesis
    goal = (
        "Design a system that combines concepts from:\n"
        "1. Distributed consensus (Raft/Paxos)\n"
        "2. Machine learning model serving\n"
        "3. Financial risk management\n\n"
        "Create an 'ML Model Governance Protocol' that ensures:\n"
        "- Consensus among N validator nodes before deploying a model\n"
        "- Rollback safety using log-based state machines\n"
        "- Risk scoring using VaR (Value at Risk) principles for model predictions\n"
        "- Quorum-based approval for model weight updates\n\n"
        "Provide: Architecture diagram (ASCII), Python implementation with classes, "
        "and formal correctness arguments. Include 5+ test cases.\n\n"
        "Do NOT use any tools."
    )

    start = time.time()
    try:
        result = await asyncio.wait_for(
            orch.chat(
                message=goal,
                provider="anthropic",
                model=EVAL_MODEL,
                learn=True,
                enabled_tools=[],
                max_steps=1,
                temperature=0,
            ),
            timeout=TIMEOUT,
        )
        content = getattr(result, "content", str(result))
        elapsed = time.time() - start

        checks = {
            "Consensus protocol": rx(content, r"consensus|quorum|vote|raft|paxos"),
            "ML model serving": rx(content, r"model.*serv|inference|deploy|predict"),
            "Risk scoring (VaR)": rx(content, r"VaR|value.?at.?risk|risk.*scor"),
            "Rollback mechanism": rx(content, r"rollback|revert|undo|log.*state"),
            "Python classes": rx(content, r"class\s+\w+"),
            "Test cases": rx(content, r"def\s+test_|assert\s+"),
            "Architecture": rx(content, r"─|│|┌|ASCII|diagram|architecture"),
            "500+ words": word_count(content, 500),
            "Code blocks": code_blocks(content, 2, 5),
            "Formal arguments": rx(content, r"correctness|proof|invariant|guarantee|safety"),
        }
        quality = sum(v for v in checks.values()) / len(checks)
        results.append(
            TestResult(
                "PURE_REASONING",
                "cross_domain_synthesis",
                quality >= 0.6,
                quality,
                len(content),
                elapsed,
                checks,
            )
        )
    except Exception as e:
        results.append(
            TestResult("PURE_REASONING", "cross_domain_synthesis", False, 0.0, error=str(e)[:200])
        )

    # Test 1B: Mathematical proof + implementation
    goal2 = (
        "Prove that the halting problem is undecidable using diagonalization.\n\n"
        "Then implement a practical 'Bounded Halting Checker' in Python that:\n"
        "1. Takes a Python function + arguments + time limit\n"
        "2. Runs it in a sandboxed subprocess with resource limits\n"
        "3. Returns: HALTS(result), TIMEOUT, or ERROR(exception)\n"
        "4. Implements Turing degree approximation for common patterns\n"
        "   (infinite loops, recursive depth, memory growth detection)\n\n"
        "Include the formal proof, full implementation, and 8+ test cases.\n\n"
        "Do NOT use any tools."
    )

    start = time.time()
    try:
        result = await asyncio.wait_for(
            orch.chat(
                message=goal2,
                provider="anthropic",
                model=EVAL_MODEL,
                learn=True,
                enabled_tools=[],
                max_steps=1,
                temperature=0,
            ),
            timeout=TIMEOUT,
        )
        content = getattr(result, "content", str(result))
        elapsed = time.time() - start

        checks = {
            "Halting problem stated": rx(content, r"halting.?problem|undecidable"),
            "Diagonalization": rx(content, r"diagonaliz|diagonal.*argument|contradiction"),
            "Formal proof structure": rx(content, r"assume|suppose|contradiction|therefore|QED|∎"),
            "BoundedHalting class": rx(content, r"class\s+\w*[Hh]alt|def\s+\w*halt"),
            "Subprocess/sandbox": rx(content, r"subprocess|sandbox|multiprocessing|Process"),
            "Resource limits": rx(content, r"timeout|resource|limit|signal|SIGALRM|setrlimit"),
            "Pattern detection": rx(content, r"infinite.*loop|recursion|memory.*growth|detection"),
            "Test cases": rx(content, r"def\s+test_"),
            "Code blocks": code_blocks(content, 2, 5),
            "500+ words": word_count(content, 500),
        }
        quality = sum(v for v in checks.values()) / len(checks)
        results.append(
            TestResult(
                "PURE_REASONING",
                "math_proof_implementation",
                quality >= 0.6,
                quality,
                len(content),
                elapsed,
                checks,
            )
        )
    except Exception as e:
        results.append(
            TestResult(
                "PURE_REASONING", "math_proof_implementation", False, 0.0, error=str(e)[:200]
            )
        )

    return results


# =============================================================================
# DIMENSION 2: TOOL CALLING (real tool use)
# =============================================================================


async def test_tool_calling(orch) -> List[TestResult]:
    """Test multi-step tool use with real skills."""
    results = []

    # Test 2A: Web search + synthesis
    goal = "Search the web for 'latest AI agent frameworks 2025' and summarize the top 3 results with pros and cons for each."

    start = time.time()
    try:
        result = await asyncio.wait_for(
            orch.chat(
                message=goal,
                provider="anthropic",
                model=EVAL_MODEL,
                learn=True,
                max_steps=5,
                temperature=0,
            ),
            timeout=TIMEOUT,
        )
        content = getattr(result, "content", str(result))
        elapsed = time.time() - start
        tools_used = getattr(result, "tools_used", []) or getattr(result, "tool_calls", []) or []

        checks = {
            "Has content": len(content) > 200,
            "Mentions AI frameworks": rx(content, r"(?:agent|AI|framework|LLM)"),
            "Has structure (headers/lists)": has_structure(content),
            "Pros mentioned": rx(content, r"pros?|advantage|strength|benefit"),
            "Cons mentioned": rx(content, r"cons?|disadvantage|weakness|limitation|drawback"),
            "Multiple frameworks": len(
                re.findall(
                    r"(?:AutoGen|CrewAI|LangGraph|LangChain|Swarm|Agency|DSPy|Jotty)", content, re.I
                )
            )
            >= 2,
            "200+ words": word_count(content, 200),
        }
        quality = sum(v for v in checks.values()) / len(checks)
        results.append(
            TestResult(
                "TOOL_CALLING",
                "web_search_synthesis",
                quality >= 0.5,
                quality,
                len(content),
                elapsed,
                checks,
                notes=f"tools_used={len(tools_used)}",
            )
        )
    except Exception as e:
        results.append(
            TestResult("TOOL_CALLING", "web_search_synthesis", False, 0.0, error=str(e)[:200])
        )

    # Test 2B: Calculator / math tool
    goal2 = "Calculate: What is 17^5 + sqrt(144) - ln(e^3)? Show your work step by step."

    start = time.time()
    try:
        result = await asyncio.wait_for(
            orch.chat(
                message=goal2,
                provider="anthropic",
                model=EVAL_MODEL,
                learn=True,
                max_steps=5,
                temperature=0,
            ),
            timeout=TIMEOUT,
        )
        content = getattr(result, "content", str(result))
        elapsed = time.time() - start

        checks = {
            "Has result": len(content) > 50,
            "Correct 17^5 (1419857)": rx(content, r"1,?419,?857"),
            "Correct sqrt(144) (12)": rx(content, r"\b12\b"),
            "Correct ln(e^3) (3)": rx(content, r"\b3\b"),
            "Final answer ~1419866": rx(content, r"1,?419,?866"),
            "Shows steps": rx(content, r"step|first|then|next|finally"),
        }
        quality = sum(v for v in checks.values()) / len(checks)
        results.append(
            TestResult(
                "TOOL_CALLING",
                "math_calculation",
                quality >= 0.4,
                quality,
                len(content),
                elapsed,
                checks,
            )
        )
    except Exception as e:
        results.append(
            TestResult("TOOL_CALLING", "math_calculation", False, 0.0, error=str(e)[:200])
        )

    return results


# =============================================================================
# DIMENSION 3: MULTI-TURN CONVERSATION
# =============================================================================


async def test_multi_turn(orch) -> List[TestResult]:
    """Test coherence across multiple conversation turns."""
    results = []

    turns = [
        "I'm building a distributed key-value store in Python. What architecture should I use?",
        "Good suggestions. Now implement the consistent hashing ring from your recommendation. Show the code.",
        "Now add virtual nodes to reduce hotspots. Modify the code you just wrote.",
        "Write 5 test cases for the consistent hashing implementation with virtual nodes.",
    ]

    history = []
    all_contents = []
    total_elapsed = 0
    errors = []

    for i, msg in enumerate(turns):
        start = time.time()
        try:
            result = await asyncio.wait_for(
                orch.chat(
                    message=msg,
                    provider="anthropic",
                    model=EVAL_MODEL,
                    history=history,
                    learn=True,
                    enabled_tools=[],
                    max_steps=1,
                    temperature=0,
                ),
                timeout=TIMEOUT,
            )
            content = getattr(result, "content", str(result))
            elapsed = time.time() - start
            total_elapsed += elapsed
            all_contents.append(content)

            history.append({"role": "user", "content": msg})
            history.append({"role": "assistant", "content": content[:2000]})
        except Exception as e:
            errors.append(f"Turn {i+1}: {str(e)[:100]}")
            all_contents.append("")
            history.append({"role": "user", "content": msg})
            history.append({"role": "assistant", "content": f"Error: {e}"})

    combined = "\n\n".join(all_contents)

    checks = {
        "Turn 1: Architecture suggestion": rx(
            all_contents[0] if all_contents[0] else "",
            r"consistent.?hash|partition|replication|shard",
        ),
        "Turn 2: Code implementation": rx(
            all_contents[1] if len(all_contents) > 1 else "", r"class\s+\w*Hash|def\s+\w*hash|ring"
        ),
        "Turn 3: Virtual nodes added": rx(
            all_contents[2] if len(all_contents) > 2 else "", r"virtual.*node|v_?node|replica"
        ),
        "Turn 4: Test cases": rx(
            all_contents[3] if len(all_contents) > 3 else "", r"def\s+test_|assert\s+"
        ),
        "Coherent across turns": rx(combined, r"consistent.?hash") and rx(combined, r"virtual"),
        "No errors": len(errors) == 0,
        "All turns responded": len([c for c in all_contents if len(c) > 50]) == 4,
    }
    quality = sum(v for v in checks.values()) / len(checks)
    results.append(
        TestResult(
            "MULTI_TURN",
            "4_turn_conversation",
            quality >= 0.5,
            quality,
            len(combined),
            total_elapsed,
            checks,
            error="; ".join(errors) if errors else None,
        )
    )

    return results


# =============================================================================
# DIMENSION 4: SDK LOCAL MODE
# =============================================================================


async def test_sdk_local(orch) -> List[TestResult]:
    """Test the SDK client in local mode."""
    results = []

    try:
        from Jotty.sdk.client import Jotty as JottySDK

        client = JottySDK()
        client.use_local()

        # Test 4A: SDK chat
        start = time.time()
        response = await asyncio.wait_for(
            client.chat("Explain the CAP theorem in distributed systems. Be concise but thorough."),
            timeout=TIMEOUT,
        )
        elapsed = time.time() - start

        content = response.content if hasattr(response, "content") else str(response)
        if isinstance(content, dict):
            content = json.dumps(content)
        content = str(content or "")

        checks = {
            "Response received": len(content) > 100,
            "Success flag": response.success if hasattr(response, "success") else True,
            "CAP mentioned": rx(content, r"CAP|consistency|availability|partition"),
            "All 3 properties": (
                rx(content, r"[Cc]onsistency")
                and rx(content, r"[Aa]vailability")
                and rx(content, r"[Pp]artition")
            ),
            "Trade-off explained": rx(content, r"trade.?off|choose|sacrifice|cannot.*all"),
            "Under 60s": elapsed < 60,
        }
        quality = sum(v for v in checks.values()) / len(checks)
        results.append(
            TestResult(
                "SDK_LOCAL",
                "chat_mode",
                quality >= 0.5,
                quality,
                len(content),
                elapsed,
                checks,
            )
        )

        # Test 4B: SDK workflow
        start = time.time()
        try:
            response = await asyncio.wait_for(
                client.workflow("List the top 3 sorting algorithms by time complexity"),
                timeout=TIMEOUT,
            )
            elapsed = time.time() - start
            content = str(response.content or "")

            checks = {
                "Response received": len(content) > 50,
                "Success flag": response.success if hasattr(response, "success") else True,
                "Sorting mentioned": rx(content, r"sort|quick|merge|heap|bubble"),
                "Complexity mentioned": rx(content, r"O\(n|log\s*n|complexity|time"),
            }
            quality = sum(v for v in checks.values()) / len(checks)
            results.append(
                TestResult(
                    "SDK_LOCAL",
                    "workflow_mode",
                    quality >= 0.3,
                    quality,
                    len(content),
                    elapsed,
                    checks,
                )
            )
        except Exception as e:
            results.append(TestResult("SDK_LOCAL", "workflow_mode", False, 0.0, error=str(e)[:200]))

    except Exception as e:
        results.append(TestResult("SDK_LOCAL", "chat_mode", False, 0.0, error=str(e)[:200]))

    return results


# =============================================================================
# DIMENSION 5: LEARNING SYSTEM
# =============================================================================


async def test_learning(orch) -> List[TestResult]:
    """Test whether the learning system records and retrieves knowledge."""
    results = []

    try:
        from Jotty.core.intelligence.learning.learning_service import (
            LearningService,
            classify_domain,
        )

        ls = LearningService.get_instance()

        # Snapshot before
        snap_before = {
            "episodes": 0,
            "guidance": {},
        }
        try:
            guidance = ls.query("eval_test", "")
            snap_before["episodes"] = guidance.get("total_episodes", 0)
            snap_before["guidance"] = guidance
        except Exception:
            pass

        # Run a task that should create a learning episode
        start = time.time()
        result = await asyncio.wait_for(
            orch.chat(
                message="Explain binary search trees and their time complexity for insertion, deletion, and search operations.",
                provider="anthropic",
                model=HAIKU,
                learn=True,
                enabled_tools=[],
                max_steps=1,
                temperature=0,
            ),
            timeout=TIMEOUT,
        )
        content = getattr(result, "content", str(result))
        elapsed = time.time() - start

        await asyncio.sleep(2)  # Let learning process

        # Snapshot after
        snap_after = {"episodes": 0, "guidance": {}}
        try:
            guidance = ls.query("data_structures", "")
            snap_after["episodes"] = guidance.get("total_episodes", 0)
            snap_after["guidance"] = guidance

            # Also check global count
            conn = ls._store._get_conn()
            row = conn.execute("SELECT COUNT(*) as cnt FROM episodes").fetchone()
            snap_after["global_episodes"] = row["cnt"] if row else 0
        except Exception as e:
            snap_after["error"] = str(e)

        # Check context injection
        ctx = ls.build_context_string("data_structures", "")
        ret = ls.build_retrieval_context("data_structures", "")

        checks = {
            "Task completed": len(content) > 200,
            "Episode recorded": snap_after.get("global_episodes", 0) > 0,
            "Guidance available": bool(snap_after.get("guidance")),
            "Context string non-empty": len(ctx) > 0 or len(ret) > 0,
            "BST content correct": rx(content, r"binary.?search.?tree|BST"),
            "Complexity mentioned": rx(content, r"O\(log|O\(n|logarithmic"),
        }
        quality = sum(v for v in checks.values()) / len(checks)
        results.append(
            TestResult(
                "LEARNING",
                "record_and_query",
                quality >= 0.5,
                quality,
                len(content),
                elapsed,
                checks,
                notes=f"episodes_before={snap_before['episodes']}, after={snap_after.get('global_episodes', '?')}, ctx={len(ctx)}ch, ret={len(ret)}ch",
            )
        )

    except Exception as e:
        results.append(
            TestResult(
                "LEARNING",
                "record_and_query",
                False,
                0.0,
                error=f"{str(e)[:200]}\n{traceback.format_exc()[-300:]}",
            )
        )

    return results


# =============================================================================
# DIMENSION 6: ERROR RECOVERY
# =============================================================================


async def test_error_recovery(orch) -> List[TestResult]:
    """Test graceful handling of edge cases."""
    results = []

    # Test 6A: Empty input
    start = time.time()
    try:
        result = await asyncio.wait_for(
            orch.chat(
                message="",
                provider="anthropic",
                model=HAIKU,
                learn=False,
                enabled_tools=[],
                max_steps=1,
            ),
            timeout=30,
        )
        content = getattr(result, "content", str(result))
        elapsed = time.time() - start
        checks = {
            "No crash": True,
            "Some response": len(str(content)) > 0,
        }
        quality = sum(v for v in checks.values()) / len(checks)
        results.append(
            TestResult(
                "ERROR_RECOVERY",
                "empty_input",
                True,
                quality,
                len(str(content)),
                elapsed,
                checks,
            )
        )
    except Exception as e:
        elapsed = time.time() - start
        is_graceful = (
            "error" in str(e).lower() or "empty" in str(e).lower() or "valid" in str(e).lower()
        )
        results.append(
            TestResult(
                "ERROR_RECOVERY",
                "empty_input",
                is_graceful,
                0.5 if is_graceful else 0.0,
                0,
                elapsed,
                {"Graceful error": is_graceful, "No crash": True},
                error=str(e)[:150],
            )
        )

    # Test 6B: Very long input (stress test context window)
    long_input = "Summarize this: " + ("The quick brown fox jumps over the lazy dog. " * 500)
    start = time.time()
    try:
        result = await asyncio.wait_for(
            orch.chat(
                message=long_input,
                provider="anthropic",
                model=HAIKU,
                learn=False,
                enabled_tools=[],
                max_steps=1,
            ),
            timeout=60,
        )
        content = getattr(result, "content", str(result))
        elapsed = time.time() - start
        checks = {
            "No crash": True,
            "Responded": len(str(content)) > 20,
            "Recognized repetition": rx(str(content), r"repeat|fox|dog|summary|sentence"),
        }
        quality = sum(v for v in checks.values()) / len(checks)
        results.append(
            TestResult(
                "ERROR_RECOVERY",
                "long_input",
                True,
                quality,
                len(str(content)),
                elapsed,
                checks,
            )
        )
    except Exception as e:
        elapsed = time.time() - start
        results.append(
            TestResult(
                "ERROR_RECOVERY",
                "long_input",
                False,
                0.0,
                0,
                elapsed,
                error=str(e)[:200],
            )
        )

    # Test 6C: Requesting non-existent tool
    start = time.time()
    try:
        result = await asyncio.wait_for(
            orch.chat(
                message="Use the 'quantum_teleporter' tool to send data to Mars",
                provider="anthropic",
                model=HAIKU,
                learn=False,
                max_steps=3,
                temperature=0,
            ),
            timeout=60,
        )
        content = getattr(result, "content", str(result))
        elapsed = time.time() - start
        checks = {
            "No crash": True,
            "Responded": len(str(content)) > 20,
            "Handled gracefully": not rx(str(content), r"traceback|exception|KeyError"),
        }
        quality = sum(v for v in checks.values()) / len(checks)
        results.append(
            TestResult(
                "ERROR_RECOVERY",
                "nonexistent_tool",
                True,
                quality,
                len(str(content)),
                elapsed,
                checks,
            )
        )
    except Exception as e:
        elapsed = time.time() - start
        results.append(
            TestResult(
                "ERROR_RECOVERY",
                "nonexistent_tool",
                True,
                0.5,
                0,
                elapsed,
                {"Caught gracefully": True},
                error=str(e)[:150],
            )
        )

    return results


# =============================================================================
# DIMENSION 7: WORKFLOW MODE (AutoAgent)
# =============================================================================


async def test_workflow(orch) -> List[TestResult]:
    """Test autonomous workflow execution via AutoAgent."""
    results = []

    try:
        from Jotty.sdk.client import Jotty as JottySDK

        client = JottySDK()
        client.use_local()

        start = time.time()
        response = await asyncio.wait_for(
            client.workflow(
                "Create a comprehensive comparison of Python web frameworks: "
                "Django, FastAPI, and Flask. Compare them on performance, "
                "learning curve, ecosystem, and best use cases."
            ),
            timeout=TIMEOUT,
        )
        elapsed = time.time() - start
        content = str(response.content or "")

        checks = {
            "Response received": len(content) > 200,
            "Django mentioned": rx(content, r"[Dd]jango"),
            "FastAPI mentioned": rx(content, r"[Ff]ast\s*API"),
            "Flask mentioned": rx(content, r"[Ff]lask"),
            "Performance discussed": rx(content, r"performance|speed|fast|benchmark|throughput"),
            "Learning curve": rx(content, r"learn|beginner|easy|simple|curve"),
            "Use cases": rx(content, r"use.?case|scenario|when.*to.*use|best.*for"),
            "Structured output": has_structure(content),
        }
        quality = sum(v for v in checks.values()) / len(checks)
        results.append(
            TestResult(
                "WORKFLOW",
                "framework_comparison",
                quality >= 0.5,
                quality,
                len(content),
                elapsed,
                checks,
                notes=f"skills_used={getattr(response, 'skills_used', [])}, steps={getattr(response, 'steps_executed', 0)}",
            )
        )
    except Exception as e:
        results.append(
            TestResult("WORKFLOW", "framework_comparison", False, 0.0, error=str(e)[:200])
        )

    return results


# =============================================================================
# DIMENSION 8: MEMORY SYSTEM
# =============================================================================


async def test_memory(orch) -> List[TestResult]:
    """Test the memory subsystem: store and retrieve."""
    results = []

    try:
        from Jotty.core.intelligence.memory.facade import get_memory_system

        mem = get_memory_system()

        # Store some memories
        mem_ids = []
        test_memories = [
            (
                "The Jotty framework uses TD-Lambda for reinforcement learning with gamma=0.99",
                "semantic",
            ),
            (
                "When building swarms, always use the Orchestrator.chat() method for tool-calling tasks",
                "procedural",
            ),
            (
                "Previous eval showed Haiku scores 75% on JWT auth tasks, improving to 87% by run 3",
                "episodic",
            ),
        ]

        for content, level in test_memories:
            try:
                mid = mem.store(
                    content=content,
                    level=level,
                    goal="eval_test",
                    metadata={"source": "eval_supercomplex"},
                )
                mem_ids.append(mid)
            except Exception:
                mem_ids.append(None)

        # Retrieve memories
        retrieve_results = []
        queries = [
            "How does Jotty learn from experience?",
            "What's the best way to run tasks in Jotty?",
        ]
        for q in queries:
            try:
                retrieved = mem.retrieve(query=q, goal="eval_test", top_k=3)
                retrieve_results.append(retrieved)
            except Exception:
                retrieve_results.append([])

        # Check status
        try:
            status = mem.status()
        except Exception:
            status = {}

        checks = {
            "Stored all memories": all(mid is not None for mid in mem_ids),
            "Retrieved results for Q1": len(retrieve_results[0]) > 0 if retrieve_results else False,
            "Retrieved results for Q2": (
                len(retrieve_results[1]) > 0 if len(retrieve_results) > 1 else False
            ),
            "Status available": bool(status),
            "Relevant retrieval": any(
                rx(str(getattr(r, "content", "")), r"TD-Lambda|reinforcement|learn")
                for r in (retrieve_results[0] if retrieve_results else [])
            ),
        }
        quality = sum(v for v in checks.values()) / len(checks)
        results.append(
            TestResult(
                "MEMORY",
                "store_retrieve",
                quality >= 0.4,
                quality,
                0,
                0,
                checks,
                notes=f"stored={len([m for m in mem_ids if m])}, retrieved={[len(r) for r in retrieve_results]}, status={status.get('total_memories', '?')}",
            )
        )

    except Exception as e:
        results.append(TestResult("MEMORY", "store_retrieve", False, 0.0, error=f"{str(e)[:200]}"))

    return results


# =============================================================================
# DIMENSION 9: MULTI-AGENT SWARM (MAS)
# =============================================================================


async def test_mas_swarm(orch) -> List[TestResult]:
    """Test multi-agent swarm execution via orch.run(swarm=...)."""
    results = []

    # Test 9A: CodingSwarm MAS — multi-agent collaborative code generation
    start = time.time()
    try:
        from Jotty.core.intelligence.orchestration.swarms.coding_swarm.swarm import CodingSwarm
        from Jotty.core.intelligence.orchestration.swarms.coding_swarm.types import CodingConfig

        coding_swarm = CodingSwarm(CodingConfig())
        result = await asyncio.wait_for(
            orch.run(
                "Build a Python rate limiter class using the token bucket algorithm. "
                "Include: TokenBucket class with consume(tokens) and refill(), "
                "a decorator @rate_limit(requests_per_second=10) for Flask routes, "
                "and 5 unit tests.",
                swarm=coding_swarm,
                learn=True,
            ),
            timeout=180,
        )
        elapsed = time.time() - start

        content = ""
        if hasattr(result, "output"):
            content = str(result.output or "")
        elif hasattr(result, "final_output"):
            content = str(result.final_output or "")
        elif isinstance(result, dict):
            content = str(result.get("output", result.get("final_output", result)))
        else:
            content = str(result)

        success_flag = getattr(result, "success", True)
        agents_used = getattr(result, "agent_contributions", {})

        checks = {
            "Response received": len(content) > 200,
            "Success flag": success_flag,
            "TokenBucket class": rx(content, r"class\s+TokenBucket"),
            "consume method": rx(content, r"def\s+consume"),
            "refill logic": rx(content, r"refill|token.*bucket|replenish"),
            "Rate limit decorator": rx(content, r"rate_limit|@.*limit"),
            "Test cases": rx(content, r"def\s+test_|assert\s+"),
            "Code blocks": code_blocks(content, 1, 5),
            "Multiple agents contributed": len(agents_used) >= 2 if agents_used else True,
        }
        quality = sum(v for v in checks.values()) / len(checks)
        results.append(
            TestResult(
                "MAS_SWARM",
                "coding_swarm",
                quality >= 0.5,
                quality,
                len(content),
                elapsed,
                checks,
                notes=f"agents={list(agents_used.keys()) if agents_used else 'unknown'}, success={success_flag}",
            )
        )
    except Exception as e:
        elapsed = time.time() - start
        results.append(
            TestResult(
                "MAS_SWARM", "coding_swarm", False, 0.0, elapsed=elapsed, error=f"{str(e)[:300]}"
            )
        )

    # Test 9B: Auto-detect via orch.run() (no explicit swarm=)
    start = time.time()
    try:
        result = await asyncio.wait_for(
            orch.run(
                "Explain the differences between microservices and monolithic architecture. "
                "Cover: scalability, deployment, team structure, and data management. "
                "Give concrete examples of when to use each.",
                learn=True,
            ),
            timeout=120,
        )
        elapsed = time.time() - start

        content = ""
        # AgenticExecutionResult: .final_output, EpisodeResult: .output
        for attr in ("final_output", "output", "content"):
            val = getattr(result, attr, None)
            if val and isinstance(val, str) and len(val) > len(content):
                content = val
        if not content:
            content = str(result)

        checks = {
            "Response received": len(content) > 200,
            "Microservices mentioned": rx(content, r"microservice"),
            "Monolithic mentioned": rx(content, r"monolith"),
            "Scalability discussed": rx(content, r"scal(e|able|ability)"),
            "Deployment discussed": rx(content, r"deploy"),
            "Examples given": rx(
                content, r"example|e\.g\.|such as|for instance|Netflix|Amazon|Uber"
            ),
            "Structured output": has_structure(content),
        }
        quality = sum(v for v in checks.values()) / len(checks)
        results.append(
            TestResult(
                "MAS_SWARM",
                "auto_detect_run",
                quality >= 0.5,
                quality,
                len(content),
                elapsed,
                checks,
            )
        )
    except Exception as e:
        elapsed = time.time() - start
        results.append(
            TestResult(
                "MAS_SWARM", "auto_detect_run", False, 0.0, elapsed=elapsed, error=f"{str(e)[:300]}"
            )
        )

    return results


# =============================================================================
# MAIN EVALUATOR
# =============================================================================


class SuperComplexEval:
    def __init__(self):
        self.results: List[TestResult] = []
        self._orch = None

    def _get_orchestrator(self):
        if self._orch is None:
            from Jotty.core.intelligence.orchestration.core.swarm_manager import Orchestrator

            self._orch = Orchestrator()
        return self._orch

    async def run_all(self):
        print(f"\n{'=' * 80}")
        print(f"  JOTTY SUPER-COMPLEX REAL-WORLD EVALUATION")
        print(f"{'=' * 80}")
        print(f"  Date:  {datetime.now().isoformat()}")
        print(f"  Model: {EVAL_MODEL}")
        print(
            f"  API:   Anthropic (key={'present' if os.getenv('ANTHROPIC_API_KEY') else 'MISSING'})"
        )
        print(f"{'=' * 80}\n")

        orch = self._get_orchestrator()

        dimensions = [
            ("1. PURE REASONING", test_pure_reasoning),
            ("2. TOOL CALLING", test_tool_calling),
            ("3. MULTI-TURN", test_multi_turn),
            ("4. SDK LOCAL MODE", test_sdk_local),
            ("5. LEARNING SYSTEM", test_learning),
            ("6. ERROR RECOVERY", test_error_recovery),
            ("7. WORKFLOW MODE", test_workflow),
            ("8. MEMORY SYSTEM", test_memory),
            ("9. MAS SWARM", test_mas_swarm),
        ]

        for dim_name, dim_func in dimensions:
            print(f"\n{'─' * 80}")
            print(f"  DIMENSION: {dim_name}")
            print(f"{'─' * 80}\n")

            try:
                dim_results = await dim_func(orch)
                self.results.extend(dim_results)

                for r in dim_results:
                    status = "PASS" if r.success else "FAIL"
                    print(
                        f"  [{status}] {r.test_name:<30s} Q={r.quality:.2f} "
                        f"({sum(r.checks.values())}/{len(r.checks)} checks) "
                        f"{r.elapsed:.1f}s {r.content_length}ch"
                    )

                    if r.error:
                        print(f"         ERROR: {r.error[:120]}")
                    if r.notes:
                        print(f"         {r.notes}")

                    for check_name, ok in r.checks.items():
                        print(f"         [{'+'if ok else '-'}] {check_name}")

            except Exception as e:
                print(f"  DIMENSION FAILED: {e}")
                traceback.print_exc()

        self._print_report()
        self._save_results()

    def _print_report(self):
        print(f"\n{'=' * 80}")
        print(f"  FINAL REPORT — JOTTY EVALUATION")
        print(f"{'=' * 80}\n")

        # Per-dimension scores
        dimensions = {}
        for r in self.results:
            if r.dimension not in dimensions:
                dimensions[r.dimension] = []
            dimensions[r.dimension].append(r)

        dim_scores = {}
        for dim, tests in dimensions.items():
            avg_quality = sum(t.quality for t in tests) / len(tests)
            pass_rate = sum(1 for t in tests if t.success) / len(tests)
            dim_scores[dim] = {"quality": avg_quality, "pass_rate": pass_rate, "tests": len(tests)}

            bar_len = int(avg_quality * 30)
            bar = "█" * bar_len + "░" * (30 - bar_len)
            pass_str = f"{sum(1 for t in tests if t.success)}/{len(tests)}"
            print(f"  {dim:<20s} {bar} {avg_quality:.0%} (pass: {pass_str})")

        # Overall scores
        total_tests = len(self.results)
        total_pass = sum(1 for r in self.results if r.success)
        avg_quality = sum(r.quality for r in self.results) / max(total_tests, 1)
        avg_elapsed = sum(r.elapsed for r in self.results) / max(total_tests, 1)
        total_elapsed = sum(r.elapsed for r in self.results)

        print(f"\n{'─' * 80}")
        print(f"  OVERALL METRICS")
        print(f"{'─' * 80}")
        print(f"  Tests run:      {total_tests}")
        print(f"  Tests passed:   {total_pass}/{total_tests} ({total_pass/max(total_tests,1):.0%})")
        print(f"  Avg quality:    {avg_quality:.2f}")
        print(f"  Avg latency:    {avg_elapsed:.1f}s")
        print(f"  Total time:     {total_elapsed:.0f}s")

        # Letter grade
        if avg_quality >= 0.9:
            grade = "A+"
        elif avg_quality >= 0.85:
            grade = "A"
        elif avg_quality >= 0.80:
            grade = "A-"
        elif avg_quality >= 0.75:
            grade = "B+"
        elif avg_quality >= 0.70:
            grade = "B"
        elif avg_quality >= 0.65:
            grade = "B-"
        elif avg_quality >= 0.60:
            grade = "C+"
        elif avg_quality >= 0.55:
            grade = "C"
        elif avg_quality >= 0.50:
            grade = "C-"
        elif avg_quality >= 0.40:
            grade = "D"
        else:
            grade = "F"

        print(f"\n  ╔══════════════════════════════╗")
        print(f"  ║  OVERALL GRADE:  {grade:>4s}        ║")
        print(f"  ╚══════════════════════════════╝")

        # Strengths and weaknesses
        sorted_dims = sorted(dim_scores.items(), key=lambda x: x[1]["quality"], reverse=True)
        print(f"\n  STRENGTHS:")
        for dim, scores in sorted_dims[:3]:
            if scores["quality"] >= 0.5:
                print(f"    + {dim}: {scores['quality']:.0%}")

        print(f"\n  WEAKNESSES:")
        for dim, scores in sorted_dims[-3:]:
            if scores["quality"] < 0.7:
                print(f"    - {dim}: {scores['quality']:.0%}")

        # Detailed ratings
        print(f"\n{'─' * 80}")
        print(f"  DIMENSION RATINGS (out of 10)")
        print(f"{'─' * 80}")

        for dim, scores in sorted(dim_scores.items()):
            rating = scores["quality"] * 10
            stars = "★" * int(rating) + "☆" * (10 - int(rating))
            print(f"  {dim:<20s} {stars} {rating:.1f}/10")

        print(
            f"\n  {'OVERALL':<20s} {'★' * int(avg_quality * 10) + '☆' * (10 - int(avg_quality * 10))} {avg_quality * 10:.1f}/10"
        )

    def _save_results(self):
        results_dir = Path(__file__).parent / "eval_results"
        results_dir.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = results_dir / f"{ts}_supercomplex.json"

        data = {
            "timestamp": datetime.now().isoformat(),
            "model": EVAL_MODEL,
            "results": [
                {
                    "dimension": r.dimension,
                    "test_name": r.test_name,
                    "success": r.success,
                    "quality": r.quality,
                    "content_length": r.content_length,
                    "elapsed": round(r.elapsed, 2),
                    "checks": r.checks,
                    "check_ratio": round(r.check_ratio, 3),
                    "error": r.error,
                    "notes": r.notes,
                }
                for r in self.results
            ],
        }

        with open(out, "w") as f:
            json.dump(data, f, indent=2, default=str)
        print(f"\n  Saved: {out}")


async def main():
    await SuperComplexEval().run_all()


if __name__ == "__main__":
    asyncio.run(main())
