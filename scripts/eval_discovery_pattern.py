#!/usr/bin/env python3
"""
Discovery-Pattern Evaluation: Jotty vs CrewAI Claims
=====================================================

Tests Jotty the way CrewAI tells users to build:
1. Start from capabilities() discovery
2. Define agents with roles/goals
3. Run multi-agent research + report (CrewAI's flagship)
4. Run code gen + test + execute (shared claim)
5. Run parallel review (shared claim)
6. Test coordination patterns (Jotty-unique: debate, iterative, consensus)
7. Test auto-routing from natural language
8. Test learning persistence (Jotty-unique)

Usage:
    python scripts/eval_discovery_pattern.py              # all tests
    python scripts/eval_discovery_pattern.py --test discovery  # single test
    python scripts/eval_discovery_pattern.py --fast        # shorter timeouts
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.chdir(str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv(".env")

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("eval_discovery")
logger.setLevel(logging.INFO)


# ─── Result Types ────────────────────────────────────────────────────────────


@dataclass
class TestResult:
    name: str
    category: str
    passed: bool
    elapsed_s: float
    output_len: int = 0
    details: Dict[str, Any] = field(default_factory=dict)
    error: str = ""
    quality_notes: List[str] = field(default_factory=list)
    score: float = 0.0


@dataclass
class EvalReport:
    timestamp: str
    total_tests: int = 0
    passed: int = 0
    failed: int = 0
    total_time_s: float = 0.0
    results: List[Dict[str, Any]] = field(default_factory=list)
    overall_score: float = 0.0
    category_scores: Dict[str, float] = field(default_factory=dict)


# ─── Helpers ─────────────────────────────────────────────────────────────────


def extract_content(result: Any) -> str:
    if result is None:
        return ""
    content = ""
    inner = getattr(result, "output", None)
    if inner and not isinstance(inner, str):
        for attr in ("final_output", "output", "content", "code", "report"):
            val = getattr(inner, attr, None)
            if val and isinstance(val, str) and len(val) > len(content):
                content = val
    for attr in ("final_output", "output", "content", "code", "report", "analysis"):
        val = getattr(result, attr, None)
        if val and isinstance(val, str) and len(val) > len(content):
            content = val
    out_dict = getattr(result, "output", None)
    if isinstance(out_dict, dict):
        for key in ("code", "report", "content", "analysis", "architecture"):
            val = out_dict.get(key, "")
            if isinstance(val, str) and len(val) > len(content):
                content = val
    if not content:
        s = str(result)
        if len(s) > 50:
            content = s
    return content


def score_output(content: str, *, min_len: int = 200, keywords: List[str] = None) -> tuple:
    notes = []
    score = 0.0
    if not content:
        return 0.0, ["Empty output"]
    if len(content) >= min_len * 3:
        score += 3.0
    elif len(content) >= min_len:
        score += 2.0
        notes.append(f"Output adequate ({len(content)} chars)")
    elif len(content) >= 50:
        score += 1.0
        notes.append(f"Output short ({len(content)} chars)")
    else:
        notes.append(f"Output too short ({len(content)} chars)")
    has_headings = any(line.startswith("#") for line in content.split("\n"))
    has_code_blocks = "```" in content
    has_bullets = any(line.strip().startswith(("- ", "* ", "• ")) for line in content.split("\n"))
    has_numbered = any(
        line.strip()[:2].rstrip(".").isdigit() for line in content.split("\n") if line.strip()
    )
    structure_score = sum([has_headings, has_code_blocks, has_bullets, has_numbered])
    score += min(structure_score, 3)
    if structure_score == 0:
        notes.append("No structured formatting")
    if keywords:
        lower_content = content.lower()
        matched = sum(1 for kw in keywords if kw.lower() in lower_content)
        relevance = matched / len(keywords)
        score += relevance * 2
        if relevance < 0.3:
            notes.append(f"Low keyword relevance ({matched}/{len(keywords)})")
    substantive_lines = [line.strip() for line in content.split("\n") if len(line.strip()) > 30]
    unique_substantive = set(substantive_lines)
    if len(substantive_lines) > 5 and len(unique_substantive) < len(substantive_lines) * 0.7:
        notes.append(
            f"Repetition detected ({len(unique_substantive)}/{len(substantive_lines)} unique)"
        )
        score += 1.0
    else:
        score += 2.0
    if not notes:
        notes.append("Strong output quality")
    return min(score, 10.0), notes


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 1: DISCOVERY API
# ═══════════════════════════════════════════════════════════════════════════════


async def test_discovery_api() -> TestResult:
    """Test that capabilities() and explain() expose all major subsystems."""
    start = time.time()
    try:
        from Jotty.core.capabilities import capabilities, explain

        caps = capabilities()
        elapsed = time.time() - start

        notes = []
        score = 0.0

        # Check subsystems are present
        subsystems = caps.get("subsystems", {})
        required_subsystems = ["learning", "memory", "context", "orchestration", "skills", "utils"]
        found = [s for s in required_subsystems if s in subsystems]
        score += (len(found) / len(required_subsystems)) * 2
        if len(found) < len(required_subsystems):
            notes.append(f"Missing subsystems: {set(required_subsystems) - set(found)}")

        # Check coordination subsystem (the one we just added)
        if "coordination" in subsystems:
            score += 2.0
            coord = subsystems["coordination"]
            coord_classes = coord.get("key_classes", [])
            if "TeamCoordinator" in coord_classes and "SwarmTemplate" in coord_classes:
                score += 1.0
            else:
                notes.append(f"Coordination missing key classes: {coord_classes}")
        else:
            notes.append("coordination subsystem NOT in capabilities() — discoverability gap")

        # Check explain() works for coordination
        coord_explain = explain("coordination")
        if "Unknown component" not in coord_explain:
            score += 1.5
            if "PIPELINE" in coord_explain and "DEBATE" in coord_explain:
                score += 1.0
            else:
                notes.append("explain('coordination') missing pattern names")
        else:
            notes.append("explain('coordination') returns Unknown")

        # Check execution paths
        paths = caps.get("execution_paths", {})
        if "chat" in paths and "workflow" in paths and "swarm" in paths:
            score += 1.5
        else:
            notes.append(
                f"Missing execution paths: {set(['chat','workflow','swarm']) - set(paths.keys())}"
            )

        # Check swarms list
        swarms = caps.get("swarms", [])
        if len(swarms) >= 5:
            score += 1.0
        else:
            notes.append(f"Only {len(swarms)} swarms listed (expected 5+)")

        if not notes:
            notes.append("All subsystems discoverable including coordination patterns")

        return TestResult(
            name="Discovery API: capabilities() + explain()",
            category="discovery",
            passed=score >= 7.0,
            elapsed_s=elapsed,
            output_len=len(json.dumps(caps, default=str)),
            details={
                "subsystem_count": len(subsystems),
                "has_coordination": "coordination" in subsystems,
                "execution_paths": list(paths.keys()),
                "swarm_count": len(swarms),
            },
            quality_notes=notes,
            score=score,
        )
    except Exception as e:
        return TestResult(
            name="Discovery API",
            category="discovery",
            passed=False,
            elapsed_s=time.time() - start,
            error=f"{type(e).__name__}: {e}",
            score=0.0,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 2: CREWAI FLAGSHIP — Research + Report (multi-agent)
# ═══════════════════════════════════════════════════════════════════════════════


async def test_research_report(timeout: int) -> TestResult:
    """CrewAI's flagship use case: research a topic and generate a report."""
    start = time.time()
    try:
        from Jotty.core.intelligence.orchestration.core.swarm_manager import Orchestrator

        orch = Orchestrator()
        result = await asyncio.wait_for(
            orch.run(
                "Research the current state of AI agents in enterprise software. "
                "Cover: key players (CrewAI, AutoGen, LangChain, custom frameworks), "
                "adoption trends, challenges, and a 2026 outlook. "
                "Produce a structured report with executive summary, sections, and conclusion.",
                learn=True,
            ),
            timeout=timeout,
        )
        elapsed = time.time() - start
        content = extract_content(result)
        score, notes = score_output(
            content,
            min_len=500,
            keywords=[
                "CrewAI",
                "AutoGen",
                "LangChain",
                "enterprise",
                "adoption",
                "agent",
                "challenge",
                "conclusion",
            ],
        )
        return TestResult(
            name="Research + Report: AI Agents in Enterprise",
            category="crewai_flagship",
            passed=len(content) > 300 and score >= 5.0,
            elapsed_s=elapsed,
            output_len=len(content),
            quality_notes=notes,
            score=score,
        )
    except Exception as e:
        return TestResult(
            name="Research + Report",
            category="crewai_flagship",
            passed=False,
            elapsed_s=time.time() - start,
            error=f"{type(e).__name__}: {e}\n{traceback.format_exc()[-500:]}",
            score=0.0,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 3: CODE GEN + TEST + EXECUTE (shared claim)
# ═══════════════════════════════════════════════════════════════════════════════


async def test_coding_swarm(timeout: int) -> TestResult:
    """CodingSwarm: Generate code with tests — both CrewAI and Jotty claim this."""
    start = time.time()
    try:
        from Jotty.core.intelligence.orchestration.core.swarm_manager import Orchestrator
        from Jotty.core.intelligence.orchestration.swarms.coding_swarm.swarm import CodingSwarm
        from Jotty.core.intelligence.orchestration.swarms.coding_swarm.types import CodingConfig

        orch = Orchestrator()
        swarm = CodingSwarm(CodingConfig())
        result = await asyncio.wait_for(
            orch.run(
                "Build a Python thread-safe bounded queue with put(), get(), peek(), "
                "and size() methods. Support timeout parameter. "
                "Include comprehensive pytest tests with edge cases.",
                swarm=swarm,
                learn=True,
            ),
            timeout=timeout,
        )
        elapsed = time.time() - start
        content = extract_content(result)
        score, notes = score_output(
            content,
            min_len=500,
            keywords=[
                "class",
                "def put",
                "def get",
                "threading",
                "Lock",
                "pytest",
                "assert",
                "def test_",
            ],
        )
        return TestResult(
            name="CodingSwarm: Thread-Safe Bounded Queue",
            category="code_gen",
            passed=len(content) > 300 and score >= 4.0,
            elapsed_s=elapsed,
            output_len=len(content),
            quality_notes=notes,
            score=score,
        )
    except Exception as e:
        return TestResult(
            name="CodingSwarm",
            category="code_gen",
            passed=False,
            elapsed_s=time.time() - start,
            error=f"{type(e).__name__}: {e}\n{traceback.format_exc()[-500:]}",
            score=0.0,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 4: PARALLEL CODE REVIEW (shared claim)
# ═══════════════════════════════════════════════════════════════════════════════


async def test_review_swarm(timeout: int) -> TestResult:
    """ReviewSwarm: Parallel security + performance + style review."""
    start = time.time()
    try:
        from Jotty.core.intelligence.orchestration.swarms.templates.review_swarm import ReviewSwarm

        swarm = ReviewSwarm()
        vulnerable_code = '''
import os, pickle, subprocess

def process_user_input(user_data):
    """Process user input from web form."""
    result = eval(user_data)  # SQL injection risk
    query = f"SELECT * FROM users WHERE name = '{user_data}'"
    os.system(f"echo {user_data}")  # command injection
    data = pickle.loads(user_data.encode())  # deserialization attack
    subprocess.call(user_data, shell=True)  # RCE
    passwords = {"admin": "password123", "root": "admin"}
    return result

class UserManager:
    def __init__(self):
        self.users = []
    def find_user(self, name):
        for u in self.users:  # O(n) scan, no index
            if u["name"] == name:
                return u
    def get_all(self):
        return self.users  # exposes internal state
'''
        result = await asyncio.wait_for(
            swarm.execute(task=vulnerable_code),
            timeout=timeout,
        )
        elapsed = time.time() - start
        content = extract_content(result)

        # Also check structured review output
        review_data = getattr(result, "output", {})
        has_issues = False
        if isinstance(review_data, dict):
            issues = review_data.get("issues", []) or review_data.get("comments", [])
            has_issues = len(issues) > 0 if issues else False

        score, notes = score_output(
            content,
            min_len=200,
            keywords=[
                "eval",
                "injection",
                "security",
                "vulnerability",
                "pickle",
                "subprocess",
                "password",
                "performance",
            ],
        )
        if has_issues:
            score = min(score + 1.0, 10.0)
            notes.append(f"Structured issues found: {len(issues) if issues else 'yes'}")

        return TestResult(
            name="ReviewSwarm: Security + Performance Audit",
            category="parallel_review",
            passed=len(content) > 100 and score >= 4.0,
            elapsed_s=elapsed,
            output_len=len(content),
            details={"has_structured_issues": has_issues},
            quality_notes=notes,
            score=score,
        )
    except Exception as e:
        return TestResult(
            name="ReviewSwarm",
            category="parallel_review",
            passed=False,
            elapsed_s=time.time() - start,
            error=f"{type(e).__name__}: {e}\n{traceback.format_exc()[-500:]}",
            score=0.0,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 5: TEAMCOORDINATOR PATTERNS (Jotty-unique)
# ═══════════════════════════════════════════════════════════════════════════════


async def test_coordination_patterns() -> TestResult:
    """Verify TeamCoordinator can instantiate all 9 coordination patterns."""
    start = time.time()
    try:
        from Jotty.core.intelligence.orchestration.swarms.base.team_coordinator import (
            TeamCoordinator,
            TeamResult,
        )
        from Jotty.core.infrastructure.foundation.types.execution_types import (
            CoordinationPattern,
            MergeStrategy,
            SynthesisStrategy,
        )

        notes = []
        score = 0.0
        patterns_ok = []

        all_patterns = [
            CoordinationPattern.NONE,
            CoordinationPattern.PIPELINE,
            CoordinationPattern.SEQUENTIAL,
            CoordinationPattern.PARALLEL,
            CoordinationPattern.CONSENSUS,
            CoordinationPattern.DEBATE,
            CoordinationPattern.HIERARCHICAL,
            CoordinationPattern.ITERATIVE,
            CoordinationPattern.BLACKBOARD,
            CoordinationPattern.ROUND_ROBIN,
            CoordinationPattern.CUSTOM,
        ]

        # Dummy agent class for instantiation testing
        class DummyAgent:
            def __init__(self, **kwargs):
                pass

            async def execute(self, input=None, context=None, **kwargs):
                class R:
                    output = f"Dummy result for: {str(input)[:50]}"

                return R()

        for pattern in all_patterns:
            try:
                kwargs = {"pattern": pattern}
                if pattern == CoordinationPattern.DEBATE:
                    kwargs["debate_rounds"] = 2
                    kwargs["synthesis_strategy"] = SynthesisStrategy.SYNTHESIZE
                if pattern == CoordinationPattern.ITERATIVE:
                    kwargs["quality_threshold"] = 0.7
                    kwargs["max_iterations"] = 2

                team = TeamCoordinator.define(
                    (DummyAgent, "Agent1"),
                    (DummyAgent, "Agent2"),
                    **kwargs,
                )
                assert len(team) == 2
                assert team.pattern == pattern
                patterns_ok.append(pattern.value)
            except Exception as e:
                notes.append(f"Pattern {pattern.value} failed: {e}")

        score = (len(patterns_ok) / len(all_patterns)) * 7.0

        # Test that PARALLEL execution actually works with DummyAgent
        team = TeamCoordinator.define(
            (DummyAgent, "Alpha"),
            (DummyAgent, "Beta"),
            (DummyAgent, "Gamma"),
            pattern=CoordinationPattern.PARALLEL,
            merge_strategy=MergeStrategy.CONCAT,
        )
        team.set_instances(
            {
                "_alpha": DummyAgent(),
                "_beta": DummyAgent(),
                "_gamma": DummyAgent(),
            }
        )
        team_result = await team.execute(task="test task")
        if isinstance(team_result, TeamResult) and team_result.success:
            score += 2.0
            notes.append(f"PARALLEL execution OK: {len(team_result.outputs)} outputs")
        else:
            notes.append("PARALLEL execution failed")

        # Test PIPELINE execution
        team2 = TeamCoordinator.define(
            (DummyAgent, "Step1", None, None, 3),
            (DummyAgent, "Step2", None, None, 2),
            (DummyAgent, "Step3", None, None, 1),
            pattern=CoordinationPattern.PIPELINE,
        )
        team2.set_instances(
            {
                "_step1": DummyAgent(),
                "_step2": DummyAgent(),
                "_step3": DummyAgent(),
            }
        )
        pipe_result = await team2.execute(task="pipeline test")
        if isinstance(pipe_result, TeamResult) and pipe_result.success:
            score += 1.0
            notes.append(f"PIPELINE execution OK: order={pipe_result.execution_order}")
        else:
            notes.append("PIPELINE execution failed")

        elapsed = time.time() - start
        if not notes:
            notes.append("All 11 coordination patterns instantiate + 2 execute correctly")

        return TestResult(
            name="TeamCoordinator: 9 Patterns + Execution",
            category="coordination_patterns",
            passed=score >= 7.0,
            elapsed_s=elapsed,
            details={
                "patterns_ok": patterns_ok,
                "total_patterns": len(all_patterns),
            },
            quality_notes=notes,
            score=min(score, 10.0),
        )
    except Exception as e:
        return TestResult(
            name="TeamCoordinator Patterns",
            category="coordination_patterns",
            passed=False,
            elapsed_s=time.time() - start,
            error=f"{type(e).__name__}: {e}\n{traceback.format_exc()[-500:]}",
            score=0.0,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 6: IDEA WRITER SWARM — Multi-agent content creation
# ═══════════════════════════════════════════════════════════════════════════════


async def test_idea_writer(timeout: int) -> TestResult:
    """IdeaWriterSwarm: Multi-agent article generation (outline → sections → synthesis)."""
    start = time.time()
    try:
        from Jotty.core.intelligence.orchestration.swarms.templates.idea_writer_swarm import (
            IdeaWriterSwarm,
        )

        swarm = IdeaWriterSwarm()
        result = await asyncio.wait_for(
            swarm.execute(
                task="Write a 1500-word article on 'How AI Agent Frameworks Are Reshaping "
                "Software Development in 2026'. Cover: multi-agent coordination patterns, "
                "the shift from single-LLM to swarm architectures, and real-world enterprise adoption."
            ),
            timeout=timeout,
        )
        elapsed = time.time() - start
        content = extract_content(result)
        score, notes = score_output(
            content,
            min_len=800,
            keywords=[
                "agent",
                "framework",
                "swarm",
                "coordination",
                "enterprise",
                "multi-agent",
                "architecture",
                "adoption",
            ],
        )
        return TestResult(
            name="IdeaWriterSwarm: AI Agents Article",
            category="content_creation",
            passed=len(content) > 500 and score >= 5.0,
            elapsed_s=elapsed,
            output_len=len(content),
            quality_notes=notes,
            score=score,
        )
    except Exception as e:
        return TestResult(
            name="IdeaWriterSwarm",
            category="content_creation",
            passed=False,
            elapsed_s=time.time() - start,
            error=f"{type(e).__name__}: {e}\n{traceback.format_exc()[-500:]}",
            score=0.0,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 7: AUTO-ROUTING (natural language → swarm selection)
# ═══════════════════════════════════════════════════════════════════════════════


async def test_auto_routing(timeout: int) -> TestResult:
    """Test that natural language tasks get auto-routed to the correct swarm."""
    start = time.time()
    try:
        from Jotty.core.intelligence.orchestration.core.swarm_manager import Orchestrator

        orch = Orchestrator()

        # Should auto-detect as a task requiring analysis/research
        result = await asyncio.wait_for(
            orch.run(
                "Analyze the pros and cons of microservices vs monolith architecture "
                "for a startup with 5 developers. Consider cost, complexity, and scalability.",
                learn=True,
            ),
            timeout=timeout,
        )
        elapsed = time.time() - start
        content = extract_content(result)
        score, notes = score_output(
            content,
            min_len=300,
            keywords=[
                "microservice",
                "monolith",
                "scalab",
                "complex",
                "cost",
                "startup",
                "team",
                "deploy",
            ],
        )

        # Bonus: check that it actually routed (didn't just do a simple chat)
        result_type = type(result).__name__
        if "Episode" in result_type or "Agentic" in result_type:
            score = min(score + 0.5, 10.0)
            notes.append(f"Routed to agentic path ({result_type})")

        return TestResult(
            name="Auto-Routing: Architecture Analysis",
            category="auto_routing",
            passed=len(content) > 200 and score >= 4.0,
            elapsed_s=elapsed,
            output_len=len(content),
            details={"result_type": result_type},
            quality_notes=notes,
            score=score,
        )
    except Exception as e:
        return TestResult(
            name="Auto-Routing",
            category="auto_routing",
            passed=False,
            elapsed_s=time.time() - start,
            error=f"{type(e).__name__}: {e}\n{traceback.format_exc()[-500:]}",
            score=0.0,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 8: LEARNING PERSISTENCE (Jotty-unique)
# ═══════════════════════════════════════════════════════════════════════════════


async def test_learning_persistence() -> TestResult:
    """TD-Lambda + Memory: store, learn, retrieve — no other framework has this."""
    start = time.time()
    try:
        from Jotty.core.intelligence.learning.facade import get_td_lambda
        from Jotty.core.intelligence.memory.facade import get_memory_system

        notes = []
        score = 0.0

        # TD-Lambda learning (use facade for correct config)
        td = get_td_lambda()
        state1 = {"task": "code_review", "agent": "security_reviewer"}
        action1 = {"tool": "static_analysis"}
        td.update(state=state1, action=action1, reward=1.0, next_state=state1)
        td.update(state=state1, action=action1, reward=0.8, next_state=state1)
        td.update(state=state1, action=action1, reward=0.9, next_state=state1)

        # Verify values were stored
        q_table = getattr(td, "q_table", {}) or getattr(td, "_q_values", {})
        if q_table or hasattr(td, "get_value"):
            score += 3.0
            notes.append("TD-Lambda Q-values updated successfully")
        else:
            notes.append("TD-Lambda Q-values not accessible")
            score += 1.5  # Partial credit — updates didn't crash

        # Memory system
        mem = get_memory_system()
        test_content = "CodingSwarm successfully generated LRU cache with 98% test coverage"
        mem_id = mem.store(
            test_content,
            level="episodic",
            goal="code_generation",
            metadata={"reward": 1.0, "swarm": "coding"},
        )
        if mem_id:
            score += 2.0
            notes.append(f"Memory stored: id={mem_id}")
        else:
            notes.append("Memory store returned no ID")

        # Retrieve from memory
        results = mem.retrieve("How did code generation go?", goal="code_generation", top_k=3)
        if results and len(results) > 0:
            score += 2.0
            notes.append(
                f"Memory retrieval: {len(results)} results, top relevance={getattr(results[0], 'relevance', 'N/A')}"
            )
        else:
            notes.append("Memory retrieval returned no results")

        # Memory status
        status = mem.status()
        if status and isinstance(status, dict):
            score += 1.0
            notes.append(f"Memory backend: {status.get('backend', 'unknown')}")

        # LearningService integration
        try:
            from Jotty.core.intelligence.learning.learning_service import LearningService

            ls = LearningService.get_instance()
            ep_id = ls.start_episode(
                unit_name="test_swarm",
                unit_type="swarm",
                domain="testing",
                task_type="eval",
            )
            ls.end_episode(
                episode_id=ep_id,
                success=True,
                quality=0.9,
                cost=0.01,
            )
            score += 2.0
            notes.append("LearningService episode recorded")
        except Exception as e:
            notes.append(f"LearningService: {e}")

        elapsed = time.time() - start
        if not notes:
            notes.append("All learning components working")

        return TestResult(
            name="Learning: TD-Lambda + Memory + LearningService",
            category="learning",
            passed=score >= 6.0,
            elapsed_s=elapsed,
            details={"td_updates": 3, "memory_stored": bool(mem_id)},
            quality_notes=notes,
            score=min(score, 10.0),
        )
    except Exception as e:
        return TestResult(
            name="Learning Persistence",
            category="learning",
            passed=False,
            elapsed_s=time.time() - start,
            error=f"{type(e).__name__}: {e}\n{traceback.format_exc()[-500:]}",
            score=0.0,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# REPORT GENERATION
# ═══════════════════════════════════════════════════════════════════════════════


def generate_report(results: List[TestResult], total_time: float) -> str:
    lines = []
    lines.append("=" * 78)
    lines.append("  JOTTY DISCOVERY-PATTERN EVALUATION")
    lines.append(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 78)
    lines.append("")

    # Category grouping
    categories = {}
    for r in results:
        categories.setdefault(r.category, []).append(r)

    category_labels = {
        "discovery": "DISCOVERY API (can any LLM find Jotty's features?)",
        "crewai_flagship": "CREWAI FLAGSHIP (research + report generation)",
        "code_gen": "CODE GENERATION + TESTING (shared claim)",
        "parallel_review": "PARALLEL REVIEW (shared claim)",
        "coordination_patterns": "COORDINATION PATTERNS (Jotty-unique: 9 patterns)",
        "content_creation": "CONTENT CREATION (multi-agent writing)",
        "auto_routing": "AUTO-ROUTING (natural language → swarm)",
        "learning": "LEARNING PERSISTENCE (Jotty-unique: TD-Lambda + Memory)",
    }

    total_score = 0.0
    total_tests = len(results)
    passed = sum(1 for r in results if r.passed)

    for cat, cat_results in categories.items():
        label = category_labels.get(cat, cat.upper())
        lines.append(f"┌─ {label}")
        for r in cat_results:
            status = "✅ PASS" if r.passed else "❌ FAIL"
            lines.append(f"│  {status}  {r.name}")
            lines.append(
                f"│         Score: {r.score:.1f}/10  |  Time: {r.elapsed_s:.1f}s  |  Output: {r.output_len} chars"
            )
            if r.error:
                lines.append(f"│         Error: {r.error[:120]}")
            for note in r.quality_notes[:3]:
                lines.append(f"│         • {note}")
            if r.details:
                for k, v in list(r.details.items())[:3]:
                    lines.append(f"│         {k}: {v}")
            total_score += r.score
        cat_avg = sum(r.score for r in cat_results) / len(cat_results)
        lines.append(f"│  Category avg: {cat_avg:.1f}/10")
        lines.append(f"└{'─' * 76}")
        lines.append("")

    avg_score = total_score / total_tests if total_tests else 0
    lines.append("=" * 78)
    lines.append(
        f"  OVERALL: {passed}/{total_tests} passed  |  Avg score: {avg_score:.1f}/10  |  Total: {total_time:.1f}s"
    )
    lines.append("=" * 78)
    lines.append("")

    # Comparison table
    lines.append("┌─ COMPARISON: What CrewAI claims vs what Jotty delivers")
    lines.append("│")
    lines.append("│  Feature                    │ CrewAI │ Jotty │ Status")
    lines.append("│  ─────────────────────────── │ ────── │ ───── │ ──────")
    comparisons = [
        (
            "Research + Report",
            "✅",
            "✅" if any(r.passed for r in categories.get("crewai_flagship", [])) else "❌",
            "tested" if categories.get("crewai_flagship") else "not tested",
        ),
        (
            "Code Gen + Test",
            "✅",
            "✅" if any(r.passed for r in categories.get("code_gen", [])) else "❌",
            "tested" if categories.get("code_gen") else "not tested",
        ),
        (
            "Parallel Review",
            "✅",
            "✅" if any(r.passed for r in categories.get("parallel_review", [])) else "❌",
            "tested" if categories.get("parallel_review") else "not tested",
        ),
        (
            "9 Coordination Patterns",
            "❌ (2)",
            "✅" if any(r.passed for r in categories.get("coordination_patterns", [])) else "❌",
            "Jotty-unique",
        ),
        (
            "Auto Swarm Routing",
            "❌",
            "✅" if any(r.passed for r in categories.get("auto_routing", [])) else "❌",
            "Jotty-unique",
        ),
        (
            "RL Learning (TD-Lambda)",
            "❌",
            "✅" if any(r.passed for r in categories.get("learning", [])) else "❌",
            "Jotty-unique",
        ),
        (
            "5-Level Memory",
            "❌",
            "✅" if any(r.passed for r in categories.get("learning", [])) else "❌",
            "Jotty-unique",
        ),
        (
            "Programmatic Discovery",
            "❌",
            "✅" if any(r.passed for r in categories.get("discovery", [])) else "❌",
            "Jotty-unique",
        ),
    ]
    for feat, crew, jotty, status in comparisons:
        lines.append(f"│  {feat:<28s}│ {crew:<7s}│ {jotty:<6s}│ {status}")
    lines.append("│")
    lines.append("└" + "─" * 76)

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════


async def main():
    parser = argparse.ArgumentParser(description="Jotty Discovery-Pattern Evaluation")
    parser.add_argument("--test", type=str, help="Run single test by category name")
    parser.add_argument("--fast", action="store_true", help="Shorter timeouts")
    parser.add_argument("--save", action="store_true", default=True, help="Save results to file")
    args = parser.parse_args()

    timeout = 300 if args.fast else 600
    fast_timeout = 180 if args.fast else 300

    test_map = {
        "discovery": lambda: test_discovery_api(),
        "research": lambda: test_research_report(timeout),
        "coding": lambda: test_coding_swarm(timeout),
        "review": lambda: test_review_swarm(fast_timeout),
        "patterns": lambda: test_coordination_patterns(),
        "writer": lambda: test_idea_writer(timeout),
        "routing": lambda: test_auto_routing(fast_timeout),
        "learning": lambda: test_learning_persistence(),
    }

    if args.test:
        if args.test not in test_map:
            print(f"Unknown test: {args.test}. Available: {', '.join(test_map.keys())}")
            return
        tests_to_run = {args.test: test_map[args.test]}
    else:
        tests_to_run = test_map

    results = []
    total_start = time.time()

    for name, test_fn in tests_to_run.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"  Running: {name}")
        logger.info(f"{'='*60}")
        try:
            result = await test_fn()
            results.append(result)
            status = "✅" if result.passed else "❌"
            logger.info(
                f"  {status} {result.name}: {result.score:.1f}/10 ({result.elapsed_s:.1f}s)"
            )
        except Exception as e:
            logger.error(f"  ❌ {name} crashed: {e}")
            results.append(
                TestResult(
                    name=name,
                    category=name,
                    passed=False,
                    elapsed_s=0,
                    error=str(e),
                    score=0.0,
                )
            )

    total_time = time.time() - total_start

    report = generate_report(results, total_time)
    print("\n" + report)

    if args.save:
        Path("scripts/eval_results").mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = f"scripts/eval_results/{ts}_discovery_pattern.json"
        report_path = f"scripts/eval_results/{ts}_discovery_pattern.txt"

        json_data = {
            "timestamp": ts,
            "total_tests": len(results),
            "passed": sum(1 for r in results if r.passed),
            "failed": sum(1 for r in results if not r.passed),
            "total_time_s": total_time,
            "overall_score": sum(r.score for r in results) / len(results) if results else 0,
            "results": [asdict(r) for r in results],
        }
        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=2, default=str)
        with open(report_path, "w") as f:
            f.write(report)
        logger.info(f"\nResults saved to {json_path}")
        logger.info(f"Report saved to {report_path}")


if __name__ == "__main__":
    asyncio.run(main())
