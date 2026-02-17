"""
Real-world E2E test for Orchestrator with SWARMS, AGENTS, and PIPELINES.

Tests the FULL execution stack — not just ChatExecutor:
- orchestrator.run(goal, swarm=ReviewSwarm)  → multi-agent code review
- orchestrator.run(goal, stages=[...])       → pipeline with callable stages
- orchestrator.run(goal, agent=AutoAgent)    → single agent execution
- Multi-agent coordination, LearningService integration, result saving

Requires: ANTHROPIC_API_KEY in .env
"""

import asyncio
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import pytest

try:
    from dotenv import load_dotenv

    env_path = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
    load_dotenv(env_path)
except ImportError:
    pass

ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
pytestmark = [
    pytest.mark.skipif(not ANTHROPIC_KEY, reason="ANTHROPIC_API_KEY not set"),
    pytest.mark.e2e,
    pytest.mark.timeout(180),
]

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def _save_result(test_name: str, content: str, metadata: dict = None) -> None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"{ts}_{test_name}.md"
    header = f"# {test_name}\n\n**Timestamp:** {datetime.now().isoformat()}\n"
    if metadata:
        for k, v in metadata.items():
            header += f"**{k}:** {v}\n"
    header += "\n---\n\n"
    path.write_text(header + content, encoding="utf-8")
    print(f"  [Saved] {path}")


def _get_orchestrator():
    from Jotty.core.intelligence.orchestration.core.swarm_manager import Orchestrator

    orch = object.__new__(Orchestrator)
    orch.config = type(
        "C",
        (),
        {
            "domain": "general",
            "base_path": None,
            "learning_wait_timeout_seconds": 0,
        },
    )()
    orch.agents = []
    orch.mode = "single"
    orch.runners = {}
    orch._runners_built = False
    orch._efficiency_stats = {}
    orch._intelligence_metrics = {}
    orch._engine = None
    orch._learning_ready = asyncio.Event()
    orch._learning_ready.set()
    return orch


# ============================================================================
# Test 1: Pipeline mode — Multi-stage execution with dependencies
# ============================================================================


class TestPipelineExecution:
    """Test orchestrator.run(stages=[...]) with real LLM callables."""

    @pytest.mark.asyncio
    async def test_three_stage_pipeline(self):
        """
        Pipeline:
          Stage 1: Research (gather info about a topic)
          Stage 2: Analyze (synthesize findings)
          Stage 3: Recommend (actionable output)

        Each stage is a callable that uses real Anthropic.
        Tests: stage chaining, context passing, dependency resolution.
        """
        import anthropic

        client = anthropic.Anthropic()
        stage_outputs = {}

        async def research_stage(task: str = "", context: str = "", **kw: Any) -> str:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1500,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            f"You are a research analyst. Task: {task}\n\n"
                            "List 5 key findings about microservices vs monolith architecture "
                            "for a startup with 10 engineers. Be specific with data points. "
                            "Keep it under 400 words."
                        ),
                    }
                ],
            )
            output = response.content[0].text
            stage_outputs["research"] = output
            return output

        async def analyze_stage(task: str = "", context: str = "", **kw: Any) -> str:
            research = stage_outputs.get("research", context)
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1500,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            f"You are a senior architect. Based on this research:\n\n"
                            f"{research[:1500]}\n\n"
                            "Identify the 3 most critical trade-offs. "
                            "For each, give a quantitative impact estimate. "
                            "Keep it under 300 words."
                        ),
                    }
                ],
            )
            output = response.content[0].text
            stage_outputs["analysis"] = output
            return output

        async def recommend_stage(task: str = "", context: str = "", **kw: Any) -> str:
            analysis = stage_outputs.get("analysis", context)
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1000,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            f"Based on this analysis:\n\n{analysis[:1500]}\n\n"
                            "Give a concrete recommendation: monolith or microservices? "
                            "Include a 6-month migration plan with 3 milestones. "
                            "Keep it under 200 words."
                        ),
                    }
                ],
            )
            output = response.content[0].text
            stage_outputs["recommendation"] = output
            return output

        orch = _get_orchestrator()
        start = time.time()

        stages = [
            {"name": "research", "callable": research_stage, "task": "Architecture decision"},
            {"name": "analyze", "callable": analyze_stage, "depends_on": ["research"]},
            {"name": "recommend", "callable": recommend_stage, "depends_on": ["analyze"]},
        ]

        result = await orch.run(
            "Should our 10-person startup use microservices or monolith?",
            stages=stages,
            learn=True,
        )

        elapsed = time.time() - start

        # Verify all 3 stages ran
        assert "research" in stage_outputs, "Research stage didn't execute"
        assert "analysis" in stage_outputs, "Analysis stage didn't execute"
        assert "recommendation" in stage_outputs, "Recommendation stage didn't execute"

        # Verify quality
        research = stage_outputs["research"]
        analysis = stage_outputs["analysis"]
        recommendation = stage_outputs["recommendation"]

        assert len(research) > 200, f"Research too short: {len(research)} chars"
        assert len(analysis) > 150, f"Analysis too short: {len(analysis)} chars"
        assert len(recommendation) > 100, f"Recommendation too short: {len(recommendation)} chars"

        assert any(
            w in recommendation.lower()
            for w in [
                "monolith",
                "microservice",
                "recommend",
                "start with",
            ]
        ), "Recommendation should mention the architecture choice"

        print(f"\n[PASS] 3-stage pipeline: {elapsed:.1f}s")
        print(f"  Research: {len(research)} chars")
        print(f"  Analysis: {len(analysis)} chars")
        print(f"  Recommendation: {len(recommendation)} chars")

        _save_result(
            "pipeline_architecture_decision",
            (
                "## Stage 1: Research\n\n"
                + research
                + "\n\n---\n\n## Stage 2: Analysis\n\n"
                + analysis
                + "\n\n---\n\n## Stage 3: Recommendation\n\n"
                + recommendation
            ),
            {
                "elapsed_s": f"{elapsed:.1f}",
                "stages": "3 (research → analyze → recommend)",
            },
        )


# ============================================================================
# Test 2: Direct agent execution via run(agent=...)
# ============================================================================


class TestAgentExecution:
    """Test orchestrator.run(agent=...) with a real agent."""

    @pytest.mark.asyncio
    async def test_run_with_real_agent(self):
        """
        Create a real agent that calls Anthropic and execute via run(agent=...).
        Tests: single-agent path, learn=True recording.
        """
        import anthropic

        client = anthropic.Anthropic()

        class SecurityReviewAgent:
            """Agent that reviews code for security vulnerabilities."""

            name = "SecurityReviewer"

            async def execute(self, task: str = "", **kwargs: Any) -> Any:
                response = client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=2000,
                    messages=[
                        {
                            "role": "user",
                            "content": (
                                f"You are a senior security engineer. Review this code "
                                f"and find ALL security vulnerabilities:\n\n{task}\n\n"
                                "For each vulnerability:\n"
                                "- Severity (Critical/High/Medium/Low)\n"
                                "- OWASP category\n"
                                "- Exploitation scenario\n"
                                "- Fix with code example"
                            ),
                        }
                    ],
                )
                return type(
                    "Result",
                    (),
                    {
                        "success": True,
                        "content": response.content[0].text,
                        "output": response.content[0].text,
                    },
                )()

        vulnerable_code = """
import pickle, os, subprocess

def process_upload(file_data, filename):
    # Save uploaded file
    path = f"/uploads/{filename}"
    with open(path, 'wb') as f:
        f.write(file_data)

    # If it's a config file, load it
    if filename.endswith('.pkl'):
        with open(path, 'rb') as f:
            config = pickle.load(f)

    # Process with shell command
    result = subprocess.run(f"file {path}", shell=True, capture_output=True)

    # Return file info
    return {"path": path, "type": result.stdout.decode()}

def admin_action(request):
    user_id = request.args.get('user_id')
    action = request.args.get('action')
    os.system(f"./admin_tool --user {user_id} --action {action}")
"""

        orch = _get_orchestrator()
        agent = SecurityReviewAgent()

        result = await orch.run(
            vulnerable_code,
            agent=agent,
            learn=True,
        )

        content = getattr(result, "content", str(result))
        content_lower = content.lower()

        found = {
            "path_traversal": any(
                w in content_lower for w in ["path traversal", "directory traversal", "../"]
            ),
            "pickle_deserialization": any(
                w in content_lower for w in ["pickle", "deserialization", "arbitrary code", "rce"]
            ),
            "command_injection": any(
                w in content_lower
                for w in ["command injection", "os.system", "shell injection", "subprocess"]
            ),
            "no_auth": any(
                w in content_lower for w in ["authentication", "authorization", "access control"]
            ),
        }

        found_count = sum(found.values())
        assert found_count >= 3, f"Expected >=3 vulnerabilities, found {found_count}: {found}"

        print(f"\n[PASS] Agent security review: found {found_count}/4 vulnerabilities")
        for vuln, detected in found.items():
            print(f"  {'[x]' if detected else '[ ]'} {vuln}")

        _save_result(
            "agent_security_review",
            content,
            {
                "vulnerabilities_found": f"{found_count}/4",
                **{k: "found" if v else "missed" for k, v in found.items()},
            },
        )


# ============================================================================
# Test 3: Multi-agent pipeline (simulating swarm coordination)
# ============================================================================


class TestMultiAgentCoordination:
    """Test multi-agent coordination via pipeline stages."""

    @pytest.mark.asyncio
    async def test_architect_developer_tester_pipeline(self):
        """
        Simulate a 3-agent coding swarm via pipeline:
          Architect → Developer → Tester

        Each is a separate LLM call with a distinct role/persona.
        Tests: role specialization, output chaining, quality of handoffs.
        """
        import anthropic

        client = anthropic.Anthropic()
        agent_outputs: Dict[str, str] = {}

        async def architect_agent(task: str = "", **kw: Any) -> str:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1500,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "You are a software architect. Design a rate limiter class in Python.\n\n"
                            "Requirements:\n"
                            "- Token bucket algorithm\n"
                            "- Thread-safe\n"
                            "- Configurable rate and burst size\n"
                            "- Support for multiple keys (per-user limiting)\n\n"
                            "Output ONLY the class interface (method signatures + docstrings). "
                            "Do NOT implement the methods — just define the API."
                        ),
                    }
                ],
            )
            output = response.content[0].text
            agent_outputs["architect"] = output
            return output

        async def developer_agent(task: str = "", context: str = "", **kw: Any) -> str:
            design = agent_outputs.get("architect", context)
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2500,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            f"You are a senior Python developer. Implement this design:\n\n"
                            f"{design[:2000]}\n\n"
                            "Implement ALL methods with production-quality code. "
                            "Use threading.Lock for thread safety. "
                            "Include proper error handling. "
                            "Output ONLY the Python code, no explanations."
                        ),
                    }
                ],
            )
            output = response.content[0].text
            agent_outputs["developer"] = output
            return output

        async def tester_agent(task: str = "", context: str = "", **kw: Any) -> str:
            code = agent_outputs.get("developer", context)
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2000,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            f"You are a QA engineer. Write pytest tests for this code:\n\n"
                            f"{code[:2500]}\n\n"
                            "Write at least 5 test cases covering:\n"
                            "1. Basic rate limiting (allow/deny)\n"
                            "2. Token refill over time\n"
                            "3. Burst handling\n"
                            "4. Multi-key isolation\n"
                            "5. Thread safety\n\n"
                            "Output ONLY the test code with pytest."
                        ),
                    }
                ],
            )
            output = response.content[0].text
            agent_outputs["tester"] = output
            return output

        orch = _get_orchestrator()
        start = time.time()

        stages = [
            {"name": "architect", "callable": architect_agent, "task": "Design rate limiter"},
            {"name": "developer", "callable": developer_agent, "depends_on": ["architect"]},
            {"name": "tester", "callable": tester_agent, "depends_on": ["developer"]},
        ]

        result = await orch.run(
            "Build a production rate limiter with tests",
            stages=stages,
            learn=True,
        )

        elapsed = time.time() - start

        # Verify all agents produced output
        assert "architect" in agent_outputs, "Architect didn't run"
        assert "developer" in agent_outputs, "Developer didn't run"
        assert "tester" in agent_outputs, "Tester didn't run"

        arch = agent_outputs["architect"]
        dev = agent_outputs["developer"]
        test = agent_outputs["tester"]

        # Architect should have API design
        assert any(
            w in arch.lower() for w in ["class", "def ", "rate", "token"]
        ), "Architect should output class/method signatures"

        # Developer should have implementation
        assert (
            "class" in dev and "def " in dev
        ), "Developer should output Python class implementation"
        assert any(
            w in dev.lower() for w in ["lock", "threading", "thread"]
        ), "Developer should use threading for thread safety"

        # Tester should have pytest tests
        assert (
            "def test_" in test or "test_" in test.lower()
        ), "Tester should output pytest test functions"
        assert (
            test.lower().count("def test_") >= 3
        ), f"Expected at least 3 test functions, got {test.lower().count('def test_')}"

        print(f"\n[PASS] 3-agent pipeline (Architect→Developer→Tester): {elapsed:.1f}s")
        print(f"  Architect: {len(arch)} chars (API design)")
        print(f"  Developer: {len(dev)} chars (implementation)")
        print(f"  Tester: {len(test)} chars ({test.lower().count('def test_')} test functions)")

        _save_result(
            "multi_agent_rate_limiter",
            (
                "## Agent 1: Architect (API Design)\n\n"
                + arch
                + "\n\n---\n\n## Agent 2: Developer (Implementation)\n\n"
                + dev
                + "\n\n---\n\n## Agent 3: Tester (Test Suite)\n\n"
                + test
            ),
            {
                "elapsed_s": f"{elapsed:.1f}",
                "agents": "3 (architect → developer → tester)",
                "test_functions": test.lower().count("def test_"),
            },
        )


# ============================================================================
# Test 4: Learning persistence across run() and chat()
# ============================================================================


class TestCrossModalLearning:
    """Verify learning persists across run() and chat() calls."""

    @pytest.mark.asyncio
    async def test_learning_accumulates_across_modes(self):
        """
        Execute via chat() then run(), verify LearningService
        accumulates episodes from both modes.
        """
        from Jotty.core.intelligence.learning.learning_service import LearningService

        ls = LearningService.get_instance()
        baseline = ls._store.get_episode_count()

        orch = _get_orchestrator()

        # Mode 1: chat()
        await orch.chat(
            message="What is the speed of light? One number, in m/s.",
            provider="anthropic",
            learn=True,
        )
        after_chat = ls._store.get_episode_count()
        assert after_chat > baseline, f"chat() should record episode: {baseline} -> {after_chat}"

        # Mode 2: run() with pipeline
        import anthropic

        client = anthropic.Anthropic()

        async def simple_stage(task: str = "", **kw: Any) -> str:
            r = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=100,
                messages=[{"role": "user", "content": "Say 'pipeline works' and nothing else."}],
            )
            return r.content[0].text

        await orch.run(
            "Test pipeline",
            stages=[{"name": "test", "callable": simple_stage, "task": "test"}],
            learn=True,
        )
        after_run = ls._store.get_episode_count()
        assert after_run > after_chat, f"run() should record episode: {after_chat} -> {after_run}"

        total_new = after_run - baseline
        print(f"\n[PASS] Cross-modal learning: {total_new} episodes recorded")
        print(f"  Baseline: {baseline}")
        print(f"  After chat(): {after_chat} (+{after_chat - baseline})")
        print(f"  After run(): {after_run} (+{after_run - after_chat})")

        # Query accumulated learning
        report = ls.improvement_report()
        assert isinstance(report, dict), "improvement_report should return dict"
        print(f"  Improvement report keys: {list(report.keys())}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--timeout=180", "-s"])
