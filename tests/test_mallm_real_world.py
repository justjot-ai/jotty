"""
Real-world integration tests for MALLM-inspired Jotty enhancements.

Uses REAL LLM calls but minimizes overhead:
  - Pre-built agents (no zero-config LLM call)
  - skip_validation=True (no architect/auditor LLM calls)
  - Each paradigm test = 2-4 LLM calls only

Run:  pytest tests/test_mallm_real_world.py -v -s --timeout=180
"""

import asyncio
import os
import pytest
import logging

pytestmark = pytest.mark.skipif(
    not os.getenv('ANTHROPIC_API_KEY'),
    reason="Requires ANTHROPIC_API_KEY for real LLM calls"
)

logging.basicConfig(level=logging.INFO, format='%(name)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def _make_lightweight_swarm(agent_specs):
    """
    Create a Orchestrator with pre-defined agents (no zero-config LLM call).

    agent_specs: list of (name, sub_goal) tuples
    """
    from core.orchestration.swarm_manager import Orchestrator
    from core.foundation.agent_config import AgentConfig
    from core.agents.auto_agent import AutoAgent
    from core.foundation.data_structures import SwarmConfig

    agents = []
    for name, sub_goal in agent_specs:
        agent = AutoAgent()
        agents.append(AgentConfig(
            name=name,
            agent=agent,
            capabilities=[sub_goal],
        ))

    config = SwarmConfig()
    sm = Orchestrator(
        agents=agents,
        config=config,
        enable_zero_config=False,  # No LLM call for agent creation
        max_concurrent_agents=2,
    )
    return sm


# =========================================================================
# 1. DISCUSSION PARADIGMS — real LLM (skip_validation for speed)
# =========================================================================

class TestRealParadigms:

    @pytest.mark.asyncio
    @pytest.mark.timeout(180)
    async def test_relay_real(self):
        """Relay: agent A researches, agent B summarizes A's output."""
        status_log = []

        def cb(stage, detail=""):
            status_log.append(stage)
            logger.info(f"  📍 {stage}: {detail}")

        sm = _make_lightweight_swarm([
            ("researcher", "List 3 benefits of exercise"),
            ("summarizer", "Summarize the key points in one sentence"),
        ])

        result = await sm.run(
            goal="What are the benefits of exercise?",
            discussion_paradigm='relay',
            skip_autonomous_setup=True,
            status_callback=cb,
            skip_validation=True,  # Skip architect/auditor = 1 LLM call per agent
        )

        print(f"\n--- RELAY ---")
        print(f"Success: {result.success}")
        print(f"Output: {str(result.output)[:400]}")
        print(f"Status steps: {len(status_log)}")

        assert result is not None
        output_str = str(result.output)
        assert len(output_str) > 10, f"Output too short: {output_str}"
        # Verify relay status messages appeared
        relay_steps = [s for s in status_log if 'Relay' in s]
        assert len(relay_steps) >= 1, f"No relay status: {status_log}"

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_debate_real(self):
        """Debate: two agents argue, then critique each other."""
        sm = _make_lightweight_swarm([
            ("supporter", "Without using any tools, give 3 reasons why morning exercise is better"),
            ("opposer", "Without using any tools, give 3 reasons why evening exercise is better"),
        ])

        result = await sm.run(
            goal="Do not search the web. Just answer from your own knowledge: is morning or evening exercise better? Give reasons for both sides in 2-3 sentences each.",
            discussion_paradigm='debate',
            debate_rounds=2,
            skip_autonomous_setup=True,
            skip_validation=True,
        )

        print(f"\n--- DEBATE ---")
        print(f"Success: {result.success}")
        print(f"Output: {str(result.output)[:400]}")

        assert result is not None
        output_str = str(result.output)
        assert len(output_str) > 10, f"Output too short: {output_str}"

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_refinement_real(self):
        """Refinement: drafter writes, editor improves."""
        sm = _make_lightweight_swarm([
            ("drafter", "Without using any tools, write a 4-line poem about the moon"),
            ("editor", "Without using any tools, improve the poem's word choice"),
        ])

        result = await sm.run(
            goal="Do not search the web. Write a short 4-line poem about the moon. Use only your creativity.",
            discussion_paradigm='refinement',
            refinement_iterations=2,
            skip_autonomous_setup=True,
            skip_validation=True,
        )

        print(f"\n--- REFINEMENT ---")
        print(f"Success: {result.success}")
        print(f"Output: {str(result.output)[:400]}")

        assert result is not None
        output_str = str(result.output)
        assert len(output_str) > 10, f"Output too short: {output_str}"


# =========================================================================
# 2. DECISION PROTOCOLS — real vote data (no LLM needed)
# =========================================================================

class TestRealDecisionProtocols:

    @pytest.mark.asyncio
    async def test_majority_real(self):
        """Majority vote with realistic agent opinions."""
        from core.orchestration.swarm_intelligence import SwarmIntelligence

        si = SwarmIntelligence()

        def vote_func(agent, question, options):
            prefs = {
                "senior_dev": ("microservices", 0.9, "Scalability and team independence"),
                "junior_dev": ("microservices", 0.6, "Everyone uses microservices now"),
                "ops_lead":   ("monolith", 0.85, "Simpler to deploy and monitor"),
                "cto":        ("microservices", 0.75, "Long-term flexibility"),
            }
            return prefs[agent]

        decision = await si.gather_consensus(
            question="Best architecture for our payment service?",
            options=["microservices", "monolith"],
            agents=["senior_dev", "junior_dev", "ops_lead", "cto"],
            vote_func=vote_func,
            protocol='majority',
        )

        print(f"\n--- MAJORITY ---")
        print(f"Decision: {decision.final_decision} (strength: {decision.consensus_strength:.2f})")
        print(f"Dissenting: {decision.dissenting_views}")

        assert decision.final_decision == "microservices"
        assert decision.consensus_strength >= 0.5  # 3/4 = 0.75
        assert len(decision.dissenting_views) == 1
        assert "ops_lead" in decision.dissenting_views[0]

    @pytest.mark.asyncio
    async def test_supermajority_blocks_weak_consensus(self):
        """Supermajority blocks when only simple majority exists."""
        from core.orchestration.swarm_intelligence import SwarmIntelligence

        si = SwarmIntelligence()

        def vote_func(agent, question, options):
            prefs = {
                "dev_1": ("deploy", 0.9, "Tests pass, ready to ship"),
                "dev_2": ("deploy", 0.7, "Looks good to me"),
                "qa":    ("wait", 0.8, "Edge case not covered"),
                "ops":   ("wait", 0.9, "No rollback plan yet"),
            }
            return prefs[agent]

        decision = await si.gather_consensus(
            question="Deploy to production today?",
            options=["deploy", "wait"],
            agents=["dev_1", "dev_2", "qa", "ops"],
            vote_func=vote_func,
            protocol='supermajority',
        )

        print(f"\n--- SUPERMAJORITY ---")
        print(f"Decision: {decision.final_decision} (strength: {decision.consensus_strength:.2f})")

        # 2/4 = 50% < 66% → penalized strength
        assert decision.consensus_strength < 0.66, \
            f"Supermajority should NOT be reached at 50%: {decision.consensus_strength}"

    @pytest.mark.asyncio
    async def test_unanimity_with_full_agreement(self):
        """Unanimity passes when everyone agrees."""
        from core.orchestration.swarm_intelligence import SwarmIntelligence

        si = SwarmIntelligence()

        def vote_func(agent, question, options):
            return "patch_now", 0.95, f"{agent}: Critical vulnerability, patch immediately"

        decision = await si.gather_consensus(
            question="How to handle the CVE-2025-1234 vulnerability?",
            options=["patch_now", "schedule_next_sprint", "ignore"],
            agents=["sec_lead", "dev_lead", "cto"],
            vote_func=vote_func,
            protocol='unanimity',
        )

        print(f"\n--- UNANIMITY ---")
        print(f"Decision: {decision.final_decision} (strength: {decision.consensus_strength:.2f})")

        assert decision.final_decision == "patch_now"
        assert decision.consensus_strength == 1.0
        assert len(decision.dissenting_views) == 0

    @pytest.mark.asyncio
    async def test_ranked_eliminates_weakest(self):
        """Ranked-choice eliminates least popular option."""
        from core.orchestration.swarm_intelligence import SwarmIntelligence

        si = SwarmIntelligence()

        def vote_func(agent, question, options):
            prefs = {
                "alice":  ("react", 0.9, "Ecosystem and community"),
                "bob":    ("vue", 0.8, "Simpler learning curve"),
                "carol":  ("react", 0.7, "More jobs available"),
                "dave":   ("svelte", 0.6, "Modern and fast"),
                "eve":    ("react", 0.85, "TypeScript support"),
            }
            return prefs[agent]

        decision = await si.gather_consensus(
            question="Which frontend framework for our new project?",
            options=["react", "vue", "svelte"],
            agents=["alice", "bob", "carol", "dave", "eve"],
            vote_func=vote_func,
            protocol='ranked',
        )

        print(f"\n--- RANKED ---")
        print(f"Decision: {decision.final_decision} (strength: {decision.consensus_strength:.2f})")

        assert decision.final_decision == "react"  # 3/5 first-choice votes

    @pytest.mark.asyncio
    async def test_approval_filters_low_confidence(self):
        """Approval voting ignores low-confidence votes."""
        from core.orchestration.swarm_intelligence import SwarmIntelligence

        si = SwarmIntelligence()

        def vote_func(agent, question, options):
            prefs = {
                "hiring_mgr":  ("hire", 0.95, "Strong culture fit"),
                "tech_lead":   ("hire", 0.8, "Good technical skills"),
                "team_member": ("hire", 0.3, "Not sure, seemed ok"),  # Low confidence
                "hr":          ("pass", 0.4, "Salary expectations high"),  # Low confidence
            }
            return prefs[agent]

        decision = await si.gather_consensus(
            question="Should we extend an offer to candidate X?",
            options=["hire", "pass"],
            agents=["hiring_mgr", "tech_lead", "team_member", "hr"],
            vote_func=vote_func,
            protocol='approval',
        )

        print(f"\n--- APPROVAL ---")
        print(f"Decision: {decision.final_decision} (strength: {decision.consensus_strength:.2f})")

        # Only hiring_mgr (0.95) and tech_lead (0.8) are confident enough
        # team_member (0.3) and hr (0.4) are below 0.5 threshold
        assert decision.final_decision == "hire"


# =========================================================================
# 3. STRUCTURED COMPRESSION — real content
# =========================================================================

class TestRealCompression:

    def test_compress_agent_execution_log(self):
        """Compress a realistic multi-agent execution log."""
        from core.context.context_guard import LLMContextManager

        mgr = LLMContextManager()

        log = "\n".join([
            "Agent researcher started execution",
            "Searching web for: machine learning deployment best practices",
            "Found 12 results from Google",
            'Result 1: "ML Model Deployment Patterns" - score 0.95',
            'Result 2: "MLOps Best Practices 2025" - score 0.91',
            'Result 3: "Scaling ML in Production" - score 0.87',
            "Processing search results...",
            "Error: Rate limit hit on result 7 (retrying)",
            "Successfully extracted key information",
            "Summary: ML deployment best practices include:",
            "  1. Use model versioning (MLflow, DVC)",
            "  2. Implement A/B testing for model rollouts",
            "  3. Monitor model drift with statistical tests",
            "  4. Use feature stores for consistency",
            "  5. Containerize models (Docker + K8s)",
            "Agent researcher completed in 8.2s",
            "Tokens: 2,400 input, 600 output",
        ] * 8)  # Repeat to make it long

        compressed = mgr.compress_structured(
            log,
            goal="machine learning deployment",
            max_chars=900,
        )

        print(f"\n--- COMPRESSION ---")
        print(f"Original: {len(log)} chars → Compressed: {len(compressed)} chars")
        print(f"Ratio: {len(compressed)/len(log):.1%}")
        print(f"\n{compressed}")

        assert '[Compressed' in compressed
        assert 'Key findings:' in compressed
        assert 'machine learning' in compressed.lower() or 'deployment' in compressed.lower() or 'ml' in compressed.lower()
        assert len(compressed) < len(log)

    def test_smart_compress_in_context_guard_flow(self):
        """Test that _smart_compress works in the build_context flow."""
        from core.context.context_guard import LLMContextManager

        mgr = LLMContextManager(max_tokens=500, safety_margin=100)

        # Register content that exceeds the budget
        mgr.register_critical("Goal", "Deploy ML model to production")
        mgr.register("Logs", "x " * 5000, priority=LLMContextManager.HIGH)

        # build_context will trigger compression
        context, meta = asyncio.run(
            mgr.build_context()
        )

        print(f"\n--- CONTEXT GUARD FLOW ---")
        print(f"Context length: {len(context)} chars")
        print(f"Utilization: {meta['utilization']:.1%}")

        assert len(context) > 0
        assert meta['utilization'] <= 1.1  # Should be within budget


# =========================================================================
# 4. SCOPED TOOLS — real registry
# =========================================================================

class TestRealScopedTools:

    def test_scoped_tools_web_task(self):
        from core.registry.unified_registry import get_unified_registry

        registry = get_unified_registry()
        all_count = len(registry.list_skills())

        tools = registry.get_scoped_tools(
            "search the web for AI news and create a PDF report",
            max_tools=8,
            format='names',
        )

        print(f"\n--- SCOPED TOOLS (web) ---")
        print(f"All skills: {all_count}, Scoped: {len(tools)}")
        print(f"Tools: {tools}")

        assert len(tools) <= 8
        assert len(tools) < all_count

    def test_scoped_tools_telegram_task(self):
        from core.registry.unified_registry import get_unified_registry

        registry = get_unified_registry()

        tools = registry.get_scoped_tools(
            "send a message on telegram",
            max_tools=5,
            format='names',
        )

        print(f"\n--- SCOPED TOOLS (telegram) ---")
        print(f"Tools: {tools}")

        assert len(tools) <= 5
        telegram_related = [t for t in tools if 'telegram' in t.lower()]
        assert len(telegram_related) > 0, f"No telegram tool found in {tools}"


# =========================================================================
# 5. LIFECYCLE HOOKS — real agent run
# =========================================================================

class TestRealHooks:

    @pytest.mark.asyncio
    @pytest.mark.timeout(120)
    async def test_hooks_fire_on_real_run(self):
        """Verify hooks fire during a real agent execution."""
        hook_log = []

        sm = _make_lightweight_swarm([
            ("math", "Solve math problems"),
        ])
        sm._ensure_runners()

        # Register hooks
        runner = sm.runners["math"]
        runner.add_hook('pre_run', lambda **ctx: hook_log.append(
            f"pre_run: {ctx.get('goal', '?')[:30]}"
        ))
        runner.add_hook('post_run', lambda **ctx: hook_log.append(
            f"post_run: success={ctx.get('success', '?')}"
        ))

        result = await sm.run(
            goal="What is 15 * 7? Reply with just the number.",
            skip_autonomous_setup=True,
            skip_validation=True,
        )

        print(f"\n--- HOOKS ---")
        print(f"Hook log: {hook_log}")
        print(f"Output: {str(result.output)[:200]}")

        assert any('pre_run' in h for h in hook_log), f"pre_run didn't fire: {hook_log}"
        assert any('post_run' in h for h in hook_log), f"post_run didn't fire: {hook_log}"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s', '--timeout=180'])
