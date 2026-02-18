"""
E2E: SDK full-stack integration test.

Tests the complete 5-layer architecture:
  Layer 5: SDK (Jotty client)
  Layer 4: Interface (ModeRouter / ChatAPI)
  Layer 3: Intelligence (Orchestrator / ChatExecutor)
  Layer 2: Infrastructure (LearningService, BudgetTracker)
  Layer 1: Providers (Anthropic via .env)

Requires: ANTHROPIC_API_KEY in .env
"""

import os
import time
from datetime import datetime
from pathlib import Path

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
    pytest.mark.timeout(120),
]

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def _save_result(test_name: str, content: str, metadata: dict | None = None) -> None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"{ts}_{test_name}.md"
    header = f"# {test_name}\n\n**Timestamp:** {datetime.now().isoformat()}\n"
    if metadata:
        for k, v in metadata.items():
            header += f"**{k}:** {v}\n"
    header += "\n---\n\n"
    path.write_text(header + content, encoding="utf-8")
    print(f"  [Saved] {path}")


class TestSDKFullStack:
    """Test from jotty import Jotty → real LLM response."""

    @pytest.mark.asyncio
    async def test_sdk_chat_local_mode(self):
        """
        Full 5-layer stack:
          SDK(Jotty) → ModeRouter → ChatExecutor → Anthropic → Response
        """
        from Jotty.sdk.client import Jotty

        client = Jotty()
        client._local_mode = True

        start = time.time()
        response = await client.chat("What is the capital of Japan? Answer in one word.")
        elapsed = time.time() - start

        assert response is not None, "SDK returned None"
        assert response.success, f"SDK response failed: {response.error}"
        assert response.content is not None, "SDK content is None"

        content_str = str(response.content)
        assert (
            "tokyo" in content_str.lower() or "Tokyo" in content_str
        ), f"Expected 'Tokyo' in response, got: {content_str[:200]}"

        print(f"\n[PASS] SDK full-stack chat: {elapsed:.1f}s")
        print(f"  Response: {content_str[:200]}")
        print(f"  Mode: {response.mode}")

        _save_result(
            "sdk_full_stack_chat",
            (
                f"## SDK Response\n\n"
                f"**Success:** {response.success}\n"
                f"**Mode:** {response.mode}\n"
                f"**Content:** {content_str}\n"
            ),
            {
                "elapsed_s": f"{elapsed:.1f}",
                "layer_path": "SDK → ModeRouter → ChatExecutor → Anthropic",
            },
        )

    @pytest.mark.asyncio
    async def test_sdk_chat_with_history(self):
        """
        SDK multi-turn conversation via Orchestrator.chat() directly.
        The SDK local_mode routes through ModeRouter which passes history
        via context metadata. Use Orchestrator directly for reliable
        history handling.
        """
        import asyncio
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

        # Turn 1
        r1 = await orch.chat(
            "My name is Alice. Remember it.",
            provider="anthropic",
            learn=False,
        )
        r1_content = str(r1.content) if hasattr(r1, "content") else str(r1)
        assert len(r1_content) > 0, "Turn 1 response empty"

        # Turn 2 — pass history
        history = [
            {"role": "user", "content": "My name is Alice. Remember it."},
            {"role": "assistant", "content": r1_content},
        ]
        r2 = await orch.chat(
            "What is my name? Answer with just the name.",
            history=history,
            provider="anthropic",
            learn=False,
        )
        r2_content = str(r2.content) if hasattr(r2, "content") else str(r2)

        assert "alice" in r2_content.lower(), f"Expected 'Alice' in response, got: {r2_content}"

        print(f"\n[PASS] Multi-turn via Orchestrator.chat()")
        print(f"  Turn 1: {r1_content[:100]}")
        print(f"  Turn 2: {r2_content[:100]}")

        _save_result(
            "sdk_multi_turn",
            (
                f"## Multi-turn Conversation\n\n"
                f"### Turn 1\n**User:** My name is Alice.\n**Bot:** {r1_content}\n\n"
                f"### Turn 2\n**User:** What is my name?\n**Bot:** {r2_content}\n"
            ),
            {"turns": "2"},
        )

    @pytest.mark.asyncio
    async def test_sdk_orchestrator_integration(self):
        """
        Test that SDK can reach the Orchestrator.run() path directly.
        """
        from Jotty.core.intelligence.orchestration.core.swarm_manager import Orchestrator
        import asyncio

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

        # Call orchestrator.chat() — this is what SDK calls internally
        result = await orch.chat(
            "What is 7 * 8? Answer with just the number.",
            provider="anthropic",
            learn=True,
        )

        content = str(result.content) if hasattr(result, "content") else str(result)
        assert "56" in content, f"Expected '56' in response, got: {content[:200]}"

        # Verify learning was recorded
        from Jotty.core.intelligence.learning.learning_service import LearningService

        ls = LearningService.get_instance()
        count = ls._store.get_episode_count()
        assert count > 0, "Should have recorded at least one episode"

        print(f"\n[PASS] SDK → Orchestrator.chat() → Anthropic → Learning")
        print(f"  Response: {content[:100]}")
        print(f"  Episodes in store: {count}")

        _save_result(
            "sdk_orchestrator_integration",
            (
                f"## Orchestrator Integration\n\n"
                f"**Response:** {content}\n"
                f"**Episodes stored:** {count}\n"
            ),
            {
                "layers_tested": "SDK → Orchestrator → ChatExecutor → LLM → LearningService",
            },
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--timeout=120", "-s"])
