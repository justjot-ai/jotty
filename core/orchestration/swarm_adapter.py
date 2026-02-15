"""
Swarm Adapter - Zero-Wrapper Multi-Swarm Integration
=====================================================

Adapts Jotty's existing components to Multi-Swarm interface automatically.

KISS PRINCIPLE: Simple adapters, no complex wrappers.
DRY PRINCIPLE: Reuses all existing Jotty infrastructure.

Usage:
    from Jotty.core.orchestration import SwarmAdapter, get_multi_swarm_coordinator

    # Option 1: From BaseSwarm
    swarms = SwarmAdapter.from_swarms([
        research_swarm,
        coding_swarm,
        testing_swarm
    ])

    # Option 2: Quick setup with prompts
    swarms = SwarmAdapter.quick_swarms([
        ("Technical", "You're a technical expert"),
        ("Business", "You're a business expert"),
        ("Ethics", "You're an ethics expert"),
    ])

    # Execute
    coordinator = get_multi_swarm_coordinator()
    result = await coordinator.execute_parallel(swarms, task)
"""

import logging
from typing import List, Tuple, Any, Optional
import asyncio

logger = logging.getLogger(__name__)


class SwarmAdapter:
    """
    Zero-code adapter for Multi-Swarm Coordinator.

    Automatically adapts Jotty components to Multi-Swarm interface.
    Users don't need to write wrapper classes.
    """

    @staticmethod
    def from_swarms(swarms: List[Any]) -> List[Any]:
        """
        Adapt existing Jotty swarms to Multi-Swarm interface.

        If swarm already has execute(), uses it directly.
        If swarm has run(), wraps it.

        Args:
            swarms: List of existing Jotty swarms

        Returns:
            List of Multi-Swarm compatible swarms
        """
        from .multi_swarm_coordinator import SwarmResult

        adapted = []

        for swarm in swarms:
            # Check if already compatible
            if hasattr(swarm, 'execute') and asyncio.iscoroutinefunction(swarm.execute):
                adapted.append(swarm)
                logger.debug(f"✅ {getattr(swarm, 'name', 'unnamed')} already compatible")

            # Wrap run() → execute()
            elif hasattr(swarm, 'run'):
                adapted.append(_RunToExecuteAdapter(swarm))
                logger.debug(f"🔧 {getattr(swarm, 'name', 'unnamed')} wrapped (run→execute)")

            else:
                logger.warning(f"⚠️  Swarm {swarm} has no execute() or run() method, skipping")

        return adapted

    @staticmethod
    def quick_swarms(
        configs: List[Tuple[str, str]],
        model: str = "claude-3-5-haiku-20241022",
        max_tokens: int = 200
    ) -> List[Any]:
        """
        Quickly create swarms from (name, system_prompt) tuples.

        Perfect for simple parallel tasks without creating swarm classes.

        Example:
            swarms = SwarmAdapter.quick_swarms([
                ("Researcher", "You research topics deeply"),
                ("Analyst", "You analyze data critically"),
                ("Writer", "You write clear summaries"),
            ])

        Args:
            configs: List of (name, system_prompt) tuples
            model: Model to use for all swarms
            max_tokens: Max tokens per swarm

        Returns:
            List of ready-to-use swarms
        """
        swarms = []

        for name, system_prompt in configs:
            swarms.append(_QuickSwarm(
                name=name,
                system_prompt=system_prompt,
                model=model,
                max_tokens=max_tokens
            ))

        logger.info(f"✅ Created {len(swarms)} quick swarms")

        return swarms

    @staticmethod
    def from_agents(agents: List[Any]) -> List[Any]:
        """
        Adapt Jotty agents to Multi-Swarm interface.

        Args:
            agents: List of Jotty agents

        Returns:
            List of Multi-Swarm compatible swarms
        """
        swarms = []

        for agent in agents:
            swarms.append(_AgentToSwarmAdapter(agent))

        logger.info(f"✅ Wrapped {len(swarms)} agents as swarms")

        return swarms


# =============================================================================
# INTERNAL ADAPTERS
# =============================================================================

class _RunToExecuteAdapter:
    """Adapts swarm.run() → swarm.execute() for Multi-Swarm compatibility."""

    def __init__(self, swarm: Any) -> None:
        self.swarm = swarm
        self.name = getattr(swarm, 'name', 'unnamed')

    async def execute(self, task: str) -> Any:
        """Execute by calling swarm.run()."""
        from .multi_swarm_coordinator import SwarmResult

        try:
            # Call original run() method
            if asyncio.iscoroutinefunction(self.swarm.run):
                result = await self.swarm.run(goal=task)
            else:
                result = self.swarm.run(goal=task)

            # Convert to SwarmResult
            if isinstance(result, dict):
                return SwarmResult(
                    swarm_name=self.name,
                    output=result.get('output', str(result)),
                    success=result.get('success', True),
                    confidence=result.get('confidence', 0.8),
                    metadata=result
                )
            else:
                return SwarmResult(
                    swarm_name=self.name,
                    output=str(result),
                    success=True,
                    confidence=0.8
                )

        except Exception as e:
            logger.error(f"❌ {self.name} execution failed: {e}")
            return SwarmResult(
                swarm_name=self.name,
                output=f"Error: {str(e)}",
                success=False,
                confidence=0.0
            )


class _QuickSwarm:
    """Quick swarm created from name + system prompt."""

    def __init__(self, name: str, system_prompt: str, model: str = 'claude-3-5-haiku-20241022', max_tokens: int = 200) -> None:
        self.name = name
        self.system_prompt = system_prompt
        self.model = model
        self.max_tokens = max_tokens

    async def execute(self, task: str) -> Any:
        """Execute using Anthropic API directly."""
        from .multi_swarm_coordinator import SwarmResult
        import os

        try:
            # Import Anthropic
            from anthropic import AsyncAnthropic

            client = AsyncAnthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))

            # Make API call
            response = await client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=0.7,
                messages=[{
                    "role": "user",
                    "content": f"{self.system_prompt}\n\nTask: {task}"
                }]
            )

            output = response.content[0].text

            # Calculate cost
            cost_usd = (
                (response.usage.input_tokens * 0.25 / 1_000_000) +
                (response.usage.output_tokens * 1.25 / 1_000_000)
            )

            return SwarmResult(
                swarm_name=self.name,
                output=output,
                success=True,
                confidence=0.85,
                metadata={
                    'model': self.model,
                    'input_tokens': response.usage.input_tokens,
                    'output_tokens': response.usage.output_tokens,
                    'cost_usd': cost_usd
                }
            )

        except Exception as e:
            logger.error(f"❌ {self.name} failed: {e}")
            return SwarmResult(
                swarm_name=self.name,
                output=f"Error: {str(e)}",
                success=False,
                confidence=0.0
            )


class _AgentToSwarmAdapter:
    """Adapts Jotty agent to Multi-Swarm interface."""

    def __init__(self, agent: Any) -> None:
        self.agent = agent
        self.name = getattr(agent, 'name', 'agent')

    async def execute(self, task: str) -> Any:
        """Execute by calling agent.run() or agent.execute()."""
        from .multi_swarm_coordinator import SwarmResult

        try:
            # Try execute() first
            if hasattr(self.agent, 'execute'):
                if asyncio.iscoroutinefunction(self.agent.execute):
                    result = await self.agent.execute(task)
                else:
                    result = self.agent.execute(task)
            # Fall back to run()
            elif hasattr(self.agent, 'run'):
                if asyncio.iscoroutinefunction(self.agent.run):
                    result = await self.agent.run(task)
                else:
                    result = self.agent.run(task)
            else:
                raise AttributeError(f"Agent {self.name} has no execute() or run() method")

            return SwarmResult(
                swarm_name=self.name,
                output=str(result),
                success=True,
                confidence=0.8
            )

        except Exception as e:
            logger.error(f"❌ Agent {self.name} failed: {e}")
            return SwarmResult(
                swarm_name=self.name,
                output=f"Error: {str(e)}",
                success=False,
                confidence=0.0
            )


__all__ = [
    'SwarmAdapter',
]
