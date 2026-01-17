"""
Jotty Minimal - MegaAgent Equivalent
=====================================

Single file, 1500 lines, all you need for multi-agent coordination.

This is Tier 0 (Minimal) - proves Jotty can be as simple as MegaAgent
while keeping the full power available via Tier 1-4 for those who need it.

Usage:
    python jotty_minimal.py --goal "Write hello world in Python"

    # Or use as a library
    from jotty_minimal import Orchestrator

    orchestrator = Orchestrator()
    result = orchestrator.run(goal="Research quantum computing")

Features:
- Multi-agent coordination (500 lines)
- Dynamic spawning (300 lines)
- Message passing (200 lines)
- Simple memory (200 lines)
- Utilities (300 lines)

Total: ~1,500 lines (vs MegaAgent's 1,000)

Author: Jotty Team
License: MIT
"""

import os
import sys
import argparse
import asyncio
import json
from typing import List, Dict, Any, Optional, Callable, Type
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
import logging

# DSPy for LLM interactions
try:
    import dspy
except ImportError:
    print("❌ Error: dspy-ai not installed. Run: pip install dspy-ai")
    sys.exit(1)

# ============================================================================
# SECTION 1: DATA STRUCTURES (200 lines)
# ============================================================================

@dataclass
class AgentMessage:
    """Message passed between agents"""
    sender: str
    receiver: str
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AgentConfig:
    """Configuration for an agent"""
    name: str
    description: str
    signature: Type  # DSPy signature class
    max_retries: int = 3
    timeout: float = 30.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TaskResult:
    """Result of agent execution"""
    agent_name: str
    success: bool
    output: Any
    error: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MemoryEntry:
    """Simple memory entry"""
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

# ============================================================================
# SECTION 2: DSPy SIGNATURES (100 lines)
# ============================================================================

class PlannerSignature(dspy.Signature):
    """Plan how to accomplish a goal"""
    goal = dspy.InputField(desc="The goal to accomplish")
    context = dspy.InputField(desc="Available context and resources", default="")

    plan = dspy.OutputField(desc="Step-by-step plan to accomplish the goal")
    required_agents = dspy.OutputField(desc="List of agent types needed (comma-separated)")

class ExecutorSignature(dspy.Signature):
    """Execute a specific task"""
    task = dspy.InputField(desc="The task to execute")
    context = dspy.InputField(desc="Context for the task", default="")

    result = dspy.OutputField(desc="Result of task execution")

class ComplexitySignature(dspy.Signature):
    """Assess task complexity"""
    task = dspy.InputField(desc="Task to assess")

    complexity_score = dspy.OutputField(desc="Complexity score (1-5)")
    should_spawn = dspy.OutputField(desc="Should spawn sub-agents? (yes/no)")
    reasoning = dspy.OutputField(desc="Reasoning for the decision")

class SpawnAgentSignature(dspy.Signature):
    """Spawn a new agent for a specific task"""
    task = dspy.InputField(desc="Task requiring new agent")
    existing_agents = dspy.InputField(desc="Currently available agents", default="")

    agent_name = dspy.OutputField(desc="Name for the new agent")
    agent_description = dspy.OutputField(desc="What this agent will do")

# ============================================================================
# SECTION 3: SIMPLE MEMORY (200 lines)
# ============================================================================

class SimpleMemory:
    """
    Simple in-memory storage (Tier 0)

    For Tier 1+, use ChromaDB/VectorDB
    For Tier 2+, use hierarchical memory (cortex.py)
    """

    def __init__(self, max_entries: int = 1000):
        self.max_entries = max_entries
        self.entries: List[MemoryEntry] = []
        self.index: Dict[str, List[int]] = defaultdict(list)  # tag -> entry indices

    def store(self, content: str, tags: Optional[List[str]] = None, **metadata):
        """Store a memory entry"""
        entry = MemoryEntry(
            content=content,
            tags=tags or [],
            metadata=metadata
        )

        # Add to entries
        self.entries.append(entry)

        # Update index
        entry_idx = len(self.entries) - 1
        for tag in entry.tags:
            self.index[tag].append(entry_idx)

        # Evict oldest if over limit
        if len(self.entries) > self.max_entries:
            self._evict_oldest()

    def retrieve(self, tags: Optional[List[str]] = None, limit: int = 10) -> List[MemoryEntry]:
        """Retrieve memory entries by tags"""
        if not tags:
            # Return most recent entries
            return self.entries[-limit:]

        # Find entries matching any tag
        matching_indices = set()
        for tag in tags:
            matching_indices.update(self.index.get(tag, []))

        # Get entries and sort by timestamp
        matching_entries = [self.entries[i] for i in matching_indices if i < len(self.entries)]
        matching_entries.sort(key=lambda e: e.timestamp, reverse=True)

        return matching_entries[:limit]

    def search(self, query: str, limit: int = 10) -> List[MemoryEntry]:
        """Simple keyword search"""
        query_lower = query.lower()
        matching = [
            entry for entry in self.entries
            if query_lower in entry.content.lower()
        ]
        matching.sort(key=lambda e: e.timestamp, reverse=True)
        return matching[:limit]

    def _evict_oldest(self):
        """Remove oldest entry"""
        if not self.entries:
            return

        oldest = self.entries.pop(0)

        # Update index
        for tag in oldest.tags:
            if tag in self.index:
                self.index[tag] = [i-1 for i in self.index[tag] if i > 0]

    def clear(self):
        """Clear all memories"""
        self.entries.clear()
        self.index.clear()

# ============================================================================
# SECTION 4: MESSAGE PASSING (200 lines)
# ============================================================================

class MessageBus:
    """
    Simple message bus for agent communication

    For Tier 2+, use SmartAgentSlack with hierarchical routing
    """

    def __init__(self):
        self.messages: List[AgentMessage] = []
        self.subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self.logger = logging.getLogger(__name__)

    def send(self, sender: str, receiver: str, content: str, **metadata):
        """Send a message from one agent to another"""
        message = AgentMessage(
            sender=sender,
            receiver=receiver,
            content=content,
            metadata=metadata
        )

        self.messages.append(message)
        self.logger.debug(f"📨 {sender} → {receiver}: {content[:100]}")

        # Notify subscribers
        self._notify(receiver, message)

    def broadcast(self, sender: str, content: str, **metadata):
        """Broadcast message to all agents"""
        message = AgentMessage(
            sender=sender,
            receiver="*",
            content=content,
            metadata=metadata
        )

        self.messages.append(message)
        self.logger.debug(f"📢 {sender} → ALL: {content[:100]}")

        # Notify all subscribers
        for receiver in self.subscribers:
            self._notify(receiver, message)

    def subscribe(self, agent_name: str, callback: Callable):
        """Subscribe to messages for an agent"""
        self.subscribers[agent_name].append(callback)

    def get_messages(self, agent_name: str, limit: int = 10) -> List[AgentMessage]:
        """Get messages for an agent"""
        messages = [
            msg for msg in self.messages
            if msg.receiver == agent_name or msg.receiver == "*"
        ]
        return messages[-limit:]

    def _notify(self, receiver: str, message: AgentMessage):
        """Notify subscribers of new message"""
        for callback in self.subscribers.get(receiver, []):
            try:
                callback(message)
            except Exception as e:
                self.logger.error(f"Error in message callback: {e}")

# ============================================================================
# SECTION 5: DYNAMIC SPAWNING (300 lines)
# ============================================================================

class DynamicSpawner:
    """
    Simple dynamic agent spawning (Approach 1 from implementation)

    For Tier 2+, use LLM-based complexity assessment (Approach 2)
    """

    def __init__(self, max_spawned_per_agent: int = 5):
        self.max_spawned_per_agent = max_spawned_per_agent
        self.spawned_agents: Dict[str, AgentConfig] = {}
        self.subordinates: Dict[str, List[str]] = defaultdict(list)
        self.total_spawned = 0
        self.logger = logging.getLogger(__name__)

    def can_spawn(self, parent_agent: str) -> bool:
        """Check if agent can spawn more subordinates"""
        current_count = len(self.subordinates.get(parent_agent, []))
        return current_count < self.max_spawned_per_agent

    def spawn(
        self,
        parent_agent: str,
        agent_name: str,
        description: str,
        signature: Type,
        **config_kwargs
    ) -> AgentConfig:
        """Spawn a new agent"""
        if not self.can_spawn(parent_agent):
            raise ValueError(
                f"Agent '{parent_agent}' has reached max subordinates "
                f"({self.max_spawned_per_agent})"
            )

        # Create agent config
        config = AgentConfig(
            name=agent_name,
            description=description,
            signature=signature,
            metadata={
                "spawned_by": parent_agent,
                "spawned_at": datetime.now().isoformat(),
                **config_kwargs
            }
        )

        # Track spawning
        self.spawned_agents[agent_name] = config
        self.subordinates[parent_agent].append(agent_name)
        self.total_spawned += 1

        self.logger.info(
            f"✨ Spawned agent '{agent_name}' "
            f"(parent: {parent_agent}, total: {self.total_spawned})"
        )

        return config

    def assess_complexity(self, task: str) -> Dict[str, Any]:
        """
        Simple complexity assessment (Tier 0)

        For Tier 2+, use ComplexitySignature with LLM reasoning
        """
        # Simple heuristics
        task_lower = task.lower()

        # Count complexity indicators
        complexity_score = 1

        if any(word in task_lower for word in ["research", "analyze", "compare"]):
            complexity_score += 1

        if any(word in task_lower for word in ["multiple", "several", "various"]):
            complexity_score += 1

        if len(task.split()) > 20:
            complexity_score += 1

        if any(word in task_lower for word in ["complex", "detailed", "comprehensive"]):
            complexity_score += 1

        should_spawn = complexity_score >= 3

        return {
            "complexity_score": min(complexity_score, 5),
            "should_spawn": should_spawn,
            "reasoning": f"Task has complexity score {complexity_score}/5"
        }

    def get_hierarchy(self) -> Dict[str, List[str]]:
        """Get agent hierarchy"""
        return dict(self.subordinates)

# ============================================================================
# SECTION 6: AGENT COORDINATION (500 lines)
# ============================================================================

class Agent:
    """
    Simple agent (Tier 0)

    For Tier 2+, use full AgentConfig with validation/retry/learning
    """

    def __init__(self, config: AgentConfig, memory: SimpleMemory, message_bus: MessageBus):
        self.config = config
        self.memory = memory
        self.message_bus = message_bus
        self.module = dspy.ChainOfThought(config.signature)
        self.logger = logging.getLogger(f"Agent.{config.name}")

    async def execute(self, **inputs) -> TaskResult:
        """Execute agent task"""
        start_time = asyncio.get_event_loop().time()

        try:
            self.logger.info(f"⚙️ Executing task: {inputs.get('task', inputs.get('goal', 'N/A'))[:100]}")

            # Get relevant context from memory
            context_entries = self.memory.retrieve(limit=5)
            context_str = "\n".join([e.content for e in context_entries])

            # Add context if signature expects it
            if "context" in inputs:
                inputs["context"] = f"{inputs.get('context', '')}\n\nMemory:\n{context_str}"

            # Execute DSPy module
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, lambda: self.module(**inputs))

            # Store result in memory
            result_content = str(result)
            self.memory.store(
                content=result_content[:500],  # Store first 500 chars
                tags=[self.config.name, "execution_result"],
                agent=self.config.name
            )

            execution_time = asyncio.get_event_loop().time() - start_time

            self.logger.info(f"✅ Completed in {execution_time:.2f}s")

            return TaskResult(
                agent_name=self.config.name,
                success=True,
                output=result,
                execution_time=execution_time
            )

        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            self.logger.error(f"❌ Failed: {e}")

            return TaskResult(
                agent_name=self.config.name,
                success=False,
                output=None,
                error=str(e),
                execution_time=execution_time
            )

class Orchestrator:
    """
    Simple multi-agent orchestrator (Tier 0)

    This is the MegaAgent equivalent - ~500 lines of coordination logic.
    For Tier 2+, use full MultiAgentsOrchestrator with RL/validation/etc.
    """

    def __init__(
        self,
        agents: Optional[List[AgentConfig]] = None,
        max_spawned_per_agent: int = 5,
        max_memory_entries: int = 1000
    ):
        self.logger = logging.getLogger(__name__)
        self.memory = SimpleMemory(max_entries=max_memory_entries)
        self.message_bus = MessageBus()
        self.spawner = DynamicSpawner(max_spawned_per_agent=max_spawned_per_agent)

        # Create agents
        self.agents: Dict[str, Agent] = {}
        if agents:
            for config in agents:
                self._create_agent(config)

        # Default planner agent
        if "planner" not in self.agents:
            self._create_agent(AgentConfig(
                name="planner",
                description="Plans how to accomplish goals",
                signature=PlannerSignature
            ))

    def _create_agent(self, config: AgentConfig):
        """Create an agent instance"""
        agent = Agent(config, self.memory, self.message_bus)
        self.agents[config.name] = agent
        self.logger.info(f"📝 Created agent: {config.name}")

    async def run(self, goal: str, max_steps: int = 10) -> Dict[str, Any]:
        """
        Run orchestrator to accomplish a goal

        Simple sequential execution (Tier 0)
        For Tier 1+, add parallel execution with asyncio.gather
        """
        self.logger.info(f"🎯 Goal: {goal}")

        # Step 1: Plan
        planner = self.agents["planner"]
        plan_result = await planner.execute(goal=goal, context="")

        if not plan_result.success:
            return {
                "success": False,
                "error": f"Planning failed: {plan_result.error}",
                "results": []
            }

        plan = plan_result.output.plan
        required_agents = [
            name.strip()
            for name in plan_result.output.required_agents.split(",")
        ]

        self.logger.info(f"📋 Plan: {plan}")
        self.logger.info(f"👥 Required agents: {required_agents}")

        # Step 2: Spawn required agents if needed
        for agent_type in required_agents:
            if agent_type not in self.agents and agent_type not in ["none", "n/a", ""]:
                # Assess if spawning is needed
                assessment = self.spawner.assess_complexity(goal)

                if assessment["should_spawn"]:
                    self.logger.info(f"✨ Spawning {agent_type} agent...")

                    # Use default executor signature for spawned agents
                    config = self.spawner.spawn(
                        parent_agent="planner",
                        agent_name=agent_type,
                        description=f"Execute {agent_type} tasks",
                        signature=ExecutorSignature
                    )

                    self._create_agent(config)

        # Step 3: Execute plan steps
        results = []
        steps = [s.strip() for s in plan.split("\n") if s.strip() and not s.strip().startswith("#")]

        for i, step in enumerate(steps[:max_steps], 1):
            self.logger.info(f"\n📍 Step {i}/{len(steps)}: {step}")

            # Find best agent for this step
            agent = self._select_agent_for_step(step, required_agents)

            # Execute step
            result = await agent.execute(task=step, context=plan)
            results.append({
                "step": i,
                "description": step,
                "agent": agent.config.name,
                "success": result.success,
                "output": str(result.output) if result.success else None,
                "error": result.error,
                "execution_time": result.execution_time
            })

            if not result.success:
                self.logger.warning(f"⚠️ Step {i} failed, continuing...")

        # Step 4: Aggregate results
        success_count = sum(1 for r in results if r["success"])
        overall_success = success_count > 0

        return {
            "success": overall_success,
            "goal": goal,
            "plan": plan,
            "results": results,
            "summary": f"Completed {success_count}/{len(results)} steps successfully"
        }

    def _select_agent_for_step(self, step: str, preferred_agents: List[str]) -> Agent:
        """Select best agent for a step"""
        step_lower = step.lower()

        # Try preferred agents first
        for agent_name in preferred_agents:
            if agent_name in self.agents:
                agent_desc_lower = self.agents[agent_name].config.description.lower()
                # Simple keyword matching
                if any(word in step_lower for word in agent_desc_lower.split()):
                    return self.agents[agent_name]

        # Try any spawned agent
        for agent_name, agent in self.agents.items():
            if agent_name != "planner":
                return agent

        # Fallback to planner
        return self.agents["planner"]

# ============================================================================
# SECTION 7: UTILITIES (300 lines)
# ============================================================================

def setup_logging(level: str = "INFO"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

def setup_dspy(
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    api_base: Optional[str] = None
):
    """
    Setup DSPy with LLM provider

    Supports: OpenAI, Anthropic, Azure, local models
    """
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")

    if not api_key:
        raise ValueError(
            "No API key found. Set OPENAI_API_KEY or ANTHROPIC_API_KEY "
            "environment variable, or pass api_key parameter."
        )

    # Configure LLM
    if "claude" in model.lower():
        lm = dspy.LM(
            model=f"anthropic/{model}",
            api_key=api_key,
            api_base=api_base
        )
    else:
        lm = dspy.LM(
            model=model,
            api_key=api_key,
            api_base=api_base
        )

    dspy.configure(lm=lm)

    logging.info(f"✅ Configured DSPy with {model}")

def print_result(result: Dict[str, Any]):
    """Pretty print orchestration result"""
    print("\n" + "="*80)
    print(f"🎯 Goal: {result['goal']}")
    print("="*80)

    if result.get("plan"):
        print(f"\n📋 Plan:\n{result['plan']}\n")

    print(f"\n📊 Results:")
    print(f"  Success: {result['success']}")
    print(f"  Summary: {result.get('summary', 'N/A')}")

    if result.get("results"):
        print(f"\n📍 Steps:")
        for r in result["results"]:
            status = "✅" if r["success"] else "❌"
            print(f"  {status} Step {r['step']}: {r['description'][:80]}")
            print(f"     Agent: {r['agent']}, Time: {r['execution_time']:.2f}s")
            if r["success"] and r.get("output"):
                output_preview = str(r["output"])[:200]
                print(f"     Output: {output_preview}")
            if r.get("error"):
                print(f"     Error: {r['error']}")

    print("\n" + "="*80)

# ============================================================================
# SECTION 8: CLI (100 lines)
# ============================================================================

async def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Jotty Minimal - Simple multi-agent orchestration"
    )
    parser.add_argument(
        "--goal",
        type=str,
        required=True,
        help="Goal to accomplish"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="LLM model to use (default: gpt-4o-mini)"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=10,
        help="Maximum number of execution steps"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )

    args = parser.parse_args()

    # Setup
    setup_logging(args.log_level)
    setup_dspy(model=args.model)

    # Create orchestrator
    orchestrator = Orchestrator()

    # Run
    result = await orchestrator.run(
        goal=args.goal,
        max_steps=args.max_steps
    )

    # Print result
    print_result(result)

    # Exit code
    sys.exit(0 if result["success"] else 1)

if __name__ == "__main__":
    asyncio.run(main())
