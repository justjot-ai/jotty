# Tool Integration Solution - Hybrid Workflow with Existing Tools

## Problem Identified

✅ **Jotty ALREADY HAS**:
- `MetadataToolRegistry` - Discovers and registers all tools
- `ToolManager` - Creates DSPy tools with smart parameter resolution
- `AgenticToolSelector` - LLM-based tool selection (no hardcoding)
- `CapabilityIndex` - Maps outputs to producers
- `Conductor` - Orchestrates actors with tools

❌ **What I created** (`run_hybrid_minimal_task.py`):
- Bypassed entire tool system
- Used raw DSPy agents without tools
- Agents only had text input/output
- No file write, no code execution, no git operations

---

## Solution: Integrate Tools into Hybrid Workflow

### Option 1: Extend Conductor with Hybrid Mode (RECOMMENDED)

Add hybrid workflow support to existing Conductor:

```python
# core/orchestration/conductor.py

async def run_hybrid(
    self,
    goal: str,
    data_location: str,
    discovery_agents: int = 3,
    delivery_agents: int = 3,
    **kwargs
) -> SwarmResult:
    """
    Run hybrid workflow: P2P Discovery + Sequential Delivery.

    ALL AGENTS GET TOOLS from MetadataToolRegistry!

    Args:
        goal: What to build
        data_location: Where data is
        discovery_agents: Number of P2P discovery agents
        delivery_agents: Number of sequential delivery agents
    """

    # Get all tools from registry (ALREADY EXISTS!)
    all_tools = self._get_auto_discovered_dspy_tools()

    # PHASE 1: P2P Discovery (parallel)
    discovery_actors = []
    for i in range(discovery_agents):
        actor_config = AgentConfig(
            name=f"Discovery Agent {i+1}",
            description=f"Explore data and discover requirements",
            tools=all_tools,  # ← GIVE THEM TOOLS!
            module=DiscoveryModule
        )
        discovery_actors.append(actor_config)

    # Run P2P (parallel)
    discovery_results = await self._run_p2p_phase(
        actors=discovery_actors,
        goal=goal,
        shared_context=self.shared_context,
        scratchpad=self.shared_scratchpad
    )

    # PHASE 2: Sequential Delivery (ordered)
    delivery_actors = []
    for i in range(delivery_agents):
        actor_config = AgentConfig(
            name=f"Delivery Agent {i+1}",
            description=f"Build system using discoveries",
            tools=all_tools,  # ← GIVE THEM TOOLS!
            module=DeliveryModule
        )
        delivery_actors.append(actor_config)

    # Run Sequential (ordered)
    delivery_results = await self._run_sequential_phase(
        actors=delivery_actors,
        goal=goal,
        discoveries=discovery_results,
        shared_context=self.shared_context,
        scratchpad=self.shared_scratchpad
    )

    return SwarmResult(...)
```

**Benefits**:
- Uses existing tool infrastructure
- ALL agents get file write, execution, git tools
- Agentic tool selection (LLM decides which tools to use)
- No code duplication

---

### Option 2: Create ToolAwareHybridWorkflow (Wrapper)

Create a new workflow class that wraps Conductor:

```python
# templates/tool_aware_hybrid_workflow.py

class ToolAwareHybridWorkflow:
    """Hybrid workflow with full tool access."""

    def __init__(self, metadata_provider):
        # Create Conductor to get tools
        self.conductor = Conductor(actors=[], config=JottyConfig())

        # Get all tools
        self.tools = self.conductor._get_auto_discovered_dspy_tools()

        # Get tool selector
        from core.metadata.tool_shed import AgenticToolSelector
        self.tool_selector = AgenticToolSelector()

    async def run(self, goal: str, data_location: str):
        # PHASE 1: P2P Discovery (with tools)
        async def run_discovery_with_tools(agent_num: int):
            # Agent decides which tools it needs
            selected_tools = self.tool_selector.select_tools(
                task=f"Explore {data_location} for {goal}",
                required_output="data analysis findings",
                available_tools=self.tools,
                current_context={}
            )

            # Agent has access to selected tools
            agent = dspy.ChainOfThought(DiscoveryAgent)
            result = agent(
                goal=goal,
                data_location=data_location,
                available_tools=selected_tools  # ← TOOLS PROVIDED!
            )

            return result

        # Run discoveries in parallel
        discoveries = await asyncio.gather(*[
            run_discovery_with_tools(i)
            for i in range(3)
        ])

        # PHASE 2: Sequential Delivery (with tools)
        deliverables = []
        for i in range(3):
            # Agent decides which tools it needs
            selected_tools = self.tool_selector.select_tools(
                task=f"Build system for {goal}",
                required_output="working code",
                available_tools=self.tools,
                current_context={'discoveries': discoveries}
            )

            # Agent has access to selected tools
            agent = dspy.ChainOfThought(DeliveryAgent)
            result = agent(
                goal=goal,
                discoveries=discoveries,
                available_tools=selected_tools  # ← TOOLS PROVIDED!
            )

            deliverables.append(result)

        return deliverables
```

---

### Option 3: Minimal - Just Add File Write Tool

Quick fix for current hybrid workflow:

```python
# run_hybrid_minimal_task.py

class FileWriteTool:
    """Minimal file write tool for agents."""

    @staticmethod
    def write_file(path: str, content: str) -> bool:
        from pathlib import Path
        try:
            Path(path).write_text(content)
            return True
        except Exception as e:
            logger.error(f"Failed to write {path}: {e}")
            return False

    @staticmethod
    def read_file(path: str) -> str:
        from pathlib import Path
        return Path(path).read_text()

# Modify agent signatures to include tools
class DeliveryAgent(dspy.Signature):
    goal: str = dspy.InputField()
    discoveries: str = dspy.InputField()
    available_tools: str = dspy.InputField(desc="read_file(path), write_file(path, content)")

    # Agent must call tools via structured output
    tool_calls: List[dict] = dspy.OutputField(desc="[{'tool': 'write_file', 'args': {'path': '...', 'content': '...'}}]")
    deliverable: str = dspy.OutputField()

# Execute tool calls
for tool_call in result.tool_calls:
    if tool_call['tool'] == 'write_file':
        FileWriteTool.write_file(**tool_call['args'])
```

---

## Recommended Approach

**Option 1 (Extend Conductor)** is best because:

1. ✅ Uses existing tool infrastructure (no duplication)
2. ✅ ALL tools available (file, execution, git, data, etc.)
3. ✅ Agentic tool selection (LLM chooses tools)
4. ✅ Tool caching, error handling already built
5. ✅ Consistent with rest of Jotty architecture

---

## What Tools Agents Would Get

From existing MetadataToolRegistry, agents would automatically get:

### File Operations
- `read_file(path)` → str
- `write_file(path, content)` → bool
- `list_files(pattern)` → List[str]
- `find_in_files(pattern, content)` → List[Match]

### Data Operations
- `load_dataframe(path)` → DataFrame
- `query_data(df, query)` → DataFrame
- `save_csv(df, path)` → bool

### Code Execution
- `execute_python(code)` → {'stdout', 'stderr', 'exit_code'}
- `run_tests(path)` → {'passed', 'failed', 'errors'}

### Git Operations
- `git_status()` → {'modified', 'untracked'}
- `git_commit(message)` → str (commit hash)
- `git_push()` → bool

### Metadata (if metadata provider configured)
- `get_tables()` → List[str]
- `get_schema(table)` → dict
- `query_metadata(sql)` → DataFrame

---

## Next Steps

1. **Add `run_hybrid()` method to Conductor**
2. **Test with stock screener task**
3. **Verify agents can write files, execute code, commit changes**
4. **Document tool usage patterns for agents**

This makes Jotty truly LM-agnostic with full tool support! 🚀
