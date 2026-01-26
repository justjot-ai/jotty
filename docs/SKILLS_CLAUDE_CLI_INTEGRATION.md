# Skills Integration with Claude CLI LM

## Current Status

✅ **Skills Registry**: Jotty's skills registry successfully loads the `last30days` skill from `~/.claude/skills/`

✅ **Skill Detection**: The skill is detected and registered as a callable tool

⚠️ **Claude CLI Integration**: Claude CLI LM wrapper doesn't automatically expose skills to Claude CLI

## How It Works

### 1. Skill Loading

Jotty's `SkillsRegistry` automatically:
- Checks `~/jotty/skills/` for standard Python skills (`tools.py`)
- Checks `~/.claude/skills/` for Claude Code Python skills (`scripts/*.py`)
- Loads `last30days` skill from `~/.claude/skills/last30days/`
- Creates wrapper functions that execute the Python scripts

### 2. Skill Execution

The `last30days` skill is registered as a callable function:

```python
from core.registry.skills_registry import get_skills_registry

registry = get_skills_registry()
registry.init()

# Get the skill
skill = registry.get_skill('last30days')
tool_func = skill.tools['last30days']

# Execute it
result = await tool_func({
    'topic': 'prompting techniques for ChatGPT',
    'tool': 'ChatGPT'
})
```

### 3. Claude CLI Integration

**Current Limitation**: Claude CLI doesn't automatically discover skills from `~/.claude/skills/` when called via `--print` mode.

**Solution Options**:

#### Option A: Direct Tool Execution (Current)
Skills are available as Python functions that can be called directly:

```python
from core.foundation.claude_cli_lm import ClaudeCLILM
from core.registry.skills_registry import get_skills_registry

lm = ClaudeCLILM(model="sonnet")
registry = get_skills_registry()
registry.init()

# Get skill tool
last30days_tool = registry.get_skill('last30days').tools['last30days']

# Use in agent workflow
result = await last30days_tool({'topic': 'AI trends'})
```

#### Option B: Function Calling Integration (Future)
Expose skills as function calling tools to Claude CLI:

```python
# Convert skills to Claude CLI function calling format
skills = registry.list_skills()
tools_schema = []
for skill in skills:
    for tool_name, tool_func in skill.tools.items():
        tools_schema.append({
            'name': tool_name,
            'description': skill.description,
            'parameters': {...}  # Extract from tool_func
        })

# Pass to Claude CLI via --tools parameter (if supported)
```

## Testing

To verify the `last30days` skill is loaded:

```python
from core.registry.skills_registry import get_skills_registry

registry = get_skills_registry()
registry.init()

# List all skills
skills = registry.list_skills()
print(f"Loaded skills: {[s['name'] for s in skills]}")

# Check last30days
skill = registry.get_skill('last30days')
if skill:
    print(f"✓ last30days skill found")
    print(f"  Description: {skill.description}")
    print(f"  Tools: {list(skill.tools.keys())}")
    print(f"  Path: {skill.metadata['path']}")
```

## Usage in Agents

Agents can use skills directly:

```python
from core.registry.skills_registry import get_skills_registry

class MyAgent:
    def __init__(self):
        self.registry = get_skills_registry()
        self.registry.init()
    
    async def research_topic(self, topic: str):
        skill = self.registry.get_skill('last30days')
        if skill:
            tool = skill.tools['last30days']
            result = await tool({
                'topic': topic,
                'deep': True,
                'sources': 'both'
            })
            return result
```

## Next Steps

1. ✅ Skills registry loads Claude Code skills
2. ✅ Skills are registered as callable tools
3. ⚠️ Claude CLI LM wrapper needs to expose skills (optional enhancement)
4. ✅ Agents can use skills directly via registry

**Current State**: Skills work in Jotty agents, but aren't automatically exposed to Claude CLI's function calling. This is fine because:
- Agents can call skills directly via the registry
- Skills are Python functions, not Claude CLI commands
- The registry provides a clean API for skill access
