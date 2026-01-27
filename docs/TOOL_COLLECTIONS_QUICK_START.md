# Tool Collections - Quick Start Guide

**Quick reference for using Tool Collections in Jotty.**

---

## Installation

### Core (No Dependencies)
```bash
# ToolCollection works out of the box for local collections
# No installation needed!
```

### Hub Integration (Optional)
```bash
pip install huggingface_hub
```

### MCP Integration (Optional)
```bash
pip install mcp
# Or for easier integration:
pip install mcpadapt
```

---

## Quick Examples

### 1. Load from Local Directory

```python
from core.registry import ToolCollection, get_skills_registry

# Load collection from local skills
collection = ToolCollection.from_local("./skills")

# Load into registry
registry = get_skills_registry()
registry.init()
registry.load_collection(collection)

# Use tools
tools = registry.get_registered_tools()
print(f"Available tools: {len(tools)}")
```

### 2. Load from HuggingFace Hub

```python
from core.registry import ToolCollection, get_skills_registry

# Load collection from Hub
collection = ToolCollection.from_hub(
    collection_slug="huggingface-tools/diffusion-tools",
    trust_remote_code=True  # ⚠️ Always inspect tools first!
)

# Load into registry
registry = get_skills_registry()
registry.init()
registry.load_collection(collection, collection_name="hub_tools")
```

### 3. Load from MCP Server

```python
from core.registry import ToolCollection, get_skills_registry
from mcp import StdioServerParameters
import os

# Configure MCP server
server_params = StdioServerParameters(
    command="uv",
    args=["--quiet", "pubmedmcp@0.1.3"],
    env={"UV_PYTHON": "3.12", **os.environ}
)

# Load collection from MCP
with ToolCollection.from_mcp(server_params, trust_remote_code=True) as collection:
    # Load into registry
    registry = get_skills_registry()
    registry.init()
    registry.load_collection(collection, collection_name="mcp_tools")
```

### 4. List Collections

```python
from core.registry import get_skills_registry

registry = get_skills_registry()
registry.init()

# List all collections
collections = registry.list_collections()
for coll in collections:
    print(f"{coll['name']}: {coll['tool_count']} tools from {coll['source']}")
```

---

## API Reference

### ToolCollection

```python
class ToolCollection:
    # Load from Hub
    @classmethod
    def from_hub(cls, collection_slug: str, token: Optional[str] = None, 
                  trust_remote_code: bool = False) -> "ToolCollection"
    
    # Load from MCP (context manager)
    @classmethod
    @contextmanager
    def from_mcp(cls, server_parameters: Any, 
                  trust_remote_code: bool = False) -> ContextManager["ToolCollection"]
    
    # Load from local
    @classmethod
    def from_local(cls, collection_path: Union[str, Path]) -> "ToolCollection"
    
    # Convert to SkillDefinitions
    def to_skill_definitions(self) -> List[SkillDefinition]
    
    # Save to local
    def save_to_local(self, output_path: Union[str, Path]) -> None
    
    # List tools
    def list_tools(self) -> List[Dict[str, Any]]
```

### SkillsRegistry Integration

```python
# Load collection
registry.load_collection(collection, collection_name: Optional[str] = None)

# List collections
collections = registry.list_collections()

# Get collection
collection_info = registry.get_collection(name: str)
```

---

## Security Best Practices

1. **Always inspect tools** before loading
2. **Use trusted sources** only
3. **Set trust_remote_code=True** only after inspection
4. **Test in isolated environment** first
5. **Monitor tool execution** (use monitoring framework)

---

## Common Use Cases

### Use Case 1: Load Community Tools

```python
# Load popular tools from Hub
collection = ToolCollection.from_hub(
    "huggingface-tools/diffusion-tools",
    trust_remote_code=True
)
registry.load_collection(collection)
```

### Use Case 2: Load MCP Tools

```python
# Load tools from MCP server
with ToolCollection.from_mcp(mcp_params, trust_remote_code=True) as collection:
    registry.load_collection(collection)
```

### Use Case 3: Organize Local Tools

```python
# Group local tools into collections
collection = ToolCollection.from_local("./my_tools")
collection.save_to_local("./collections/my_tools")
```

---

## Troubleshooting

### Hub Integration Not Working
```bash
pip install huggingface_hub
```

### MCP Integration Not Working
```bash
pip install mcp
```

### Tools Not Loading
- Check `trust_remote_code=True` is set
- Verify tool format matches expected structure
- Check logs for error messages

---

**See**: `TOOL_COLLECTIONS_IMPLEMENTATION.md` for full documentation.
