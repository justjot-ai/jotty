# Memory Identification Integration Plan

## Current HierarchicalMemory Key Generation

**Current code** (`core/memory/cortex.py`):
```python
def store(self, content: str, level: MemoryLevel, context: Dict[str, Any], goal: str, ...):
    # Current: timestamp-based key (problematic!)
    key = hashlib.md5(f"{content}{datetime.now().isoformat()}".encode()).hexdigest()
    
    entry = MemoryEntry(
        key=key,
        content=content,
        level=level,
        context=context,  # domain/task_type could be in context!
        ...
    )
```

**Problem**: Same content at different times = different keys (no deduplication!)

---

## Integration Plan: Add Domain/Task_Type to Key Generation

### Step 1: Update `HierarchicalMemory.store()` Method

**Add optional parameters** for domain/task_type:

```python
def store(self,
          content: str,
          level: MemoryLevel,
          context: Dict[str, Any],
          goal: str,
          domain: Optional[str] = None,  # NEW
          task_type: Optional[str] = None,  # NEW
          initial_value: float = 0.5,
          causal_links: List[str] = None) -> MemoryEntry:
    """
    Store a new memory with hierarchical key generation.
    
    Args:
        domain: Domain identifier (e.g., 'sql', 'mermaid', 'plantuml')
        task_type: Task type (e.g., 'date_filter', 'sequence_diagram')
        context: Context dict (domain/task_type can also come from here)
    """
    # Extract domain/task_type from context if not provided
    if domain is None:
        domain = context.get('domain', 'general')
    if task_type is None:
        task_type = context.get('task_type', context.get('operation_type', 'general'))
    
    # Generate hierarchical key: {domain}:{task_type}:{content_hash}
    content_hash = hashlib.md5(content.encode()).hexdigest()[:16]
    key = f"{domain}:{task_type}:{content_hash}"
    
    # Check if key already exists (deduplication)
    if level in self.memories and key in self.memories[level]:
        # Update existing memory instead of creating duplicate
        existing = self.memories[level][key]
        existing.access_count += 1
        existing.last_accessed = datetime.now()
        return existing
    
    # Create entry (rest of existing code)
    entry = MemoryEntry(
        key=key,
        content=content,
        level=level,
        context=context,
        ...
    )
    
    # Store domain/task_type in metadata for easy filtering
    entry.metadata = {
        **entry.metadata,
        'domain': domain,
        'task_type': task_type
    }
    
    # Store
    self.memories[level][key] = entry
    return entry
```

### Step 2: Add Retrieval Methods by Domain/Task_Type

**Add helper methods** for filtering:

```python
def retrieve_by_domain(
    self,
    domain: str,
    goal: str,
    budget_tokens: int,
    levels: List[MemoryLevel] = None
) -> List[MemoryEntry]:
    """Retrieve memories filtered by domain."""
    if levels is None:
        levels = list(MemoryLevel)
    
    # Collect memories from specified levels
    domain_memories = []
    for level in levels:
        if level in self.memories:
            # Filter by domain prefix
            for key, memory in self.memories[level].items():
                if key.startswith(f'{domain}:'):
                    domain_memories.append(memory)
    
    # Use existing retriever for ranking
    selected = self.retriever.retrieve(
        query=f"Domain: {domain}",
        goal=goal,
        memories=domain_memories,
        budget_tokens=budget_tokens
    )
    
    return selected

def retrieve_by_task_type(
    self,
    task_type: str,
    goal: str,
    budget_tokens: int,
    levels: List[MemoryLevel] = None
) -> List[MemoryEntry]:
    """Retrieve memories filtered by task type."""
    if levels is None:
        levels = list(MemoryLevel)
    
    # Collect memories from specified levels
    task_memories = []
    for level in levels:
        if level in self.memories:
            # Filter by task type (second part of key)
            for key, memory in self.memories[level].items():
                if f':{task_type}:' in key:
                    task_memories.append(memory)
    
    # Use existing retriever for ranking
    selected = self.retriever.retrieve(
        query=f"Task type: {task_type}",
        goal=goal,
        memories=task_memories,
        budget_tokens=budget_tokens
    )
    
    return selected
```

### Step 3: Update Conductor to Pass Domain/Task_Type

**Update Conductor** to extract and pass domain/task_type:

```python
# In conductor.py, when storing memories:

# Extract domain from agent/context
domain = None
task_type = None

# Try to infer from agent name
if 'sql' in actor_config.name.lower() or 'query' in actor_config.name.lower():
    domain = 'sql'
elif 'mermaid' in actor_config.name.lower():
    domain = 'mermaid'
elif 'plantuml' in actor_config.name.lower():
    domain = 'plantuml'
elif 'latex' in actor_config.name.lower():
    domain = 'latex'

# Try to infer task type from task description
if 'filter' in task.description.lower() or 'where' in task.description.lower():
    task_type = 'filter'
elif 'sequence' in task.description.lower():
    task_type = 'sequence_diagram'
elif 'class' in task.description.lower():
    task_type = 'class_diagram'
# ... etc

# Store with domain/task_type
self.shared_memory.store(
    content=content,
    level=MemoryLevel.EPISODIC,
    context=context,
    goal=goal,
    domain=domain,  # NEW
    task_type=task_type  # NEW
)
```

### Step 4: Backward Compatibility

**Ensure existing code still works**:

```python
def store(self, ..., domain: Optional[str] = None, task_type: Optional[str] = None, ...):
    # If domain/task_type not provided, use 'general'
    if domain is None:
        domain = context.get('domain', 'general')
    if task_type is None:
        task_type = context.get('task_type', 'general')
    
    # Generate key
    key = f"{domain}:{task_type}:{content_hash}"
    
    # Existing code continues to work!
    # Old memories without domain/task_type will have 'general:general:hash'
    # New memories will have proper domain/task_type
```

---

## Integration Benefits

### ✅ Backward Compatible

- Existing code continues to work
- Old memories: `general:general:hash` (fallback)
- New memories: `sql:date_filter:hash` (proper)

### ✅ Better Deduplication

**Before**:
```python
# Same content, different times = different keys
key1 = hash("Use partition column" + "2024-01-01T10:00:00")  # abc123
key2 = hash("Use partition column" + "2024-01-01T11:00:00")  # def456
# ❌ Duplicate memories!
```

**After**:
```python
# Same content = same key (regardless of time)
key1 = "sql:date_filter:" + hash("Use partition column")  # sql:date_filter:abc123
key2 = "sql:date_filter:" + hash("Use partition column")  # sql:date_filter:abc123
# ✅ Same key = deduplication works!
```

### ✅ Domain/Task Filtering

**Before**: Had to retrieve all memories, filter in Python
```python
all_memories = memory.retrieve(query, goal, budget)
sql_memories = [m for m in all_memories if 'sql' in m.content]  # Slow!
```

**After**: Filter at key level (fast!)
```python
sql_memories = memory.retrieve_by_domain('sql', goal, budget)  # Fast!
```

### ✅ RL Layer Integration

**RL layer can now use domain/task_type**:

```python
# RL layer identifies memories by domain + task_type
learner.record_access(
    memory=hierarchical_memory,
    domain='sql',
    task_type='date_filter',
    content='Use partition column',
    step_reward=0.1
)

# RL updates go directly to HierarchicalMemory
learner.end_episode(
    memory=hierarchical_memory,
    domain='sql',
    goal=goal,
    final_reward=1.0
)
```

---

## Implementation Checklist

- [ ] Update `HierarchicalMemory.store()` to accept `domain`/`task_type`
- [ ] Update key generation to use hierarchical format
- [ ] Add `retrieve_by_domain()` method
- [ ] Add `retrieve_by_task_type()` method
- [ ] Update Conductor to extract/pass domain/task_type
- [ ] Update RL layer to use domain/task_type
- [ ] Ensure backward compatibility (fallback to 'general')
- [ ] Test deduplication works correctly
- [ ] Test filtering by domain/task_type

---

## Example: Before vs After

### Before (Current):

```python
# Store memory
memory.store(
    content="Use partition column for date filters",
    level=MemoryLevel.SEMANTIC,
    context={"agent": "SQLGenerator"},
    goal="optimize_query"
)
# Key: hash("Use partition column..." + timestamp) = abc123def456...

# Retrieve (no domain filtering)
memories = memory.retrieve("date filter query", goal="optimize_query", budget=5000)
# Gets ALL memories, then filters by content (slow)
```

### After (With Integration):

```python
# Store memory with domain/task_type
memory.store(
    content="Use partition column for date filters",
    level=MemoryLevel.SEMANTIC,
    context={"agent": "SQLGenerator"},
    goal="optimize_query",
    domain="sql",  # NEW
    task_type="date_filter"  # NEW
)
# Key: "sql:date_filter:abc123..." (deduplication works!)

# Retrieve by domain (fast filtering)
sql_memories = memory.retrieve_by_domain("sql", goal="optimize_query", budget=5000)
# Only SQL memories, filtered at key level (fast!)
```

---

## Summary

**✅ Option 1 (Hierarchical Keys) CAN integrate with existing HierarchicalMemory!**

**Changes needed**:
1. Update `store()` to accept `domain`/`task_type` parameters
2. Change key generation from timestamp-based to hierarchical
3. Add retrieval methods for domain/task_type filtering
4. Update Conductor to pass domain/task_type
5. Ensure backward compatibility

**Benefits**:
- ✅ Better deduplication
- ✅ Faster filtering
- ✅ RL layer integration
- ✅ Backward compatible

**Result**: Unified memory system with proper identification! 🎯
