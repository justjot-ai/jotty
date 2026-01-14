# Memory Identification Scheme

## The Problem

**How should memories be identified in a multi-agent orchestrator?**

- ❌ Can't use agent names (multiple agents might learn same thing)
- ❌ Can't use task IDs (same task type appears multiple times)
- ❌ Need to avoid conflicts when multiple agents access same memory
- ❌ Need to support shared learning across agents

---

## Current State

### How Memory Keys Are Generated

**Current approach**: Content-based hashing
```python
key = hashlib.md5(content.encode()).hexdigest()
```

**Problem**: 
- Same content → same key (good for deduplication)
- But doesn't capture context (which agent, which task type, etc.)

### How Memories Are Stored

**HierarchicalMemory structure**:
```python
memories: Dict[MemoryLevel, Dict[str, MemoryEntry]]
# Level → {key → MemoryEntry}
```

**No entity/agent/task separation** - all memories in same dict per level!

---

## Proposed Identification Scheme

### Option 1: Hierarchical Keys (Recommended)

**Format**: `{domain}:{task_type}:{content_hash}`

**Example**:
```
sql:date_filter:abc123def456...
mermaid:sequence_diagram:xyz789...
plantuml:class_diagram:def456...
```

**Benefits**:
- ✅ Domain separation (SQL vs Mermaid vs PlantUML)
- ✅ Task type separation (date_filter vs sequence_diagram)
- ✅ Content deduplication (same content = same hash)
- ✅ Easy filtering by domain/task_type

**Structure**:
```python
memories: Dict[MemoryLevel, Dict[str, MemoryEntry]]
# Level → {domain:task_type:hash → MemoryEntry}

# Retrieve by domain
sql_memories = {k: v for k, v in memories[level].items() if k.startswith('sql:')}

# Retrieve by task type
date_filter_memories = {k: v for k, v in memories[level].items() if ':date_filter:' in k}
```

---

### Option 2: Entity-Based with Namespace

**Format**: `{entity_id}:{namespace}:{content_hash}`

**Where**:
- `entity_id`: Orchestrator ID (e.g., "conductor_1", "expert_swarm")
- `namespace`: Domain/task_type (e.g., "sql", "mermaid", "plantuml")
- `content_hash`: Content hash for deduplication

**Example**:
```
conductor_1:sql:abc123...
conductor_1:mermaid:xyz789...
expert_swarm:plantuml:def456...
```

**Benefits**:
- ✅ Entity separation (different orchestrators)
- ✅ Namespace separation (domains/task types)
- ✅ Content deduplication
- ✅ Supports multi-orchestrator scenarios

---

### Option 3: Multi-Dimensional Index

**Structure**: Separate indices by dimension

```python
# By domain
domain_index: Dict[str, Dict[str, MemoryEntry]]  # domain → {key → MemoryEntry}

# By task type
task_index: Dict[str, Dict[str, MemoryEntry]]  # task_type → {key → MemoryEntry}

# By agent (for agent-specific memories)
agent_index: Dict[str, Dict[str, MemoryEntry]]  # agent_name → {key → MemoryEntry}

# By goal
goal_index: Dict[str, Dict[str, MemoryEntry]]  # goal → {key → MemoryEntry}
```

**Benefits**:
- ✅ Multiple access patterns
- ✅ Fast filtering by any dimension
- ✅ Supports complex queries

**Drawbacks**:
- ⚠️ More complex to maintain
- ⚠️ Memory overhead (multiple indices)

---

## Recommended: Hierarchical Keys (Option 1)

### Why Hierarchical Keys?

1. **Simple**: Single key structure
2. **Flexible**: Easy to filter by prefix
3. **Deduplication**: Same content = same hash
4. **Domain-aware**: Natural separation by domain
5. **Task-aware**: Natural separation by task type

### Implementation

```python
def generate_memory_key(
    content: str,
    domain: Optional[str] = None,
    task_type: Optional[str] = None,
    agent_name: Optional[str] = None
) -> str:
    """
    Generate hierarchical memory key.
    
    Format: {domain}:{task_type}:{content_hash}
    
    If domain/task_type not provided, uses 'general'
    """
    import hashlib
    
    # Hash content for deduplication
    content_hash = hashlib.md5(content.encode()).hexdigest()[:16]
    
    # Build hierarchical key
    parts = []
    
    if domain:
        parts.append(domain)
    else:
        parts.append('general')
    
    if task_type:
        parts.append(task_type)
    else:
        parts.append('general')
    
    # Agent name is metadata, not part of key (for shared learning)
    # But can be stored in MemoryEntry.metadata
    
    parts.append(content_hash)
    
    return ':'.join(parts)

# Examples:
# "sql:date_filter:abc123def456"
# "mermaid:sequence_diagram:xyz789"
# "plantuml:class_diagram:def456"
# "general:general:hash123"  # Fallback
```

### Memory Storage Structure

```python
class HierarchicalMemory:
    # Current: Dict[MemoryLevel, Dict[str, MemoryEntry]]
    memories: Dict[MemoryLevel, Dict[str, MemoryEntry]]
    
    def store(
        self,
        content: str,
        level: MemoryLevel,
        domain: Optional[str] = None,
        task_type: Optional[str] = None,
        agent_name: Optional[str] = None,
        **kwargs
    ):
        # Generate hierarchical key
        key = generate_memory_key(content, domain, task_type)
        
        # Create MemoryEntry
        entry = MemoryEntry(
            key=key,
            content=content,
            metadata={
                'domain': domain,
                'task_type': task_type,
                'agent_name': agent_name,  # Stored as metadata
                **kwargs
            }
        )
        
        # Store in appropriate level
        if level not in self.memories:
            self.memories[level] = {}
        
        self.memories[level][key] = entry
    
    def retrieve_by_domain(
        self,
        domain: str,
        level: MemoryLevel,
        goal: Optional[str] = None,
        top_k: int = 10
    ) -> List[MemoryEntry]:
        """Retrieve memories by domain."""
        if level not in self.memories:
            return []
        
        # Filter by domain prefix
        domain_memories = {
            k: v for k, v in self.memories[level].items()
            if k.startswith(f'{domain}:')
        }
        
        # Sort by value and return top_k
        sorted_memories = sorted(
            domain_memories.values(),
            key=lambda m: m.get_value(goal) if goal else m.default_value,
            reverse=True
        )
        
        return sorted_memories[:top_k]
    
    def retrieve_by_task_type(
        self,
        task_type: str,
        level: MemoryLevel,
        goal: Optional[str] = None,
        top_k: int = 10
    ) -> List[MemoryEntry]:
        """Retrieve memories by task type."""
        if level not in self.memories:
            return []
        
        # Filter by task type (second part of key)
        task_memories = {
            k: v for k, v in self.memories[level].items()
            if f':{task_type}:' in k
        }
        
        # Sort by value and return top_k
        sorted_memories = sorted(
            task_memories.values(),
            key=lambda m: m.get_value(goal) if goal else m.default_value,
            reverse=True
        )
        
        return sorted_memories[:top_k]
```

---

## For RL Layer Integration

### How RL Should Identify Memories

**RL layer should use domain + task_type for identification:**

```python
class TDLambdaLearner:
    def record_access(
        self,
        memory: HierarchicalMemory,
        domain: str,
        task_type: str,
        content: str,
        step_reward: float = 0.0
    ):
        """
        Record memory access for RL learning.
        
        Uses hierarchical key: {domain}:{task_type}:{content_hash}
        """
        # Generate key
        key = generate_memory_key(content, domain, task_type)
        
        # Retrieve memory entry
        # Check all levels, find entry by key
        memory_entry = None
        for level in MemoryLevel:
            if level in memory.memories:
                if key in memory.memories[level]:
                    memory_entry = memory.memories[level][key]
                    break
        
        if not memory_entry:
            # Memory doesn't exist yet - create it
            memory.store(
                content=content,
                level=MemoryLevel.EPISODIC,  # Start as episodic
                domain=domain,
                task_type=task_type
            )
            memory_entry = memory.memories[MemoryLevel.EPISODIC][key]
        
        # Record access for TD(λ) learning
        trace = self._update_trace(key, step_reward)
        self.values_at_access[key] = memory_entry.get_value(self.current_goal)
        
        return trace
    
    def end_episode(
        self,
        memory: HierarchicalMemory,
        domain: str,
        goal: str,
        final_reward: float
    ):
        """
        Perform TD updates on memories in HierarchicalMemory.
        
        Updates values directly in HierarchicalMemory.
        """
        updates = []
        
        for key, trace in self.traces.items():
            # Find memory entry across all levels
            memory_entry = None
            for level in MemoryLevel:
                if level in memory.memories:
                    if key in memory.memories[level]:
                        memory_entry = memory.memories[level][key]
                        break
            
            if not memory_entry:
                continue
            
            # Perform TD update
            old_value = self.values_at_access.get(key, memory_entry.get_value(goal))
            td_error = final_reward - old_value
            new_value = old_value + self.alpha * td_error * trace
            new_value = max(0.0, min(1.0, new_value))
            
            # Update value in HierarchicalMemory
            if goal not in memory_entry.goal_values:
                memory_entry.goal_values[goal] = GoalValue()
            
            memory_entry.goal_values[goal].value = new_value
            memory_entry.goal_values[goal].last_updated = datetime.now()
            
            updates.append((key, old_value, new_value))
        
        return updates
```

---

## Summary

### Recommended Identification Scheme

**Format**: `{domain}:{task_type}:{content_hash}`

**Examples**:
- `sql:date_filter:abc123...` - SQL domain, date filter task
- `mermaid:sequence_diagram:xyz789...` - Mermaid domain, sequence diagram task
- `plantuml:class_diagram:def456...` - PlantUML domain, class diagram task

**Why**:
- ✅ Domain separation (different expert domains)
- ✅ Task type separation (different task types within domain)
- ✅ Content deduplication (same content = same hash)
- ✅ Easy filtering (`k.startswith('sql:')`)
- ✅ Supports shared learning (same memory across agents)

**Not**:
- ❌ Agent names (agents share memories)
- ❌ Task IDs (same task type appears multiple times)
- ❌ Entity IDs (orchestrator-level, not memory-level)

### For Multi-Agent Orchestrator

**Memory is shared across agents**:
- Same domain + task_type → same memory
- Multiple agents can learn from same memory
- RL updates benefit all agents using that memory

**Agent-specific info stored as metadata**:
- `MemoryEntry.metadata['agent_name']` - Which agent created/accessed it
- `MemoryEntry.metadata['access_count']` - How many agents accessed it
- But key is domain + task_type + content (shared)

---

## Next Steps

1. **Update `generate_memory_key()`** to use hierarchical format
2. **Update `HierarchicalMemory.store()`** to accept domain/task_type
3. **Update `HierarchicalMemory.retrieve()`** to support domain/task_type filtering
4. **Update RL layer** to use domain/task_type for identification
5. **Update Conductor** to pass domain/task_type when storing memories
