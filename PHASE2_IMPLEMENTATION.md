# Phase 2: Unified BaseSwarm Implementation
## Converting ALL to Templates

**Total Conversions:** 16 swarms/templates → templates on BaseSwarm

---

## Directory Structure

```
core/intelligence/swarms/
├── base_swarm.py (ENHANCED - all 8 learning layers)
├── templates/ (NEW - all swarms become templates)
│   ├── __init__.py
│   ├── coding.py (was CodingSwarm)
│   ├── research.py (was ResearchSwarm)
│   ├── ml.py (was SwarmML)
│   ├── ml_comprehensive.py (was SwarmMLComprehensive)
│   ├── arxiv_learning.py (was ArxivLearningSwarm)
│   ├── olympiad_learning.py (was OlympiadLearningSwarm)
│   ├── perspective_learning.py (was PerspectiveLearningSwarm)
│   ├── pilot.py (was PilotSwarm)
│   ├── review.py (was ReviewSwarm)
│   ├── testing.py (was TestingSwarm)
│   ├── data_analysis.py (was DataAnalysisSwarm)
│   ├── devops.py (was DevOpsSwarm)
│   ├── fundamental.py (was FundamentalSwarm)
│   ├── idea_writer.py (was IdeaWriterSwarm)
│   ├── learning.py (was LearningSwarm)
│   └── team_patterns/
│       ├── collaborative.py
│       ├── hybrid.py
│       └── sequential.py
├── coding_swarm/ (KEPT - backward compat alias)
├── research_swarm/ (KEPT - backward compat alias)
└── ... (all kept as aliases)
```

---

## Template Format

Each template is just configuration:

```python
# core/intelligence/swarms/templates/coding.py

from ..base_swarm import BaseSwarm
from ..base.agent_team import AgentTeam, CoordinationPattern
from .agents import ArchitectAgent, DeveloperAgent, TestWriterAgent
from .types import CodingConfig, CodingResult

class CodingTemplate(BaseSwarm):
    """
    Coding swarm template - production-quality code generation.

    Template (not base class) - inherits ALL learning from BaseSwarm.
    """

    # Agent team definition
    AGENT_TEAM = AgentTeam.define(
        (ArchitectAgent, "Architect", "_architect"),
        (DeveloperAgent, "Developer", "_developer"),
        (TestWriterAgent, "TestWriter", "_test_writer"),
    )

    # Coordination pattern (AUTO learns best approach)
    COORDINATION = CoordinationPattern.AUTO

    # Template metadata
    TEMPLATE_NAME = "coding"
    TEMPLATE_VERSION = "2.0.0"
    RESULT_CLASS = CodingResult

    def __init__(self, config: CodingConfig = None):
        super().__init__(config or CodingConfig())

    # Optional: Custom execution (if not using standard patterns)
    async def _execute_domain(self, **kwargs):
        """Custom coding workflow (or use AUTO pattern)."""
        # Can implement custom logic OR rely on AUTO pattern selection
        pass

# Backward compatibility
CodingSwarm = CodingTemplate  # Alias
```

---

## Conversion Strategy

For each swarm, extract:
1. **AGENT_TEAM** → Keep as-is
2. **Custom logic** → Move to _execute_domain() or STAGES
3. **Mixins** → Move to separate utilities (if needed)
4. **Learning** → Already in BaseSwarm!
5. **Everything else** → Template config

---

## Example Conversions

### 1. Simple Template (ReviewSwarm)

```python
# BEFORE (domain swarm)
class ReviewSwarm(DomainSwarm):
    AGENT_TEAM = AgentTeam.define(
        (SecurityReviewer, "Security"),
        (PerformanceReviewer, "Performance"),
        pattern=CoordinationPattern.PARALLEL,
    )

# AFTER (template)
class ReviewTemplate(BaseSwarm):
    AGENT_TEAM = AgentTeam.define(
        (SecurityReviewer, "Security"),
        (PerformanceReviewer, "Performance"),
    )
    COORDINATION = CoordinationPattern.PARALLEL  # ← Moved here
    TEMPLATE_NAME = "review"

ReviewSwarm = ReviewTemplate  # Backward compat
```

### 2. Complex Template (CodingSwarm)

```python
# BEFORE (domain swarm with custom logic)
class CodingSwarm(DomainSwarm):
    AGENT_TEAM = AgentTeam.define(...)

    async def _execute_domain(self, **kwargs):
        # Custom multi-stage workflow
        arch = await self._architect.design()
        code = await self._developer.implement(arch)
        tests = await self._test_writer.test(code)
        return result

# AFTER (template with STAGES)
class CodingTemplate(BaseSwarm):
    AGENT_TEAM = AgentTeam.define(...)

    # Option 1: Use STAGES (declarative)
    STAGES = [
        StageConfig("design", ["_architect"]),
        StageConfig("implement", ["_developer"], needs=["design"]),
        StageConfig("test", ["_test_writer"], needs=["implement"]),
    ]
    COORDINATION = CoordinationPattern.CUSTOM

    # Option 2: Keep custom logic
    async def _execute_domain(self, **kwargs):
        arch = await self._architect.design()
        code = await self._developer.implement(arch)
        tests = await self._test_writer.test(code)
        return result

CodingSwarm = CodingTemplate
```

### 3. ML Template (was SwarmTemplate)

```python
# BEFORE (SwarmTemplate with agents dict)
class SwarmML(SwarmTemplate):
    agents = {
        "data_analyst": AgentConfig(...),
        "feature_engineer": AgentConfig(...),
    }
    pipeline = [...]

# AFTER (template on BaseSwarm)
class MLTemplate(BaseSwarm):
    AGENT_TEAM = AgentTeam.define(
        (DataAnalystAgent, "DataAnalyst"),
        (FeatureEngineerAgent, "FeatureEngineer"),
    )
    STAGES = [
        StageConfig("analyze", ["_data_analyst"]),
        StageConfig("engineer", ["_feature_engineer"], needs=["analyze"]),
    ]
    COORDINATION = CoordinationPattern.CUSTOM
    TEMPLATE_NAME = "ml"

SwarmML = MLTemplate  # Backward compat
```

---

## Backward Compatibility

Keep all old imports working:

```python
# core/intelligence/swarms/coding_swarm/__init__.py
from ..templates.coding import CodingTemplate as CodingSwarm
__all__ = ["CodingSwarm"]

# core/intelligence/orchestration/templates/swarm_ml.py
from core.intelligence.swarms.templates.ml import MLTemplate as SwarmML
__all__ = ["SwarmML"]
```

---

## Benefits

✅ **Single source of truth** - BaseSwarm has ALL learning
✅ **Templates are simple** - Just configuration
✅ **Zero learning lost** - All 8 layers in every template
✅ **Easy to create new** - Copy template, modify config
✅ **Backward compatible** - Old code works
✅ **AUTO pattern** - All templates can use it
✅ **Consistent** - Same architecture everywhere

---

## Testing Strategy

For each template, ensure:
1. ✅ Can instantiate
2. ✅ Can execute
3. ✅ All learning layers active
4. ✅ Backward compat alias works
5. ✅ Examples run
6. ✅ Results equivalent to old version

---

## Implementation Order

1. ✅ Enhance BaseSwarm (add all learning)
2. ✅ Create templates/ directory
3. ✅ Convert simple templates first (ReviewTemplate, etc.)
4. ✅ Convert complex templates (CodingTemplate, MLTemplate)
5. ✅ Set up backward compat aliases
6. ✅ Update examples
7. ✅ Test everything
8. ✅ Update documentation

---

Ready to implement!
