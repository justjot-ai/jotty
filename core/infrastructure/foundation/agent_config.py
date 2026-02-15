"""
Agent Configuration for JOTTY.

Defines configuration for individual agents in the swarm.

# GENERIC: No domain-specific logic.

JOTTY Naming Convention:
- Architect = Pre-execution planner
- Auditor = Post-execution validator
- Agent = User's actor module
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

# A-TEAM: Import ContextRequirements
try:
    from ..utils.context_logger import ContextRequirements
except ImportError:
    ContextRequirements = None


@dataclass
class AgentConfig:
    """
    Specification for a single agent in the JOTTY swarm.
    
    JOTTY v1.0: Complete naming overhaul
    - Architect = Plans execution, assesses inputs
    - Auditor = Validates outputs, quality check
    
    NEW in v9.1: Full tool support with separate Architect/Auditor tools.
    NEW in v10.0 (A-Team v2.0): Capabilities for dynamic orchestration.
    NEW in v11.0 (A-Team FINAL): Declarative input/output specification.
    """
    name: str
    agent: Any  # DSPy Module
    architect_prompts: Optional[List[str]] = None  # Pre-execution planning prompts
    auditor_prompts: Optional[List[str]] = None    # Post-execution validation prompts
    
    # 🆕 Parameter Mappings (v11.0 - A-Team FINAL - USER CORRECTION)
    parameter_mappings: Optional[Dict[str, str]] = None
    outputs: Optional[List[str]] = None  # Output field names this agent produces
    provides: Optional[List[str]] = None  # Parameter names this agent can provide to others
    
    # A-TEAM: Context management
    context_requirements: Optional[Any] = None
    
    # Tool configuration (JOTTY v1.0)
    architect_tools: List[Any] = field(default_factory=list)  # Tools for Architect
    auditor_tools: List[Any] = field(default_factory=list)    # Tools for Auditor
    
    # Feedback routing
    feedback_rules: Optional[List[Dict[str, Any]]] = None
    capabilities: Optional[List[str]] = None
    dependencies: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    # Validation Control
    enable_architect: bool = True   # Enable Architect for this agent
    enable_auditor: bool = True     # Enable Auditor for this agent
    validation_mode: str = "standard"  # 'quick', 'standard', 'thorough'
    
    # Critical Agent Control
    is_critical: bool = False
    max_retries: int = 0  # 0 → resolved in __post_init__
    retry_strategy: str = "with_hints"
    
    # JOTTY v1.0: Executor flag (replaces hardcoded SQL checks)
    # Set to True for agents that execute actions and return execution metadata
    # (e.g., query executors, API callers, file writers)
    is_executor: bool = False
    
    enabled: bool = True
    
    def __post_init__(self) -> None:
        """Initialize with validation."""
        # Ensure prompts are lists
        if self.architect_prompts is None:
            self.architect_prompts = []
        if self.auditor_prompts is None:
            self.auditor_prompts = []
        # Resolve sentinel defaults from centralized config
        if self.max_retries <= 0:
            from Jotty.core.infrastructure.foundation.config_defaults import MAX_RETRIES
            self.max_retries = MAX_RETRIES


