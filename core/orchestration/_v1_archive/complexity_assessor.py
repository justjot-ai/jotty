"""
LLM-Based Complexity Assessment for Dynamic Agent Spawning
===========================================================

Approach 2: Intelligent decision-making for when to spawn new agents.

Uses LLM to assess task complexity and determine if spawning is needed.
More sophisticated than MegaAgent's implicit prompt-based approach.

Key Features:
- LLM-based complexity scoring
- Explicit reasoning about spawning decisions
- Task decomposition recommendations
- Integration with DynamicAgentSpawner

Dr. Agarwal: "Complexity assessment is key to efficient agent allocation"
Dr. Manning: "Let the LLM decide when to divide and conquer"
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

try:
    import dspy
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False

logger = logging.getLogger(__name__)


class ComplexityLevel(Enum):
    """Task complexity levels"""
    TRIVIAL = 1      # Single agent, single step
    SIMPLE = 2       # Single agent, multiple steps
    MODERATE = 3     # Multiple agents OR complex logic
    COMPLEX = 4      # Multiple agents AND complex coordination
    VERY_COMPLEX = 5 # Requires hierarchical spawning


@dataclass
class AssessmentResult:
    """Result of complexity assessment"""
    task: str
    complexity_level: ComplexityLevel
    should_spawn: bool
    reasoning: str
    recommended_agents: List[Dict[str, str]]  # [{"name": "...", "description": "...", "signature": "..."}]
    confidence: float  # 0-1


# =============================================================================
# DSPy SIGNATURES FOR COMPLEXITY ASSESSMENT
# =============================================================================

if DSPY_AVAILABLE:
    class ComplexityAssessorSignature(dspy.Signature):
        """
        LLM-based task complexity assessment

        Analyzes a task and determines if spawning new agents is beneficial.
        """
        task: str = dspy.InputField(
            desc="Task description to assess for complexity"
        )
        existing_agents: str = dspy.InputField(
            desc="List of currently available agents and their capabilities. "
            "Format: 'agent_name: description, agent2: description, ...'"
        )
        current_progress: str = dspy.InputField(
            desc="What has been accomplished so far. "
            "Empty if starting a new task."
        )

        complexity_score: int = dspy.OutputField(
            desc="Complexity score from 1-5:\n"
            "1 = Trivial (single agent, single step)\n"
            "2 = Simple (single agent, multiple steps)\n"
            "3 = Moderate (multiple agents OR complex logic)\n"
            "4 = Complex (multiple agents AND coordination)\n"
            "5 = Very Complex (hierarchical spawning needed)"
        )
        should_spawn: str = dspy.OutputField(
            desc="'yes' if spawning new agents would help, 'no' if existing agents sufficient. "
            "Only say 'yes' if task truly benefits from parallel execution or specialization."
        )
        reasoning: str = dspy.OutputField(
            desc="Detailed reasoning about the complexity assessment. "
            "Explain WHY spawning is/isn't needed. "
            "Consider: task divisibility, parallelization opportunities, specialization needs."
        )

    class AgentRecommenderSignature(dspy.Signature):
        """
        Recommend specific agents to spawn for a complex task

        Only called if ComplexityAssessor determined spawning is needed.
        """
        task: str = dspy.InputField(
            desc="Complex task that needs decomposition"
        )
        existing_agents: str = dspy.InputField(
            desc="Currently available agents"
        )
        available_signatures: str = dspy.InputField(
            desc="Available DSPy signatures for spawning. "
            "Format: 'SignatureName: description, SignatureName2: description, ...'"
        )

        recommended_agents: str = dspy.OutputField(
            desc="List of agents to spawn, one per line. "
            "Format: 'agent_name|signature_name|description'\n"
            "Example:\n"
            "researcher_1|ResearcherSignature|Research introduction section\n"
            "researcher_2|ResearcherSignature|Research methodology section\n"
            "writer_1|ContentWriterSignature|Write introduction from research"
        )


# =============================================================================
# COMPLEXITY ASSESSOR
# =============================================================================

class ComplexityAssessor:
    """
    LLM-based complexity assessment for dynamic spawning decisions

    Approach 2: Explicit LLM-based decision making

    Usage:
        assessor = ComplexityAssessor()

        result = assessor.assess_task(
            task="Write 15-section comprehensive guide",
            existing_agents=["planner", "researcher"],
            current_progress="Planning complete, 0/15 sections written"
        )

        if result.should_spawn:
            for agent_spec in result.recommended_agents:
                spawner.spawn_agent(**agent_spec)

    Features:
    - LLM-based complexity scoring (1-5)
    - Explicit reasoning about spawning decisions
    - Recommendations for specific agents to spawn
    - Signature-aware recommendations
    """

    def __init__(self, signature_registry: Optional[Dict[str, type]] = None):
        """
        Initialize complexity assessor

        Args:
            signature_registry: Dict mapping signature names to classes
                                (e.g., {"ResearcherSignature": ResearcherSignature})
        """
        if not DSPY_AVAILABLE:
            raise RuntimeError("DSPy required for ComplexityAssessor")

        self.signature_registry = signature_registry or {}

        # Initialize DSPy modules
        self.complexity_module = dspy.ChainOfThought(ComplexityAssessorSignature)
        self.recommender_module = dspy.ChainOfThought(AgentRecommenderSignature)

        logger.info(f"🧠 ComplexityAssessor initialized with {len(self.signature_registry)} signatures")

    def assess_task(
        self,
        task: str,
        existing_agents: List[str],
        current_progress: str = "",
        existing_agent_descriptions: Optional[Dict[str, str]] = None
    ) -> AssessmentResult:
        """
        Assess task complexity and determine if spawning is needed

        Args:
            task: Task description
            existing_agents: List of available agent names
            current_progress: What's been accomplished so far
            existing_agent_descriptions: Optional descriptions of existing agents

        Returns:
            AssessmentResult with spawning recommendation
        """
        logger.info(f"🔍 Assessing task complexity: {task[:100]}...")

        # Format existing agents
        if existing_agent_descriptions:
            agents_str = ", ".join([
                f"{name}: {desc}" for name, desc in existing_agent_descriptions.items()
            ])
        else:
            agents_str = ", ".join(existing_agents)

        # Run complexity assessment
        complexity_result = self.complexity_module(
            task=task,
            existing_agents=agents_str,
            current_progress=current_progress or "Starting new task"
        )

        # Parse results
        complexity_score = int(complexity_result.complexity_score)
        should_spawn_str = complexity_result.should_spawn.lower().strip()
        should_spawn = should_spawn_str in ["yes", "true", "1"]

        complexity_level = ComplexityLevel(complexity_score)

        logger.info(f"   Complexity: {complexity_level.name} (score: {complexity_score})")
        logger.info(f"   Should spawn: {should_spawn}")
        logger.info(f"   Reasoning: {complexity_result.reasoning[:200]}...")

        # Get agent recommendations if spawning needed
        recommended_agents = []
        if should_spawn and self.signature_registry:
            recommended_agents = self._get_recommendations(
                task=task,
                existing_agents=agents_str
            )

        return AssessmentResult(
            task=task,
            complexity_level=complexity_level,
            should_spawn=should_spawn,
            reasoning=complexity_result.reasoning,
            recommended_agents=recommended_agents,
            confidence=0.8  # Could be enhanced with actual confidence scoring
        )

    def _get_recommendations(
        self,
        task: str,
        existing_agents: str
    ) -> List[Dict[str, str]]:
        """
        Get specific agent recommendations for spawning

        Args:
            task: Task description
            existing_agents: Formatted existing agents string

        Returns:
            List of agent specifications to spawn
        """
        # Format available signatures
        signatures_str = ", ".join([
            f"{name}: {sig.__doc__ or 'DSPy signature'}"
            for name, sig in self.signature_registry.items()
        ])

        logger.info(f"   Getting agent recommendations...")

        # Run recommender
        recommendations = self.recommender_module(
            task=task,
            existing_agents=existing_agents,
            available_signatures=signatures_str
        )

        # Parse recommendations
        agents = []
        for line in recommendations.recommended_agents.strip().split('\n'):
            line = line.strip()
            if not line or '|' not in line:
                continue

            parts = line.split('|')
            if len(parts) >= 3:
                agent_name = parts[0].strip()
                signature_name = parts[1].strip()
                description = parts[2].strip()

                # Validate signature exists
                if signature_name in self.signature_registry:
                    agents.append({
                        "name": agent_name,
                        "signature_name": signature_name,
                        "description": description
                    })
                else:
                    logger.warning(f"   Unknown signature '{signature_name}' recommended, skipping")

        logger.info(f"   ✅ Recommended {len(agents)} agents to spawn")
        for agent in agents:
            logger.info(f"      - {agent['name']} ({agent['signature_name']}): {agent['description'][:50]}...")

        return agents

    def register_signature(self, name: str, signature: type):
        """
        Register a DSPy signature for spawning recommendations

        Args:
            name: Signature name (e.g., "ResearcherSignature")
            signature: DSPy signature class
        """
        self.signature_registry[name] = signature
        logger.info(f"📝 Registered signature: {name}")

    def __repr__(self) -> str:
        return f"ComplexityAssessor(signatures={len(self.signature_registry)})"


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def assess_and_spawn(
    assessor: ComplexityAssessor,
    spawner: 'DynamicAgentSpawner',  # Type hint forward reference
    task: str,
    existing_agents: List[str],
    parent_agent: str,
    current_progress: str = ""
) -> Tuple[bool, List[str]]:
    """
    Assess task and automatically spawn recommended agents

    Convenience function combining assessment + spawning.

    Args:
        assessor: ComplexityAssessor instance
        spawner: DynamicAgentSpawner instance
        task: Task to assess
        existing_agents: Current available agents
        parent_agent: Agent requesting spawning
        current_progress: What's been done so far

    Returns:
        Tuple of (spawned, agent_names)
        - spawned: True if any agents were spawned
        - agent_names: List of spawned agent names
    """
    # Assess complexity
    result = assessor.assess_task(
        task=task,
        existing_agents=existing_agents,
        current_progress=current_progress
    )

    if not result.should_spawn:
        logger.info(f"❌ No spawning needed: {result.reasoning[:100]}...")
        return False, []

    # Spawn recommended agents
    spawned_names = []
    for agent_spec in result.recommended_agents:
        try:
            signature = assessor.signature_registry[agent_spec["signature_name"]]
            config = spawner.spawn_agent(
                name=agent_spec["name"],
                description=agent_spec["description"],
                signature=signature,
                spawned_by=parent_agent
            )
            spawned_names.append(agent_spec["name"])
            logger.info(f"✅ Spawned: {agent_spec['name']}")

        except Exception as e:
            logger.error(f"❌ Failed to spawn {agent_spec['name']}: {e}")

    logger.info(f"🌱 Spawned {len(spawned_names)} agents based on complexity assessment")

    return len(spawned_names) > 0, spawned_names
