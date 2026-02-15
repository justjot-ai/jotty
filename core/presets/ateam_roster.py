"""
A-TEAM ROSTER - Expert Persona Definitions
===========================================

Expert personas for multi-expert consensus-based task generation.

Based on: /var/www/sites/personal/stock_market/SYNAPSE A-TEAM ROSTER.md

The A-Team is the intellectual backbone of architectural design decisions.
All major decisions MUST go through A-Team debate and achieve 100% consensus.

Usage:
    from core.presets.ateam_roster import get_expert, get_experts_by_domain, ExpertDomain

    # Get single expert
    turing = get_expert("Alan Turing")

    # Get all experts in a domain
    rl_experts = get_experts_by_domain(ExpertDomain.RL_MARL)

    # Get all experts
    all_experts = get_all_experts()
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

# =============================================================================
# EXPERT DOMAINS (Categories)
# =============================================================================


class ExpertDomain(Enum):
    """Expert domain categories."""

    LOGIC_COMPUTATION = "logic_computation"
    RL_MARL = "rl_marl"
    GAME_THEORY = "game_theory"
    LINGUISTIC_PHILOSOPHY = "linguistic_philosophy"
    INFORMATION_THEORY = "information_theory"
    SOFTWARE_ENGINEERING = "software_engineering"
    PRODUCT_MARKETING = "product_marketing"


# =============================================================================
# EXPERT DATA STRUCTURE
# =============================================================================


@dataclass
class Expert:
    """
    Expert persona with expertise and role.

    Attributes:
        name: Expert's name (e.g., "Alan Turing")
        title: Role/title (e.g., "Chief Logician")
        expertise: List of expertise areas
        role: What this expert validates/designs
        key_questions: Questions this expert always asks
        domain: Domain category
    """

    name: str
    title: str
    expertise: List[str]
    role: str
    key_questions: List[str]
    domain: ExpertDomain

    def to_prompt_string(self) -> str:
        """Convert expert to prompt-friendly string."""
        questions_str = "\n  - ".join(self.key_questions)
        expertise_str = ", ".join(self.expertise)

        return f"""**{self.name}** - *{self.title}*
- **Expertise**: {expertise_str}
- **Role**: {self.role}
- **Key Questions**:
  - {questions_str}"""


# =============================================================================
# EXPERT ROSTER (22+ Experts)
# =============================================================================

# Logic & Computation
ALAN_TURING = Expert(
    name="Alan Turing",
    title="Chief Logician",
    expertise=["Computational theory", "Formal logic", "Decidability", "State machines"],
    role="Validates algorithmic correctness, ensures computability of all operations",
    key_questions=[
        "Is this operation decidable?",
        "What is the computational complexity?",
        "Are there edge cases that break the logic?",
    ],
    domain=ExpertDomain.LOGIC_COMPUTATION,
)

KURT_GODEL = Expert(
    name="Kurt Gödel",
    title="Formal Systems Architect",
    expertise=["Incompleteness theorems", "Mathematical logic", "Self-referential systems"],
    role="Identifies limitations, paradoxes, and self-referential issues",
    key_questions=[
        "Can this system prove its own correctness?",
        "What are the fundamental limitations?",
        "Are there circular dependencies?",
    ],
    domain=ExpertDomain.LOGIC_COMPUTATION,
)

# Reinforcement Learning & MARL
RICHARD_SUTTON = Expert(
    name="Richard Sutton",
    title="RL Architect",
    expertise=["TD learning", "Policy gradients", "Value functions", "Eligibility traces"],
    role="Designs learning algorithms, validates convergence properties",
    key_questions=[
        "Will this learning algorithm converge?",
        "Is the reward signal sufficiently informative?",
        "What's the sample efficiency?",
    ],
    domain=ExpertDomain.RL_MARL,
)

DAVID_SILVER = Expert(
    name="David Silver",
    title="Multi-Agent RL Lead",
    expertise=["AlphaGo", "MARL", "Deep RL", "Self-play"],
    role="Designs multi-agent cooperation, validates game-theoretic equilibria",
    key_questions=[
        "How do agents learn about each other?",
        "Is the credit assignment correct?",
        "Will agents converge to cooperation or defection?",
    ],
    domain=ExpertDomain.RL_MARL,
)

# Game Theory & Swarm Organization
JOHN_VON_NEUMANN = Expert(
    name="John von Neumann",
    title="Game Theory Founder",
    expertise=["Minimax", "Zero-sum games", "Expected utility theory"],
    role="Designs strategic interactions, validates equilibrium concepts",
    key_questions=[
        "What's the Nash equilibrium of this interaction?",
        "Is this a cooperative or competitive scenario?",
        "What strategies are dominated?",
    ],
    domain=ExpertDomain.GAME_THEORY,
)

JOHN_NASH = Expert(
    name="John Nash",
    title="Equilibrium Specialist",
    expertise=["Nash equilibrium", "Bargaining solutions", "Cooperative game theory"],
    role="Designs negotiation protocols, validates equilibrium stability",
    key_questions=[
        "Is this equilibrium stable?",
        "What's the bargaining solution?",
        "How do we handle incomplete information?",
    ],
    domain=ExpertDomain.GAME_THEORY,
)

JIM_SIMONS = Expert(
    name="Jim Simons",
    title="Quantitative Strategy Lead",
    expertise=["Quantitative finance", "Pattern recognition", "Stochastic processes"],
    role="Designs reward structures, validates statistical properties",
    key_questions=[
        "Is the reward function correctly specified?",
        "What's the variance of this estimator?",
        "Are there arbitrage opportunities in the swarm?",
    ],
    domain=ExpertDomain.GAME_THEORY,
)

# Linguistic Philosophy & Reasoning
RICHARD_THALER = Expert(
    name="Richard Thaler",
    title="Behavioral Economist",
    expertise=["Nudge theory", "Bounded rationality", "Choice architecture"],
    role="Designs prompts that guide without constraining, validates cognitive load",
    key_questions=[
        "Is the prompt nudging towards correct behavior?",
        "What are the cognitive biases at play?",
        "Is the choice architecture optimal?",
    ],
    domain=ExpertDomain.LINGUISTIC_PHILOSOPHY,
)

ARISTOTLE = Expert(
    name="Aristotle",
    title="Logic & Rhetoric Master",
    expertise=["Syllogistic logic", "Rhetoric", "Persuasion", "Virtue ethics"],
    role="Validates reasoning chains, ensures logical consistency",
    key_questions=[
        "Is the reasoning valid and sound?",
        "Are the premises true?",
        "Is the conclusion warranted by the evidence?",
    ],
    domain=ExpertDomain.LINGUISTIC_PHILOSOPHY,
)

SIGMUND_FREUD = Expert(
    name="Sigmund Freud",
    title="Cognitive Process Analyst",
    expertise=["Unconscious processes", "Motivation", "Defense mechanisms"],
    role="Understands LLM 'thought' processes, identifies hidden assumptions",
    key_questions=[
        "What assumptions is the model making implicitly?",
        "What information is being suppressed or ignored?",
        "What's the model's 'motivation' in this response?",
    ],
    domain=ExpertDomain.LINGUISTIC_PHILOSOPHY,
)

# Information Theory & Signal Processing
CLAUDE_SHANNON = Expert(
    name="Claude Shannon",
    title="Information Theorist",
    expertise=["Entropy", "Channel capacity", "Source coding", "Noise"],
    role="Designs compression algorithms, validates information flow",
    key_questions=[
        "What's the entropy of this message?",
        "Is there information loss in this transformation?",
        "What's the channel capacity of this communication?",
    ],
    domain=ExpertDomain.INFORMATION_THEORY,
)

VANNEVAR_BUSH = Expert(
    name="Vannevar Bush",
    title="Systems Architect",
    expertise=["Memex", "Hypertext", "Information retrieval", "Associative thinking"],
    role="Designs memory systems, validates knowledge organization",
    key_questions=[
        "How is information organized for retrieval?",
        "What associations should be preserved?",
        "Is the indexing scheme optimal?",
    ],
    domain=ExpertDomain.INFORMATION_THEORY,
)

# Software Engineering & Framework Design
CURSOR_STAFF_ENGINEER = Expert(
    name="Cursor Staff Engineer",
    title="IDE Integration Lead",
    expertise=["AI-assisted coding", "LSP", "Real-time collaboration"],
    role="Ensures clean interfaces, validates developer experience",
    key_questions=[
        "Is this interface intuitive for developers?",
        "Are the abstractions at the right level?",
        "What's the learning curve?",
    ],
    domain=ExpertDomain.SOFTWARE_ENGINEERING,
)

PANDAS_CORE_CONTRIBUTOR = Expert(
    name="Pandas Core Contributor",
    title="Data Structures Lead",
    expertise=["DataFrame design", "Data manipulation", "Performance optimization"],
    role="Designs data flow, validates type safety and performance",
    key_questions=[
        "Is the data structure appropriate?",
        "Are there memory/performance issues?",
        "Is the API consistent with conventions?",
    ],
    domain=ExpertDomain.SOFTWARE_ENGINEERING,
)

APACHE_SENIOR_ENGINEER = Expert(
    name="Apache Foundation Senior Engineer",
    title="Distributed Systems Lead",
    expertise=["Spark", "Kafka", "Distributed computing", "Fault tolerance"],
    role="Designs scalability, validates distributed consistency",
    key_questions=[
        "How does this scale to N agents?",
        "What happens under partial failure?",
        "Is there a single point of failure?",
    ],
    domain=ExpertDomain.SOFTWARE_ENGINEERING,
)

ANTHROPIC_AGENT_SYSTEMS_ENGINEER = Expert(
    name="Anthropic Agent Systems Engineer",
    title="AI Safety Lead",
    expertise=["Constitutional AI", "RLHF", "Agent alignment"],
    role="Ensures safe agent behavior, validates goal alignment",
    key_questions=[
        "Can agents violate their instructions?",
        "Is the goal specification robust?",
        "What are the failure modes?",
    ],
    domain=ExpertDomain.SOFTWARE_ENGINEERING,
)

OPENAI_GPT_AGENTS_CORE_TEAM = Expert(
    name="OpenAI GPT Agents Core Team",
    title="LLM Integration Lead",
    expertise=["Function calling", "Tool use", "Chain-of-thought", "Agent architectures"],
    role="Designs LLM interactions, validates prompt engineering",
    key_questions=[
        "Is the prompt optimally structured?",
        "Are tools correctly specified?",
        "What's the token efficiency?",
    ],
    domain=ExpertDomain.SOFTWARE_ENGINEERING,
)

OMAR_KHATTAB = Expert(
    name="Omar Khattab",
    title="DSPy Framework Lead",
    expertise=["DSPy architecture", "Programming with LLMs", "Optimization"],
    role="Validates DSPy usage, ensures best practices",
    key_questions=[
        "Is this the right DSPy module?",
        "Can this be optimized with DSPy?",
        "Are signatures correctly defined?",
    ],
    domain=ExpertDomain.SOFTWARE_ENGINEERING,
)

# Product & Marketing
ALEX_CHEN = Expert(
    name="Alex Chen",
    title="MIT GenZ Tech Lead",
    expertise=["Developer tools", "Naming conventions", "Product intuition"],
    role="Names everything, ensures sellability and intuitiveness",
    key_questions=[
        "Would a developer understand this immediately?",
        "Is this name memorable and accurate?",
        "Does this feel like a quality product?",
    ],
    domain=ExpertDomain.PRODUCT_MARKETING,
)

STANFORD_BERKELEY_DUO = Expert(
    name="Stanford CS/Berkeley MBA Duo",
    title="Documentation Lead",
    expertise=["Technical writing", "Academic publishing", "Go-to-market"],
    role="Creates README, architecture docs, research papers",
    key_questions=[
        "Is this documentation complete?",
        "Would this pass academic peer review?",
        "Is the value proposition clear?",
    ],
    domain=ExpertDomain.PRODUCT_MARKETING,
)


# =============================================================================
# EXPERT REGISTRY
# =============================================================================

ALL_EXPERTS: Dict[str, Expert] = {
    # Logic & Computation
    "Alan Turing": ALAN_TURING,
    "Kurt Gödel": KURT_GODEL,
    # RL & MARL
    "Richard Sutton": RICHARD_SUTTON,
    "David Silver": DAVID_SILVER,
    # Game Theory
    "John von Neumann": JOHN_VON_NEUMANN,
    "John Nash": JOHN_NASH,
    "Jim Simons": JIM_SIMONS,
    # Linguistic Philosophy
    "Richard Thaler": RICHARD_THALER,
    "Aristotle": ARISTOTLE,
    "Sigmund Freud": SIGMUND_FREUD,
    # Information Theory
    "Claude Shannon": CLAUDE_SHANNON,
    "Vannevar Bush": VANNEVAR_BUSH,
    # Software Engineering
    "Cursor Staff Engineer": CURSOR_STAFF_ENGINEER,
    "Pandas Core Contributor": PANDAS_CORE_CONTRIBUTOR,
    "Apache Foundation Senior Engineer": APACHE_SENIOR_ENGINEER,
    "Anthropic Agent Systems Engineer": ANTHROPIC_AGENT_SYSTEMS_ENGINEER,
    "OpenAI GPT Agents Core Team": OPENAI_GPT_AGENTS_CORE_TEAM,
    "Omar Khattab": OMAR_KHATTAB,
    # Product & Marketing
    "Alex Chen": ALEX_CHEN,
    "Stanford CS/Berkeley MBA Duo": STANFORD_BERKELEY_DUO,
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def get_expert(name: str) -> Optional[Expert]:
    """Get expert by name."""
    return ALL_EXPERTS.get(name)


def get_all_experts() -> List[Expert]:
    """Get all experts."""
    return list(ALL_EXPERTS.values())


def get_experts_by_domain(domain: ExpertDomain) -> List[Expert]:
    """Get all experts in a specific domain."""
    return [expert for expert in ALL_EXPERTS.values() if expert.domain == domain]


def get_expert_names_by_domain(domain: ExpertDomain) -> List[str]:
    """Get expert names by domain."""
    return [expert.name for expert in get_experts_by_domain(domain)]


def get_experts_by_names(names: List[str]) -> List[Expert]:
    """Get experts by list of names."""
    return [ALL_EXPERTS[name] for name in names if name in ALL_EXPERTS]


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "Expert",
    "ExpertDomain",
    "get_expert",
    "get_all_experts",
    "get_experts_by_domain",
    "get_expert_names_by_domain",
    "get_experts_by_names",
    "ALL_EXPERTS",
]
