"""
Swarm Templates - All swarms as configuration on unified SwarmLearning
===================================================================

All templates inherit from SwarmLearning and define:
- AGENT_TEAM: Agent composition
- COORDINATION: Pattern (AUTO, SEQUENTIAL, PARALLEL, etc.)
- Optional: STAGES for CUSTOM pattern

Templates get ALL 8 learning layers automatically:
1. Memory (5 levels)
2. TD-Lambda reinforcement learning
3. Swarm Intelligence meta-learning
4. Gold Standard evaluation
5. Improvement Agents (Expert, Reviewer, Planner, Actor, Auditor, Learner)
6. Pattern Learning
7. Transfer Learning
8. Adaptive Components

Author: Jotty Team
Date: February 2026
"""

# Learning templates
from .arxiv_learning import ArxivLearningTemplate

# Core templates
from .coding import CodingTemplate
from .data_analysis import DataAnalysisTemplate
from .devops import DevopsTemplate
from .fundamental import FundamentalTemplate
from .idea_writer import IdeaWriterTemplate
from .learning import LearningTemplate

# ML templates
from .ml import MLTemplate
from .ml_comprehensive import MlComprehensiveTemplate
from .olympiad_learning import OlympiadLearningTemplate
from .perspective_learning import PerspectiveLearningTemplate

# Pilot template
from .pilot import PilotTemplate
from .research import ResearchTemplate
from .review import ReviewTemplate

# Team pattern templates
from .team_patterns.collaborative import CollaborativeTemplate
from .team_patterns.hybrid import HybridTemplate
from .team_patterns.sequential import SequentialTemplate
from .testing import TestingTemplate

# Backward compatibility aliases
CodingSwarm = CodingTemplate
ResearchSwarm = ResearchTemplate
ReviewSwarm = ReviewTemplate
TestingSwarm = TestingTemplate
DataAnalysisSwarm = DataAnalysisTemplate
DevOpsSwarm = DevopsTemplate
FundamentalSwarm = FundamentalTemplate
IdeaWriterSwarm = IdeaWriterTemplate
LearningSwarm = LearningTemplate

ArxivLearningSwarm = ArxivLearningTemplate
OlympiadLearningSwarm = OlympiadLearningTemplate
PerspectiveLearningSwarm = PerspectiveLearningTemplate

PilotSwarm = PilotTemplate

SwarmML = MLTemplate
SwarmMLComprehensive = MlComprehensiveTemplate

CollaborativeTeam = CollaborativeTemplate
HybridTeam = HybridTemplate
SequentialTeam = SequentialTemplate

__all__ = [
    # Templates
    "CodingTemplate",
    "ResearchTemplate",
    "ReviewTemplate",
    "TestingTemplate",
    "DataAnalysisTemplate",
    "DevopsTemplate",
    "FundamentalTemplate",
    "IdeaWriterTemplate",
    "LearningTemplate",
    "ArxivLearningTemplate",
    "OlympiadLearningTemplate",
    "PerspectiveLearningTemplate",
    "PilotTemplate",
    "MLTemplate",
    "MlComprehensiveTemplate",
    "CollaborativeTemplate",
    "HybridTemplate",
    "SequentialTemplate",
    # Aliases (backward compat)
    "CodingSwarm",
    "ResearchSwarm",
    "ReviewSwarm",
    "TestingSwarm",
    "DataAnalysisSwarm",
    "DevOpsSwarm",
    "FundamentalSwarm",
    "IdeaWriterSwarm",
    "LearningSwarm",
    "ArxivLearningSwarm",
    "OlympiadLearningSwarm",
    "PerspectiveLearningSwarm",
    "PilotSwarm",
    "SwarmML",
    "SwarmMLComprehensive",
    "CollaborativeTeam",
    "HybridTeam",
    "SequentialTeam",
]
