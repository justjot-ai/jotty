"""
Jotty World-Class Swarm Templates
==================================

Production-grade multi-agent swarms with:
- LLM-powered analysis (DSPy Chain of Thought)
- Parallel agent execution
- Shared resources (memory, context, bus)
- Self-improving feedback loop
- Learning from past research

Available Swarms:
-----------------

RESEARCH & ANALYSIS:
- ResearchSwarm: Comprehensive stock research with sentiment, peers, charts
- FundamentalSwarm: Financial statement analysis, valuation, investment thesis
- DataAnalysisSwarm: EDA, statistics, ML recommendations, insights

DEVELOPMENT:
- CodingSwarm: Code generation, architecture, testing, documentation
- TestingSwarm: Test generation, coverage analysis, quality assessment
- ReviewSwarm: Code review, security scanning, performance analysis

CONTENT & OPERATIONS:
- IdeaWriterSwarm: Multi-section content generation with section registry
- DevOpsSwarm: Infrastructure, CI/CD, containers, security, monitoring

EDUCATION:
- ArxivLearningSwarm: Paper understanding with progressive, intuitive explanations

META & LEARNING:
- BaseSwarm: Foundation with self-improving loop (Expert, Reviewer, Planner, Actor)
- LearningSwarm: Meta-swarm that improves all other swarms

Usage:
    # Research
    from core.swarms import ResearchSwarm, research
    result = await research("Paytm", send_telegram=True)

    # Coding
    from core.swarms import CodingSwarm, code
    result = await code("Create a REST API for user management")

    # Testing
    from core.swarms import TestingSwarm, test
    result = await test(my_code, language="python")

    # Fundamental Analysis
    from core.swarms import FundamentalSwarm, analyze_fundamentals
    result = await analyze_fundamentals("RELIANCE")

    # Data Analysis
    from core.swarms import DataAnalysisSwarm, analyze_data
    result = await analyze_data(df, question="What drives sales?")

    # Content Writing
    from core.swarms import IdeaWriterSwarm, write
    result = await write("The Future of AI")

    # DevOps
    from core.swarms import DevOpsSwarm, deploy
    result = await deploy("myapp", cloud="aws")

    # Code Review
    from core.swarms import ReviewSwarm, review_code
    result = await review_code(code, language="python")

    # ArXiv Paper Learning
    from core.swarms import ArxivLearningSwarm, learn_paper
    result = await learn_paper("1706.03762")  # Attention Is All You Need
    result = await learn_paper(topic="diffusion models")

    # Meta-Learning
    from core.swarms import LearningSwarm, improve_swarm
    result = await improve_swarm("coding")

Sync versions:
    from core.swarms import research_sync, code_sync, test_sync
    result = research_sync("TCS")
"""

# =============================================================================
# BASE SWARM INFRASTRUCTURE
# =============================================================================

from .base_swarm import (
    # Enums
    AgentRole,
    EvaluationResult,
    ImprovementType,

    # Data classes
    GoldStandard,
    Evaluation,
    ImprovementSuggestion,
    AgentConfig,
    ExecutionTrace,
    SwarmConfig,
    SwarmResult,

    # DSPy Signatures
    ExpertEvaluationSignature,
    ReviewerAnalysisSignature,
    PlannerOptimizationSignature,
    ActorExecutionSignature,

    # Core classes
    GoldStandardDB,
    ImprovementHistory,
    ExpertAgent,
    ReviewerAgent,
    PlannerAgent,
    ActorAgent,
    BaseSwarm,
    SwarmRegistry,
    register_swarm,
)

# =============================================================================
# RESEARCH SWARM
# =============================================================================

from .research_swarm import (
    # Main swarm
    ResearchSwarm,
    ResearchConfig,
    ResearchResult,
    RatingType,

    # Convenience functions
    research,
    research_sync,

    # Individual agents (for customization)
    BaseAgent,
    DataFetcherAgent,
    WebSearchAgent,
    SentimentAgent,
    LLMAnalysisAgent,
    PeerComparisonAgent,
    ChartGeneratorAgent,
    ReportGeneratorAgent,

    # New agents
    TechnicalAnalysisAgent,
    EnhancedChartGeneratorAgent,
    ScreenerAgent,
    SocialSentimentAgent,

    # DSPy Signatures (for extension)
    StockAnalysisSignature,
    SentimentAnalysisSignature,
    PeerSelectionSignature,
    SocialSentimentSignature,
    TechnicalSignalsSignature,
)

# =============================================================================
# CODING SWARM
# =============================================================================

from .coding_swarm import (
    CodingSwarm,
    CodingConfig,
    CodingResult,
    CodeOutput,
    CodeLanguage,
    CodeStyle,
    code,
    code_sync,
    # Agents
    ArchitectAgent,
    DeveloperAgent,
    DebuggerAgent,
    OptimizerAgent,
    TestWriterAgent,
    DocWriterAgent,
)

# =============================================================================
# TESTING SWARM
# =============================================================================

from .testing_swarm import (
    TestingSwarm,
    TestingConfig,
    TestingResult,
    TestSuite,
    TestCase,
    CoverageReport,
    TestType,
    TestFramework,
    CoverageTarget,
    test,
    test_sync,
    # Agents
    CodeAnalyzerAgent,
    UnitTestAgent,
    IntegrationTestAgent,
    E2ETestAgent,
    CoverageAgent,
    QualityAgent,
)

# =============================================================================
# FUNDAMENTAL ANALYSIS SWARM
# =============================================================================

from .fundamental_swarm import (
    FundamentalSwarm,
    FundamentalConfig,
    FundamentalResult,
    FinancialMetrics,
    ValuationMetrics,
    QualityMetrics,
    ValuationResult,
    InvestmentThesis,
    ValuationType,
    InvestmentStyle,
    RatingScale,
    analyze_fundamentals,
    analyze_fundamentals_sync,
    # Agents
    FinancialStatementAgent,
    RatioAnalysisAgent,
    ValuationAgent,
    QualityEarningsAgent,
    ManagementAgent,
    MoatAgent,
    ThesisAgent,
)

# =============================================================================
# IDEA WRITER SWARM
# =============================================================================

from .idea_writer_swarm import (
    IdeaWriterSwarm,
    WriterConfig,
    WriterResult,
    ContentResult,
    Section,
    Outline,
    ContentType,
    Tone,
    OutputFormat,
    write,
    write_sync,
    # Section Registry
    SectionRegistry,
    SectionWriter,
    # Built-in writers
    IntroductionWriter,
    BodySectionWriter,
    ConclusionWriter,
    MarketAnalysisWriter,
    CaseStudiesWriter,
    TechnicalDeepDiveWriter,
    ResearchFindingsWriter,
    # Agents
    OutlineAgent,
    ResearchAgent,
    PolishAgent,
)

# =============================================================================
# DATA ANALYSIS SWARM
# =============================================================================

from .data_analysis_swarm import (
    DataAnalysisSwarm,
    DataAnalysisConfig,
    AnalysisResult,
    DataProfile,
    StatisticalResult,
    Insight,
    MLRecommendation,
    Visualization,
    AnalysisType,
    DataType,
    VisualizationType,
    analyze_data,
    analyze_data_sync,
    # Agents
    DataProfilerAgent,
    EDAAgent,
    StatisticalAgent,
    InsightAgent,
    MLRecommenderAgent,
    VisualizationAgent,
)

# =============================================================================
# DEVOPS SWARM
# =============================================================================

from .devops_swarm import (
    DevOpsSwarm,
    DevOpsConfig,
    DevOpsResult,
    InfrastructureSpec,
    PipelineSpec,
    ContainerSpec,
    SecurityConfig,
    MonitoringConfig,
    CloudProvider,
    IaCTool,
    CIProvider,
    ContainerPlatform,
    deploy,
    deploy_sync,
    # Agents
    InfrastructureArchitect,
    CICDDesigner,
    ContainerSpecialist,
    SecurityHardener,
    MonitoringSpecialist,
    IaCGenerator,
)

# =============================================================================
# REVIEW SWARM
# =============================================================================

from .review_swarm import (
    ReviewSwarm,
    ReviewConfig,
    ReviewResult,
    ReviewComment,
    SecurityFinding,
    PerformanceFinding,
    ArchitectureFinding,
    ReviewType,
    Severity,
    ReviewStatus,
    review_code,
    review_code_sync,
    # Agents
    CodeReviewer,
    SecurityScanner,
    PerformanceAnalyzer,
    ArchitectureReviewer,
    StyleChecker,
    ReviewSynthesizer,
)

# =============================================================================
# ARXIV LEARNING SWARM
# =============================================================================

from .arxiv_learning_swarm import (
    ArxivLearningSwarm,
    ArxivLearningConfig,
    ArxivLearningResult,
    LearningContent,
    LearningSection,
    Concept,
    PaperInfo,
    LearningDepth,
    ContentStyle,
    AudienceLevel,
    learn_paper,
    learn_paper_sync,
    # Agents
    PaperFetcherAgent,
    ConceptExtractorAgent,
    IntuitionBuilderAgent,
    MathSimplifierAgent,
    ExampleGeneratorAgent,
    ProgressiveBuilderAgent,
    ContentPolisherAgent,
)

# =============================================================================
# LEARNING SWARM (META)
# =============================================================================

from .learning_swarm import (
    LearningSwarm,
    LearningConfig,
    LearningResult,
    SwarmPerformance,
    OptimizationResult,
    LearningMode,
    OptimizationType,
    improve_swarm,
    improve_swarm_sync,
    # Agents
    PerformanceEvaluator,
    GoldCurator,
    PromptOptimizer,
    WorkflowOptimizer,
    ParameterTuner,
    MetaLearner,
)

# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # ==========================================================================
    # BASE INFRASTRUCTURE
    # ==========================================================================
    # Enums
    "AgentRole",
    "EvaluationResult",
    "ImprovementType",

    # Data classes
    "GoldStandard",
    "Evaluation",
    "ImprovementSuggestion",
    "AgentConfig",
    "ExecutionTrace",
    "SwarmConfig",
    "SwarmResult",

    # DSPy Signatures
    "ExpertEvaluationSignature",
    "ReviewerAnalysisSignature",
    "PlannerOptimizationSignature",
    "ActorExecutionSignature",

    # Core classes
    "GoldStandardDB",
    "ImprovementHistory",
    "ExpertAgent",
    "ReviewerAgent",
    "PlannerAgent",
    "ActorAgent",
    "BaseSwarm",
    "SwarmRegistry",
    "register_swarm",

    # ==========================================================================
    # RESEARCH SWARM
    # ==========================================================================
    "ResearchSwarm",
    "ResearchConfig",
    "ResearchResult",
    "RatingType",
    "research",
    "research_sync",
    "BaseAgent",
    "DataFetcherAgent",
    "WebSearchAgent",
    "SentimentAgent",
    "LLMAnalysisAgent",
    "PeerComparisonAgent",
    "ChartGeneratorAgent",
    "ReportGeneratorAgent",
    "TechnicalAnalysisAgent",
    "EnhancedChartGeneratorAgent",
    "ScreenerAgent",
    "SocialSentimentAgent",
    "StockAnalysisSignature",
    "SentimentAnalysisSignature",
    "PeerSelectionSignature",
    "SocialSentimentSignature",
    "TechnicalSignalsSignature",

    # ==========================================================================
    # CODING SWARM
    # ==========================================================================
    "CodingSwarm",
    "CodingConfig",
    "CodingResult",
    "CodeOutput",
    "CodeLanguage",
    "CodeStyle",
    "code",
    "code_sync",
    "ArchitectAgent",
    "DeveloperAgent",
    "DebuggerAgent",
    "OptimizerAgent",
    "TestWriterAgent",
    "DocWriterAgent",

    # ==========================================================================
    # TESTING SWARM
    # ==========================================================================
    "TestingSwarm",
    "TestingConfig",
    "TestingResult",
    "TestSuite",
    "TestCase",
    "CoverageReport",
    "TestType",
    "TestFramework",
    "CoverageTarget",
    "test",
    "test_sync",
    "CodeAnalyzerAgent",
    "UnitTestAgent",
    "IntegrationTestAgent",
    "E2ETestAgent",
    "CoverageAgent",
    "QualityAgent",

    # ==========================================================================
    # FUNDAMENTAL SWARM
    # ==========================================================================
    "FundamentalSwarm",
    "FundamentalConfig",
    "FundamentalResult",
    "FinancialMetrics",
    "ValuationMetrics",
    "QualityMetrics",
    "ValuationResult",
    "InvestmentThesis",
    "ValuationType",
    "InvestmentStyle",
    "RatingScale",
    "analyze_fundamentals",
    "analyze_fundamentals_sync",
    "FinancialStatementAgent",
    "RatioAnalysisAgent",
    "ValuationAgent",
    "QualityEarningsAgent",
    "ManagementAgent",
    "MoatAgent",
    "ThesisAgent",

    # ==========================================================================
    # IDEA WRITER SWARM
    # ==========================================================================
    "IdeaWriterSwarm",
    "WriterConfig",
    "WriterResult",
    "ContentResult",
    "Section",
    "Outline",
    "ContentType",
    "Tone",
    "OutputFormat",
    "write",
    "write_sync",
    "SectionRegistry",
    "SectionWriter",
    "IntroductionWriter",
    "BodySectionWriter",
    "ConclusionWriter",
    "MarketAnalysisWriter",
    "CaseStudiesWriter",
    "TechnicalDeepDiveWriter",
    "ResearchFindingsWriter",
    "OutlineAgent",
    "ResearchAgent",
    "PolishAgent",

    # ==========================================================================
    # DATA ANALYSIS SWARM
    # ==========================================================================
    "DataAnalysisSwarm",
    "DataAnalysisConfig",
    "AnalysisResult",
    "DataProfile",
    "StatisticalResult",
    "Insight",
    "MLRecommendation",
    "Visualization",
    "AnalysisType",
    "DataType",
    "VisualizationType",
    "analyze_data",
    "analyze_data_sync",
    "DataProfilerAgent",
    "EDAAgent",
    "StatisticalAgent",
    "InsightAgent",
    "MLRecommenderAgent",
    "VisualizationAgent",

    # ==========================================================================
    # DEVOPS SWARM
    # ==========================================================================
    "DevOpsSwarm",
    "DevOpsConfig",
    "DevOpsResult",
    "InfrastructureSpec",
    "PipelineSpec",
    "ContainerSpec",
    "SecurityConfig",
    "MonitoringConfig",
    "CloudProvider",
    "IaCTool",
    "CIProvider",
    "ContainerPlatform",
    "deploy",
    "deploy_sync",
    "InfrastructureArchitect",
    "CICDDesigner",
    "ContainerSpecialist",
    "SecurityHardener",
    "MonitoringSpecialist",
    "IaCGenerator",

    # ==========================================================================
    # REVIEW SWARM
    # ==========================================================================
    "ReviewSwarm",
    "ReviewConfig",
    "ReviewResult",
    "ReviewComment",
    "SecurityFinding",
    "PerformanceFinding",
    "ArchitectureFinding",
    "ReviewType",
    "Severity",
    "ReviewStatus",
    "review_code",
    "review_code_sync",
    "CodeReviewer",
    "SecurityScanner",
    "PerformanceAnalyzer",
    "ArchitectureReviewer",
    "StyleChecker",
    "ReviewSynthesizer",

    # ==========================================================================
    # ARXIV LEARNING SWARM
    # ==========================================================================
    "ArxivLearningSwarm",
    "ArxivLearningConfig",
    "ArxivLearningResult",
    "LearningContent",
    "LearningSection",
    "Concept",
    "PaperInfo",
    "LearningDepth",
    "ContentStyle",
    "AudienceLevel",
    "learn_paper",
    "learn_paper_sync",
    "PaperFetcherAgent",
    "ConceptExtractorAgent",
    "IntuitionBuilderAgent",
    "MathSimplifierAgent",
    "ExampleGeneratorAgent",
    "ProgressiveBuilderAgent",
    "ContentPolisherAgent",

    # ==========================================================================
    # LEARNING SWARM (META)
    # ==========================================================================
    "LearningSwarm",
    "LearningConfig",
    "LearningResult",
    "SwarmPerformance",
    "OptimizationResult",
    "LearningMode",
    "OptimizationType",
    "improve_swarm",
    "improve_swarm_sync",
    "PerformanceEvaluator",
    "GoldCurator",
    "PromptOptimizer",
    "WorkflowOptimizer",
    "ParameterTuner",
    "MetaLearner",
]
