"""Topic Research Swarm — DSPy Signatures for LLM-powered analysis."""

import dspy


class TopicDecompositionSignature(dspy.Signature):
    """Decompose a research topic into focused sub-topics for parallel investigation.

    You are a research strategist. Given a topic, break it down into 4-8
    focused sub-topics that together provide comprehensive coverage.
    Each sub-topic should be specific enough for a single web search.
    """

    topic: str = dspy.InputField(desc="Main research topic or question")
    context: str = dspy.InputField(
        desc="Additional context: mode (single/comparison/landscape), depth, entities"
    )

    sub_topics: str = dspy.OutputField(
        desc="4-8 focused sub-topics separated by |. Each should be a specific search query."
    )
    search_queries: str = dspy.OutputField(
        desc="2-3 web search queries per sub-topic, separated by |. "
        "Total queries separated by ||"
    )
    key_questions: str = dspy.OutputField(
        desc="5-10 key questions this research should answer, separated by |"
    )


class DataSynthesisSignature(dspy.Signature):
    """Synthesize multiple web search results into structured findings.

    You are an expert research analyst. Combine raw web search data into
    clear, sourced findings. Prioritize recent data, identify contradictions,
    and flag claims that need more evidence.
    """

    topic: str = dspy.InputField(desc="Research topic")
    sub_topic: str = dspy.InputField(desc="Specific sub-topic being synthesized")
    raw_data: str = dspy.InputField(desc="Raw web search results and extracted text")
    key_questions: str = dspy.InputField(desc="Questions this research should answer")

    findings: str = dspy.OutputField(
        desc="3-8 key findings, each as: CLAIM: ... | EVIDENCE: ... | CONFIDENCE: 0.0-1.0. "
        "Separated by ||"
    )
    data_gaps: str = dspy.OutputField(
        desc="Areas where data is insufficient or contradictory, separated by |"
    )
    key_numbers: str = dspy.OutputField(
        desc="Important quantitative data points found, as KEY: VALUE pairs separated by |"
    )


class ComparativeAnalysisSignature(dspy.Signature):
    """Compare entities across multiple dimensions with structured scoring.

    You are a strategy consultant. Produce a rigorous head-to-head comparison
    with specific data points, not vague generalizations. Every claim must
    reference a number or fact.
    """

    entities: str = dspy.InputField(desc="Entities being compared, separated by |")
    dimensions: str = dspy.InputField(
        desc="Comparison dimensions (e.g., Revenue, Market Share, Profitability), separated by |"
    )
    research_data: str = dspy.InputField(desc="All research findings about the entities")

    comparison: str = dspy.OutputField(
        desc="For each dimension: DIMENSION: ... | ENTITY1: value | ENTITY2: value | "
        "WINNER: ... | ANALYSIS: ... Dimensions separated by ||"
    )
    verdict: str = dspy.OutputField(
        desc="Overall verdict: which entity wins and why (2-3 sentences)"
    )
    key_asymmetries: str = dspy.OutputField(
        desc="3-5 critical differences that matter most, separated by |"
    )


class ReportCompilationSignature(dspy.Signature):
    """Compile research findings into a professional report.

    You are a senior analyst at McKinsey. Write a structured, data-rich
    report that executives would find actionable. Use markdown formatting.
    Include tables, bullet points, and clear section headers.
    """

    topic: str = dspy.InputField(desc="Research topic")
    findings: str = dspy.InputField(desc="All synthesized findings")
    comparison: str = dspy.InputField(desc="Comparison analysis (empty if not a comparison)")
    key_numbers: str = dspy.InputField(desc="Key quantitative data points")
    sources: str = dspy.InputField(desc="Source URLs and titles")

    report: str = dspy.OutputField(
        desc="Full markdown report with: Executive Summary, Key Findings, "
        "Detailed Analysis (per sub-topic), Data Tables, Comparison Matrix "
        "(if applicable), Sources. Use ## headers, tables, bullet points."
    )
    executive_summary: str = dspy.OutputField(
        desc="3-5 bullet point executive summary for busy readers"
    )
    one_liner: str = dspy.OutputField(desc="Single sentence thesis/conclusion")


class FactCheckSignature(dspy.Signature):
    """Fact-check research findings for accuracy and consistency.

    You are a fact-checker. Review claims against evidence. Flag
    unsupported claims, contradictions, and outdated data.
    """

    claims: str = dspy.InputField(desc="Claims to fact-check, separated by ||")
    evidence: str = dspy.InputField(desc="Available evidence and sources")

    verified_claims: str = dspy.OutputField(
        desc="For each claim: CLAIM: ... | STATUS: verified/unverified/contradicted | "
        "CONFIDENCE: 0.0-1.0 | NOTE: ... Separated by ||"
    )
    overall_reliability: float = dspy.OutputField(desc="Overall reliability score 0.0-1.0")
    issues: str = dspy.OutputField(
        desc="Specific issues found (contradictions, outdated data, unsupported claims), "
        "separated by |"
    )
