#!/usr/bin/env python3
"""
Research Expert Team for Jotty
Creates academic-quality research papers using multi-agent workflow
"""

from typing import Any, Dict, List

import dspy

# =============================================================================
# RESEARCH EXPERT SIGNATURES
# =============================================================================


class LiteratureReviewSignature(dspy.Signature):
    """Literature review expert - surveys existing research"""

    topic: str = dspy.InputField(desc="Research topic to review")
    goal: str = dspy.InputField(desc="Specific focus area or research question")

    key_papers: str = dspy.OutputField(desc="List of seminal papers with citations")
    historical_context: str = dspy.OutputField(desc="Historical development of the field")
    current_state: str = dspy.OutputField(desc="Current state of research")
    literature_summary: str = dspy.OutputField(desc="Comprehensive literature review section")


class ConceptExplainerSignature(dspy.Signature):
    """Concept explainer - breaks down complex ideas into understandable parts"""

    concept: str = dspy.InputField(desc="Complex concept to explain")
    goal: str = dspy.InputField(desc="Target audience and explanation depth")
    context: str = dspy.InputField(desc="Background context and prerequisites")

    intuitive_explanation: str = dspy.OutputField(desc="Simple intuitive explanation")
    technical_details: str = dspy.OutputField(desc="Technical details and mechanics")
    analogies: str = dspy.OutputField(desc="Helpful analogies and metaphors")
    why_it_matters: str = dspy.OutputField(desc="Importance and applications")
    full_explanation: str = dspy.OutputField(desc="Complete explanation section")


class MathematicalAnalysisSignature(dspy.Signature):
    """Mathematical analyst - formulates and explains mathematical foundations"""

    topic: str = dspy.InputField(desc="Topic requiring mathematical analysis")
    goal: str = dspy.InputField(desc="What to derive or prove")

    mathematical_formulation: str = dspy.OutputField(desc="Core mathematical formulas in LaTeX")
    derivation: str = dspy.OutputField(desc="Step-by-step derivation")
    notation_guide: str = dspy.OutputField(desc="Explanation of notation used")
    complexity_analysis: str = dspy.OutputField(desc="Computational complexity if applicable")
    math_section: str = dspy.OutputField(
        desc="Complete mathematical section in markdown with LaTeX"
    )


class DiagramCreatorSignature(dspy.Signature):
    """Diagram creator - creates visual representations using Mermaid"""

    concept: str = dspy.InputField(desc="Concept to visualize")
    goal: str = dspy.InputField(desc="What the diagram should illustrate")

    diagram_type: str = dspy.OutputField(desc="Type of diagram (flowchart, sequence, architecture)")
    mermaid_code: str = dspy.OutputField(desc="Complete Mermaid diagram code")
    caption: str = dspy.OutputField(desc="Diagram caption and explanation")


class ReportWriterSignature(dspy.Signature):
    """Report writer - synthesizes research into coherent academic paper"""

    topic: str = dspy.InputField(desc="Research paper topic")
    goal: str = dspy.InputField(desc="Paper objective and target audience")
    literature_review: str = dspy.InputField(desc="Literature review section")
    explanations: str = dspy.InputField(desc="Concept explanations")
    mathematics: str = dspy.InputField(desc="Mathematical analysis")
    diagrams: str = dspy.InputField(desc="Mermaid diagrams")

    abstract: str = dspy.OutputField(desc="Paper abstract (150-250 words)")
    introduction: str = dspy.OutputField(desc="Introduction section")
    body_structure: str = dspy.OutputField(desc="Main body sections organized")
    conclusion: str = dspy.OutputField(desc="Conclusion and future work")
    references: str = dspy.OutputField(desc="References section")
    full_paper: str = dspy.OutputField(desc="Complete paper in markdown format")


# =============================================================================
# RESEARCH TEAM CREATION
# =============================================================================


def create_research_team() -> List[Dict[str, Any]]:
    """
    Create research expert team for academic paper generation

    Returns:
        List of agent configurations (dict format matching workflow templates)
    """

    agents = [
        {
            "name": "LiteratureReviewer",
            "agent": dspy.ChainOfThought(LiteratureReviewSignature),
            "expert": None,
            "tools": [],
            "role": "Expert in surveying and synthesizing academic literature",
        },
        {
            "name": "ConceptExplainer",
            "agent": dspy.ChainOfThought(ConceptExplainerSignature),
            "expert": None,
            "tools": [],
            "role": "Expert in breaking down complex concepts into clear explanations",
        },
        {
            "name": "MathematicalAnalyst",
            "agent": dspy.ChainOfThought(MathematicalAnalysisSignature),
            "expert": None,
            "tools": [],
            "role": "Expert in mathematical formulation and rigorous derivations",
        },
        {
            "name": "DiagramCreator",
            "agent": dspy.ChainOfThought(DiagramCreatorSignature),
            "expert": None,
            "tools": [],
            "role": "Expert in creating clear visual diagrams using Mermaid",
        },
        {
            "name": "ReportWriter",
            "agent": dspy.ChainOfThought(ReportWriterSignature),
            "expert": None,
            "tools": [],
            "role": "Expert in synthesizing research into publication-quality papers",
        },
    ]

    return agents


# =============================================================================
# TRANSFORMER-SPECIFIC SIGNATURES (for the demo)
# =============================================================================


class TransformerExplainerSignature(dspy.Signature):
    """Transformer architecture expert - comprehensive explanation"""

    goal: str = dspy.InputField(desc="What to explain about Transformers")

    # Core concepts
    attention_mechanism: str = dspy.OutputField(desc="How self-attention works")
    architecture_overview: str = dspy.OutputField(desc="Overall Transformer architecture")
    encoder_decoder: str = dspy.OutputField(desc="Encoder and decoder components")
    positional_encoding: str = dspy.OutputField(desc="Why and how positional encoding works")

    # Mathematical foundations
    attention_formula: str = dspy.OutputField(desc="Attention formula in LaTeX")
    multi_head_attention: str = dspy.OutputField(desc="Multi-head attention mathematics")

    # Why it matters
    advantages: str = dspy.OutputField(desc="Advantages over RNNs and CNNs")
    applications: str = dspy.OutputField(desc="Real-world applications")
    impact: str = dspy.OutputField(desc="Impact on NLP and beyond")

    # Diagrams
    architecture_diagram: str = dspy.OutputField(desc="Mermaid diagram of Transformer architecture")
    attention_diagram: str = dspy.OutputField(desc="Mermaid diagram of attention mechanism")

    # Complete paper
    full_paper: str = dspy.OutputField(desc="Complete research paper on Transformers in markdown")


def create_transformer_expert() -> Dict[str, Any]:
    """Create expert for Transformer paper generation"""
    return {
        "name": "TransformerExpert",
        "agent": dspy.ChainOfThought(TransformerExplainerSignature),
        "expert": None,
        "tools": [],
        "role": "Expert in Transformer architecture and attention mechanisms",
    }
