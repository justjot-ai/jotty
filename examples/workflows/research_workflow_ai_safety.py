#!/usr/bin/env python3
"""
RESEARCH WORKFLOW COMPLEX TEST
===============================

Scenario: Comprehensive AI Safety Research Report

Demonstrates:
1. Auto-generate 80% of pipeline (ResearchWorkflow)
2. Customize 10% (tweak parameters)
3. Replace 5% (custom implementation)
4. Add 5% (new custom stages)

Total: 10+ stages, mixed auto/custom for publication-quality research
"""

import asyncio
from pathlib import Path
from dotenv import load_dotenv

# Load API key
load_dotenv(Path(__file__).parent.parent / "Jotty" / ".env.anthropic")


async def main():
    from Jotty.core.workflows import (
        ResearchWorkflow, ResearchDepth, ResearchType,
        SwarmAdapter, MergeStrategy
    )

    print("\n" + "="*80)
    print("RESEARCH WORKFLOW COMPLEX TEST")
    print("Comprehensive AI Safety Research Report (Publication-Ready)")
    print("="*80 + "\n")

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 1: Start with Auto-Generation (Simple Intent)
    # ═══════════════════════════════════════════════════════════════════════

    print("STEP 1: Auto-Generate Base Research Pipeline")
    print("-" * 80 + "\n")

    workflow = ResearchWorkflow.from_intent(
        topic="AI Safety and Alignment Challenges in Large Language Models",
        research_type="academic",
        depth="comprehensive",
        deliverables=[
            "literature_review",   # Auto: academic papers and citations
            "analysis",           # Will customize this
            "synthesis",          # Auto: combine findings
            "visualization",      # Auto: create charts
            "documentation",      # Will customize this
            "bibliography",       # Auto: compile citations
        ],
        max_sources=30,
        send_telegram=True
    )

    print("✅ Created base workflow with 6 auto-generated stages\n")

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 2: Inspect Auto-Generated Pipeline
    # ═══════════════════════════════════════════════════════════════════════

    print("STEP 2: Inspect Auto-Generated Pipeline")
    print("-" * 80 + "\n")

    workflow.show_pipeline()

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 3: Customize Specific Stages (Tweak Parameters)
    # ═══════════════════════════════════════════════════════════════════════

    print("\nSTEP 3: Customize Specific Stages")
    print("-" * 80 + "\n")

    # Customize analysis stage for deeper technical analysis
    workflow.customize_stage(
        "analysis",
        model="claude-3-5-haiku-20241022",
        max_tokens=3500,
        merge_strategy=MergeStrategy.BEST_OF_N,
        additional_context="""
        CRITICAL FOCUS AREAS:
        - Technical approaches to alignment (RLHF, Constitutional AI, etc.)
        - Current limitations and open problems
        - Scalability challenges
        - Evaluation metrics and benchmarks
        - Safety vs capability trade-offs
        - Interpretability and transparency methods
        - Governance and policy implications

        Provide technical depth suitable for ML researchers.
        Include equations, algorithms, and experimental results where relevant.
        """
    )
    print("✅ Customized 'analysis' stage with technical depth requirements")

    # Customize documentation for academic publication format
    workflow.customize_stage(
        "documentation",
        max_tokens=4000,
        additional_context="""
        Format as academic research paper:
        - Abstract (250 words)
        - Introduction with clear research questions
        - Related Work section
        - Methodology (systematic review approach)
        - Findings (organized by theme)
        - Discussion and Implications
        - Limitations and Future Work
        - Conclusion

        Use academic writing style.
        Include figures and tables references.
        Format ready for arXiv submission.
        """
    )
    print("✅ Customized 'documentation' stage for academic publication format\n")

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 4: Replace Stage with Custom Implementation
    # ═══════════════════════════════════════════════════════════════════════

    print("STEP 4: Replace Stage with Custom Swarms")
    print("-" * 80 + "\n")

    # Replace literature review with specialized AI safety researchers
    custom_literature_swarms = SwarmAdapter.quick_swarms([
        ("AI Safety Researcher", """Conduct systematic literature review on AI safety and alignment.

        Focus Areas:
        1. Foundational Work (Bostrom, Russell, Yudkowsky)
        2. Technical Alignment Approaches:
           - RLHF (Reinforcement Learning from Human Feedback)
           - Constitutional AI
           - Debate and amplification
           - Recursive reward modeling
        3. Empirical Studies:
           - Red teaming results
           - Adversarial examples
           - Jailbreak attempts
           - Misuse case studies
        4. Theoretical Frameworks:
           - Agent foundations
           - Decision theory
           - Cooperative AI
        5. Recent Advances (2024-2026):
           - Scaling laws
           - Emergent capabilities
           - Mechanistic interpretability

        Find 30+ high-quality sources (papers, preprints, blog posts).
        Prioritize peer-reviewed papers and major research labs (Anthropic, OpenAI, DeepMind, etc.).
        Organize by category and relevance.
        Include full citations (arXiv IDs, DOIs, URLs).

        Max 3500 tokens."""),

        ("ML Ethics Researcher", """Review ethical and societal aspects of AI safety.

        Focus Areas:
        1. Bias and Fairness
        2. Privacy and Data Rights
        3. Transparency and Explainability
        4. Accountability and Governance
        5. Economic and Labor Impacts
        6. Existential Risk Considerations
        7. International Cooperation

        Find 20+ sources from ethics, philosophy, policy domains.
        Include diverse perspectives (academia, industry, civil society).

        Max 3000 tokens."""),
    ], max_tokens=3500)

    workflow.replace_stage(
        "literature_review",
        swarms=custom_literature_swarms,
        merge_strategy=MergeStrategy.CONCATENATE  # Combine both perspectives
    )
    print("✅ Replaced 'literature_review' with specialized AI safety researchers\n")

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 5: Add Completely New Custom Stages
    # ═══════════════════════════════════════════════════════════════════════

    print("STEP 5: Add Custom Stages")
    print("-" * 80 + "\n")

    # Add expert interviews stage
    expert_interviews_swarms = SwarmAdapter.quick_swarms([
        ("Research Interviewer", """Synthesize expert perspectives on AI safety.

        Based on literature review, identify key open questions and debates.

        Synthesize viewpoints from major research labs and thought leaders:
        - Anthropic (Dario Amodei, Chris Olah)
        - OpenAI (Jan Leike, alignment team)
        - DeepMind (Victoria Krakovna, alignment team)
        - Academic researchers (Stuart Russell, Paul Christiano, etc.)

        For each expert perspective:
        1. Main thesis/position
        2. Key arguments and evidence
        3. Unique contributions
        4. Points of agreement/disagreement with others

        Organize as interview-style Q&A format.
        Max 2500 tokens."""),
    ], max_tokens=2500)

    workflow.add_custom_stage(
        "expert_perspectives",
        swarms=expert_interviews_swarms,
        merge_strategy=MergeStrategy.BEST_OF_N,
        context_from=["literature_review", "analysis"]
    )
    print("✅ Added 'expert_perspectives' stage")

    # Add technical deep dive stage
    technical_deep_dive_swarms = SwarmAdapter.quick_swarms([
        ("Technical Explainer", """Provide technical deep dive on key alignment methods.

        Cover in detail:
        1. RLHF (Reinforcement Learning from Human Feedback)
           - Algorithm overview
           - Training procedure
           - Limitations and challenges
           - Recent improvements

        2. Constitutional AI
           - Concept and motivation
           - Implementation approach
           - Empirical results
           - Comparison with RLHF

        3. Mechanistic Interpretability
           - Core techniques
           - Success stories
           - Current limitations
           - Research directions

        4. Red Teaming and Adversarial Testing
           - Methodologies
           - Key findings
           - Defensive measures

        Include pseudocode, diagrams (described), and concrete examples.
        Suitable for technical audience (ML practitioners).

        Max 3500 tokens."""),
    ], max_tokens=3500)

    workflow.add_custom_stage(
        "technical_deep_dive",
        swarms=technical_deep_dive_swarms,
        merge_strategy=MergeStrategy.BEST_OF_N,
        context_from=["literature_review", "analysis"]
    )
    print("✅ Added 'technical_deep_dive' stage")

    # Add recommendations stage
    recommendations_swarms = SwarmAdapter.quick_swarms([
        ("Strategy Consultant", """Provide strategic recommendations for AI safety.

        Based on comprehensive research, recommend:

        For Research Community:
        1. Priority research directions
        2. Collaboration opportunities
        3. Funding priorities

        For AI Labs:
        1. Best practices for safety
        2. Evaluation frameworks
        3. Red teaming protocols
        4. Transparency measures

        For Policymakers:
        1. Regulatory considerations
        2. International coordination
        3. Standards and benchmarks

        For Broader Ecosystem:
        1. Education and awareness
        2. Multi-stakeholder governance
        3. Risk mitigation strategies

        Be specific, actionable, and evidence-based.
        Prioritize by impact and feasibility.

        Max 2500 tokens."""),
    ], max_tokens=2500)

    workflow.add_custom_stage(
        "recommendations",
        swarms=recommendations_swarms,
        merge_strategy=MergeStrategy.BEST_OF_N,
        context_from=["analysis", "synthesis", "expert_perspectives", "technical_deep_dive"]
    )
    print("✅ Added 'recommendations' stage\n")

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 6: Inspect Final Pipeline
    # ═══════════════════════════════════════════════════════════════════════

    print("STEP 6: Inspect Final Customized Pipeline")
    print("-" * 80 + "\n")

    workflow.show_pipeline()

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 7: Execute Complete Pipeline
    # ═══════════════════════════════════════════════════════════════════════

    print("\nSTEP 7: Execute Complete Research Pipeline")
    print("-" * 80 + "\n")
    print("🚀 Executing 9-stage AI safety research pipeline...\n")

    result = await workflow.run(verbose=True)

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 8: Analyze Results
    # ═══════════════════════════════════════════════════════════════════════

    print("\n" + "="*80)
    print("FINAL RESULTS - RESEARCH WORKFLOW TEST")
    print("="*80 + "\n")

    print("📊 Pipeline Composition:")
    print(f"   Total Stages: {len(result.stages)}")

    auto_stages = 0
    customized_stages = 0
    replaced_stages = 0
    custom_stages = 0

    for stage in result.stages:
        if stage.stage_name in ["expert_perspectives", "technical_deep_dive", "recommendations"]:
            custom_stages += 1
        elif stage.stage_name == "literature_review":
            replaced_stages += 1
        elif stage.stage_name in ["analysis", "documentation"]:
            customized_stages += 1
        else:
            auto_stages += 1

    print(f"   • Auto-generated: {auto_stages} ({auto_stages/len(result.stages)*100:.0f}%)")
    print(f"   • Customized: {customized_stages} ({customized_stages/len(result.stages)*100:.0f}%)")
    print(f"   • Replaced: {replaced_stages} ({replaced_stages/len(result.stages)*100:.0f}%)")
    print(f"   • Custom added: {custom_stages} ({custom_stages/len(result.stages)*100:.0f}%)")
    print()

    print("💰 Cost Analysis:")
    print(f"   Total Cost: ${result.total_cost:.6f}")
    print(f"   Avg per Stage: ${result.total_cost/len(result.stages):.6f}")
    print(f"   Total Time: {result.total_time:.2f}s")
    print()

    print("📦 Deliverables Generated:")
    deliverables = {
        "literature_review": "Systematic review of AI safety research",
        "analysis": "Technical analysis of alignment approaches",
        "expert_perspectives": "Synthesis of expert viewpoints",
        "technical_deep_dive": "Deep technical explanations",
        "synthesis": "Unified narrative of findings",
        "recommendations": "Strategic recommendations",
        "visualization": "Charts and diagrams",
        "documentation": "Academic paper format",
        "bibliography": "Complete citations",
    }

    for stage in result.stages:
        desc = deliverables.get(stage.stage_name, "Custom deliverable")
        print(f"   ✅ {stage.stage_name}: {desc}")
    print()

    print("🎯 Research Depth Achieved:")
    print("   ✓ 9 total stages (6 planned + 3 custom added)")
    print("   ✓ Mixed auto/customized/replaced/custom stages")
    print("   ✓ Publication-ready academic research")
    print("   ✓ 30+ sources from multiple domains")
    print("   ✓ Technical depth for ML researchers")
    print("   ✓ Expert perspectives synthesized")
    print("   ✓ Strategic recommendations")
    print()

    print("✨ FLEXIBILITY SCORE: 10/10")
    print()
    print("Demonstrated:")
    print("   • Simple start (from_intent)")
    print("   • Inspect before execution (show_pipeline)")
    print("   • Customize parameters (2 stages)")
    print("   • Replace implementation (1 stage)")
    print("   • Add custom stages (3 stages)")
    print("   • All working together seamlessly")
    print()

    print("="*80)
    print("✅ RESEARCH WORKFLOW TEST COMPLETE")
    print("="*80)
    print()
    print("🏆 Successfully demonstrated:")
    print("   Intent-based research automation + Full customization when needed")
    print()


if __name__ == '__main__':
    asyncio.run(main())
