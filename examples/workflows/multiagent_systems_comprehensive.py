#!/usr/bin/env python3
"""
COMPREHENSIVE MULTI-AGENT SYSTEMS LEARNING COURSE
==================================================

Uses LearningWorkflow (proven to generate 10,000+ words of comprehensive content).
Delivers PDF to both Telegram + WhatsApp (#my-notes) via environment config.

Approach: Treat multi-agent systems as an advanced learning topic (university/research level)
instead of research paper - this generates much more detailed educational content.
"""

import asyncio
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment
env_path = Path(__file__).parent.parent.parent / "Jotty" / ".env.anthropic"
if env_path.exists():
    load_dotenv(env_path)


async def main():
    from Jotty.core.workflows import (
        LearningWorkflow,
        LearningLevel,
        LearningDepth,
        Subject,
        OutputFormatManager,
        OutputChannelManager,
    )
    from Jotty.core.orchestration.swarm_adapter import SwarmAdapter
    from Jotty.core.orchestration.multi_stage_pipeline import MergeStrategy

    print("\n" + "="*80)
    print("COMPREHENSIVE MULTI-AGENT SYSTEMS COURSE")
    print("University/Research Level - Full Technical Deep Dive")
    print("="*80 + "\n")

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 1: Auto-Generate Comprehensive Learning Course
    # ═══════════════════════════════════════════════════════════════════════

    print("STEP 1: Auto-Generate Comprehensive Learning Course")
    print("-" * 80 + "\n")

    # Use LearningWorkflow with maximum comprehensiveness
    workflow = LearningWorkflow.from_intent(
        subject="computer_science",  # CS domain
        topic="Multi-Agent Systems: Architectures, Coordination, Memory, and Learning",
        student_name="Advanced Researcher",
        depth="marathon",  # Maximum depth - full day workshop
        level="research",  # Research/PhD level content
        deliverables=[
            "curriculum",         # Learning plan
            "concepts",          # Fundamental concepts
            "intuition",         # Deep intuitive understanding
            "patterns",          # Advanced patterns & techniques
            "examples",          # Detailed worked examples
            "problems",          # Challenge problems
            "solutions",         # Comprehensive solutions
            "mistakes",          # Common pitfalls
            "connections",       # Links to other topics
            "assessment",        # Tests and quizzes
            "content_assembly",  # Final integration
        ],
        output_formats=["pdf"],  # Generate PDF
        send_telegram=False,  # We'll handle delivery manually
        include_assessment=True
    )

    print("✅ Created learning workflow with 11 comprehensive stages")
    print("   Level: Research/PhD")
    print("   Depth: Marathon (full day)")
    print()

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 2: Customize for Multi-Agent Systems Depth
    # ═══════════════════════════════════════════════════════════════════════

    print("STEP 2: Customize for Multi-Agent Systems Technical Depth")
    print("-" * 80 + "\n")

    # Customize concepts for comprehensive coverage
    workflow.customize_stage(
        "concepts",
        max_tokens=6000,
        additional_context="""
        COMPREHENSIVE MULTI-AGENT SYSTEMS CONCEPTS:

        Cover ALL fundamental concepts in depth (2000+ words total):

        1. AGENT FUNDAMENTALS (400 words):
           - Autonomy: goal-directed, independent decision-making
           - Reactivity: perceive environment, respond in timely fashion
           - Pro-activeness: goal-directed, take initiative
           - Social ability: interact with other agents
           - Rationality: act to achieve goals (bounded rationality)
           - Learning: improve from experience

        2. AGENT ARCHITECTURES (500 words):
           - Reactive agents: subsumption architecture (Brooks)
             * No internal state, stimulus-response
             * Layered behaviors: avoid, wander, explore, map
             * Example: Roomba vacuum, simple robots
           - Deliberative agents: BDI (Belief-Desire-Intention)
             * Internal world model, reasoning, planning
             * Beliefs: what agent knows about world
             * Desires: goals agent wants to achieve
             * Intentions: commitments to plans
             * Example: JACK, Jason, Jadex frameworks
           - Hybrid: combining reactive + deliberative
             * InteRRaP: 3-layer (reactive, local planning, cooperative)
             * Advantages: responsiveness + rationality

        3. COMMUNICATION & COORDINATION (500 words):
           - Agent Communication Languages (ACL)
             * FIPA ACL: speech acts, performatives
             * Message structure: sender, receiver, content, protocol
           - Coordination mechanisms:
             * Negotiation: alternating offers, game theory
             * Contract Net: task announcement → bidding → award
             * Voting: consensus building, Byzantine agreement
             * Market mechanisms: auctions, trading
           - Coordination topologies:
             * Centralized: hub-spoke, single coordinator
             * Decentralized: peer-to-peer, distributed
             * Hierarchical: tree structures, command chains

        4. MEMORY & KNOWLEDGE (300 words):
           - Episodic memory: experience sequences
           - Semantic memory: facts, concepts, ontologies
           - Procedural memory: skills, compiled knowledge
           - Shared memory: blackboard systems, tuple spaces
           - Knowledge representation: logic, rules, graphs

        5. LEARNING & ADAPTATION (300 words):
           - Reinforcement learning in MAS
           - Multi-agent RL: non-stationarity, credit assignment
           - Opponent modeling: predicting other agents
           - Emergent communication: learning to communicate
           - Evolution: genetic algorithms, coevolution

        Write as complete educational textbook content with:
        - Full explanations, not bullet points
        - Technical depth with examples
        - Clear definitions and formulas
        - Real-world applications
        """
    )
    print("✅ Customized 'concepts' for comprehensive technical depth (6000 tokens)")

    # Customize patterns for advanced techniques
    workflow.customize_stage(
        "patterns",
        max_tokens=6000,
        additional_context="""
        ADVANCED MULTI-AGENT SYSTEMS PATTERNS:

        Cover advanced design patterns and techniques (2000+ words):

        1. ARCHITECTURAL PATTERNS (500 words):
           - Mediator pattern: central coordinator
           - Blackboard pattern: shared knowledge space
           - Publish-Subscribe: event-driven communication
           - Pipeline: sequential processing chain
           - Hierarchical decomposition: layered control

        2. COORDINATION PATTERNS (600 words):
           - Master-Slave: centralized control
           - Peer-to-Peer: symmetric cooperation
           - Team: shared goals, joint plans
           - Coalition: temporary alliances
           - Emergence: self-organization from local rules
           - Stigmergy: indirect coordination via environment

        3. NEGOTIATION PATTERNS (400 words):
           - One-to-one bargaining: alternating offers
           - One-to-many: auctions (English, Dutch, sealed-bid)
           - Many-to-many: double auctions, markets
           - Argumentation: reason-based negotiation
           - Game-theoretic: Nash equilibrium, mechanism design

        4. LEARNING PATTERNS (300 words):
           - Independent learners: treat others as environment
           - Joint action learners: model other agents
           - Centralized training, decentralized execution (CTDE)
           - Self-play: learn from copies of self
           - Population-based: evolutionary approaches

        5. ROBUSTNESS PATTERNS (200 words):
           - Redundancy: backup agents
           - Diversity: heterogeneous agent types
           - Graceful degradation: partial functionality
           - Fault tolerance: Byzantine agreement

        For EACH pattern:
        - When to use it (recognition triggers)
        - How to implement it (step-by-step)
        - Pros and cons
        - Real-world examples
        - Common variations
        """
    )
    print("✅ Customized 'patterns' for advanced design patterns (6000 tokens)")

    # Customize problems for real-world challenges
    workflow.customize_stage(
        "problems",
        max_tokens=6000,
        additional_context="""
        COMPREHENSIVE PROBLEM SET:

        Generate 15-20 progressively challenging problems covering:

        LEVEL 1: Foundational (5 problems)
        - Design a reactive agent for obstacle avoidance
        - Implement BDI agent for delivery robot
        - Create simple negotiation protocol
        - Design blackboard system for collaborative filtering
        - Implement basic reinforcement learning agent

        LEVEL 2: Intermediate (5 problems)
        - Multi-agent path planning in grid world
        - Auction-based task allocation
        - Consensus in presence of Byzantine agents
        - Emergent flocking behavior from local rules
        - Multi-agent coordination game (prisoner's dilemma)

        LEVEL 3: Advanced (5-10 problems)
        - AlphaStar-style multi-agent RL for StarCraft
        - Swarm robotics: warehouse automation
        - Smart city traffic light coordination
        - Multi-agent adversarial search
        - Coalition formation with dynamic preferences
        - Emergent communication in referential games
        - Multi-agent planning with partial observability
        - Mechanism design for truthful bidding

        For EACH problem:
        - Clear problem statement (2-3 paragraphs)
        - Input/output specification
        - Constraints and assumptions
        - Hints for approach
        - Difficulty rating (⭐ to ⭐⭐⭐⭐⭐)
        """
    )
    print("✅ Customized 'problems' for real-world challenges (6000 tokens)")

    # Customize solutions for comprehensive explanations
    workflow.customize_stage(
        "solutions",
        max_tokens=8000,  # Maximum for detailed solutions
        additional_context="""
        COMPREHENSIVE SOLUTIONS:

        For EACH problem, provide:

        1. MULTIPLE APPROACHES (3-5 different solutions):
           - Approach 1: Simple/naive solution
           - Approach 2: Optimized solution
           - Approach 3: Advanced/research-level solution
           - Compare time/space complexity

        2. DETAILED EXPLANATION:
           - Intuition: Why this approach works
           - Algorithm: Step-by-step pseudocode
           - Implementation: Key code snippets
           - Analysis: Time/space complexity, correctness proof

        3. WORKED EXAMPLE:
           - Concrete input
           - Trace through algorithm step-by-step
           - Show intermediate states
           - Final output with explanation

        4. VARIATIONS & EXTENSIONS:
           - What if constraints change?
           - How to scale to 1000+ agents?
           - Real-world considerations
           - Research directions

        Write solutions as complete tutorial content, not brief answers.
        Target: 400-500 words per problem solution.
        """
    )
    print("✅ Customized 'solutions' for comprehensive explanations (8000 tokens)")

    # Customize content assembly for publication quality
    workflow.customize_stage(
        "content_assembly",
        max_tokens=6000,
        additional_context="""
        ASSEMBLE PUBLICATION-QUALITY COURSE:

        Create comprehensive course document integrating ALL previous stages:

        STRUCTURE:
        1. Title Page
           - Course title: "Multi-Agent Systems: Comprehensive Technical Course"
           - Subtitle: "From Foundations to Advanced Research"
           - Level: University/Research

        2. Table of Contents (auto-generated from sections)

        3. Course Overview (from curriculum stage)
           - Learning objectives
           - Prerequisites
           - Course structure
           - Time commitment

        4. Part I: Fundamentals (from concepts stage)
           - All core concepts in full
           - 2000+ words of educational content

        5. Part II: Advanced Patterns (from patterns stage)
           - All design patterns in full
           - 2000+ words of technique explanation

        6. Part III: Worked Examples (from examples stage)
           - Complete examples with full explanations

        7. Part IV: Problem Set (from problems stage)
           - All 15-20 problems

        8. Part V: Comprehensive Solutions (from solutions stage)
           - All detailed solutions

        9. Part VI: Common Mistakes (from mistakes stage)
           - Pitfalls and how to avoid them

        10. Part VII: Connections (from connections stage)
            - Links to AI, distributed systems, game theory

        11. Part VIII: Assessment (from assessment stage)
            - Quizzes, tests, projects

        12. Appendices
            - Further reading
            - Resources
            - Research papers

        Format as professional educational textbook.
        Use clear headers, subheaders, emphasis.
        Include diagrams (described textually).
        Target: 10,000-15,000 words total.
        """
    )
    print("✅ Customized 'content_assembly' for publication quality (6000 tokens)")

    print()

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 3: Execute Comprehensive Learning Course Generation
    # ═══════════════════════════════════════════════════════════════════════

    print("STEP 3: Execute Comprehensive Course Generation")
    print("-" * 80 + "\n")

    print("🚀 Starting comprehensive course generation...")
    print("   This will take 8-12 minutes due to depth...")
    print("   Expected: 10,000-15,000 words, $0.04-0.06 cost")
    print()

    result = await workflow.run(verbose=True)

    print()
    print("✅ Course Generation Complete!")
    print(f"   Stages: {len(result.stages)}")
    print(f"   Cost: ${result.total_cost:.6f}")
    print(f"   Time: {result.total_time:.2f}s")
    print()

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 4: Extract Content
    # ═══════════════════════════════════════════════════════════════════════

    print("STEP 4: Extract Course Content")
    print("-" * 80 + "\n")

    # Find content assembly stage (has the complete integrated content)
    content_stage = next((s for s in result.stages if s.stage_name == "content_assembly"), None)
    if not content_stage:
        print("❌ Content assembly stage not found")
        return

    # Save markdown
    output_dir = Path.home() / "jotty" / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    markdown_path = output_dir / "multiagent_systems_course.md"
    with open(markdown_path, 'w', encoding='utf-8') as f:
        f.write(content_stage.result.output)

    content_size = len(content_stage.result.output)
    word_count = content_size // 5
    page_count = word_count // 500

    print(f"✅ Saved course content: {markdown_path}")
    print(f"   Size: {content_size:,} characters ({content_size/1024:.1f}KB)")
    print(f"   Words: ~{word_count:,} words")
    print(f"   Pages: ~{page_count} pages")
    print()

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 5: Generate PDF
    # ═══════════════════════════════════════════════════════════════════════

    print("STEP 5: Generate PDF")
    print("-" * 80 + "\n")

    format_manager = OutputFormatManager(output_dir=str(output_dir))

    print("🔄 Generating PDF...")
    pdf_result = format_manager.generate_pdf(
        markdown_path=str(markdown_path),
        title="Multi-Agent Systems: Comprehensive Technical Course",
        author="Jotty AI Learning Framework",
        page_size="a4"
    )

    if pdf_result.success:
        file_to_send = pdf_result.file_path
        file_format = "PDF"
        pdf_size_kb = Path(pdf_result.file_path).stat().st_size / 1024
        print(f"✅ PDF generated: {Path(pdf_result.file_path).name}")
        print(f"   Size: {pdf_size_kb:.1f} KB")
    else:
        print(f"❌ PDF generation failed: {pdf_result.error}")
        print(f"⚠️  Falling back to Markdown")
        file_to_send = str(markdown_path)
        file_format = "Markdown"

    print()

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 6: Send to Telegram (from environment)
    # ═══════════════════════════════════════════════════════════════════════

    print("STEP 6: Send to Telegram")
    print("-" * 80 + "\n")

    channel_manager = OutputChannelManager()

    telegram_caption = f"""📚 <b>Multi-Agent Systems: Comprehensive Course</b>

<b>University/Research Level Technical Course</b>

📄 {len(result.stages)} learning stages
📊 ~{word_count:,} words (~{page_count} pages)
💰 ${result.total_cost:.6f}

<b>Complete Coverage:</b>
✅ Fundamental Concepts (2000+ words)
✅ Advanced Patterns (2000+ words)
✅ 15-20 Challenge Problems
✅ Comprehensive Solutions
✅ Common Mistakes & Pitfalls
✅ Real-World Applications
✅ Assessment & Quizzes

<i>Generated by Jotty AI Learning Framework</i>"""

    print(f"📤 Sending {file_format} to Telegram...")
    telegram_result = await channel_manager.send_to_telegram(
        file_path=file_to_send,
        caption=telegram_caption,
        parse_mode="HTML"
    )

    if telegram_result.success:
        print(f"✅ Sent to Telegram!")
        print(f"   Message ID: {telegram_result.message_id}")
        print(f"   Format: {file_format}")
    else:
        print(f"❌ Telegram send failed: {telegram_result.error}")

    print()

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 7: Send to WhatsApp #my-notes (from environment)
    # ═══════════════════════════════════════════════════════════════════════

    print("STEP 7: Send to WhatsApp #my-notes")
    print("-" * 80 + "\n")

    # Get WhatsApp config from environment
    whatsapp_to = os.environ.get('WHATSAPP_CHANNEL') or os.environ.get('WHATSAPP_TO')
    whatsapp_result = None

    if whatsapp_to:
        print(f"📤 Sending to WhatsApp: {whatsapp_to}")

        whatsapp_caption = f"""📚 *Multi-Agent Systems: Comprehensive Course*

*University/Research Level*

📄 {len(result.stages)} stages
📊 ~{word_count:,} words ({page_count} pages)
💰 ${result.total_cost:.6f}

*Complete Coverage:*
✅ Fundamentals (2000+ words)
✅ Advanced Patterns
✅ 15-20 Problems + Solutions
✅ Mistakes & Pitfalls
✅ Real-World Applications
✅ Assessment

_Generated by Jotty AI_"""

        whatsapp_result = await channel_manager.send_to_whatsapp(
            to=whatsapp_to,
            file_path=file_to_send,
            caption=whatsapp_caption,
            provider="baileys"
        )

        if whatsapp_result.success:
            print(f"✅ Sent to WhatsApp #my-notes!")
            print(f"   Message ID: {whatsapp_result.message_id}")
            print(f"   Provider: Bailey")
            print(f"   Format: {file_format}")
        else:
            print(f"❌ WhatsApp send failed: {whatsapp_result.error}")
    else:
        print("⚠️  WhatsApp not configured")
        print("   Set in .env.anthropic:")
        print("   WHATSAPP_CHANNEL='your-group-jid@g.us'")
        print("   WHATSAPP_TO='14155238886'  # or phone number")

    print()

    # ═══════════════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ═══════════════════════════════════════════════════════════════════════

    print("="*80)
    print("COMPREHENSIVE MULTI-AGENT SYSTEMS COURSE COMPLETE")
    print("="*80 + "\n")

    channels_sent = []
    if telegram_result.success:
        channels_sent.append("Telegram")
    if whatsapp_result and whatsapp_result.success:
        channels_sent.append("WhatsApp (#my-notes)")

    print("📤 Delivery Summary:")
    if channels_sent:
        for ch in channels_sent:
            print(f"   ✅ {ch}")
    else:
        print("   ⚠️  No channels delivered")
    print()

    print(f"📊 Content Generated:")
    print(f"   Words: ~{word_count:,}")
    print(f"   Pages: ~{page_count}")
    print(f"   Format: {file_format}")
    print()

    print(f"💰 Total Cost: ${result.total_cost:.6f}")
    print(f"⏱️  Total Time: {result.total_time:.0f}s ({result.total_time/60:.1f} minutes)")
    print()

    print("📁 Files:")
    print(f"   {output_dir}/")
    if pdf_result.success:
        print(f"   - {Path(pdf_result.file_path).name} ✅ PRIMARY")
    print(f"   - {markdown_path.name}")
    print()

    print("🎊 Comprehensive technical course complete!")
    print()


if __name__ == '__main__':
    asyncio.run(main())
