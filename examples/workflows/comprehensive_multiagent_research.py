#!/usr/bin/env python3
"""
COMPREHENSIVE MULTI-AGENT SYSTEM RESEARCH
==========================================

Deep comprehensive research on multi-agent systems (like olympiad/research examples).

Generates FULL detailed content:
- 15+ stages with built-in comprehensive prompts
- 8000-12000+ words of actual technical content
- PDF output as primary format
- WhatsApp Bailey + Telegram delivery
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
    from Jotty.core.modes.workflow import (
        ResearchWorkflow,
        ResearchDepth,
        ResearchType,
        OutputFormatManager,
        OutputChannelManager,
    )
    from Jotty.core.intelligence.orchestration.swarm_adapter import SwarmAdapter
    from Jotty.core.intelligence.orchestration.multi_stage_pipeline import MergeStrategy

    print("\n" + "="*80)
    print("COMPREHENSIVE MULTI-AGENT SYSTEM RESEARCH")
    print("Publication-Quality Deep Dive (12,000+ words)")
    print("="*80 + "\n")

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 1: Auto-Generate Comprehensive Research Pipeline
    # ═══════════════════════════════════════════════════════════════════════

    print("STEP 1: Auto-Generate Comprehensive Research Pipeline")
    print("-" * 80 + "\n")

    # Use ALL built-in deliverables for maximum comprehensiveness
    workflow = ResearchWorkflow.from_intent(
        topic="Multi-Agent Systems: Architectures, Coordination, Memory, and Learning",
        research_type="academic",
        depth="comprehensive",  # Deep research with 25+ sources
        deliverables=[
            "overview",           # High-level summary
            "literature_review",  # 25+ sources, comprehensive analysis
            "analysis",          # Deep technical analysis
            "synthesis",         # Integration of findings
            "visualization",     # Diagrams and charts
            "documentation",     # Final comprehensive paper
            "bibliography",      # Academic citations
        ],
        max_sources=30,
        send_telegram=False  # We'll handle delivery manually
    )

    print("✅ Created base workflow with 7 comprehensive stages\n")

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 2: Customize for Academic Publication Quality
    # ═══════════════════════════════════════════════════════════════════════

    print("STEP 2: Customize Stages for Publication Quality")
    print("-" * 80 + "\n")

    # Customize literature review for depth
    workflow.customize_stage(
        "literature_review",
        max_tokens=6000,  # Allow long comprehensive review
        additional_context="""
        COMPREHENSIVE LITERATURE REVIEW REQUIREMENTS:

        Cover ALL major aspects with 30+ academic sources:

        1. FOUNDATIONAL WORK (5-7 sources):
           - Early multi-agent systems (1980s-1990s)
           - Distributed AI foundations
           - Agent-oriented programming origins
           - Cite: Brooks (subsumption), Minsky (society of mind), Wooldridge & Jennings

        2. AGENT ARCHITECTURES (7-10 sources):
           - Reactive agents (Brooks, behavioral robotics)
           - BDI model (Rao & Georgeff, practical reasoning)
           - Hybrid architectures (InteRRaP, TouringMachines)
           - Cognitive architectures (SOAR, ACT-R integration)
           - Modern deep learning agents
           - Cite recent papers from AAMAS, IJCAI, NeurIPS

        3. COMMUNICATION & COORDINATION (8-10 sources):
           - FIPA ACL and communication languages
           - Contract Net Protocol (Smith 1980)
           - Market-based coordination (auctions, trading agents)
           - Consensus algorithms (Paxos, Raft, Byzantine)
           - Swarm intelligence (ant colony, particle swarm, bees)
           - Multi-agent planning (distributed planning, task allocation)

        4. LEARNING & ADAPTATION (6-8 sources):
           - Multi-agent reinforcement learning (independent learners)
           - Centralized training, decentralized execution (CTDE)
           - Opponent modeling and theory of mind
           - Emergent communication and language evolution
           - Recent breakthroughs: AlphaStar, OpenAI Five, Hide and Seek

        5. APPLICATIONS (4-6 sources):
           - Smart cities and traffic management
           - Multi-robot systems and warehouses
           - Trading and market simulation
           - Distributed sensor networks
           - Healthcare and epidemic modeling

        For EACH source:
        - Full citation (Author, Year, Title, Venue)
        - Key contribution (2-3 sentences)
        - Relevance to multi-agent systems
        - Connections to other work

        Organize chronologically AND thematically.
        Total length: 2500+ words minimum.
        """
    )
    print("✅ Customized 'literature_review' for comprehensive depth (6000 tokens)")

    # Customize analysis for technical depth
    workflow.customize_stage(
        "analysis",
        max_tokens=6000,
        merge_strategy=MergeStrategy.BEST_OF_N,
        additional_context="""
        DEEP TECHNICAL ANALYSIS REQUIREMENTS:

        Generate 2500+ words of detailed technical analysis covering:

        1. AGENT TYPES & ARCHITECTURES (600 words):
           - Reactive agents: Subsumption architecture, behavior-based design
             * Layered control (avoid, wander, explore, map)
             * No internal state, purely reactive
             * Examples: Brooks' robots, Roomba vacuums
           - Deliberative agents (BDI): Belief-Desire-Intention model
             * Formal logic representation (modal logic)
             * Practical reasoning algorithm
             * Plan library, intention selection
             * Examples: JACK, Jason, Jadex platforms
           - Hybrid architectures: Combining reactive + deliberative
             * InteRRaP: 3-layer (reactive, planning, cooperative)
             * TouringMachines: vertical vs horizontal composition
             * Trade-offs: responsiveness vs rationality
           - Learning agents: Integration with ML/RL
             * Perception-action loops with learning
             * Examples: DeepMind agents, robotic learning

        2. COMMUNICATION TOPOLOGIES (500 words):
           - Centralized: hub-spoke, master-slave
             * Advantages: simple coordination, guaranteed consistency
             * Disadvantages: single point of failure, scalability limits
             * Examples: client-server, hierarchical control
           - Decentralized: peer-to-peer, fully connected
             * Advantages: robustness, no bottleneck
             * Disadvantages: message complexity O(n²), coordination difficulty
             * Examples: BitTorrent, blockchain, gossip protocols
           - Hierarchical: tree structures, command chains
             * Multi-level coordination
             * Examples: military organizations, corporate structures
           - Dynamic/Adaptive: topology changes with context
             * Self-organization principles
             * Examples: mobile ad-hoc networks, swarms

        3. MEMORY SYSTEMS (500 words):
           - Episodic memory: event sequences, temporal patterns
             * Storage: time-indexed experiences
             * Retrieval: similarity search, sequence matching
             * Applications: case-based reasoning, learning from history
           - Semantic memory: facts, concepts, knowledge graphs
             * Representation: ontologies, description logics
             * Reasoning: inference, deduction, abduction
             * Examples: Cyc, WordNet, knowledge bases
           - Procedural memory: skills, habits, policies
             * Compiled knowledge for fast execution
             * Examples: learned motor skills, cached strategies
           - Shared memory: coordination through common state
             * Blackboard systems: knowledge sources + control
             * Tuple spaces: Linda model, JavaSpaces
             * Challenges: consistency, concurrency, synchronization

        4. COORDINATION MECHANISMS (600 words):
           - Negotiation protocols:
             * Alternating offers (Rubinstein bargaining)
             * Auctions (English, Dutch, sealed-bid, Vickrey)
             * Game theory: Nash equilibrium, mechanism design
             * Examples: automated trading, resource allocation
           - Contract Net Protocol:
             * Task announcement → bidding → award → execution
             * Extensions: CNP with deadlines, iterated CNP
             * Applications: task allocation in robotics
           - Voting and consensus:
             * Byzantine agreement: tolerating faulty agents
             * Paxos: majority-based consensus
             * Raft: leader election, log replication
             * Blockchain consensus: proof-of-work, proof-of-stake
           - Swarm intelligence:
             * Ant colony optimization: pheromone trails
             * Particle swarm: social + cognitive learning
             * Stigmergy: indirect coordination via environment
             * Examples: optimization, routing, foraging

        5. LEARNING & ADAPTATION (300 words):
           - Multi-agent RL: non-stationarity challenges
             * Independent learners: treat others as environment
             * Joint action learners: model other agents
             * CTDE: centralized training, decentralized execution
           - Opponent modeling: predicting others' behaviors
             * Explicit models: Bayesian, game-theoretic
             * Implicit learning: neural networks
             * Theory of mind in agents
           - Emergent communication:
             * Learning to communicate without predefined language
             * Examples: DeepMind's communication games

        For EACH topic:
        - Provide mathematical formulations where relevant
        - Include pseudocode for key algorithms
        - Compare different approaches with pros/cons tables
        - Reference specific implementations and frameworks
        - Discuss scalability, complexity, and practical considerations

        Write as cohesive technical narrative with smooth transitions.
        Total: 2500+ words minimum.
        """
    )
    print("✅ Customized 'analysis' for deep technical content (6000 tokens)")

    # Customize synthesis for integration
    workflow.customize_stage(
        "synthesis",
        max_tokens=5000,
        additional_context="""
        COMPREHENSIVE SYNTHESIS REQUIREMENTS:

        Generate 2000+ words synthesizing ALL findings:

        1. INTEGRATION OF CONCEPTS (500 words):
           How do agent types, topologies, memory, and coordination interact?
           - Which agent types work best with which topologies?
           - How does memory architecture affect coordination strategy?
           - What are the emergent properties from component combinations?
           - Provide concrete examples of integrated systems

        2. COMPARATIVE ANALYSIS (500 words):
           Compare different design choices:
           - Reactive vs deliberative: when to use each?
           - Centralized vs decentralized: scalability trade-offs
           - Learning vs hard-coded: adaptability vs reliability
           - Create decision matrices for system designers

        3. STATE OF THE ART (2023-2026) (400 words):
           Recent breakthroughs and trends:
           - Deep RL in multi-agent settings (AlphaStar, DOTA)
           - Large language models as agents (AutoGPT, BabyAGI)
           - Neurosymbolic approaches
           - Quantum multi-agent systems
           - Edge AI and federated learning

        4. RESEARCH GAPS & CHALLENGES (300 words):
           What remains unsolved?
           - Scalability to 1000+ agents
           - Handling non-stationary environments
           - Explainability and interpretability
           - Safety and robustness guarantees
           - Ethical AI and fairness

        5. FUTURE DIRECTIONS (300 words):
           Where is the field heading?
           - Integration with neuroscience (brain-inspired MAS)
           - Quantum coordination protocols
           - Human-AI teaming and collaboration
           - Self-evolving multi-agent ecosystems

        Write as narrative essay with clear argument flow.
        Use subheadings, bullet points, and emphasis for readability.
        Total: 2000+ words minimum.
        """
    )
    print("✅ Customized 'synthesis' for comprehensive integration (5000 tokens)")

    # Customize documentation for publication quality
    workflow.customize_stage(
        "documentation",
        max_tokens=8000,  # Maximum for full paper
        additional_context="""
        PUBLICATION-READY ACADEMIC PAPER:

        Generate COMPLETE 4000+ word academic paper (NOT outline):

        FORMAT AS FINAL PUBLISHED DOCUMENT:
        - NO meta-commentary ("I'll help you...", "Would you like me to...")
        - Write as if this is the camera-ready version
        - Include ALL sections with full content

        STRUCTURE:

        1. TITLE & ABSTRACT (300 words):
           - Compelling technical title
           - Comprehensive abstract covering all findings
           - 5-7 keywords

        2. INTRODUCTION (600 words):
           - Motivation: Why multi-agent systems matter
           - Historical context and evolution
           - Research questions addressed
           - Contributions of this survey
           - Paper organization

        3. BACKGROUND (400 words):
           - Foundational concepts
           - Historical development
           - Key terminology and definitions

        4. MAIN CONTENT (2000 words):
           Integrate ALL previous stages:
           - Agent Types & Architectures (from analysis)
           - Communication Topologies (from analysis)
           - Memory Systems (from analysis)
           - Coordination Mechanisms (from analysis)
           - Learning & Adaptation (from analysis)
           Use full content from analysis stage, don't summarize.

        5. APPLICATIONS (400 words):
           Real-world case studies:
           - Smart cities (Singapore, Barcelona)
           - Multi-robot warehouses (Amazon, Ocado)
           - Trading agents (financial markets)
           - Game AI (AlphaStar, OpenAI Five)
           Include metrics, performance data, lessons learned.

        6. DISCUSSION (500 words):
           - Key findings and insights
           - Implications for research and practice
           - Limitations of current approaches
           - Comparison with existing surveys
           - Threats to validity

        7. CONCLUSION (300 words):
           - Summary of contributions
           - Future research directions
           - Final thoughts on field trajectory

        8. REFERENCES (30+ sources):
           Full academic citations in format:
           Author(s). (Year). Title. Venue/Publisher. DOI/URL.

        WRITING STYLE:
        - Formal academic tone
        - Technical precision
        - Cite sources inline [Author, Year]
        - Use figures/tables (describe them textually)
        - Include equations for key algorithms
        - Write full paragraphs, not bullet points for main text

        CRITICAL: This must be the FINAL complete paper, not a template.
        Total length: 4000+ words minimum.
        """
    )
    print("✅ Customized 'documentation' for publication-ready paper (8000 tokens)")

    print()

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 3: Add Custom Deep-Dive Stages for Extra Depth
    # ═══════════════════════════════════════════════════════════════════════

    print("STEP 3: Add Custom Deep-Dive Stages")
    print("-" * 80 + "\n")

    # Custom stage: Real-world applications with metrics
    applications_swarms = SwarmAdapter.quick_swarms([
        ("Application Specialist", """
        Generate COMPLETE 1500+ word section on real-world multi-agent applications.

        NO meta-commentary. Write final content directly.

        Cover these domains in depth:

        1. SMART CITIES (400 words):
           - Traffic management: SCOOT (London), adaptive signals
             * Multi-agent traffic light coordination
             * Vehicle-to-infrastructure communication
             * Results: 20-30% congestion reduction
           - Energy grids: smart grid coordination
             * Distributed energy resource management
             * Demand response, peak shaving
             * Case study: Pecan Street Project (Austin)
           - Emergency response: agent-based simulation
             * Evacuation planning, resource allocation
             * Real-time coordination during disasters

        2. ROBOTICS (400 words):
           - Warehouse automation:
             * Amazon Kiva robots: 20,000+ agents
             * Collision avoidance, path planning
             * 2-3x productivity improvement
           - Ocado: grocery fulfillment
             * 3D grid navigation, task allocation
             * 65,000 orders/week per warehouse
           - Multi-drone coordination:
             * Intel's light shows: 500-2000 drones
             * Swarm algorithms, collision avoidance
             * Applications: delivery, surveillance, search & rescue

        3. GAME AI (400 words):
           - AlphaStar (StarCraft II):
             * League training: self-play with diverse strategies
             * 600 years of game time in 14 days
             * Grandmaster level (99.8 percentile)
           - OpenAI Five (Dota 2):
             * 5-agent coordination, 20,000 actions/game
             * Beat world champions (OG, Team Liquid)
             * Emergent strategies: smoke ganks, creep pulling
           - Hide and Seek (OpenAI):
             * Emergent tool use, cooperative behavior
             * 6 distinct strategic phases discovered

        4. FINANCIAL MARKETS (300 words):
           - High-frequency trading agents
           - Market making and liquidity provision
           - Flash crashes and systemic risk
           - Agent-based models for policy testing

        For EACH application:
        - Technical architecture details
        - Quantitative metrics and results
        - Lessons learned and challenges
        - Future directions

        Total: 1500+ words with specific examples and data.
        """)
    ], max_tokens=5000)

    workflow.add_custom_stage(
        "real_world_applications",
        swarms=applications_swarms,
        context_from=["analysis", "synthesis"]
    )
    print("✅ Added: Real-World Applications (5000 tokens, metrics & case studies)")

    print()
    workflow.show_pipeline()
    print()

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 4: Execute Comprehensive Research
    # ═══════════════════════════════════════════════════════════════════════

    print("STEP 4: Execute Comprehensive Research")
    print("-" * 80 + "\n")

    print("🚀 Starting comprehensive research...")
    print("   This will take 6-10 minutes due to depth...")
    print("   Expected: 10,000+ words, $0.03-0.05 cost")
    print()

    result = await workflow.run(verbose=True)

    print()
    print("✅ Research Complete!")
    print(f"   Stages: {len(result.stages)}")
    print(f"   Cost: ${result.total_cost:.6f}")
    print(f"   Time: {result.total_time:.2f}s")
    print()

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 5: Extract and Save Documentation
    # ═══════════════════════════════════════════════════════════════════════

    print("STEP 5: Extract Documentation")
    print("-" * 80 + "\n")

    # Find documentation stage
    doc_stage = next((s for s in result.stages if s.stage_name == "documentation"), None)
    if not doc_stage:
        print("❌ Documentation stage not found")
        return

    # Save markdown
    output_dir = Path.home() / "jotty" / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    markdown_path = output_dir / "multiagent_systems_comprehensive.md"
    with open(markdown_path, 'w', encoding='utf-8') as f:
        f.write(doc_stage.result.output)

    content_size = len(doc_stage.result.output)
    word_count = content_size // 5  # Rough estimate: 5 chars per word
    page_count = word_count // 500  # Rough estimate: 500 words per page

    print(f"✅ Saved markdown: {markdown_path}")
    print(f"   Size: {content_size:,} characters ({content_size/1024:.1f}KB)")
    print(f"   Words: ~{word_count:,} words")
    print(f"   Pages: ~{page_count} pages")
    print()

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 6: Generate PDF (Primary Output Format)
    # ═══════════════════════════════════════════════════════════════════════

    print("STEP 6: Generate PDF Output")
    print("-" * 80 + "\n")

    format_manager = OutputFormatManager(output_dir=str(output_dir))

    print("🔄 Generating PDF...")
    pdf_result = format_manager.generate_pdf(
        markdown_path=str(markdown_path),
        title="Multi-Agent Systems: Comprehensive Technical Survey",
        author="Jotty Research AI",
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
    # STEP 7: Send to Telegram
    # ═══════════════════════════════════════════════════════════════════════

    print("STEP 7: Send to Telegram")
    print("-" * 80 + "\n")

    channel_manager = OutputChannelManager()

    telegram_caption = f"""📚 <b>Multi-Agent Systems: Comprehensive Survey</b>

<b>Publication-Quality Research Paper</b>

📄 {len(result.stages)} research stages
📊 ~{word_count:,} words (~{page_count} pages)
💰 ${result.total_cost:.6f}

<b>Comprehensive Coverage:</b>
✅ Agent Architectures (Reactive, BDI, Hybrid)
✅ Communication Topologies
✅ Memory Systems
✅ Coordination Mechanisms
✅ Learning & Adaptation
✅ Real-World Applications
✅ 30+ Academic Citations

Generated by Jotty AI Research Framework"""

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
    # STEP 8: Send to WhatsApp #my-notes via Bailey
    # ═══════════════════════════════════════════════════════════════════════

    print("STEP 8: Send to WhatsApp #my-notes via Bailey")
    print("-" * 80 + "\n")

    # Get WhatsApp channel from environment (default configured)
    whatsapp_to = os.environ.get('WHATSAPP_CHANNEL') or os.environ.get('WHATSAPP_TO')
    whatsapp_result = None

    if whatsapp_to:
        print(f"📤 Sending to WhatsApp: {whatsapp_to}")

        whatsapp_caption = f"""📚 *Multi-Agent Systems: Comprehensive Survey*

*Publication-Quality Research Paper*

📄 {len(result.stages)} research stages
📊 ~{word_count:,} words (~{page_count} pages)
💰 ${result.total_cost:.6f}

*Comprehensive Coverage:*
✅ Agent Architectures
✅ Communication Topologies
✅ Memory Systems
✅ Coordination Mechanisms
✅ Learning & Adaptation
✅ Real-World Applications
✅ 30+ Academic Citations

Generated by Jotty AI"""

        whatsapp_result = await channel_manager.send_to_whatsapp(
            to=whatsapp_to,
            file_path=file_to_send,
            caption=whatsapp_caption,
            provider="baileys"  # Explicitly use Bailey
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
        print("   Edit .env.anthropic and set:")
        print("   WHATSAPP_CHANNEL='your-group-jid@g.us'")

    print()

    # ═══════════════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ═══════════════════════════════════════════════════════════════════════

    print("="*80)
    print("COMPREHENSIVE MULTI-AGENT RESEARCH COMPLETE")
    print("="*80 + "\n")

    # Delivery summary
    channels_sent = []
    if telegram_result.success:
        channels_sent.append("Telegram")
    if whatsapp_result and whatsapp_result.success:
        channels_sent.append("WhatsApp (#my-notes via Bailey)")

    print("📤 Channel Delivery:")
    if channels_sent:
        for ch in channels_sent:
            print(f"   ✅ {ch}")
    else:
        print("   ⚠️  No channels delivered (check configuration)")
    print()

    print(f"📊 Content Generated:")
    print(f"   Words: ~{word_count:,}")
    print(f"   Pages: ~{page_count}")
    print(f"   Format: {file_format}")
    print()

    print(f"💰 Total Cost: ${result.total_cost:.6f}")
    print(f"⏱️  Total Time: {result.total_time:.0f}s")
    print()

    print("📁 Files saved to:")
    print(f"   {output_dir}/")
    if pdf_result.success:
        print(f"   - {Path(pdf_result.file_path).name} ✅ PRIMARY")
    print(f"   - {markdown_path.name}")
    print()

    print("🎊 Publication-quality comprehensive research complete!")
    print()


if __name__ == '__main__':
    asyncio.run(main())
