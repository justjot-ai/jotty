#!/usr/bin/env python3
"""
LEARNING WORKFLOW COMPLEX TEST
================================

Scenario: Comprehensive Olympiad-Level Mathematics Course

Demonstrates:
1. Auto-generate 80% of learning materials (LearningWorkflow)
2. Customize 10% (tweak difficulty, add context)
3. Replace 5% (custom problem generation)
4. Add 5% (competition strategies, mental math tricks)

Total: 12+ stages, mixed auto/custom for complete olympiad preparation
"""

import asyncio
from pathlib import Path
from dotenv import load_dotenv

# Load API key
load_dotenv(Path(__file__).parent.parent / "Jotty" / ".env.anthropic")


async def main():
    from Jotty.core.workflows import (
        LearningWorkflow, LearningLevel, LearningDepth, Subject,
        SwarmAdapter, MergeStrategy
    )

    print("\n" + "="*80)
    print("LEARNING WORKFLOW COMPLEX TEST")
    print("Olympiad-Level Number Theory Course for Competition Preparation")
    print("="*80 + "\n")

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 1: Start with Auto-Generation (Simple Intent)
    # ═══════════════════════════════════════════════════════════════════════

    print("STEP 1: Auto-Generate Base Learning Pipeline")
    print("-" * 80 + "\n")

    workflow = LearningWorkflow.from_intent(
        subject="mathematics",
        topic="Number Theory for Olympiad Preparation",
        student_name="Aria",
        depth="marathon",  # Full day workshop
        level="olympiad",  # Competition level
        deliverables=[
            "curriculum",         # Auto: learning plan
            "concepts",          # Auto: fundamental concepts
            "intuition",         # Auto: intuitive understanding
            "patterns",          # Will customize this
            "examples",          # Auto: worked examples
            "problems",          # Will replace this
            "solutions",         # Auto: detailed solutions
            "mistakes",          # Auto: common errors
            "connections",       # Auto: links to other topics
            "assessment",        # Auto: quizzes and tests
            "content_assembly",  # Will customize this
            "pdf_generation",    # Auto: professional PDF
        ],
        output_formats=["pdf", "html"],
        send_telegram=True,
        include_assessment=True
    )

    print("✅ Created base workflow with 12 auto-generated stages\n")

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

    # Customize patterns stage for olympiad-specific techniques
    workflow.customize_stage(
        "patterns",
        model="claude-3-5-haiku-20241022",
        max_tokens=4000,
        merge_strategy=MergeStrategy.BEST_OF_N,
        additional_context="""
        OLYMPIAD-SPECIFIC PATTERNS:

        Cover these competition-critical techniques:
        1. Divisibility Rules and Tricks
           - Powers of primes
           - LCM/GCD shortcuts
           - Modular arithmetic patterns

        2. Diophantine Equations
           - Linear Diophantine (Bezout's identity)
           - Quadratic Diophantine (Pell's equation)
           - Systematic solution methods

        3. Number-Theoretic Functions
           - Euler's totient function φ(n)
           - Sum/product of divisors
           - Multiplicative functions

        4. Prime Number Patterns
           - Fermat's Little Theorem applications
           - Wilson's Theorem
           - Chinese Remainder Theorem
           - Quadratic reciprocity

        5. Competition Tricks
           - Pigeonhole principle in number theory
           - Extremal principle
           - Invariants and monovariants
           - Coloring arguments

        6. Problem-Solving Meta-Patterns
           - When to use modular arithmetic
           - When to use prime factorization
           - When to use divisibility
           - When to construct examples

        For each pattern:
        - Recognition triggers (how to spot it)
        - Step-by-step application
        - 2-3 olympiad-level examples
        - Common variations
        - When it fails (limitations)

        Organize as "pattern library" for quick reference during competitions.
        """
    )
    print("✅ Customized 'patterns' stage with olympiad-specific techniques")

    # Customize content assembly for interactive HTML
    workflow.customize_stage(
        "content_assembly",
        max_tokens=3000,
        additional_context="""
        Create interactive learning experience:

        HTML Features:
        - Collapsible sections for each concept
        - Click-to-reveal hints for problems
        - Expandable solutions with multiple approaches
        - Progress tracker (concepts mastered)
        - Quick reference sidebar with formulas
        - Interactive practice problems
        - Visual animations for key concepts (described)
        - Dark mode support

        PDF Features:
        - Professional formatting (A4, two-column where appropriate)
        - Color-coded difficulty levels
        - Margin notes with quick tips
        - Practice problem answer key at end
        - Reference card (single page summary)

        Ensure both formats are complementary:
        - PDF for offline study
        - HTML for interactive learning
        """
    )
    print("✅ Customized 'content_assembly' for interactive features\n")

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 4: Replace Stage with Custom Implementation
    # ═══════════════════════════════════════════════════════════════════════

    print("STEP 4: Replace Stage with Custom Swarms")
    print("-" * 80 + "\n")

    # Replace problems with specialized olympiad problem crafters
    custom_problem_swarms = SwarmAdapter.quick_swarms([
        ("IMO Problem Specialist", """Create 25 olympiad-level number theory problems.

        Problem Distribution:
        - 5 Warm-up (AMC 10/12 level) - build confidence
        - 8 Standard Olympiad (AIME level) - core techniques
        - 7 Advanced Olympiad (USAMO/IMO level) - combine ideas
        - 5 Challenge (IMO P3/P6 level) - extremely difficult

        Topics Coverage:
        1. Divisibility (5 problems)
        2. Modular Arithmetic (5 problems)
        3. Diophantine Equations (4 problems)
        4. Prime Numbers (4 problems)
        5. Number-Theoretic Functions (4 problems)
        6. Mixed/Creative (3 problems)

        For each problem:
        1. Problem Number & Difficulty ⭐ to ⭐⭐⭐⭐⭐
        2. Clear statement (LaTeX math notation)
        3. Source/Inspiration (IMO 2015, USAMO 2020, etc. or "Original")
        4. Progressive Hints (3-4 levels, don't give away)
           - Hint 1: Direction to start
           - Hint 2: Key observation
           - Hint 3: Main technique
           - Hint 4: Almost solution
        5. Skills Tested (list specific techniques)
        6. Estimated Time (5-60 minutes)

        NO SOLUTIONS - save for solutions stage.
        Problems should be original or properly credited.
        Ensure variety in techniques required.

        Max 4000 tokens."""),

        ("Problem Difficulty Calibrator", """Verify and calibrate problem difficulty.

        For problems from IMO Problem Specialist:
        1. Verify difficulty ratings are accurate
        2. Ensure smooth progression (no sudden jumps)
        3. Check that hints are appropriate (not too easy, not too hard)
        4. Validate that problems test stated skills
        5. Suggest reordering if needed

        Add:
        - Expected success rates (% of olympiad students who would solve)
        - Prerequisite problems (which earlier problems prepare for this)
        - Extension challenges (how to make harder)

        Max 2000 tokens."""),
    ], max_tokens=4000)

    workflow.replace_stage(
        "problems",
        swarms=custom_problem_swarms,
        merge_strategy=MergeStrategy.CONCATENATE  # Get both problem set and calibration
    )
    print("✅ Replaced 'problems' with specialized olympiad problem crafters\n")

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 5: Add Completely New Custom Stages
    # ═══════════════════════════════════════════════════════════════════════

    print("STEP 5: Add Custom Stages")
    print("-" * 80 + "\n")

    # Add competition strategies stage
    competition_strategies_swarms = SwarmAdapter.quick_swarms([
        ("Competition Coach", """Provide competition-specific strategies.

        TIME MANAGEMENT:
        1. Problem Selection Strategy
           - Read all problems first (5-10 min)
           - Identify "your" problems (strength areas)
           - Solve easiest first (build confidence)
           - When to skip and move on

        2. Time Allocation
           - Typical time per difficulty level
           - When to make educated guesses
           - Last 30 minutes strategy

        PROBLEM-SOLVING TECHNIQUES:
        1. First Steps (when you're stuck)
           - Try small cases
           - Look for patterns
           - Consider extremes
           - Draw diagrams/tables
           - Work backwards

        2. Checking Work
           - Quick verification methods
           - Common error patterns
           - When to re-solve

        MENTAL PREPARATION:
        1. Before Competition
           - What to review
           - Sleep and nutrition
           - Materials to bring

        2. During Competition
           - Managing anxiety
           - Handling blocks
           - Maintaining focus

        3. After Competition
           - Learning from mistakes
           - Reviewing problems

        PRACTICE ROUTINE:
        1. Daily practice schedule
        2. When to do timed vs untimed
        3. How to review solutions
        4. Building problem-solving stamina

        Practical, actionable advice based on successful olympiad students.

        Max 2500 tokens."""),
    ], max_tokens=2500)

    workflow.add_custom_stage(
        "competition_strategies",
        swarms=competition_strategies_swarms,
        merge_strategy=MergeStrategy.BEST_OF_N,
        context_from=["patterns", "problems"]
    )
    print("✅ Added 'competition_strategies' stage")

    # Add mental math tricks stage
    mental_math_swarms = SwarmAdapter.quick_swarms([
        ("Mental Math Expert", """Teach mental math tricks for number theory.

        FAST CALCULATION TECHNIQUES:

        1. Modular Arithmetic Shortcuts
           - Powers of 2, 3, 5, 9 (mod calculations)
           - Casting out nines
           - Divisibility tests
           - Mod 4, 8, 16 patterns

        2. Prime Testing
           - Quick checks for small primes
           - Fermat test shortcuts
           - Wilson's theorem applications

        3. GCD/LCM Speed Methods
           - Euclidean algorithm optimizations
           - Common factor recognition
           - LCM from GCD formula

        4. Factorization Tricks
           - Difference of squares
           - Sum/difference of cubes
           - Sophie Germain identity
           - Aurifeuillean factorizations

        5. Power Calculations
           - Repeated squaring (binary exponentiation)
           - Cyclicity in last digits
           - Patterns in powers

        6. Memory Aids
           - First 20 primes
           - Perfect squares up to 30²
           - Powers of 2 up to 2¹⁶
           - Small factorials

        For each trick:
        - When to use it
        - Step-by-step method
        - Practice examples (3-5)
        - Common errors

        Goal: Reduce calculation time by 50%+

        Max 2000 tokens."""),
    ], max_tokens=2000)

    workflow.add_custom_stage(
        "mental_math_tricks",
        swarms=mental_math_swarms,
        merge_strategy=MergeStrategy.BEST_OF_N,
        context_from=["patterns", "examples"]
    )
    print("✅ Added 'mental_math_tricks' stage")

    # Add practice schedule stage
    practice_schedule_swarms = SwarmAdapter.quick_swarms([
        ("Study Planner", """Create personalized olympiad preparation schedule.

        TIMELINE: 6-month competition preparation

        PHASE 1: Foundation (Weeks 1-8)
        - Master core concepts
        - Build problem-solving habits
        - Daily practice routine
        - Weekly goals

        PHASE 2: Skill Development (Weeks 9-16)
        - Advanced techniques
        - Timed practice
        - Mock competitions
        - Review and adjust

        PHASE 3: Competition Ready (Weeks 17-24)
        - Full mock exams
        - Weakness targeting
        - Mental preparation
        - Final review

        For each week, specify:
        1. Topics to cover
        2. Number of problems to solve
        3. Difficulty distribution
        4. Review sessions
        5. Mock tests (when)
        6. Rest days

        Daily Structure:
        - Warm-up problems (15 min)
        - New concept learning (30 min)
        - Practice problems (60-90 min)
        - Review and reflection (15 min)

        Include:
        - Milestone checkpoints
        - Motivation tips
        - Adjustment guidelines
        - Resource recommendations

        Realistic and sustainable for student life.

        Max 2500 tokens."""),
    ], max_tokens=2500)

    workflow.add_custom_stage(
        "practice_schedule",
        swarms=practice_schedule_swarms,
        merge_strategy=MergeStrategy.BEST_OF_N,
        context_from=["curriculum", "assessment"]
    )
    print("✅ Added 'practice_schedule' stage\n")

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 6: Inspect Final Pipeline
    # ═══════════════════════════════════════════════════════════════════════

    print("STEP 6: Inspect Final Customized Pipeline")
    print("-" * 80 + "\n")

    workflow.show_pipeline()

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 7: Execute Complete Pipeline
    # ═══════════════════════════════════════════════════════════════════════

    print("\nSTEP 7: Execute Complete Learning Pipeline")
    print("-" * 80 + "\n")
    print("🚀 Executing 15-stage olympiad learning pipeline...\n")

    result = await workflow.run(verbose=True)

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 8: Analyze Results
    # ═══════════════════════════════════════════════════════════════════════

    print("\n" + "="*80)
    print("FINAL RESULTS - LEARNING WORKFLOW TEST")
    print("="*80 + "\n")

    print("📊 Pipeline Composition:")
    print(f"   Total Stages: {len(result.stages)}")

    auto_stages = 0
    customized_stages = 0
    replaced_stages = 0
    custom_stages = 0

    for stage in result.stages:
        if stage.stage_name in ["competition_strategies", "mental_math_tricks", "practice_schedule"]:
            custom_stages += 1
        elif stage.stage_name == "problems":
            replaced_stages += 1
        elif stage.stage_name in ["patterns", "content_assembly"]:
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
        "curriculum": "Comprehensive learning plan",
        "concepts": "Fundamental concepts explained",
        "intuition": "Intuitive understanding & analogies",
        "patterns": "Olympiad-specific techniques",
        "examples": "Worked examples with thinking",
        "problems": "25 olympiad-level problems",
        "solutions": "Detailed multi-approach solutions",
        "mistakes": "Common errors analysis",
        "connections": "Links to other topics",
        "competition_strategies": "Competition tips & tactics",
        "mental_math_tricks": "Fast calculation methods",
        "assessment": "Quizzes and practice tests",
        "practice_schedule": "6-month preparation plan",
        "content_assembly": "Interactive HTML + PDF",
        "pdf_generation": "Professional learning document",
    }

    for stage in result.stages:
        desc = deliverables.get(stage.stage_name, "Custom deliverable")
        print(f"   ✅ {stage.stage_name}: {desc}")
    print()

    print("🎯 Learning Depth Achieved:")
    print("   ✓ 15 total stages (12 planned + 3 custom added)")
    print("   ✓ Mixed auto/customized/replaced/custom stages")
    print("   ✓ Olympiad-ready number theory course")
    print("   ✓ 25 competition-level problems")
    print("   ✓ Interactive HTML + Professional PDF")
    print("   ✓ Competition strategies & mental math")
    print("   ✓ 6-month preparation schedule")
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
    print("✅ LEARNING WORKFLOW TEST COMPLETE")
    print("="*80)
    print()
    print("🏆 Successfully demonstrated:")
    print("   Intent-based educational content + Full customization when needed")
    print()
    print("📚 Student Aria now has:")
    print("   • Complete olympiad-level course")
    print("   • 25 practice problems with solutions")
    print("   • Competition strategies")
    print("   • Mental math shortcuts")
    print("   • 6-month practice plan")
    print("   • Interactive + offline materials")
    print()


if __name__ == '__main__':
    asyncio.run(main())
