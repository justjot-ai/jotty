#!/usr/bin/env python3
"""
LearningWorkflow - Intent-Based Educational Content Pipeline
=============================================================

Automatically creates comprehensive learning materials for any subject
from K-12 to Olympiad level.

Usage:
    # Simplest - just provide subject and topic
    result = await learn("mathematics", "Number Theory", student_name="Aria")

    # With depth and level control
    result = await learn(
        subject="physics",
        topic="Quantum Mechanics",
        student_name="Alex",
        depth="deep",
        level="olympiad"
    )

    # Full customization
    workflow = LearningWorkflow.from_intent(
        subject="mathematics",
        topic="Combinatorics",
        student_name="Aria",
        depth="marathon",
        level="olympiad",
        deliverables=["concepts", "patterns", "problems", "solutions", "practice"]
    )
    workflow.customize_stage("problems", max_tokens=4000)
    result = await workflow.run()
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
from ..orchestration import (
    MultiStagePipeline,
    SwarmAdapter,
    MergeStrategy,
    PipelineResult,
)


class Subject(Enum):
    """Learning subjects."""
    MATHEMATICS = "mathematics"
    PHYSICS = "physics"
    CHEMISTRY = "chemistry"
    COMPUTER_SCIENCE = "computer_science"
    BIOLOGY = "biology"
    ASTRONOMY = "astronomy"
    ECONOMICS = "economics"
    GENERAL = "general"


class LearningLevel(Enum):
    """Difficulty levels."""
    FOUNDATION = "foundation"      # K-5 (ages 5-10)
    INTERMEDIATE = "intermediate"  # 6-8 (ages 11-13)
    ADVANCED = "advanced"          # 9-12 (ages 14-18)
    OLYMPIAD = "olympiad"          # Competition level
    UNIVERSITY = "university"      # Undergraduate
    RESEARCH = "research"          # Graduate/Research


class LearningDepth(Enum):
    """Content depth levels."""
    QUICK = "quick"              # 15-30 min lesson
    STANDARD = "standard"        # 1-2 hour lesson
    DEEP = "deep"               # 3-5 hour comprehensive
    MARATHON = "marathon"        # Full day workshop


@dataclass
class LearningIntent:
    """High-level learning intent."""
    subject: str
    topic: str
    student_name: str
    depth: Optional[str] = "standard"
    level: Optional[str] = "intermediate"
    deliverables: Optional[List[str]] = None
    output_formats: Optional[List[str]] = None  # ["pdf", "html", "interactive"]
    send_telegram: bool = True
    include_assessment: bool = True


class LearningWorkflow:
    """
    Automatically creates comprehensive learning materials.

    Learning Pipeline Stages:
    1. Curriculum Planning - Define learning objectives, prerequisites
    2. Concept Breakdown - Decompose topic into teachable units
    3. Intuition Building - Build intuitive understanding
    4. Pattern Recognition - Identify key patterns and techniques
    5. Problem Crafting - Create practice problems (easy → hard)
    6. Solution Strategies - Detailed solutions with multiple approaches
    7. Common Mistakes - Identify and explain common errors
    8. Connections - Link to other topics and real-world applications
    9. Assessment - Quizzes and tests
    10. Interactive Content - HTML with visualizations
    11. PDF Generation - Professional document
    """

    def __init__(self, intent: LearningIntent):
        """Initialize learning workflow."""
        self.intent = intent
        self.pipeline = None
        self.stage_configs = {}

    @classmethod
    def from_intent(
        cls,
        subject: str,
        topic: str,
        student_name: str,
        depth: Optional[str] = "standard",
        level: Optional[str] = "intermediate",
        deliverables: Optional[List[str]] = None,
        output_formats: Optional[List[str]] = None,
        send_telegram: bool = True,
        include_assessment: bool = True
    ) -> "LearningWorkflow":
        """
        Create learning workflow from intent (without executing).

        Args:
            subject: Subject area
            topic: Specific topic to teach
            student_name: Student's name for personalization
            depth: Content depth (quick/standard/deep/marathon)
            level: Difficulty level (foundation/intermediate/advanced/olympiad)
            deliverables: What to generate (optional)
            output_formats: Output formats (pdf, html, interactive)
            send_telegram: Send results via Telegram
            include_assessment: Include quizzes/tests

        Returns:
            LearningWorkflow instance (not yet executed)
        """
        intent = LearningIntent(
            subject=subject,
            topic=topic,
            student_name=student_name,
            depth=depth,
            level=level,
            deliverables=deliverables,
            output_formats=output_formats or ["pdf", "html"],
            send_telegram=send_telegram,
            include_assessment=include_assessment
        )
        return cls(intent)

    @classmethod
    async def execute(
        cls,
        subject: str,
        topic: str,
        student_name: str,
        depth: Optional[str] = "standard",
        level: Optional[str] = "intermediate",
        deliverables: Optional[List[str]] = None,
        output_formats: Optional[List[str]] = None,
        send_telegram: bool = True,
        include_assessment: bool = True,
        verbose: bool = True
    ) -> PipelineResult:
        """
        Execute learning workflow from high-level intent (simplest API).

        Args:
            subject: Subject area
            topic: Topic to teach
            student_name: Student's name
            depth: Content depth
            level: Difficulty level
            deliverables: What to generate
            output_formats: Output formats
            send_telegram: Send via Telegram
            include_assessment: Include quizzes
            verbose: Print progress

        Returns:
            PipelineResult with all outputs
        """
        intent = LearningIntent(
            subject=subject,
            topic=topic,
            student_name=student_name,
            depth=depth,
            level=level,
            deliverables=deliverables,
            output_formats=output_formats or ["pdf", "html"],
            send_telegram=send_telegram,
            include_assessment=include_assessment
        )

        workflow = cls(intent)
        workflow.build_pipeline()
        return await workflow.run(verbose=verbose)

    def build_pipeline(self):
        """Build pipeline from learning intent."""
        deliverables = self.intent.deliverables or self._infer_deliverables()

        self.pipeline = MultiStagePipeline(task=f"Learning: {self.intent.topic} for {self.intent.student_name}")

        self._add_learning_stages(deliverables)

    def _infer_deliverables(self) -> List[str]:
        """Infer deliverables from depth and level."""
        depth = self.intent.depth
        level = self.intent.level

        # Base deliverables
        base = ["curriculum", "concepts", "intuition", "patterns"]

        # Add based on depth
        if depth == "quick":
            base.extend(["examples", "summary"])
        elif depth == "standard":
            base.extend(["examples", "problems", "solutions"])
        elif depth == "deep":
            base.extend(["examples", "problems", "solutions", "mistakes", "connections"])
        else:  # marathon
            base.extend(["examples", "problems", "solutions", "mistakes", "connections", "advanced_topics"])

        # Add assessment for intermediate and above
        if level in ["intermediate", "advanced", "olympiad", "university"]:
            if self.intent.include_assessment:
                base.append("assessment")

        # Always add final stages
        base.extend(["content_assembly", "pdf_generation"])

        return base

    def _add_learning_stages(self, deliverables: List[str]):
        """Add pipeline stages for learning deliverables."""

        # Determine problem counts based on depth
        depth_to_problems = {
            "quick": 5,
            "standard": 10,
            "deep": 20,
            "marathon": 50
        }
        num_problems = depth_to_problems.get(self.intent.depth, 10)

        # Build base context
        base_context = f"""
Subject: {self.intent.subject}
Topic: {self.intent.topic}
Student: {self.intent.student_name}
Level: {self.intent.level}
Depth: {self.intent.depth}

Teaching Philosophy:
- Build intuition before formalism
- Use concrete examples first
- Progress from simple to complex
- Connect to real-world applications
- Encourage pattern recognition
- Celebrate mistakes as learning opportunities
"""

        # Track previous stages
        previous_stages = []

        # Stage prompts
        stage_prompts = {
            "curriculum": ("Curriculum Architect", f"""Design comprehensive curriculum plan for this topic.

{base_context}

Create:
1. Learning Objectives (specific, measurable)
2. Prerequisites (what student should know)
3. Key Concepts (list in learning order)
4. Estimated Time for Each Section
5. Assessment Criteria

Output in structured markdown."""),

            "concepts": ("Concept Expert", f"""Break down the topic into fundamental concepts.

{base_context}

For each core concept:
1. Name & Definition (clear, age-appropriate)
2. Why It Matters (motivation)
3. Building Blocks (sub-concepts)
4. Key Properties/Rules
5. Visual Representation (describe diagram)

Make it crystal clear and engaging."""),

            "intuition": ("Intuition Builder", f"""Build deep intuitive understanding.

{base_context}

For each concept:
1. Concrete Analogy (relate to everyday life)
2. Visual Mental Model
3. "Aha!" Moment (key insight)
4. Common Misconceptions (what NOT to think)
5. Intuitive Explanation (before formal definition)

Use storytelling and metaphors. Make it memorable."""),

            "patterns": ("Pattern Hunter", f"""Identify key patterns and techniques.

{base_context}

Find:
1. Recurring Patterns (what keeps showing up)
2. Problem-Solving Techniques (when to use each)
3. Recognition Triggers (how to spot pattern)
4. Variations & Extensions
5. Power Moves (advanced tricks)

Create pattern recognition guide with examples."""),

            "examples": ("Example Creator", f"""Create illustrative examples.

{base_context}

Provide:
1. Worked Examples (5-7, increasing difficulty)
   - Problem statement
   - Thought process
   - Step-by-step solution
   - Key insights
2. Counter-Examples (common mistakes)
3. Edge Cases

Show the THINKING, not just steps."""),

            "problems": ("Problem Crafter", f"""Create {num_problems} practice problems.

{base_context}

Problem Distribution:
- {num_problems//3} Easy (build confidence)
- {num_problems//3} Medium (apply concepts)
- {num_problems//3} Hard (combine ideas)
- Remaining: Challenge/Olympiad level

For each problem:
1. Problem Number & Difficulty
2. Clear Statement
3. Hints (progressive, don't give away)
4. Learning Goal (what skill it develops)

No solutions here - only problems."""),

            "solutions": ("Solution Strategist", f"""Provide detailed solutions to all problems.

{base_context}

For each problem solution:
1. Multiple Approaches (at least 2 where possible)
2. Step-by-Step Reasoning
3. Key Insights & Techniques Used
4. Common Pitfalls to Avoid
5. Extensions (what if we change X?)

Explain WHY each step works, not just HOW."""),

            "mistakes": ("Mistake Analyzer", f"""Analyze common mistakes and misconceptions.

{base_context}

Identify:
1. Top 5-7 Common Mistakes
   - What students typically do wrong
   - Why it's wrong
   - Correct approach
   - How to remember the right way
2. Conceptual Misconceptions
3. Calculation Errors (typical traps)
4. Prevention Strategies

Turn mistakes into learning moments."""),

            "connections": ("Connection Mapper", f"""Map connections to other topics and real-world.

{base_context}

Show:
1. Prerequisites (what led to this topic)
2. Next Steps (where this leads)
3. Related Topics in Same Subject
4. Cross-Subject Connections
5. Real-World Applications (specific examples)
6. Historical Context (who/when/why)

Create concept map with clear links."""),

            "advanced_topics": ("Advanced Topics Guide", f"""Introduce advanced extensions.

{base_context}

For motivated students:
1. Advanced Theorems/Results
2. Open Problems (age-appropriate)
3. Research Directions
4. Competition-Level Techniques
5. Further Reading (books, papers, resources)

Inspire curiosity and deeper exploration."""),

            "assessment": ("Assessment Designer", f"""Create comprehensive assessment.

{base_context}

Design:
1. Quick Quiz (5 questions, 10 min)
   - Multiple choice
   - Auto-gradable
2. Practice Test (10 questions, 30 min)
   - Short answer
   - Apply concepts
3. Challenge Problems (3 problems, 60 min)
   - Open-ended
   - Require synthesis
4. Answer Keys for All

Include grading rubrics."""),

            "content_assembly": ("Content Assembler", f"""Assemble all content into cohesive learning module.

{base_context}

Create structured document:
1. Cover Page (topic, student name, date)
2. Table of Contents
3. Introduction & Learning Path
4. All Content Sections (in logical order)
5. Summary & Key Takeaways
6. Resources & Next Steps

Professional formatting. Clear navigation.
Output Format: {', '.join(self.intent.output_formats)}"""),

            "pdf_generation": ("PDF Generator", f"""Generate professional PDF document.

{base_context}

Specifications:
- Page Size: A4
- Font: Professional, readable
- Colors: Subject-appropriate
- Margins: Comfortable
- Headers/Footers: Topic, page numbers
- Sections: Clear hierarchy
- Images: Placeholder descriptions

Create publication-quality learning material."""),

            "summary": ("Summarizer", f"""Create concise summary.

{base_context}

One-page summary:
1. Key Concepts (bullet points)
2. Essential Formulas/Rules
3. Problem-Solving Checklist
4. Quick Reference Guide

Perfect for review before test."""),
        }

        # Add stages based on deliverables
        for deliverable in deliverables:
            stage_config = self.stage_configs.get(deliverable, {})

            # Check if stage is completely replaced
            if stage_config.get("replace"):
                swarms = stage_config["custom_swarms"]
                merge_strategy = stage_config.get("merge_strategy", MergeStrategy.BEST_OF_N)
                context_from = stage_config.get("context_from", previous_stages.copy() if previous_stages else None)
            else:
                # Auto-generate with prompts
                if deliverable not in stage_prompts:
                    print(f"⚠️  Unknown deliverable: {deliverable}, skipping")
                    continue

                agent_name, prompt = stage_prompts[deliverable]

                # Apply customizations
                if stage_config.get("additional_context"):
                    prompt = f"{prompt}\n\nAdditional Context:\n{stage_config['additional_context']}"

                # Create swarms
                swarms = SwarmAdapter.quick_swarms(
                    [(agent_name, prompt)],
                    model=stage_config.get("model", "claude-3-5-haiku-20241022"),
                    max_tokens=stage_config.get("max_tokens", 2500)  # Higher default for educational content
                )

                merge_strategy = stage_config.get("merge_strategy", MergeStrategy.BEST_OF_N)
                context_from = previous_stages.copy() if previous_stages else None

            # Add stage
            self.pipeline.add_stage(
                name=deliverable,
                swarms=swarms,
                merge_strategy=merge_strategy,
                context_from=context_from,
                max_context_chars=2500  # Educational content needs more context
            )

            previous_stages.append(deliverable)

        # Add any custom stages
        for stage_name, config in self.stage_configs.items():
            if config.get("custom_stage"):
                swarms = config["swarms"]
                merge_strategy = config.get("merge_strategy", MergeStrategy.BEST_OF_N)
                context_from = config.get("context_from")

                self.pipeline.add_stage(
                    name=stage_name,
                    swarms=swarms,
                    merge_strategy=merge_strategy,
                    context_from=context_from,
                    max_context_chars=2500
                )

    def customize_stage(
        self,
        stage_name: str,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        merge_strategy: Optional[MergeStrategy] = None,
        additional_context: Optional[str] = None
    ):
        """Customize a specific stage."""
        if stage_name not in self.stage_configs:
            self.stage_configs[stage_name] = {}

        if model:
            self.stage_configs[stage_name]["model"] = model
        if max_tokens:
            self.stage_configs[stage_name]["max_tokens"] = max_tokens
        if merge_strategy:
            self.stage_configs[stage_name]["merge_strategy"] = merge_strategy
        if additional_context:
            self.stage_configs[stage_name]["additional_context"] = additional_context

    def replace_stage(
        self,
        stage_name: str,
        swarms: List[Any],
        merge_strategy: Optional[MergeStrategy] = None,
        context_from: Optional[List[str]] = None
    ):
        """Completely replace a stage with custom swarms."""
        if stage_name not in self.stage_configs:
            self.stage_configs[stage_name] = {}

        self.stage_configs[stage_name]["replace"] = True
        self.stage_configs[stage_name]["custom_swarms"] = swarms
        if merge_strategy:
            self.stage_configs[stage_name]["merge_strategy"] = merge_strategy
        if context_from:
            self.stage_configs[stage_name]["context_from"] = context_from

    def add_custom_stage(
        self,
        stage_name: str,
        swarms: List[Any],
        position: Optional[int] = None,
        merge_strategy: MergeStrategy = MergeStrategy.BEST_OF_N,
        context_from: Optional[List[str]] = None
    ):
        """Add a completely custom stage."""
        self.stage_configs[stage_name] = {
            "custom_stage": True,
            "swarms": swarms,
            "position": position,
            "merge_strategy": merge_strategy,
            "context_from": context_from
        }

    def show_pipeline(self, verbose: bool = True):
        """Print pipeline structure."""
        deliverables = self.intent.deliverables or self._infer_deliverables()

        if verbose:
            print("\n" + "="*70)
            print("LEARNING PIPELINE INSPECTION")
            print("="*70)
            print(f"\n🎯 Topic: {self.intent.topic}")
            print(f"📚 Subject: {self.intent.subject}")
            print(f"👤 Student: {self.intent.student_name}")
            print(f"📊 Level: {self.intent.level}")
            print(f"📖 Depth: {self.intent.depth}")
            print(f"📄 Stages: {len(deliverables)}")
            print()

            for i, stage in enumerate(deliverables, 1):
                status = ""
                if self.stage_configs.get(stage, {}).get("replace"):
                    status = "🔧 REPLACED (custom swarms)"
                elif stage in self.stage_configs:
                    status = "⚙️  CUSTOMIZED (tweaked)"
                else:
                    status = "🤖 AUTO (built-in prompts)"

                print(f"{i}. {stage:<25} {status}")

            print("\n" + "="*70 + "\n")

    async def run(self, verbose: bool = True) -> PipelineResult:
        """Execute the learning pipeline."""
        if self.pipeline is None:
            self.build_pipeline()

        return await self.pipeline.execute(auto_trace=True, verbose=verbose)

    async def run_with_outputs(
        self,
        output_formats: Optional[List[str]] = None,
        output_dir: Optional[str] = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Execute pipeline and generate multiple output formats.

        Args:
            output_formats: List of formats (pdf, epub, html, docx, presentation)
            output_dir: Output directory (default: ~/jotty/outputs)
            verbose: Print progress

        Returns:
            Dict with 'pipeline_result' and 'outputs' (dict of OutputSinkResults)
        """
        # Execute pipeline
        pipeline_result = await self.run(verbose=verbose)

        # Use configured output formats if not specified
        formats = output_formats or self.intent.output_formats or ["pdf", "html"]

        # Generate outputs from content_assembly stage if it exists
        content_stage = None
        for stage in pipeline_result.stages:
            if stage.stage_name == "content_assembly":
                content_stage = stage
                break

        if not content_stage:
            # No content assembly stage, just return pipeline result
            return {
                'pipeline_result': pipeline_result,
                'outputs': {},
                'note': 'No content_assembly stage found - outputs not generated'
            }

        # Save markdown content to file
        from pathlib import Path
        import os

        output_path = Path(output_dir or os.path.expanduser("~/jotty/outputs"))
        output_path.mkdir(parents=True, exist_ok=True)

        # Create markdown file
        safe_topic = "".join(c for c in self.intent.topic if c.isalnum() or c in (' ', '-', '_')).strip()
        safe_topic = safe_topic.replace(' ', '_')[:50]
        markdown_path = str(output_path / f"learning_{self.intent.student_name}_{safe_topic}.md")

        with open(markdown_path, 'w', encoding='utf-8') as f:
            f.write(f"# {self.intent.topic}\n\n")
            f.write(f"**Student:** {self.intent.student_name}\n\n")
            f.write(f"**Subject:** {self.intent.subject} | **Level:** {self.intent.level} | **Depth:** {self.intent.depth}\n\n")
            f.write("---\n\n")
            f.write(content_stage.result.output)

        if verbose:
            print(f"\n📄 Saved markdown: {markdown_path}")

        # Check if we should build EPUB with chapters
        build_epub_chapters = "epub" in formats and len(pipeline_result.stages) > 3

        if build_epub_chapters:
            # Build chapter list from pipeline stages
            chapters = []
            for stage in pipeline_result.stages:
                if stage.stage_name not in ["pdf_generation", "content_assembly"]:
                    chapters.append({
                        'title': stage.stage_name.replace('_', ' ').title(),
                        'content': stage.result.output
                    })

        # Generate outputs using OutputFormatManager
        try:
            from .output_formats import OutputFormatManager

            manager = OutputFormatManager(output_dir=str(output_path))

            # Generate regular outputs
            outputs = manager.generate_all(
                markdown_path=markdown_path,
                formats=[fmt for fmt in formats if fmt not in ["markdown", "epub"]],
                title=f"{self.intent.topic} - {self.intent.student_name}",
                author="Jotty Learning Workflow",
                n_slides=15,  # For presentations
                tone="educational"
            )

            # Generate rich EPUB with chapters if requested
            if build_epub_chapters:
                epub_result = manager.generate_epub_with_chapters(
                    chapters=chapters,
                    title=self.intent.topic,
                    author=f"For {self.intent.student_name}",
                    description=f"{self.intent.subject} - {self.intent.level} level",
                    language="en"
                )
                outputs['epub'] = epub_result

            if verbose:
                summary = manager.get_summary(outputs)
                print(f"\n📦 Generated {summary['successful']}/{summary['total']} output formats:")
                for fmt in summary['successful_formats']:
                    print(f"   ✅ {fmt}: {summary['file_paths'][fmt]}")
                if summary['failed_formats']:
                    print(f"\n❌ Failed formats: {', '.join(summary['failed_formats'])}")

            return {
                'pipeline_result': pipeline_result,
                'outputs': outputs,
                'markdown_path': markdown_path
            }

        except Exception as e:
            if verbose:
                print(f"\n⚠️  Output generation failed: {e}")
            return {
                'pipeline_result': pipeline_result,
                'outputs': {},
                'markdown_path': markdown_path,
                'error': str(e)
            }


# Convenience functions
async def learn(
    subject: str,
    topic: str,
    student_name: str,
    **kwargs
) -> PipelineResult:
    """
    Simplest API - create learning materials.

    Usage:
        result = await learn("mathematics", "Number Theory", "Aria")
        result = await learn("physics", "Quantum Mechanics", "Alex", depth="deep", level="olympiad")
    """
    return await LearningWorkflow.execute(subject, topic, student_name, **kwargs)
