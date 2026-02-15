"""
SwarmScience — World-Class Science Teaching Template
=====================================================

A comprehensive swarm template for teaching any science topic from building
blocks to competition level (e.g. International Science Olympiads).

Design principles (built into prompts; no external style names):
- One concept at a time; intuition before formalism
- Clear structure: what we're learning, why it matters, then the idea, then connections
- Build from prerequisites → foundations → core → depth → practice
- Personalize with student name; outcome-focused explanations
- No jargon until defined; every claim tied to "so what"

Inputs: subject (e.g. Science), topic (e.g. Photosynthesis), student_name (e.g. Maya)

Usage:
    from Jotty.core.intelligence.orchestration.templates import SwarmScience
    t = SwarmScience()
    goal = t.build_goal(subject="Science", topic="Photosynthesis", student_name="Maya")
    # Then: await swarm.run(goal)
    # Or: Swarm.solve(template="science", subject="Science", topic="Photosynthesis", student_name="Maya")
"""

from typing import Any, Dict

from .base import AgentConfig, ModelTier, StageConfig, SwarmTemplate

# =============================================================================
# PEDAGOGY CONSTANTS (used in prompts)
# =============================================================================

STYLE_RULES = """
- Explain one idea at a time. Do not introduce a second idea until the first is clear.
- Start with intuition and everyday analogies; only then introduce formal terms and equations.
- For every concept, state clearly: (1) what we are learning, (2) why it matters, (3) the idea in plain language, (4) how it connects to what came before.
- Use a clear, friendly tone. Address the student by name. No filler; every sentence should earn its place.
- Define every technical term the first time you use it. Never assume prior jargon.
- End sections with a single-sentence takeaway and one "so what" (why this matters in the real world or in later topics).
"""

SCAFFOLD_RULES = """
- Order content from building blocks to advanced. List prerequisites explicitly before the main concept.
- Each section must depend only on ideas already introduced. If a concept needs X and Y, teach X and Y first.
- For competition/Olympiad depth: add a final section that extends the topic (harder applications, connections to other topics, or past Olympiad-style questions) without breaking the flow.
"""


# =============================================================================
# SWARM SCIENCE TEMPLATE
# =============================================================================


class SwarmScience(SwarmTemplate):
    """
    World-class science teaching template.

    Pipeline: Parse request → Prerequisites → Foundations → Core → Depth → Practice → Deliver.
    Prompts enforce intuition-first, one-concept-at-a-time, building-block-to-advanced pedagogy.
    """

    name = "SwarmScience"
    version = "1.0.0"
    description = (
        "World-class science teaching: building blocks to Olympiad-level, one concept at a time"
    )

    supported_problem_types = [
        "science_teaching",
        "science_explanation",
        "science_olympiad_prep",
        "science_topic",
    ]

    # ================================================================
    # AGENTS
    # ================================================================
    agents = {
        "teacher": AgentConfig(
            name="teacher",
            skills=[
                "web-search",
                "content-research-writer",
                "markdown-writer",
                "file-write",
                "file-read",
            ],
            model=ModelTier.BALANCED,
            max_concurrent=1,
            timeout=600,
            retry_count=2,
        ),
    }

    # ================================================================
    # PIPELINE
    # ================================================================
    pipeline = [
        StageConfig(
            name="PARSE_REQUEST",
            agents=["teacher"],
            parallel=False,
            inputs=["task", "subject", "topic", "student_name", "business_context"],
            outputs=["parsed_subject", "parsed_topic", "parsed_student", "scope", "prereqs_list"],
            weight=5,
            description="Parse subject, topic, student name and identify scope and prerequisites",
        ),
        StageConfig(
            name="PREREQUISITES",
            agents=["teacher"],
            parallel=False,
            inputs=["parsed_topic", "prereqs_list", "parsed_student"],
            outputs=["prereqs_content"],
            weight=15,
            description="Explain prerequisites from building blocks",
        ),
        StageConfig(
            name="FOUNDATIONS",
            agents=["teacher"],
            parallel=False,
            inputs=["parsed_topic", "prereqs_content", "parsed_student"],
            outputs=["foundations_content"],
            weight=20,
            description="Foundations: intuition and first principles",
        ),
        StageConfig(
            name="CORE",
            agents=["teacher"],
            parallel=False,
            inputs=["parsed_topic", "foundations_content", "parsed_student"],
            outputs=["core_content"],
            weight=25,
            description="Core concept in depth with clear structure",
        ),
        StageConfig(
            name="DEPTH",
            agents=["teacher"],
            parallel=False,
            inputs=["parsed_topic", "core_content", "parsed_student"],
            outputs=["depth_content"],
            weight=20,
            description="Extension to Olympiad/competition level",
        ),
        StageConfig(
            name="PRACTICE",
            agents=["teacher"],
            parallel=False,
            inputs=["parsed_topic", "core_content", "depth_content", "parsed_student"],
            outputs=["practice_content"],
            weight=10,
            description="Practice problems with solutions",
        ),
        StageConfig(
            name="DELIVER",
            agents=["teacher"],
            parallel=False,
            inputs=[
                "prereqs_content",
                "foundations_content",
                "core_content",
                "depth_content",
                "practice_content",
                "parsed_student",
                "parsed_topic",
            ],
            outputs=["final_module", "output_path"],
            weight=5,
            description="Assemble and save final learning module",
        ),
    ]

    feedback_config = None

    # ================================================================
    # LLM PROMPTS (pedagogy baked in)
    # ================================================================
    llm_prompts = {
        "parse_request": """You are designing a world-class learning module for one student.

Inputs:
- Subject: {subject}
- Topic: {topic}
- Student name: {student_name}

Tasks:
1. Identify the exact scope of the topic (what is in and out).
2. List prerequisites (building blocks) that the student must understand before this topic. Order them from most basic to most advanced.
3. Note any common misconceptions about this topic to address later.

Output a structured summary (you can use bullets or short paragraphs):
- Scope: ...
- Prerequisites (ordered): ...
- Misconceptions to address: ...

Keep it concise. This will guide the rest of the module.
"""
        + STYLE_RULES,
        "prerequisites": """You are writing the PREREQUISITES section of a learning module.

Topic (main): {topic}
Student name: {student_name}
Prerequisites to cover (in order): {prereqs_list}

For each prerequisite:
- Explain it in one short block (one concept at a time).
- Use intuition first, then a simple definition. No jargon before defining it.
- End with one sentence: why this matters for understanding "{topic}".

Rules:
"""
        + STYLE_RULES
        + SCAFFOLD_RULES
        + """
Write the full Prerequisites section. Use clear headings for each prerequisite. Address {student_name} by name once at the start.
""",
        "foundations": """You are writing the FOUNDATIONS section of a learning module.

Topic: {topic}
Student name: {student_name}
Context: The student has just read the prerequisites. Now we introduce the topic from first principles.

Tasks:
1. Start with a single big question or real-world hook (why should we care about this?).
2. Introduce the main idea using an everyday analogy or a minimal example first.
3. Only then introduce any formal terms or symbols. Define each term the first time.
4. Keep to one main idea per paragraph. End the section with a one-sentence takeaway and a "so what."

Rules:
"""
        + STYLE_RULES
        + SCAFFOLD_RULES
        + """
Write the full Foundations section. Use headings. Address {student_name} occasionally. No filler.
""",
        "core": """You are writing the CORE section of a learning module.

Topic: {topic}
Student name: {student_name}
Context: Foundations are done. Now go deep into the main concept.

Tasks:
1. Structure the core as: (a) What we are learning, (b) Why it matters, (c) The idea step by step, (d) How it connects to the foundations and to the real world.
2. One concept at a time. If you need to use an equation or a diagram description, introduce it only after the intuition.
3. Include one or two concrete examples. End with a clear takeaway and a "so what."

Rules:
"""
        + STYLE_RULES
        + SCAFFOLD_RULES
        + """
Write the full Core section. Use headings and short paragraphs. Define all technical terms. Address {student_name}.
""",
        "depth": """You are writing the DEPTH section for a student aiming for competition level (e.g. International Science Olympiads).

Topic: {topic}
Student name: {student_name}
Context: Core content is done. Now extend to harder applications and connections.

Tasks:
1. Add one or two extensions: a harder application, a connection to another topic, or a past Olympiad-style idea (do not copy past problems verbatim; inspire from their style).
2. Keep the same style: one idea at a time, intuition first, clear takeaways.
3. End with a short "Where this can go next" (what topics or competitions naturally follow).

Rules:
"""
        + STYLE_RULES
        + SCAFFOLD_RULES
        + """
Write the full Depth section. Encourage {student_name} by name. Stay rigorous but readable.
""",
        "practice": """You are writing the PRACTICE section of a learning module.

Topic: {topic}
Student name: {student_name}
Context: The student has read prerequisites, foundations, core, and depth.

Tasks:
1. Propose 3–5 practice tasks: mix of (a) quick checks, (b) medium application, (c) one harder/extension question.
2. For each task: state the question clearly, then provide a solution with short explanations (not just the answer).
3. Tie each task to a specific idea from the module ("This checks your understanding of ...").

Rules: Clear wording. No trick questions; the goal is to reinforce understanding. Address {student_name}.
""",
        "deliver": """You are assembling the final learning module.

Topic: {topic}
Student name: {student_name}

Sections you have:
- Prerequisites: {prereqs_content}
- Foundations: {foundations_content}
- Core: {core_content}
- Depth: {depth_content}
- Practice: {practice_content}

Tasks:
1. Combine these into one coherent markdown document.
2. Add a short title and a one-paragraph intro for {student_name} ("In this module we will ...").
3. Add a brief table of contents at the top.
4. Ensure section order: Intro → Prerequisites → Foundations → Core → Depth → Practice.
5. Do not duplicate content; smooth transitions between sections if needed.

Output the full markdown. It should be ready to save as a single file (e.g. science_{topic_slug}_for_{student_name}.md).
""",
    }

    def __init__(self) -> None:
        super().__init__()

    def detect_problem_type(self, X: Any = None, y: Any = None, **kwargs: Any) -> str:
        """Detect if this is a science-teaching request from task/context."""
        task = kwargs.get("task") or kwargs.get("business_context") or ""
        if isinstance(X, str):
            task = task or X
        if not isinstance(task, str):
            return "unknown"
        t = task.lower()
        if any(
            w in t for w in ["science", "teach", "learn", "topic", "student", "olympiad", "module"]
        ):
            return "science_teaching"
        return "unknown"

    def validate_inputs(self, **kwargs: Any) -> bool:
        task = kwargs.get("task") or kwargs.get("business_context") or ""
        subject = kwargs.get("subject", "").strip()
        topic = kwargs.get("topic", "").strip()
        student_name = kwargs.get("student_name", "").strip()
        if topic:
            return True
        if isinstance(task, str) and (
            "topic" in task.lower() or "teach" in task.lower() or "science" in task.lower()
        ):
            return True
        return bool(subject and topic and student_name)

    def get_prompt(self, prompt_name: str, **format_kwargs: Any) -> str:
        defaults = {
            "subject": "Science",
            "topic": "General science topic",
            "student_name": "the student",
            "prereqs_list": "basic concepts needed for the topic",
            "prereqs_content": "[Prerequisites section content]",
            "foundations_content": "[Foundations section content]",
            "core_content": "[Core section content]",
            "depth_content": "[Depth section content]",
            "practice_content": "[Practice section content]",
            "topic_slug": "topic",
        }
        for k, v in defaults.items():
            if k not in format_kwargs:
                format_kwargs[k] = v
        return super().get_prompt(prompt_name, **format_kwargs)

    def build_goal(
        self,
        subject: str = "Science",
        topic: str = "",
        student_name: str = "",
    ) -> str:
        """
        Build a single, detailed goal string that can be passed to Orchestrator.run().
        The planner will break this into steps (research, write, save).
        """
        if not topic:
            return ""
        student = student_name.strip() or "the student"
        subject = (subject or "Science").strip()

        return (
            f"Create a world-class science learning module for {student}. "
            f"Subject: {subject}. Topic: {topic}. "
            "Use this exact pedagogy: "
            "(1) Start with prerequisites—list and explain every building block needed, one concept at a time, intuition before formalism. "
            "(2) Then foundations—introduce the topic from first principles with a real-world hook and everyday analogies before any formal terms. "
            "(3) Then core—deep dive with clear structure: what we are learning, why it matters, the idea step by step, how it connects; one idea per paragraph, define every technical term. "
            "(4) Then depth—extend to competition/Olympiad level: harder applications, connections to other topics, or Olympiad-style thinking. "
            "(5) Then practice—3 to 5 tasks (quick check, medium, one harder) with full solutions and short explanations. "
            "Rules: one concept at a time; no jargon until defined; every section ends with a one-sentence takeaway and a 'so what'; address the student by name; clear headings and short paragraphs. "
            f"Save the final module as one markdown file (e.g. science_{_slug(topic)}_for_{_slug(student)}.md)."
        )

    def get_learning_plan(
        self,
        subject: str = "Science",
        topic: str = "",
        student_name: str = "",
    ) -> Dict[str, Any]:
        """
        Return a structured learning plan (stages and short descriptions).
        Useful for UI or for a skill that runs each stage via separate run() calls.
        """
        if not topic:
            return {}
        student = student_name.strip() or "the student"
        return {
            "subject": subject or "Science",
            "topic": topic,
            "student_name": student,
            "stages": [
                {
                    "name": "Prerequisites",
                    "description": "Identify and explain building blocks, one concept at a time.",
                },
                {
                    "name": "Foundations",
                    "description": "Introduce topic from first principles; intuition before formalism.",
                },
                {"name": "Core", "description": "Deep dive with clear structure and takeaways."},
                {"name": "Depth", "description": "Extend to Olympiad/competition level."},
                {"name": "Practice", "description": "3–5 tasks with full solutions."},
                {"name": "Deliver", "description": "Assemble and save as one markdown module."},
            ],
            "goal": self.build_goal(subject=subject, topic=topic, student_name=student),
        }


def _slug(s: str) -> str:
    """Simple slug for filenames."""
    if not s:
        return "module"
    return "".join(c if c.isalnum() else "_" for c in s.strip().lower()).strip("_") or "module"
