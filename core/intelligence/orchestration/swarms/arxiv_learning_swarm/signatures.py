"""ArXiv Learning Swarm - DSPy Signatures."""

import dspy


class ConceptExtractionSignature(dspy.Signature):
    """Extract key concepts from a paper.

    You are extracting concepts that need to be TAUGHT, not just listed.
    For each concept, identify:
    1. What problem does it solve? (the WHY)
    2. What do you need to know first? (prerequisites)
    3. How hard is it to understand?

    Think like a teacher planning a lesson.
    """

    paper_title: str = dspy.InputField(desc="Paper title")
    abstract: str = dspy.InputField(desc="Paper abstract")
    full_text_summary: str = dspy.InputField(desc="Summary of full paper if available")

    concepts: str = dspy.OutputField(
        desc="JSON list of concepts with name, description, why_it_matters, prerequisites, difficulty(1-5)"
    )
    learning_order: str = dspy.OutputField(
        desc="Recommended order to learn concepts, separated by |"
    )
    key_innovation: str = dspy.OutputField(desc="The ONE key innovation in simple terms")


class IntuitionBuilderSignature(dspy.Signature):
    """Build intuition for a concept BEFORE diving into math.

    Your goal is to make the reader FEEL why this concept makes sense.

    RULES:
    1. Start with a real-world problem or analogy
    2. Build up step by step - no jumping ahead
    3. Use "imagine if..." scenarios
    4. Make the reader predict what comes next
    5. Celebrate understanding with enthusiasm!

    NO jargon until intuition is solid.
    """

    concept: str = dspy.InputField(desc="Concept to explain")
    why_it_matters: str = dspy.InputField(desc="Why this concept matters")
    audience_level: str = dspy.InputField(desc="Starting knowledge level")
    prerequisites: str = dspy.InputField(desc="What they should already know")

    hook: str = dspy.OutputField(desc="Opening hook that grabs attention - why should they care?")
    analogy: str = dspy.OutputField(desc="Real-world analogy that captures the essence")
    intuition_build: str = dspy.OutputField(
        desc="Step-by-step intuition building, each step on new line"
    )
    aha_moment: str = dspy.OutputField(desc="The 'Bingo!' moment where it all clicks")
    check_understanding: str = dspy.OutputField(desc="Question to verify they got it")


class MathSimplifierSignature(dspy.Signature):
    """Make math accessible by building from basics.

    Your job is to make math feel INEVITABLE, not arbitrary.

    APPROACH:
    1. Start with what they know (basic algebra/calculus)
    2. Each new symbol EARNS its place by solving a problem
    3. Show WHY the math looks the way it does
    4. Connect equations to the intuition built earlier
    5. Use concrete numbers before variables

    Math should feel like a natural next step, not a wall.
    """

    concept: str = dspy.InputField(desc="Concept with its math")
    intuition: str = dspy.InputField(desc="The intuition already built")
    equations: str = dspy.InputField(desc="Key equations to explain")
    audience_level: str = dspy.InputField(desc="Math background")

    math_motivation: str = dspy.OutputField(desc="Why we need math here (the problem it solves)")
    building_blocks: str = dspy.OutputField(
        desc="Basic math building blocks needed, each on new line"
    )
    step_by_step: str = dspy.OutputField(
        desc="Step-by-step derivation. Format: 'Step 1: [explanation]\\nStep 2: [explanation]\\n...' - EACH STEP MUST START ON A NEW LINE"
    )
    concrete_example: str = dspy.OutputField(
        desc="Worked example with actual numbers, each calculation step on new line"
    )
    connection_to_intuition: str = dspy.OutputField(
        desc="How this math connects to earlier intuition"
    )


class ExampleGeneratorSignature(dspy.Signature):
    """Generate examples that reinforce understanding.

    Examples should:
    1. Start simple, get progressively harder
    2. Cover different angles of the concept
    3. Include "what if" variations
    4. Have clear, checkable answers

    Make examples that teach, not just test.
    """

    concept: str = dspy.InputField(desc="Concept to exemplify")
    intuition: str = dspy.InputField(desc="Intuition built")
    math_explanation: str = dspy.InputField(desc="Math explanation")

    simple_example: str = dspy.OutputField(desc="Simple example anyone can follow")
    intermediate_example: str = dspy.OutputField(desc="Example that tests understanding")
    challenging_example: str = dspy.OutputField(desc="Example that pushes boundaries")
    code_example: str = dspy.OutputField(desc="Python code demonstrating the concept")
    what_if_variations: str = dspy.OutputField(desc="What-if variations to explore, separated by |")


class ProgressiveBuilderSignature(dspy.Signature):
    """Build complete learning content progressively.

    Structure (ALWAYS in this order):
    1. THE HOOK - Why should anyone care? What problem does this solve?
    2. THE BASICS - What do we need to know first? (brief review)
    3. THE INTUITION - Build understanding without math
    4. THE MATH - Now that we get it, here's the precise formulation
    5. THE APPLICATION - See it in action
    6. THE DEEP DIVE - For those who want more

    Each section builds on the previous. No skipping!
    """

    paper_info: str = dspy.InputField(desc="Paper information")
    concepts: str = dspy.InputField(desc="Concepts to cover")
    intuitions: str = dspy.InputField(desc="Intuitions built")
    math_explanations: str = dspy.InputField(desc="Math explanations")
    examples: str = dspy.InputField(desc="Examples generated")
    celebration_word: str = dspy.InputField(desc="Word to celebrate insights")

    complete_content: str = dspy.OutputField(desc="Complete learning content with all sections")
    key_insights: str = dspy.OutputField(desc="Key insights (celebration moments), separated by |")
    summary: str = dspy.OutputField(desc="Concise summary of what was learned")
    next_steps: str = dspy.OutputField(desc="What to learn next, separated by |")


class ContentPolisherSignature(dspy.Signature):
    """Polish content to be engaging and clear.

    Make sure:
    1. Language is conversational, not academic
    2. Enthusiasm comes through (but not fake)
    3. Complex ideas have simple explanations
    4. Flow is smooth between sections
    5. Reader feels capable, not intimidated

    The reader should WANT to keep reading.
    """

    draft_content: str = dspy.InputField(desc="Draft learning content")
    style: str = dspy.InputField(desc="Desired style")
    audience: str = dspy.InputField(desc="Target audience")

    polished_content: str = dspy.OutputField(desc="Polished, engaging content")
    engagement_score: float = dspy.OutputField(desc="Estimated engagement 0-100")
    clarity_score: float = dspy.OutputField(desc="Clarity score 0-100")


# =============================================================================
# UNIFIED LEARNING SIGNATURE - MEGA-OPTIMIZATION
# =============================================================================
# This single signature replaces 8+ separate LLM calls with ONE comprehensive call.
# Benefits: 80% faster, better coherence, full context awareness, no huge prompts.


class UnifiedConceptLearningSignature(dspy.Signature):
    """Generate complete learning content for ALL concepts in ONE pass.

    You are a world-class educator creating an engaging learning experience.
    For each concept, provide intuition, math (if needed), and examples.

    TEACHING PHILOSOPHY:
    1. Hook first - why should they care?
    2. Intuition before math - make it FEEL right
    3. Math earns its place - solve real problems
    4. Examples reinforce - simple to challenging
    5. Celebrate insights with enthusiasm

    OUTPUT FORMAT (JSON):
    {
        "hook": "Opening hook - why this paper matters",
        "concepts": [
            {
                "name": "Concept Name",
                "analogy": "Real-world analogy",
                "intuition": "Detailed step-by-step intuition building",
                "aha_moment": "The key insight moment",
                "math_motivation": "Why we need math here",
                "math_steps": "Detailed step-by-step math explanation",
                "simple_example": "Easy example with walkthrough",
                "code_example": "Python code demonstrating concept"
            }
        ],
        "key_insights": ["Insight 1", "Insight 2", "Insight 3"],
        "summary": "Comprehensive summary of what was learned",
        "next_steps": ["What to learn next 1", "What to learn next 2"]
    }
    """

    paper_title: str = dspy.InputField(desc="Paper title")
    paper_abstract: str = dspy.InputField(desc="Paper abstract (key content)")
    concepts_json: str = dspy.InputField(
        desc="JSON list of concepts with name, description, why_it_matters, difficulty"
    )
    audience_level: str = dspy.InputField(desc="Target audience: beginner, intermediate, advanced")
    celebration_word: str = dspy.InputField(desc="Word to celebrate insights (e.g., Bingo!)")

    learning_content_json: str = dspy.OutputField(
        desc="Complete learning content as JSON (see format above)"
    )


class SingleConceptDeepSignature(dspy.Signature):
    """Generate DEEP, comprehensive learning content for ONE concept.

    Create engaging, thorough educational content with:
    - Rich analogies and real-world connections
    - Step-by-step intuition building (multiple paragraphs)
    - Detailed math derivations with explanations
    - Multiple examples from simple to advanced
    - Working code with comments

    Be thorough - this will become multiple pages of content.
    """

    concept_name: str = dspy.InputField(desc="Concept name")
    concept_description: str = dspy.InputField(desc="What this concept is")
    why_it_matters: str = dspy.InputField(desc="Why this concept is important")
    paper_context: str = dspy.InputField(desc="Paper title and key context")
    audience_level: str = dspy.InputField(desc="beginner/intermediate/advanced")

    analogy: str = dspy.OutputField(desc="Rich real-world analogy (2-3 sentences)")
    intuition: str = dspy.OutputField(
        desc="Detailed intuition building (3-5 paragraphs, use newlines)"
    )
    aha_moment: str = dspy.OutputField(desc="The key insight that makes it click")
    math_motivation: str = dspy.OutputField(desc="Why math is needed here")
    math_steps: str = dspy.OutputField(
        desc="Step-by-step math with explanations (use newlines between steps)"
    )
    simple_example: str = dspy.OutputField(desc="Simple worked example with explanation")
    advanced_example: str = dspy.OutputField(desc="More challenging example")
    code_example: str = dspy.OutputField(desc="Python code with detailed comments (10-20 lines)")


# =============================================================================
# AGENTS
# =============================================================================
