"""Olympiad Learning Swarm - DSPy Signatures.

Each signature embeds world-class teaching philosophy:
- Start with WHY before WHAT
- Build from the simplest possible example
- Every formula earns its place by solving a real problem
- Use concrete data/numbers before abstract variables
- Celebrate insight moments
- Pattern recognition is the key to competition success

OUTPUT LENGTH CONTRACT:
Every output field specifies a minimum length. The LLM MUST meet these
minimums to produce world-class content. Short, terse answers are never
acceptable — every field should be elaborate, detailed, and richly specific.
"""

import dspy


class CurriculumArchitectSignature(dspy.Signature):
    """Design a complete, elaborately detailed learning roadmap from building blocks to mastery.

    You are the world's best curriculum designer. Your roadmap must be THOROUGH and DETAILED.

    1. IDENTIFY THE FOUNDATIONS — What building blocks does the student absolutely need?
       For EACH building block, provide a detailed description (3-4 sentences minimum),
       explain WHY it's needed with a specific example, give a thorough quick review
       (4-5 sentences with a worked example), and a check question.

    2. MAP THE DEPENDENCY GRAPH — Which concepts depend on which? Never teach B before A.

    3. CREATE MILESTONE CHECKPOINTS — After each section, how do we verify understanding?
       Each checkpoint should be a specific question or mini-problem, not just "understand X".

    4. PLAN THE DIFFICULTY CURVE — Smooth progression, never a cliff.

    5. DESIGN A RUNNING EXAMPLE — Create a vivid, engaging real-world scenario (8-10 sentences)
       that will be threaded through EVERY section of the lesson. This scenario should be
       relatable to the student, involve specific characters/names/places, and naturally
       illustrate the core concepts. Examples:
       - For fractions: "Aria and her 3 friends are planning a pizza party at Mario's Pizzeria..."
       - For body parts: "Dr. Sarah is giving a guided tour of the human body museum..."
       - For physics: "The SpaceX launch team needs to calculate the rocket trajectory..."

    6. PLAN SECTION DEPTH — Specify how many words/paragraphs each section needs:
       - Building Blocks: 400-600 words (3-4 paragraphs per block with worked examples)
       - Discovery: 600-800 words (step-by-step guided discovery with specific numbers)
       - Intuition: 500-700 words (vivid analogy + multi-step progression)
       - Patterns: 500-800 words (4-6 patterns, each with trigger + template + example)
       - Problems: 200-400 words per problem (statement + hints + full solution)
       - Mistakes: 400-600 words (4-6 traps with wrong vs correct worked examples)
       - Strategies: 400-600 words (4-6 strategies with step-by-step procedures)
       - Connections: 300-400 words (surprising links with specific examples)
       Total target: 4000-6000 words of world-class content.

    Think: "If I had to teach this to a brilliant kid who's never seen it, what's the
    perfect sequence so they discover it themselves?"
    """
    subject: str = dspy.InputField(desc="Subject area (e.g., mathematics, biology, physics)")
    topic: str = dspy.InputField(desc="Specific topic to teach (e.g., Number Theory, Human Body Parts)")
    student_name: str = dspy.InputField(desc="Student's name for personalization")
    current_level: str = dspy.InputField(desc="Student's current knowledge level")
    target_level: str = dspy.InputField(desc="Target mastery level (e.g., olympiad, 5th_grader)")

    building_blocks_json: str = dspy.OutputField(
        desc="JSON list of 4-8 building blocks. EACH block MUST have: "
        "{name (str), description (3-4 sentences explaining what this is and why it matters), "
        "why_needed (2-3 sentences on how this connects to the main topic), "
        "quick_review (4-5 sentences with a specific worked example using real numbers/data — "
        "not just 'review this concept' but actually walk through it), "
        "check_question (a specific question with a concrete answer, not vague), "
        "difficulty (1-5)}. Be thorough — each block should be a mini-lesson."
    )
    learning_sequence: str = dspy.OutputField(
        desc="Ordered list of 6-12 concepts to teach, separated by |. "
        "For each concept, include a brief rationale for why it comes at this position "
        "(e.g., 'Equivalent fractions | builds on equal parts, needed before addition')"
    )
    milestone_checkpoints: str = dspy.OutputField(
        desc="6-10 checkpoints to verify understanding, separated by |. "
        "Each checkpoint must be a SPECIFIC question or mini-problem with expected answer "
        "(e.g., 'Can you explain why 2/4 = 1/2 using the pizza model? Expected: cutting each of 2 slices in half gives 4 slices total')"
    )
    running_example_scenario: str = dspy.OutputField(
        desc="A vivid, engaging real-world scenario (8-10 sentences minimum) that will be "
        "threaded through EVERY section of the lesson. Include specific character names, "
        "a setting, concrete quantities/numbers, and a narrative arc that naturally "
        "illustrates the core concepts. This scenario should feel like the opening of a "
        "story that makes the student want to keep reading. It must be reusable in "
        "building blocks, discovery, patterns, problems, and mistakes sections."
    )
    section_depth_plan: str = dspy.OutputField(
        desc="Content depth targets as pipe-separated plan: "
        "'Building Blocks: 500 words, 4 blocks with worked examples | "
        "Discovery: 700 words, 3 guided examples building to the key insight | "
        "Patterns: 600 words, 5 patterns with triggers | "
        "Problems: 1200 words, 10 problems across 4 tiers with full solutions | "
        "Mistakes: 500 words, 5 traps with wrong vs correct | "
        "Strategies: 500 words, 4 strategies with steps'. "
        "Adjust word counts based on topic complexity. Total should target 4000-6000 words."
    )
    estimated_sessions: str = dspy.OutputField(
        desc="Estimated number of focused sessions to reach mastery, with brief breakdown"
    )


class ConceptDecomposerSignature(dspy.Signature):
    """Break down a concept into its absolute simplest components with ELABORATE detail.

    You explain like the best data science educator: take ANYTHING complex and make it
    feel obvious by starting with the simplest possible example.

    RULES (you MUST follow ALL of these):
    1. Start with ONE concrete example using small numbers (2, 3, 5... not n, k, m).
       Walk through it step by step. Show your work. Use at least 4-5 sentences.
    2. Show the pattern emerging from 3-4 examples that BUILD on each other.
       Each example should be 2-3 sentences. The progression must be clear.
    3. Let the student "discover" the formula by seeing it work across examples.
    4. Only THEN introduce the formal definition — which should feel INEVITABLE.
    5. Each step must feel like "well, obviously..." to the student.

    Example flow for "modular arithmetic":
    - "What's the remainder when you divide 17 by 5? Let's work it out: 5×3=15, so 17-15=2. The remainder is 2."
    - "Now divide 22 by 5. We get 5×4=20, so 22-20=2. The remainder is also 2! Interesting..."
    - "And 27÷5? 5×5=25, so 27-25=2. Three different numbers, ALL with remainder 2."
    - "Both 17, 22, and 27 behave the same way when divided by 5. They're all '2 more than a multiple of 5'. Let's call that 'equivalent mod 5'."
    - Now they DISCOVERED modular arithmetic. The formula just names what they already know.

    EVERY output field must be DETAILED and SPECIFIC. No one-liners.
    """
    concept_name: str = dspy.InputField(desc="Name of the concept to decompose")
    concept_description: str = dspy.InputField(desc="Brief description of the concept")
    subject: str = dspy.InputField(desc="Subject area")
    prerequisites: str = dspy.InputField(desc="What the student already knows")
    student_name: str = dspy.InputField(desc="Student's name")

    simplest_example: str = dspy.OutputField(
        desc="The absolute simplest concrete example (MINIMUM 5-7 sentences). "
        "Use real numbers, specific items, and a step-by-step walkthrough. "
        "Show each step of your work, like a patient tutor sitting next to the student. "
        "Example: 'Imagine you have a chocolate bar with 8 equal pieces. You eat 3 of them. "
        "How much did you eat? Well, each piece is 1/8 of the bar. You ate 3 pieces. "
        "So you ate 3/8 of the bar. The 8 tells you how many equal pieces exist (the denominator). "
        "The 3 tells you how many pieces you took (the numerator).'"
    )
    pattern_discovery: str = dspy.OutputField(
        desc="Guide the student through 3-4 progressively complex examples (MINIMUM 8-12 sentences total). "
        "Each example should build on the previous one, adding one new element. "
        "After each example, highlight what's the SAME and what's DIFFERENT. "
        "By the end, the student should see the general pattern WITHOUT being told. "
        "Use phrases like 'Notice something?' and 'See how this is like the first example?' "
        "Include the actual numbers and calculations, not just descriptions."
    )
    formal_definition: str = dspy.OutputField(
        desc="The formal definition and framework (MINIMUM 4-6 sentences, ideally 2 paragraphs). "
        "Start with 'Now that you've seen the pattern...' and give the formal name. "
        "Define every term precisely. Include a worked example using the formal notation. "
        "Connect BACK to the discovery examples: 'Remember when we found that 3/8 of the "
        "chocolate? In formal terms, that's...' The student should feel the definition "
        "is just giving a name to what they already understand."
    )
    why_it_works: str = dspy.OutputField(
        desc="Deep explanation of WHY this works (MINIMUM 4-6 sentences, ideally 2 paragraphs). "
        "Connect back to the concrete examples. Explain the underlying LOGIC, not just the rule. "
        "Include a concrete proof or demonstration with specific numbers. "
        "The student should finish thinking 'Now I understand WHY, not just HOW.'"
    )
    key_insight: str = dspy.OutputField(
        desc="The ONE key insight that makes everything click (2-3 sentences). "
        "This should be the 'aha!' moment — a statement so clear that the student "
        "will remember it months from now. Make it vivid and specific to this concept."
    )
    common_misconceptions: str = dspy.OutputField(
        desc="Top 3-4 misconceptions, separated by |. For EACH misconception: "
        "state what students wrongly believe, then in parentheses explain WHY it's "
        "wrong with a specific counterexample. Example: 'Bigger denominator means bigger "
        "fraction (WRONG: 1/8 is actually smaller than 1/4 because you're splitting "
        "into more pieces, making each piece smaller)'"
    )


class IntuitionBuilderSignature(dspy.Signature):
    """Build deep intuition for a concept BEFORE any formulas — with rich, vivid detail.

    Your goal: make the student FEEL why this concept exists and why it HAS to work
    the way it does. After your explanation, the formula should feel inevitable.

    APPROACH (be ELABORATE on each step):
    1. Start with a REAL problem that naturally leads to this concept.
       Paint a vivid scene with specific details — names, places, quantities.
       ("Imagine you're at Aria's birthday party. There are 3 pizzas and 4 friends...")
    2. Walk through the problem step by step, showing your work at each stage.
       Don't skip any reasoning steps. Use actual numbers and calculations.
    3. Use analogies from everyday life — make them SPECIFIC, not generic.
       ("Think of fractions like slicing a cake — but here's the key: all slices must be EQUAL.")
    4. Build up to the "aha!" moment where the general principle becomes obvious.
    5. Only THEN connect to the formal framework.

    The student should think "of course it works that way!" before seeing any formula.
    """
    concept: str = dspy.InputField(desc="Concept with description")
    why_it_matters: str = dspy.InputField(desc="Why this concept matters for success")
    student_name: str = dspy.InputField(desc="Student's name")
    audience_level: str = dspy.InputField(desc="Current knowledge level")
    subject: str = dspy.InputField(desc="Subject area")

    real_world_hook: str = dspy.OutputField(
        desc="A compelling real-world scenario (MINIMUM 6-8 sentences). "
        "Include specific character names, a concrete setting, exact quantities, "
        "and sensory details that make the student feel present. Paint a vivid scene. "
        "The scenario must naturally lead to needing this concept. "
        "Example: 'You and your friend Maya are at Antonio's Pizza Kitchen. You ordered "
        "2 large pizzas — one pepperoni, one veggie. Each pizza is cut into 8 equal slices. "
        "Maya says she's super hungry and wants 3 slices of pepperoni and 2 of veggie. "
        "You want to know: what fraction of ALL the pizza is Maya eating? You look down "
        "at the 16 total slices across both boxes and start thinking...' "
        "This is the kind of detail we need."
    )
    analogy: str = dspy.OutputField(
        desc="A vivid, memorable analogy (MINIMUM 4-5 sentences). "
        "Compare the concept to something the student already deeply understands. "
        "Extend the analogy — show where it works AND where it breaks down. "
        "The analogy should create a mental image so strong the student never forgets it."
    )
    intuition_build: str = dspy.OutputField(
        desc="Step-by-step intuition building (MINIMUM 6-8 steps, each 2-3 sentences). "
        "Walk through the scenario progressively. Each step adds ONE new idea. "
        "Use the student's name. Show actual numbers/data at each step. "
        "Include 'What do you think happens next?' between steps to build curiosity. "
        "By the end, the general principle should feel OBVIOUS and INEVITABLE. "
        "Format: 'Step 1: [content]\nStep 2: [content]...' on separate lines."
    )
    aha_moment: str = dspy.OutputField(
        desc="The breakthrough moment (MINIMUM 3-4 sentences). "
        "The general principle becomes obvious from the specific examples. "
        "State it as a revelation: 'And here's the magical part — it ALWAYS works this way! "
        "No matter how many slices, no matter how many friends...' "
        "This should feel like the climax of a story."
    )
    visual_description: str = dspy.OutputField(
        desc="Description of a visual/diagram (MINIMUM 4-5 sentences). "
        "Describe exactly what to draw: shapes, labels, arrows, colors. "
        "Be specific enough that someone could sketch it from your description alone."
    )


class PatternHunterSignature(dspy.Signature):
    """Identify problem-solving PATTERNS with elaborate detail and worked examples.

    In competitions, the student who wins is the one who RECOGNIZES which pattern applies.
    Your job is to catalog the patterns for this topic.

    For EACH pattern (include 4-6 patterns minimum):
    1. NAME it clearly (students need vocabulary to think about patterns)
    2. DESCRIBE it in 3-4 sentences — what is this pattern and why does it work?
    3. Show the TRIGGER in 2-3 sentences — what in a problem signals this pattern?
       Use specific phrases or structural cues the student should watch for.
    4. Give a TEMPLATE in 3-4 sentences — the step-by-step general approach
    5. Include a WORKED EXAMPLE (4-5 sentences) showing the pattern applied to a specific problem

    Think: "What would a grandmaster competitor's pattern library look like for this topic?"
    """
    topic: str = dspy.InputField(desc="Topic to find patterns in")
    subject: str = dspy.InputField(desc="Subject area")
    concepts: str = dspy.InputField(desc="Key concepts covered")
    target_level: str = dspy.InputField(desc="Competition level to target")

    patterns_json: str = dspy.OutputField(
        desc="JSON list of 4-6 patterns. EACH pattern MUST have: "
        "{name (str), "
        "description (3-4 sentences explaining the pattern and WHY it works), "
        "when_to_use (2-3 sentences with specific trigger words/phrases to look for in problems), "
        "example_trigger (a specific problem excerpt that would trigger this pattern, 2-3 sentences), "
        "template (3-4 sentence step-by-step procedure for applying this pattern)}. "
        "Be thorough — each pattern should be a mini-tutorial."
    )
    pattern_connections: str = dspy.OutputField(
        desc="How these patterns connect (MINIMUM 4-6 sentences). "
        "Which patterns chain together? Which pattern do you try FIRST, and when do "
        "you switch to another? Give a specific example of a problem that requires "
        "combining two patterns."
    )
    meta_strategy: str = dspy.OutputField(
        desc="The meta-strategy for pattern selection (MINIMUM 4-6 sentences). "
        "A decision tree or flowchart in words: 'If you see X, try Pattern A first. "
        "If that doesn't work because of Y, switch to Pattern B.' "
        "Include specific signal words to look for."
    )


class ProblemCrafterSignature(dspy.Signature):
    """Create a progressive problem set with DETAILED statements and FULL solutions.

    You craft problems like a competition problem setter:

    FOUNDATION (warm-up): Direct application, builds confidence
    - "Given that... find..." (straightforward)

    INTERMEDIATE (school competition): Requires connecting 2 concepts
    - Needs one insight, one technique

    ADVANCED (national level): Requires creative combination of techniques
    - The solution path isn't obvious, need to explore

    OLYMPIAD (international level): Requires deep insight + elegant technique
    - Beautiful problems with surprising solutions
    - Often combines ideas from different areas

    EVERY problem MUST include:
    1. A CURIOSITY HOOK opening question that makes the student wonder
       ("What happens if we try to share 3 pizzas among 4 friends?")
    2. A clear, detailed statement (3-5 sentences minimum — set the scene, not just the math)
    3. 2-3 progressive hints (each hint reveals a bit more of the solution path)
    4. A COMPLETE step-by-step solution (6-10 sentences minimum — show every step,
       explain WHY each step works, don't skip anything)
    5. A key insight explaining what this problem teaches
    6. Which pattern from the pattern library it exercises
    7. Where natural, connect back to the running_example scenario
    """
    topic: str = dspy.InputField(desc="Topic for problems")
    subject: str = dspy.InputField(desc="Subject area")
    concepts: str = dspy.InputField(desc="Concepts to test")
    patterns: str = dspy.InputField(desc="Patterns to exercise (JSON with name, description, when_to_use, template, trigger)")
    running_example: str = dspy.InputField(desc="Running example/scenario to reference in problems where natural")
    tier: str = dspy.InputField(desc="Difficulty tier: foundation/intermediate/advanced/olympiad")
    count: str = dspy.InputField(desc="Number of problems to generate")
    student_name: str = dspy.InputField(desc="Student's name")

    problems_json: str = dspy.OutputField(
        desc="JSON list of problems. EACH problem MUST have: "
        "{statement (3-5 sentences — set the scene, ask the question clearly, include specific numbers), "
        "hints (list of 2-3 progressive hints — each hint is 1-2 sentences, reveals one piece of the approach), "
        "solution (DETAILED step-by-step, MINIMUM 6-10 sentences — show every calculation, explain WHY "
        "each step works, highlight the key move, don't skip any reasoning), "
        "strategy_used (which strategy applies), "
        "time_estimate_minutes (int), "
        "key_insight (2-3 sentences — what this problem teaches that's NEW), "
        "common_mistakes (list of 1-2 mistakes students make on THIS specific problem), "
        "relates_to_pattern (name of pattern this exercises from the pattern library)}. "
        "Solutions should read like a patient tutor explaining to the student."
    )


class SolutionStrategistSignature(dspy.Signature):
    """Teach problem-solving STRATEGIES with detailed examples and reasoning.

    This is the difference between knowing the subject and WINNING competitions.

    STRATEGY CATEGORIES:
    1. ATTACK STRATEGIES — How to start a problem you've never seen
       (Try small cases, work backwards, draw a picture, guess-and-check)
    2. PROOF TECHNIQUES — When and how to use each
       (Contradiction, induction, pigeonhole, extremal principle, invariants)
    3. COMPUTATION TRICKS — Speed techniques for competitions
       (Shortcuts, symmetry exploitation, clever substitutions)
    4. TIME MANAGEMENT — How to allocate time in a competition
    5. STUCK STRATEGIES — What to do when you're stuck
       (Change representation, consider the complement, add auxiliary elements)

    For EACH strategy, be ELABORATE:
    - Description: 3-5 sentences explaining what this strategy IS and WHY it's powerful
    - When to use: 2-3 sentences with specific trigger signals
    - Steps: 4-6 detailed steps (each step is 1-2 sentences)
    - Example: A full worked example (5-8 sentences) showing the strategy in action
    - Include a "WHY does this work?" reasoning section for every strategy
    """
    topic: str = dspy.InputField(desc="Topic area")
    subject: str = dspy.InputField(desc="Subject area")
    concepts: str = dspy.InputField(desc="Concepts being taught")
    target_level: str = dspy.InputField(desc="Competition level")

    strategies_json: str = dspy.OutputField(
        desc="JSON list of 4-6 strategies. EACH strategy MUST have: "
        "{name (str), "
        "description (3-5 sentences explaining what this strategy does and WHY it's powerful), "
        "when_to_use (2-3 sentences with specific trigger conditions/signal words), "
        "steps (list of 4-6 detailed steps — each step 1-2 sentences, not just keywords), "
        "example_problem (a specific problem statement, 2-3 sentences), "
        "example_solution (a FULL worked solution applying this strategy, MINIMUM 5-8 sentences "
        "showing each step of the strategy in action), "
        "pitfalls (list of 2-3 common mistakes when applying this strategy)}."
    )
    speed_techniques: str = dspy.OutputField(
        desc="3-5 speed techniques separated by |. Each technique MUST be 2-3 sentences with a "
        "specific example (e.g., 'When comparing fractions with same numerator, the one with "
        "the SMALLER denominator is larger — this saves time vs. cross-multiplying. For example, "
        "3/4 > 3/7 because 4ths are bigger pieces than 7ths.')"
    )
    stuck_toolkit: str = dspy.OutputField(
        desc="5-7 things to try when stuck, separated by |. Each item MUST be 2-3 sentences with "
        "a specific example of how it applies to this topic."
    )


class MistakeAnalyzerSignature(dspy.Signature):
    """Identify and explain common mistakes with ELABORATE trap scenarios.

    Prevention is better than cure. Frame each mistake as a TRAP QUESTION:

    1. Present the trap scenario (3-4 sentences — set up a realistic problem)
    2. Ask "What do you think the answer is?" — let the student think first
    3. Show the WRONG way in detail (3-4 sentences — actually work through the incorrect approach)
    4. Explain WHY it's wrong (2-3 sentences — pinpoint the exact flaw in reasoning)
    5. Show the CORRECT way in detail (3-4 sentences — work through the right approach)
    6. Give a DETECTOR (1-2 sentences — a quick check to catch this mistake)

    Where possible, use the running_example scenario to show the trap in a
    familiar context. Focus on mistakes that are SUBTLE — where students feel
    confident they're right but are actually wrong.
    """
    topic: str = dspy.InputField(desc="Topic to analyze mistakes for")
    subject: str = dspy.InputField(desc="Subject area")
    concepts: str = dspy.InputField(desc="Concepts covered")
    running_example: str = dspy.InputField(desc="Running example/scenario to reference in trap questions")
    target_level: str = dspy.InputField(desc="Competition level")

    mistakes_json: str = dspy.OutputField(
        desc="JSON list of 4-6 common mistakes. EACH mistake MUST have: "
        "{description (2-3 sentences explaining the trap — what students wrongly believe/do), "
        "why_it_happens (2-3 sentences on the underlying reasoning flaw), "
        "how_to_avoid (2-3 sentences with a specific detection/prevention technique), "
        "example_wrong (3-4 sentences showing a specific worked example done INCORRECTLY — "
        "include actual numbers and calculations that lead to the wrong answer), "
        "example_correct (3-4 sentences showing the SAME problem done CORRECTLY — "
        "include actual numbers and calculations that lead to the right answer)}. "
        "The wrong and correct examples should use THE SAME problem so the student "
        "can directly compare the approaches."
    )
    trap_problems: str = dspy.OutputField(
        desc="2-3 trap problems separated by |. Each trap problem is 3-5 sentences: "
        "a problem specifically designed to trigger a common mistake, with the "
        "expected wrong answer and explanation of why it's wrong."
    )


class ConnectionMapperSignature(dspy.Signature):
    """Map connections between this topic and other areas with rich detail and wonder.

    In advanced study, the BEST solutions often come from unexpected connections.
    A biology problem connected to physics. A math problem solved with real-world thinking.

    Create a sense of WONDER with each connection:
    - "Did you know that this is secretly related to...?"
    - "Here's something that will blow your mind..."
    - "The reason this works is the SAME reason that..."

    Map with SPECIFIC, DETAILED examples:
    1. DIRECT CONNECTIONS — 3-5 topics that directly build on this one,
       each with 2-3 sentences explaining HOW they connect
    2. SURPRISING CONNECTIONS — 2-3 non-obvious links with 3-4 sentences each,
       framed with wonder and showing a specific example
    3. POWERFUL COMBINATIONS — 2-3 combinations with 2-3 sentences each,
       showing what becomes possible when you combine knowledge
    4. NEXT TOPICS — Recommended study order with rationale
    """
    topic: str = dspy.InputField(desc="Current topic")
    subject: str = dspy.InputField(desc="Subject area")
    concepts: str = dspy.InputField(desc="Concepts covered")

    direct_connections: str = dspy.OutputField(
        desc="3-5 topics that directly build on this, separated by |. "
        "Each connection MUST be 2-3 sentences explaining specifically HOW this topic "
        "leads to or enables the next one, with a brief example. "
        "Not just 'relates to X' but 'Understanding fractions is essential for ratios "
        "because every ratio IS a fraction — when you see 3:5, that's really 3/5...'"
    )
    surprising_connections: str = dspy.OutputField(
        desc="2-3 non-obvious, surprising connections separated by |. "
        "Each MUST be 3-4 sentences, framed with curiosity and wonder. Start each with "
        "a hook: 'Here's something amazing: ...' or 'Did you know that...'. "
        "Include a specific example showing the surprising link in action."
    )
    powerful_combinations: str = dspy.OutputField(
        desc="2-3 powerful combinations separated by |. Each MUST be 2-3 sentences "
        "showing what becomes possible when this topic is combined with another. "
        "Include a specific example of a problem that requires both topics."
    )
    next_topics: str = dspy.OutputField(
        desc="4-6 recommended next topics in optimal study order, separated by |. "
        "Each MUST include a brief rationale (e.g., 'Decimals | natural next step — "
        "fractions and decimals are two ways to write the same number, learning conversion "
        "doubles your problem-solving toolkit')"
    )


class ContentAssemblerSignature(dspy.Signature):
    """Assemble all components into a beautifully flowing lesson of 4000-6000 words.

    You are assembling a masterclass. The lesson must FLOW like a STORY, not
    read like 8 separate documents concatenated together.

    CRITICAL RULES FOR COHERENCE:
    - Thread the running_example through EVERY section (reference it in patterns,
      problems, mistakes — not just the intro)
    - Before each major idea, ask a SOCRATIC QUESTION: "What do you think
      happens when...?" or "Before we reveal the answer, what's your guess?"
    - Write NATURAL TRANSITIONS between sections ("Now that you've seen the
      building blocks, something exciting happens when we combine them...")
    - Never repeat the same example twice — build on previous examples

    STRUCTURE (target 4000-6000 words total):
    1. HOOK (200-300 words) — Why should {student_name} care? Make them WANT to learn this.
    2. FOUNDATIONS CHECK (400-600 words) — Quick review of building blocks with worked examples
    3. THE DISCOVERY (600-800 words) — Walk through the concept like they're discovering it
    4. THE FORMALIZATION (300-400 words) — Precise language now that they understand WHY
    5. PATTERN LIBRARY (500-800 words) — Patterns with triggers, templates, and examples
    6. PROBLEM LADDER (1000-1500 words) — Progressive problems with full solutions
    7. STRATEGY TOOLKIT (400-600 words) — Competition-specific techniques
    8. MISTAKE PREVENTION (400-600 words) — Traps with worked wrong/correct examples
    9. CONNECTIONS (300-400 words) — Surprising links to other topics
    10. CHALLENGE ARENA (300-500 words) — Advanced problems for those ready

    Each transition should feel natural. The student should NEVER feel lost.
    Personalize for {student_name} throughout.
    """
    student_name: str = dspy.InputField(desc="Student's name")
    topic: str = dspy.InputField(desc="Topic being taught")
    subject: str = dspy.InputField(desc="Subject area")
    building_blocks: str = dspy.InputField(desc="Building blocks JSON")
    intuition: str = dspy.InputField(desc="Intuition content")
    decomposition: str = dspy.InputField(desc="Concept decomposition")
    patterns: str = dspy.InputField(desc="Pattern library JSON")
    strategies: str = dspy.InputField(desc="Strategy toolkit JSON")
    problems_summary: str = dspy.InputField(desc="Summary of problems across tiers")
    mistakes: str = dspy.InputField(desc="Common mistakes JSON")
    connections: str = dspy.InputField(desc="Topic connections")
    celebration_word: str = dspy.InputField(desc="Word to celebrate insights")

    complete_content: str = dspy.OutputField(
        desc="Complete assembled lesson content in markdown format. "
        "MINIMUM 4000 words. Follow the 10-section structure above. "
        "Include section headings, Socratic questions, transitions, "
        "and running example references throughout."
    )
    key_insights: str = dspy.OutputField(
        desc="The top 3-5 breakthrough insight moments from this lesson, separated by |. "
        "Each insight MUST be a specific, memorable statement (2-3 sentences), "
        "not a generic placeholder."
    )
    summary: str = dspy.OutputField(
        desc="Concise but thorough summary (4-6 sentences): what was learned, "
        "what the key breakthroughs were, and what this unlocks for future study."
    )
    competition_tips: str = dspy.OutputField(
        desc="Top 3-5 competition-specific tips, separated by |. "
        "Each tip MUST be 2-3 sentences with specific actionable advice."
    )


class SingleTopicDeepSignature(dspy.Signature):
    """Generate comprehensive, deep content for ONE topic in a single pass.

    You are the world's greatest teacher creating content that could turn
    any motivated student into a top performer.

    TEACHING STYLE:
    - Before EVERY major reveal, ask "What do you think happens when...?"
    - Use "Pause and think..." prompts before showing solutions
    - Create mini-mysteries: set up a puzzle, let curiosity build, then resolve
    - Thread ONE running example through the entire lesson (introduce it in the
      hook, use it in patterns, reference it in problems, bring it back in traps)

    Create thorough content covering ALL of these sections:
    - Building blocks review (400-600 words with concrete worked examples)
    - Core concept decomposition (600-800 words, simplest example first, build up)
    - Deep intuition building (500-700 words, real-world hook, analogy, step-by-step)
    - Pattern recognition (500-800 words, competition patterns with triggers and examples)
    - Problem-solving strategies (400-600 words, with worked examples for each)
    - Progressive problems (1000-1500 words, foundation through advanced with full solutions)
    - Common mistakes (400-600 words, trap scenarios with wrong vs correct)
    - Connections to other topics (300-400 words, surprising links)

    TOTAL TARGET: 4000-6000 words of world-class content.
    Use {student_name} throughout for personalization.
    """
    student_name: str = dspy.InputField(desc="Student's name")
    topic: str = dspy.InputField(desc="Topic to teach")
    subject: str = dspy.InputField(desc="Subject area")
    target_level: str = dspy.InputField(desc="Target mastery level")
    celebration_word: str = dspy.InputField(desc="Word to celebrate breakthroughs")

    hook: str = dspy.OutputField(
        desc="Compelling opening (MINIMUM 4-6 sentences). Paint a vivid real-world scene "
        "that makes the student WANT to learn this. Include specific names, places, quantities. "
        "This becomes the running example threaded through the entire lesson."
    )
    building_blocks: str = dspy.OutputField(
        desc="Quick review of prerequisites (MINIMUM 3-4 paragraphs, 400-600 words). "
        "For each prerequisite, include a specific worked example with real numbers."
    )
    concept_discovery: str = dspy.OutputField(
        desc="Lead the student to DISCOVER the concept (MINIMUM 4-6 paragraphs, 600-800 words). "
        "Start with simplest example, build through 3-4 progressive examples. "
        "Include 'Pause and think...' before the key revelation."
    )
    formal_definition: str = dspy.OutputField(
        desc="Formal definition and key theorems (MINIMUM 2-3 paragraphs). "
        "Now that they understand WHY, here's the precise language. Include worked examples."
    )
    key_insight: str = dspy.OutputField(
        desc="The ONE insight that makes everything click (2-3 sentences). "
        "A memorable, vivid statement the student will never forget."
    )
    patterns: str = dspy.OutputField(
        desc="4-6 competition patterns (MINIMUM 500-800 words). For each pattern: "
        "name, description (3 sentences), trigger (2 sentences), template (3 sentences), "
        "and a quick worked example (3-4 sentences)."
    )
    strategies: str = dspy.OutputField(
        desc="3-4 problem-solving strategies (MINIMUM 400-600 words). For each: "
        "name, when to use (2 sentences), step-by-step procedure (4-6 steps), "
        "and a worked example (4-5 sentences)."
    )
    problems_foundation: str = dspy.OutputField(
        desc="2-3 foundation problems (MINIMUM 300-500 words total). Each problem: "
        "clear statement (2-3 sentences), 2 hints, and a COMPLETE step-by-step solution "
        "(5-8 sentences showing every step)."
    )
    problems_intermediate: str = dspy.OutputField(
        desc="2-3 intermediate problems (MINIMUM 400-600 words total). Each problem: "
        "contextual statement (3-4 sentences), 2-3 hints, and a DETAILED solution "
        "(6-10 sentences with full reasoning)."
    )
    problems_advanced: str = dspy.OutputField(
        desc="2 advanced problems (MINIMUM 300-500 words total). Each problem: "
        "challenging statement (3-4 sentences), 2-3 progressive hints, and a "
        "thorough solution (8-12 sentences with key insight highlighted)."
    )
    common_mistakes: str = dspy.OutputField(
        desc="4-5 common mistakes (MINIMUM 400-600 words total). For each: "
        "the trap (2 sentences), wrong approach with specific numbers (3 sentences), "
        "correct approach with same numbers (3 sentences), and how to detect it (1 sentence)."
    )
    connections: str = dspy.OutputField(
        desc="Connections to other topics (MINIMUM 200-300 words). "
        "Include 2-3 surprising connections with specific examples that create wonder."
    )
    competition_tips: str = dspy.OutputField(
        desc="3-5 competition-specific tips (MINIMUM 200-300 words). "
        "Each tip must be 2-3 sentences with specific, actionable advice for this topic."
    )


class NarrativeEditorSignature(dspy.Signature):
    """Edit assembled lesson content to be curious, inquisitive, and coherent.

    You are a master narrative editor. The content you receive was generated by
    multiple independent agents — your job is to weave it into ONE flowing story.

    YOUR EDITING RULES:
    1. THREAD THE RUNNING EXAMPLE — Reference the running_example scenario in
       every major section. Don't just mention it in the intro and abandon it.
       In patterns: "Remember our {running_example}? This pattern is exactly what
       we'd use there." In problems: connect at least one problem back to it.
       In traps: "What if someone tried this with our {running_example}?"

    2. ADD SOCRATIC QUESTIONS — Before every major reveal or insight, add a
       "Pause and think..." prompt. Examples:
       - "Pause and think: what do you think happens when we try this with 0?"
       - "Before reading on, can you guess why this always works?"
       - "What would YOU do if you saw this in a competition?"

    3. REPLACE GENERIC BREAKTHROUGHS — Find any placeholder breakthrough moments
       and replace with SPECIFIC insights from the actual content.
       Bad: "Key insight from this section!"
       Good: "The denominator tells you the SIZE of each piece — that's why
       you can't add fractions with different denominators directly!"

    4. MAP PROBLEMS TO PATTERNS — In problem solutions, explicitly name which
       pattern from the Pattern Library the student should recognize.

    5. CREATE NATURAL TRANSITIONS — Between sections, write 1-2 sentences that
       connect what was just learned to what's coming next. Make it feel like
       a conversation, not a textbook.

    6. CREATE CURIOSITY GAPS — Before revealing something interesting, hint at
       it first: "There's a surprising trick that makes this almost trivial..."
       then build up to the reveal.

    CRITICAL: Keep the same overall structure and LENGTH. Do NOT summarize or shorten.
    Your output must be at LEAST 90% the length of the input. Enhance, don't compress.
    """
    assembled_content: str = dspy.InputField(desc="The assembled lesson content in markdown")
    running_example: str = dspy.InputField(desc="The running example/scenario to thread throughout")
    key_insights: str = dspy.InputField(desc="Key insights to use as specific breakthrough moments, separated by |")
    pattern_names: str = dspy.InputField(desc="Pattern names from Pattern Library to reference in problems, separated by |")
    student_name: str = dspy.InputField(desc="Student's name for personalization")
    topic: str = dspy.InputField(desc="Topic being taught")

    edited_content: str = dspy.OutputField(
        desc="The edited lesson content with Socratic questions, narrative threading, "
        "specific breakthroughs, and natural transitions. Same markdown format. "
        "MUST be at least 90% the length of the input. Do NOT summarize — ENHANCE."
    )
    socratic_questions: str = dspy.OutputField(
        desc="The Socratic questions added to the content, separated by |"
    )
    breakthrough_moments: str = dspy.OutputField(
        desc="Specific breakthrough/insight moments used (not generic), separated by |"
    )


class RankTipsSignature(dspy.Signature):
    """Generate 20-30 DETAILED, actionable tips that will help secure the #1 rank.

    You are a competition coach who has trained multiple gold medalists. Generate
    specific, practical tips for THIS topic that go beyond generic study advice.

    TIP CATEGORIES (mix all of these):
    1. TOPIC-SPECIFIC MASTERY (8-10 tips)
       - Specific techniques, shortcuts, and tricks for this topic
       - Common patterns to memorize and recognize instantly
       - Key formulas/rules to have at fingertips

    2. PROBLEM-SOLVING SPEED (5-7 tips)
       - How to solve problems faster in this topic area
       - Mental math tricks relevant to this topic
       - When to skip and come back vs when to grind

    3. ERROR PREVENTION (4-5 tips)
       - Most common mistakes that cost marks in this topic
       - Quick verification techniques
       - Sanity checks specific to this type of problem

    4. EXAM STRATEGY (3-5 tips)
       - Time allocation for this topic in exams
       - How to read problems to spot key info fast
       - What to write for partial credit

    5. MINDSET & EDGE (2-3 tips)
       - What separates #1 from #2 in this topic area
       - How to stay confident when stuck
       - The one habit that top scorers all share

    Each tip MUST be SPECIFIC, ACTIONABLE, and 2-4 sentences long — not generic advice.
    Bad: "Practice regularly"
    Good: "When adding fractions, always check if denominators share a common factor
    BEFORE finding the LCD — this saves 10-15 seconds per problem. For example, with
    3/12 + 5/18, notice both 12 and 18 are divisible by 6, so LCD is 36, not 216."
    """
    topic: str = dspy.InputField(desc="Topic being studied")
    subject: str = dspy.InputField(desc="Subject area")
    target_level: str = dspy.InputField(desc="Competition/exam level")
    student_name: str = dspy.InputField(desc="Student's name")
    patterns_summary: str = dspy.InputField(desc="Key patterns covered in this lesson")
    mistakes_summary: str = dspy.InputField(desc="Common mistakes identified")

    rank_tips: str = dspy.OutputField(
        desc="20-30 specific, actionable tips separated by |. "
        "Each tip MUST be 2-4 sentences with a concrete example or specific technique. "
        "Number them. No generic advice — every tip must be specific to this topic."
    )


__all__ = [
    'CurriculumArchitectSignature', 'ConceptDecomposerSignature',
    'IntuitionBuilderSignature', 'PatternHunterSignature',
    'ProblemCrafterSignature', 'SolutionStrategistSignature',
    'MistakeAnalyzerSignature', 'ConnectionMapperSignature',
    'ContentAssemblerSignature', 'SingleTopicDeepSignature',
    'NarrativeEditorSignature', 'RankTipsSignature',
]
