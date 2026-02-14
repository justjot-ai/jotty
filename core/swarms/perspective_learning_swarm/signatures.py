"""Perspective Learning Swarm - DSPy Signatures.

Each signature embeds multi-perspective teaching philosophy:
- Start with WHY — why does this matter to a child's life?
- Use the student's name for personalization
- Age-appropriate language and examples
- Running example threading across all perspectives
- Minimum output lengths enforced (no terse responses)
- Celebrate insight moments

Two embedded teaching styles (never named explicitly):
1. Visual/intuitive step-by-step discovery (StatQuest-inspired)
2. Structured framework/mental-model thinking (PM-inspired)

OUTPUT LENGTH CONTRACT:
Every output field specifies a minimum length. The LLM MUST meet these
minimums to produce world-class content. Short, terse answers are never
acceptable — every field should be elaborate, detailed, and richly specific.
"""

import dspy


class CurriculumDesignerSignature(dspy.Signature):
    """Design a complete, elaborately detailed multi-perspective learning plan for ANY topic.

    You are the world's best IB/PYP curriculum designer. Your plan must enable a student
    to explore a topic from EVERY angle — visual, narrative, critical, hands-on, real-world,
    and structured — while making connections across subjects and languages.

    1. IDENTIFY THE CENTRAL IDEA — Distill the topic into one powerful statement
       (e.g., "Media is a tool that influences the decisions people make").

    2. DEFINE LEARNING OBJECTIVES — 6-8 specific, measurable objectives covering
       knowledge, skills, and attitudes across multiple perspectives.

    3. EXTRACT KEY CONCEPTS — 4-6 concepts with kid-friendly descriptions that
       will be explored from all 6 perspectives.

    4. CREATE A RUNNING EXAMPLE — A vivid, engaging real-world scenario (8-10 sentences)
       that will be threaded through EVERY perspective. Include specific character names,
       a setting, concrete details, and a narrative arc.

    5. BUILD VOCABULARY — 10-15 key terms with age-appropriate definitions that will
       be translated across all 4 languages.

    6. PLAN SECTION DEPTH — How many words/paragraphs each perspective section needs.

    7. MAP TRANSDISCIPLINARY CONNECTIONS — How this topic connects to other subjects
       (math, science, art, social studies, language arts).

    Think: "If I had to help a child understand this from every possible angle,
    what's the perfect plan so they discover it themselves?"
    """
    topic: str = dspy.InputField(desc="The topic to teach (e.g., 'Media and its influence on decisions')")
    student_name: str = dspy.InputField(desc="Student's name for personalization")
    age_group: str = dspy.InputField(desc="Age group: early_primary, primary, middle, high, general")
    central_idea: str = dspy.InputField(desc="Optional central idea/statement to explore (leave empty to auto-generate)")

    learning_objectives: str = dspy.OutputField(
        desc="6-8 specific learning objectives separated by |. Each objective should be "
        "measurable and cover different Bloom's levels (know, understand, apply, analyze, "
        "evaluate, create). Example: 'Identify at least 3 types of media and explain how "
        "each one tries to influence decisions | Analyze an advertisement to spot hidden "
        "persuasion techniques | Create a poster presenting both sides of a media topic'"
    )
    key_concepts_json: str = dspy.OutputField(
        desc="JSON list of 4-6 key concepts. EACH concept MUST have: "
        "{name (str), description (3-4 sentences, age-appropriate, explaining what this "
        "concept means and why it matters in the student's daily life), "
        "perspective_hooks (a brief note on how each of the 6 perspectives can explore this)}. "
        "Be thorough — these concepts drive ALL downstream content generation."
    )
    running_example_scenario: str = dspy.OutputField(
        desc="A vivid, engaging real-world scenario (8-10 sentences minimum) that will be "
        "threaded through EVERY perspective of the lesson. Include specific character names, "
        "a setting, concrete details, and a narrative arc. The scenario must be relatable "
        "to the student's age group and naturally illustrate the core concepts. "
        "Example: 'Riya is scrolling through her tablet when she sees a colorful ad for "
        "a new chocolate brand. It says: \"Kids who eat ChocoBurst are smarter!\" Her friend "
        "Aarav says it must be true because a famous cricketer is holding the chocolate. "
        "But Riya's older sister Priya asks: \"Who paid for this ad? Do they have proof?\" "
        "Now Riya is confused — should she believe the ad or question it?...'"
    )
    vocabulary_json: str = dspy.OutputField(
        desc="JSON list of 10-15 key terms. EACH term MUST have: "
        "{term (str), definition (2-3 sentences, kid-friendly, with a relatable example), "
        "example_in_context (1 sentence showing the term used in the running example)}. "
        "Terms should span all key concepts and be useful for multilingual translation."
    )
    section_depth_plan: str = dspy.OutputField(
        desc="Content depth targets as pipe-separated plan: "
        "'Intuitive Visual: 500 words, step-by-step discovery with concrete examples | "
        "Structured Framework: 500 words, 2-3 mental models with visual layouts | "
        "Storytelling: 600 words, complete narrative arc | "
        "Debate: 500 words, central question with 3-4 arguments each side | "
        "Hands-On: 500 words, 3-4 activities with materials and steps | "
        "Real-World: 400 words, daily life + career + current events | "
        "Each Language: 300 words summary + vocabulary + reflections'. "
        "Total target: 5000-8000 words of world-class content."
    )
    transdisciplinary_connections: str = dspy.OutputField(
        desc="How this topic connects to other subjects (MINIMUM 4-5 sentences). "
        "Map connections to math, science, art, social studies, language arts, and "
        "physical education where relevant. Be specific: 'In math, students can "
        "analyze survey data about media habits using bar graphs and percentages...'"
    )


class IntuitiveExplainerSignature(dspy.Signature):
    """Explain the topic like you're drawing on a whiteboard for a curious child.

    Start with ONE tiny, concrete example. Build up gradually. Use specific numbers
    and details, not abstractions. Draw pictures with words. Make every step feel
    like "well, obviously..." to the student.

    APPROACH:
    1. Start with the SIMPLEST possible example — something from the student's daily life
    2. Build up step-by-step, adding ONE new idea at each step
    3. Describe what to visualize at each step (like drawing on a whiteboard)
    4. Lead to the "aha!" moment where the big idea clicks
    5. Include "check your understanding" questions

    The student should feel like they DISCOVERED the idea, not that it was lectured at them.
    """
    topic: str = dspy.InputField(desc="Topic to explain")
    concepts: str = dspy.InputField(desc="Key concepts (JSON or comma-separated)")
    student_name: str = dspy.InputField(desc="Student's name")
    age_group: str = dspy.InputField(desc="Age group")
    running_example: str = dspy.InputField(desc="Running example scenario to reference")

    simplest_example: str = dspy.OutputField(
        desc="The absolute simplest concrete example (MINIMUM 5-7 sentences). "
        "Use the student's name. Reference the running example. "
        "Walk through it step by step like a patient tutor."
    )
    step_by_step_build: str = dspy.OutputField(
        desc="6-8 progressive steps building understanding (MINIMUM 12-16 sentences total). "
        "Each step adds ONE new idea. Use 'Step 1: ... Step 2: ...' format on separate lines. "
        "Include specific details and 'What do you think happens next?' prompts."
    )
    visual_descriptions: str = dspy.OutputField(
        desc="3-4 'imagine this picture' descriptions separated by |. Each MUST be 3-4 sentences "
        "describing exactly what to visualize — shapes, colors, arrows, labels. "
        "Specific enough that someone could draw it from the description."
    )
    aha_moment: str = dspy.OutputField(
        desc="The insight that makes everything click (MINIMUM 3-4 sentences). "
        "State it as a revelation: 'And here's the amazing part...' "
        "This should feel like the climax of a story."
    )
    check_your_understanding: str = dspy.OutputField(
        desc="2-3 quick questions separated by | to verify understanding. "
        "Each question should be answerable from the content above and test "
        "whether the student truly understood the concept, not just memorized it."
    )


class FrameworkBuilderSignature(dspy.Signature):
    """Think in structures. Every complex idea can be broken into a framework.

    Use 2x2 matrices, decision trees, prioritization lists, comparison tables.
    Make thinking systematic and organized. Give the student TOOLS for thinking,
    not just information.

    APPROACH:
    1. Identify 2-3 mental models that help organize this topic
    2. For each model, explain it simply, show its visual layout, and apply it
    3. Create a decision tree for navigating the topic's key questions
    4. Build a comparison matrix that organizes ideas into clear categories
    5. Distill 3-5 key principles or rules
    6. Create a step-by-step thinking checklist

    The student should feel empowered: "Now I have a SYSTEM for thinking about this!"
    """
    topic: str = dspy.InputField(desc="Topic to build frameworks for")
    concepts: str = dspy.InputField(desc="Key concepts")
    student_name: str = dspy.InputField(desc="Student's name")
    age_group: str = dspy.InputField(desc="Age group")
    running_example: str = dspy.InputField(desc="Running example scenario")

    frameworks_json: str = dspy.OutputField(
        desc="JSON list of 2-3 mental models. EACH model MUST have: "
        "{name (str), description (3-4 sentences explaining the framework), "
        "how_to_use (2-3 sentences on when and how to apply it), "
        "visual_layout (describe the visual structure — grid, tree, matrix, etc. in 3-4 sentences), "
        "example_applied (apply the framework to the running example in 3-4 sentences)}. "
        "Make frameworks age-appropriate but genuinely useful for thinking."
    )
    decision_tree: str = dspy.OutputField(
        desc="A decision tree for navigating this topic (MINIMUM 6-8 sentences). "
        "Format as: 'Start here: [question] -> If YES: [path] -> If NO: [path]'. "
        "Include 3-4 decision points. Apply it to the running example."
    )
    comparison_matrix: str = dspy.OutputField(
        desc="A comparison table organizing key ideas (MINIMUM 6-8 sentences). "
        "Define the categories/dimensions, then classify each key concept. "
        "Format clearly so a student could draw it as a table."
    )
    key_principles: str = dspy.OutputField(
        desc="3-5 key rules or principles separated by |. Each MUST be 2-3 sentences: "
        "state the principle clearly, then give a specific example from the topic."
    )
    thinking_checklist: str = dspy.OutputField(
        desc="A step-by-step thinking process (5-7 steps separated by |). "
        "Each step is 1-2 sentences. The student can use this checklist "
        "whenever they encounter this type of topic or question."
    )


class StorytellerSignature(dspy.Signature):
    """Every concept has a story. Create characters, conflict, resolution.

    The student should FEEL the concept, not just know it. Use narrative to make
    abstract ideas concrete and memorable. The story should have a clear beginning,
    middle, and end — with the concept woven naturally into the plot.

    APPROACH:
    1. Create 2-3 relatable characters (use the running example if possible)
    2. Establish a setting and situation that naturally involves the topic
    3. Build CONFLICT that requires understanding the concept to resolve
    4. Let the characters discover the concept through their journey
    5. Resolve the conflict with the concept as the key insight
    6. End with a moral/lesson and questions for reflection
    """
    topic: str = dspy.InputField(desc="Topic to tell a story about")
    concepts: str = dspy.InputField(desc="Key concepts to weave into the story")
    student_name: str = dspy.InputField(desc="Student's name")
    age_group: str = dspy.InputField(desc="Age group")
    running_example: str = dspy.InputField(desc="Running example scenario to build on")

    story: str = dspy.OutputField(
        desc="A complete story (MINIMUM 600-800 words) with a narrative arc. "
        "Include at least 3 scenes with different settings. "
        "The student ({student_name}) should be the protagonist or a central character. "
        "Include dialogue, sensory details, and emotional moments. "
        "The concept should be woven naturally — never feel like a lecture. "
        "Use paragraph breaks. End with a clear resolution."
    )
    characters_json: str = dspy.OutputField(
        desc="JSON list of 2-3 characters. EACH character MUST have: "
        "{name (str), role (str — e.g., 'the questioner', 'the believer'), "
        "personality (2 sentences), connection_to_concept (how they embody or "
        "challenge the concept)}."
    )
    moral_or_lesson: str = dspy.OutputField(
        desc="The moral or lesson of the story (MINIMUM 3-4 sentences). "
        "State what the characters learned and how it connects to the bigger idea. "
        "Make it feel earned — not preachy."
    )
    discussion_questions: str = dspy.OutputField(
        desc="3-4 discussion questions separated by |. Each question should connect "
        "the story to the student's own life. Use 'What would YOU do if...' format."
    )
    connect_to_life: str = dspy.OutputField(
        desc="How does this story relate to the student's own life? (MINIMUM 3-4 sentences). "
        "Use the student's name. Give specific, relatable examples from a child's "
        "daily experience — school, friends, family, hobbies."
    )


class DebateArchitectSignature(dspy.Signature):
    """Critical thinking is a superpower. Present both sides fairly.

    Teach students to spot bias, ask tough questions, and form their own opinions.
    Never tell the student what to think — give them the tools to think for themselves.

    CRITICAL: Both sides must be EQUALLY compelling. Do NOT bias toward one position.
    A reader should genuinely struggle to pick a side after reading both arguments.

    APPROACH:
    1. Frame a CENTRAL DEBATABLE QUESTION (not a factual question)
    2. Present 3-4 strong arguments FOR one side with evidence
    3. Present 3-4 EQUALLY STRONG arguments AGAINST with evidence
    4. Teach bias-spotting techniques
    5. Guide the student to form their OWN opinion with reasoning
    6. Provide critical questions that challenge both sides
    """
    topic: str = dspy.InputField(desc="Topic to debate")
    concepts: str = dspy.InputField(desc="Key concepts")
    student_name: str = dspy.InputField(desc="Student's name")
    age_group: str = dspy.InputField(desc="Age group")
    running_example: str = dspy.InputField(desc="Running example scenario")

    central_question: str = dspy.OutputField(
        desc="The big debatable question (MINIMUM 2-3 sentences). Frame it as a "
        "genuine dilemma with no easy answer. Connect to the running example. "
        "Example: 'Should children be allowed to watch any advertisement they want, "
        "or should parents and schools filter what they see?'"
    )
    points_for_json: str = dspy.OutputField(
        desc="JSON list of 3-4 DISTINCT arguments FOR one side. EACH argument MUST have: "
        "{position (str — the claim), argument (2-3 sentences explaining the reasoning), "
        "evidence (1-2 sentences with a REAL example, statistic, or named scenario — not generic), "
        "counterargument (1-2 sentences acknowledging a weakness in this argument), "
        "critical_question (a question that tests this argument)}. "
        "Present this side as GENUINELY compelling — a reader should find these arguments persuasive."
    )
    points_against_json: str = dspy.OutputField(
        desc="JSON list of 3-4 EQUALLY STRONG arguments for the opposing side. Same structure. "
        "Do NOT make this side weaker — both sides must be fair and well-evidenced. "
        "Each argument needs its own REAL evidence/statistics, not just opinions. "
        "A reader should find THESE arguments just as persuasive as the FOR arguments."
    )
    bias_spotting_tips: str = dspy.OutputField(
        desc="3-4 bias-spotting tips separated by |. Each tip MUST be 2-3 sentences "
        "with a specific example from the topic. Teach students to ask: "
        "'Who benefits from this message? What are they NOT telling me? "
        "Would I feel differently if this came from someone else?'"
    )
    form_your_opinion: str = dspy.OutputField(
        desc="A guided reflection prompt (MINIMUM 4-5 sentences). Walk the student "
        "through forming their own opinion: 'Now that you've heard both sides, "
        "{student_name}, what do YOU think? Here's how to decide...' "
        "Include a simple framework for weighing arguments."
    )
    critical_questions: str = dspy.OutputField(
        desc="4-5 critical thinking questions separated by |. These should challenge "
        "the student to think deeper — questions with no easy answer that spark "
        "genuine curiosity and discussion."
    )


class ProjectDesignerSignature(dspy.Signature):
    """You learn by doing. Design activities that make concepts stick.

    Poster-making, role-play, experiments, surveys, presentations — hands-on
    activities that bring the topic to life and make learning tangible.

    APPROACH:
    1. Design 3-4 diverse activities (different learning styles)
    2. Include materials lists, step-by-step instructions, and assessment criteria
    3. Create a poster design brief with specific layout instructions
    4. Design a role-play scenario with characters and situations
    5. Create a presentation outline the student can use
    6. End with a reflection activity
    """
    topic: str = dspy.InputField(desc="Topic for hands-on projects")
    concepts: str = dspy.InputField(desc="Key concepts to reinforce")
    student_name: str = dspy.InputField(desc="Student's name")
    age_group: str = dspy.InputField(desc="Age group")
    running_example: str = dspy.InputField(desc="Running example scenario")

    projects_json: str = dspy.OutputField(
        desc="JSON list of 3-4 project activities. Each project must be DIFFERENT in type "
        "(e.g., poster, roleplay, experiment, survey/interview — NO duplicate types). "
        "EACH project MUST have: "
        "{title (str), type (str — one of: poster, roleplay, experiment, survey, presentation, craft), "
        "description (2-3 sentences), "
        "materials (list of materials needed — keep simple and accessible), "
        "steps (list of 4-6 clear steps), "
        "learning_outcome (1-2 sentences — what the student will learn), "
        "assessment_criteria (1-2 sentences — how to know if it worked)}. "
        "Activities should be doable at home with common materials."
    )
    poster_design_brief: str = dspy.OutputField(
        desc="A detailed poster design brief (MINIMUM 5-7 sentences). "
        "Include specific layout with sections: what goes at the TOP (title, slogan), "
        "MIDDLE (main content, drawings), and BOTTOM (conclusion, call to action). "
        "Specify what to draw/write in each section, and suggested colors. "
        "Example: 'Title at top in large blue letters. Draw 3 boxes in the middle row, "
        "each showing a different type of media...'"
    )
    role_play_scenario: str = dspy.OutputField(
        desc="A role-play scenario (MINIMUM 4-5 sentences). Define: "
        "2-3 characters with roles, the situation/setting, what each character "
        "should argue/present, and how it ends. Connect to the running example."
    )
    presentation_outline: str = dspy.OutputField(
        desc="A presentation outline for the student (MINIMUM 5-6 points separated by |). "
        "Each point is 1-2 sentences covering: opening hook, key points, "
        "example/story, activity for audience, conclusion, call to action."
    )
    reflection_activity: str = dspy.OutputField(
        desc="A reflection activity (MINIMUM 3-4 sentences). Something the student "
        "does AFTER the project to consolidate learning — journaling prompt, "
        "self-assessment questions, or a letter to a friend explaining what they learned."
    )


class RealWorldConnectorSignature(dspy.Signature):
    """Nothing is abstract. Every concept lives in the real world.

    Connect the topic to careers, news, daily decisions, and future impact.
    Show the student that what they're learning matters RIGHT NOW.

    APPROACH:
    1. Find 4-5 examples from daily life where this topic appears
    2. Connect to 3-4 careers where this knowledge is essential
    3. Link to 2-3 current events or recent news stories
    4. Show how this shapes the student's future
    5. Provide interview questions for parents/guests
    """
    topic: str = dspy.InputField(desc="Topic to connect to real world")
    concepts: str = dspy.InputField(desc="Key concepts")
    student_name: str = dspy.InputField(desc="Student's name")
    age_group: str = dspy.InputField(desc="Age group")
    running_example: str = dspy.InputField(desc="Running example scenario")

    daily_life_connections: str = dspy.OutputField(
        desc="5-7 SPECIFIC real scenarios separated by |. Each MUST be 2-3 sentences "
        "with names, places, and concrete situations (not generic). Be specific: "
        "'When you see a YouTube ad before your favorite video, that's media trying "
        "to influence you — the company PAID to put that ad there because they know "
        "kids your age watch these videos.' Include morning, school, after-school, "
        "and weekend scenarios."
    )
    career_connections: str = dspy.OutputField(
        desc="3-4 career connections separated by |. Each MUST be 3-4 sentences "
        "explaining how this topic matters in a real job. Include: job title, "
        "what they ACTUALLY DO day-to-day, and how this topic appears in their work. "
        "Make careers sound exciting and specific — not just the job title."
    )
    current_events_link: str = dspy.OutputField(
        desc="3-4 current events or recent examples separated by |. Each MUST be "
        "2-3 sentences connecting a real news story or cultural event to the topic. "
        "Include approximate dates and specific details. "
        "Keep it age-appropriate and engaging."
    )
    future_impact: str = dspy.OutputField(
        desc="How this topic shapes the student's future (MINIMUM 6-8 sentences). "
        "Use the student's name. Paint a vivid picture of how understanding this topic "
        "will help them make better decisions, have more opportunities, or understand "
        "the world better as they grow up. Include specific scenarios at ages 15, 18, "
        "and as an adult."
    )
    interview_questions: str = dspy.OutputField(
        desc="4-5 interview questions separated by | that the student can ask parents, "
        "family members, or guests to discuss this topic. Each question should spark "
        "a meaningful conversation. Example: 'Have you ever bought something because "
        "of an ad and later regretted it? What happened?'"
    )


class MultilingualContentSignature(dspy.Signature):
    """Learning in multiple languages deepens understanding.

    Each language brings its own cultural perspective. Generate content that isn't
    just translated — it's culturally adapted for that language's context.

    IMPORTANT: Generate ALL content in the TARGET LANGUAGE, not English.
    Only keep the topic name in English for reference.

    For Hindi: Use Devanagari script. Focus on essay writing and slogan creation.
    For Kannada: Use Kannada script. Focus on role-play and poster activities.
    For French: Use proper French. Focus on persuasive writing and debate.
    """
    topic: str = dspy.InputField(desc="Topic (in English for reference)")
    key_concepts: str = dspy.InputField(desc="Key concepts to cover")
    vocabulary: str = dspy.InputField(desc="Key vocabulary terms to translate")
    student_name: str = dspy.InputField(desc="Student's name")
    running_example: str = dspy.InputField(desc="Running example scenario")
    target_language: str = dspy.InputField(desc="Target language: hindi, kannada, or french")

    summary: str = dspy.OutputField(
        desc="A comprehensive summary of the topic IN THE TARGET LANGUAGE "
        "(MINIMUM 200-300 words). Cover the main ideas, why they matter, "
        "and connect to the student's life. Use age-appropriate language. "
        "For Hindi: write in Devanagari. For Kannada: write in Kannada script. "
        "For French: write in proper French."
    )
    key_vocabulary_translated: str = dspy.OutputField(
        desc="10-15 key terms translated into the target language, separated by |. "
        "Format: 'English term = target language term (brief definition in target language)'. "
        "Include pronunciation guide where helpful."
    )
    reflection_prompts: str = dspy.OutputField(
        desc="3-4 reflection prompts IN THE TARGET LANGUAGE, separated by |. "
        "These should encourage the student to think deeply about the topic "
        "in the cultural context of that language."
    )
    activity: str = dspy.OutputField(
        desc="A language-specific activity IN THE TARGET LANGUAGE (MINIMUM 4-5 sentences). "
        "For Hindi: essay writing or slogan creation. "
        "For Kannada: role-play script or poster text. "
        "For French: persuasive writing or debate preparation. "
        "Include clear instructions."
    )
    slogans: str = dspy.OutputField(
        desc="2-3 catchy slogans or taglines IN THE TARGET LANGUAGE, separated by |. "
        "These should capture the essence of the topic in a memorable way. "
        "Think: campaign slogans, poster headlines, rally cries."
    )


class ContentAssemblerSignature(dspy.Signature):
    """Weave all 6 perspectives + 3 languages into ONE cohesive learning document.

    You are assembling a masterclass. The document must FLOW like a guided journey,
    not read like 9 separate documents stapled together.

    CRITICAL RULES:
    - Thread the running_example through EVERY section
    - Write NATURAL TRANSITIONS between perspectives
    - Build on what came before — "Now that you've SEEN it, let's THINK about it..."
    - Use the student's name throughout
    - Include a Table of Contents
    - End with key insights that synthesize ALL perspectives

    STRUCTURE (target 5000-8000 words total):
    1. WHY THIS MATTERS (200-300 words) — Running example + learning objectives
    2. SEE IT CLEARLY — Intuitive visual perspective
    3. THINK IT THROUGH — Structured frameworks perspective
    4. FEEL THE STORY — Narrative perspective
    5. DEBATE IT — Critical thinking perspective
    6. BUILD IT — Hands-on project perspective
    7. LIVE IT — Real-world connections perspective
    8-10. LANGUAGE SECTIONS — Hindi, Kannada, French
    11. PARENT'S GUIDE — Tips for parents
    12. KEY INSIGHTS & REFLECTION — Synthesis
    """
    student_name: str = dspy.InputField(desc="Student's name")
    topic: str = dspy.InputField(desc="Topic being taught")
    central_idea: str = dspy.InputField(desc="Central idea statement")
    running_example: str = dspy.InputField(desc="Running example scenario")
    learning_objectives: str = dspy.InputField(desc="Learning objectives")
    intuitive_content: str = dspy.InputField(desc="Intuitive/visual perspective content")
    framework_content: str = dspy.InputField(desc="Structured framework perspective content")
    story_content: str = dspy.InputField(desc="Storytelling perspective content")
    debate_content: str = dspy.InputField(desc="Debate/critical thinking perspective content")
    project_content: str = dspy.InputField(desc="Hands-on project perspective content")
    real_world_content: str = dspy.InputField(desc="Real-world connections content")
    celebration_word: str = dspy.InputField(desc="Word to celebrate insights")

    assembled_content: str = dspy.OutputField(
        desc="Complete assembled lesson content in markdown format. "
        "MINIMUM 3000 words. Follow the 12-section structure above. "
        "Include section headings, transitions between perspectives, "
        "and running example references throughout."
    )
    table_of_contents: str = dspy.OutputField(
        desc="A numbered table of contents with all sections, separated by |."
    )
    key_insights: str = dspy.OutputField(
        desc="5-7 key insights that synthesize ALL perspectives, separated by |. "
        "Each insight MUST be a specific, memorable statement (2-3 sentences). "
        "These should show how different perspectives illuminate the SAME truth."
    )


class NarrativeEditorSignature(dspy.Signature):
    """Generate supplementary content for a multi-perspective lesson.

    You receive a SUMMARY of a lesson that was assembled by multiple agents.
    Your job is NOT to rewrite the content — it is to produce three supplementary pieces:

    1. SOCRATIC QUESTIONS — 5-7 thought-provoking questions that a teacher or parent
       could ask at key moments during the lesson. Each question should challenge the
       student to think deeper, not just recall facts.

    2. PARENT'S GUIDE — A practical guide for parents to support learning at home.
       Include conversation starters, activities, and what to look for.

    3. KEY TAKEAWAYS — 5-7 memorable bullet points that synthesize the most important
       lessons from ALL perspectives. Each takeaway should be one sentence that a
       student could remember and apply.
    """
    content_summary: str = dspy.InputField(desc="Summary of the lesson content (key concepts and themes)")
    running_example: str = dspy.InputField(desc="The running example scenario used throughout the lesson")
    student_name: str = dspy.InputField(desc="Student's name")
    topic: str = dspy.InputField(desc="Topic being taught")

    socratic_questions: str = dspy.OutputField(
        desc="5-7 Socratic questions separated by |. Each question should challenge "
        "the student to think beyond the surface. Use formats like: "
        "'What would happen if...', 'Why do you think...', 'How is this different from...', "
        "'What evidence would change your mind about...'. "
        "Questions should span all perspectives (visual, narrative, critical, real-world)."
    )
    parent_guide: str = dspy.OutputField(
        desc="A parent's guide (MINIMUM 200-300 words, 8-10 sentences). Include: "
        "1. Key conversation starters for discussing this topic at home (3-4 specific questions) "
        "2. Activities parents can do with their child (2-3 concrete activities with steps) "
        "3. What to look for in the student's understanding (signs of comprehension) "
        "4. How to extend the learning beyond this lesson (books, videos, real-world observations)"
    )
    key_takeaways: str = dspy.OutputField(
        desc="5-7 key takeaways separated by |. Each takeaway MUST be a single memorable "
        "sentence that synthesizes a lesson from the content. Cover different perspectives: "
        "one from the visual/intuitive angle, one from the story, one from the debate, "
        "one from real-world connections, etc. These should feel like 'aha!' moments "
        "the student can carry with them."
    )


__all__ = [
    'CurriculumDesignerSignature', 'IntuitiveExplainerSignature',
    'FrameworkBuilderSignature', 'StorytellerSignature',
    'DebateArchitectSignature', 'ProjectDesignerSignature',
    'RealWorldConnectorSignature', 'MultilingualContentSignature',
    'ContentAssemblerSignature', 'NarrativeEditorSignature',
]
