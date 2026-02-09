"""
DSPy Signatures for Agentic Planning
=====================================

Extracted from agentic_planner.py for modularity.
Contains all DSPy signature definitions used by AgenticPlanner.
"""

from typing import List

try:
    import dspy
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False

try:
    from pydantic import BaseModel
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False

# ExecutionStepSchema is defined in agentic_planner.py BEFORE it imports us,
# so this circular import is safe at module-load time.
if PYDANTIC_AVAILABLE:
    from .agentic_planner import ExecutionStepSchema


if DSPY_AVAILABLE:

    class TaskTypeInferenceSignature(dspy.Signature):
        """Classify the task type from description.

        You are a CLASSIFIER. You MUST classify ANY input - even vague or incomplete ones.
        NEVER ask for clarification. ALWAYS provide a classification.

        CLASSIFICATION GUIDE:
        - creation: Create files, build apps, write code, generate content, make something new
          Examples: "Create a Python file", "Build a todo app", "Write a UI component"
        - research: Search web, find information, discover facts, investigate topics
          Examples: "Research best practices", "Find documentation", "Search for tutorials"
        - comparison: Compare options, vs analysis, evaluate alternatives
          Examples: "Compare React vs Vue", "Which database is better"
        - analysis: Analyze data, evaluate code, review content, assess quality, calculate, compute
          Examples: "Analyze this code", "Review the architecture", "Calculate the answer"
        - communication: Send messages, notify, email, communicate with users
          Examples: "Send an email", "Notify the team", "Post an update"
        - automation: Automate workflows, schedule tasks, set up pipelines
          Examples: "Automate deployment", "Schedule backups", "Set up CI/CD"
        - unknown: ONLY if task is completely unintelligible (random characters, empty)

        CRITICAL RULES:
        1. For ANY math/calculation task -> 'analysis'
        2. For ANY vague task mentioning "answer", "help", "do" -> 'analysis'
        3. For building/creating -> 'creation'
        4. NEVER output questions or ask for clarification
        5. Default to 'analysis' for ambiguous tasks (NOT 'unknown')
        """
        task_description: str = dspy.InputField(desc="The task description to classify. May be vague - classify it anyway.")

        task_type: str = dspy.OutputField(
            desc="Output EXACTLY one word: creation, research, comparison, analysis, communication, automation, or unknown. For vague tasks, default to 'analysis'. NEVER ask questions."
        )
        reasoning: str = dspy.OutputField(
            desc="Brief 1-2 sentence explanation. If task is vague, explain your best guess."
        )
        confidence: float = dspy.OutputField(
            desc="A number between 0.0 and 1.0. Use 0.5 for vague tasks."
        )


    class CapabilityInferenceSignature(dspy.Signature):
        """Infer what capabilities are needed to complete a task.

        You are a CAPABILITY CLASSIFIER. Analyze the task and output what types of
        capabilities are needed to complete it.

        AVAILABLE CAPABILITIES:
        - data-fetch: Get data from external sources (weather, stocks, web, APIs)
        - research: Search and gather information from the web
        - analyze: Process, analyze, or compute data
        - visualize: Create charts, graphs, slides, diagrams
        - document: Create documents, PDFs, reports
        - communicate: Send messages via telegram, slack, email, etc.
        - file-ops: Read, write, or manipulate files
        - code: Write or execute code
        - media: Generate or process images, audio, video

        EXAMPLES:
        - "Delhi weather" -> ["data-fetch"]
        - "Research AI trends and create PDF" -> ["research", "document"]
        - "Stock analysis with charts on telegram" -> ["data-fetch", "analyze", "visualize", "communicate"]
        - "Send meeting notes to slack" -> ["communicate"]

        Output 1-4 capabilities that best match the task requirements.
        """
        task_description: str = dspy.InputField(desc="The task to analyze")

        capabilities: str = dspy.OutputField(
            desc='JSON array of capabilities needed, e.g., ["data-fetch", "communicate"]. Max 4 items.'
        )
        reasoning: str = dspy.OutputField(
            desc="Brief explanation of why these capabilities are needed"
        )


    class ExecutionPlanningSignature(dspy.Signature):
        """Create an executable plan using the available skills. Audience: senior engineer.

        You will be penalized for wrong skill names, empty params, or generic tool names.

        PHASE 1 - UNDERSTAND (Plan-and-Solve):
        Parse the task_description. Extract all variables (names, locations, topics, quantities).
        Identify constraints (output format, delivery channel, dependencies).

        PHASE 2 - EVALUATE (Tree of Thoughts):
        Consider 2-3 possible approaches using available_skills. Assess feasibility of each.
        For SIMPLE tasks: Choose the approach with fewest steps that covers ALL requirements.
        For COMPLEX/COMPARISON tasks: Choose the approach that MAXIMIZES quality — separate
        research per entity, dedicated synthesis step, dedicated output formatting step.
        Quality > speed. More steps = better output for complex tasks.

        COMPARISON TASK PATTERN (when task contains "vs", "compare", "versus", "difference"):
        1. Research Entity A: web-search with targeted query for entity A
        2. Research Entity B: web-search with targeted query for entity B
        3. (Optional) Research Entity C: if 3+ entities
        4. Synthesize: Use claude-cli-llm or summarize to create structured comparison
           (feature matrix, pricing table, pros/cons, recommendation)
        5. Format: Generate final output (PDF, slides, etc.)
        6. Deliver: Send via requested channel (telegram, slack, etc.)
        This pattern produces MUCH higher quality than using a composite skill that does one generic search.

        PHASE 3 - PLAN (ReAct):
        For each step, reason about: What could go wrong? What does this step need from prior steps?
        Each step MUST have:
        - skill_name: EXACT name from available_skills (e.g., "web-search", "file-operations")
        - tool_name: EXACT tool name from that skill's "tools" array - COPY IT EXACTLY, never use generic names like "use_skill"
        - params: EXTRACT actual values from task_description! ("Delhi weather" -> {"location": "Delhi"})
        - description: What this step accomplishes
        - verification: How to confirm this step succeeded (e.g., "output contains weather data")
        - fallback_skill: Alternative skill if this one fails (e.g., "http-client")
        - output_key: A unique key for this step's output (e.g., "step_0", "step_1", "search_results")
        - depends_on: List of step INDICES (0-based integers) this step needs

        REFERENCING PREVIOUS STEP OUTPUTS (CRITICAL):
        When a step needs output from a previous step, use ${output_key} syntax in params.
        Examples:
        - Step 0 has output_key "step_0" -> Step 1 references it as "${step_0}"
        - Step 0 has output_key "search_results" -> Step 1 references it as "${search_results}"
        - To access a specific field: "${step_0.results}" or "${search_results.content}"
        NEVER use placeholder names like "CONTENT_FROM_STEP_1" — always use ${output_key}.

        PHASE 4 - VERIFY (Self-Refine):
        Before outputting, self-check: Are all dependencies satisfied? Any missing steps?
        Are params populated (not empty)? Does every skill_name exist in available_skills?
        Do all inter-step references use ${output_key} syntax?

        PARAMETER EXTRACTION (CRITICAL - you will be penalized for empty params):
        - "Delhi weather" -> params: {"location": "Delhi"}
        - "research on Tesla" -> params: {"query": "Tesla", "topic": "Tesla"}
        - "Paytm stock" -> params: {"ticker": "PAYTM", "company_name": "Paytm"}
        - "slides about AI" -> params: {"topic": "AI"}

        FALLBACK: If a needed capability doesn't exist, use "web-search" for research or "file-operations" for creation.
        """

        task_description: str = dspy.InputField(desc="Task to accomplish")
        task_type: str = dspy.InputField(desc="Task type: research, analysis, creation, etc.")
        available_skills: str = dspy.InputField(desc="JSON array of available skills. ONLY use skill_name values from this list!")
        previous_outputs: str = dspy.InputField(desc="JSON dict of outputs from previous steps")
        max_steps: int = dspy.InputField(desc="Maximum number of steps")

        # Use typed Pydantic model - DSPy JSONAdapter enforces schema
        if PYDANTIC_AVAILABLE:
            execution_plan: List[ExecutionStepSchema] = dspy.OutputField(
                desc="List of execution steps"
            )
        else:
            execution_plan: List[dict] = dspy.OutputField(
                desc='Steps array. Each: {"skill_name": "...", "tool_name": "...", "params": {...}, "description": "..."}'
            )
        reasoning: str = dspy.OutputField(desc="Brief explanation of the plan including why alternative approaches were rejected")
        estimated_complexity: str = dspy.OutputField(desc="simple, medium, or complex")


    class SkillSelectionSignature(dspy.Signature):
        """Select the BEST skills for the task from the full skill catalog.

        Each skill has: name, description, type (base/derived/composite), caps (capabilities).
        - "type":"composite" + "combines":[...] = pre-built workflow using multiple skills
        - "type":"derived" + "extends":"..." = specialized version of a base skill
        - "type":"base" = atomic building block

        SELECTION STRATEGY:
        1. DECOMPOSE the task: subject, action, output format, delivery channel
        2. For SIMPLE single-step tasks -> prefer a composite skill (one skill does it all)
           Example: "Delhi weather on telegram" -> "weather-to-telegram" (composite)
        3. For COMPLEX multi-step tasks -> prefer granular base/derived skills
           Example: "Compare X vs Y, create PDF, send telegram" -> web-search + claude-cli-llm + document-converter + telegram-sender
        4. Match "caps" to task needs: research, data-fetch, analyze, visualize, document, communicate, file-ops, code, media
        5. VERIFY: does the selection cover ALL requirements?

        You are selecting skills, NOT executing. Justify each choice.
        """
        task_description: str = dspy.InputField(desc="The task to complete")
        available_skills: str = dspy.InputField(
            desc="JSON skill catalog. Each: {name, description, type, caps, combines/extends}"
        )
        max_skills: int = dspy.InputField(
            desc="Maximum number of skills to select"
        )

        selected_skills: str = dspy.OutputField(
            desc='Return ONLY a JSON array like ["skill-name-1", "skill-name-2"]. No markdown, no explanation, just the JSON array.'
        )
        reasoning: str = dspy.OutputField(
            desc="Per-skill justification: which sub-requirement each skill satisfies and why alternatives were not chosen"
        )
        skill_priorities: str = dspy.OutputField(
            desc="JSON dict mapping skill names to priority (0.0-1.0). Higher = execute earlier. Order by logical workflow."
        )


    class ReflectivePlanningSignature(dspy.Signature):
        """Replan after failure using Reflexion-style analysis. Audience: senior engineer.

        You will be penalized for repeating the same failed approach or ignoring completed work.

        STEP 1 - REFLECT:
        Analyze WHY each failed step failed. Categorize: wrong skill? bad params? missing dependency? service down?

        STEP 2 - ADAPT:
        Do NOT retry the same skill/tool combination that failed structurally.
        Use excluded_skills to avoid blacklisted skills entirely.
        Preserve outputs from completed_outputs - do NOT redo successful work.

        STEP 3 - REPLAN:
        Create a corrected plan that routes around the failures.
        Each step MUST have: skill_name, tool_name, params, description, verification, fallback_skill.
        Only plan the REMAINING work - completed steps should not be repeated.
        """
        task_description: str = dspy.InputField(desc="Original task to accomplish")
        task_type: str = dspy.InputField(desc="Task type: research, analysis, creation, etc.")
        available_skills: str = dspy.InputField(desc="JSON array of available skills (excluding blacklisted ones)")
        failed_steps: str = dspy.InputField(desc="JSON array of failed steps with error details: [{skill_name, tool_name, error, params}]")
        completed_outputs: str = dspy.InputField(desc="JSON dict of outputs from successfully completed steps")
        excluded_skills: str = dspy.InputField(desc="JSON array of skill names to NEVER use (blacklisted due to structural failures)")
        max_steps: int = dspy.InputField(desc="Maximum number of remaining steps")

        if PYDANTIC_AVAILABLE:
            corrected_plan: List[ExecutionStepSchema] = dspy.OutputField(
                desc="List of corrected execution steps that avoid previous failures"
            )
        else:
            corrected_plan: List[dict] = dspy.OutputField(
                desc='Corrected steps array. Each: {"skill_name": "...", "tool_name": "...", "params": {...}, "description": "...", "verification": "...", "fallback_skill": "..."}'
            )
        reflection: str = dspy.OutputField(desc="Analysis of WHY each step failed and what to do differently")
        reasoning: str = dspy.OutputField(desc="Explanation of the corrected plan and how it avoids previous failures")
