"""
DSPy Signatures for Agentic Planning
=====================================

Extracted from agentic_planner.py for modularity.
Contains all DSPy signature definitions used by TaskPlanner.
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

# ExecutionStepSchema lives in _execution_types (no circular dependency)
if PYDANTIC_AVAILABLE:
    from ._execution_types import ExecutionStepSchema


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

        task_description: str = dspy.InputField(
            desc="The task description to classify. May be vague - classify it anyway."
        )

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

        RESEARCH + SYNTHESIS TASK PATTERN (when task asks to research/analyze THEN create a report/recommendation/summary/ranking):
        1. Research: Use web-search for each entity/topic (one search per entity for quality)
        2. Synthesize: Use claude-cli-llm with prompt that REFERENCES prior outputs:
           params: {"prompt": "Using this research data:\n${search_entity_1}\n${search_entity_2}\n\nCreate a detailed [report/ranking/comparison] in Markdown format."}
           CRITICAL: The prompt MUST include ${output_key} references to inject actual research data. Never ask the LLM to fabricate or simulate data.
        3. Save: Use file-operations/write_file_tool with .md extension:
           params: {"path": "descriptive_name.md", "content": "${synthesis_output}"}
           Do NOT save as .py — synthesis output is text/markdown, not code.

        CODE GENERATION / CREATION TASK PATTERN (when task says "generate", "build", "create", "write" a script/tool/app):
        NEVER pass the task description or LLM output as the shell-exec command. Instead:
        1. Generate: Use claude-cli-llm/generate_code_tool to generate the actual code/script content.
           BEST PRACTICE: generate_code_tool returns clean code in its 'code' field — no fences, no preamble.
           Wire ${output_key.code} directly to write_file_tool's 'content' param for zero extraction issues.
        2. Save: Use file-operations/write_file_tool with params {"path": "<descriptive_name>.py", "content": "${generated_code.code}"}
           File name MUST be descriptive (e.g., "weather_fetcher.py", "github_scraper.py" — NOT "script.py")
        3. Execute: Use shell-exec/execute_command_tool with LITERAL command string:
           params: {"command": "python <descriptive_name>.py"} — NEVER use ${template} for the command!
           The command MUST be a real shell command like "python weather_fetcher.py", NOT "${step_0}" or the task text.
        4. Verify: Use file-operations/read_file_tool to read output files and confirm correctness

        SHELL-EXEC CRITICAL RULES:
        - The "command" param must ALWAYS be a literal shell command (e.g., "python script.py", "ls -la", "curl ...")
        - NEVER set command to "${step_0}" or "${generated_code}" — those resolve to LLM output text, NOT shell commands
        - NEVER set command to the task description text
        - If you need to run a Python script that was saved in step N, use: {"command": "python <filename_from_step_N>.py"}

        FILE-OPERATIONS CRITICAL RULES:
        - write_file_tool requires: {"path": "filename.ext", "content": "..."}
        - read_file_tool requires: {"path": "filename.ext"}
        - The param name is "path" (NOT "file_path", "filepath", or "filename")

        PHASE 3 - PLAN (ReAct):
        For each step, reason about: What could go wrong? What does this step need from prior steps?
        Each step MUST have:
        - skill_name: EXACT name from available_skills (e.g., "web-search", "file-operations")
        - tool_name: EXACT tool name from that skill's "tools" array - COPY IT EXACTLY, never use generic names like "use_skill"
        - params: EXTRACT actual values from task_description! ("Delhi weather" -> {"location": "Delhi"})
        - description: What this step accomplishes
        - verification: How to confirm this step succeeded (e.g., "output contains weather data")
        - fallback_skill: Alternative skill if this one fails (e.g., "http-client")
        - output_key: A DESCRIPTIVE key for this step's output (e.g., "hn_scraper", "search_results", "generated_code" — NOT "step_0")
        - depends_on: List of step INDICES (0-based integers) this step needs

        TOOL OUTPUT SCHEMAS (READ the "returns" field):
        Each tool in available_skills may have a "returns" array listing the output fields it produces.
        Example: get_portfolio_tool returns: [{"name": "holdings", "type": "list"}, {"name": "total_pnl", "type": "float"}]
        Use these field names when referencing step outputs in later steps.
        Example: Step 0 uses get_portfolio_tool -> Step 1 references "${step_0.holdings}" or "${step_0.total_pnl}"
        This ensures correct data wiring between steps.

        STEP I/O CONTRACTS (CRITICAL for correct wiring):
        Each step MUST declare:
        - inputs_needed: Dict mapping each param to its data source.
          Use "step_key.field" for dependency data, "literal:value" for constants.
          Example: {"content": "step_0.holdings", "path": "literal:/tmp/portfolio.pdf"}
        - outputs_produced: List of keys this step's output will contain (copy from the tool's "returns" names).
          Example: ["holdings", "total_value", "total_pnl"]
        The execution engine uses these for SCOPED resolution — only declared sources are checked.

        REFERENCING PREVIOUS STEP OUTPUTS (CRITICAL):
        When a step needs output from a previous step, use ${output_key.field} syntax in params.
        ALWAYS use field-level references when the tool has a "returns" schema:
        - Step 0 has output_key "portfolio" and returns ["holdings", "total_pnl"]
          -> Step 1 references "${portfolio.holdings}" or "${portfolio.total_pnl}"
        - Step 0 has output_key "search_results" and returns ["results", "count"]
          -> Step 1 references "${search_results.results}"
        Only use bare ${output_key} (no field) when the entire output dict is needed as content.
        NEVER use placeholder names like "CONTENT_FROM_STEP_1" — always use ${output_key} or ${output_key.field}.

        HYBRID ACTION ROUTING (when both GUI and API skills are available):
        Skills may have an "executor_type" field: "api", "gui", "hybrid", or "general".
        - "api" skills (http-client, pmi-*, messaging) execute programmatically — fast, reliable
        - "gui" skills (browser-automation, android-automation) interact via UI — slow, fragile
        - ALWAYS prefer API/shortcut skills over GUI skills when both can achieve the subtask
        - Example: "Add item to cart" — use cart API (1 step) NOT tap through UI (5+ steps)
        - Example: "Search for product" — use deep-link/search API NOT type in search bar
        - ONLY use GUI skills when no API alternative exists for that specific action
        - When mixing GUI and API: do API steps first (data fetching), GUI only for steps
          that genuinely require screen interaction (visual verification, manual navigation)

        PHASE 4 - VERIFY (Self-Refine):
        Before outputting, self-check: Are all dependencies satisfied? Any missing steps?
        Are params populated (not empty)? Does every skill_name exist in available_skills?
        Do all inter-step references use ${output_key} syntax?
        If both GUI and API skills are used, verify that API is preferred where possible.

        PARAMETER EXTRACTION (CRITICAL - you will be penalized for empty params):
        - "Delhi weather" -> params: {"location": "Delhi"}
        - "research on Tesla" -> params: {"query": "Tesla", "topic": "Tesla"}
        - "Paytm stock" -> params: {"ticker": "PAYTM", "company_name": "Paytm"}
        - "slides about AI" -> params: {"topic": "AI"}

        FALLBACK: If a needed capability doesn't exist, use "web-search" for research or "file-operations" for creation.
        """

        task_description: str = dspy.InputField(desc="Task to accomplish")
        task_type: str = dspy.InputField(desc="Task type: research, analysis, creation, etc.")
        available_skills: str = dspy.InputField(
            desc="JSON array of available skills. ONLY use skill_name values from this list!"
        )
        previous_outputs: str = dspy.InputField(
            desc="JSON dict of outputs from previous steps. May contain '_learning_guidance' with lessons from past executions — use these to avoid known failures and replicate successful approaches."
        )
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
        reasoning: str = dspy.OutputField(
            desc="Brief explanation of the plan including why alternative approaches were rejected"
        )
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
        max_skills: int = dspy.InputField(desc="Maximum number of skills to select")

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

        CRITICAL PARAMETER RULES (common replan failures):
        - shell-exec/execute_command_tool: "command" must be a LITERAL shell command
          (e.g., {"command": "python weather_fetcher.py"})
          NEVER set command to ${step_0} or the task description text.
        - shell-exec/execute_script_tool: "script" must be the actual Python code string.
          NOT "script_path" — the param name is "script".
        - file-operations/write_file_tool: requires {"path": "...", "content": "..."}
          The param name is "path" (NOT "file_path" or "filepath").
        - file-operations/read_file_tool: requires {"path": "filename.ext"}
          Use the ACTUAL filename, NOT a step reference like "step_2".
        - If a step wrote a file, reference it by its ACTUAL filename in later steps.
        """

        task_description: str = dspy.InputField(desc="Original task to accomplish")
        task_type: str = dspy.InputField(desc="Task type: research, analysis, creation, etc.")
        available_skills: str = dspy.InputField(
            desc="JSON array of available skills (excluding blacklisted ones)"
        )
        failed_steps: str = dspy.InputField(
            desc="JSON array of failed steps with error details: [{skill_name, tool_name, error, params}]"
        )
        completed_outputs: str = dspy.InputField(
            desc="JSON dict of outputs from successfully completed steps"
        )
        excluded_skills: str = dspy.InputField(
            desc="JSON array of skill names to NEVER use (blacklisted due to structural failures)"
        )
        max_steps: int = dspy.InputField(desc="Maximum number of remaining steps")

        if PYDANTIC_AVAILABLE:
            corrected_plan: List[ExecutionStepSchema] = dspy.OutputField(
                desc="List of corrected execution steps that avoid previous failures"
            )
        else:
            corrected_plan: List[dict] = dspy.OutputField(
                desc='Corrected steps array. Each: {"skill_name": "...", "tool_name": "...", "params": {...}, "description": "...", "verification": "...", "fallback_skill": "..."}'
            )
        reflection: str = dspy.OutputField(
            desc="Analysis of WHY each step failed and what to do differently"
        )
        reasoning: str = dspy.OutputField(
            desc="Explanation of the corrected plan and how it avoids previous failures"
        )
