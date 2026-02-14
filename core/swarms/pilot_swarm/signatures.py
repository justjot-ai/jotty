"""Pilot Swarm - DSPy Signatures.

Each signature defines the LLM reasoning contract for one agent:
- PlannerSignature: goal → ordered subtask list
- SearchSignature: question → queries + synthesis
- CoderSignature: task → file operations
- TerminalSignature: task → safe commands
- SkillWriterSignature: description → skill.yaml + tools.py
- ValidatorSignature: results → success assessment
"""

import dspy


class PlannerSignature(dspy.Signature):
    """Decompose a goal into an ordered list of executable subtasks.

    You are an expert strategist. Given any goal, break it into concrete,
    actionable subtasks that specialized agents can execute.

    RULES:
    1. Each subtask has a TYPE: search, code, terminal, create_skill, delegate, analyze, browse
    2. Use 'delegate' when a task clearly matches a specialized swarm (coding, research, etc.)
    3. Keep the plan MINIMAL — fewest subtasks to achieve the goal
    4. Be specific about what each subtask should PRODUCE (not vague "look into...")
    5. Order by dependency — downstream tasks reference upstream results
    6. MAXIMUM 8 subtasks

    TYPE GUIDE:
    - search: find information on the web, look up documentation, research a topic
    - code: write code, create files, edit configurations, generate scripts
    - terminal: run shell commands (install packages, check status, run tests, execute scripts)
    - create_skill: create a new reusable Jotty skill (skill.yaml + tools.py)
    - delegate: hand off to a specialized swarm (coding, research, testing, etc.)
    - analyze: think through / synthesize information using LLM reasoning
    - browse: open a specific URL to read its content, scrape a webpage, OR visually inspect
      images/screenshots/PDFs/PPTX files. Put the URL or file path in tool_hint.
      Use browse when you need the ACTUAL content of a specific URL or file.
    """
    goal: str = dspy.InputField(desc="The goal to accomplish")
    available_swarms: str = dspy.InputField(desc="Available specialized swarms for delegation")
    context: str = dspy.InputField(desc="Previous results or context (empty if first attempt). "
                                        "On re-planning after validation failure, includes completed work and remaining gaps.")

    subtasks_json: str = dspy.OutputField(
        desc="JSON list of subtasks. Each MUST have: "
        "{id (str, e.g. 's1'), type (str — search/code/terminal/create_skill/delegate/analyze/browse), "
        "description (2-3 sentences: what to do and what to produce), "
        "tool_hint (str — suggested tool/skill/swarm name, or empty), "
        "depends_on (list of subtask IDs that must complete first, e.g. ['s1'])}. "
        "MAXIMUM 8 subtasks. Keep it minimal and actionable."
    )
    reasoning: str = dspy.OutputField(
        desc="Brief explanation (2-3 sentences) of the plan's strategy and expected outcome."
    )


class SearchSignature(dspy.Signature):
    """Search for information and synthesize findings.

    Generate targeted search queries, then synthesize what you know
    into a clear, factual answer. Be specific — include numbers, names, dates.
    """
    task: str = dspy.InputField(desc="What information is needed and why")
    context: str = dspy.InputField(desc="Previous results providing context")

    search_queries: str = dspy.OutputField(
        desc="2-3 specific search queries separated by |. "
        "Each query targets a different angle of the information needed. "
        "Be specific: 'Python FastAPI JWT authentication tutorial 2024' not 'auth tutorial'."
    )
    synthesis: str = dspy.OutputField(
        desc="Synthesized answer (MINIMUM 4-6 sentences). "
        "Include specific facts, names, numbers, and actionable details. "
        "Note any uncertainties or areas needing verification."
    )
    key_findings: str = dspy.OutputField(
        desc="3-5 key findings separated by |. Each is one specific fact or actionable insight."
    )


class CoderSignature(dspy.Signature):
    """Generate code or file content to accomplish a task.

    Write clean, production-quality code. Follow existing patterns if context
    shows an existing codebase. Include imports, error handling, and docstrings.

    RULES:
    1. Each file operation specifies the FULL file path
    2. Code must be complete and runnable — no placeholders or unimplemented sections
    3. Include all necessary imports
    4. Handle errors gracefully
    5. Follow the language's conventions (PEP 8 for Python, etc.)
    """
    task: str = dspy.InputField(desc="What code to write, what files to create or edit")
    context: str = dspy.InputField(desc="Relevant context: existing code, requirements, previous results")

    file_operations_json: str = dspy.OutputField(
        desc="JSON list of file operations. Each MUST have: "
        "{file_path (str — full path), "
        "action (str — 'create', 'append', 'read', or 'edit'), "
        "content (str — file content for create/append, new content for edit), "
        "old_content (str — required for 'edit': the exact text to replace), "
        "description (str — what this operation does)}. "
        "Use 'read' to load existing file content for context. "
        "Use 'edit' for surgical replacement of specific text in existing files. "
        "Include ALL necessary files. Code must be complete — no stubs."
    )
    explanation: str = dspy.OutputField(
        desc="Brief explanation of what was created and how it works (2-3 sentences)."
    )


class TerminalSignature(dspy.Signature):
    """Generate shell commands to accomplish a system task.

    SAFETY RULES:
    - Mark commands as safe=true for MOST normal operations:
      running Python scripts, pip install, curl, wget, git,
      creating directories, writing files, running tests, etc.
    - Mark commands as safe=false ONLY for truly DESTRUCTIVE operations:
      rm -rf, drop database, kill -9, format, dd, mkfs, shutdown,
      chmod 777, deleting git branches, force push
    - NEVER generate destructive commands unless explicitly asked
    """
    task: str = dspy.InputField(desc="What to accomplish via terminal")
    context: str = dspy.InputField(desc="Relevant context (OS, previous output, working directory)")

    commands_json: str = dspy.OutputField(
        desc="JSON list of commands. Each MUST have: "
        "{command (str — the shell command to run), "
        "purpose (str — what this command does in plain English), "
        "safe (bool — true for normal operations like running scripts, installing packages, "
        "creating files, git operations; false ONLY for destructive operations like rm -rf, "
        "drop database, kill -9, format disk)}. "
        "Order matters — commands execute sequentially."
    )
    safety_assessment: str = dspy.OutputField(
        desc="Safety assessment (1-2 sentences). List any risks. "
        "If any command modifies system state, explain what changes."
    )


class SkillWriterSignature(dspy.Signature):
    """Create a new Jotty skill with skill.yaml and tools.py.

    A Jotty skill is a reusable capability package:
    1. skill.yaml — metadata: name, description, version, tool list, dependencies
    2. tools.py — Python functions implementing the skill's tools

    TOOL FUNCTION CONTRACT:
    - Each tool function takes a single `params: dict` argument
    - Returns a dict with 'result' key on success, 'error' key on failure
    - Uses the @tool_wrapper decorator for parameter validation
    - Has a clear docstring explaining params and return value
    - Handles exceptions internally
    """
    skill_description: str = dspy.InputField(desc="What the skill should do — be specific about inputs and outputs")
    skill_name: str = dspy.InputField(desc="Skill name in kebab-case (e.g., 'my-cool-skill')")
    reference_patterns: str = dspy.InputField(desc="Example skill patterns showing the expected format")

    skill_yaml: str = dspy.OutputField(
        desc="Complete skill.yaml content in YAML format. Must include: "
        "name, description, version (1.0.0), tools (list of function names from tools.py), "
        "and optionally: dependencies (list of pip packages), category, tags."
    )
    tools_py: str = dspy.OutputField(
        desc="Complete tools.py content. MUST include: "
        "1. All necessary imports at the top "
        "2. Each tool function taking params: dict and returning dict "
        "3. Docstrings with Args/Returns documentation "
        "4. try/except blocks returning {'error': str(e)} on failure "
        "5. Type hints on function signatures "
        "Code must be complete and runnable."
    )
    usage_example: str = dspy.OutputField(
        desc="A brief usage example showing how to call the skill's main tool (2-3 lines of Python)."
    )


class ValidatorSignature(dspy.Signature):
    """Validate whether a goal has been achieved by examining results.

    Be honest and strict. Don't report success if there are clear gaps.
    If partially successful, identify exactly what's missing.
    """
    goal: str = dspy.InputField(desc="The original goal to achieve")
    results_summary: str = dspy.InputField(desc="Summary of all subtask results and their outcomes")

    success: str = dspy.OutputField(
        desc="'true' if the goal is FULLY achieved, 'false' if not. "
        "Be strict — partial completion is 'false'."
    )
    assessment: str = dspy.OutputField(
        desc="Assessment of results (3-5 sentences). What was accomplished? "
        "What's the quality? What gaps remain?"
    )
    remaining_gaps: str = dspy.OutputField(
        desc="Remaining gaps separated by | (empty string if fully successful). "
        "Each gap is a specific, actionable item that would need to be done."
    )


__all__ = [
    'PlannerSignature', 'SearchSignature', 'CoderSignature',
    'TerminalSignature', 'SkillWriterSignature', 'ValidatorSignature',
]
