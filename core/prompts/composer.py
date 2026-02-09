"""
PromptComposer — composable, model-family-aware prompt builder.

Inspired by Cline's PromptRegistry + PromptBuilder but KISS:
- No template engine, no YAML configs, no plugin system
- Just ordered sections with model-family-aware formatting
- Each section is a plain function (str) → str
- Model family auto-detected from model name string

The five sections (in order):
1. IDENTITY   — Who the agent is
2. TOOLS      — What tools are available + trust levels
3. LEARNING   — Past experience, stigmergy hints, track record
4. CONSTRAINTS — Rules, guardrails, output format
5. TASK       — The actual task to perform

Each section is optional. The composer joins non-empty sections
with model-family-appropriate separators.
"""

import logging
from enum import Enum
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)


class ModelFamily(Enum):
    """Model families with different prompt preferences."""
    CLAUDE = "claude"       # XML structure, verbose OK
    GPT = "gpt"             # Markdown structure, function_calling
    GROQ = "groq"           # Fast inference, minimal prompt preferred
    GEMINI = "gemini"       # Markdown, system instruction slot
    GENERIC = "generic"     # Safe middle ground


def detect_model_family(model: str) -> ModelFamily:
    """Auto-detect model family from model name string. KISS heuristic."""
    m = model.lower()
    if 'claude' in m or 'anthropic' in m or 'sonnet' in m or 'opus' in m or 'haiku' in m:
        return ModelFamily.CLAUDE
    if 'gpt' in m or 'o1' in m or 'o3' in m or 'openai' in m:
        return ModelFamily.GPT
    if 'groq' in m or 'llama' in m or 'mixtral' in m or 'gemma' in m:
        return ModelFamily.GROQ
    if 'gemini' in m or 'google' in m:
        return ModelFamily.GEMINI
    return ModelFamily.GENERIC


# Model-family-specific section separators and wrappers
_SECTION_SEP = {
    ModelFamily.CLAUDE:  "\n\n",        # Claude handles XML sections well
    ModelFamily.GPT:     "\n\n---\n\n", # GPT likes markdown separators
    ModelFamily.GROQ:    "\n\n",        # Minimal
    ModelFamily.GEMINI:  "\n\n",
    ModelFamily.GENERIC: "\n\n---\n\n",
}

# Max prompt length hints per family (chars, not tokens)
_MAX_PROMPT_CHARS = {
    ModelFamily.CLAUDE:  32000,  # Large context, verbose OK
    ModelFamily.GPT:     24000,  # Moderate
    ModelFamily.GROQ:    12000,  # Fast = short
    ModelFamily.GEMINI:  24000,
    ModelFamily.GENERIC: 16000,
}


class PromptComposer:
    """
    Compose agent system prompts from reusable sections.

    Model-family aware: adjusts formatting and verbosity per LLM provider.
    """

    def __init__(self, model: str = ""):
        self.family = detect_model_family(model) if model else ModelFamily.GENERIC
        self.max_chars = _MAX_PROMPT_CHARS[self.family]

    def compose(
        self,
        identity: str = "",
        tools: Optional[List[str]] = None,
        tool_descriptions: Optional[Dict[str, str]] = None,
        trust_levels: Optional[Dict[str, str]] = None,
        learning_context: Optional[List[str]] = None,
        constraints: Optional[List[str]] = None,
        task: str = "",
        extra_sections: Optional[Dict[str, str]] = None,
        workspace_dir: Optional[str] = None,
    ) -> str:
        """
        Compose a full system prompt from sections.

        Args:
            identity: Agent identity/role description
            tools: List of available tool names
            tool_descriptions: tool_name → description mapping
            trust_levels: tool_name → trust_level (safe/side_effect/destructive)
            learning_context: Past experience strings to inject
            constraints: Rules and guardrails
            task: The actual task (goes last)
            extra_sections: Additional named sections

        Returns:
            Composed system prompt string
        """
        sections = []

        # 1. IDENTITY
        if identity:
            sections.append(self._format_identity(identity))

        # 2. TOOLS
        if tools:
            sections.append(self._format_tools(
                tools, tool_descriptions or {}, trust_levels or {}
            ))

        # 3. LEARNING CONTEXT
        if learning_context:
            sections.append(self._format_learning(learning_context))

        # 4. CONSTRAINTS
        if constraints:
            sections.append(self._format_constraints(constraints))

        # 5. PROJECT RULES (.jottyrules / .clinerules / .cursorrules / CLAUDE.md)
        if workspace_dir:
            try:
                from .rules import load_project_rules
                rules = load_project_rules(workspace_dir)
                if rules:
                    sections.append(self._wrap_section("Project Rules", rules))
            except Exception as e:
                logger.debug(f"Rule loading skipped: {e}")

        # 6. EXTRA SECTIONS
        if extra_sections:
            for name, content in extra_sections.items():
                if content.strip():
                    sections.append(self._wrap_section(name, content))

        # 6. TASK (always last)
        if task:
            sections.append(self._format_task(task))

        sep = _SECTION_SEP[self.family]
        prompt = sep.join(s for s in sections if s.strip())

        # Budget guard: compress if over limit
        if len(prompt) > self.max_chars:
            prompt = self._compress_prompt(prompt)

        return prompt

    # =========================================================================
    # SECTION FORMATTERS (model-family aware)
    # =========================================================================

    def _format_identity(self, identity: str) -> str:
        if self.family == ModelFamily.CLAUDE:
            return f"<identity>\n{identity}\n</identity>"
        return f"# Identity\n{identity}"

    def _format_tools(
        self,
        tools: List[str],
        descriptions: Dict[str, str],
        trust_levels: Dict[str, str],
    ) -> str:
        lines = []
        if self.family == ModelFamily.CLAUDE:
            lines.append("<available_tools>")
        else:
            lines.append("# Available Tools")

        for t in tools:
            desc = descriptions.get(t, "")
            trust = trust_levels.get(t, "safe")
            trust_badge = ""
            if trust == "side_effect":
                trust_badge = " [SIDE_EFFECT]"
            elif trust == "destructive":
                trust_badge = " [DESTRUCTIVE - verify before use]"

            if self.family == ModelFamily.GROQ:
                # Minimal: just name + trust
                lines.append(f"- {t}{trust_badge}")
            else:
                if desc:
                    lines.append(f"- **{t}**{trust_badge}: {desc}")
                else:
                    lines.append(f"- **{t}**{trust_badge}")

        if self.family == ModelFamily.CLAUDE:
            lines.append("</available_tools>")

        return "\n".join(lines)

    def _format_learning(self, context: List[str]) -> str:
        if not context:
            return ""

        if self.family == ModelFamily.GROQ:
            # Groq: compress learning to essentials only
            compressed = [c[:200] for c in context[:3]]
            return "## Past Experience\n" + "\n".join(f"- {c}" for c in compressed)

        header = "<learning_context>" if self.family == ModelFamily.CLAUDE else "# Past Experience"
        footer = "</learning_context>" if self.family == ModelFamily.CLAUDE else ""

        lines = [header]
        for c in context:
            lines.append(f"- {c}")
        if footer:
            lines.append(footer)
        return "\n".join(lines)

    def _format_constraints(self, constraints: List[str]) -> str:
        if self.family == ModelFamily.CLAUDE:
            items = "\n".join(f"- {c}" for c in constraints)
            return f"<constraints>\n{items}\n</constraints>"
        items = "\n".join(f"- {c}" for c in constraints)
        return f"# Constraints\n{items}"

    def _format_task(self, task: str) -> str:
        if self.family == ModelFamily.CLAUDE:
            return f"<task>\n{task}\n</task>"
        return f"# Task\n{task}"

    def _wrap_section(self, name: str, content: str) -> str:
        slug = name.lower().replace(' ', '_')
        if self.family == ModelFamily.CLAUDE:
            return f"<{slug}>\n{content}\n</{slug}>"
        return f"# {name}\n{content}"

    # =========================================================================
    # COMPRESSION
    # =========================================================================

    def _compress_prompt(self, prompt: str) -> str:
        """Compress prompt to fit within model family's budget.
        Keep start + end, compress middle (same pattern as SmartContextManager).
        """
        if len(prompt) <= self.max_chars:
            return prompt

        half = self.max_chars // 2
        compressed = (
            prompt[:half]
            + "\n\n[...context compressed to fit model limit...]\n\n"
            + prompt[-half:]
        )
        logger.debug(
            f"Prompt compressed: {len(prompt)} → {len(compressed)} chars "
            f"(family={self.family.value}, limit={self.max_chars})"
        )
        return compressed
