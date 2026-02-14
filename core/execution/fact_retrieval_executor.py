"""
Fact-Retrieval Executor
=======================

Optimized executor for direct Q&A tasks like GAIA benchmark.
Designed for perfect accuracy on fact-retrieval questions.

Key Features:
- Multi-step question decomposition
- Direct tool access (no swarm indirection)
- Exact answer extraction
- Format-aware output
"""

import re
import logging
from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class AnswerFormat(Enum):
    """Expected answer format types."""

    TEXT = "text"            # Free-form text
    NUMBER = "number"        # Numeric value
    DATE = "date"            # Date or year
    PERSON = "person"        # Person name
    LOCATION = "location"    # Place or location
    YES_NO = "yes_no"        # Boolean answer
    LIST = "list"            # Multiple items


@dataclass
class QuestionAnalysis:
    """Analysis of question structure and requirements."""

    is_multi_hop: bool
    answer_format: AnswerFormat
    domain: str
    tools_needed: List[str]
    complexity: str  # simple, medium, complex


@dataclass
class ExecutionStep:
    """Single step in multi-step execution."""

    text: str
    depends_on: List[int]  # Indices of previous steps
    tools: List[str]
    result: Optional[Any] = None


class FactRetrievalExecutor:
    """Optimized executor for fact-retrieval tasks (GAIA benchmark)."""

    def __init__(self, registry=None, lm=None):
        """
        Initialize executor.

        Args:
            registry: Skill registry for tool access
            lm: Language model for execution
        """
        self._registry = registry
        self._lm = lm

    @property
    def registry(self):
        """Lazy-load registry."""
        if self._registry is None:
            from Jotty.core.registry import get_unified_registry
            self._registry = get_unified_registry()
        return self._registry

    @property
    def lm(self):
        """Lazy-load language model."""
        if self._lm is None:
            from Jotty.core.foundation.unified_lm_provider import get_lm
            self._lm = get_lm(model='gpt-4o', temperature=0.0)  # Deterministic
        return self._lm

    async def execute(
        self,
        question: str,
        attachments: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Execute fact-retrieval task with perfect accuracy.

        Args:
            question: Question to answer
            attachments: Optional attachment files
            context: Optional additional context

        Returns:
            Exact answer string
        """
        logger.info(f"FactRetrievalExecutor: {question[:100]}...")

        # Step 1: Analyze question
        analysis = await self._analyze_question(question, attachments)
        logger.info(f"Question analysis: {analysis}")

        # Step 2: Decompose into steps if multi-hop
        if analysis.is_multi_hop:
            steps = await self._decompose_question(question, analysis)
            logger.info(f"Decomposed into {len(steps)} steps")
        else:
            steps = [ExecutionStep(
                text=question,
                depends_on=[],
                tools=analysis.tools_needed
            )]

        # Step 3: Execute each step
        execution_context = {}
        for i, step in enumerate(steps):
            logger.info(f"Executing step {i+1}/{len(steps)}: {step.text}")

            # Resolve dependencies
            resolved_text = self._resolve_dependencies(step.text, steps, execution_context)

            # Execute step
            result = await self._execute_step(
                question=resolved_text,
                tools=step.tools,
                attachments=attachments if i == 0 else None,  # Attachments only for first step
                context=execution_context
            )

            step.result = result
            execution_context[f"step_{i+1}"] = result
            logger.info(f"Step {i+1} result: {result}")

        # Step 4: Extract exact answer
        final_answer = await self._extract_answer(
            question=question,
            steps=steps,
            expected_format=analysis.answer_format,
            context=execution_context
        )

        logger.info(f"Final answer: {final_answer}")
        return final_answer

    async def _analyze_question(
        self,
        question: str,
        attachments: Optional[List[str]]
    ) -> QuestionAnalysis:
        """Analyze question structure and requirements."""

        prompt = f"""Analyze this question to determine how to answer it.

Question: {question}
{f"Attachments: {', '.join(attachments)}" if attachments else ""}

Provide analysis in this format:

MULTI_HOP: [yes/no] - Does this require multiple reasoning steps?
FORMAT: [text/number/date/person/location/yes_no/list] - Expected answer format
DOMAIN: [general/science/math/geography/history/etc.] - Question domain
TOOLS: [comma-separated tool names] - Tools needed (web-search, calculator, document-reader, etc.)
COMPLEXITY: [simple/medium/complex] - Question complexity

Analysis:"""

        response = self.lm(prompt)
        return self._parse_question_analysis(response, question, attachments)

    def _parse_question_analysis(
        self,
        response: str,
        question: str,
        attachments: Optional[List[str]]
    ) -> QuestionAnalysis:
        """Parse question analysis from LLM response."""

        lines = response.strip().split('\n')
        data = {}

        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                data[key.strip().upper()] = value.strip()

        # Parse fields
        is_multi_hop = data.get('MULTI_HOP', 'no').lower() == 'yes'

        format_str = data.get('FORMAT', 'text').lower()
        format_map = {
            'text': AnswerFormat.TEXT,
            'number': AnswerFormat.NUMBER,
            'date': AnswerFormat.DATE,
            'person': AnswerFormat.PERSON,
            'location': AnswerFormat.LOCATION,
            'yes_no': AnswerFormat.YES_NO,
            'list': AnswerFormat.LIST,
        }
        answer_format = format_map.get(format_str, AnswerFormat.TEXT)

        domain = data.get('DOMAIN', 'general')

        tools_str = data.get('TOOLS', '')
        if tools_str and tools_str.lower() != 'none':
            tools_needed = [t.strip() for t in tools_str.split(',')]
        else:
            tools_needed = []

        # Auto-detect additional tools
        auto_tools = self._auto_detect_tools(question, attachments)
        tools_needed.extend([t for t in auto_tools if t not in tools_needed])

        complexity = data.get('COMPLEXITY', 'medium')

        return QuestionAnalysis(
            is_multi_hop=is_multi_hop,
            answer_format=answer_format,
            domain=domain,
            tools_needed=tools_needed,
            complexity=complexity
        )

    def _auto_detect_tools(self, question: str, attachments: Optional[List[str]]) -> List[str]:
        """Auto-detect required tools from question patterns."""

        tools = []
        q_lower = question.lower()

        # Math/calculation
        if any(op in question for op in ['+', '-', '*', '/', '=']) or \
           any(word in q_lower for word in ['calculate', 'compute', 'sum', 'multiply', 'divide']):
            tools.append('calculator')

        # Web search
        if any(word in q_lower for word in ['what is', 'who is', 'when', 'where', 'find', 'search']):
            tools.append('web-search')

        # Attachments
        if attachments:
            for filename in attachments:
                ext = filename.split('.')[-1].lower()
                if ext in ['mp3', 'wav', 'm4a']:
                    tools.append('whisper')
                elif ext in ['pdf', 'docx', 'txt']:
                    tools.append('document-reader')
                elif ext in ['jpg', 'png', 'jpeg']:
                    tools.append('vision')

        return tools

    async def _decompose_question(
        self,
        question: str,
        analysis: QuestionAnalysis
    ) -> List[ExecutionStep]:
        """Decompose multi-hop question into atomic steps."""

        prompt = f"""Decompose this multi-step question into atomic sub-questions.
Each step should be answerable independently or reference previous steps.

Question: {question}
Domain: {analysis.domain}

Provide steps in this EXACT format:

Step 1: [atomic question]
Tools: [tool1, tool2]

Step 2: [question that may reference {{Step 1}}]
Tools: [tool1]

Steps:"""

        response = self.lm(prompt)
        return self._parse_decomposition(response)

    def _parse_decomposition(self, response: str) -> List[ExecutionStep]:
        """Parse step decomposition from LLM response."""

        steps = []
        lines = response.strip().split('\n')
        current_step = None
        current_tools = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if line.startswith('Step '):
                # Save previous step
                if current_step:
                    steps.append(ExecutionStep(
                        text=current_step,
                        depends_on=self._extract_dependencies(current_step),
                        tools=current_tools
                    ))

                # Extract new step text
                match = re.match(r'Step \d+:\s*(.+)', line)
                if match:
                    current_step = match.group(1)
                    current_tools = []

            elif line.startswith('Tools:'):
                # Extract tools
                tools_str = line.replace('Tools:', '').strip()
                if tools_str and tools_str.lower() != 'none':
                    current_tools = [t.strip() for t in tools_str.split(',')]

        # Add last step
        if current_step:
            steps.append(ExecutionStep(
                text=current_step,
                depends_on=self._extract_dependencies(current_step),
                tools=current_tools
            ))

        return steps

    def _extract_dependencies(self, step_text: str) -> List[int]:
        """Extract which previous steps this step depends on."""

        deps = []
        # Look for {Step N} references
        for match in re.finditer(r'\{Step (\d+)\}', step_text):
            step_num = int(match.group(1))
            deps.append(step_num - 1)  # Convert to 0-indexed

        return deps

    def _resolve_dependencies(
        self,
        step_text: str,
        steps: List[ExecutionStep],
        context: Dict[str, Any]
    ) -> str:
        """Replace {Step N} references with actual results."""

        resolved = step_text

        # Find all {Step N} patterns
        for match in re.finditer(r'\{Step (\d+)\}', step_text):
            step_num = int(match.group(1))
            step_key = f"step_{step_num}"

            if step_key in context:
                # Replace with actual result
                result = context[step_key]
                resolved = resolved.replace(match.group(0), str(result))

        return resolved

    async def _execute_step(
        self,
        question: str,
        tools: List[str],
        attachments: Optional[List[str]],
        context: Dict[str, Any]
    ) -> str:
        """Execute single step with specified tools."""

        # Build context from previous steps
        context_str = ""
        if context:
            context_str = "\n\nPrevious steps:\n"
            for key, value in context.items():
                context_str += f"- {key}: {value}\n"

        # Execute with tools
        prompt = f"""Answer this question using the available tools.

Question: {question}
{f"Attachments: {', '.join(attachments)}" if attachments else ""}
{context_str if context_str else ""}

Available tools: {', '.join(tools)}

Provide a direct, concise answer. If you need to use a tool, use it.

Answer:"""

        # FUNDAMENTAL FIX: Actually execute tools, don't just prompt
        #
        # Use LLM with tool-calling to properly execute tools
        # This is the CORRECT way - let the LLM decide which tools to use

        try:
            # Get tool definitions from registry
            tool_defs = []
            for tool_name in tools:
                try:
                    skill = self.registry.get_skill(tool_name)
                    if skill and hasattr(skill, 'get_tool_definition'):
                        tool_defs.append(skill.get_tool_definition())
                except Exception as e:
                    logger.warning(f"Could not load tool {tool_name}: {e}")

            # Execute with tool-calling enabled
            if tool_defs:
                logger.info(f"Executing with {len(tool_defs)} tools available")

                # Use tool-calling LLM (e.g., Claude with tools)
                from Jotty.core.foundation.unified_lm_provider import get_lm

                tool_lm = get_lm(model='claude-3-5-sonnet-20241022', temperature=0.0)

                # Let LLM use tools to answer the question
                response = tool_lm(
                    prompt,
                    tools=tool_defs,
                    tool_choice='auto'
                )

                # Extract final answer from tool-calling response
                if isinstance(response, dict) and 'content' in response:
                    result = response['content']
                else:
                    result = str(response)

                return result.strip()
            else:
                # No tools available, use regular LLM
                logger.warning("No tools available, falling back to LLM-only")
                response = self.lm(prompt)
                return response.strip()

        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            # Fallback to regular LLM
            response = self.lm(prompt)
            return response.strip()

    async def _extract_answer(
        self,
        question: str,
        steps: List[ExecutionStep],
        expected_format: AnswerFormat,
        context: Dict[str, Any]
    ) -> str:
        """Extract exact answer in expected format."""

        # Collect all step results
        results_text = "\n\n".join(
            f"Step {i+1}: {step.text}\nResult: {step.result}"
            for i, step in enumerate(steps)
        )

        prompt = f"""Extract the exact answer to this question from the execution results.

Original Question: {question}
Expected Format: {expected_format.value}

Execution Results:
{results_text}

Provide ONLY the final answer in the expected format. No explanation, no reasoning.

{self._format_instruction(expected_format)}

Final Answer:"""

        response = self.lm(prompt)
        answer = response.strip()

        # Validate and fix format
        answer = self._validate_and_fix_format(answer, expected_format)

        return answer

    def _format_instruction(self, format: AnswerFormat) -> str:
        """Get format-specific instructions."""

        instructions = {
            AnswerFormat.NUMBER: "Provide just the number, no units or explanation.",
            AnswerFormat.DATE: "Provide the date/year only.",
            AnswerFormat.PERSON: "Provide just the person's name.",
            AnswerFormat.LOCATION: "Provide just the location name.",
            AnswerFormat.YES_NO: "Provide just 'Yes' or 'No'.",
            AnswerFormat.LIST: "Provide items separated by commas.",
            AnswerFormat.TEXT: "Provide a concise text answer.",
        }

        return instructions.get(format, "Provide a concise answer.")

    def _validate_and_fix_format(self, answer: str, format: AnswerFormat) -> str:
        """Validate answer format and fix if needed."""

        if format == AnswerFormat.NUMBER:
            # Extract first number
            match = re.search(r'-?\d+\.?\d*', answer)
            if match:
                return match.group()

        elif format == AnswerFormat.YES_NO:
            # Normalize to Yes/No
            answer_lower = answer.lower()
            if 'yes' in answer_lower or 'true' in answer_lower:
                return 'Yes'
            elif 'no' in answer_lower or 'false' in answer_lower:
                return 'No'

        elif format == AnswerFormat.LOCATION:
            # Remove prefixes like "in", "at"
            answer = re.sub(r'^(in|at|near)\s+', '', answer, flags=re.IGNORECASE)

        return answer.strip()


# Singleton instance
_executor_instance: Optional[FactRetrievalExecutor] = None


def get_fact_retrieval_executor() -> FactRetrievalExecutor:
    """Get or create singleton executor instance."""
    global _executor_instance
    if _executor_instance is None:
        _executor_instance = FactRetrievalExecutor()
    return _executor_instance
