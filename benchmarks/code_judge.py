"""
LLM Code Judge - Automatic Code Quality Assessment for Swarm Outputs
=====================================================================

This module provides automated code quality evaluation using both:
1. Static heuristics (fast, deterministic checks)
2. LLM-based analysis (nuanced, contextual evaluation)

The judge evaluates code against principles:
- Complexity should match the problem
- YAGNI (You Aren't Gonna Need It)
- No premature abstraction
- No cargo-cult engineering

Usage:
    from benchmarks.code_judge import CodeJudge

    judge = CodeJudge()
    result = await judge.evaluate(
        task="Make a tic-tac-toe game",
        code_files={"game.py": "..."},
    )
    print(result.score, result.verdict)
"""

import asyncio
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)


@dataclass
class CodeMetrics:
    """Static code analysis metrics."""
    total_files: int = 0
    total_lines: int = 0
    total_classes: int = 0
    total_functions: int = 0
    protocol_count: int = 0
    abstract_count: int = 0
    decorator_count: int = 0
    import_count: int = 0
    docstring_ratio: float = 0.0
    comment_ratio: float = 0.0
    avg_function_length: float = 0.0
    max_function_length: int = 0
    cyclomatic_complexity_estimate: int = 0
    redundant_implementations: List[str] = field(default_factory=list)


@dataclass
class QualityIssue:
    """A specific code quality issue found."""
    category: str  # "over-engineering", "yagni", "duplication", "complexity"
    severity: str  # "critical", "major", "minor"
    description: str
    file: Optional[str] = None
    line: Optional[int] = None
    suggestion: str = ""


@dataclass
class JudgeResult:
    """Result of code quality evaluation."""
    task: str
    score: int  # 1-10
    verdict: str  # "excellent", "good", "acceptable", "poor", "bad"
    metrics: CodeMetrics
    issues: List[QualityIssue]
    complexity_match: str  # "under", "appropriate", "over"
    summary: str
    recommendations: List[str]
    llm_analysis: Optional[str] = None
    evaluation_time: float = 0.0


class CodeJudge:
    """
    Evaluates code quality with focus on appropriate complexity.

    Uses a two-phase approach:
    1. Static heuristics for fast, deterministic checks
    2. LLM analysis for nuanced, contextual evaluation
    """

    # Task complexity estimates (lines of code for "appropriate" solution)
    TASK_COMPLEXITY_HINTS = {
        "tic-tac-toe": (50, 150),
        "todo": (30, 100),
        "calculator": (20, 80),
        "snake": (100, 250),
        "hangman": (50, 120),
        "rest api": (50, 200),
        "cli tool": (40, 150),
    }

    # Over-engineering patterns to detect
    OVER_ENGINEERING_PATTERNS = {
        r'Protocol\[': ("Protocol usage", "Protocols often overkill for single implementations"),
        r'@abstractmethod': ("Abstract methods", "Abstract classes often premature for simple apps"),
        r'class.*Factory': ("Factory pattern", "Factories often unnecessary for simple object creation"),
        r'CircuitBreaker|circuit_breaker': ("Circuit breaker", "Circuit breakers for simple loops indicate over-engineering"),
        r'bit_mask|bitmask|BitMask': ("Bit masking", "Bit operations rarely needed for simple games"),
        r'O\(1\)|O\(n\)': ("Big-O claims", "Performance claims for trivial operations"),
        r'Production.?Ready|Enterprise': ("Enterprise language", "Enterprise claims for simple apps"),
        r'Optimized|High.?Performance': ("Performance claims", "Optimization claims should match complexity"),
        r'class.*Protocol\(Protocol\)': ("Protocol definition", "Custom protocols for single use"),
        r'signal\.signal|SIGALRM': ("Signal handlers", "Signal handling for simple input is excessive"),
        r'accessibility|screen.?reader': ("Accessibility", "Check if genuinely needed or over-scoped"),
        r'InputProcessor|OutputRenderer': ("Processor patterns", "Processor abstractions for simple I/O"),
    }

    # Duplication patterns
    DUPLICATION_INDICATORS = [
        (r'class.*Game.*:', "Game class", 2),  # Flagif more than 2
        (r'class.*Board.*:', "Board class", 2),
        (r'class.*Engine.*:', "Engine class", 1),
        (r'class.*Controller.*:', "Controller class", 1),
        (r'def (check_win|check_winner)', "Win check function", 1),
    ]

    def __init__(self, llm_provider: Optional[Any] = None):
        """
        Initialize code judge.

        Args:
            llm_provider: Optional LLM for nuanced analysis. If None, uses heuristics only.
        """
        self.llm_provider = llm_provider

    def _analyze_metrics(self, code_files: Dict[str, str]) -> CodeMetrics:
        """Extract static metrics from code."""
        metrics = CodeMetrics()
        metrics.total_files = len(code_files)

        all_code = "\n".join(code_files.values())
        lines = all_code.split('\n')
        metrics.total_lines = len(lines)

        # Count classes and functions
        metrics.total_classes = len(re.findall(r'^class\s+\w+', all_code, re.MULTILINE))
        metrics.total_functions = len(re.findall(r'^def\s+\w+', all_code, re.MULTILINE))

        # Count abstractions
        metrics.protocol_count = len(re.findall(r'Protocol\[|Protocol\)', all_code))
        metrics.abstract_count = len(re.findall(r'@abstractmethod|ABC\)', all_code))
        metrics.decorator_count = len(re.findall(r'^@\w+', all_code, re.MULTILINE))
        metrics.import_count = len(re.findall(r'^(?:from|import)\s+', all_code, re.MULTILINE))

        # Docstring and comment ratios
        docstrings = len(re.findall(r'"""[\s\S]*?"""', all_code))
        comments = len(re.findall(r'#.*$', all_code, re.MULTILINE))
        non_empty_lines = len([l for l in lines if l.strip()])
        if non_empty_lines > 0:
            metrics.docstring_ratio = docstrings / max(1, metrics.total_functions + metrics.total_classes)
            metrics.comment_ratio = comments / non_empty_lines

        # Function lengths
        func_lengths = []
        current_func_lines = 0
        in_function = False
        indent_level = 0

        for line in lines:
            if re.match(r'^def\s+\w+', line):
                if in_function and current_func_lines > 0:
                    func_lengths.append(current_func_lines)
                in_function = True
                current_func_lines = 1
                indent_level = len(line) - len(line.lstrip())
            elif in_function:
                if line.strip() and (len(line) - len(line.lstrip())) > indent_level:
                    current_func_lines += 1
                elif line.strip() and (len(line) - len(line.lstrip())) <= indent_level:
                    func_lengths.append(current_func_lines)
                    in_function = False

        if in_function and current_func_lines > 0:
            func_lengths.append(current_func_lines)

        if func_lengths:
            metrics.avg_function_length = sum(func_lengths) / len(func_lengths)
            metrics.max_function_length = max(func_lengths)

        # Cyclomatic complexity estimate (branching)
        metrics.cyclomatic_complexity_estimate = (
            len(re.findall(r'\bif\b', all_code)) +
            len(re.findall(r'\belif\b', all_code)) +
            len(re.findall(r'\bfor\b', all_code)) +
            len(re.findall(r'\bwhile\b', all_code)) +
            len(re.findall(r'\band\b|\bor\b', all_code)) +
            len(re.findall(r'\btry\b', all_code))
        )

        # Check for redundant implementations
        for pattern, name, max_count in self.DUPLICATION_INDICATORS:
            count = len(re.findall(pattern, all_code, re.MULTILINE))
            if count > max_count:
                metrics.redundant_implementations.append(
                    f"{name}: found {count} (expected max {max_count})"
                )

        return metrics

    def _estimate_task_complexity(self, task: str) -> Tuple[int, int]:
        """Estimate appropriate LOC range for a task."""
        task_lower = task.lower()

        for hint_key, (min_loc, max_loc) in self.TASK_COMPLEXITY_HINTS.items():
            if hint_key in task_lower:
                return (min_loc, max_loc)

        # Default: moderate complexity
        return (50, 200)

    def _check_over_engineering(
        self,
        code_files: Dict[str, str],
        task: str
    ) -> List[QualityIssue]:
        """Detect over-engineering patterns."""
        issues = []
        all_code = "\n".join(code_files.values())
        task_lower = task.lower()

        # Check each pattern
        for pattern, (name, reason) in self.OVER_ENGINEERING_PATTERNS.items():
            matches = re.findall(pattern, all_code, re.IGNORECASE)
            if matches:
                # Context-aware severity
                severity = "major"

                # Simple tasks get harsher ratings for complex patterns
                simple_keywords = ["tic", "tac", "toe", "simple", "basic", "hello", "todo"]
                if any(kw in task_lower for kw in simple_keywords):
                    if name in ["Circuit breaker", "Signal handlers", "Bit masking", "Enterprise language"]:
                        severity = "critical"

                issues.append(QualityIssue(
                    category="over-engineering",
                    severity=severity,
                    description=f"{name} detected: {reason}",
                    suggestion=f"Consider if {name.lower()} is truly needed for this task"
                ))

        return issues

    def _check_yagni(
        self,
        code_files: Dict[str, str],
        metrics: CodeMetrics
    ) -> List[QualityIssue]:
        """Check for YAGNI violations."""
        issues = []
        all_code = "\n".join(code_files.values())

        # Too many protocols/abstractions
        if metrics.protocol_count > 0 and metrics.total_classes < metrics.protocol_count * 3:
            issues.append(QualityIssue(
                category="yagni",
                severity="major",
                description=f"Protocol definitions ({metrics.protocol_count}) without sufficient implementations",
                suggestion="Protocols are useful when you have multiple implementations. Consider removing."
            ))

        # Abstract base classes for single implementations
        if metrics.abstract_count > 0:
            abc_classes = re.findall(r'class\s+(\w+).*ABC\)', all_code)
            for abc_name in abc_classes:
                impl_count = len(re.findall(rf'class\s+\w+\([^)]*{abc_name}', all_code))
                if impl_count <= 1:
                    issues.append(QualityIssue(
                        category="yagni",
                        severity="major",
                        description=f"Abstract class '{abc_name}' has {impl_count} implementation(s)",
                        suggestion="Remove abstract base class if there's only one implementation"
                    ))

        # Excessive imports for simple tasks
        if metrics.import_count > 15 and metrics.total_lines < 500:
            issues.append(QualityIssue(
                category="yagni",
                severity="minor",
                description=f"High import count ({metrics.import_count}) for code size ({metrics.total_lines} lines)",
                suggestion="Review if all imports are necessary"
            ))

        return issues

    def _check_duplication(
        self,
        code_files: Dict[str, str],
        metrics: CodeMetrics
    ) -> List[QualityIssue]:
        """Check for code duplication and redundant implementations."""
        issues = []

        for dup in metrics.redundant_implementations:
            issues.append(QualityIssue(
                category="duplication",
                severity="critical",
                description=f"Redundant implementations: {dup}",
                suggestion="Consolidate into a single, well-designed implementation"
            ))

        # Check for similar function names suggesting duplication
        all_code = "\n".join(code_files.values())
        func_names = re.findall(r'def\s+(\w+)', all_code)

        # Group by base name (ignoring numbers and common suffixes)
        base_names: Dict[str, List[str]] = {}
        for name in func_names:
            base = re.sub(r'_?v?\d+$|_optimized|_fast|_new', '', name)
            if base not in base_names:
                base_names[base] = []
            base_names[base].append(name)

        for base, variants in base_names.items():
            if len(variants) > 2:
                issues.append(QualityIssue(
                    category="duplication",
                    severity="major",
                    description=f"Multiple variants of function '{base}': {variants}",
                    suggestion="Consolidate function variants"
                ))

        return issues

    def _check_complexity_match(
        self,
        task: str,
        metrics: CodeMetrics
    ) -> Tuple[str, List[QualityIssue]]:
        """Check if code complexity matches task complexity."""
        issues = []
        min_loc, max_loc = self._estimate_task_complexity(task)

        if metrics.total_lines < min_loc * 0.5:
            return "under", [QualityIssue(
                category="complexity",
                severity="minor",
                description=f"Code seems too minimal ({metrics.total_lines} lines) for the task",
                suggestion="Ensure all requirements are met"
            )]
        elif metrics.total_lines > max_loc * 3:
            ratio = metrics.total_lines / max_loc
            severity = "critical" if ratio > 5 else "major"
            issues.append(QualityIssue(
                category="complexity",
                severity=severity,
                description=f"Code is {ratio:.1f}x larger than expected ({metrics.total_lines} lines, expected ~{max_loc})",
                suggestion="Simplify implementation. Good code matches complexity to the problem."
            ))
            return "over", issues
        elif metrics.total_lines > max_loc * 1.5:
            issues.append(QualityIssue(
                category="complexity",
                severity="minor",
                description=f"Code is moderately larger than typical ({metrics.total_lines} vs ~{max_loc} lines)",
                suggestion="Review for unnecessary abstractions"
            ))
            return "over", issues

        return "appropriate", issues

    def _calculate_score(
        self,
        issues: List[QualityIssue],
        complexity_match: str
    ) -> int:
        """Calculate final score from 1-10."""
        score = 10

        # Deduct for issues
        severity_deductions = {
            "critical": 2.5,
            "major": 1.5,
            "minor": 0.5
        }

        for issue in issues:
            score -= severity_deductions.get(issue.severity, 0.5)

        # Complexity mismatch penalty
        if complexity_match == "over":
            score -= 1
        elif complexity_match == "under":
            score -= 0.5

        return max(1, min(10, round(score)))

    def _get_verdict(self, score: int) -> str:
        """Get verdict string from score."""
        if score >= 9:
            return "excellent"
        elif score >= 7:
            return "good"
        elif score >= 5:
            return "acceptable"
        elif score >= 3:
            return "poor"
        else:
            return "bad"

    def _generate_summary(
        self,
        task: str,
        metrics: CodeMetrics,
        issues: List[QualityIssue],
        complexity_match: str,
        score: int
    ) -> str:
        """Generate human-readable summary."""
        parts = []

        parts.append(f"Code Quality Assessment: {task}")
        parts.append(f"Score: {score}/10")
        parts.append("")

        # The Problem
        if complexity_match == "over":
            min_loc, max_loc = self._estimate_task_complexity(task)
            parts.append("## The Problem")
            parts.append(f"This task typically requires {min_loc}-{max_loc} lines of code.")
            parts.append(f"Delivered: {metrics.total_files} files, ~{metrics.total_lines} lines")
            parts.append("")

        # Critical issues
        critical = [i for i in issues if i.severity == "critical"]
        if critical:
            parts.append("## Critical Issues")
            for issue in critical:
                parts.append(f"- **{issue.description}**")
                if issue.suggestion:
                    parts.append(f"  - {issue.suggestion}")
            parts.append("")

        # Major issues
        major = [i for i in issues if i.severity == "major"]
        if major:
            parts.append("## Major Issues")
            for issue in major:
                parts.append(f"- {issue.description}")
            parts.append("")

        # What's good (if any)
        positives = []
        if metrics.docstring_ratio > 0.5:
            positives.append("Good documentation coverage")
        if metrics.avg_function_length < 20:
            positives.append("Functions are reasonably sized")
        if not issues:
            positives.append("No significant issues found")

        if positives:
            parts.append("## What's Good")
            for p in positives:
                parts.append(f"- {p}")
            parts.append("")

        return "\n".join(parts)

    def _generate_recommendations(
        self,
        issues: List[QualityIssue],
        complexity_match: str,
        metrics: CodeMetrics
    ) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []

        if complexity_match == "over":
            recommendations.append(
                "SIMPLIFY: Good code matches complexity to the problem. "
                "Start fresh with the minimal solution that works."
            )

        if any(i.category == "duplication" for i in issues):
            recommendations.append(
                "CONSOLIDATE: Merge redundant implementations into one clean solution."
            )

        if any(i.category == "over-engineering" for i in issues):
            recommendations.append(
                "REMOVE ABSTRACTIONS: Delete protocols, factories, and patterns "
                "that don't have multiple implementations."
            )

        if metrics.total_files > 5 and metrics.total_lines < 500:
            recommendations.append(
                f"CONSOLIDATE FILES: {metrics.total_files} files for {metrics.total_lines} lines "
                "suggests over-modularization."
            )

        if not recommendations:
            recommendations.append("Code quality is acceptable. Minor improvements may be possible.")

        return recommendations

    async def _llm_analyze(
        self,
        task: str,
        code_files: Dict[str, str],
        metrics: CodeMetrics,
        issues: List[QualityIssue]
    ) -> Optional[str]:
        """Use LLM for nuanced analysis."""
        if not self.llm_provider:
            return None

        # Build prompt parts separately to avoid f-string backslash issues
        newline = "\n"
        issues_list = newline.join(f"- {i.description}" for i in issues)

        code_snippets = []
        for name, code in list(code_files.items())[:5]:
            snippet = code[:2000] + "..." if len(code) > 2000 else code
            code_snippets.append(f"=== {name} ==={newline}{snippet}")
        code_section = newline.join(code_snippets)

        prompt = f"""Analyze this code for quality issues. The task was: "{task}"

Code Metrics:
- Files: {metrics.total_files}
- Lines: {metrics.total_lines}
- Classes: {metrics.total_classes}
- Functions: {metrics.total_functions}
- Protocols: {metrics.protocol_count}
- Abstract classes: {metrics.abstract_count}

Already detected issues:
{issues_list}

Code:
{code_section}

Provide a brief (3-4 sentences) assessment focusing on:
1. Is the complexity appropriate for the task?
2. Any cargo-cult engineering patterns?
3. One key recommendation for improvement.
"""
        try:
            # Try to use the LLM provider
            if hasattr(self.llm_provider, 'generate'):
                response = await self.llm_provider.generate(prompt)
                return response
            elif hasattr(self.llm_provider, '__call__'):
                response = await self.llm_provider(prompt)
                return response
        except Exception as e:
            logger.warning(f"LLM analysis failed: {e}")

        return None

    async def evaluate(
        self,
        task: str,
        code_files: Dict[str, str],
        use_llm: bool = True
    ) -> JudgeResult:
        """
        Evaluate code quality.

        Args:
            task: Original task description
            code_files: Dict of filename -> content
            use_llm: Whether to use LLM for additional analysis

        Returns:
            JudgeResult with score, issues, and recommendations
        """
        start_time = datetime.now()

        # Phase 1: Static analysis
        metrics = self._analyze_metrics(code_files)

        # Collect issues
        issues: List[QualityIssue] = []

        # Check various quality aspects
        issues.extend(self._check_over_engineering(code_files, task))
        issues.extend(self._check_yagni(code_files, metrics))
        issues.extend(self._check_duplication(code_files, metrics))

        complexity_match, complexity_issues = self._check_complexity_match(task, metrics)
        issues.extend(complexity_issues)

        # Calculate score
        score = self._calculate_score(issues, complexity_match)
        verdict = self._get_verdict(score)

        # Generate summary and recommendations
        summary = self._generate_summary(task, metrics, issues, complexity_match, score)
        recommendations = self._generate_recommendations(issues, complexity_match, metrics)

        # Phase 2: LLM analysis (optional)
        llm_analysis = None
        if use_llm and self.llm_provider:
            llm_analysis = await self._llm_analyze(task, code_files, metrics, issues)

        evaluation_time = (datetime.now() - start_time).total_seconds()

        return JudgeResult(
            task=task,
            score=score,
            verdict=verdict,
            metrics=metrics,
            issues=issues,
            complexity_match=complexity_match,
            summary=summary,
            recommendations=recommendations,
            llm_analysis=llm_analysis,
            evaluation_time=evaluation_time
        )

    def evaluate_sync(
        self,
        task: str,
        code_files: Dict[str, str],
        use_llm: bool = False
    ) -> JudgeResult:
        """Synchronous evaluation (no LLM)."""
        return asyncio.run(self.evaluate(task, code_files, use_llm=use_llm))


class SwarmCodeReviewer:
    """
    Integration point for swarm output review.

    Use this to automatically review code before presenting to users.
    """

    def __init__(
        self,
        min_acceptable_score: int = 5,
        auto_flag_threshold: int = 3,
        llm_provider: Optional[Any] = None
    ):
        """
        Initialize swarm code reviewer.

        Args:
            min_acceptable_score: Minimum score to pass (1-10)
            auto_flag_threshold: Score below which to auto-flag for human review
            llm_provider: Optional LLM for enhanced analysis
        """
        self.judge = CodeJudge(llm_provider)
        self.min_acceptable_score = min_acceptable_score
        self.auto_flag_threshold = auto_flag_threshold

    async def review(
        self,
        task: str,
        generated_files: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Review swarm-generated code.

        Returns dict with:
        - passed: bool
        - needs_human_review: bool
        - result: JudgeResult
        - action: suggested action ("accept", "revise", "reject")
        """
        result = await self.judge.evaluate(task, generated_files)

        passed = result.score >= self.min_acceptable_score
        needs_human_review = result.score < self.auto_flag_threshold

        if result.score >= 8:
            action = "accept"
        elif result.score >= 5:
            action = "revise"
        else:
            action = "reject"

        return {
            "passed": passed,
            "needs_human_review": needs_human_review,
            "result": result,
            "action": action,
            "reason": result.recommendations[0] if result.recommendations else None
        }


# =============================================================================
# EXAMPLE: PROPER TIC-TAC-TOE
# =============================================================================
# This shows what the code SHOULD look like

PROPER_TIC_TAC_TOE = '''"""
Tic-Tac-Toe - Simple keyboard-based game.

Controls: 1-9 for positions, Q to quit.
"""
import os


class TicTacToe:
    """Simple tic-tac-toe game with keyboard input."""

    WIN_PATTERNS = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],  # rows
        [0, 3, 6], [1, 4, 7], [2, 5, 8],  # cols
        [0, 4, 8], [2, 4, 6]              # diagonals
    ]

    def __init__(self):
        self.board = [" "] * 9
        self.current = "X"

    def display(self):
        """Show the board with position hints."""
        os.system("clear" if os.name == "posix" else "cls")
        print("\\nTic-Tac-Toe - Press 1-9 to play, Q to quit\\n")
        for i in range(3):
            row = self.board[i*3:(i+1)*3]
            hints = [str(i*3 + j + 1) if c == " " else c for j, c in enumerate(row)]
            print(f" {hints[0]} | {hints[1]} | {hints[2]} ")
            if i < 2:
                print("-----------")
        print(f"\\nPlayer {self.current}'s turn")

    def check_winner(self) -> str | None:
        """Return winner (X/O) or None."""
        for pattern in self.WIN_PATTERNS:
            vals = [self.board[i] for i in pattern]
            if vals[0] != " " and vals[0] == vals[1] == vals[2]:
                return vals[0]
        return None

    def is_draw(self) -> bool:
        """Check if board is full with no winner."""
        return " " not in self.board and not self.check_winner()

    def make_move(self, pos: int) -> bool:
        """Make move at position (1-9). Returns True if valid."""
        if 1 <= pos <= 9 and self.board[pos - 1] == " ":
            self.board[pos - 1] = self.current
            self.current = "O" if self.current == "X" else "X"
            return True
        return False

    def play(self):
        """Main game loop."""
        while True:
            self.display()

            if winner := self.check_winner():
                print(f"\\n{winner} wins!")
                break
            if self.is_draw():
                print("\\nIt's a draw!")
                break

            try:
                key = input("\\nEnter position (1-9) or Q to quit: ").strip().upper()
                if key == "Q":
                    print("Thanks for playing!")
                    break
                if key.isdigit() and not self.make_move(int(key)):
                    input("Invalid move! Press Enter...")
            except (EOFError, KeyboardInterrupt):
                break


if __name__ == "__main__":
    TicTacToe().play()
'''


# =============================================================================
# DEMO / CLI
# =============================================================================

async def demo():
    """Demonstrate the code judge."""
    judge = CodeJudge()

    # Test with the proper implementation
    print("=" * 60)
    print("Testing PROPER tic-tac-toe implementation:")
    print("=" * 60)

    result = await judge.evaluate(
        task="Make a tic-tac-toe game with keyboard input",
        code_files={"tic_tac_toe.py": PROPER_TIC_TAC_TOE}
    )

    print(result.summary)
    print(f"\nScore: {result.score}/10 ({result.verdict})")
    print(f"Complexity match: {result.complexity_match}")
    print("\nRecommendations:")
    for rec in result.recommendations:
        print(f"  - {rec}")

    # Test with over-engineered example
    print("\n" + "=" * 60)
    print("Testing OVER-ENGINEERED implementation (simulated):")
    print("=" * 60)

    over_engineered = {
        "game_board.py": '''
class GameBoardProtocol(Protocol):
    """Protocol for game board implementations."""
    def get_state(self) -> List[str]: ...
    def set_cell(self, pos: int, value: str) -> bool: ...

class OptimizedGameEngine:
    """Production-Ready High-Performance game engine with O(1) win detection."""
    WIN_MASKS = [0b111000000, 0b000111000]  # Bit masking for efficiency

class GameController:
    """MVC controller for game state management."""
    pass
''',
        "game_engine.py": '''
class GameEngine:
    """Another game engine implementation."""
    def check_winner(self): pass

class CircuitBreaker:
    """Circuit breaker for input loop protection."""
    pass
''',
        "tic_tac_toe_app.py": '''
import signal

class InputProcessorProtocol(Protocol):
    """Input processing abstraction."""
    pass

class TicTacToeGame:
    """Yet another game implementation."""
    def __init__(self):
        signal.signal(signal.SIGALRM, self._timeout_handler)

    def _timeout_handler(self, signum, frame):
        pass
'''
    }

    result2 = await judge.evaluate(
        task="Make a tic-tac-toe game with keyboard input",
        code_files=over_engineered
    )

    print(result2.summary)
    print(f"\nScore: {result2.score}/10 ({result2.verdict})")
    print("\nIssues found:")
    for issue in result2.issues:
        print(f"  [{issue.severity.upper()}] {issue.description}")


if __name__ == "__main__":
    asyncio.run(demo())
