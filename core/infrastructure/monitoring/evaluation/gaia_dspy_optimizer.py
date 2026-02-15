"""
GAIA DSPy Optimizer — BootstrapFewShot compilation from successful examples.

Compiles a DSPy module from successful GAIA (question, answer) pairs so that
future runs benefit from accumulated few-shot examples.

Usage:
    from Jotty.core.infrastructure.monitoring.evaluation.gaia_dspy_optimizer import compile_gaia_module

    successful = [
        {'question': 'What is 2+2?', 'expected': '4', 'success': True},
        ...
    ]
    compiled = compile_gaia_module(successful, '~/.jotty/gaia_compiled.json')
"""

import logging
from pathlib import Path
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

try:
    import dspy

    class GAIATaskSignature(dspy.Signature):
        """Solve a GAIA benchmark question using available tools."""

        question: str = dspy.InputField()
        tools_available: str = dspy.InputField(
            desc="Available tools: web_search, voice_to_text, read_file"
        )
        answer: str = dspy.OutputField(desc="Concise final answer only")

    class GAIAOptimizedSolver(dspy.Module):
        def __init__(self) -> None:
            super().__init__()
            self.solve = dspy.ChainOfThought(GAIATaskSignature)

        def forward(
            self, question: Any, tools_available: Any = "web_search, voice_to_text, read_file"
        ) -> Any:
            return self.solve(question=question, tools_available=tools_available)

    _DSPY_AVAILABLE = True
except ImportError:
    _DSPY_AVAILABLE = False


def compile_gaia_module(
    successful_examples: List[dict],
    save_path: str,
) -> Optional[object]:
    """Compile a DSPy module from successful (question, answer) pairs.

    Args:
        successful_examples: List of dicts with 'question', 'expected', 'success' keys.
        save_path: Path to save the compiled module JSON.

    Returns:
        The compiled DSPy module, or None if not enough examples or DSPy unavailable.
    """
    if not _DSPY_AVAILABLE:
        logger.warning("DSPy not available, skipping GAIA module compilation")
        return None

    examples = [
        dspy.Example(
            question=e["question"],
            tools_available="web_search, voice_to_text, read_file",
            answer=e["expected"],
        ).with_inputs("question", "tools_available")
        for e in successful_examples
        if e.get("success") and e.get("question") and e.get("expected")
    ]

    if len(examples) < 3:
        logger.info(f"Only {len(examples)} successful examples, need >= 3 for compilation")
        return None

    module = GAIAOptimizedSolver()

    try:
        optimizer = dspy.BootstrapFewShot(
            max_bootstrapped_demos=4,
            max_labeled_demos=4,
        )
        compiled = optimizer.compile(module, trainset=examples)

        # Ensure save directory exists
        save_dir = Path(save_path).parent
        save_dir.mkdir(parents=True, exist_ok=True)
        compiled.save(save_path)

        logger.info(f"GAIA DSPy module compiled from {len(examples)} examples -> {save_path}")
        return compiled
    except Exception as e:
        logger.warning(f"DSPy compilation failed: {e}")
        return None
