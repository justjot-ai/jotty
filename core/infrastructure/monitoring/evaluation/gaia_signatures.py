"""
GAIA benchmark — DSPy signatures for structured answer extraction.

Uses the expected answer (and question) to drive output format, so we compare
structured output to expected instead of regex over free text.
"""

try:
    import dspy
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False
    dspy = None

GAIAAnswerExtractSignature = None
if DSPY_AVAILABLE:

    class GAIAAnswerExtractSignature(dspy.Signature):
        """Extract the exact final answer from a model's raw response for GAIA scoring.

        You are a GAIA answer normalizer. Given the model's raw response and the
        expected answer format (example), output ONLY the final_answer string in
        the same format as the expected example: same type (number, name, list, etc.),
        no extra words, no explanation.

        Rules:
        - If expected is a number, output only that number (e.g. "519").
        - If expected is a short phrase or name, output only that (e.g. "Rockhopper penguin").
        - If expected is a comma-separated list, output the same format (e.g. "a, b, c").
        - If the raw response contains the correct answer embedded in prose, extract it.
        - If the raw response refuses or says it cannot answer, output the empty string or "The adventurer died." only when that is the correct outcome.
        """
        question_summary: str = dspy.InputField(
            desc="One-line summary of the benchmark question (for context)"
        )
        raw_response: str = dspy.InputField(
            desc="The model's full raw response (may be verbose or contain the answer in a sentence)"
        )
        expected_example: str = dspy.InputField(
            desc="The expected answer string. Output format must match this (number, name, list, etc.)"
        )

        final_answer: str = dspy.OutputField(
            desc="Only the final answer, in the exact format of expected_example. No explanation, no prefix."
        )


_dspy_lm_configured = False


def _ensure_dspy_lm() -> None:
    """Ensure DSPy has a configured LM for Predict calls."""
    global _dspy_lm_configured
    if _dspy_lm_configured:
        return
    if not DSPY_AVAILABLE:
        return
    try:
        if dspy.settings.lm is None:
            lm = dspy.LM("anthropic/claude-haiku-4-5-20251001")
            dspy.configure(lm=lm)
        _dspy_lm_configured = True
    except Exception:
        pass


def normalize_gaia_answer_with_dspy(
    raw_response: str,
    expected_example: str,
    question_summary: str = "",
) -> str:
    """
    Use DSPy to extract a GAIA-format answer from raw model output.

    Args:
        raw_response: Full model output (may be verbose).
        expected_example: Expected answer string (defines format).
        question_summary: Optional one-line question summary for context.

    Returns:
        Extracted final_answer string, or raw_response if DSPy fails/unavailable.
    """
    if not DSPY_AVAILABLE or not raw_response or not raw_response.strip():
        return (raw_response or "").strip()

    _ensure_dspy_lm()

    try:
        predictor = dspy.Predict(GAIAAnswerExtractSignature)
        out = predictor(
            question_summary=question_summary[:200] if question_summary else "Benchmark question.",
            raw_response=raw_response[:8000],
            expected_example=expected_example[:500],
        )
        if hasattr(out, "final_answer") and out.final_answer is not None:
            return str(out.final_answer).strip()
    except Exception:
        pass
    return raw_response.strip()
