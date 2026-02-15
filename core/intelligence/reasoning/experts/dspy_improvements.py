"""
DSPy Improvements Integration

This module provides utilities to integrate stored improvements
back into DSPy modules so they can be used in future runs.
"""

import logging
from typing import Any, Dict, List

try:
    import dspy

    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False
    dspy = None

logger = logging.getLogger(__name__)


def apply_improvements_to_dspy_module(
    module: Any, improvements: List[Dict[str, Any]], max_improvements: int = 10
) -> None:
    """
    Apply stored improvements to a DSPy module.

    This updates the module's instructions or signature description
    so DSPy can use the learned patterns in future generations.

    Args:
        module: DSPy module to update
        improvements: List of improvement dictionaries
        max_improvements: Maximum number of improvements to apply
    """
    if not DSPY_AVAILABLE:
        logger.warning("DSPy not available, cannot apply improvements")
        return

    if not isinstance(module, dspy.Module):
        logger.warning(f"Module is not a DSPy module: {type(module)}")
        return

    if not improvements:
        return

    # Get the most recent/relevant improvements
    recent_improvements = improvements[-max_improvements:]

    # Extract learned patterns
    patterns = []
    for imp in recent_improvements:
        pattern = imp.get("learned_pattern", "")
        if pattern:
            patterns.append(pattern)

    if not patterns:
        return

    # Format patterns as instructions
    instructions_text = "\n\n".join(
        [f"Learned Pattern {i+1}: {pattern}" for i, pattern in enumerate(patterns)]
    )

    # Try to update module instructions
    # Method 1: Update signature description
    if hasattr(module, "signature") and module.signature:
        original_desc = getattr(module.signature, "__doc__", "") or ""
        if instructions_text not in original_desc:
            new_desc = f"{original_desc}\n\n## Learned Patterns\n\n{instructions_text}"
            module.signature.__doc__ = new_desc
            logger.info(f"Updated signature description with {len(patterns)} patterns")

    # Method 2: Update module instructions attribute
    if hasattr(module, "instructions"):
        if isinstance(module.instructions, list):
            module.instructions.extend(patterns)
        elif isinstance(module.instructions, str):
            module.instructions = (
                f"{module.instructions}\n\n## Learned Patterns\n\n{instructions_text}"
            )
        else:
            module.instructions = patterns
        logger.info(f"Updated module instructions with {len(patterns)} patterns")

    # Method 3: Store in metadata
    if hasattr(module, "_metadata"):
        if module._metadata is None:
            module._metadata = {}
        module._metadata["learned_patterns"] = patterns
        module._metadata["improvements_count"] = len(improvements)

    logger.info(f"Applied {len(patterns)} improvements to DSPy module")


def create_improvements_context(improvements: List[Dict[str, Any]]) -> str:
    """
    Create a context string from improvements to pass to DSPy module.

    Args:
        improvements: List of improvement dictionaries

    Returns:
        Formatted string with learned patterns
    """
    if not improvements:
        return ""

    # Get most recent improvements
    recent = improvements[-5:]  # Last 5 improvements

    patterns = []
    for imp in recent:
        pattern = imp.get("learned_pattern", "")
        if pattern:
            patterns.append(f"- {pattern}")

    if not patterns:
        return ""

    return "\n".join(["## Previously Learned Patterns:", *patterns])


def inject_improvements_into_signature(
    signature_class: type, improvements: List[Dict[str, Any]]
) -> type:
    """
    Create a new signature class with improvements injected into the docstring.

    Args:
        signature_class: Original DSPy Signature class
        improvements: List of improvement dictionaries

    Returns:
        New signature class with improvements in docstring
    """
    if not DSPY_AVAILABLE:
        return signature_class

    if not improvements:
        return signature_class

    # Get original docstring
    original_doc = signature_class.__doc__ or ""

    # Extract patterns
    patterns = []
    for imp in improvements[-5:]:  # Last 5
        pattern = imp.get("learned_pattern", "")
        if pattern:
            patterns.append(f"  - {pattern}")

    if not patterns:
        return signature_class

    # Create new docstring with improvements
    improvements_section = "\n\n## Learned Patterns (from previous training):\n" + "\n".join(
        patterns
    )
    new_doc = original_doc + improvements_section

    # Create new class with updated docstring
    new_signature = type(signature_class.__name__, (signature_class,), {"__doc__": new_doc})

    # Copy all fields from original
    for attr_name in dir(signature_class):
        if not attr_name.startswith("_"):
            attr = getattr(signature_class, attr_name)
            if not hasattr(new_signature, attr_name):
                setattr(new_signature, attr_name, attr)

    return new_signature
