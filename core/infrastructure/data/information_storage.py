"""
Information Storage — Re-export Shim
======================================

Canonical location: ``core.intelligence.memory.information_storage``
This module re-exports for backward compatibility.
"""

from Jotty.core.intelligence.memory.information_storage import (
    InformationTheoreticStorage,
    InformationWeightedMemory,
    SurpriseEstimator,
)

# SurpriseSignature requires DSPy - conditionally re-export
try:
    from Jotty.core.intelligence.memory.information_storage import SurpriseSignature
except ImportError:
    pass

__all__ = [
    "InformationTheoreticStorage",
    "InformationWeightedMemory",
    "SurpriseEstimator",
    "SurpriseSignature",
]
