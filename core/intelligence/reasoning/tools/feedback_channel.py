"""
Re-export shim — canonical location is now:
    Jotty.core.intelligence.orchestration.learning.feedback_channel
"""

from Jotty.core.intelligence.orchestration.learning.feedback_channel import (
    FeedbackChannel,
    FeedbackMessage,
    FeedbackType,
)

__all__ = ["FeedbackChannel", "FeedbackMessage", "FeedbackType"]
