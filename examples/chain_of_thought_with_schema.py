"""
Custom ChainOfThought that enforces JSON schema for Claude CLI.

This wrapper ensures the signature is passed to the LM so it can extract
the JSON schema and enforce structured output via --json-schema flag.
"""

import dspy
from typing import Any


class ChainOfThoughtWithSchema(dspy.ChainOfThought):
    """ChainOfThought that passes signature to LM for JSON schema enforcement."""

    def __init__(self, signature, **kwargs):
        super().__init__(signature, **kwargs)
        # Store signature for later use
        self._signature_class = signature

    def forward(self, **kwargs):
        """Forward pass with signature injection for JSON schema."""
        # Inject signature into kwargs so enhanced Claude CLI can extract schema
        kwargs['signature'] = self._signature_class

        # Call parent forward
        return super().forward(**kwargs)
