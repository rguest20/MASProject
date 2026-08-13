"""
agents/semantics.py

Compatibility shim.

The implementation moved to `agents/cognition/semantic_utils.py` as part of the
systems/mixins refactor. Existing imports continue to work.
"""

from agents.cognition.semantic_utils import *  # noqa: F403
from agents.cognition.semantic_utils import __all__  # noqa: F401
