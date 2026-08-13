"""
agents/language.py

Compatibility shim.

The implementation moved to `agents/cognition/language_organ.py` as part of the
systems/mixins refactor. Existing imports continue to work.
"""

from agents.cognition.language_organ import (  # noqa: F401
    CONSONANTS,
    SYLLABLE_PATTERNS,
    VOWELS,
    LanguageOrgan,
    invent_word,
    make_syllable,
    mutate_syllable,
)

__all__ = [
    "VOWELS",
    "CONSONANTS",
    "SYLLABLE_PATTERNS",
    "make_syllable",
    "mutate_syllable",
    "invent_word",
    "LanguageOrgan",
]

