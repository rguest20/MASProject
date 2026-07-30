import math
import random
import re
from collections import Counter

from agents.cognition.semantic_utils import add, scale, cos_sim
from evolution.coordinator_settings import (
    REFERENTIAL_BONUS,
    PHASE2_LR,
    REWARD_EPS,
    REWARD_TEMP,
    TEACH_PAIRS_PER_PASS,
    TEACH_ACC_TEMP,
)

from evolution.mixins.coordinator_reading_mixin import CoordinatorReadingMixin
from evolution.mixins.coordinator_dialogue_mixin import CoordinatorDialogueMixin

class CoordinatorLanguageMixin(CoordinatorReadingMixin, CoordinatorDialogueMixin):
    """
    Language, teaching, and dialogue utilities factored out of the core
    Coordinator.  These methods manipulate agent communication without
    needing to live inside the already large coordinator module.
    """

    _READING_TOKEN = re.compile(r"[a-z][a-z'-]{0,31}", re.IGNORECASE)

    _READING_BRIDGE_STOP_WORDS = frozenset({
        "a", "all", "an", "and", "are", "as", "at", "be", "been", "but", "by",
        "can", "could", "did", "do", "does", "doing", "done", "each", "every", "for",
        "from", "had", "has", "have", "he", "her", "his", "i", "if", "in", "into",
        "is", "it", "its", "just", "let", "may", "me", "might", "more", "most", "much",
        "must", "my", "no", "not", "of", "on", "only", "or", "ought", "our", "out",
        "over", "shall", "she", "should", "so", "that", "than", "the", "their", "them",
        "then", "there", "they", "this", "to", "too", "up", "very", "was", "we", "were",
        "when", "will", "with", "would", "you", "your", "yet",
    })
