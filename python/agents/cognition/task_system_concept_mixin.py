"""
agents/cognition/task_system_v2.py
Task solving system (v2).
"""

import random
import re

from agents.agent_constants import REL_GT, REL_LT, REL_EQ

from agents.cognition.task_system_concept_dialogue_mixin import TaskSystemConceptDialogueMixin
from agents.cognition.task_system_concept_reasoning_mixin import TaskSystemConceptReasoningMixin

class TaskSystemConceptMixin(TaskSystemConceptDialogueMixin, TaskSystemConceptReasoningMixin):
    pass
