# agents/mixins/semantic_mixin.py

import random
import numpy as np
from collections import defaultdict

from agents.cognition.semantic_utils import (
    rand_vec, add, sub, scale, cos_sim,
)
from agents.agent_constants import SYLLABLES

from agents.mixins.semantic_initialisation_mixin import SemanticInitialisationMixin
from agents.mixins.semantic_graph_mixin import SemanticGraphMixin
from agents.mixins.semantic_operations_mixin import SemanticOperationsMixin

class SemanticMixin(SemanticInitialisationMixin, SemanticGraphMixin, SemanticOperationsMixin):
    pass
