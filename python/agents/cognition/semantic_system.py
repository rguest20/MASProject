"""Semantic system for managing agent semantic memory and associations."""

from __future__ import annotations

import random
import math
import re
import numpy as np
from config import DIMS
from agents.cognition.semantic_utils import add, sub, scale
from typing import Optional

from agents.cognition.semantic_vector_mixin import SemanticVectorMixin
from agents.cognition.semantic_association_mixin import SemanticAssociationMixin
from agents.cognition.semantic_family_mixin import SemanticFamilyMixin

class SemanticSystem(SemanticVectorMixin, SemanticAssociationMixin, SemanticFamilyMixin):
    def __init__(self, owner):
        self.owner = owner

        # Shared semantic container (some mixins/systems expect this to exist).
        if not hasattr(self.owner, "semantic") or not isinstance(getattr(self.owner, "semantic", None), dict):
            self.owner.semantic = {}
        self.owner.semantic.setdefault("concept_tokens", {})
        self.concept_tokens = self.owner.semantic["concept_tokens"]

        self.family_system = self.SemanticFamily(owner)
        # Back-compat: older code expects direct access
        self.families = self.family_system.families
        self.vectors = {}
        self.tokens = {}
        self.links = {}
        self.last_used = {}

        # Flavour channels and latent fields
        self.semantic_flavour = {}
        dim = self._semantic_dim()
        self.flavour_axes = {
            "objectness": np.random.normal(size=dim),
            "processness": np.random.normal(size=dim),
            "relationness": np.random.normal(size=dim),
            "transformness": np.random.normal(size=dim),
            "causativeness": np.random.normal(size=dim),
            "temporalness": np.random.normal(size=dim),
        }
        for k, v in list(self.flavour_axes.items()):
            n = np.linalg.norm(v) + 1e-9
            self.flavour_axes[k] = v / n

        self.flavour_gain = 0.02

        base_dim = dim
        self.latent_attractors = {
            "objectness": self._randvec(scale=1.0),
            "processness": self._randvec(scale=1.0),
            "relationness": self._randvec(scale=1.0),
            "transformness": self._randvec(scale=1.0),
            "causativeness": self._randvec(scale=1.0),
            "temporalness": self._randvec(scale=1.0),
        }

        # Predefine "why" token for reasoning
        self.vectors["why"] = self._randvec()
        self.owner.vocab.add("why")
