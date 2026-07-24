"""
agents/cognition/decision_system.py
Decision system for selecting agent actions based on state and traits.
"""

from __future__ import annotations

import random


class DecisionSystem:
    """Owns action selection; mixins should delegate here."""

    def __init__(self, owner):
        self.owner = owner

    def decide_action(self) -> str:
        o = self.owner

        state = getattr(o, "state", {}) or {}

        loneliness = state.get("loneliness", 0.5)
        curiosity = state.get("curiosity", 0.5)
        satisfaction = state.get("satisfaction", 0.5)
        happiness = state.get("happiness", 0.5)
        frustration = state.get("frustration", 0.0)
        confidence = state.get("confidence", 0.5)
        purpose = state.get("purpose", 0.5)

        energy = getattr(o, "energy", 50.0)

        curiosity_level = getattr(o, "curiosity_level", curiosity)

        traits = getattr(o, "traits", {}) or {}
        curiosity_trait = traits.get("curiosity", 0.5)
        teaching_drive = traits.get("teaching_drive", 0.5)
        cooperation = traits.get("cooperation_weight", 0.5)
        expressiveness = traits.get("expressiveness", 0.5)

        # These action names are the public vocabulary accepted by
        # ``orchestrate_action``.  Keeping the mapping here prevents agents
        # from paying an energy cost for a decision the environment cannot
        # execute.
        weights = {
            "idle": 0.1,
            "attempt_learning": (0.6 * curiosity_level + 0.4 * curiosity_trait),
            "attempt_math_challenge": (0.7 * curiosity_level + 0.3 * confidence),
            "attempt_language_challenge": (0.6 * expressiveness + 0.4 * curiosity_trait),
            "seek_social": (0.8 * loneliness + 0.4 * cooperation),
            "attempt_teaching": (0.6 * teaching_drive + 0.2 * satisfaction + 0.2 * purpose),
            "explore_semantic_space": (
                0.3 * curiosity + 0.3 * (1.0 - satisfaction) + 0.2 * frustration + 0.2 * purpose
            ),
            "reorganize_concepts": (0.5 * curiosity_trait + 0.5 * (1.0 - satisfaction)),
        }

        if energy < 20:
            weights["idle"] += 2.0
            weights["attempt_learning"] *= 0.4
            weights["attempt_math_challenge"] *= 0.2
            weights["attempt_language_challenge"] *= 0.5
            weights["attempt_teaching"] *= 0.3

        if frustration > 0.7:
            weights["seek_social"] += 0.5
            weights["reorganize_concepts"] += 0.4

        if happiness > 0.7 and purpose > 0.6:
            weights["attempt_teaching"] += 0.5

        for k in list(weights.keys()):
            if weights[k] < 0.0:
                weights[k] = 0.0

        total = sum(weights.values())
        if total <= 0:
            return "idle"

        actions = list(weights.keys())
        probs = [w / total for w in weights.values()]
        return random.choices(actions, probs)[0]
