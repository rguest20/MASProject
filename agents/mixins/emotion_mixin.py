# agents/mixins/emotion_mixin.py

import math
import random

class EmotionMixin:
    """
    Provides:
      - emotional state dictionary
      - homeostasis drift
      - event-based adjustments
      - emotional updates after actions
    """

    # ----------------------------------------------------
    # Initialisation
    # ----------------------------------------------------
    def _init_emotion_system(self):

        # Core emotional dimensions (normalised 0..1)
        self.state = {
            "happiness":     0.5,
            "loneliness":    0.5,
            "confidence":    0.5,
            "frustration":   0.5,
            "curiosity":     0.5,
            "satisfaction":  0.5,
            "purpose":       0.5,
            "expressiveness":0.5,
        }

        # event rule table — mixins can use it
        self.STATE_RULES = {
            "successful_comm": {
                "loneliness":  -0.05,
                "happiness":   +0.05,
                "confidence":  +0.04,
                "frustration": -0.03,
            },
            "failed_comm": {
                "loneliness":  +0.03,
                "happiness":   -0.03,
                "confidence":  -0.04,
                "frustration": +0.07,
            },
            "teaching_success": {
                "loneliness":  -0.03,
                "happiness":   +0.06,
                "confidence":  +0.08,
                "purpose":     +0.04,
                "frustration": -0.02,
            },
            "teaching_failure": {
                "happiness":   -0.04,
                "confidence":  -0.06,
                "frustration": +0.06,
            },
            "learning_success": {
                "curiosity":   +0.06,
                "happiness":   +0.03,
                "confidence":  +0.04,
                "frustration": -0.03,
                "purpose":     +0.03,
            },
            "learning_failure": {
                "frustration": +0.05,
                "confidence":  -0.03,
                "curiosity":   +0.01,
            },
            "concept_cleanup": {
                "frustration": -0.12,
                "satisfaction": +0.05,
                "purpose":      +0.04,
            },
            "curiosity_boost": {
                "curiosity": +0.03,
                "frustration": -0.02,
            },
        }

    # ----------------------------------------------------
    # Clamp utility
    # ----------------------------------------------------
    def _clamp01(self, v):
        return max(0.0, min(1.0, float(v)))

    # ----------------------------------------------------
    # Event-based state update
    # ----------------------------------------------------
    def state_event(self, event_name):
        if not hasattr(self, "STATE_RULES"):
            return
        if event_name not in self.STATE_RULES:
            return

        rules = self.STATE_RULES[event_name]
        S = self.state

        for key, delta in rules.items():
            if key in S:
                S[key] = self._clamp01(S[key] + delta)

    # ----------------------------------------------------
    # Passive emotional drift each tick
    # ----------------------------------------------------
    def decay_states(self):
        """
        Pull emotions back toward neutral (0.5) slowly.
        Frustration decays faster.
        """
        S = self.state
        for k in S:
            target = 0.5
            rate = 0.02 if k == "frustration" else 0.01
            S[k] = self._clamp01(S[k] + rate * (target - S[k]))

    # ----------------------------------------------------
    # Action-based emotional updates  
    # ----------------------------------------------------
    def update_emotional_state(self, action, success=None):
        """
        Applies emotional dynamics after an action.
        """
        S = self.state

        curiosity_trait = self.traits.get("curiosity", 0.5)
        patience_trait  = self.traits.get("patience", 0.5)
        cooperation     = self.traits.get("cooperation_weight", 0.5)
        expressiveness  = self.traits.get("expressiveness", 0.5)

        # --- natural drift first ---
        self.decay_states()

        # --- apply action-driven rules ---
        if action == "seek_social":
            if success:
                S["loneliness"] -= 0.15 * cooperation
                S["happiness"]  += 0.10 * cooperation
                S["confidence"] += 0.05
            else:
                S["loneliness"]  += 0.10 * (1 - patience_trait)
                S["frustration"] += 0.05

        elif action == "explore_semantic_space":
            S["curiosity"]   += 0.04
            S["frustration"] -= 0.05

        elif action == "reorganize_concepts":
            S["frustration"]  -= 0.12
            S["satisfaction"] += 0.05
            S["purpose"]      += 0.04

        # clamp everything
        for k in S:
            S[k] = self._clamp01(S[k])

        # === NUMERIC COLLISION DISCOMFORT (explorer-driven) ===
        collisions = self._numeric_collision_score()
        if collisions > 0:
            # scaled by curiosity
            self.intrinsic_discomfort = \
                self.intrinsic_discomfort + 0.05 * self.curiosity * collisions
        else:
            # slight relief if improving
            self.intrinsic_discomfort *= 0.97