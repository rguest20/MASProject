# agents/decision.py

import random


class DecisionMixin:
    """
    Provides decide_action(), using emotional state and traits.
    Assumes EmotionMixin.init_emotional_state() has been called
    so self.state exists.
    """

    def decide_action(self) -> str:
        """
        Choose an action based on motivational state (curiosity, loneliness,
        satisfaction), personality traits, and energy.

        Returns:
            action (str) chosen from a fixed action set.
        """

        # ------ 1. Read emotional state safely ------
        state = getattr(self, "state", {}) or {}

        loneliness   = state.get("loneliness",   0.5)
        curiosity    = state.get("curiosity",    0.5)
        satisfaction = state.get("satisfaction", 0.5)
        happiness    = state.get("happiness",    0.5)
        frustration  = state.get("frustration",  0.0)
        confidence   = state.get("confidence",   0.5)
        purpose      = state.get("purpose",      0.5)

        energy       = getattr(self, "energy", 50.0)

        # legacy fields, if present, slightly influence
        curiosity_level = getattr(self, "curiosity_level", curiosity)

        # ------ 2. Personality traits ------
        traits = getattr(self, "traits", {}) or {}
        curiosity_trait = traits.get("curiosity",          0.5)
        teaching_drive  = traits.get("teaching_drive",     0.5)
        cooperation     = traits.get("cooperation_weight", 0.5)
        expressiveness  = traits.get("expressiveness",     0.5)

        # ------ 3. Base weights for each action ------
        weights = {
            "idle": 0.1,

            # reading is “safe curiosity”
            "read": (
                0.6 * curiosity_level +
                0.4 * curiosity_trait
            ),

            # math: more effortful curiosity
            "attempt_math": (
                0.7 * curiosity_level +
                0.3 * confidence
            ),

            # language: curiosity + expressiveness
            "attempt_language": (
                0.6 * expressiveness +
                0.4 * curiosity_trait
            ),

            # social: driven by loneliness + cooperation
            "seek_social": (
                0.8 * loneliness +
                0.4 * cooperation
            ),

            # teaching: driven by teaching drive + satisfaction + purpose
            "teach_other": (
                0.6 * teaching_drive +
                0.2 * satisfaction +
                0.2 * purpose
            ),

            # reflective semantic stuff:
            "reflect_semantic_space": (
                0.3 * curiosity +
                0.3 * (1.0 - satisfaction) +
                0.2 * frustration +
                0.2 * purpose
            ),

            "reorganize_concepts": (
                0.5 * curiosity_trait +
                0.5 * (1.0 - satisfaction)
            ),
        }

        # ------ 4. Low-energy gating ------
        if energy < 20:
            # collapse into rest / low-effort behaviours
            weights["idle"]            += 2.0
            weights["read"]            *= 0.4
            weights["attempt_math"]    *= 0.2
            weights["attempt_language"] *= 0.5
            weights["teach_other"]     *= 0.3

        # if very frustrated, push toward social or concept cleanup
        if frustration > 0.7:
            weights["seek_social"]        += 0.5
            weights["reorganize_concepts"] += 0.4

        # if very happy & purposeful, more likely to teach
        if happiness > 0.7 and purpose > 0.6:
            weights["teach_other"] += 0.5

        # ------ 5. Normalise and sample ------
        # ensure no negative weights
        for k in list(weights.keys()):
            if weights[k] < 0.0:
                weights[k] = 0.0

        total = sum(weights.values())
        if total <= 0:
            return "idle"

        actions = list(weights.keys())
        probs   = [w / total for w in weights.values()]

        return random.choices(actions, probs)[0]