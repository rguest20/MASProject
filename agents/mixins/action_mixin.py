# agents/mixins/action_mixin.py

import random

class ActionMixin:

    def compute_action_weights(self):
        S = self.state
        w = {}

        w["seek_social"] = (
            0.6*S["loneliness"] +
            0.2*(1-S["confidence"]) +
            0.2*S["happiness"]
        )

        w["attempt_learning"] = (
            0.6*S["curiosity"] +
            0.2*(1-S["confidence"]) +
            0.2*(1-S["frustration"])
        )

        w["attempt_teaching"] = (
            0.5*S["confidence"] +
            0.3*S["purpose"] +
            0.2*(1-S["loneliness"])
        )

        w["attempt_math_challenge"] = (
            0.4*S["confidence"] +
            0.6*S["curiosity"]
        )

        w["attempt_language_challenge"] = (
            0.7*S["curiosity"] +
            0.3*(1-S["frustration"])
        )

        w["explore_semantic_space"] = (
            0.8*S["curiosity"] +
            0.2*S["happiness"]
        )

        w["reorganize_concepts"] = (
            0.8*S["frustration"] +
            0.2*(1-S["confidence"])
        )

        return {k:max(0.01,min(1.0,v)) for k,v in w.items()}

    def choose_action(self):
        w = self.compute_action_weights()
        keys = list(w.keys())
        vals = list(w.values())
        action = random.choices(keys, vals)[0]
        self.last_action = action
        return action

    def perform_action(self, coordinator=None):
        action = self.choose_action()

        if action == "explore_semantic_space":
            self.state_event("curiosity_boost")

        if coordinator:
            coordinator.enqueue_action(self, action)

        return action