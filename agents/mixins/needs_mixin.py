# agents/mixins/needs_mixin.py

import math

class NeedsMixin:
    """
    Lightweight motivational-needs subsystem reconstructed from the
    original agent code. Coordinator expects `update_needs()` to exist.
    """

    def _init_needs_system(self):
        # High-level motivational categories
        self.needs = {
            "energy": 1.0,        # vitality / exhaustion
            "social": 0.8,        # desire for contact
            "curiosity": 0.7,     # desire to learn
            "esteem": 0.5,        # sense of competence
            "play": 0.6,          # fun/creativity impulse
        }

        # smoothing / decay parameters
        self._needs_decay = 0.98

    # ------------------------------------------------------------
    # Main update → called once per generation by Coordinator
    # ------------------------------------------------------------
    def update_needs(self):
        """
        Recompute needs from emotional state + energy.
        Cheap, stable, non-explosive.
        """

        S = self.state
        N = self.needs

        # Energy need grows when energy low
        N["energy"] = 1.0 - min(1.0, self.energy / 100.0)

        # Loneliness maps into social need
        N["social"] = S["loneliness"]

        # Curiosity maps directly
        N["curiosity"] = S["curiosity"]

        # Esteem from confidence + happiness
        N["esteem"] = max(0.0, min(1.0, 0.5*S["confidence"] + 0.5*S["happiness"]))

        # Play from happiness minus frustration
        N["play"] = max(0.0, min(1.0, S["happiness"] - 0.5*S["frustration"]))

        # Apply smoothing to avoid violent jumps
        for k in N:
            N[k] = 0.5*N[k] + 0.5*max(0.0, min(1.0, N[k]))

    # ------------------------------------------------------------
    # Optional — slight decay every generation
    # ------------------------------------------------------------
    def decay_needs(self):
        for k in self.needs:
            target = 0.5
            self.needs[k] += 0.01 * (target - self.needs[k])
            self.needs[k] = max(0.0, min(1.0, self.needs[k]))