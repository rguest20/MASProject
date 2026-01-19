# agents/mixins/decision_mixin.py

from agents.cognition.decision_system import DecisionSystem


class DecisionMixin:
    """Coordinator mixin that delegates action selection to DecisionSystem."""

    def _init_decision_system(self):
        if not hasattr(self, "decision_system"):
            self.decision_system = DecisionSystem(owner=self)
