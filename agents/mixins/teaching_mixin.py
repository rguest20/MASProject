# agents/mixins/teaching_mixin.py

from agents.cognition.teaching_system import TeachingSystem


class TeachingMixin:
    """Coordinator mixin that delegates teaching to TeachingSystem."""

    def _init_teaching_system(self):
        if not hasattr(self, "teaching_system"):
            self.teaching_system = TeachingSystem(owner=self)
        self.teaching_system.init()
