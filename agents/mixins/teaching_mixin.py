# agents/mixins/teaching_mixin.py

from agents.cognition.teaching_system import TeachingSystem


class TeachingMixin:
    """Coordinator mixin that delegates teaching to TeachingSystem."""

    def _init_teaching_system(self):
        if not hasattr(self, "teaching_system"):
            self.teaching_system = TeachingSystem(owner=self)
        self.teaching_system.init()

    # ------------------------------------------------------------
    # Back-compat API (CoordinatorLanguageMixin calls these)
    # ------------------------------------------------------------
    def apply_teaching_reward(self, student_id, reward):
        if hasattr(self, "teaching_system"):
            return self.teaching_system.apply_teaching_reward(student_id, reward)
        return None

    def apply_learning_reward(self, teacher_id, reward):
        if hasattr(self, "teaching_system"):
            return self.teaching_system.apply_learning_reward(teacher_id, reward)
        return None
