# agents/mixins/task_mixin_v2.py

from agents.cognition.task_system_v2 import TaskSystemV2
from agents.mixins.task_mixin import TaskMixin


class TaskMixinV2(TaskMixin):
    """Coordinator mixin that delegates task solving to TaskSystemV2."""

    def _init_task_system(self):
        if not hasattr(self, "task_system_v2"):
            self.task_system_v2 = TaskSystemV2(owner=self)
        # Keep legacy init name; system forwards attribute writes to the agent.
        self.task_system_v2._init_task_system()
