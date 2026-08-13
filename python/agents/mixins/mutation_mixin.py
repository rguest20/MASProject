# agents/mixins/mutation_mixin.py

from agents.cognition.mutation_system import MutationSystem


class MutationMixin:
    """
    Handles all mutation processes:
      • DSL genome mutation
      • trait mutation
      • language-bias drift
      • symbol map inheritance + drift
      • counting system inheritance + occasional mutation
      • memory mutation
      • soft social pruning
      • semantic vocabulary inheritance safety
    """

    def _init_mutation_system(self):
        if not hasattr(self, "mutation_system"):
            self.mutation_system = MutationSystem(owner=self)

    # ============================================================
    #  MAIN MUTATION ENTRY POINT
    # ============================================================
    def mutate(self, parent_program, parent_agent=None):
        if not hasattr(self, "mutation_system"):
            self._init_mutation_system()
        return self.mutation_system.mutate(parent_program, parent_agent=parent_agent)
