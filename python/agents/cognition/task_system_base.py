"""
agents/cognition/task_system_v2.py
Task solving system (v2).
"""

import random
import re

from agents.agent_constants import REL_GT, REL_LT, REL_EQ


class TaskSystemBase:
    # Keep the task vocabulary in one inspectable place.  The coordinator can
    # still stage any of these tasks, while a missing handler now fails closed
    # instead of being hidden in a long conditional chain.
    TASK_HANDLERS = {
        "compare_numbers": "_solve_compare_numbers",
        "cooperative_compare_numbers": "_solve_cooperative_compare_numbers",
        "reconcile_counts": "_solve_reconcile_counts",
        "translate_number": "_solve_reconcile_counts",
        "translate_quantity": "_solve_reconcile_counts",
        "referential_signal": "_solve_referential_signal",
        "action_signal": "_solve_action_signal",
        "compositional_signal": "_solve_compositional_signal",
        "compositional_action_signal": "_solve_compositional_action_signal",
        "human_dictionary_link": "_solve_human_dictionary_link",
        "semantic_alignment": "_solve_semantic_gap",
        "agreement_dialogue": "_solve_agreement_dialogue",
        "explain_partner": "_solve_explain_partner_answer",
        "token_compress": "_solve_token_compression",
        "pref_align": "_solve_preference_alignment_dialogue",
        "describe_concept": "_solve_describe_concept",
        "action_reconstruction": "_solve_action_reconstruction",
        "similarity_debate": "_solve_similarity_debate",
        "narrative_chain": "_solve_narrative_chain",
        "role_assignment": "_solve_role_assignment",
        "misunderstanding_detection": "_solve_misunderstanding_detection",
        "property_attribution": "_solve_property_attribution",
        "verb_noun_compat": "_solve_verb_noun_compat",
        "definition_swap": "_solve_definition_swap",
        "prediction_task": "_solve_prediction_task",
    }

    def __init__(self, owner):
        object.__setattr__(self, "owner", owner)

    def __getattr__(self, name):
        return getattr(self.owner, name)

    def __setattr__(self, name, value):
        if name == "owner":
            object.__setattr__(self, name, value)
            return
        setattr(self.owner, name, value)

    def _init_task_system(self):
        # base task state
        self.solved_tasks = {}
        self.failed_tasks = {}
        self.last_task_gen = -1
        self.tasks_attempted_this_gen = set()
        self.last_teach_seen = 0

        # optional logging hooks
        self.last_coop_log_gen = -1

        # dialogue bookkeeping
        self.last_dialogue_proposal = None

    def _task_fail(self, task_id):
        """Record a failed task attempt on the owning agent."""
        if task_id is not None:
            self.failed_tasks[task_id] = self.failed_tasks.get(task_id, 0) + 1
        if hasattr(self, "state"):
            self.state["frustration"] = min(
                1.0,
                self.state.get("frustration", 0.0) + 0.05,
            )
        return None

    def try_solve_tasks(self, task_list, generation_index):
        # Always keep numeric help / teaching / family broadcast alive
        try:
            if hasattr(self, "maybe_answer_numeric_help"):
                self.maybe_answer_numeric_help()
        except Exception:
            pass

        try:
            if hasattr(self, "process_numeric_teaching"):
                self.process_numeric_teaching()
        except Exception:
            pass

        try:
            if hasattr(self, "maybe_broadcast_families"):
                self.maybe_broadcast_families()
        except Exception:
            pass

        if not task_list:
            return

        # generation reset
        if self.last_task_gen != generation_index:
            self.last_task_gen = generation_index
            self.tasks_attempted_this_gen = set()

        my_id = getattr(self, "id", None)

        def _is_task_available(t):
            tid = t.get("task_id")
            if tid in self.tasks_attempted_this_gen:
                return False

            assigned = t.get("assigned_agents", None)
            if assigned is not None and my_id is not None:
                # only explicitly assigned agents may attempt
                return my_id in assigned

            return True

        candidates = [t for t in task_list if _is_task_available(t)]
        if not candidates:
            return

        # An explicit assignment is a commitment by the coordinator: do not
        # let an unrelated task lottery make the intended listener skip the
        # communicative trial.  Unassigned background tasks remain bounded to
        # one random attempt per agent per generation.
        assigned = [task for task in candidates if task.get("assigned_agents") is not None]
        tasks_to_attempt = assigned or [random.choice(candidates)]
        for task in tasks_to_attempt:
            self._attempt_task(task)
