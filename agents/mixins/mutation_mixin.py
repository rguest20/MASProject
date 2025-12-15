# agents/mixins/mutation_mixin.py

import random
from agents.agent_constants import SYLLABLES
from evolution.counting import CountingSystem
from evolution.programs import mutate_program


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

    # ============================================================
    #  MAIN MUTATION ENTRY POINT
    # ============================================================
    def mutate(self, parent_program, parent_agent=None):
        """
        Called by Coordinator when creating a new agent.
        parent_program : genome list (DSL)
        parent_agent   : the agent to inherit culture/traits from
        """

        # --- 0. Genetic code (DSL list) --------------------------------
        if isinstance(parent_program, list):
            try:
                self.program = mutate_program(parent_program)
            except Exception:
                self.program = list(parent_program)

        # ============================================================
        # 1. Mutate behavioural traits
        # ============================================================
        for key in self.traits:
            drift = random.uniform(-0.05, 0.05)     # soft mutation
            self.traits[key] = self._clamp(
                self.traits[key] + drift, 0.0, 1.0
            )

        # Strengthen teaching/learning reliability slightly over gens
        if random.random() < 0.1:
            self.traits["curiosity"] = self._clamp(
                self.traits["curiosity"] + random.uniform(0.0, 0.05), 0.0, 1.0
            )

        # ============================================================
        # 2. Memory mutation
        # ============================================================
        self.memory_influence = self._clamp(
            self.memory_influence + random.uniform(-0.05, 0.05),
            0.0, 1.0
        )

        self.memory_decay_rate = self._clamp(
            self.memory_decay_rate + random.uniform(-0.05, 0.05),
            0.0, 1.0
        )

        # ============================================================
        # 3. Soft social pruning (forget weak ties)
        # ============================================================
        if random.random() < 0.06 and self.social_memory:
            partner = random.choice(list(self.social_memory.keys()))
            rec = self.social_memory.get(partner)
            if rec:
                rec["trust_delta"] *= 0.5
                rec["mean_outcome"] *= 0.5

        # ============================================================
        # 4. Language bias drift
        # ============================================================
        self.utter_bias["length_bias"] = self._clamp(
            self.utter_bias["length_bias"] + random.uniform(-0.05, 0.05),
            0.0, 1.0
        )

        self.utter_bias["repeat_bias"] = self._clamp(
            self.utter_bias["repeat_bias"] + random.uniform(-0.05, 0.05),
            0.0, 1.0
        )

        for s in SYLLABLES:
            if s in self.utter_bias["symbol_preferences"]:
                self.utter_bias["symbol_preferences"][s] = self._clamp(
                    self.utter_bias["symbol_preferences"][s] + random.uniform(-0.05, 0.05),
                    -1.5, 1.5
                )

        # ============================================================
        # 5. Counting system inheritance
        # ============================================================
        if parent_agent:
            try:
                parent_counting = parent_agent.counting

                # inherit base + symbols
                if not hasattr(self, "counting"):
                    raise RuntimeError("CountingSystem must be initialised in InitMixin")
                self.counting.base = parent_counting.base
                self.counting.symbols = dict(parent_counting.symbols)
                self.counting.inverse = dict(parent_counting.inverse)

                # small chance to mutate base
                if random.random() < 0.05:
                    self.counting.mutate_base()

            except Exception:
                # fallback: new counting system
                if not hasattr(self, "counting"):
                    raise RuntimeError("CountingSystem must be initialised in InitMixin")
        else:
            # brand new lineage
            if not hasattr(self, "counting"):
                raise RuntimeError("CountingSystem must be initialised in InitMixin")

        # Ensure symbols exist for all digits
        try:
            base = self.counting.base
            for n in range(base):
                if n not in self.counting.symbols:
                    new_word = random.choice(SYLLABLES)
                    self.counting.symbols[n] = new_word
                    self.counting.inverse[new_word] = n
        except Exception:
            pass

        # ============================================================
        # 6. Symbol map cultural inheritance
        # ============================================================
        if parent_agent and getattr(parent_agent, "symbol_map", None):
            # deep copy from parent
            self.symbol_map = dict(parent_agent.symbol_map)

            # low chance drift on a single number word
            if self.symbol_map and random.random() < 0.1:
                k = random.choice(list(self.symbol_map.keys()))
                old = self.symbol_map[k]
                mutated = f"{old}{random.choice(['a','e','o','i','u'])}"
                self.symbol_map[k] = mutated

        else:
            self.symbol_map = {}

        # ============================================================
        # 7. Minor numeric symbol drift
        # ============================================================
        for k in list(self.symbol_map.keys()):
            if random.random() < 0.02:
                self.symbol_map[k] = f"{self.symbol_map[k]}{random.choice(['-x','-y','-z'])}"

        # ============================================================
        # 8. Post-mutation emotional nudges
        # ============================================================
        if hasattr(self, "state_event"):
            self.state_event("cultural_sync")  # newborn feels aligned to parent