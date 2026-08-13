"""
agents/cognition/mutation_system.py
Mutation system for evolving agent traits and symbolic structures.
"""

from __future__ import annotations

import random

from agents.agent_constants import SYLLABLES
from evolution.programs import mutate_program


class MutationSystem:
    """Owns the mutation mechanics; mixins should only delegate here."""

    def __init__(self, owner):
        self.owner = owner

    def mutate(self, parent_program, parent_agent=None):
        o = self.owner

        # --- 0. Genetic code (DSL list) --------------------------------
        if isinstance(parent_program, list):
            try:
                o.program = mutate_program(parent_program)
            except Exception:
                o.program = list(parent_program)

        # ============================================================
        # 1. Mutate behavioural traits
        # ============================================================
        for key in getattr(o, "traits", {}):
            drift = random.uniform(-0.05, 0.05)
            o.traits[key] = o._clamp(o.traits[key] + drift, 0.0, 1.0)

        if random.random() < 0.1:
            o.traits["curiosity"] = o._clamp(
                o.traits.get("curiosity", 0.5) + random.uniform(0.0, 0.05),
                0.0,
                1.0,
            )

        # ============================================================
        # 2. Memory mutation
        # ============================================================
        if hasattr(o, "memory_influence"):
            o.memory_influence = o._clamp(
                o.memory_influence + random.uniform(-0.05, 0.05), 0.0, 1.0
            )
        if hasattr(o, "memory_decay_rate"):
            o.memory_decay_rate = o._clamp(
                o.memory_decay_rate + random.uniform(-0.05, 0.05), 0.0, 1.0
            )

        # ============================================================
        # 3. Soft social pruning (forget weak ties)
        # ============================================================
        social_memory = getattr(o, "social_memory", None)
        if random.random() < 0.06 and isinstance(social_memory, dict) and social_memory:
            partner = random.choice(list(social_memory.keys()))
            rec = social_memory.get(partner)
            if isinstance(rec, dict):
                if "trust_delta" in rec:
                    rec["trust_delta"] *= 0.5
                if "mean_outcome" in rec:
                    rec["mean_outcome"] *= 0.5

        # ============================================================
        # 4. Language bias drift
        # ============================================================
        utter_bias = getattr(o, "utter_bias", None)
        if isinstance(utter_bias, dict):
            utter_bias["length_bias"] = o._clamp(
                float(utter_bias.get("length_bias", 0.5)) + random.uniform(-0.05, 0.05),
                0.0,
                1.0,
            )
            utter_bias["repeat_bias"] = o._clamp(
                float(utter_bias.get("repeat_bias", 0.5)) + random.uniform(-0.05, 0.05),
                0.0,
                1.0,
            )

            prefs = utter_bias.get("symbol_preferences")
            if isinstance(prefs, dict):
                for s in SYLLABLES:
                    if s in prefs:
                        prefs[s] = o._clamp(float(prefs[s]) + random.uniform(-0.05, 0.05), -1.5, 1.5)

        # ============================================================
        # 5. Counting/numeric system inheritance
        # ============================================================
        if not hasattr(o, "counting"):
            raise RuntimeError("NumericSystem must be initialised in InitMixin")

        if parent_agent is not None and hasattr(parent_agent, "counting"):
            try:
                parent_counting = parent_agent.counting
                o.counting.base = parent_counting.base
                o.counting.symbols = dict(parent_counting.symbols)
                o.counting.inverse = dict(parent_counting.inverse)
                if random.random() < 0.05:
                    o.counting.mutate_base()
            except Exception:
                pass

        # Ensure symbols exist for all digits
        try:
            base = int(getattr(o.counting, "base", 10))
            for n in range(base):
                if n not in o.counting.symbols:
                    new_word = random.choice(SYLLABLES)
                    o.counting.symbols[n] = new_word
                    o.counting.inverse[new_word] = n
        except Exception:
            pass

        # ============================================================
        # 6. Symbol map cultural inheritance
        # ============================================================
        if parent_agent and getattr(parent_agent, "symbol_map", None):
            o.symbol_map = dict(parent_agent.symbol_map)
            if o.symbol_map and random.random() < 0.1:
                k = random.choice(list(o.symbol_map.keys()))
                old = o.symbol_map[k]
                o.symbol_map[k] = f"{old}{random.choice(['a', 'e', 'o', 'i', 'u'])}"
        else:
            o.symbol_map = {}

        # ============================================================
        # 7. Minor numeric symbol drift
        # ============================================================
        for k in list(getattr(o, "symbol_map", {}).keys()):
            if random.random() < 0.02:
                o.symbol_map[k] = f"{o.symbol_map[k]}{random.choice(['-x', '-y', '-z'])}"

        # ============================================================
        # 8. Post-mutation emotional nudges
        # ============================================================
        if hasattr(o, "state_event"):
            o.state_event("cultural_sync")

