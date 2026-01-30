"""
agents/cognition/pragmatic_system.py
Pragmatic system for managing agent communication and behavior.
"""

from __future__ import annotations

import random


class PragmaticSystem:
    """
    Manages the pragmatic aspects of an agent's communication and behavior.
    """
    def __init__(self, owner):
        self.owner = owner
        self.mode_bias = {}       # e.g. ask/assert/mirror
        self.state_bias = {}      # alpha/beta/...
        self.repetition_score = {}
        self.dialogue_stats = {"turns_seen": 0, "turns_spoken": 0}
        # legacy access: some code expects this on the agent
        self.owner.dialogue_stats = self.dialogue_stats

    # -------------------------------------------------
    # Internal helpers
    # -------------------------------------------------
    def _clamp(self, x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, x))

    def get_trust_to(self, other_id: int) -> float:
        tc = getattr(self.owner, "trust_channels", None)
        if not isinstance(tc, dict):
            return 0.0
        prof = tc.get(other_id)
        if not isinstance(prof, dict):
            return float(prof) if isinstance(prof, (int, float)) else 0.0
        vals = [v for v in prof.values() if isinstance(v, (int, float))]
        if not vals:
            return 0.0
        return self._clamp(sum(vals) / len(vals), -1.0, 1.0)

    def base_talk_probability(self) -> float:
        traits = getattr(self.owner, "traits", {})
        chattiness = traits.get("chattiness", 0.5)
        curiosity = traits.get("curiosity", 0.5)
        p = 0.05 + 0.25 * float(chattiness) + 0.20 * float(curiosity)
        mot = getattr(self.owner, "motivation", None)
        if isinstance(mot, dict) and "social" in mot:
            p *= (0.5 + float(mot.get("social", 0.5)))
        return self._clamp(p, 0.02, 0.60)

    def semantic_affinity_to(self, other_id: int) -> float:
        if not hasattr(self.owner, "semantic_system"):
            return 0.0
        if not hasattr(self.owner, "identity_system"):
            return 0.0

        me = getattr(self.owner, "identity_token", None)
        if not isinstance(me, str):
            me = getattr(self.owner.identity_system, "identity_token", None)
        if not isinstance(me, str):
            return 0.0

        other_tok = self.owner.identity_system.get_or_create_nickname_for_id(other_id)
        if not other_tok:
            return 0.0
        vecs = self.owner.semantic_system.vectors
        if me not in vecs or other_tok not in vecs:
            return 0.0
        if hasattr(self.owner, "semantic_distance"):
            return 1.0 - float(self.owner.semantic_distance(me, other_tok))
        return float(self.owner.semantic_system.similarity(me, other_tok))

    def social_affinity(self, other) -> float:
        trust = self.get_trust_to(other.id)
        sem = self.semantic_affinity_to(other.id)
        curiosity = float(getattr(self.owner, "traits", {}).get("curiosity", 0.5))
        noise = random.uniform(-0.1, 0.1)
        score = (0.55 * trust) + (0.35 * sem) + (0.10 * curiosity) + noise
        return float(score)

    # -------------------------------------------------
    # Partner selection
    # -------------------------------------------------
    def choose_conversation_partner(self, agents):
        if random.random() > self.base_talk_probability():
            return None
        best_id = None
        best_score = None
        for other in agents:
            if other is self.owner:
                continue
            s = self.social_affinity(other)
            if best_score is None or s > best_score:
                best_score = s
                best_id = other.id
        if best_id is None or best_score is None or best_score < 0.05:
            return None
        return best_id

    # -------------------------------------------------
    # Dialogue bookkeeping
    # -------------------------------------------------
    def mark_spoken_turn(self):
        self.dialogue_stats["turns_spoken"] += 1

    def receive_message(self, from_id: int, utterance: str):
        if not utterance:
            return
        toks = [t for t in utterance.split() if t.strip()]
        if not toks:
            return

        if hasattr(self.owner, "identity_system"):
            try:
                self.owner.identity_system.observe_nickname_use(from_id, utterance)
            except Exception:
                pass

        if hasattr(self.owner, "_observe_language_tokens"):
            try:
                self.owner._observe_language_tokens(toks, gain=0.05)
            except Exception:
                pass
        elif hasattr(self.owner, "semantic_system"):
            try:
                self.owner.semantic_system.observe_tokens(toks, gain=0.05)
            except Exception:
                pass

        tc = getattr(self.owner, "trust_channels", None)
        if isinstance(tc, dict):
            prof = tc.get(from_id)
            if isinstance(prof, dict):
                prof["affinity"] = self._clamp(float(prof.get("affinity", 0.0)) + 0.01, -2.0, 2.0)
            else:
                tc[from_id] = self._clamp(float(prof or 0.0) + 0.01, -1.0, 1.0)

        mot = getattr(self.owner, "motivation", None)
        if isinstance(mot, dict) and "social" in mot:
            mot["social"] = self._clamp(mot.get("social", 0.5) + 0.02, 0.0, 1.0)

        self.dialogue_stats["turns_seen"] += 1

        # Optional identity drift toward trusted partners
        prof = tc.get(from_id) if isinstance(tc, dict) else None
        trust_val = None
        if isinstance(prof, dict) and prof:
            vals = [v for v in prof.values() if isinstance(v, (int, float))]
            trust_val = sum(vals) / len(vals) if vals else None
        if trust_val is not None and hasattr(self.owner, "coordinator"):
            teacher = next(
                (a for a in getattr(self.owner.coordinator, "agents", []) if a.id == from_id),
                None,
            )
            if teacher is not None and hasattr(self.owner, "identity_system"):
                self.owner.identity_system.identity_social_drift(teacher, amount=0.02 * float(trust_val))

    def produce_addressed_utterance(self, listener) -> str:
        if hasattr(self.owner, "produce_utterance"):
            base = self.owner.produce_utterance() or ""
        else:
            base = ""

        if not listener or random.random() < 0.4:
            return base
        if not hasattr(self.owner, "identity_system"):
            return base
        try:
            nick = self.owner.identity_system.get_or_create_nickname_for_agent(listener)
        except Exception:
            return base
        if not nick:
            return base
        return f"{nick} {base}".strip()

    def observe_utterance(self, utterance):
        """Process an observed utterance pragmatically."""
        return None

    def choose_state(self):
        """Choose the pragmatic state for the next utterance."""
        return None

    def score_novelty(self, utterance):
        """Score the novelty of an utterance."""
        return None
