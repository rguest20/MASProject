"""
agents/cognition/identity_system.py
Identity system for managing agent identities and profiles.
"""

from collections import defaultdict
import random


class IdentitySystem:
    """Manages the identity and profiles of agents."""

    def __init__(self, owner):
        self.owner = owner
        self.identity_tokens = set()
        # Back-compat: older code expects `agent.identity_tokens` to exist.
        self.owner.identity_tokens = self.identity_tokens

        # primary self-identity token (stable across lifetime)
        self.identity_token = f"agent_{owner.id}"
        self.owner.identity_token = self.identity_token
        self.owner.vocab.add(self.identity_token)
        self.owner.semantic_system._ensure_vec(self.identity_token)
        self.identity_tokens.add(self.identity_token)

        # initialise identity vector near a stable root with small jitter
        dim = self.owner.semantic_system._semantic_dim()
        base = self._identity_root_vec(dim)
        jitter = [random.uniform(-0.05, 0.05) for _ in range(dim)]
        vec = [b + j for b, j in zip(base, jitter)]
        self.owner.semantic_system.vectors[self.identity_token] = vec

        self.identity_vec = list(vec)
        self.owner.identity_vec = list(vec)
        self.nicknames_for_others = {}
        self.heard_nicknames = defaultdict(set)
        # legacy access
        self.owner.nicknames_for_others = self.nicknames_for_others
        self.owner.heard_nicknames = self.heard_nicknames

    def is_identity_like(self, tok: str) -> bool:
        if not isinstance(tok, str) or not tok:
            return False
        if tok in self.identity_tokens:
            return True
        if tok.startswith("agent_") and tok[6:].isdigit():
            return True
        if (tok[0] in ("a", "A")) and tok[1:].isdigit():
            return True
        if tok.startswith("id") and tok[2:].isdigit():
            return True
        return False

    # Back-compat: SemanticSystem historically called this private name.
    def _is_identity_like(self, tok: str) -> bool:
        return self.is_identity_like(tok)

    # ------------------------------------------------------------------
    # Core helpers
    # ------------------------------------------------------------------
    def _identity_root_vec(self, dim: int):
        base = [
            4.0,
            -3.0,
            2.5,
            -4.2,
            3.1,
            1.7,
            -2.3,
            2.8,
            -3.7,
            1.9,
            -1.4,
            0.6,
            2.2,
            -0.8,
            1.5,
            -2.9,
        ]
        if dim <= len(base):
            return base[:dim]
        return base + [0.0] * (dim - len(base))

    def is_identity_token(self, tok: str) -> bool:
        return tok in self.identity_tokens

    def mark_identity_token(self, tok: str):
        self.owner.semantic_system._ensure_vec(tok)
        dim = self.owner.semantic_system._semantic_dim()
        root = self._identity_root_vec(dim)
        jitter = [random.uniform(-0.05, 0.05) for _ in range(dim)]
        vec = [b + j for b, j in zip(root, jitter)]
        self.owner.semantic_system.vectors[tok] = vec
        self.identity_tokens.add(tok)

    # ------------------------------------------------------------------
    # Social identity semantics
    # ------------------------------------------------------------------
    def seed_agent_identity(self, agent_id: int):
        token = f"A{agent_id}"
        self.owner.semantic_system._ensure_vec(token)

        vecs = self.owner.semantic_system.vectors
        if token not in vecs:
            dim = self.owner.semantic_system._semantic_dim()
            vecs[token] = self.owner.semantic_system._randvec(scale=1.0)

        trust = getattr(self.owner, "trust_channels", {}).get(agent_id, {})
        if trust:
            base = vecs[token]
            for _, v in trust.items():
                factor = 0.01 * v
                base = [x + factor for x in base]
            vecs[token] = base

        if hasattr(self.owner, "dict_vocab"):
            self.owner.dict_vocab.add(token)

    def reinforce_identity(self, partner_id: int, val: float):
        tok = f"a{partner_id}"
        self.owner.semantic_system._ensure_vec(tok)
        v = self.owner.semantic_system.vectors[tok]
        nudge = self.owner.semantic_system._randvec(scale=1.0)
        updated = [a + 0.15 * val * b for a, b in zip(v, nudge)]
        self.owner.semantic_system.vectors[tok] = updated

        if hasattr(self.owner, "name_token"):
            me = self.owner.name_token
            self.owner.semantic_system._ensure_vec(me)
            if hasattr(self.owner, "_link"):
                self.owner._link(me, tok, 0.05 * val)

    # not used currently, but could be useful for generating consistent names
    def _invent_name_token(self, other_id: int) -> str:
        tok = f"agent_{other_id}"
        self.owner.vocab.add(tok)
        self.owner.semantic_system._ensure_vec(tok)
        self.identity_tokens.add(tok)
        return tok

    def get_or_create_nickname_for_id(self, other_id: int) -> str:
        """Get or create a nickname token for another agent by their ID."""
        tok = f"agent_{other_id}"
        self.nicknames_for_others[other_id] = tok
        self.owner.vocab.add(tok)
        self.owner.semantic_system._ensure_vec(tok)

        id_tok = f"agent_{other_id}"
        base_vec = self.owner.semantic_system.vectors.get(id_tok)
        if base_vec is not None:
            v = self.owner.semantic_system.vectors[tok]
            self.owner.semantic_system.vectors[tok] = [
                a + 0.3 * (b - a) for a, b in zip(v, base_vec)
            ]
        return tok

    def get_or_create_nickname_for_agent(self, other):
        other_id = other if isinstance(other, int) else getattr(other, "id", None)
        if other_id is None:
            return None
        return self.get_or_create_nickname_for_id(int(other_id))

    def observe_nickname_use(self, from_id: int, utterance: str):
        if not utterance:
            return
        tokens = utterance.split()
        if not tokens:
            return
        first = tokens[0]
        if first.startswith("agent_") and first[6:].isdigit():
            self.heard_nicknames[from_id].add(first)

    def identity_social_drift(self, other, amount: float = 0.01):
        if other is None:
            return
        me = self.identity_token
        other_tok = getattr(other, "identity_token", None)
        if not isinstance(other_tok, str) and hasattr(other, "identity_system"):
            other_tok = getattr(other.identity_system, "identity_token", None)
        if not isinstance(other_tok, str):
            return

        vecs = self.owner.semantic_system.vectors
        a = vecs.get(me)
        b = vecs.get(other_tok)
        if a is None or b is None:
            return

        amount = float(amount)
        new = [x + amount * (y - x) for x, y in zip(a, b)]
        vecs[me] = new
        self.identity_vec = list(new)
        self.owner.identity_vec = list(new)
