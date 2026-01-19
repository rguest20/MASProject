"""
agents/cognition/identity_system.py
Identity system for managing agent identities and profiles.
"""

from collections import defaultdict
import random
class IdentitySystem:
    """
    Manages the identity and profiles of agents.
    """
    def __init__(self, owner):
        self.owner = owner
        self.identity_tokens = set()
        self.identity_token = f"agent_{owner.id}"

        self.owner.vocab.add(self.identity_token)
        self.owner.semantic_system._ensure_vec(self.identity_token)
        self.identity_tokens.add(self.identity_token)

        dim = self.owner.semantic_system._semantic_dim()
        base = self.owner.semantic_system._randvec(scale=0.6)
        jitter = [random.uniform(-0.05, 0.05) for _ in range(dim)]
        self.owner.semantic["vecs"][self.identity_token] = [
            b + j for b, j in zip(base, jitter)
        ]

        self.identity_vec = list(self.owner.semantic_system.vectors[self.identity_token])
        self.nicknames_for_others = {}
        self.heard_nicknames = defaultdict(set)

    # not used currently, but could be useful for generating consistent names
    def _invent_name_token(self, other_id: int) -> str:
        tok = f"agent_{other_id}"
        self.owner.vocab.add(tok)
        self.owner.semantic_system._ensure_vec(tok)
        if hasattr(self, "identity_tokens"):
            self.identity_tokens.add(tok)
        return tok
    
    def get_or_create_nickname_for_id(self, other_id: int) -> str:
        """
        Get or create a nickname token for another agent by their ID.
        """
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
