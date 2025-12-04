# agents/mixins/reasoning_mixin.py
import math
import random


class ReasoningMixin:
    """
    Semantic reasoning module for comparisons.

    Patch 3 changes:
      - Removes any fallbacks that use "_after_".
      - Guarantees meaningful operator strings ("greater_than", "less_than", "equal_to")
      - Strengthens operator learning and justification building.
      - Ensures justification always returns a clean string.
    """

    # ============================================================
    # INITIALISATION
    # ============================================================
    def _init_reasoning_system(self):
        # Ensure semantic structure exists
        if not hasattr(self, "semantic") or not isinstance(getattr(self, "semantic"), dict):
            self.semantic = {"vecs": {}, "dictionary_gain": 0.10}
        if "vecs" not in self.semantic:
            self.semantic["vecs"] = {}

        # Reasoning operator tokens (symbolic syllables)
        self.reasoning_tokens = {
            "gt":   "tor",    # greater-than
            "lt":   "lem",    # less-than
            "eq":   "zu",     # equal-to
            "next": "nex",    # successor (future)
            "prev": "pre",    # predecessor (future)
        }

        # Anchor semantic operator names: stable English tokens
        for op in ("greater_than", "less_than", "equal_to"):
            self._ensure_vec(op)

        # Legacy dictionary (optional to keep)
        self.reason_ops = {
            ">": self.semantic["vecs"]["greater_than"],
            "<": self.semantic["vecs"]["less_than"],
            "=": self.semantic["vecs"]["equal_to"],
        }

        # Ensure vocab exists
        if not hasattr(self, "vocab"):
            self.vocab = set()

        # Create vectors for reasoning tokens and add them to vocab
        for tok in self.reasoning_tokens.values():
            self.vocab.add(tok)
            self._r_ensure_vec(tok)

    # ============================================================
    # INTERNAL VECTOR HELPERS
    # ============================================================
    def _r_ensure_vec(self, tok):
        """Ensure a semantic vector exists for tok."""
        vecs = self.semantic.setdefault("vecs", {})
        if tok in vecs:
            return

        if vecs:
            dim = len(next(iter(vecs.values())))
        else:
            dim = 32

        if hasattr(self, "_randvec"):
            vecs[tok] = self._randvec()
        else:
            vecs[tok] = [random.uniform(-0.5, 0.5) for _ in range(dim)]

    def _r_get_vec(self, tok):
        return self.semantic["vecs"].get(tok)

    def _r_mean_vec(self, toks):
        vecs = [self._r_get_vec(t) for t in toks if self._r_get_vec(t) is not None]
        if not vecs:
            return None
        dim = len(vecs[0])
        out = [0.0] * dim
        for v in vecs:
            for i in range(dim):
                out[i] += v[i]
        n = float(len(vecs))
        return [x / n for x in out]

    def _r_sub(self, a, b):
        return [x - y for x, y in zip(a, b)]

    def _r_add_scaled(self, base, direction, gain):
        return [b * (1.0 - gain) + direction[i] * gain for i, b in enumerate(base)]

    def _r_norm(self, v):
        return math.sqrt(sum(x * x for x in v))

    def _r_normalize(self, v):
        n = self._r_norm(v)
        if n < 1e-8:
            return None
        return [x / n for x in v]

    def _r_dot(self, a, b):
        return sum(x * y for x, y in zip(a, b))

    # ============================================================
    # OPERATOR LEARNING
    # ============================================================
    def update_comparison_operator(self, phrase_A, phrase_B, relation):
        """
        Learns directional meaning of >, <, = based on comparisons.
        """

        if relation not in ("gt", "lt", "eq"):
            return
        if not hasattr(self, "reasoning_tokens"):
            return

        A_vec = self._r_mean_vec(phrase_A)
        B_vec = self._r_mean_vec(phrase_B)

        # mix in numeric anchor structure
        num_vecs = [self.numeric_semantic[t] for t in phrase_A if t in self.numeric_semantic]
        if num_vecs:
            # weight numeric structure strongly
            A_vec = [
                0.7*A_vec[i] + 0.3*(sum(v[i] for v in num_vecs)/len(num_vecs))
                for i in range(len(A_vec))
            ]
        num_vecs = [self.numeric_semantic[t] for t in phrase_B if t in self.numeric_semantic]
        if num_vecs:
            B_vec = [
                0.7*B_vec[i] + 0.3*(sum(v[i] for v in num_vecs)/len(num_vecs))
                for i in range(len(B_vec))
            ]
        
        if A_vec is None or B_vec is None:
            return

        delta = self._r_sub(A_vec, B_vec)
        delta = self._r_normalize(delta)
        if delta is None:
            return

        # Ensure operator vectors exist
        for key in ("gt", "lt", "eq"):
            self._r_ensure_vec(self.reasoning_tokens[key])

        vecs = self.semantic["vecs"]

        if relation in ("gt", "lt"):
            sign_gt = +1.0 if relation == "gt" else -1.0
            sign_lt = -sign_gt

            gt_tok = self.reasoning_tokens["gt"]
            lt_tok = self.reasoning_tokens["lt"]

            vecs[gt_tok] = self._r_add_scaled(vecs[gt_tok],
                                               [sign_gt * x for x in delta],
                                               gain=0.25)

            vecs[lt_tok] = self._r_add_scaled(vecs[lt_tok],
                                               [sign_lt * x for x in delta],
                                               gain=0.25)

            eq_tok = self.reasoning_tokens["eq"]
            vecs[eq_tok] = [x * 0.96 for x in vecs[eq_tok]]

        else:  # eq
            for key in ("gt", "lt", "eq"):
                tok = self.reasoning_tokens[key]
                vecs[tok] = [x * 0.995 for x in vecs[tok]]

    # # ============================================================
    # # JUSTIFICATION BUILDER
    # # ============================================================
    # def build_numeric_justification(self, phrase_A, phrase_B, relation=None):
    #     """
    #     Creates a justification in learned operator terms.
    #     Does NOT fall back to any '_after_' patterns.
    #     """

    #     if not hasattr(self, "reasoning_tokens"):
    #         # Clean fallback only:
    #         if relation == "gt":
    #             return f"{' '.join(phrase_A)} greater_than {' '.join(phrase_B)}"
    #         elif relation == "lt":
    #             return f"{' '.join(phrase_A)} less_than {' '.join(phrase_B)}"
    #         else:
    #             return f"{' '.join(phrase_A)} equal_to {' '.join(phrase_B)}"

    #     # One more learning update
    #     if relation in ("gt", "lt", "eq"):
    #         self.update_comparison_operator(phrase_A, phrase_B, relation)

    #     A_vec = self._r_mean_vec(phrase_A)
    #     B_vec = self._r_mean_vec(phrase_B)
    #     if A_vec is None or B_vec is None:
    #         return f"{' '.join(phrase_A)} equal_to {' '.join(phrase_B)}"

    #     delta = self._r_sub(A_vec, B_vec)
    #     delta = self._r_normalize(delta)
    #     if delta is None:
    #         return f"{' '.join(phrase_A)} equal_to {' '.join(phrase_B)}"

    #     # Score operators
    #     scores = {}
    #     for key in ("gt", "lt", "eq"):
    #         tok = self.reasoning_tokens[key]
    #         self._ensure_token_semantic(tok)
    #         v = self.semantic["vecs"].get(tok)
    #         if v is None:
    #             continue
    #         scores[key] = self._r_dot(v, delta)

    #     if not scores:
    #         return f"{' '.join(phrase_A)} equal_to {' '.join(phrase_B)}"

    #     best_key = max(scores, key=scores.get)
    #     op_tok = self.reasoning_tokens[best_key]

    #     justification_tokens = list(phrase_A) + [op_tok] + list(phrase_B)

    #     # Co-occurrence learning
    #     if hasattr(self, "_observe_tokens"):
    #         self._observe_tokens(justification_tokens, gain=0.20)

    #     return " ".join(justification_tokens)

    # ============================================================
    # OPTIONAL: adjacency learning
    # ============================================================
    def update_adjacency_from_tokens(self, smaller_tok, bigger_tok):
        """
        Future successor/predecessor reasoning.
        """
        if not hasattr(self, "reasoning_tokens"):
            return

        v_small = self._r_get_vec(smaller_tok)
        v_big = self._r_get_vec(bigger_tok)
        if v_small is None or v_big is None:
            return

        delta = self._r_sub(v_big, v_small)
        delta = self._r_normalize(delta)
        if delta is None:
            return

        self._r_ensure_vec(self.reasoning_tokens["next"])
        self._r_ensure_vec(self.reasoning_tokens["prev"])

        vecs = self.semantic["vecs"]
        nex_tok = self.reasoning_tokens["next"]
        pre_tok = self.reasoning_tokens["prev"]

        vecs[nex_tok] = self._r_add_scaled(vecs[nex_tok], delta, gain=0.12)
        vecs[pre_tok] = self._r_add_scaled(vecs[pre_tok], [-x for x in delta], gain=0.12)