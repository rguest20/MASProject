"""
agents/cognition/numeric_system.py
Numeric system for managing agent numeric representations.
"""
import random
from typing import Optional

POSSIBLE_BASES = [4, 6, 8, 10, 12, 16]
class NumericSystem:
    """
    Manages the numeric representations of an agent.
    """
    def __init__(self, owner):
        self.owner = owner
        self.base = random.choice(POSSIBLE_BASES)
        self.symbols = {}
        self.inverse = {}
        self._init_seed_symbols()
        # The agent owns the map used by task and communication code.  The
        # numeric system keeps a reference to that same object rather than a
        # divergent private copy.
        if not isinstance(getattr(owner, "symbol_map", None), dict):
            owner.symbol_map = {}
        self.symbol_map = owner.symbol_map
        self.inverse_symbol_map = {}
        
        # Numeric semantic mapping: digit -> token (keep in sync with symbols).
        self.numeric_semantic = self.symbols
        for d in range(self.base):
            tok = self.get_symbol(d)
            self.numeric_semantic[d] = tok
            self._register_numeric_token(tok, digit=d)

    def _numeric_collision_score(self):
        mapping = self.numeric_semantic
        tokens = list(mapping.values())
        unique = set(tokens)
        return max(0, len(tokens) - len(unique))

    def _fresh_symbol(self, reserved_tokens=None):
        """Create a token not already used by another digit in this system."""
        reserved = {str(token).strip().lower() for token in (reserved_tokens or []) if token}
        reserved.update(
            str(token).strip().lower()
            for token in getattr(self, "symbols", {}).values()
            if token
        )
        reserved.update(
            str(token).strip().lower()
            for token in getattr(self, "symbol_map", {}).values()
            if token
        )

        vowels = "aeiou"
        consonants = "bcdfghjklmnpqrstvwxyz"
        for _ in range(256):
            token = random.choice(consonants) + random.choice(vowels)
            if token not in reserved:
                return token

        # The two-syllable fallback makes exhaustion impossible in practice.
        while True:
            token = (
                random.choice(consonants) + random.choice(vowels) +
                random.choice(consonants) + random.choice(vowels)
            )
            if token not in reserved:
                return token

    def _register_numeric_token(self, tok: str, digit: Optional[int] = None):
        if not isinstance(tok, str) or not tok.strip():
            return
        tok = tok.strip().lower()

        o = self.owner
        if hasattr(o, "vocab") and isinstance(getattr(o, "vocab", None), set):
            o.vocab.add(tok)

        reg = getattr(o, "token_registry", None)
        if reg is not None and hasattr(reg, "register_numeric"):
            try:
                reg.register_numeric(tok)
            except Exception:
                pass

        if hasattr(o, "semantic_system"):
            try:
                o.semantic_system.ensure_numeric_token(tok, digit=digit, base=self.base)
            except Exception:
                pass

    def decode_token(self, tok):
        """
        Extended decoding:
        1. Try direct match.
        2. Try multi-token composite.
        3. Try neighbourhood inference.
        4. Try creative extrapolation (number invention).
        """
        # Cleanup formatting: allow 'qa hu', 'qa-hu', 'qa,hu'
        if isinstance(tok, str) and any(sep in tok for sep in [" ", "-", ","]):
            parts = self._split_composite(tok)
            if parts:
                return self._decode_composite(parts)

        # Direct known token
        inv = self.inverse_symbol_map
        if tok in inv:
            return inv[tok]

        # 3. Neighbourhood inference
        guess = self._infer_from_neighbours(tok)
        if guess is not None:
            return guess

        # 4. Extrapolation / invention
        newval = self._invent_number(tok)
        return newval

    def _split_composite(self, tok):
        for sep in [" ", "-", ","]:
            if sep in tok:
                parts = tok.split(sep)
                return [p.strip() for p in parts if p.strip()]
        return None

    def _decode_composite(self, parts):
        """
        Base-N positional interpretation.
        Unknown pieces go through normal decoder.
        """
        vals = []
        for p in parts:
            try:
                v = self.decode_token(p)
                vals.append(v)
            except Exception:
                return None

        base = self.base
        # [[1, 0]] → 1*base + 0
        result = 0
        for v in vals:
            result = result * base + v
        return result

    def _infer_from_neighbours(self, tok):
        """
        Look for tokens similar in phonetic/symbolic space:
        same prefix, suffix, onset, etc.
        """
        inv = self.inverse_symbol_map

        sims = []
        for known_tok, val in inv.items():
            score = 0
            if tok[0] == known_tok[0]:   # similar onset
                score += 1
            if tok[-1] == known_tok[-1]: # similar coda
                score += 1
            # Levenshtein-lite
            score -= abs(len(tok) - len(known_tok))

            if score > 0:
                sims.append((score, val))

        if not sims:
            return None

        # Weighted average guess
        sims.sort(reverse=True)
        top = sims[:3]
        num = sum(v for _, v in top)
        return round(num / len(top))

    def _invent_number(self, tok):
        """
        Create a new number for an unseen token.
        """
        max_val = max(self.symbol_map.keys(), default=-1)
        new_val = max_val + 1

        # Store it: dynamic expansion
        self.symbol_map[new_val] = tok
        self.inverse_symbol_map[tok] = new_val
        self._register_numeric_token(tok, digit=(new_val % max(2, self.base)))
        return new_val

    def set_symbol_map(self, symbol_map):
        """
        Accept an int->token map from the Agent and build a reverse
        token->int map for decoding.  A digit system must be injective: one
        token cannot name two distinct digits.
        """
        clean_map = {}
        used_tokens = set()
        for digit, token in (symbol_map or {}).items():
            if not isinstance(digit, int) or digit < 0:
                continue
            token = str(token).strip().lower() if token is not None else ""
            if not token or token in used_tokens:
                token = self._fresh_symbol(used_tokens)
            clean_map[digit] = token
            used_tokens.add(token)

        self.symbol_map = clean_map
        self.owner.symbol_map = self.symbol_map
        self.inverse_symbol_map = {v: k for k, v in self.symbol_map.items()}
        for digit, token in self.symbol_map.items():
            self.symbols[digit] = token
        self.inverse = {token: digit for digit, token in self.symbols.items()}
        self.numeric_semantic = self.symbols

    def learn_digit_mapping(self, token: str, digit: int):
        """Ground a peer's one-token numeral in this agent's number system."""
        if not isinstance(token, str) or not token.strip():
            return
        if not isinstance(digit, int) or not (0 <= digit < self.base):
            return

        token = token.strip().lower()
        mapping = dict(self.symbol_map)
        old_token = mapping.get(digit)
        old_digit = self.inverse_symbol_map.get(token)

        # Swapping retains the previous token for the displaced digit, rather
        # than leaving two digits with the newly learned peer token.
        if old_digit is not None and old_digit != digit:
            mapping[old_digit] = old_token or self._fresh_symbol(mapping.values())

        mapping[digit] = token
        self.set_symbol_map(mapping)
        self._register_numeric_token(token, digit=digit)

    def _init_seed_symbols(self):
        """Seed with minimal tokens up to base-1."""
        for n in range(self.base):
            token = self._fresh_symbol(self.symbols.values())
            self.symbols[n] = token
            self.inverse[token] = n
            self._register_numeric_token(token, digit=n)

    def interpret(self, raw_number):
        """
        Interpret an external integer (e.g., from challenge) using current base.
        Returns a list of component digits in this base.
        """
        if raw_number == 0:
            return [0]
        digits = []
        n = raw_number
        while n > 0:
            digits.append(n % self.base)
            n //= self.base
        digits.reverse()
        return digits

    def express(self, raw_number):
        """
        Convert number → string form using internal symbols.
        """
        digits = self.interpret(raw_number)
        parts = [self.symbols.get(d, "?") for d in digits]
        return " ".join(parts)

    def learn_from(self, peer):
        """
        Merge or drift toward peer’s base and symbols with some probability.
        """
        if random.random() < 0.2:
            # occasional base drift toward peer’s base
            self.base = peer.base

        for k, v in peer.symbols.items():
            if k not in self.symbols and random.random() < 0.3:
                self.symbols[k] = v
                self.inverse[v] = k
                self._register_numeric_token(v, digit=k)

    def mutate(self):
        """
        Randomly perturb base or rename one symbol.
        """
        if random.random() < 0.1:
            self.base = random.choice(POSSIBLE_BASES)
        if random.random() < 0.3 and self.symbols:
            k = random.choice(list(self.symbols.keys()))
            old = self.symbols[k]
            new = f"{old}{random.choice('aeiou')}"
            self.symbols[k] = new
            self.inverse[new] = k

    def mutate_base(self):
        """Compatibility shim: older code calls `counting.mutate_base()`."""
        self.base = random.choice(POSSIBLE_BASES)

    def get_symbol(self, n: int) -> str:
        """
        Return the symbolic token for a given integer n in this counting system.
        If n is outside the current base, it will wrap around or extend lazily.
        """
        tok = self.symbols.get(n)

        # If tok is missing OR is None OR empty → regenerate safely
        other_tokens = {
            value for digit, value in self.symbols.items()
            if digit != n and isinstance(value, str)
        }
        if not tok or not isinstance(tok, str) or tok in other_tokens:
            tok = self._fresh_symbol(other_tokens)
            self.symbols[n] = tok
            self.inverse[tok] = n
            self._register_numeric_token(tok, digit=n)
        
        return tok


    def merge_from(self, *others):
        """
        Blend this counting system with one or more others.
        Keeps the same base unless multiple distinct bases dominate,
        in which case adopts the mode (most common) base.
        Also merges symbolic mappings with light drift.
        """
        if not others:
            return self

        # 1. Base blending — take the most common base among systems
        bases = [o.base for o in others if hasattr(o, "base")]
        if bases:
            self.base = max(set(bases), key=bases.count)

        # 2. Merge symbols — probabilistic, allows partial borrowing
        for o in others:
            if not hasattr(o, "symbols"):
                continue
            for k, v in o.symbols.items():
                self._register_numeric_token(v, digit=k)
                if k not in self.symbols:
                    # occasionally borrow symbol directly
                    if random.random() < 0.5:
                        self.symbols[k] = v
                        self.inverse[v] = k
                else:
                    # occasionally drift toward blended token
                    if random.random() < 0.2:
                        old = self.symbols[k]
                        # tiny drift: append one vowel from peer token
                        if v and random.random() < 0.5:
                            self.symbols[k] = old + v[-1]
                            self.inverse[self.symbols[k]] = k

        # 3. Handle base extension if merged systems have more digits
        max_digit = max(self.symbols.keys()) if self.symbols else 0
        for n in range(max_digit + 1, self.base):
            if n not in self.symbols:
                vowels = "aeiou"
                consonants = "bcdfghjklmnpqrstvwxyz"
                token = random.choice(consonants) + random.choice(vowels)
                self.symbols[n] = token
                self.inverse[token] = n
                self._register_numeric_token(token, digit=n)

        return self
    
    def _maybe_learn_numeric_from(self, teacher):
        my_score = self._numeric_collision_score()
        their_score = teacher.numeric_system._numeric_collision_score()
        if their_score < my_score:
            for n, tok in teacher.numeric_system.numeric_semantic.items():
                self.numeric_semantic[n] = tok
                if hasattr(self.owner, "semantic_system"):
                    self.owner.semantic_system.ensure_numeric_token(tok, digit=n, base=self.base)

            if hasattr(self.owner, "adjust_trust"):
                self.owner.adjust_trust(teacher.id, amount=0.02, channel=4)  # competence
            elif hasattr(self.owner, "update_trust_channels"):
                self.owner.update_trust_channels(teacher.id, reward=0.2)

            motivations = getattr(self.owner, "motivations", None)
            if not isinstance(motivations, dict):
                motivations = getattr(self.owner, "needs", None)
                if isinstance(motivations, dict):
                    self.owner.motivations = motivations
                else:
                    self.owner.motivations = {}
                motivations = self.owner.motivations
            motivations["esteem"] = min(1.0, motivations.get("esteem", 0.5) + 0.03)

            if teacher.traits.get("teaching_drive", 0) > 0.4:
                teacher_motivations = getattr(teacher, "motivations", None)
                if not isinstance(teacher_motivations, dict):
                    teacher_motivations = getattr(teacher, "needs", None)
                    if isinstance(teacher_motivations, dict):
                        teacher.motivations = teacher_motivations
                    else:
                        teacher.motivations = {}
                    teacher_motivations = teacher.motivations
                teacher_motivations["esteem"] = min(1.0, teacher_motivations.get("esteem", 0.5) + 0.02)

    def speak_number(self, n):
        try:
            digits = self.interpret(n)
        except Exception:
            return str(n)

        owner_map = getattr(self.owner, "symbol_map", None)
        if isinstance(owner_map, dict) and owner_map is not self.symbol_map:
            self.symbol_map = owner_map

        self.set_symbol_map(self.symbol_map or {})

        tokens = []
        for d in digits:
            if d in self.symbol_map:
                tok = self.symbol_map[d]
            else:
                try:
                    tok = self.get_symbol(d)
                except Exception:
                    tok = f"num{d}"
                self.symbol_map[d] = tok
                self.inverse_symbol_map[tok] = d

            if tok is None:
                continue
            tok = str(tok).strip()
            if not tok:
                continue

            self.owner.vocab.add(tok)
            self.owner.semantic_system._ensure_vec(tok)
            self.owner._ensure_token_semantic(tok)
            self.owner._semantic_tick_token(tok)
            if hasattr(self.owner, "semantic_system"):
                self.owner.semantic_system.ensure_numeric_token(tok, digit=d, base=self.base)
            tokens.append(tok)

        if not tokens:
            tokens = ["na"]

        clean = [t.lower() for t in tokens]
        self.owner._observe_language_tokens(clean)

        utter = " ".join(tokens)
        self.last_written_word = utter
        self._last_tokens = clean
        self.owner.last_written_word = utter
        self.owner._last_tokens = clean
        return utter
