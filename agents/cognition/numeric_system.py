"""
agents/cognition/numeric_system.py
Numeric system for managing agent numeric representations.
"""
import random

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
        self.symbol_map = {}
        self.inverse_symbol_map = {}
        
        # Numeric semantic anchors
        self.numeric_semantic = {}
        for d in range(self.base):
            tok = self.get_symbol(d)
            self.numeric_semantic[d] = tok
            self.owner.vocab.add(tok)
            if hasattr(self.owner, "semantic_system"):
                self.owner.semantic_system.ensure_numeric_token(tok, digit=d, base=self.base)

    def _numeric_collision_score(self):
        mapping = self.numeric_semantic
        tokens = list(mapping.values())
        unique = set(tokens)
        return max(0, len(tokens) - len(unique))

    def _register_numeric_token(self, tok):
        if not tok:
            return
        if self.owner and hasattr(self.owner, "register_numeric_token"):
            self.owner.register_numeric_token(tok)

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
        max_val = max(self.symbol_map.keys())
        new_val = max_val + 1

        # Store it: dynamic expansion
        self.symbol_map[new_val] = tok
        self.inverse_symbol_map[tok] = new_val
        self._register_numeric_token(tok)
        return new_val

    def set_symbol_map(self, symbol_map):
        """
        Accept an int->token map from the Agent and build a reverse
        token->int map for decoding.
        """
        self.symbol_map = dict(symbol_map or {})
        self.inverse_symbol_map = {v: k for k, v in self.symbol_map.items()}

    def _init_seed_symbols(self):
        """Seed with minimal tokens up to base-1."""
        vowels = "aeiou"
        consonants = "bcdfghjklmnpqrstvwxyz"
        for n in range(self.base):
            token = random.choice(consonants) + random.choice(vowels)
            self.symbols[n] = token
            self.inverse[token] = n
            self._register_numeric_token(token)

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
                self._register_numeric_token(v)

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
        if not tok or not isinstance(tok, str):
            vowels = "aeiou"
            consonants = "bcdfghjklmnpqrstvwxyz"
            tok = random.choice(consonants) + random.choice(vowels)
            self.symbols[n] = tok
            self.inverse[tok] = n
            self._register_numeric_token(tok)
        
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
                self._register_numeric_token(v)
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

        return self
    
    def _maybe_learn_numeric_from(self, teacher):
        my_score = self._numeric_collision_score()
        their_score = teacher.numeric_system._numeric_collision_score()
        if their_score < my_score:
            for n, tok in teacher.numeric_system.numeric_semantic.items():
                self.numeric_semantic[n] = tok
                if hasattr(self.owner, "semantic_system"):
                    self.owner.semantic_system.ensure_numeric_token(tok, digit=n, base=self.base)
            if hasattr(self.owner, "trust_channels"):
                self.owner.trust_channels[teacher.id] = \
                    min(1.0, self.owner.trust_channels.get(teacher.id, 0.0) + 0.05)
            if hasattr(self.owner, "motivations"):
                self.owner.motivations["esteem"] = min(
                    1.0, self.owner.motivations.get("esteem", 0.5) + 0.03
                )
            if teacher.traits.get("teaching_drive", 0) > 0.4:
                teacher.motivations["esteem"] = min(
                    1.0, teacher.motivations.get("esteem", 0.5) + 0.02
                )

    def speak_number(self, n):
        try:
            digits = self.interpret(n)
        except Exception:
            return str(n)

        if not hasattr(self, "symbol_map") or self.symbol_map is None:
            self.symbol_map = {}
        else:
            self.symbol_map = {
                k: v for k, v in self.symbol_map.items()
                if isinstance(v, str) and v.strip()
            }

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
