# agents/mixins/language_mixin.py
import random
import re
import math
from agents.agent_constants import SYLLABLES, PUNCT
from evolution.counting import CountingSystem
from agents.language import LanguageOrgan


class LanguageMixin:
    """
    Clean, controlled utterance generator with semantic vectors,
    dictionary weighting, identity awareness, and numeric reinforcement.
    """

    # =====================================================
    # INITIALISATION
    # =====================================================
    def _init_language_system(self):
        self.vocab = set(SYLLABLES)
        self.dict_vocab = set()

        self.utterance_memory = {
            "associations": {},
            "usage_count": {},
        }

        self.utter_bias = {
            "symbol_preferences": {s: random.uniform(-0.25, 0.25) for s in SYLLABLES},
            "length_bias": random.uniform(0.0, 1.0),
            "repeat_bias": random.uniform(0.0, 1.0),
        }

        self.symbol_drift = {s: 0.0 for s in SYLLABLES}

        self.counting = CountingSystem(owner=self)
        self.language = LanguageOrgan(
            owner=self,
            trust_threshold=self.traits.get("trust_threshold", 0.5),
        )

        if not hasattr(self, "recent_tokens"):
            self.recent_tokens = []

        # ensure semantic map exists
        if not hasattr(self, "semantic"):
            self.semantic = {"vecs": {}, "dictionary_gain": 0.10}

        # permanent numeric anchor vectors
        if not hasattr(self, "numeric_semantic"):
            self.numeric_semantic = {}

        # base = counting system base
        base = getattr(self.counting, "base", 16)

        for d in range(base):
            tok = self.counting.get_symbol(d)  # digit token ("sa", "ti", "vu", ...)
            if tok not in self.semantic["vecs"]:
                # Each digit gets a deterministic anchor vector on a line
                # e.g. embedding ~ d * constant direction
                anchor = [(d / (base - 1)) * 1.0] + [0.0] * 31
                self.semantic["vecs"][tok] = anchor

        # 🔹 NEW: hook in reasoning, if the mixin is present
        if hasattr(self, "_init_reasoning_system"):
            self._init_reasoning_system()
    # =====================================================
    # TOKEN CLEANING
    # =====================================================
    def _parse_utterance(self, utterance):
        if not utterance:
            return []
        toks = []
        for t in utterance.split():
            t = re.sub(r"[^\w\-']+$", "", t.lower()).strip()
            if t:
                toks.append(t)
        return toks

    # =====================================================
    # EMOTIONAL MODIFIERS
    # =====================================================
    def _emotion_mod_len(self):
        S = self.state
        base = 0
        base += (S["curiosity"] - 0.5) * 3
        base += (S["loneliness"] - 0.5) * 2
        base += (S["frustration"] - 0.5) * -3
        base += (S["happiness"] - 0.5) * 2
        return int(base)

    def _emotion_mod_punct(self):
        S = self.state
        mod = 0
        mod += (S["happiness"] - 0.5) * 0.25
        mod += (S["frustration"] - 0.5) * -0.20
        mod += (S["confidence"] - 0.5) * 0.15
        return mod

    def _emotion_mod_dict_bias(self):
        S = self.state
        b = 0.75
        b += (S["confidence"] - 0.5) * 0.20
        b += (S["purpose"]    - 0.5) * 0.05
        b += (S["frustration"] - 0.5) * -0.15
        return max(0.1, min(0.95, b))

    # =====================================================
    # MAIN UTTERANCE GENERATOR
    # =====================================================
    def produce_utterance(self):
        """
        Dictionary-weighted, identity-aware utterance generator.
        Ensures:
           - dictionary > syllables
           - numeric tokens correctly reinforced
           - NO random English ingestion
        """

        # Length & punctuation
        express = self.traits.get("expressiveness", 0.5)
        base_len = random.randint(2, 6)
        length   = int(max(2, min(10, base_len + self._emotion_mod_len())))

        punct_prob = 0.15 + 0.3 * express + self._emotion_mod_punct()
        punct_prob = max(0.01, min(0.9, punct_prob))

        # Vocabulary pools
        syll_pool = list(SYLLABLES)
        dict_pool = list(self.dict_vocab)

        identity_tokens = [
            f"A{other.id}"
            for other in getattr(self, "population", [])
            if other.id != self.id
        ]

        all_tokens = syll_pool + dict_pool + identity_tokens

        # Ensure semantic vectors
        for tok in all_tokens:
            if tok not in self.semantic["vecs"]:
                self.semantic["vecs"][tok] = self._randvec()

        prefs = self.utter_bias["symbol_preferences"]

        # 🔹 NEW: reasoning tokens (if enabled)
        reasoning_pool = []
        if hasattr(self, "reasoning_tokens"):
            reasoning_pool = list(self.reasoning_tokens.values())

        all_tokens = syll_pool + dict_pool + identity_tokens + reasoning_pool
        ...
        prefs = self.utter_bias["symbol_preferences"]

        # 🔹 NEW: keep reasoning tokens available but not dominant
        for r in reasoning_pool:
            prefs.setdefault(r, 0.05)

        # Weak syllables
        for s in syll_pool:
            prefs.setdefault(s, 0.15)

        # Strong dictionary
        for w in dict_pool:
            prefs.setdefault(w, 1.5)

        # identity
        for ident in identity_tokens:
            prefs.setdefault(ident, 0.10)

        # Recency boost
        for tok in self.recent_tokens[-25:]:
            prefs[tok] = min(3.0, prefs.get(tok, 0.2) + 0.15)

        # Expressiveness → more dictionary
        if random.random() < express:
            for w in dict_pool:
                prefs[w] = min(3.0, prefs.get(w, 1.5) + 0.15)

        # ---------------------------
        # SOFTMAX
        # ---------------------------
        logits = [prefs.get(tok, 0.2) for tok in all_tokens]

        m = max(logits)
        exps = [math.exp(l - m) for l in logits]
        total = sum(exps)
        probs = [e/total for e in exps]

        # ---------------------------
        # CHOOSE TOKENS
        # ---------------------------
        toks = []
        for _ in range(length):
            tok = random.choices(all_tokens, probs)[0]
            toks.append(tok)

            # recency
            self.recent_tokens.append(tok)
            prefs[tok] = min(3.0, prefs.get(tok, 0.2) + 0.05)

        # ---------------------------
        # Punctuation & cleaning
        # ---------------------------
        utter = " ".join(toks)
        if random.random() < punct_prob:
            utter += random.choice(PUNCT)

        clean = self._parse_utterance(utter)

        # Reinforcement
        if hasattr(self, "_observe_tokens"):
            self._observe_tokens(clean)

        # Multi-token numeric reinforcement
        if len(clean) > 1 and all(tok in self.symbol_map.values() for tok in clean):
            self._observe_tokens(clean, gain=0.2)

        self._last_tokens = clean
        self.last_written_word = utter

        # frequency
        mem = self.utterance_memory["usage_count"]
        mem[utter] = mem.get(utter, 0) + 1

        return utter

    # =====================================================
    # FEEDBACK
    # =====================================================
    def get_utterance_expectation(self, utterance):
        return float(self.utterance_memory["associations"].get(utterance, 0.0))

    def learn_from_feedback(self, utterance, reward, lr=0.1):
        if not utterance:
            return
        r = max(-1.0, min(+1.0, reward))
        mem = self.utterance_memory["associations"]

        old = mem.get(utterance, 0.0)
        mem[utterance] = max(-1.0, min(1.0, old + lr*r))

        # mild decay of all memories
        for k in list(mem.keys()):
            mem[k] *= 0.999

    # =====================================================
    # EXPECTATION
    # =====================================================
    def get_overall_expectation(self):
        expectation = float(getattr(self, "numeric_bias", 0.0))
        assoc = self.utterance_memory.get("associations", {})
        if assoc:
            expectation = 0.7*expectation + 0.3*(sum(assoc.values())/len(assoc))
        return max(-1.0, min(1.0, expectation))

        # =====================================================
    # Numeric → symbolic speech
    # =====================================================
    def speak_number(self, n):
        if not hasattr(self, "counting"):
            return str(n)

        try:
            digits = self.counting.interpret(n)
        except Exception:
            return str(n)

        # Ensure we have a symbol_map
        if not hasattr(self, "symbol_map") or self.symbol_map is None:
            self.symbol_map = {}

        tokens = []
        for d in digits:
            # If we already have a symbol for this digit, use it
            if d in self.symbol_map:
                tok = self.symbol_map[d]
            else:
                # Allocate a *real* symbol for this digit via the CountingSystem,
                # instead of a "dX" placeholder.
                try:
                    tok = self.counting.get_symbol(d)
                except Exception:
                    # ultra-fallback: still avoid "dX" so it’s decodable later
                    tok = f"num{d}"
                self.symbol_map[d] = tok

            tokens.append(tok)

        clean = [t.lower() for t in tokens]
        if hasattr(self, "_observe_tokens"):
            self._observe_tokens(clean)

        self._last_tokens = clean
        utter = " ".join(tokens)
        self.last_written_word = utter
        return utter

    def language_world_ingest_step(self):
        """
        Agents ingest only:
        - their own vocab
        - dictionary words
        - numeric tokens
        - identity tokens
        BUT we now also parse structured numeric-teaching lines like:

            A12 teach_numeric ha ti digits=[9,1] value=145 map={{ 9:ha, 1:ti }}

        without allowing arbitrary English tokens through.
        """

        if not hasattr(self, "api") or self.api is None:
            return

        texts = []

        t1 = self.api.read_text("/notes.txt")
        if t1:
            texts.append(t1)

        t2 = self.api.read_text("/help_responses.txt")
        if t2:
            texts.append(t2)

        if not texts:
            return

        for txt in texts:
            for line in txt.strip().splitlines()[-10:]:

                # ===============================================
                # NEW: detect numeric teaching lines BEFORE token filtering
                # ===============================================
                if "teach_numeric" in line:
                    try:
                        self._integrate_teach_numeric_line(line)
                    except Exception:
                        pass
                # ===============================================

                # standard ingest pathway
                toks = []
                for t in line.split():
                    t = t.lower().strip(",.!?;:\"'")
                    if not t:
                        continue

                    # only accept allowed-safe tokens
                    if (
                        t in self.dict_vocab or
                        t in self.vocab or
                        t in self.symbol_map.values() or
                        (t.startswith("a") and t[1:].isdigit())
                    ):
                        toks.append(t)

                # update vocab & semantic vectors
                for t in toks:
                    self.vocab.add(t)
                    if t not in self.semantic["vecs"]:
                        self.semantic["vecs"][t] = self._randvec()

                # semantic reinforcement
                if toks and hasattr(self, "_observe_tokens"):
                    self._observe_tokens(
                        toks, gain=self.semantic.get("dictionary_gain", 0.1)
                    )

    # =====================================================
    #  Numeric teaching integration (shared lexicon)
    # =====================================================
    def _integrate_teach_numeric_line(self, line):
        """
        Parse lines like:
          A12 teach_numeric ha ti digits=[9,1] value=145 map={{ 9:ha, 1:ti }}
        and use them to update this agent's symbol_map.

        This lets agents converge on a shared number lexicon.
        """

        if "teach_numeric" not in line:
            return

        try:
            # Rough parse:
            #  A12 teach_numeric <phrase> digits=[...]
            # First, split on "teach_numeric"
            head, rest = line.split("teach_numeric", 1)
            rest = rest.strip()

            # Now find "digits=[" and "value="
            # phrase is the part before "digits="
            digits_idx = rest.find("digits=")
            if digits_idx == -1:
                return

            phrase = rest[:digits_idx].strip()
            meta = rest[digits_idx:]

            # Extract digits list
            # digits=[9,1] value=145 ...
            import re
            m_digits = re.search(r"digits=\[([0-9,\s]+)\]", meta)
            if not m_digits:
                return

            digits_str = m_digits.group(1)
            digits = [int(x.strip()) for x in digits_str.split(",") if x.strip().isdigit()]

            # Optional: extract value (not strictly needed here)
            m_val = re.search(r"value=([0-9]+)", meta)
            value = int(m_val.group(1)) if m_val else None

            toks = phrase.split()
            if len(toks) != len(digits):
                # Mismatched digit/token counts; skip
                return

            # Ensure numeric machinery exists
            if not hasattr(self, "symbol_map"):
                self.symbol_map = {}
            if not hasattr(self, "counting"):
                from evolution.counting import CountingSystem
                self.counting = CountingSystem(owner=self)

            # Do we know base?
            base = getattr(self.counting, "base", 16)

            updated = False
            for d, tok in zip(digits, toks):
                # Only accept valid digits in current base
                if d < 0:
                    continue

                if d >= base:
                    # Expand base to allow this digit
                    new_base = d + 1
                    self.traits["numeric_base"] = new_base
                    self.counting.base = new_base
                    base = new_base

                # If this digit has no symbol yet, adopt it
                if d not in self.symbol_map:
                    self.symbol_map[d] = tok
                    updated = True
                else:
                    # If we already map this digit to the *same* token, fine
                    if self.symbol_map[d] == tok:
                        continue
                    # If there's a conflict, we'll be conservative and keep existing
                    # (you could add conflict resolution here later)
                    # If we have a conflict: update, but only if multiple teachers show the same mapping
                    if getattr(self, "_numeric_update_buffer", None) is None:
                        self._numeric_update_buffer = {}

                    key = (d, tok)
                    self._numeric_update_buffer[key] = self._numeric_update_buffer.get(key, 0) + 1

                    # Require two confirmations before overwriting old mapping
                    if self._numeric_update_buffer[key] >= 2:
                        self.symbol_map[d] = tok
                        updated = True

                # Integrate semantically
                if hasattr(self, "_ensure_vec"):
                    self._ensure_vec(tok)
                if hasattr(self, "_observe_tokens"):
                    self._observe_tokens([tok], gain=0.3)

            if updated:
                # Re-sync counting system with new symbol map
                try:
                    self.counting.set_symbol_map(self.symbol_map)
                except Exception:
                    pass

        except Exception:
            # Don’t let bad log lines crash the agent
            return