# ============================================================
# language.py — Phonemic and proto-word generation & learning
# ============================================================

import random

VOWELS = "aeiou"
CONSONANTS = "bcdfghjklmnpqrstvwxyz"
SYLLABLE_PATTERNS = ["CVC", "VC", "CV", "VVC", "CVV"]

# ---------------------------
# Basic word invention utils
# ---------------------------

def make_syllable():
    pattern = random.choice(SYLLABLE_PATTERNS)
    return "".join(
        random.choice(CONSONANTS if p == "C" else VOWELS)
        for p in pattern
    )

def mutate_syllable(syll):
    if random.random() < 0.3:
        i = random.randrange(len(syll))
        if random.random() < 0.5:
            new_char = random.choice(CONSONANTS + VOWELS)
            syll = syll[:i] + new_char + syll[i+1:]
        else:
            insert_char = random.choice(CONSONANTS + VOWELS)
            syll = syll[:i] + insert_char + syll[i:]
    return syll

def invent_word():
    num_syll = 1 if random.random() < 0.7 else 2
    word = "".join(make_syllable() for _ in range(num_syll))
    return mutate_syllable(word)

# ---------------------------
# LanguageOrgan class
# ---------------------------

class LanguageOrgan:
    """
    Handles invention, storage, and transmission of symbolic mappings.
    Each agent carries its own instance.
    """

    def __init__(self, owner=None, trust_threshold=0.5):
        self.owner = owner
        self.symbol_map = {}          # {concept_id -> word}
        self.trust_threshold = trust_threshold

    # --- core methods ---

    def speak(self, concept_id):
        """
        Retrieve or invent a word for a concept (e.g. a number).
        """
        if concept_id not in self.symbol_map:
            self.symbol_map[concept_id] = invent_word()
        return self.symbol_map[concept_id]

    def learn_from(self, peer):
        """
        Absorb some of a peer's known symbols, weighted by trust.
        """
        if not isinstance(peer, LanguageOrgan):
            return

        for k, v in peer.symbol_map.items():
            if k not in self.symbol_map and random.random() < self.trust_threshold:
                self.symbol_map[k] = v

    def merge_from_parents(self, parent1, parent2):
        """
        For inheritance: blend symbol maps from both parents.
        """
        if parent1:
            self.symbol_map.update({
                k: v for k, v in parent1.symbol_map.items()
                if random.random() < 0.5
            })
        if parent2:
            self.symbol_map.update({
                k: v for k, v in parent2.symbol_map.items()
                if random.random() < 0.5
            })

    # --- optional diagnostics ---

    def summary(self, n=5):
        """
        Return a sample of current symbol mappings.
        """
        items = list(self.symbol_map.items())[:n]
        return ", ".join(f"{k}->{v}" for k, v in items) if items else "(no symbols)"