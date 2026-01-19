"""
agents/cognition/language_system.py
Language system for linguistic utilities (tokens, parsing, rendering, bias).

This system intentionally does NOT manage meaning:
- No vector creation / semantic storage writes
- No co-occurrence learning
- No semantic-walk neighbor logic

Semantic concerns live in `SemanticSystem` and are orchestrated by
`LanguageMixin` as a bridge between systems.
"""

from __future__ import annotations

import math
import random
import re
from typing import Optional

from agents.agent_constants import SYLLABLES, PUNCT


class LanguageSystem:
    """Linguistic utilities: invention, parsing, token sampling, rendering."""

    def __init__(self, owner):
        self.owner = owner

    # ------------------------------------------------------------------
    # Bootstrap
    # ------------------------------------------------------------------
    def init(self, seed_tokens: int = 100):
        o = self.owner

        # Biases / drift
        if not hasattr(o, "utter_bias") or not isinstance(o.utter_bias, dict):
            o.utter_bias = {
                "symbol_preferences": {},
                "length_bias": random.uniform(0.0, 1.0),
                "repeat_bias": random.uniform(0.0, 1.0),
            }
        prefs = o.utter_bias.setdefault("symbol_preferences", {})
        for s in SYLLABLES:
            prefs.setdefault(s, random.uniform(-0.25, 0.25))

        if not hasattr(o, "symbol_drift") or not isinstance(o.symbol_drift, dict):
            o.symbol_drift = {s: 0.0 for s in SYLLABLES}
        if not hasattr(o, "_next_token_id"):
            o._next_token_id = 0

        if not hasattr(o, "vocab") or not isinstance(o.vocab, set):
            o.vocab = set(SYLLABLES)

        if not hasattr(o, "recent_tokens"):
            o.recent_tokens = []

        # Seed invented tokens for diversity (no semantic side effects here)
        for _ in range(max(0, int(seed_tokens))):
            tok = self.invent_token(max_syllables=3)
            o.vocab.add(tok)

    # ------------------------------------------------------------------
    # Token invention
    # ------------------------------------------------------------------
    def invent_token(
        self,
        *,
        prefix: str | None = None,
        max_syllables: int = 3,
        concept: bool = False,
    ) -> str:
        o = self.owner
        prefix = (prefix or "").strip()
        max_syllables = max(1, int(max_syllables))

        for _ in range(64):
            n = random.randint(1, max_syllables)
            core = "".join(random.choice(SYLLABLES) for _ in range(n))
            tok = f"{prefix}{core}" if prefix else core
            tok = tok.strip().lower()

            if not tok or tok in SYLLABLES:
                continue
            if hasattr(o, "vocab") and tok in o.vocab:
                continue
            break
        else:
            tok = f"{prefix}{random.choice(SYLLABLES)}{random.randint(0, 9999)}".lower()

        return tok

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------
    def parse_utterance(self, utterance: str) -> list[str]:
        if not utterance:
            return []
        toks: list[str] = []
        for t in utterance.split():
            t = re.sub(r"[^\w\-']+$", "", t.lower()).strip()
            if t:
                toks.append(t)
        return toks

    # ------------------------------------------------------------------
    # Utterance helpers
    # ------------------------------------------------------------------
    def choose_utter_start(self, all_tokens: list[str]) -> str:
        o = self.owner

        if hasattr(o, "_last_tokens") and hasattr(o, "symbol_map"):
            numeric_vals = set(getattr(o, "symbol_map", {}).values())
            num_cands = [t for t in (getattr(o, "_last_tokens", []) or []) if t in numeric_vals]
            if num_cands and random.random() < 0.4:
                return random.choice(num_cands)

        recent_tokens = getattr(o, "recent_tokens", None)
        if recent_tokens:
            recent = [t for t in recent_tokens[-20:] if t in all_tokens]
            if recent and random.random() < 0.7:
                return random.choice(recent)

        if all_tokens:
            return random.choice(all_tokens)
        return "na"

    def sample_token_from_pool(self, pool_tokens: list[str], prefs: dict[str, float]) -> Optional[str]:
        o = self.owner
        pool_tokens = [t for t in pool_tokens if isinstance(t, str) and t.strip()]
        if not pool_tokens:
            return None

        logits = [prefs.get(t, 0.2) for t in pool_tokens]
        if any(not isinstance(l, (int, float)) or math.isnan(l) for l in logits):
            return random.choice(pool_tokens)

        m = max(logits)
        exps = [math.exp(l - m) for l in logits]
        total = sum(exps)
        if not total or math.isnan(total):
            return random.choice(pool_tokens)

        probs = [e / total for e in exps]
        if hasattr(o, "stab_adjust_token_probs"):
            probs = o.stab_adjust_token_probs(pool_tokens, probs)
        return random.choices(pool_tokens, probs)[0]

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def render_utterance(self, tokens: list[str], punct_prob: float) -> str:
        tokens = [t for t in tokens if isinstance(t, str) and t.strip()]
        if not tokens:
            tokens = ["na"]
        utter = " ".join(tokens)
        if random.random() < float(punct_prob):
            utter += random.choice(PUNCT)
        return utter

    # ------------------------------------------------------------------
    # Feedback / expectation
    # ------------------------------------------------------------------
    def get_overall_expectation(self) -> float:
        o = self.owner
        expectation = float(getattr(o, "numeric_bias", 0.0))
        assoc = getattr(o, "utterance_memory", {}).get("associations", {})
        if assoc:
            expectation = 0.7 * expectation + 0.3 * (sum(assoc.values()) / len(assoc))
        return max(-1.0, min(1.0, expectation))

    def get_utterance_expectation(self, utterance: str) -> float:
        o = self.owner
        return float(getattr(o, "utterance_memory", {}).get("associations", {}).get(utterance, 0.0))

    def learn_from_feedback(self, utterance: str, reward: float, lr: float = 0.1):
        o = self.owner
        if not utterance:
            return
        r = max(-1.0, min(+1.0, float(reward)))
        mem = o.utterance_memory["associations"]
        old = mem.get(utterance, 0.0)
        mem[utterance] = max(-1.0, min(1.0, old + float(lr) * r))
        for k in list(mem.keys()):
            mem[k] *= 0.999
